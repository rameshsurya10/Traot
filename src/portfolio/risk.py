"""
Risk Management Module (Lean-Inspired)
======================================
Portfolio-level risk management and constraints.

Features:
- Maximum drawdown protection
- Position size limits
- Sector/correlation exposure
- Portfolio volatility targets
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .manager import PortfolioManager

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Risk management actions."""
    ALLOW = "allow"          # Allow the trade
    REDUCE = "reduce"        # Reduce position size
    BLOCK = "block"          # Block the trade
    LIQUIDATE = "liquidate"  # Close existing position


@dataclass
class RiskAssessment:
    """Result of risk evaluation."""
    action: RiskAction
    adjusted_quantity: float
    reason: str
    risk_score: float  # 0.0 = no risk, 1.0 = maximum risk


class RiskModel(ABC):
    """
    Abstract base class for risk models (Lean-inspired).

    Implement custom risk rules by subclassing.
    """

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    @abstractmethod
    def evaluate(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskAssessment:
        """
        Evaluate trade against risk rules.

        Args:
            portfolio: Current portfolio state
            symbol: Trade symbol
            quantity: Proposed quantity
            price: Trade price
            side: "BUY" or "SELL"

        Returns:
            RiskAssessment with action and adjusted quantity
        """
        pass


class MaximumDrawdownRisk(RiskModel):
    """
    Maximum Drawdown Protection.

    Blocks new long positions when portfolio drawdown exceeds threshold.
    """

    def __init__(self, max_drawdown_percent: float = 20.0):
        """
        Args:
            max_drawdown_percent: Maximum allowed drawdown (e.g., 20 = 20%)
        """
        super().__init__("MaximumDrawdown")
        self.max_drawdown = max_drawdown_percent

    def evaluate(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskAssessment:
        """Evaluate drawdown risk."""
        current_dd = portfolio.drawdown

        # Always allow closing positions
        if side == "SELL" and portfolio.has_position(symbol):
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="Closing position allowed",
                risk_score=current_dd / 100
            )

        # Check drawdown threshold
        if current_dd >= self.max_drawdown:
            return RiskAssessment(
                action=RiskAction.BLOCK,
                adjusted_quantity=0,
                reason=f"Drawdown {current_dd:.1f}% exceeds max {self.max_drawdown}%",
                risk_score=1.0
            )

        # Progressive risk as drawdown approaches limit
        risk_score = current_dd / self.max_drawdown

        # Reduce position size as drawdown increases
        if current_dd > self.max_drawdown * 0.5:
            reduction = 1 - ((current_dd - self.max_drawdown * 0.5) / (self.max_drawdown * 0.5))
            adjusted_qty = quantity * max(reduction, 0.25)  # Min 25% of original
            return RiskAssessment(
                action=RiskAction.REDUCE,
                adjusted_quantity=adjusted_qty,
                reason=f"Reduced due to drawdown {current_dd:.1f}%",
                risk_score=risk_score
            )

        return RiskAssessment(
            action=RiskAction.ALLOW,
            adjusted_quantity=quantity,
            reason="OK",
            risk_score=risk_score
        )


class MaximumPositionSizeRisk(RiskModel):
    """
    Maximum Position Size Constraint.

    Limits individual position size as percentage of portfolio.
    """

    def __init__(
        self,
        max_position_percent: float = 0.25,
        max_total_exposure: float = 1.0
    ):
        """
        Args:
            max_position_percent: Max single position (0.25 = 25%)
            max_total_exposure: Max total exposure (1.0 = 100%)
        """
        super().__init__("MaximumPositionSize")
        self.max_position_percent = max_position_percent
        self.max_total_exposure = max_total_exposure

    def evaluate(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskAssessment:
        """Evaluate position size risk."""
        # Calculate proposed position value
        proposed_value = quantity * price
        portfolio_value = portfolio.total_value

        if portfolio_value == 0:
            return RiskAssessment(
                action=RiskAction.BLOCK,
                adjusted_quantity=0,
                reason="Portfolio value is zero",
                risk_score=1.0
            )

        # Existing position
        holding = portfolio.get_holding(symbol)
        current_value = holding.holdings_value if holding else 0

        # New total position value
        if side == "BUY":
            new_value = current_value + proposed_value
        else:
            new_value = current_value - proposed_value

        # Calculate position percentage
        position_percent = abs(new_value) / portfolio_value

        # Check single position limit
        max_value = portfolio_value * self.max_position_percent
        if position_percent > self.max_position_percent:
            # Calculate allowed quantity
            allowed_value = max_value - abs(current_value)
            allowed_qty = max(allowed_value / price, 0) if price > 0 else 0

            if allowed_qty <= 0:
                return RiskAssessment(
                    action=RiskAction.BLOCK,
                    adjusted_quantity=0,
                    reason=f"Position would exceed {self.max_position_percent:.0%} limit",
                    risk_score=position_percent
                )

            return RiskAssessment(
                action=RiskAction.REDUCE,
                adjusted_quantity=allowed_qty,
                reason=f"Reduced to stay within {self.max_position_percent:.0%} limit",
                risk_score=position_percent
            )

        # Check total exposure
        total_exposure = (portfolio.total_holdings_value + proposed_value) / portfolio_value
        if total_exposure > self.max_total_exposure:
            allowed_exposure = (self.max_total_exposure * portfolio_value -
                              portfolio.total_holdings_value)
            allowed_qty = max(allowed_exposure / price, 0) if price > 0 else 0

            if allowed_qty <= 0:
                return RiskAssessment(
                    action=RiskAction.BLOCK,
                    adjusted_quantity=0,
                    reason=f"Total exposure would exceed {self.max_total_exposure:.0%}",
                    risk_score=total_exposure
                )

            return RiskAssessment(
                action=RiskAction.REDUCE,
                adjusted_quantity=allowed_qty,
                reason="Reduced for total exposure limit",
                risk_score=total_exposure
            )

        return RiskAssessment(
            action=RiskAction.ALLOW,
            adjusted_quantity=quantity,
            reason="OK",
            risk_score=position_percent
        )


class SectorExposureRisk(RiskModel):
    """
    Sector/Category Exposure Limits.

    Limits exposure to any single sector or asset class.
    """

    def __init__(
        self,
        max_sector_exposure: float = 0.4,
        sector_mapping: Dict[str, str] = None
    ):
        """
        Args:
            max_sector_exposure: Max exposure per sector (0.4 = 40%)
            sector_mapping: Symbol to sector mapping
        """
        super().__init__("SectorExposure")
        self.max_sector_exposure = max_sector_exposure
        self.sector_mapping = sector_mapping or {}

        # Default sector mappings
        self._default_sectors = {
            # Crypto
            'BTC': 'crypto', 'ETH': 'crypto', 'SOL': 'crypto',
            'ADA': 'crypto', 'DOGE': 'crypto', 'XRP': 'crypto',
            # Forex
            'EUR': 'forex', 'GBP': 'forex', 'JPY': 'forex',
            'CHF': 'forex', 'AUD': 'forex', 'CAD': 'forex',
            # Tech
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech',
            'AMZN': 'tech', 'META': 'tech', 'NVDA': 'tech',
        }

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol."""
        # Check explicit mapping
        if symbol in self.sector_mapping:
            return self.sector_mapping[symbol]

        # Check default mappings
        base = symbol.split('/')[0].split('-')[0].upper()
        if base in self._default_sectors:
            return self._default_sectors[base]

        return 'other'

    def evaluate(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskAssessment:
        """Evaluate sector exposure risk."""
        sector = self._get_sector(symbol)
        portfolio_value = portfolio.total_value

        if portfolio_value == 0:
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="OK",
                risk_score=0.0
            )

        # Calculate current sector exposure
        sector_value = 0.0
        for sym, holding in portfolio.holdings.items():
            if holding.invested and self._get_sector(sym) == sector:
                sector_value += abs(holding.holdings_value)

        # Add proposed trade
        proposed_value = quantity * price
        new_sector_value = sector_value + (proposed_value if side == "BUY" else -proposed_value)
        sector_exposure = new_sector_value / portfolio_value

        # Check limit
        if sector_exposure > self.max_sector_exposure:
            allowed_value = (self.max_sector_exposure * portfolio_value - sector_value)
            allowed_qty = max(allowed_value / price, 0) if price > 0 else 0

            if allowed_qty <= 0:
                return RiskAssessment(
                    action=RiskAction.BLOCK,
                    adjusted_quantity=0,
                    reason=f"Sector '{sector}' would exceed {self.max_sector_exposure:.0%}",
                    risk_score=sector_exposure
                )

            return RiskAssessment(
                action=RiskAction.REDUCE,
                adjusted_quantity=allowed_qty,
                reason=f"Reduced for sector '{sector}' limit",
                risk_score=sector_exposure
            )

        return RiskAssessment(
            action=RiskAction.ALLOW,
            adjusted_quantity=quantity,
            reason="OK",
            risk_score=sector_exposure
        )


class RiskManager:
    """
    Composite Risk Manager (Lean-Inspired).

    Evaluates trades against multiple risk models.
    Most restrictive constraint wins.

    Example:
        risk = RiskManager()
        risk.add_model(MaximumDrawdownRisk(max_drawdown_percent=20))
        risk.add_model(MaximumPositionSizeRisk(max_position_percent=0.25))

        # Before placing order
        assessment = risk.evaluate_trade(portfolio, "AAPL", 100, 150.0, "BUY")
        if assessment.action == RiskAction.ALLOW:
            brokerage.place_order(...)
        elif assessment.action == RiskAction.REDUCE:
            order.quantity = assessment.adjusted_quantity
            brokerage.place_order(...)
    """

    def __init__(self):
        self._models: List[RiskModel] = []
        self._last_assessment: Optional[RiskAssessment] = None

    def add_model(self, model: RiskModel):
        """Add a risk model."""
        self._models.append(model)
        logger.info(f"Added risk model: {model.name}")

    def remove_model(self, name: str):
        """Remove a risk model by name."""
        self._models = [m for m in self._models if m.name != name]

    def evaluate_trade(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskAssessment:
        """
        Evaluate trade against all risk models.

        Returns the most restrictive assessment.
        """
        if not self._models:
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="No risk models configured",
                risk_score=0.0
            )

        # Collect all assessments
        assessments = []
        for model in self._models:
            if not model.enabled:
                continue
            try:
                assessment = model.evaluate(portfolio, symbol, quantity, price, side)
                assessments.append((model.name, assessment))
            except Exception as e:
                logger.error(f"Risk model {model.name} error: {e}")

        if not assessments:
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="OK",
                risk_score=0.0
            )

        # Priority: LIQUIDATE > BLOCK > REDUCE > ALLOW
        priority = {
            RiskAction.LIQUIDATE: 4,
            RiskAction.BLOCK: 3,
            RiskAction.REDUCE: 2,
            RiskAction.ALLOW: 1
        }

        # Sort by priority (descending) then by adjusted quantity (ascending)
        assessments.sort(
            key=lambda x: (priority[x[1].action], -x[1].adjusted_quantity),
            reverse=True
        )

        # Get most restrictive
        model_name, result = assessments[0]

        # Combine reasons if multiple models triggered
        restrictive_models = [
            (name, a) for name, a in assessments
            if a.action in (RiskAction.BLOCK, RiskAction.REDUCE, RiskAction.LIQUIDATE)
        ]

        if len(restrictive_models) > 1:
            reasons = "; ".join(f"{name}: {a.reason}" for name, a in restrictive_models)
            result = RiskAssessment(
                action=result.action,
                adjusted_quantity=result.adjusted_quantity,
                reason=reasons,
                risk_score=max(a.risk_score for _, a in restrictive_models)
            )

        self._last_assessment = result
        return result

    def get_portfolio_risk_score(self, portfolio: 'PortfolioManager') -> float:
        """
        Calculate overall portfolio risk score.

        Returns:
            Risk score 0.0 (low risk) to 1.0 (high risk)
        """
        if not self._models:
            return 0.0

        # Evaluate a hypothetical small trade to get current risk state
        scores = []
        for model in self._models:
            if not model.enabled:
                continue
            try:
                # Use a dummy evaluation
                assessment = model.evaluate(portfolio, "_CHECK_", 0, 1.0, "BUY")
                scores.append(assessment.risk_score)
            except Exception:
                pass

        return max(scores) if scores else 0.0

    @property
    def last_assessment(self) -> Optional[RiskAssessment]:
        """Get last risk assessment."""
        return self._last_assessment

    def get_status(self) -> dict:
        """Get risk manager status."""
        return {
            'models': [
                {'name': m.name, 'enabled': m.enabled}
                for m in self._models
            ],
            'last_assessment': {
                'action': self._last_assessment.action.value,
                'reason': self._last_assessment.reason,
                'risk_score': self._last_assessment.risk_score
            } if self._last_assessment else None
        }
