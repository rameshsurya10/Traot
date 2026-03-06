"""
Forex-Specific Risk Models
==========================

Risk models tailored for Forex trading that integrate with the existing
RiskManager framework.

Models:
- ForexLeverageRisk: Enforce leverage limits (e.g., 50:1)
- SpreadAdjustmentRisk: Block/reduce trades during wide spreads
- ForexCorrelationRisk: Limit exposure to correlated pairs

Usage:
    from src.portfolio.forex.risk_models import (
        ForexLeverageRisk,
        SpreadAdjustmentRisk,
        ForexCorrelationRisk
    )

    # Add to risk manager
    risk_manager.add_model(ForexLeverageRisk(max_leverage=50.0))
    risk_manager.add_model(SpreadAdjustmentRisk())
"""

import logging
from typing import TYPE_CHECKING, Optional

from ..risk import RiskModel, RiskAssessment, RiskAction
from .leverage_manager import LeverageManager
from .spread_tracker import SpreadTracker
from .constants import (
    US_MAX_LEVERAGE,
    CORRELATION_GROUPS,
    is_forex_pair,
)

if TYPE_CHECKING:
    from ..manager import PortfolioManager

logger = logging.getLogger(__name__)


class ForexLeverageRisk(RiskModel):
    """
    Forex Leverage Risk Model.

    Evaluates trades against leverage constraints:
    - Maximum leverage limit (default 50:1 for US)
    - Progressive reduction as leverage increases
    - Blocks trades exceeding limit

    Example:
        risk = ForexLeverageRisk(max_leverage=50.0)
        assessment = risk.evaluate(portfolio, "EUR/USD", 100000, 1.1000, "BUY")

        if assessment.action == RiskAction.BLOCK:
            print(f"Trade blocked: {assessment.reason}")
    """

    def __init__(
        self,
        max_leverage: float = US_MAX_LEVERAGE,
        warning_leverage: float = 40.0,
        leverage_manager: Optional[LeverageManager] = None
    ):
        """
        Initialize Forex leverage risk model.

        Args:
            max_leverage: Maximum allowed leverage (default 50:1)
            warning_leverage: Leverage level to start reducing positions
            leverage_manager: Optional pre-configured LeverageManager
        """
        super().__init__("ForexLeverage")
        self.max_leverage = max_leverage
        self.warning_leverage = warning_leverage
        self.leverage_mgr = leverage_manager or LeverageManager(max_leverage)

    def evaluate(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskAssessment:
        """
        Evaluate trade against Forex leverage constraints.

        Args:
            portfolio: Current portfolio state
            symbol: Currency pair (e.g., "EUR/USD")
            quantity: Position size in units (not lots)
            price: Current exchange rate
            side: "BUY" or "SELL"

        Returns:
            RiskAssessment with action and adjusted quantity
        """
        # Skip non-Forex symbols
        if not is_forex_pair(symbol):
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="Not a Forex pair",
                risk_score=0.0
            )

        # Always allow closing positions
        if side.upper() == "SELL" and portfolio.has_position(symbol):
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="Closing position allowed",
                risk_score=0.0
            )

        # Calculate position value
        position_value = quantity * price
        account_equity = portfolio.total_value

        if account_equity <= 0:
            return RiskAssessment(
                action=RiskAction.BLOCK,
                adjusted_quantity=0,
                reason="No account equity",
                risk_score=1.0
            )

        # Get current Forex exposure
        current_exposure = self._get_forex_exposure(portfolio)

        # Check if trade can be opened
        can_open, margin_required, reason = self.leverage_mgr.can_open_position(
            account_equity=account_equity,
            position_value=position_value,
            current_exposure=current_exposure
        )

        if not can_open:
            return RiskAssessment(
                action=RiskAction.BLOCK,
                adjusted_quantity=0,
                reason=reason,
                risk_score=1.0
            )

        # Calculate new leverage
        new_exposure = current_exposure + position_value
        new_leverage = new_exposure / account_equity
        risk_score = min(new_leverage / self.max_leverage, 1.0)

        # Reduce position if approaching limit
        if new_leverage > self.warning_leverage:
            max_additional = self.leverage_mgr.calculate_max_position(
                account_equity, current_exposure
            )
            allowed_quantity = (max_additional / price) * 0.9  # 90% safety margin

            if allowed_quantity < quantity:
                return RiskAssessment(
                    action=RiskAction.REDUCE,
                    adjusted_quantity=allowed_quantity,
                    reason=f"Leverage {new_leverage:.1f}x approaching limit, reduced",
                    risk_score=risk_score
                )

        return RiskAssessment(
            action=RiskAction.ALLOW,
            adjusted_quantity=quantity,
            reason="OK",
            risk_score=risk_score
        )

    def _get_forex_exposure(self, portfolio: 'PortfolioManager') -> float:
        """Get current Forex exposure from portfolio."""
        exposure = 0.0
        for symbol, holding in portfolio.holdings.items():
            if is_forex_pair(symbol) and holding.invested:
                exposure += abs(holding.holdings_value)
        return exposure


class SpreadAdjustmentRisk(RiskModel):
    """
    Spread-Based Risk Model.

    Evaluates trades based on current spread conditions:
    - Blocks trades during excessive spread widening (3x average)
    - Reduces position size during moderate widening (2x average)

    Example:
        tracker = SpreadTracker()
        risk = SpreadAdjustmentRisk(spread_tracker=tracker)

        assessment = risk.evaluate(portfolio, "EUR/USD", 100000, 1.1000, "BUY")
    """

    def __init__(
        self,
        spread_tracker: Optional[SpreadTracker] = None,
        max_spread_multiplier: float = 3.0,
        reduce_spread_multiplier: float = 2.0
    ):
        """
        Initialize spread adjustment risk model.

        Args:
            spread_tracker: SpreadTracker instance
            max_spread_multiplier: Block trades if spread > this * average
            reduce_spread_multiplier: Reduce size if spread > this * average
        """
        super().__init__("SpreadAdjustment")
        self.spread_tracker = spread_tracker or SpreadTracker()
        self.max_multiplier = max_spread_multiplier
        self.reduce_multiplier = reduce_spread_multiplier

    def evaluate(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskAssessment:
        """
        Evaluate trade against spread conditions.

        Args:
            portfolio: Current portfolio state
            symbol: Currency pair
            quantity: Position size in units
            price: Current price
            side: "BUY" or "SELL"

        Returns:
            RiskAssessment with action and adjusted quantity
        """
        # Skip non-Forex symbols
        if not is_forex_pair(symbol):
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="Not a Forex pair",
                risk_score=0.0
            )

        # Always allow closing positions
        if side.upper() == "SELL" and portfolio.has_position(symbol):
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="Closing position allowed",
                risk_score=0.0
            )

        # Get spread stats
        stats = self.spread_tracker.get_spread_stats(symbol)

        if stats is None:
            # No spread data - allow but warn
            logger.debug(f"No spread data for {symbol}")
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="No spread data available",
                risk_score=0.2
            )

        # Calculate spread ratio
        if stats.average_spread == 0:
            spread_ratio = 1.0
        else:
            spread_ratio = stats.current_spread / stats.average_spread

        risk_score = min(spread_ratio / self.max_multiplier, 1.0)

        # Block if spread too wide
        if spread_ratio >= self.max_multiplier:
            return RiskAssessment(
                action=RiskAction.BLOCK,
                adjusted_quantity=0,
                reason=f"Spread too wide: {stats.current_spread:.1f} pips ({spread_ratio:.1f}x avg)",
                risk_score=1.0
            )

        # Reduce if spread elevated
        if spread_ratio >= self.reduce_multiplier:
            reduction = 1 - ((spread_ratio - self.reduce_multiplier) /
                           (self.max_multiplier - self.reduce_multiplier))
            reduced_qty = quantity * max(reduction, 0.25)

            return RiskAssessment(
                action=RiskAction.REDUCE,
                adjusted_quantity=reduced_qty,
                reason=f"Spread elevated: {stats.current_spread:.1f} pips, reduced",
                risk_score=risk_score
            )

        return RiskAssessment(
            action=RiskAction.ALLOW,
            adjusted_quantity=quantity,
            reason="OK",
            risk_score=risk_score
        )


class ForexCorrelationRisk(RiskModel):
    """
    Forex Correlation Risk Model.

    Limits exposure to highly correlated pairs:
    - EUR/USD and GBP/USD often move together
    - USD/JPY and USD/CHF often move inversely

    Prevents over-concentration in similar trades.

    Example:
        risk = ForexCorrelationRisk(max_group_exposure=0.40)
        assessment = risk.evaluate(portfolio, "GBP/USD", 100000, 1.25, "BUY")
    """

    def __init__(self, max_group_exposure: float = 0.40):
        """
        Initialize correlation risk model.

        Args:
            max_group_exposure: Maximum exposure to correlated group (40% default)
        """
        super().__init__("ForexCorrelation")
        self.max_group_exposure = max_group_exposure

    def _get_symbol_groups(self, symbol: str) -> list:
        """Get correlation groups for a symbol."""
        normalized = symbol.upper().replace("_", "/")
        if "/" not in normalized and len(normalized) == 6:
            normalized = f"{normalized[:3]}/{normalized[3:]}"

        groups = []
        for group_name, symbols in CORRELATION_GROUPS.items():
            if normalized in symbols:
                groups.append(group_name)
        return groups

    def evaluate(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskAssessment:
        """
        Evaluate correlation risk.

        Args:
            portfolio: Current portfolio state
            symbol: Currency pair
            quantity: Position size in units
            price: Current price
            side: "BUY" or "SELL"

        Returns:
            RiskAssessment with action and adjusted quantity
        """
        if not is_forex_pair(symbol):
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="Not a Forex pair",
                risk_score=0.0
            )

        # Always allow closing positions
        if side.upper() == "SELL" and portfolio.has_position(symbol):
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="Closing position allowed",
                risk_score=0.0
            )

        portfolio_value = portfolio.total_value
        if portfolio_value <= 0:
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="OK",
                risk_score=0.0
            )

        # Get groups this symbol belongs to
        groups = self._get_symbol_groups(symbol)
        if not groups:
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="OK",
                risk_score=0.0
            )

        # Calculate exposure in each group
        proposed_value = quantity * price
        max_risk_score = 0.0

        for group_name in groups:
            group_symbols = CORRELATION_GROUPS.get(group_name, [])
            group_exposure = 0.0

            for sym in group_symbols:
                holding = portfolio.holdings.get(sym)
                if holding and holding.invested:
                    group_exposure += abs(holding.holdings_value)

            # Add proposed position
            new_group_exposure = group_exposure + proposed_value
            group_percent = new_group_exposure / portfolio_value

            if group_percent > self.max_group_exposure:
                allowed_value = (self.max_group_exposure * portfolio_value) - group_exposure

                if allowed_value <= 0:
                    return RiskAssessment(
                        action=RiskAction.BLOCK,
                        adjusted_quantity=0,
                        reason=f"Correlation group '{group_name}' at max exposure",
                        risk_score=1.0
                    )
                else:
                    allowed_qty = allowed_value / price
                    return RiskAssessment(
                        action=RiskAction.REDUCE,
                        adjusted_quantity=allowed_qty,
                        reason=f"Reduced for correlation group '{group_name}'",
                        risk_score=group_percent
                    )

            max_risk_score = max(max_risk_score, group_percent / self.max_group_exposure)

        return RiskAssessment(
            action=RiskAction.ALLOW,
            adjusted_quantity=quantity,
            reason="OK",
            risk_score=max_risk_score
        )


class ForexSessionRisk(RiskModel):
    """
    Trading Session Risk Model.

    Considers market liquidity based on trading session:
    - Reduces position size during low-liquidity sessions
    - Warns about trading during session overlaps

    Example:
        risk = ForexSessionRisk()
        assessment = risk.evaluate(portfolio, "EUR/USD", 100000, 1.1, "BUY")
    """

    # Liquidity factors by session (1.0 = normal, <1 = lower liquidity)
    SESSION_LIQUIDITY = {
        "sydney": 0.6,
        "tokyo": 0.7,
        "london": 1.0,
        "new_york": 1.0,
        "london_new_york_overlap": 1.0,
        "tokyo_london_overlap": 0.9,
        "closed": 0.3,  # Weekend or between sessions
    }

    def __init__(self, min_liquidity_factor: float = 0.5):
        """
        Initialize session risk model.

        Args:
            min_liquidity_factor: Reduce size if liquidity < this
        """
        super().__init__("ForexSession")
        self.min_liquidity = min_liquidity_factor

    def evaluate(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskAssessment:
        """Evaluate session-based liquidity risk."""
        if not is_forex_pair(symbol):
            return RiskAssessment(
                action=RiskAction.ALLOW,
                adjusted_quantity=quantity,
                reason="Not a Forex pair",
                risk_score=0.0
            )

        from .constants import get_current_session
        session = get_current_session()

        liquidity = self.SESSION_LIQUIDITY.get(session, 0.7)
        risk_score = 1 - liquidity

        if liquidity < self.min_liquidity:
            reduction = liquidity / self.min_liquidity
            reduced_qty = quantity * reduction

            return RiskAssessment(
                action=RiskAction.REDUCE,
                adjusted_quantity=reduced_qty,
                reason=f"Low liquidity session ({session}), reduced",
                risk_score=risk_score
            )

        return RiskAssessment(
            action=RiskAction.ALLOW,
            adjusted_quantity=quantity,
            reason=f"Session: {session}",
            risk_score=risk_score
        )
