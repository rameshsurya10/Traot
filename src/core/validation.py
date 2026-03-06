"""
Order Validation Module
=======================
Comprehensive order validation before execution.

Validates:
- Order parameters (quantity, price, symbol)
- Risk limits (position size, daily loss)
- Market conditions (circuit breaker, trading hours)
- Portfolio constraints (buying power, margin)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class OrderValidationResult:
    """Complete order validation result."""
    is_valid: bool
    checks: List[ValidationCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_check(self, check: ValidationCheck):
        self.checks.append(check)
        if not check.passed:
            if check.severity == "error":
                self.errors.append(check.message)
                self.is_valid = False
            elif check.severity == "warning":
                self.warnings.append(check.message)

    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'checks': [
                {'name': c.name, 'passed': c.passed, 'message': c.message}
                for c in self.checks
            ]
        }


class OrderValidator:
    """
    Comprehensive order validator.

    Usage:
        validator = OrderValidator(config)
        result = validator.validate(order, portfolio, market_data)
        if result.is_valid:
            execute_order(order)
        else:
            log_rejection(result.errors)
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}

        # Risk limits
        self.max_position_percent = config.get('max_position_percent', 0.25)
        self.max_daily_loss_percent = config.get('daily_loss_limit', 0.05)
        self.max_order_value = config.get('max_order_value', 100000)
        self.min_order_value = config.get('min_order_value', 10)

        # Circuit breaker
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = ""

        # Trading hours (UTC)
        self.trading_hours_enabled = config.get('trading_hours_enabled', False)
        self.trading_start = time(0, 0)  # 24/7 for crypto
        self.trading_end = time(23, 59)

        # Allowed symbols
        self.allowed_symbols: Optional[List[str]] = config.get('allowed_symbols')

        # Custom validators
        self._custom_validators: List[Callable] = []

    def validate(
        self,
        order: Dict,
        portfolio: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> OrderValidationResult:
        """
        Validate an order against all rules.

        Args:
            order: Order dict with symbol, quantity, side, price, order_type
            portfolio: Portfolio state dict
            market_data: Current market data dict

        Returns:
            OrderValidationResult with pass/fail and messages
        """
        result = OrderValidationResult(is_valid=True)

        # Basic order validation
        self._validate_order_params(order, result)

        # Circuit breaker check
        self._validate_circuit_breaker(result)

        # Trading hours check
        self._validate_trading_hours(result)

        # Symbol validation
        self._validate_symbol(order, result)

        # Risk validation (if portfolio provided)
        if portfolio:
            self._validate_position_size(order, portfolio, result)
            self._validate_buying_power(order, portfolio, result)
            self._validate_daily_loss(portfolio, result)

        # Market conditions (if market data provided)
        if market_data:
            self._validate_market_conditions(order, market_data, result)

        # Run custom validators
        for validator in self._custom_validators:
            try:
                validator(order, portfolio, market_data, result)
            except Exception as e:
                logger.error(f"Custom validator error: {e}")

        return result

    def _validate_order_params(self, order: Dict, result: OrderValidationResult):
        """Validate basic order parameters."""
        # Symbol required
        if not order.get('symbol'):
            result.add_check(ValidationCheck(
                name="symbol_required",
                passed=False,
                message="Order must have a symbol"
            ))

        # Quantity validation
        quantity = order.get('quantity', 0)
        if quantity <= 0:
            result.add_check(ValidationCheck(
                name="quantity_positive",
                passed=False,
                message=f"Quantity must be positive, got {quantity}"
            ))
        else:
            result.add_check(ValidationCheck(
                name="quantity_positive",
                passed=True,
                message="Quantity is valid"
            ))

        # Order type validation
        valid_types = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
        order_type = order.get('order_type', 'MARKET').upper()
        if order_type not in valid_types:
            result.add_check(ValidationCheck(
                name="order_type_valid",
                passed=False,
                message=f"Invalid order type: {order_type}"
            ))

        # Limit orders need price
        if order_type in ['LIMIT', 'STOP_LIMIT']:
            price = order.get('price', 0)
            if price <= 0:
                result.add_check(ValidationCheck(
                    name="limit_price_required",
                    passed=False,
                    message="Limit orders require a positive price"
                ))

        # Side validation
        side = order.get('side', '').upper()
        if side not in ['BUY', 'SELL']:
            result.add_check(ValidationCheck(
                name="side_valid",
                passed=False,
                message=f"Invalid side: {side}"
            ))

    def _validate_circuit_breaker(self, result: OrderValidationResult):
        """Check if circuit breaker is active."""
        if self.circuit_breaker_active:
            result.add_check(ValidationCheck(
                name="circuit_breaker",
                passed=False,
                message=f"Circuit breaker active: {self.circuit_breaker_reason}"
            ))
        else:
            result.add_check(ValidationCheck(
                name="circuit_breaker",
                passed=True,
                message="Circuit breaker not active"
            ))

    def _validate_trading_hours(self, result: OrderValidationResult):
        """Check if within trading hours."""
        if not self.trading_hours_enabled:
            result.add_check(ValidationCheck(
                name="trading_hours",
                passed=True,
                message="Trading hours check disabled (24/7)"
            ))
            return

        now = datetime.utcnow().time()
        if self.trading_start <= now <= self.trading_end:
            result.add_check(ValidationCheck(
                name="trading_hours",
                passed=True,
                message="Within trading hours"
            ))
        else:
            result.add_check(ValidationCheck(
                name="trading_hours",
                passed=False,
                message=f"Outside trading hours ({self.trading_start}-{self.trading_end} UTC)"
            ))

    def _validate_symbol(self, order: Dict, result: OrderValidationResult):
        """Validate symbol is allowed."""
        symbol = order.get('symbol', '')

        if self.allowed_symbols is None:
            result.add_check(ValidationCheck(
                name="symbol_allowed",
                passed=True,
                message="All symbols allowed"
            ))
            return

        if symbol in self.allowed_symbols:
            result.add_check(ValidationCheck(
                name="symbol_allowed",
                passed=True,
                message=f"Symbol {symbol} is allowed"
            ))
        else:
            result.add_check(ValidationCheck(
                name="symbol_allowed",
                passed=False,
                message=f"Symbol {symbol} not in allowed list"
            ))

    def _validate_position_size(
        self,
        order: Dict,
        portfolio: Dict,
        result: OrderValidationResult
    ):
        """Validate position size limits."""
        quantity = order.get('quantity', 0)
        price = order.get('price', 0) or order.get('current_price', 0)

        if price <= 0:
            result.add_check(ValidationCheck(
                name="position_size",
                passed=True,
                message="Cannot validate position size without price",
                severity="warning"
            ))
            return

        order_value = quantity * price
        portfolio_value = portfolio.get('total_value', 0)

        if portfolio_value <= 0:
            result.add_check(ValidationCheck(
                name="position_size",
                passed=False,
                message="Portfolio value is zero or negative"
            ))
            return

        position_percent = order_value / portfolio_value

        # Check max position size
        if position_percent > self.max_position_percent:
            result.add_check(ValidationCheck(
                name="position_size",
                passed=False,
                message=f"Position size {position_percent:.1%} exceeds max {self.max_position_percent:.1%}"
            ))
        else:
            result.add_check(ValidationCheck(
                name="position_size",
                passed=True,
                message=f"Position size {position_percent:.1%} within limits"
            ))

        # Check order value limits
        if order_value < self.min_order_value:
            result.add_check(ValidationCheck(
                name="min_order_value",
                passed=False,
                message=f"Order value ${order_value:.2f} below minimum ${self.min_order_value}"
            ))

        if order_value > self.max_order_value:
            result.add_check(ValidationCheck(
                name="max_order_value",
                passed=False,
                message=f"Order value ${order_value:.2f} exceeds maximum ${self.max_order_value}"
            ))

    def _validate_buying_power(
        self,
        order: Dict,
        portfolio: Dict,
        result: OrderValidationResult
    ):
        """Validate sufficient buying power."""
        if order.get('side', '').upper() != 'BUY':
            return

        quantity = order.get('quantity', 0)
        price = order.get('price', 0) or order.get('current_price', 0)

        if price <= 0:
            return

        order_value = quantity * price
        cash = portfolio.get('cash', 0)

        if order_value > cash:
            result.add_check(ValidationCheck(
                name="buying_power",
                passed=False,
                message=f"Insufficient cash: need ${order_value:.2f}, have ${cash:.2f}"
            ))
        else:
            result.add_check(ValidationCheck(
                name="buying_power",
                passed=True,
                message=f"Sufficient buying power (${cash:.2f} available)"
            ))

    def _validate_daily_loss(
        self,
        portfolio: Dict,
        result: OrderValidationResult
    ):
        """Validate daily loss limit not exceeded."""
        daily_pnl = portfolio.get('daily_pnl', 0)
        portfolio_value = portfolio.get('total_value', 0)

        if portfolio_value <= 0:
            return

        daily_loss_percent = abs(min(0, daily_pnl)) / portfolio_value

        if daily_loss_percent >= self.max_daily_loss_percent:
            result.add_check(ValidationCheck(
                name="daily_loss_limit",
                passed=False,
                message=f"Daily loss limit reached: {daily_loss_percent:.1%} (max {self.max_daily_loss_percent:.1%})"
            ))
        elif daily_loss_percent >= self.max_daily_loss_percent * 0.8:
            result.add_check(ValidationCheck(
                name="daily_loss_limit",
                passed=True,
                message=f"Approaching daily loss limit: {daily_loss_percent:.1%}",
                severity="warning"
            ))
        else:
            result.add_check(ValidationCheck(
                name="daily_loss_limit",
                passed=True,
                message="Within daily loss limits"
            ))

    def _validate_market_conditions(
        self,
        order: Dict,
        market_data: Dict,
        result: OrderValidationResult
    ):
        """Validate market conditions."""
        # Check for extreme volatility
        volatility = market_data.get('volatility', 0)
        if volatility > 0.10:  # 10% daily volatility
            result.add_check(ValidationCheck(
                name="high_volatility",
                passed=True,
                message=f"High volatility warning: {volatility:.1%}",
                severity="warning"
            ))

        # Check spread
        spread_percent = market_data.get('spread_percent', 0)
        if spread_percent > 0.02:  # 2% spread
            result.add_check(ValidationCheck(
                name="wide_spread",
                passed=True,
                message=f"Wide spread warning: {spread_percent:.2%}",
                severity="warning"
            ))

        # Check liquidity
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', volume)
        if avg_volume > 0 and volume < avg_volume * 0.3:
            result.add_check(ValidationCheck(
                name="low_liquidity",
                passed=True,
                message="Low liquidity warning",
                severity="warning"
            ))

    def activate_circuit_breaker(self, reason: str):
        """Activate circuit breaker to stop all trading."""
        self.circuit_breaker_active = True
        self.circuit_breaker_reason = reason
        logger.warning(f"Circuit breaker activated: {reason}")

    def deactivate_circuit_breaker(self):
        """Deactivate circuit breaker."""
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = ""
        logger.info("Circuit breaker deactivated")

    def add_custom_validator(self, validator: Callable):
        """Add a custom validation function."""
        self._custom_validators.append(validator)


# Convenience functions
def validate_order(order: Dict, config: Optional[Dict] = None) -> OrderValidationResult:
    """Quick order validation."""
    validator = OrderValidator(config)
    return validator.validate(order)


def is_valid_order(order: Dict) -> bool:
    """Simple boolean check."""
    return validate_order(order).is_valid


# Export
__all__ = [
    'OrderValidator',
    'OrderValidationResult',
    'ValidationCheck',
    'ValidationResult',
    'validate_order',
    'is_valid_order'
]
