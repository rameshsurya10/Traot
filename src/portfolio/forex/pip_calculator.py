"""
Forex Pip Calculator
====================

Calculate pip values for Forex currency pairs.

Pip = Percentage In Point (smallest price movement)
- Most pairs: 0.0001 (4 decimal places) = $10 per pip per standard lot
- JPY pairs: 0.01 (2 decimal places) = $6.67 per pip at 150.00

Key Formulas:
- Quote = USD: pip_value = lot_size * pip_size
- Base = USD: pip_value = (lot_size * pip_size) / price
- Cross pair: pip_value = (lot_size * pip_size) / USD_rate

Usage:
    from src.portfolio.forex.pip_calculator import PipCalculator

    calc = PipCalculator(account_currency="USD")

    # Calculate pip value for 1 standard lot EUR/USD
    pip_value = calc.get_pip_value("EUR/USD", 1.1000, lot_size=100000)
    # Returns: 10.0 (USD per pip)

    # Calculate pips between two prices
    pips = calc.price_to_pips("EUR/USD", 1.1050, 1.1000)
    # Returns: 50.0 pips
"""

import logging
from typing import Dict, Optional

from .constants import (
    CurrencyPairConfig,
    LotType,
    get_pair_config,
    DEFAULT_ACCOUNT_CURRENCY,
)

logger = logging.getLogger(__name__)


class PipCalculator:
    """
    Calculate pip values for Forex currency pairs.

    Thread-safe: All methods are stateless except for FX rate cache.

    Attributes:
        account_currency: Base currency for account (default: USD)

    Example:
        calc = PipCalculator(account_currency="USD")

        # Get pip value for EUR/USD
        pip_value = calc.get_pip_value("EUR/USD", 1.1000)
        print(f"Pip value: ${pip_value:.2f}")  # $10.00

        # Convert price difference to pips
        pips = calc.price_to_pips("USD/JPY", 150.50, 150.00)
        print(f"Move: {pips} pips")  # 50 pips
    """

    def __init__(self, account_currency: str = DEFAULT_ACCOUNT_CURRENCY):
        """
        Initialize pip calculator.

        Args:
            account_currency: Account base currency (default: USD)
        """
        self.account_currency = account_currency.upper()
        self._fx_rates: Dict[str, float] = {}

    def get_pair_config(self, symbol: str) -> CurrencyPairConfig:
        """
        Get configuration for a currency pair.

        Args:
            symbol: Currency pair (e.g., "EUR/USD")

        Returns:
            CurrencyPairConfig for the pair

        Raises:
            ValueError: If pair is not found
        """
        return get_pair_config(symbol)

    def is_jpy_pair(self, symbol: str) -> bool:
        """Check if pair involves JPY (2 decimal places)."""
        return "JPY" in symbol.upper()

    def get_pip_size(self, symbol: str) -> float:
        """
        Get pip size for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            Pip size (0.0001 for most pairs, 0.01 for JPY)
        """
        config = self.get_pair_config(symbol)
        return config.pip_size

    def get_pip_decimal_places(self, symbol: str) -> int:
        """
        Get number of decimal places for pips.

        Args:
            symbol: Currency pair

        Returns:
            4 for most pairs, 2 for JPY pairs
        """
        config = self.get_pair_config(symbol)
        return config.pip_decimal_places

    def price_to_pips(
        self,
        symbol: str,
        price1: float,
        price2: float
    ) -> float:
        """
        Convert price difference to pips.

        Args:
            symbol: Currency pair
            price1: First price (e.g., entry)
            price2: Second price (e.g., stop)

        Returns:
            Number of pips (can be negative)

        Example:
            pips = calc.price_to_pips("EUR/USD", 1.1050, 1.1000)
            # Returns: 50.0 (moved up 50 pips)
        """
        pip_size = self.get_pip_size(symbol)
        return (price1 - price2) / pip_size

    def pips_to_price(
        self,
        symbol: str,
        base_price: float,
        pips: float
    ) -> float:
        """
        Add pips to a price.

        Args:
            symbol: Currency pair
            base_price: Starting price
            pips: Number of pips to add (can be negative)

        Returns:
            Adjusted price

        Example:
            new_price = calc.pips_to_price("EUR/USD", 1.1000, 50)
            # Returns: 1.1050
        """
        pip_size = self.get_pip_size(symbol)
        return base_price + (pips * pip_size)

    def get_pip_value(
        self,
        symbol: str,
        current_price: float,
        lot_size: float = 100000,
        fx_rates: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate the value of 1 pip in account currency.

        Args:
            symbol: Currency pair (e.g., "EUR/USD")
            current_price: Current exchange rate
            lot_size: Position size in units (default: standard lot = 100,000)
            fx_rates: Optional dict of conversion rates for cross pairs

        Returns:
            Pip value in account currency (USD by default)

        Formulas:
            Quote = USD (EUR/USD): pip_value = lot_size * pip_size
            Base = USD (USD/JPY):  pip_value = (lot_size * pip_size) / price
            Cross pair (EUR/GBP): pip_value = (lot_size * pip_size) * GBP/USD rate

        Example:
            # EUR/USD at 1.1000, standard lot
            pip_value = calc.get_pip_value("EUR/USD", 1.1000)
            # Returns: 10.0 (USD)

            # USD/JPY at 150.00, standard lot
            pip_value = calc.get_pip_value("USD/JPY", 150.00)
            # Returns: 6.67 (USD)
        """
        config = self.get_pair_config(symbol)
        pip_size = config.pip_size

        # Update cached rates if provided
        if fx_rates:
            self._fx_rates.update(fx_rates)

        # Case 1: Quote currency is account currency (EUR/USD, GBP/USD)
        # Formula: pip_value = lot_size * pip_size
        if config.quote_currency == self.account_currency:
            return lot_size * pip_size

        # Case 2: Base currency is account currency (USD/JPY, USD/CHF)
        # Formula: pip_value = (lot_size * pip_size) / price
        if config.base_currency == self.account_currency:
            if current_price == 0:
                logger.warning(f"Current price is 0 for {symbol}")
                return 0.0
            return (lot_size * pip_size) / current_price

        # Case 3: Cross pair (EUR/GBP, EUR/JPY) - need conversion
        # Need to find USD rate for the quote currency
        quote = config.quote_currency

        # Try quote/USD format first (e.g., GBP/USD for EUR/GBP)
        quote_usd_pair = f"{quote}/{self.account_currency}"
        if quote_usd_pair in self._fx_rates:
            conversion_rate = self._fx_rates[quote_usd_pair]
            return (lot_size * pip_size) * conversion_rate

        # Try USD/quote format (e.g., USD/JPY for EUR/JPY)
        usd_quote_pair = f"{self.account_currency}/{quote}"
        if usd_quote_pair in self._fx_rates:
            conversion_rate = self._fx_rates[usd_quote_pair]
            if conversion_rate == 0:
                logger.warning(f"Conversion rate is 0 for {usd_quote_pair}")
                return 0.0
            return (lot_size * pip_size) / conversion_rate

        # Fallback: Use current price as rough estimate
        # This is less accurate but prevents failure
        logger.debug(
            f"No conversion rate found for {symbol}, "
            f"using price as estimate"
        )
        return lot_size * pip_size

    def get_pip_value_per_lot_type(
        self,
        symbol: str,
        current_price: float,
        lot_type: LotType = LotType.STANDARD,
        fx_rates: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Get pip value for a specific lot type.

        Args:
            symbol: Currency pair
            current_price: Current price
            lot_type: Standard, Mini, Micro, or Nano
            fx_rates: Conversion rates for cross pairs

        Returns:
            Pip value for the specified lot type

        Example:
            # Mini lot EUR/USD
            pip_value = calc.get_pip_value_per_lot_type(
                "EUR/USD", 1.1000, LotType.MINI
            )
            # Returns: 1.0 (USD per pip)
        """
        return self.get_pip_value(
            symbol,
            current_price,
            lot_size=lot_type.value,
            fx_rates=fx_rates
        )

    def update_fx_rates(self, rates: Dict[str, float]) -> None:
        """
        Update cached FX rates for cross pair calculations.

        Args:
            rates: Dict of pair -> rate (e.g., {"GBP/USD": 1.2500})
        """
        self._fx_rates.update(rates)

    def calculate_risk_in_pips(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float
    ) -> float:
        """
        Calculate risk in pips (always positive).

        Args:
            symbol: Currency pair
            entry_price: Entry price
            stop_price: Stop loss price

        Returns:
            Risk in pips (absolute value)
        """
        return abs(self.price_to_pips(symbol, entry_price, stop_price))

    def calculate_reward_in_pips(
        self,
        symbol: str,
        entry_price: float,
        target_price: float
    ) -> float:
        """
        Calculate potential reward in pips (always positive).

        Args:
            symbol: Currency pair
            entry_price: Entry price
            target_price: Take profit price

        Returns:
            Reward in pips (absolute value)
        """
        return abs(self.price_to_pips(symbol, entry_price, target_price))

    def calculate_risk_reward_ratio(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        target_price: float
    ) -> float:
        """
        Calculate risk:reward ratio.

        Args:
            symbol: Currency pair
            entry_price: Entry price
            stop_price: Stop loss price
            target_price: Take profit price

        Returns:
            Reward / Risk ratio (e.g., 2.0 for 1:2 risk:reward)
        """
        risk_pips = self.calculate_risk_in_pips(symbol, entry_price, stop_price)
        reward_pips = self.calculate_reward_in_pips(symbol, entry_price, target_price)

        if risk_pips == 0:
            return 0.0

        return reward_pips / risk_pips

    def calculate_position_size(
        self,
        symbol: str,
        account_risk: float,
        stop_pips: float,
        current_price: float,
        lot_type: LotType = LotType.STANDARD,
        fx_rates: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate position size in lots based on risk.

        Formula: Lots = Account Risk / (Stop Pips * Pip Value per Lot)

        Args:
            symbol: Currency pair
            account_risk: Amount to risk in account currency (e.g., $200)
            stop_pips: Stop loss distance in pips
            current_price: Current exchange rate
            lot_type: Lot type for pip value calculation
            fx_rates: Conversion rates for cross pairs

        Returns:
            Position size in lots

        Example:
            # Risk $200 with 50 pip stop on EUR/USD
            lots = calc.calculate_position_size(
                "EUR/USD",
                account_risk=200,
                stop_pips=50,
                current_price=1.1000
            )
            # Returns: 0.40 lots (40,000 units)
        """
        if stop_pips == 0:
            logger.warning("Stop pips is 0, cannot calculate position size")
            return 0.0

        pip_value = self.get_pip_value_per_lot_type(
            symbol, current_price, lot_type, fx_rates
        )

        if pip_value == 0:
            logger.warning(f"Pip value is 0 for {symbol}")
            return 0.0

        lots = account_risk / (stop_pips * pip_value)
        return round(lots, 2)  # Round to 2 decimal places (0.01 lot precision)

    def calculate_profit_loss(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        lot_size: float,
        side: str,
        current_usd_rates: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate profit/loss for a trade in account currency.

        Args:
            symbol: Currency pair
            entry_price: Entry price
            exit_price: Exit price
            lot_size: Position size in units
            side: "BUY" or "SELL"
            current_usd_rates: USD conversion rates for cross pairs

        Returns:
            Profit/loss in account currency (negative = loss)

        Example:
            # Long EUR/USD from 1.1000 to 1.1050, 10000 units
            pnl = calc.calculate_profit_loss(
                "EUR/USD", 1.1000, 1.1050, 10000, "BUY"
            )
            # Returns: 50.0 (profit of $50)
        """
        pips = self.price_to_pips(symbol, exit_price, entry_price)

        # For sell trades, profit is when price goes down
        if side.upper() == "SELL":
            pips = -pips

        pip_value = self.get_pip_value(
            symbol, exit_price, lot_size, current_usd_rates
        )

        return pips * pip_value
