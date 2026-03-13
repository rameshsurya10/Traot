"""
Learning State Manager
======================

Manages and persists LEARNING ↔ TRADING mode transitions.

Key Features:
- Track current mode per (symbol, interval)
- Persist mode changes to database
- Query mode history
- Thread-safe operations

Integrates with:
- ConfidenceGate: Determines mode transitions
- Database: Persists state to learning_states table
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
from src.core.database import Database

logger = logging.getLogger(__name__)


class LearningStateManager:
    """
    Manages learning/trading mode state with database persistence.

    Thread-safe: All database operations are atomic via Database class.

    State Transitions:
    - LEARNING → TRADING: When confidence ≥ threshold
    - TRADING → LEARNING: When confidence drops or loss occurs
    """

    def __init__(self, database: Database):
        """
        Initialize state manager.

        Args:
            database: Database instance for persistence
        """
        self.db = database

        # In-memory cache: (symbol, interval) -> current mode
        self._mode_cache: dict = {}

        # Statistics
        self._stats = {
            'mode_queries': 0,
            'transitions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info("LearningStateManager initialized")

    def get_current_mode(
        self,
        symbol: str,
        interval: str,
        use_cache: bool = True
    ) -> str:
        """
        Get current mode for (symbol, interval).

        Args:
            symbol: Trading pair
            interval: Timeframe
            use_cache: Use in-memory cache (default: True)

        Returns:
            'LEARNING' or 'TRADING'

        Default: 'LEARNING' (conservative - only trade when confident)
        """
        self._stats['mode_queries'] += 1

        key = (symbol, interval)

        # Check cache first
        if use_cache and key in self._mode_cache:
            self._stats['cache_hits'] += 1
            return self._mode_cache[key]

        self._stats['cache_misses'] += 1

        # Query database for latest state
        try:
            latest_state = self.db.get_latest_learning_state(symbol, interval)

            if latest_state:
                mode = latest_state['mode']
                # Update cache
                self._mode_cache[key] = mode
                return mode
            else:
                # No state in database - default to LEARNING
                mode = 'LEARNING'
                self._mode_cache[key] = mode
                return mode

        except Exception as e:
            logger.error(f"Failed to get mode for {symbol} @ {interval}: {e}")
            # Default to LEARNING on error (conservative)
            return 'LEARNING'

    def transition_to_trading(
        self,
        symbol: str,
        interval: str,
        confidence: float,
        reason: str = None
    ) -> int:
        """
        Transition to TRADING mode.

        Args:
            symbol: Trading pair
            interval: Timeframe
            confidence: Current confidence score
            reason: Reason for transition (optional)

        Returns:
            State ID from database
        """
        current_mode = self.get_current_mode(symbol, interval)

        if current_mode == 'TRADING':
            logger.debug(f"{symbol} @ {interval} already in TRADING mode")
            return None

        # Record transition
        state_id = self.db.save_learning_state(
            symbol=symbol,
            interval=interval,
            mode='TRADING',
            confidence=confidence,
            reason=reason or "Confidence threshold reached",
            metadata={
                'previous_mode': current_mode,
                'transition_type': 'LEARNING_TO_TRADING'
            }
        )

        # Update cache
        key = (symbol, interval)
        self._mode_cache[key] = 'TRADING'

        self._stats['transitions'] += 1

        logger.info(
            f"[{symbol} @ {interval}] LEARNING → TRADING "
            f"(confidence: {confidence:.1%})"
        )

        # Also record confidence history
        self._record_confidence(symbol, interval, confidence, 'TRADING')

        return state_id

    def transition_to_learning(
        self,
        symbol: str,
        interval: str,
        reason: str,
        confidence: float = None
    ) -> int:
        """
        Transition to LEARNING mode.

        Args:
            symbol: Trading pair
            interval: Timeframe
            reason: Reason for transition (e.g., "Loss detected", "Confidence dropped")
            confidence: Current confidence score (optional)

        Returns:
            State ID from database
        """
        current_mode = self.get_current_mode(symbol, interval)

        if current_mode == 'LEARNING':
            logger.debug(f"{symbol} @ {interval} already in LEARNING mode")
            return None

        # Record transition
        state_id = self.db.save_learning_state(
            symbol=symbol,
            interval=interval,
            mode='LEARNING',
            confidence=confidence or 0.0,
            reason=reason,
            metadata={
                'previous_mode': current_mode,
                'transition_type': 'TRADING_TO_LEARNING'
            }
        )

        # Update cache
        key = (symbol, interval)
        self._mode_cache[key] = 'LEARNING'

        self._stats['transitions'] += 1

        logger.warning(
            f"[{symbol} @ {interval}] TRADING → LEARNING "
            f"(reason: {reason})"
            + (f", confidence: {confidence:.1%}" if confidence else "")
        )

        # Also record confidence history
        if confidence is not None:
            self._record_confidence(symbol, interval, confidence, 'LEARNING')

        return state_id

    def _record_confidence(
        self,
        symbol: str,
        interval: str,
        confidence: float,
        mode: str
    ):
        """Record confidence score to history in database."""
        try:
            self.db.save_confidence_score(
                symbol=symbol,
                interval=interval,
                confidence=confidence,
                mode=mode
            )
            logger.debug(
                f"Confidence recorded: {symbol} @ {interval} = {confidence:.1%} ({mode})"
            )
        except Exception as e:
            logger.error(f"Failed to record confidence: {e}")

    def get_mode_history(
        self,
        symbol: str,
        interval: str = None,
        limit: int = 10
    ) -> list:
        """
        Get mode transition history.

        Args:
            symbol: Trading pair
            interval: Timeframe (None for all intervals)
            limit: Max number of records

        Returns:
            List of state dicts from database
        """
        try:
            return self.db.get_learning_state_history(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to get mode history: {e}")
            return []

    def get_time_in_mode(
        self,
        symbol: str,
        interval: str,
        mode: str = None
    ) -> Optional[timedelta]:
        """
        Get time spent in current or specific mode.

        Args:
            symbol: Trading pair
            interval: Timeframe
            mode: Specific mode to check, or None for current mode

        Returns:
            Time duration or None if no data
        """
        try:
            latest_state = self.db.get_latest_learning_state(symbol, interval)

            if not latest_state:
                return None

            # If checking specific mode, verify it matches
            if mode and latest_state['mode'] != mode:
                return None

            entered_at = datetime.fromisoformat(latest_state['entered_at'])
            duration = datetime.utcnow() - entered_at

            return duration

        except Exception as e:
            logger.error(f"Failed to calculate time in mode: {e}")
            return None

    def is_stable_in_trading(
        self,
        symbol: str,
        interval: str,
        min_duration_hours: float = 24.0
    ) -> bool:
        """
        Check if system has been stable in TRADING mode.

        Args:
            symbol: Trading pair
            interval: Timeframe
            min_duration_hours: Minimum hours in TRADING mode

        Returns:
            True if stable in TRADING for required duration
        """
        current_mode = self.get_current_mode(symbol, interval)

        if current_mode != 'TRADING':
            return False

        duration = self.get_time_in_mode(symbol, interval)

        if not duration:
            return False

        required_duration = timedelta(hours=min_duration_hours)

        return duration >= required_duration

    def clear_cache(self, symbol: str = None, interval: str = None):
        """
        Clear mode cache.

        Args:
            symbol: If specified, clear only this symbol
            interval: If specified (with symbol), clear only that specific model
        """
        if symbol is None:
            # Clear all
            self._mode_cache.clear()
            logger.info("Cleared all mode cache")
        elif interval is None:
            # Clear all intervals for symbol
            keys_to_remove = [k for k in self._mode_cache.keys() if k[0] == symbol]
            for key in keys_to_remove:
                del self._mode_cache[key]
            logger.info(f"Cleared mode cache for {symbol} (all intervals)")
        else:
            # Clear specific model
            key = (symbol, interval)
            self._mode_cache.pop(key, None)
            logger.info(f"Cleared mode cache for {symbol} @ {interval}")

    def get_stats(self) -> dict:
        """Get state manager statistics."""
        return {
            **self._stats,
            'cached_states': len(self._mode_cache),
            'cache_hit_rate': (
                self._stats['cache_hits'] /
                (self._stats['cache_hits'] + self._stats['cache_misses'])
                if (self._stats['cache_hits'] + self._stats['cache_misses']) > 0
                else 0.0
            )
        }

    def get_all_trading_pairs(self) -> list:
        """
        Get all (symbol, interval) pairs currently in TRADING mode.

        Returns:
            List of (symbol, interval) tuples
        """
        trading_pairs = []

        for key, mode in self._mode_cache.items():
            if mode == 'TRADING':
                trading_pairs.append(key)

        return trading_pairs

    def force_learning_mode(self, symbol: str, interval: str, reason: str):
        """
        Force transition to LEARNING mode (emergency stop).

        Args:
            symbol: Trading pair
            interval: Timeframe
            reason: Reason for forced transition
        """
        logger.warning(
            f"FORCED TRANSITION: {symbol} @ {interval} → LEARNING (reason: {reason})"
        )

        return self.transition_to_learning(
            symbol=symbol,
            interval=interval,
            reason=f"FORCED: {reason}",
            confidence=0.0
        )
