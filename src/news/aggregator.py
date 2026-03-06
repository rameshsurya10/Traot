"""
Sentiment Aggregator
====================

Aggregate news sentiment into candle-aligned features for trading models.

Generates 7 sentiment features:
1. sentiment_1h: Average sentiment over last 1 hour
2. sentiment_6h: Average sentiment over last 6 hours
3. sentiment_24h: Average sentiment over last 24 hours
4. sentiment_momentum: Change in sentiment (1h vs 6h)
5. sentiment_volatility: Volatility of sentiment scores
6. news_volume_1h: Number of articles in last 1 hour
7. source_diversity: Shannon entropy of news sources

These features are added to the existing 32 technical features → Total: 39 features
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
from collections import Counter
import math

from src.core.database import Database
from .types import SentimentFeatures

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """
    Aggregate sentiment features for candles.

    Thread-safe: Can be called from multiple threads
    Stateless: Each call is independent
    """

    def __init__(
        self,
        database: Database,
        config: dict = None
    ):
        """
        Initialize sentiment aggregator.

        Args:
            database: Database instance
            config: Configuration dict from config.yaml['news']['features']
        """
        self.db = database
        self.config = config or {}

        # Configuration
        self.lookback_hours = self.config.get('lookback_hours', [1, 6, 24])
        self.time_weighted = self.config.get('time_weighted', True)
        self.decay_rate = self.config.get('decay_rate', 0.1)

        # Minimum articles required for valid sentiment
        self.min_articles = self.config.get('min_articles', 1)

        # Statistics
        self._stats = {
            'features_generated': 0,
            'insufficient_data_count': 0,
            'avg_articles_per_candle': 0.0
        }

        logger.info(
            f"SentimentAggregator initialized: "
            f"lookback_hours={self.lookback_hours}, "
            f"time_weighted={self.time_weighted}"
        )

    def aggregate_for_candle(
        self,
        symbol: str,
        candle_timestamp: int,
        interval: str = '1h'
    ) -> Optional[SentimentFeatures]:
        """
        Generate sentiment features for a specific candle.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            candle_timestamp: Unix timestamp of the candle
            interval: Timeframe (e.g., '1h')

        Returns:
            SentimentFeatures object or None if insufficient data
        """
        try:
            # Extract base symbol (BTC from BTC/USDT)
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol

            # Get candle datetime
            candle_time = datetime.utcfromtimestamp(candle_timestamp)

            # OPTIMIZED: Single query for longest lookback period (was N+5 queries)
            max_lookback = max(self.lookback_hours)
            since_timestamp = int((candle_time - timedelta(hours=max_lookback)).timestamp())

            # Fetch ALL articles once for the longest period
            all_articles = self.db.get_news_articles(
                symbol=base_symbol,
                since_timestamp=since_timestamp,
                limit=1000,
                processed_only=False  # Get all, filter in memory
            )

            # Pre-compute article timestamps for efficient filtering
            articles_with_ts = []
            for article in all_articles:
                try:
                    article_time = datetime.fromisoformat(article['datetime'])
                    article['_parsed_time'] = article_time
                    articles_with_ts.append(article)
                except (KeyError, ValueError):
                    continue

            # Calculate sentiment for each lookback period (in-memory filtering)
            sentiment_scores = {}
            article_counts = {}

            for hours in self.lookback_hours:
                cutoff_time = candle_time - timedelta(hours=hours)

                # Filter articles for this period (in-memory, no DB call)
                period_articles = [
                    a for a in articles_with_ts
                    if a['_parsed_time'] >= cutoff_time and a.get('sentiment_score') is not None
                ]

                # Calculate average sentiment
                if period_articles:
                    sentiments = [a['sentiment_score'] for a in period_articles]

                    if sentiments:
                        if self.time_weighted:
                            # Apply time decay weighting
                            weights = self._calculate_time_weights(period_articles, candle_time)
                            avg_sentiment = np.average(sentiments, weights=weights)
                        else:
                            avg_sentiment = np.mean(sentiments)

                        sentiment_scores[f'{hours}h'] = float(avg_sentiment)
                        article_counts[f'{hours}h'] = len(period_articles)
                    else:
                        sentiment_scores[f'{hours}h'] = 0.0
                        article_counts[f'{hours}h'] = 0
                else:
                    sentiment_scores[f'{hours}h'] = 0.0
                    article_counts[f'{hours}h'] = 0

            # Calculate momentum (change from 1h to 6h)
            sentiment_1h = sentiment_scores.get('1h', 0.0)
            sentiment_6h = sentiment_scores.get('6h', 0.0)
            sentiment_24h = sentiment_scores.get('24h', 0.0)

            # Momentum: difference between short-term and medium-term sentiment
            sentiment_momentum = sentiment_1h - sentiment_6h

            # Calculate volatility using already-fetched articles (in-memory)
            volatility = self._calculate_volatility_from_articles(
                articles=articles_with_ts,
                candle_time=candle_time,
                hours=self.lookback_hours[0]  # Use shortest lookback for volatility
            )

            # News volume (1 hour)
            news_volume_1h = article_counts.get('1h', 0)

            # Source diversity using already-fetched articles (in-memory)
            source_diversity = self._calculate_source_diversity_from_articles(
                articles=articles_with_ts,
                candle_time=candle_time,
                hours=self.lookback_hours[1]  # Use 6h for diversity
            )

            # Create features object
            features = SentimentFeatures(
                candle_timestamp=candle_timestamp,
                symbol=symbol,
                interval=interval,
                sentiment_1h=sentiment_1h,
                sentiment_6h=sentiment_6h,
                sentiment_24h=sentiment_24h,
                sentiment_momentum=sentiment_momentum,
                sentiment_volatility=volatility,
                news_volume_1h=news_volume_1h,
                source_diversity=source_diversity,
                last_updated=datetime.utcnow().isoformat()
            )

            # Update statistics
            self._stats['features_generated'] += 1
            n = self._stats['features_generated']
            old_avg = self._stats['avg_articles_per_candle']
            self._stats['avg_articles_per_candle'] = (old_avg * (n - 1) + news_volume_1h) / n

            # Check if we have sufficient data
            if news_volume_1h < self.min_articles:
                self._stats['insufficient_data_count'] += 1
                logger.debug(
                    f"Insufficient articles for {symbol} @ {candle_time}: "
                    f"{news_volume_1h} < {self.min_articles}"
                )

            return features

        except Exception as e:
            logger.error(f"Failed to aggregate sentiment for {symbol}: {e}", exc_info=True)
            return None

    def _calculate_time_weights(
        self,
        articles: List[dict],
        candle_time: datetime
    ) -> List[float]:
        """
        Calculate time decay weights for articles.

        More recent articles get higher weights.

        Args:
            articles: List of article dicts
            candle_time: Candle datetime

        Returns:
            List of weights (same length as articles)
        """
        weights = []

        for article in articles:
            # Parse article timestamp
            article_time = datetime.fromisoformat(article['datetime'])

            # Calculate hours old
            hours_old = (candle_time - article_time).total_seconds() / 3600

            # Exponential decay: weight = e^(-decay_rate * hours)
            weight = math.exp(-self.decay_rate * hours_old)
            weights.append(weight)

        return weights

    def _calculate_volatility_from_articles(
        self,
        articles: List[dict],
        candle_time: datetime,
        hours: int
    ) -> float:
        """
        Calculate sentiment volatility from pre-fetched articles (in-memory).

        Args:
            articles: Pre-fetched articles with _parsed_time
            candle_time: Candle datetime
            hours: Hours to look back

        Returns:
            Volatility score (0.0 to ~1.0)
        """
        try:
            cutoff_time = candle_time - timedelta(hours=hours)

            # Filter articles in memory
            period_articles = [
                a for a in articles
                if a.get('_parsed_time') and a['_parsed_time'] >= cutoff_time
            ]

            if not period_articles:
                return 0.0

            sentiments = [
                a['sentiment_score'] for a in period_articles
                if a.get('sentiment_score') is not None
            ]

            if len(sentiments) < 2:
                return 0.0

            return float(np.std(sentiments))

        except Exception as e:
            logger.warning(f"Failed to calculate volatility: {e}")
            return 0.0

    def _calculate_source_diversity_from_articles(
        self,
        articles: List[dict],
        candle_time: datetime,
        hours: int
    ) -> float:
        """
        Calculate source diversity from pre-fetched articles (in-memory).

        Higher entropy = more diverse sources = more reliable sentiment.

        Args:
            articles: Pre-fetched articles with _parsed_time
            candle_time: Candle datetime
            hours: Hours to look back

        Returns:
            Diversity score (0.0 to ~1.0)
        """
        try:
            cutoff_time = candle_time - timedelta(hours=hours)

            # Filter articles in memory
            period_articles = [
                a for a in articles
                if a.get('_parsed_time') and a['_parsed_time'] >= cutoff_time
            ]

            if not period_articles:
                return 0.0

            # Count sources
            sources = [a['source'] for a in period_articles if a.get('source')]

            if not sources:
                return 0.0

            # Calculate Shannon entropy
            source_counts = Counter(sources)
            total = len(sources)
            entropy = 0.0

            for count in source_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)

            # Normalize to 0-1 (max entropy for 3 sources is log2(3) ≈ 1.58)
            max_entropy = math.log2(min(len(source_counts), 3))  # Cap at 3 sources
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            return float(normalized_entropy)

        except Exception as e:
            logger.warning(f"Failed to calculate source diversity: {e}")
            return 0.0

    # Legacy methods kept for backwards compatibility (but unused)
    def _calculate_volatility(
        self,
        base_symbol: str,
        candle_time: datetime,
        hours: int
    ) -> float:
        """Legacy method - use _calculate_volatility_from_articles instead."""
        try:
            since_timestamp = int((candle_time - timedelta(hours=hours)).timestamp())

            articles = self.db.get_news_articles(
                symbol=base_symbol,
                since_timestamp=since_timestamp,
                limit=1000,
                processed_only=True
            )

            if not articles:
                return 0.0

            sentiments = [a['sentiment_score'] for a in articles if a.get('sentiment_score') is not None]

            if len(sentiments) < 2:
                return 0.0

            return float(np.std(sentiments))

        except Exception as e:
            logger.warning(f"Failed to calculate volatility: {e}")
            return 0.0

    def _calculate_source_diversity(
        self,
        base_symbol: str,
        candle_time: datetime,
        hours: int
    ) -> float:
        """Legacy method - use _calculate_source_diversity_from_articles instead."""
        try:
            since_timestamp = int((candle_time - timedelta(hours=hours)).timestamp())

            articles = self.db.get_news_articles(
                symbol=base_symbol,
                since_timestamp=since_timestamp,
                limit=1000,
                processed_only=False  # All articles
            )

            if not articles:
                return 0.0

            # Count sources
            sources = [a['source'] for a in articles if a.get('source')]

            if not sources:
                return 0.0

            # Calculate Shannon entropy
            source_counts = Counter(sources)
            total = len(sources)
            entropy = 0.0

            for count in source_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)

            # Normalize to 0-1 (max entropy for 3 sources is log2(3) ≈ 1.58)
            max_entropy = math.log2(min(len(source_counts), 3))  # Cap at 3 sources
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            return float(normalized_entropy)

        except Exception as e:
            logger.warning(f"Failed to calculate source diversity: {e}")
            return 0.0

    def store_features(self, features: SentimentFeatures) -> bool:
        """
        Store sentiment features in database.

        Args:
            features: SentimentFeatures object

        Returns:
            True if successful
        """
        try:
            # Unpack dict to match method signature
            self.db.save_sentiment_features(**features.to_dict())
            return True
        except Exception as e:
            logger.error(f"Failed to store sentiment features: {e}")
            return False

    def get_features_for_candle(
        self,
        symbol: str,
        candle_timestamp: int,
        generate_if_missing: bool = True
    ) -> Optional[SentimentFeatures]:
        """
        Get sentiment features for a candle (from DB or generate).

        Args:
            symbol: Trading pair
            candle_timestamp: Unix timestamp
            generate_if_missing: Generate if not in database

        Returns:
            SentimentFeatures object or None
        """
        # Try to get from database first
        try:
            features_dict = self.db.get_sentiment_features(candle_timestamp)

            if features_dict:
                # Convert dict to SentimentFeatures object
                return SentimentFeatures(
                    candle_timestamp=features_dict['candle_timestamp'],
                    symbol=features_dict.get('symbol', symbol),
                    interval=features_dict.get('interval', '1h'),
                    sentiment_1h=features_dict.get('sentiment_1h', 0.0),
                    sentiment_6h=features_dict.get('sentiment_6h', 0.0),
                    sentiment_24h=features_dict.get('sentiment_24h', 0.0),
                    sentiment_momentum=features_dict.get('sentiment_momentum', 0.0),
                    sentiment_volatility=features_dict.get('sentiment_volatility', 0.0),
                    news_volume_1h=features_dict.get('news_volume_1h', 0),
                    source_diversity=features_dict.get('source_diversity', 0.0),
                    last_updated=features_dict.get('last_updated')
                )

        except Exception as e:
            logger.warning(f"Failed to get features from database: {e}")

        # Generate if missing and requested
        if generate_if_missing:
            logger.debug(f"Generating sentiment features for {symbol} @ {candle_timestamp}")
            features = self.aggregate_for_candle(symbol, candle_timestamp)

            if features:
                # Store for future use
                self.store_features(features)

            return features

        return None

    def get_stats(self) -> dict:
        """Get aggregator statistics."""
        return {
            **self._stats,
            'insufficient_data_ratio': (
                self._stats['insufficient_data_count'] /
                max(self._stats['features_generated'], 1)
            )
        }
