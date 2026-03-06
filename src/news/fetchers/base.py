"""
Base News Fetcher
==================

Abstract base class for news fetchers.

All news fetchers must inherit from this class and implement the fetch() method.
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from typing import List
from datetime import datetime

from ..types import Article

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """
    Abstract base class for news fetchers.

    All fetchers must implement:
    - fetch(): Fetch news articles for given symbols
    """

    def __init__(self, config: dict = None):
        """
        Initialize fetcher.

        Args:
            config: Configuration dict from config.yaml
        """
        self.config = config or {}
        self.name = self.__class__.__name__

        # Statistics
        self._stats = {
            'articles_fetched': 0,
            'fetch_errors': 0,
            'duplicates_skipped': 0,
            'last_fetch_time': None
        }

    @abstractmethod
    def fetch(
        self,
        symbols: List[str],
        lookback_hours: int = 6,
        max_articles: int = 100
    ) -> List[Article]:
        """
        Fetch news articles for given symbols.

        Args:
            symbols: List of trading symbols (e.g., ['BTC', 'ETH'])
            lookback_hours: How many hours back to fetch
            max_articles: Maximum number of articles to return

        Returns:
            List of Article objects

        Raises:
            Exception: If fetch fails
        """
        pass

    def _create_content_hash(self, article_data: dict) -> str:
        """
        Create SHA256 hash for article deduplication.

        Args:
            article_data: Dict with article data

        Returns:
            SHA256 hash string
        """
        # Use title + url for hash (most stable identifiers)
        hash_input = f"{article_data.get('title', '')}{article_data.get('url', '')}"
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    def _extract_symbols(self, text: str, known_symbols: List[str]) -> List[str]:
        """
        Extract symbols mentioned in text.

        Args:
            text: Text to search for symbols
            known_symbols: List of symbols to look for (e.g., ['BTC', 'ETH'])

        Returns:
            List of symbols found in text
        """
        if not text:
            return []

        text_upper = text.upper()
        found_symbols = []

        for symbol in known_symbols:
            # Check for exact symbol match (e.g., "BTC")
            if symbol.upper() in text_upper:
                found_symbols.append(symbol)

            # Also check for common variations (e.g., "BITCOIN" for BTC)
            variations = self._get_symbol_variations(symbol)
            for variation in variations:
                if variation.upper() in text_upper and symbol not in found_symbols:
                    found_symbols.append(symbol)

        return found_symbols

    def _get_symbol_variations(self, symbol: str) -> List[str]:
        """
        Get common variations of a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC')

        Returns:
            List of variations (e.g., ['BITCOIN', 'BTC'])
        """
        variations = {
            'BTC': ['BITCOIN', 'BTC'],
            'ETH': ['ETHEREUM', 'ETH'],
            'USDT': ['TETHER', 'USDT'],
            'BNB': ['BINANCE COIN', 'BNB'],
            'SOL': ['SOLANA', 'SOL'],
            'ADA': ['CARDANO', 'ADA'],
            'XRP': ['RIPPLE', 'XRP'],
            'DOT': ['POLKADOT', 'DOT'],
            'DOGE': ['DOGECOIN', 'DOGE'],
            'AVAX': ['AVALANCHE', 'AVAX']
        }

        return variations.get(symbol.upper(), [symbol])

    def _calculate_relevance_score(
        self,
        article: Article,
        primary_symbol: str
    ) -> float:
        """
        Calculate relevance score for an article.

        Args:
            article: Article object
            primary_symbol: Primary symbol to calculate relevance for

        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0

        # Title mentions (most important)
        if article.title and primary_symbol.upper() in article.title.upper():
            score += 0.5

        # Description mentions
        if article.description and primary_symbol.upper() in article.description.upper():
            score += 0.3

        # Content mentions
        if article.content and primary_symbol.upper() in article.content.upper():
            score += 0.2

        return min(score, 1.0)

    def get_stats(self) -> dict:
        """Get fetcher statistics."""
        return {
            'name': self.name,
            **self._stats
        }

    def _update_stats(self, articles_count: int, error: bool = False):
        """Update statistics."""
        if error:
            self._stats['fetch_errors'] += 1
        else:
            self._stats['articles_fetched'] += articles_count

        self._stats['last_fetch_time'] = datetime.utcnow().isoformat()
