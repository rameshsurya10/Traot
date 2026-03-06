"""
News Data Types
===============

Data classes for news articles and sentiment analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Article:
    """
    News article data structure.

    Attributes:
        timestamp: Unix timestamp
        datetime: ISO datetime string
        source: Source of the article ('newsapi', 'alphavantage', 'reddit')
        title: Article title
        description: Article description/summary
        content: Full article content
        url: Article URL
        sentiment_score: Sentiment score (-1.0 to 1.0)
        sentiment_label: Sentiment label ('positive', 'negative', 'neutral')
        sentiment_compound: VADER compound score
        symbols: List of symbols mentioned in article
        primary_symbol: Primary symbol this article is about
        relevance_score: Relevance score (0.0 to 1.0)
        processed: Whether sentiment analysis has been performed
        content_hash: SHA256 hash for deduplication
    """
    timestamp: int
    datetime: str
    source: str
    title: str
    description: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None

    # Sentiment (computed)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    sentiment_compound: Optional[float] = None

    # Symbol tracking
    symbols: List[str] = field(default_factory=list)
    primary_symbol: Optional[str] = None

    # Metadata
    relevance_score: Optional[float] = None
    processed: bool = False
    content_hash: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'timestamp': self.timestamp,
            'datetime': self.datetime,
            'source': self.source,
            'title': self.title,
            'description': self.description,
            'content': self.content,
            'url': self.url,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'sentiment_compound': self.sentiment_compound,
            'symbols': ','.join(self.symbols) if self.symbols else None,
            'primary_symbol': self.primary_symbol,
            'relevance_score': self.relevance_score,
            'processed': int(self.processed),
            'content_hash': self.content_hash
        }


@dataclass
class SentimentResult:
    """
    Sentiment analysis result from VADER.

    Attributes:
        score: Overall sentiment score (-1.0 to 1.0)
        label: Sentiment label ('positive', 'negative', 'neutral')
        compound: VADER compound score
        positive: Positive score (0.0 to 1.0)
        negative: Negative score (0.0 to 1.0)
        neutral: Neutral score (0.0 to 1.0)
        confidence: Confidence in sentiment classification (0.0 to 1.0)
    """
    score: float
    label: str
    compound: float
    positive: float
    negative: float
    neutral: float
    confidence: float = 0.0

    @classmethod
    def from_vader(cls, vader_scores: dict) -> 'SentimentResult':
        """
        Create SentimentResult from VADER scores.

        Args:
            vader_scores: Dict with keys 'compound', 'pos', 'neg', 'neu'

        Returns:
            SentimentResult instance
        """
        compound = vader_scores['compound']

        # Classify sentiment based on compound score
        if compound >= 0.05:
            label = 'positive'
            score = compound
        elif compound <= -0.05:
            label = 'negative'
            score = compound
        else:
            label = 'neutral'
            score = 0.0

        # Confidence is the absolute value of compound score
        confidence = abs(compound)

        return cls(
            score=score,
            label=label,
            compound=compound,
            positive=vader_scores['pos'],
            negative=vader_scores['neg'],
            neutral=vader_scores['neu'],
            confidence=confidence
        )


@dataclass
class SentimentFeatures:
    """
    Aggregated sentiment features for a specific candle.

    These features will be added to the existing 32 technical features.
    Total: 32 + 7 = 39 features

    Attributes:
        candle_timestamp: Unix timestamp of the candle
        symbol: Trading pair
        interval: Timeframe
        sentiment_1h: Average sentiment over last 1 hour
        sentiment_6h: Average sentiment over last 6 hours
        sentiment_24h: Average sentiment over last 24 hours
        sentiment_momentum: Change in sentiment (1h vs 6h)
        sentiment_volatility: Volatility of sentiment scores
        news_volume_1h: Number of articles in last 1 hour
        source_diversity: Shannon entropy of news sources
        last_updated: ISO datetime of last update
    """
    candle_timestamp: int
    symbol: str
    interval: str

    # Sentiment scores
    sentiment_1h: float = 0.0
    sentiment_6h: float = 0.0
    sentiment_24h: float = 0.0

    # Momentum and volatility
    sentiment_momentum: float = 0.0
    sentiment_volatility: float = 0.0

    # Volume and diversity
    news_volume_1h: int = 0
    source_diversity: float = 0.0

    last_updated: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'candle_timestamp': self.candle_timestamp,
            'symbol': self.symbol,
            'interval': self.interval,
            'sentiment_1h': self.sentiment_1h,
            'sentiment_6h': self.sentiment_6h,
            'sentiment_24h': self.sentiment_24h,
            'sentiment_momentum': self.sentiment_momentum,
            'sentiment_volatility': self.sentiment_volatility,
            'news_volume_1h': self.news_volume_1h,
            'source_diversity': self.source_diversity,
            'last_updated': self.last_updated or datetime.utcnow().isoformat()
        }

    def to_feature_vector(self) -> List[float]:
        """
        Convert to feature vector for model input.

        Returns:
            List of 7 sentiment features in order:
            [sentiment_1h, sentiment_6h, sentiment_24h, sentiment_momentum,
             sentiment_volatility, news_volume_1h (normalized), source_diversity]
        """
        return [
            self.sentiment_1h,
            self.sentiment_6h,
            self.sentiment_24h,
            self.sentiment_momentum,
            self.sentiment_volatility,
            min(self.news_volume_1h / 10.0, 1.0),  # Normalize to 0-1 (cap at 10 articles)
            self.source_diversity
        ]
