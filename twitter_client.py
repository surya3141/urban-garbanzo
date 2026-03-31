"""
Twitter / social-media data client.

Fetches recent posts for a given hashtag or keyword using the
Twitter API v2 (via Tweepy).  When Twitter credentials are not
available the client falls back to built-in mock data so the
application can still be demonstrated without API access.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import tweepy
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Post:
    """Represents a single social-media post."""

    id: str
    text: str
    created_at: datetime
    author_id: str = ""
    source: str = "twitter"
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_MOCK_TEMPLATES = [
    "Really loving {query} today! Such a great experience. 😊",
    "I can't stand {query}. Absolute disaster every single time. 😤",
    "{query} is okay I guess. Nothing special, nothing terrible.",
    "Just tried {query} for the first time — totally blown away! 🚀",
    "Why does {query} keep letting us down? So disappointing.",
    "Neutral thoughts on {query}: has pros and cons.",
    "{query} is changing the game! This is the future. 🔥",
    "Not impressed with {query} at all. Expected much better.",
    "{query} — meh. Neither here nor there honestly.",
    "Absolutely thrilled about {query}! Best thing this year! ❤️",
    "Serious problems with {query} again. When will they fix it?",
    "{query} is fine but I've seen better.",
    "Breaking: {query} just hit a new milestone! Amazing news.",
    "{query} drama unfolding — not a good look.",
    "Genuinely undecided about {query}. Mixed feelings.",
]


def _make_mock_posts(query: str, count: int) -> List[Post]:
    """Generate *count* mock posts for *query*."""
    random.seed(hash(query) % (2**31))
    posts = []
    now = time.time()
    templates = _MOCK_TEMPLATES * (count // len(_MOCK_TEMPLATES) + 1)
    for i in range(count):
        text = templates[i].format(query=query)
        created_at = datetime.fromtimestamp(now - i * 120, tz=timezone.utc)
        posts.append(
            Post(
                id=f"mock_{i}",
                text=text,
                created_at=created_at,
                author_id=f"user_{random.randint(1000, 9999)}",
                source="mock",
                like_count=random.randint(0, 500),
                retweet_count=random.randint(0, 100),
                reply_count=random.randint(0, 50),
            )
        )
    return posts


# ---------------------------------------------------------------------------
# Twitter client
# ---------------------------------------------------------------------------


class TwitterClient:
    """Fetches recent posts via the Twitter API v2, with mock-data fallback."""

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        use_mock: bool = False,
    ) -> None:
        self._use_mock = use_mock
        self._client: Optional[tweepy.Client] = None

        if not use_mock:
            token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
            if token:
                self._client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
            else:
                self._use_mock = True  # silently fall back to mock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_mock(self) -> bool:
        """True when the client is operating in mock / demo mode."""
        return self._use_mock

    def fetch_recent_posts(
        self,
        query: str,
        max_results: int = 100,
    ) -> List[Post]:
        """Return up to *max_results* recent posts matching *query*.

        Falls back to mock data automatically when Twitter credentials are
        not configured.
        """
        if self._use_mock or self._client is None:
            return _make_mock_posts(query, max_results)

        return self._fetch_from_twitter(query, max_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_from_twitter(self, query: str, max_results: int) -> List[Post]:
        """Call Twitter API v2 recent-search endpoint."""
        safe_query = f"({query}) -is:retweet lang:en"
        # Twitter API caps max_results between 10 and 100 per request
        per_request = max(10, min(max_results, 100))

        response = self._client.search_recent_tweets(
            query=safe_query,
            max_results=per_request,
            tweet_fields=["created_at", "author_id", "public_metrics"],
        )

        if not response.data:
            return []

        posts: List[Post] = []
        for tweet in response.data:
            metrics = tweet.public_metrics or {}
            posts.append(
                Post(
                    id=str(tweet.id),
                    text=tweet.text,
                    created_at=tweet.created_at or datetime.now(tz=timezone.utc),
                    author_id=str(tweet.author_id or ""),
                    source="twitter",
                    like_count=metrics.get("like_count", 0),
                    retweet_count=metrics.get("retweet_count", 0),
                    reply_count=metrics.get("reply_count", 0),
                )
            )
        return posts
