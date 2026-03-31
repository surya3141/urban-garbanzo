"""
Unit tests for TwitterClient — mock and live-credential branches.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from twitter_client import Post, TwitterClient, _make_mock_posts


# ---------------------------------------------------------------------------
# _make_mock_posts
# ---------------------------------------------------------------------------


def test_make_mock_posts_count() -> None:
    posts = _make_mock_posts("python", 20)
    assert len(posts) == 20


def test_make_mock_posts_contain_query() -> None:
    posts = _make_mock_posts("openai", 5)
    for post in posts:
        assert "openai" in post.text.lower()


def test_make_mock_posts_have_timestamps() -> None:
    posts = _make_mock_posts("test", 3)
    for post in posts:
        assert isinstance(post.created_at, datetime)
        assert post.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# TwitterClient — mock mode
# ---------------------------------------------------------------------------


def test_client_falls_back_to_mock_without_token() -> None:
    client = TwitterClient(bearer_token=None, use_mock=False)
    assert client.is_mock is True


def test_client_mock_mode_explicit() -> None:
    client = TwitterClient(use_mock=True)
    assert client.is_mock is True


def test_client_mock_fetch_returns_posts() -> None:
    client = TwitterClient(use_mock=True)
    posts = client.fetch_recent_posts("ai", max_results=15)
    assert len(posts) == 15
    for post in posts:
        assert isinstance(post, Post)
        assert post.source == "mock"


def test_client_mock_post_fields() -> None:
    client = TwitterClient(use_mock=True)
    posts = client.fetch_recent_posts("streamlit", max_results=5)
    for post in posts:
        assert post.id.startswith("mock_")
        assert isinstance(post.like_count, int)
        assert isinstance(post.retweet_count, int)


# ---------------------------------------------------------------------------
# TwitterClient — live mode (mocked Tweepy)
# ---------------------------------------------------------------------------


def _make_mock_tweet(i: int):
    tweet = MagicMock()
    tweet.id = i
    tweet.text = f"Live tweet {i}"
    tweet.created_at = datetime(2024, 1, 1, 12, i, 0, tzinfo=timezone.utc)
    tweet.author_id = f"author_{i}"
    tweet.public_metrics = {"like_count": i * 5, "retweet_count": i, "reply_count": i}
    return tweet


def test_client_live_mode_with_bearer_token() -> None:
    with patch("twitter_client.tweepy.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        client = TwitterClient(bearer_token="fake_token", use_mock=False)

    assert client.is_mock is False


def test_client_live_fetch_maps_tweets_to_posts() -> None:
    tweets = [_make_mock_tweet(i) for i in range(3)]
    mock_response = MagicMock()
    mock_response.data = tweets

    with patch("twitter_client.tweepy.Client") as mock_client_cls:
        mock_tweepy = MagicMock()
        mock_tweepy.search_recent_tweets.return_value = mock_response
        mock_client_cls.return_value = mock_tweepy
        client = TwitterClient(bearer_token="fake_token", use_mock=False)

    posts = client.fetch_recent_posts("python", max_results=3)
    assert len(posts) == 3
    for post in posts:
        assert isinstance(post, Post)
        assert post.source == "twitter"
        assert "Live tweet" in post.text


def test_client_live_fetch_empty_response() -> None:
    mock_response = MagicMock()
    mock_response.data = None

    with patch("twitter_client.tweepy.Client") as mock_client_cls:
        mock_tweepy = MagicMock()
        mock_tweepy.search_recent_tweets.return_value = mock_response
        mock_client_cls.return_value = mock_tweepy
        client = TwitterClient(bearer_token="fake_token", use_mock=False)

    posts = client.fetch_recent_posts("noresults", max_results=10)
    assert posts == []
