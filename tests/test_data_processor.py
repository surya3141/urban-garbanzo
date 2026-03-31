"""
Unit tests for the data_processor module.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from data_processor import build_dataframe, compute_summary, sentiment_over_time, top_posts
from sentiment_analyzer import SentimentResult
from twitter_client import Post


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _post(i: int, created_offset_min: int = 0) -> Post:
    return Post(
        id=str(i),
        text=f"Post number {i}",
        created_at=datetime(2024, 1, 1, 12, created_offset_min, 0, tzinfo=timezone.utc),
        author_id=f"user_{i}",
        source="mock",
        like_count=i * 10,
        retweet_count=i,
        reply_count=i,
    )


def _result(text: str, label: str, score: float = 0.9) -> SentimentResult:
    return SentimentResult(text=text, label=label, score=score)


# ---------------------------------------------------------------------------
# build_dataframe
# ---------------------------------------------------------------------------


def test_build_dataframe_basic() -> None:
    posts = [_post(0, 0), _post(1, 10), _post(2, 20)]
    results = [
        _result(p.text, lbl)
        for p, lbl in zip(posts, ["positive", "negative", "neutral"])
    ]
    df = build_dataframe(posts, results)
    assert len(df) == 3
    assert list(df.columns) >= ["id", "text", "created_at", "label", "score"]
    assert set(df["label"]) == {"positive", "negative", "neutral"}


def test_build_dataframe_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        build_dataframe([_post(0)], [])


def test_build_dataframe_empty() -> None:
    df = build_dataframe([], [])
    assert df.empty


def test_build_dataframe_sorted_by_time() -> None:
    posts = [_post(0, 30), _post(1, 10), _post(2, 20)]
    results = [_result(p.text, "neutral") for p in posts]
    df = build_dataframe(posts, results)
    times = list(df["created_at"])
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------


def test_compute_summary_basic() -> None:
    posts = [_post(i, i * 5) for i in range(6)]
    labels = ["positive", "positive", "negative", "neutral", "positive", "neutral"]
    results = [_result(p.text, lbl) for p, lbl in zip(posts, labels)]
    df = build_dataframe(posts, results)
    s = compute_summary(df)

    assert s["total"] == 6
    assert s["positive"] == 3
    assert s["negative"] == 1
    assert s["neutral"] == 2
    assert s["positive_pct"] == pytest.approx(50.0, abs=0.1)
    assert s["negative_pct"] == pytest.approx(16.7, abs=0.1)
    assert s["neutral_pct"] == pytest.approx(33.3, abs=0.1)


def test_compute_summary_empty_dataframe() -> None:
    s = compute_summary(pd.DataFrame())
    assert s["total"] == 0
    assert s["positive_pct"] == 0.0
    assert s["avg_score"] == 0.0


def test_compute_summary_all_positive() -> None:
    posts = [_post(i, i) for i in range(4)]
    results = [_result(p.text, "positive", 0.95) for p in posts]
    df = build_dataframe(posts, results)
    s = compute_summary(df)
    assert s["positive_pct"] == 100.0
    assert s["negative_pct"] == 0.0
    assert s["avg_score"] == pytest.approx(0.95, abs=1e-3)


# ---------------------------------------------------------------------------
# sentiment_over_time
# ---------------------------------------------------------------------------


def test_sentiment_over_time_basic() -> None:
    # 6 posts spread over 60 minutes
    posts = [_post(i, i * 10) for i in range(6)]
    labels = ["positive", "positive", "negative", "neutral", "negative", "positive"]
    results = [_result(p.text, lbl) for p, lbl in zip(posts, labels)]
    df = build_dataframe(posts, results)
    time_df = sentiment_over_time(df, freq="20min")
    assert not time_df.empty
    assert "positive" in time_df.columns
    assert "negative" in time_df.columns
    assert "neutral" in time_df.columns


def test_sentiment_over_time_empty() -> None:
    result = sentiment_over_time(pd.DataFrame(), freq="10min")
    assert result.empty


# ---------------------------------------------------------------------------
# top_posts
# ---------------------------------------------------------------------------


def test_top_posts_returns_correct_label() -> None:
    posts = [_post(i, i * 5) for i in range(5)]
    labels = ["positive", "negative", "positive", "neutral", "positive"]
    scores = [0.9, 0.8, 0.95, 0.6, 0.85]
    results = [_result(p.text, lbl, sc) for p, lbl, sc in zip(posts, labels, scores)]
    df = build_dataframe(posts, results)
    top = top_posts(df, "positive", n=2)
    assert len(top) == 2
    assert all(top["label"] == "positive")
    # Highest-scored positive first
    assert list(top["score"]) == sorted(top["score"], reverse=True)


def test_top_posts_missing_label() -> None:
    posts = [_post(i, i) for i in range(3)]
    results = [_result(p.text, "positive") for p in posts]
    df = build_dataframe(posts, results)
    top = top_posts(df, "negative", n=5)
    assert top.empty
