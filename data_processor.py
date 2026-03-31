"""
Data-processing utilities for aggregating and summarizing sentiment results.

Converts raw SentimentResult lists into DataFrames and summary statistics
that the Streamlit UI can visualise directly.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd

from sentiment_analyzer import SentimentResult
from twitter_client import Post


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def build_dataframe(posts: List[Post], results: List[SentimentResult]) -> pd.DataFrame:
    """Combine *posts* and their corresponding *results* into a single DataFrame.

    Columns: id, text, created_at, source, label, score,
             like_count, retweet_count, reply_count.
    """
    if len(posts) != len(results):
        raise ValueError(
            f"posts ({len(posts)}) and results ({len(results)}) must have the same length."
        )

    rows = []
    for post, result in zip(posts, results):
        rows.append(
            {
                "id": post.id,
                "text": post.text,
                "created_at": post.created_at,
                "source": post.source,
                "label": result.label,
                "score": result.score,
                "like_count": post.like_count,
                "retweet_count": post.retweet_count,
                "reply_count": post.reply_count,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df.sort_values("created_at", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def compute_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Return a dictionary with high-level sentiment statistics.

    Keys: total, positive, negative, neutral, positive_pct,
          negative_pct, neutral_pct, avg_score.
    """
    if df.empty:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "avg_score": 0.0,
        }

    counts = Counter(df["label"])
    total = len(df)
    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    neutral = counts.get("neutral", 0)

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_pct": round(positive / total * 100, 1),
        "negative_pct": round(negative / total * 100, 1),
        "neutral_pct": round(neutral / total * 100, 1),
        "avg_score": round(float(df["score"].mean()), 3),
    }


def sentiment_over_time(df: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """Resample *df* by *freq* and count sentiment labels per time bucket.

    Returns a DataFrame with columns: created_at, positive, negative, neutral.
    """
    if df.empty:
        return pd.DataFrame(columns=["created_at", "positive", "negative", "neutral"])

    tmp = df.set_index("created_at")
    grouped = (
        tmp.groupby([pd.Grouper(freq=freq), "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    grouped.columns.name = None
    # Ensure all three sentiment columns are present
    for col in ("positive", "negative", "neutral"):
        if col not in grouped.columns:
            grouped[col] = 0
    grouped.rename(columns={"created_at": "created_at"}, inplace=True)
    return grouped[["created_at", "positive", "negative", "neutral"]]


def top_posts(df: pd.DataFrame, label: str, n: int = 5) -> pd.DataFrame:
    """Return the *n* highest-scored posts with the given sentiment *label*."""
    filtered = df[df["label"] == label].copy()
    return filtered.nlargest(n, "score")[["text", "label", "score", "created_at"]]
