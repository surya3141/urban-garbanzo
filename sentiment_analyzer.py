"""
Sentiment analysis module using HuggingFace Transformers.

Uses a pre-trained model fine-tuned on Twitter data for accurate
social media sentiment classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from transformers import pipeline


SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    # cardiffnlp model labels (stored lowercase for case-insensitive lookup)
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}


@dataclass
class SentimentResult:
    """Holds the sentiment classification result for a single text."""

    text: str
    label: str  # 'positive', 'negative', or 'neutral'
    score: float  # confidence score in [0, 1]
    raw_label: str = field(default="", repr=False)


class SentimentAnalyzer:
    """Analyzes sentiment of social media posts using a pre-trained NLP model."""

    def __init__(self, model_name: str = SENTIMENT_MODEL) -> None:
        self._model_name = model_name
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=512,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> SentimentResult:
        """Classify the sentiment of a single *text* string."""
        result = self._pipeline(text)[0]
        raw_label: str = result["label"]
        normalized = _LABEL_MAP.get(raw_label.lower(), raw_label.lower())
        return SentimentResult(
            text=text,
            label=normalized,
            score=float(result["score"]),
            raw_label=raw_label,
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Classify the sentiment of a list of texts in a single batched call."""
        if not texts:
            return []
        results = self._pipeline(texts)
        sentiment_results = []
        for text, result in zip(texts, results):
            raw_label: str = result["label"]
            normalized = _LABEL_MAP.get(raw_label.lower(), raw_label.lower())
            sentiment_results.append(
                SentimentResult(
                    text=text,
                    label=normalized,
                    score=float(result["score"]),
                    raw_label=raw_label,
                )
            )
        return sentiment_results
