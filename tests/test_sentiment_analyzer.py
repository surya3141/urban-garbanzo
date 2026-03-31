"""
Unit tests for the SentimentAnalyzer.

The tests stub out the HuggingFace pipeline so that no model download
is required in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sentiment_analyzer import SentimentAnalyzer, SentimentResult, _LABEL_MAP


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_pipeline_response(label: str, score: float):
    """Return a mock pipeline call that emits *label* and *score*."""
    mock_pipeline = MagicMock(return_value=[{"label": label, "score": score}])
    return mock_pipeline


def _make_batch_pipeline_response(items):
    """Return a mock pipeline that emits a list of results."""
    mock_pipeline = MagicMock(return_value=items)
    return mock_pipeline


@pytest.fixture()
def analyzer_positive():
    with patch("sentiment_analyzer.pipeline") as mock_pipe_factory:
        mock_pipe_factory.return_value = _make_pipeline_response("positive", 0.95)
        analyzer = SentimentAnalyzer()
    return analyzer


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("positive", "positive"),
        ("negative", "negative"),
        ("neutral", "neutral"),
        ("LABEL_0", "negative"),
        ("LABEL_1", "neutral"),
        ("LABEL_2", "positive"),
    ],
)
def test_label_map_normalisation(raw: str, expected: str) -> None:
    assert _LABEL_MAP.get(raw.lower(), raw.lower()) == expected or _LABEL_MAP.get(raw, raw) == expected


# ---------------------------------------------------------------------------
# SentimentAnalyzer.analyze
# ---------------------------------------------------------------------------


def test_analyze_positive_label() -> None:
    with patch("sentiment_analyzer.pipeline") as mock_pipe_factory:
        mock_pipeline = MagicMock(return_value=[{"label": "positive", "score": 0.98}])
        mock_pipe_factory.return_value = mock_pipeline
        analyzer = SentimentAnalyzer()

    result = analyzer.analyze("I love this product!")
    assert isinstance(result, SentimentResult)
    assert result.label == "positive"
    assert pytest.approx(result.score, abs=1e-4) == 0.98
    assert result.text == "I love this product!"


def test_analyze_negative_label() -> None:
    with patch("sentiment_analyzer.pipeline") as mock_pipe_factory:
        mock_pipeline = MagicMock(return_value=[{"label": "negative", "score": 0.92}])
        mock_pipe_factory.return_value = mock_pipeline
        analyzer = SentimentAnalyzer()

    result = analyzer.analyze("This is terrible.")
    assert result.label == "negative"
    assert result.score == pytest.approx(0.92, abs=1e-4)


def test_analyze_neutral_label() -> None:
    with patch("sentiment_analyzer.pipeline") as mock_pipe_factory:
        mock_pipeline = MagicMock(return_value=[{"label": "neutral", "score": 0.75}])
        mock_pipe_factory.return_value = mock_pipeline
        analyzer = SentimentAnalyzer()

    result = analyzer.analyze("The sky is blue.")
    assert result.label == "neutral"


def test_analyze_cardiffnlp_label_mapping() -> None:
    """LABEL_2 from the CardiffNLP model should map to 'positive'."""
    with patch("sentiment_analyzer.pipeline") as mock_pipe_factory:
        mock_pipeline = MagicMock(return_value=[{"label": "LABEL_2", "score": 0.85}])
        mock_pipe_factory.return_value = mock_pipeline
        analyzer = SentimentAnalyzer()

    result = analyzer.analyze("Great day!")
    assert result.label == "positive"
    assert result.raw_label == "LABEL_2"


# ---------------------------------------------------------------------------
# SentimentAnalyzer.analyze_batch
# ---------------------------------------------------------------------------


def test_analyze_batch_empty() -> None:
    with patch("sentiment_analyzer.pipeline") as mock_pipe_factory:
        mock_pipe_factory.return_value = MagicMock()
        analyzer = SentimentAnalyzer()

    results = analyzer.analyze_batch([])
    assert results == []


def test_analyze_batch_multiple() -> None:
    batch_output = [
        {"label": "positive", "score": 0.9},
        {"label": "negative", "score": 0.8},
        {"label": "neutral", "score": 0.6},
    ]
    with patch("sentiment_analyzer.pipeline") as mock_pipe_factory:
        mock_pipeline = MagicMock(return_value=batch_output)
        mock_pipe_factory.return_value = mock_pipeline
        analyzer = SentimentAnalyzer()

    texts = ["Great!", "Awful!", "Okay."]
    results = analyzer.analyze_batch(texts)
    assert len(results) == 3
    assert results[0].label == "positive"
    assert results[1].label == "negative"
    assert results[2].label == "neutral"
    for r, t in zip(results, texts):
        assert r.text == t


def test_analyze_batch_preserves_order() -> None:
    labels = ["negative", "positive", "neutral", "positive", "negative"]
    batch_output = [{"label": lbl, "score": 0.7} for lbl in labels]
    with patch("sentiment_analyzer.pipeline") as mock_pipe_factory:
        mock_pipeline = MagicMock(return_value=batch_output)
        mock_pipe_factory.return_value = mock_pipeline
        analyzer = SentimentAnalyzer()

    texts = [f"text_{i}" for i in range(5)]
    results = analyzer.analyze_batch(texts)
    assert [r.label for r in results] == labels
