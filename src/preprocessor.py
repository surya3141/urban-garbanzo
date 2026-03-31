"""
preprocessor.py
---------------
Text preprocessing pipeline for the Fake News Detector.

Steps:
  1. HTML / URL / special-character removal
  2. Lower-casing
  3. Tokenisation (word_tokenize)
  4. Stop-word removal
  5. TF-IDF vectorisation (fit or transform)
"""

import re
import string
import logging
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK data – download once, silently skip if already present
# ---------------------------------------------------------------------------
for _pkg in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_STOP_WORDS: set = set(stopwords.words("english"))

# Regex helpers
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_HTML = re.compile(r"<[^>]+>")
_RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
_RE_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Low-level text cleaner
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Return a normalised, stop-word-free string ready for vectorisation."""
    if not isinstance(text, str):
        text = str(text)

    # 1. Remove HTML tags
    text = _RE_HTML.sub(" ", text)
    # 2. Remove URLs
    text = _RE_URL.sub(" ", text)
    # 3. Lower-case
    text = text.lower()
    # 4. Remove non-alpha chars (digits, punctuation, special symbols)
    text = _RE_NON_ALPHA.sub(" ", text)
    # 5. Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # 6. Tokenise → remove stop-words → rejoin
    tokens: List[str] = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    return " ".join(tokens)


def preprocess_batch(texts: List[str]) -> List[str]:
    """Apply `clean_text` to a list of texts."""
    return [clean_text(t) for t in texts]


# ---------------------------------------------------------------------------
# TF-IDF wrapper
# ---------------------------------------------------------------------------

class TfidfPreprocessor:
    """
    Thin wrapper around TfidfVectorizer that also handles cleaning.

    Usage
    -----
    p = TfidfPreprocessor()
    X_train = p.fit_transform(train_texts)   # during training
    X_test  = p.transform(test_texts)        # during inference
    p.save("models/tfidf.joblib")
    p2 = TfidfPreprocessor.load("models/tfidf.joblib")
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_transform(self, texts: List[str]):
        cleaned = preprocess_batch(texts)
        matrix = self.vectorizer.fit_transform(cleaned)
        self._fitted = True
        logger.info("TF-IDF fitted on %d docs | vocab size: %d",
                    len(texts), len(self.vectorizer.vocabulary_))
        return matrix

    def transform(self, texts: List[str]):
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        cleaned = preprocess_batch(texts)
        return self.vectorizer.transform(cleaned)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self.vectorizer, path)
        logger.info("TF-IDF vectorizer saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "TfidfPreprocessor":
        instance = cls.__new__(cls)
        instance.vectorizer = joblib.load(path)
        instance._fitted = True
        logger.info("TF-IDF vectorizer loaded ← %s", path)
        return instance
