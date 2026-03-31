"""
predict.py
----------
Inference wrapper for the trained Fake-News classifier.

Supports both scikit-learn (TF-IDF + LR/RF) and BERT models.
The predictor auto-detects which artefacts are present in `models/`.

Usage (Python)
--------------
from src.predict import FakeNewsPredictor

predictor = FakeNewsPredictor()           # loads from models/
result = predictor.predict("Article text here …")
# → {"label": "Fake", "confidence": 0.93, "label_id": 1}
"""

import logging
from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np

from src.preprocessor import TfidfPreprocessor

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
LABEL_MAP = {0: "Real", 1: "Fake"}


class FakeNewsPredictor:
    """
    Unified prediction interface.

    Priority:
      1. scikit-learn classifier  (fast, no GPU needed)
      2. BERT classifier          (higher accuracy, GPU recommended)
    """

    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self._clf = None
        self._tfidf: Union[TfidfPreprocessor, None] = None
        self._bert_model = None
        self._bert_tokenizer = None
        self._mode: str = ""
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        tfidf_path = self.model_dir / "tfidf.joblib"
        bert_path = self.model_dir / "bert_model"

        # Prefer sklearn if available (lighter)
        if tfidf_path.exists():
            self._load_sklearn()
        elif bert_path.exists():
            self._load_bert()
        else:
            raise FileNotFoundError(
                f"No trained model found in '{self.model_dir}/'. "
                "Run `python -m src.train` first."
            )

    def _load_sklearn(self) -> None:
        tfidf_path = self.model_dir / "tfidf.joblib"
        self._tfidf = TfidfPreprocessor.load(str(tfidf_path))

        # Find whichever classifier file exists (prefer LR)
        for name in ("classifier_lr.joblib", "classifier_rf.joblib"):
            clf_path = self.model_dir / name
            if clf_path.exists():
                self._clf = joblib.load(clf_path)
                self._mode = "sklearn"
                logger.info("Loaded sklearn classifier ← %s", clf_path)
                return

        raise FileNotFoundError(
            f"TF-IDF found but no classifier_*.joblib in '{self.model_dir}/'. "
            "Retrain with `python -m src.train`."
        )

    def _load_bert(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers + torch: pip install torch transformers")

        bert_dir = str(self.model_dir / "bert_model")
        self._bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self._bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
        self._bert_model.eval()
        self._mode = "bert"
        logger.info("Loaded BERT model ← %s", bert_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict:
        """
        Classify a single text.

        Returns
        -------
        dict with keys:
          label      : "Fake" or "Real"
          label_id   : 1 (Fake) or 0 (Real)
          confidence : float in [0, 1]
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Classify a list of texts. Returns a list of result dicts."""
        if self._mode == "sklearn":
            return self._predict_sklearn(texts)
        elif self._mode == "bert":
            return self._predict_bert(texts)
        else:
            raise RuntimeError("Predictor not initialised. Call _load() first.")

    # ------------------------------------------------------------------
    # Internal inference helpers
    # ------------------------------------------------------------------

    def _predict_sklearn(self, texts: List[str]) -> List[Dict]:
        X = self._tfidf.transform(texts)
        label_ids = self._clf.predict(X)
        probas: np.ndarray = self._clf.predict_proba(X)

        results = []
        for lid, prob_row in zip(label_ids, probas):
            results.append(
                {
                    "label": LABEL_MAP[int(lid)],
                    "label_id": int(lid),
                    "confidence": round(float(prob_row[int(lid)]), 4),
                }
            )
        return results

    def _predict_bert(self, texts: List[str]) -> List[Dict]:
        import torch
        import torch.nn.functional as F

        encodings = self._bert_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._bert_model(**encodings)
            probs = F.softmax(outputs.logits, dim=-1).numpy()

        results = []
        for prob_row in probs:
            lid = int(np.argmax(prob_row))
            results.append(
                {
                    "label": LABEL_MAP[lid],
                    "label_id": lid,
                    "confidence": round(float(prob_row[lid]), 4),
                }
            )
        return results
