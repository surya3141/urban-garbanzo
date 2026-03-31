"""
train.py
--------
Train a Fake-News classifier using TF-IDF features + Logistic Regression or
Random Forest.  An optional BERT fine-tuning path is included but kept
behind a flag so the project runs on CPU with no GPU requirement.

Usage (CLI)
-----------
  python -m src.train --data_dir data/ --model lr   # Logistic Regression
  python -m src.train --data_dir data/ --model rf   # Random Forest
  python -m src.train --data_dir data/ --model bert # Fine-tuned BERT (GPU recommended)

Expected CSV layout (from Kaggle "Fake-News" dataset):
  - fake.csv : columns [title, text, subject, date]  → label = 1
  - true.csv : columns [title, text, subject, date]  → label = 0
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Ensure local `src` package is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessor import TfidfPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LABEL_MAP = {0: "Real", 1: "Fake"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kaggle_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load the standard Kaggle Fake-News dataset (fake.csv + true.csv).
    Falls back to a single combined CSV named `news.csv` with a `label`
    column (0 = Real, 1 = Fake).
    """
    data_dir = Path(data_dir)
    fake_path = data_dir / "fake.csv"
    true_path = data_dir / "true.csv"
    combined_path = data_dir / "news.csv"

    if fake_path.exists() and true_path.exists():
        logger.info("Loading fake.csv + true.csv …")
        fake_df = pd.read_csv(fake_path)
        fake_df["label"] = 1

        true_df = pd.read_csv(true_path)
        true_df["label"] = 0

        df = pd.concat([fake_df, true_df], ignore_index=True)
    elif combined_path.exists():
        logger.info("Loading combined news.csv …")
        df = pd.read_csv(combined_path)
        if "label" not in df.columns:
            raise ValueError("news.csv must contain a 'label' column (0=Real, 1=Fake).")
    else:
        raise FileNotFoundError(
            "Dataset not found.  Run `python data/download_data.py` first, "
            "or place fake.csv + true.csv inside the data/ directory."
        )

    # Combine title + text into one field
    if "text" in df.columns and "title" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    else:
        raise ValueError("Dataset must have a 'text' column.")

    df = df[["content", "label"]].dropna()
    logger.info("Dataset loaded: %d rows | Fake: %d | Real: %d",
                len(df), df["label"].sum(), (df["label"] == 0).sum())
    return df


# ---------------------------------------------------------------------------
# Scikit-learn trainers
# ---------------------------------------------------------------------------

def train_sklearn(
    df: pd.DataFrame,
    model_type: str = "lr",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Train and evaluate a scikit-learn classifier (lr or rf)."""

    X = df["content"].tolist()
    y = df["label"].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- TF-IDF ---
    logger.info("Fitting TF-IDF vectorizer …")
    preprocessor = TfidfPreprocessor(max_features=50_000, ngram_range=(1, 2))
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # --- Classifier ---
    if model_type == "lr":
        logger.info("Training Logistic Regression …")
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state,
        )
    elif model_type == "rf":
        logger.info("Training Random Forest (100 trees) …")
        clf = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lr' or 'rf'.")

    clf.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test Accuracy: %.4f", acc)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    # --- Persist artefacts ---
    tfidf_path = MODEL_DIR / "tfidf.joblib"
    clf_path = MODEL_DIR / f"classifier_{model_type}.joblib"
    preprocessor.save(str(tfidf_path))
    joblib.dump(clf, clf_path)
    logger.info("Classifier saved → %s", clf_path)

    # Also save metadata
    meta = {"model_type": model_type, "accuracy": acc, "label_map": LABEL_MAP}
    joblib.dump(meta, MODEL_DIR / "metadata.joblib")
    logger.info("Training complete. Artefacts saved in %s/", MODEL_DIR)


# ---------------------------------------------------------------------------
# Optional BERT fine-tuning
# ---------------------------------------------------------------------------

def train_bert(df: pd.DataFrame, epochs: int = 3, batch_size: int = 16) -> None:
    """Fine-tune DistilBERT for sequence classification."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
        from torch.optim import AdamW
    except ImportError:
        logger.error("Install PyTorch + transformers: pip install torch transformers")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("BERT training on device: %s", device)

    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class NewsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=256):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    X = df["content"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        logger.info("Epoch %d/%d | Avg Loss: %.4f", epoch + 1, epochs, total_loss / len(train_loader))

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    logger.info("BERT Test Accuracy: %.4f", acc)
    logger.info("\n%s", classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

    bert_path = MODEL_DIR / "bert_model"
    model.save_pretrained(str(bert_path))
    tokenizer.save_pretrained(str(bert_path))
    logger.info("BERT model saved → %s/", bert_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fake News Classifier")
    parser.add_argument("--data_dir", default="data", help="Directory with CSV data files")
    parser.add_argument(
        "--model",
        choices=["lr", "rf", "bert"],
        default="lr",
        help="Model type: lr=Logistic Regression, rf=Random Forest, bert=DistilBERT",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--bert_epochs", type=int, default=3)
    args = parser.parse_args()

    df = load_kaggle_dataset(args.data_dir)

    if args.model in ("lr", "rf"):
        train_sklearn(df, model_type=args.model, test_size=args.test_size)
    else:
        train_bert(df, epochs=args.bert_epochs)


if __name__ == "__main__":
    main()
