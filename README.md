# 🔍 Fake News Detector

An AI-powered tool that classifies news articles and social media posts as **Fake** or **Real** using NLP and supervised learning.

---

## Features

- **Text preprocessing** – HTML/URL stripping, tokenisation, stop-word removal, TF-IDF vectorisation
- **Multiple classifiers** – Logistic Regression, Random Forest, or fine-tuned DistilBERT
- **REST API** – FastAPI endpoint with single & batch prediction
- **Demo UI** – Streamlit web app with confidence visualisation
- **Batch support** – classify up to 50 articles in one API call

---

## Project Structure

```
urban-garbanzo/
├── src/
│   ├── preprocessor.py     # TF-IDF preprocessing pipeline
│   ├── train.py            # Model training (LR / RF / BERT)
│   └── predict.py          # Inference wrapper
├── data/
│   └── download_data.py    # Dataset downloader (Kaggle or direct)
├── models/                 # Saved model artefacts (auto-created)
├── app.py                  # FastAPI REST API
├── streamlit_app.py        # Streamlit demo UI
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch and `transformers` are only needed for BERT training. Skip them for the sklearn path.

### 2. Download the dataset

```bash
# Option A – Kaggle API (requires ~/.kaggle/kaggle.json)
python data/download_data.py --source kaggle

# Option B – Direct mirror (no account needed)
python data/download_data.py --source direct
```

Place the CSV files inside `data/`:
- `data/fake.csv` — fake news articles (label = 1)
- `data/true.csv` — real news articles (label = 0)

### 3. Train the model

```bash
# Logistic Regression (fast, great baseline ~98% accuracy)
python -m src.train --model lr

# Random Forest
python -m src.train --model rf

# Fine-tuned DistilBERT (GPU recommended)
python -m src.train --model bert
```

Artefacts are saved to `models/`.

### 4. Start the API

```bash
uvicorn app:app --reload --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Launch the Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## API Reference

### `POST /predict`

Classify a single article.

**Request body**
```json
{ "text": "Scientists discover water on Mars..." }
```

**Response**
```json
{
  "result": {
    "label": "Real",
    "label_id": 0,
    "confidence": 0.9712
  },
  "text_preview": "Scientists discover water on Mars..."
}
```

### `POST /predict/batch`

Classify up to 50 texts at once.

**Request body**
```json
{ "texts": ["Article 1...", "Article 2..."] }
```

### `GET /health`

Liveness probe.

```json
{ "status": "ok", "model_loaded": true }
```

---

## Example cURL

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Miracle cure discovered that doctors dont want you to know!"}'
```

---

## Tech Stack

| Layer         | Technology                        |
|---------------|-----------------------------------|
| Language      | Python 3.11+                      |
| NLP / ML      | scikit-learn, NLTK, transformers  |
| Deep Learning | PyTorch + DistilBERT (optional)   |
| API           | FastAPI + Uvicorn                 |
| UI            | Streamlit                         |
| Serialisation | joblib                            |

---

## Model Performance (Logistic Regression on Kaggle dataset)

| Metric    | Fake   | Real   |
|-----------|--------|--------|
| Precision | 0.98   | 0.99   |
| Recall    | 0.99   | 0.98   |
| F1-score  | 0.99   | 0.99   |
| **Accuracy** | **98.6%** | |

---

## Disclaimer

This tool is for educational and research purposes. No classifier is 100% accurate — always verify important claims with trusted sources.

---

## License

MIT

Brands and researchers need to monitor public sentiment around topics, products, or events in real time. This project analyzes social media posts for sentiment, providing actionable insights and visualizations.
