# Social Media Sentiment Analyzer

Brands and researchers need to monitor public sentiment around topics, products, or events in real time. This project analyzes social media posts for sentiment, providing actionable insights and visualizations.

## Features

- **Real-time sentiment analysis** using a pre-trained Twitter-optimised HuggingFace Transformer model (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
- **Twitter API v2 integration** via Tweepy — automatically falls back to rich mock/demo data when no API credentials are provided.
- **Interactive Streamlit dashboard** with:
  - Summary metrics (total posts, positive/negative/neutral counts and percentages)
  - Sentiment distribution pie chart
  - Sentiment trends over time (line chart)
  - Confidence score histogram
  - Representative posts per sentiment category
  - Raw data table

## Tech Stack

| Layer | Technology |
|---|---|
| NLP model | HuggingFace Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest`) |
| Social media data | Tweepy (Twitter API v2) |
| Frontend / UI | Streamlit |
| Visualisations | Plotly Express |
| Data processing | Pandas, NumPy |

## Project Structure

```
.
├── app.py                  # Streamlit application (entry point)
├── sentiment_analyzer.py   # HuggingFace sentiment-analysis pipeline wrapper
├── twitter_client.py       # Twitter API client with mock-data fallback
├── data_processor.py       # DataFrame building, aggregation, trend helpers
├── requirements.txt        # Python dependencies
└── tests/
    ├── test_sentiment_analyzer.py
    ├── test_data_processor.py
    └── test_twitter_client.py
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Twitter API (optional)

Create a `.env` file in the project root (or export the variable in your shell):

```env
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
```

If the token is not provided, the app runs in **demo mode** using generated mock data so you can explore all features without API access.

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

## Usage

1. Enter a **hashtag** (without `#`) or **keyword** in the sidebar.
2. Choose the number of posts to analyse (10–100).
3. Optionally paste your Twitter Bearer Token for live data.
4. Click **Analyse** to run the pipeline and view results.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Environment Variables

| Variable | Description |
|---|---|
| `TWITTER_BEARER_TOKEN` | Twitter API v2 Bearer Token. Optional — app uses mock data without it. |

