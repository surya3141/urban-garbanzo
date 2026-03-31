"""
Social Media Sentiment Analyzer — Streamlit application.

Run with:
    streamlit run app.py

Environment variables (optional — app falls back to demo/mock data):
    TWITTER_BEARER_TOKEN  — Twitter API v2 bearer token
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from data_processor import build_dataframe, compute_summary, sentiment_over_time, top_posts
from sentiment_analyzer import SentimentAnalyzer
from twitter_client import TwitterClient

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")

query: str = st.sidebar.text_input(
    "Hashtag or keyword",
    value="python",
    help="Enter a hashtag (without #) or any keyword to analyse.",
)

max_posts: int = st.sidebar.slider(
    "Number of posts to analyse",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
)

bearer_token: Optional[str] = st.sidebar.text_input(
    "Twitter Bearer Token (optional)",
    value=os.getenv("TWITTER_BEARER_TOKEN", ""),
    type="password",
    help="Leave blank to use demo/mock data.",
)

run_button = st.sidebar.button("🔍 Analyse", use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📊 Social Media Sentiment Analyzer")
st.markdown(
    "Analyse the public sentiment around any hashtag or keyword using "
    "state-of-the-art NLP. Powered by **HuggingFace Transformers** and "
    "optionally connected to the **Twitter API**."
)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
}


@st.cache_resource(show_spinner="Loading sentiment model…")
def load_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


def run_analysis(query: str, max_posts: int, bearer_token: str) -> None:
    """Fetch posts, run sentiment analysis, and render the dashboard."""

    # 1. Fetch posts -----------------------------------------------------------
    use_mock = not bearer_token.strip()
    client = TwitterClient(
        bearer_token=bearer_token.strip() or None,
        use_mock=use_mock,
    )

    with st.spinner(f"Fetching posts for **{query}**…"):
        posts = client.fetch_recent_posts(query=query, max_results=max_posts)

    if not posts:
        st.warning("No posts found for that query. Try a different keyword.")
        return

    if client.is_mock:
        st.info(
            "🎭 **Demo mode** — showing mock data. "
            "Add a Twitter Bearer Token in the sidebar to use live data.",
            icon="ℹ️",
        )

    # 2. Sentiment analysis ----------------------------------------------------
    analyzer = load_analyzer()
    texts = [p.text for p in posts]

    with st.spinner("Analysing sentiment…"):
        results = analyzer.analyze_batch(texts)

    # 3. Build DataFrame -------------------------------------------------------
    df = build_dataframe(posts, results)
    summary = compute_summary(df)

    # 4. Summary metrics -------------------------------------------------------
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total posts", summary["total"])
    col2.metric(
        "😊 Positive",
        f"{summary['positive']} ({summary['positive_pct']}%)",
    )
    col3.metric(
        "😠 Negative",
        f"{summary['negative']} ({summary['negative_pct']}%)",
    )
    col4.metric(
        "😐 Neutral",
        f"{summary['neutral']} ({summary['neutral_pct']}%)",
    )

    # 5. Pie chart — overall distribution ------------------------------------
    st.subheader("🥧 Sentiment Distribution")
    pie_df = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [summary["positive"], summary["negative"], summary["neutral"]],
        }
    )
    fig_pie = px.pie(
        pie_df,
        values="Count",
        names="Sentiment",
        color="Sentiment",
        color_discrete_map={
            "Positive": SENTIMENT_COLORS["positive"],
            "Negative": SENTIMENT_COLORS["negative"],
            "Neutral": SENTIMENT_COLORS["neutral"],
        },
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

    # 6. Sentiment over time --------------------------------------------------
    st.subheader("📉 Sentiment Trends Over Time")
    time_df = sentiment_over_time(df, freq="10min")
    if not time_df.empty and len(time_df) > 1:
        fig_line = px.line(
            time_df,
            x="created_at",
            y=["positive", "negative", "neutral"],
            labels={"value": "Post count", "created_at": "Time", "variable": "Sentiment"},
            color_discrete_map={
                "positive": SENTIMENT_COLORS["positive"],
                "negative": SENTIMENT_COLORS["negative"],
                "neutral": SENTIMENT_COLORS["neutral"],
            },
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Not enough time variation in the fetched posts to render a trend chart.")

    # 7. Bar chart — confidence distribution ----------------------------------
    st.subheader("📊 Confidence Score Distribution")
    fig_hist = px.histogram(
        df,
        x="score",
        color="label",
        barmode="overlay",
        nbins=20,
        labels={"score": "Confidence", "label": "Sentiment"},
        color_discrete_map=SENTIMENT_COLORS,
        opacity=0.75,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 8. Top posts by label ---------------------------------------------------
    st.subheader("🏆 Representative Posts")
    tab_pos, tab_neg, tab_neu = st.tabs(["😊 Positive", "😠 Negative", "😐 Neutral"])
    for tab, label in [(tab_pos, "positive"), (tab_neg, "negative"), (tab_neu, "neutral")]:
        with tab:
            sample = top_posts(df, label, n=5)
            if sample.empty:
                st.write("No posts with this sentiment found.")
            else:
                for _, row in sample.iterrows():
                    st.markdown(
                        f"> {row['text']}  \n"
                        f"*Confidence: {row['score']:.2%}*"
                    )
                    st.divider()

    # 9. Raw data table -------------------------------------------------------
    with st.expander("📋 Raw data"):
        st.dataframe(
            df[["created_at", "text", "label", "score", "like_count", "retweet_count"]],
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if run_button and query.strip():
    run_analysis(query.strip(), max_posts, bearer_token or "")
elif not run_button:
    st.markdown(
        "👈 **Enter a hashtag or keyword in the sidebar and click Analyse** "
        "to get started."
    )
