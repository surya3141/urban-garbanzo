"""
streamlit_app.py
----------------
Streamlit demo UI for the Fake News Detector.

Run
---
  streamlit run streamlit_app.py

The UI calls the local FastAPI service; make sure the API is running:
  uvicorn app:app --port 8000
"""

import time
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def call_api(text: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the API server.  "
            "Start it with: `uvicorn app:app --port 8000`"
        )
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        st.error(f"API error: {detail}")
    return None


def confidence_bar(confidence: float, label: str) -> None:
    colour = "#e74c3c" if label == "Fake" else "#27ae60"
    pct = int(confidence * 100)
    st.markdown(
        f"""
        <div style="background:#eee;border-radius:6px;height:24px;width:100%;">
          <div style="background:{colour};border-radius:6px;height:24px;width:{pct}%;
                      display:flex;align-items:center;padding-left:8px;">
            <span style="color:white;font-weight:bold;font-size:13px;">{pct}%</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page Layout
# ---------------------------------------------------------------------------

# --- Header ---
st.markdown(
    "<h1 style='text-align:center;'>🔍 Fake News Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:grey;'>Paste a news article or social-media post "
    "below and click <b>Analyse</b> to check its credibility.</p>",
    unsafe_allow_html=True,
)
st.divider()

# --- Input ---
text_input = st.text_area(
    label="News article / social media post",
    placeholder=(
        "e.g. "Scientists discover a cure for all cancers using common household "
        "ingredients. Doctors hate this one trick…""
    ),
    height=220,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    analyse_btn = st.button("🔎 Analyse", use_container_width=True, type="primary")

# --- Example articles ---
with st.expander("📋 Load example article"):
    examples = {
        "Fake – Sensational claim": (
            "BREAKING: Scientists have discovered that drinking bleach cures "
            "cancer instantly! Big Pharma has been hiding this secret for decades. "
            "Share this before they take it down! The government doesn't want you "
            "to know about this miracle cure that works in just 24 hours."
        ),
        "Real – Factual reporting": (
            "NASA's James Webb Space Telescope has captured the deepest infrared "
            "image of the universe to date, revealing thousands of galaxies – "
            "including the faintest objects ever observed. The image covers a patch "
            "of sky approximately the size of a grain of sand held at arm's length "
            "by someone on the ground. The telescope launched on December 25, 2021."
        ),
    }
    chosen = st.selectbox("Choose an example", list(examples.keys()))
    if st.button("Load example"):
        st.session_state["example_text"] = examples[chosen]
        st.rerun()

if "example_text" in st.session_state:
    text_input = st.session_state.pop("example_text")

# --- Prediction ---
if analyse_btn or text_input:
    if not text_input or len(text_input.strip()) < 10:
        st.warning("Please enter at least 10 characters.")
    elif analyse_btn:
        with st.spinner("Analysing…"):
            start = time.time()
            response = call_api(text_input.strip())
            elapsed = time.time() - start

        if response:
            result = response["result"]
            label: str = result["label"]
            confidence: float = result["confidence"]

            st.divider()
            st.subheader("Result")

            tag_colour = "#e74c3c" if label == "Fake" else "#27ae60"
            tag_icon = "🚫" if label == "Fake" else "✅"

            st.markdown(
                f"<div style='font-size:2rem;font-weight:bold;color:{tag_colour};'>"
                f"{tag_icon} {label.upper()}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Confidence:** {confidence:.1%}")
            confidence_bar(confidence, label)

            st.caption(f"_Analysed in {elapsed:.2f}s_")

            # Interpretation note
            st.divider()
            if label == "Fake":
                st.info(
                    "⚠️  This content shows characteristics commonly found in "
                    "misleading or fabricated news. Please verify with trusted sources "
                    "before sharing."
                )
            else:
                st.success(
                    "✔️  This content appears consistent with factual reporting. "
                    "However, always cross-check important claims."
                )

# --- Batch tab ---
st.divider()
with st.expander("📦 Batch prediction (multiple items)"):
    st.markdown("Enter one article per line:")
    batch_input = st.text_area("Batch input", height=150, label_visibility="collapsed")
    if st.button("Run batch"):
        lines = [l.strip() for l in batch_input.splitlines() if len(l.strip()) >= 10]
        if not lines:
            st.warning("Enter at least one article (min 10 chars each).")
        elif len(lines) > 50:
            st.error("Maximum 50 articles per batch.")
        else:
            try:
                resp = requests.post(
                    f"{API_URL}/predict/batch",
                    json={"texts": lines},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                import pandas as pd
                df = pd.DataFrame(data["results"])
                df.insert(0, "text_preview", [t[:80] + "…" if len(t) > 80 else t for t in lines])
                df["confidence"] = df["confidence"].map("{:.1%}".format)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(str(e))

# --- Footer ---
st.divider()
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:12px;'>"
    "Built with Python · scikit-learn · FastAPI · Streamlit</p>",
    unsafe_allow_html=True,
)
