---
# Fill in the fields below to create a basic custom agent for your repository.
# The Copilot CLI can be used for local testing: https://gh.io/customagents/cli
# To make this agent available, merge this file into the default repository branch.
# For format details, see: https://gh.io/customagents/config

name:
description:
---

# My Agent

Describe what your agent does here.
**# Fake News Detector

## Prompt for GitHub Copilot
Build a Python-based AI tool that detects fake or misleading news articles shared on social media.

### Requirements:
- Input: Text of a news article or social media post.
- Output: Classification as "Fake" or "Real" with confidence score.
- Dataset: Use Kaggle’s Fake News dataset or any open-source labeled dataset.
- AI Technique: NLP + supervised learning (Logistic Regression, Random Forest, or fine-tuned BERT).
- Workflow:
  1. Preprocess text (tokenization, stopword removal, TF-IDF).
  2. Train classifier on labeled dataset.
  3. Expose prediction via a simple Flask/FastAPI endpoint.
  4. Optional: Add a Streamlit UI for demo.
- Tech Stack: Python, scikit-learn, TensorFlow/PyTorch, Flask/FastAPI, Streamlit.
- Copilot Usage: Generate preprocessing functions, model training scripts, and API boilerplate quickly.**
