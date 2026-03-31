"""
app.py
------
FastAPI REST API for the Fake News Detector.

Endpoints
---------
  POST /predict        – classify a single article / social media post
  POST /predict/batch  – classify up to 50 texts at once
  GET  /health         – liveness probe
  GET  /               – API documentation redirect

Run
---
  uvicorn app:app --reload --port 8000

Then open: http://127.0.0.1:8000/docs
"""

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

from src.predict import FakeNewsPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fake News Detector API",
    description=(
        "Classify news articles and social-media posts as **Fake** or **Real** "
        "using NLP + supervised learning."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
_predictor: FakeNewsPredictor | None = None


@app.on_event("startup")
async def load_model() -> None:
    global _predictor
    try:
        _predictor = FakeNewsPredictor()
        logger.info("Model loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning("Model not found at startup: %s", exc)
        logger.warning("Train a model first: python -m src.train")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ArticleRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=10_000,
        example=(
            "Scientists discover water on Mars, raising hopes for extraterrestrial life. "
            "NASA confirms the findings after decades of research."
        ),
    )

    @field_validator("text")
    @classmethod
    def no_empty_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        for i, t in enumerate(v):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"texts[{i}] must be a non-empty string")
            if len(t) > 10_000:
                raise ValueError(f"texts[{i}] exceeds 10,000 character limit")
        return v


class PredictionResult(BaseModel):
    label: str = Field(..., example="Fake")
    label_id: int = Field(..., example=1)
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.93)


class PredictionResponse(BaseModel):
    result: PredictionResult
    text_preview: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health_check():
    """Liveness probe – returns whether the model is ready."""
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: ArticleRequest):
    """
    Classify a **single** news article or social-media post.

    - **label**: `"Fake"` or `"Real"`
    - **confidence**: probability score in `[0, 1]`
    """
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first: python -m src.train",
        )

    result = _predictor.predict(request.text)
    preview = (request.text[:120] + "…") if len(request.text) > 120 else request.text

    return PredictionResponse(
        result=PredictionResult(**result),
        text_preview=preview,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Classify up to **50 texts** in a single request.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first: python -m src.train",
        )

    results = _predictor.predict_batch(request.texts)
    return BatchPredictionResponse(
        results=[PredictionResult(**r) for r in results],
        count=len(results),
    )
