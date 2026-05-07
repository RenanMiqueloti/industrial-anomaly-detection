"""FastAPI service for real-time anomaly scoring.

Endpoints:
    GET  /health   — liveness + provider/model info
    POST /score    — score a single window: {"signal": [...]} -> {"score", "is_anomaly"}
    WS   /stream   — stream of windows: client sends {"signal": [...]} per message

Run:
    uvicorn src.api:app --reload --port 8000

Install:
    pip install -e ".[api]"
"""

from __future__ import annotations

import json
import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.features import extract_all
from src.models.iforest import IForestDetector

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("results/iforest_model.joblib")
_THRESHOLD_PATH = Path("results/threshold.json")


class ScoreRequest(BaseModel):
    signal: list[float] = Field(..., min_length=512, description="1-D vibration window")
    fs: int = Field(20_000, gt=0, description="Sampling rate in Hz")


class ScoreResponse(BaseModel):
    score: float
    is_anomaly: bool
    threshold: float
    model: str = "IsolationForest"


def _load_threshold() -> float:
    if _THRESHOLD_PATH.exists():
        try:
            data = json.loads(_THRESHOLD_PATH.read_text())
            return float(data["iforest"])
        except (KeyError, ValueError, json.JSONDecodeError):
            warnings.warn(
                f"Could not read threshold from {_THRESHOLD_PATH}; defaulting to inf (nothing flagged)",
                stacklevel=2,
            )
    # inf means no window is flagged until a real threshold is configured via threshold.json
    return float("inf")


def _score_signal(model: IForestDetector, signal: list[float], fs: int) -> float:
    arr = np.array(signal, dtype=np.float64)
    feats = extract_all(arr, fs=fs)
    X = np.array(list(feats.values()), dtype=np.float64).reshape(1, -1)
    return float(model.score(X)[0])


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    if not _MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {_MODEL_PATH}. Run 'make train' first.")
    app.state.model = IForestDetector.load(_MODEL_PATH)
    app.state.threshold = _load_threshold()
    logger.info("IForestDetector loaded. threshold=%.4f", app.state.threshold)
    yield


app = FastAPI(
    title="Industrial Anomaly Detection API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    threshold = getattr(app.state, "threshold", float("inf"))
    return {"status": "ok", "model": "iforest", "threshold": threshold}


@app.post("/score", response_model=ScoreResponse)
def score(body: ScoreRequest) -> ScoreResponse:
    model: IForestDetector = app.state.model
    threshold: float = app.state.threshold
    s = _score_signal(model, body.signal, body.fs)
    return ScoreResponse(score=s, is_anomaly=s >= threshold, threshold=threshold)


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    model: IForestDetector = app.state.model
    threshold: float = app.state.threshold
    try:
        while True:
            msg = await ws.receive_json()
            signal: list[float] = msg["signal"]
            fs: int = int(msg.get("fs", 20_000))
            s = _score_signal(model, signal, fs)
            await ws.send_json({"score": s, "is_anomaly": s >= threshold})
    except WebSocketDisconnect:
        pass
