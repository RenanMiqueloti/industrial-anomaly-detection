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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from src.features import extract_all
from src.models.iforest import IForestDetector

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("results/iforest_model.joblib")
_THRESHOLD_PATH = Path("results/threshold.json")


class ScoreRequest(BaseModel):
    signal: list[float] = Field(..., min_length=512, description="1-D vibration window")
    fs: int = Field(20_000, gt=0, description="Sampling rate in Hz")
    bearing_id: int | None = Field(
        None,
        ge=1,
        description="Optional 1-indexed bearing id. When provided, uses the per-bearing "
        "p99 threshold calibrated at training time; otherwise falls back to the global threshold.",
    )


class ScoreResponse(BaseModel):
    score: float
    is_anomaly: bool
    threshold: float
    bearing_id: int | None = None
    model: str = "IsolationForest"


def _load_sidecar() -> dict[str, Any]:
    """Load thresholds + feature_order from threshold.json.

    Returns
    -------
    {
      "thresholds": {"iforest": float, "iforest_b<id>": float, ...},
      "feature_order": list[str] | None,
    }

    When the file is missing or malformed, returns inf threshold so nothing is
    flagged until a real configuration is in place.
    """
    if not _THRESHOLD_PATH.exists():
        return {"thresholds": {"iforest": float("inf")}, "feature_order": None}
    try:
        data = json.loads(_THRESHOLD_PATH.read_text())
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"Could not parse {_THRESHOLD_PATH}: {exc}. Defaulting to inf threshold.",
            stacklevel=2,
        )
        return {"thresholds": {"iforest": float("inf")}, "feature_order": None}

    # New sidecar format (post-PR-correctness fix) has explicit "thresholds" key.
    if "thresholds" in data and isinstance(data["thresholds"], dict):
        thresholds = {k: float(v) for k, v in data["thresholds"].items()}
    else:
        # Legacy flat format: {"iforest": 0.5, "iforest_b1": 0.52, ...}
        thresholds = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        if not thresholds:
            thresholds = {"iforest": float("inf")}

    feature_order = data.get("feature_order")
    return {"thresholds": thresholds, "feature_order": feature_order}


def _resolve_threshold(thresholds: dict[str, float], bearing_id: int | None) -> float:
    """Resolve which threshold to use for a request.

    Priority: per-bearing → global "iforest" → inf.
    """
    if bearing_id is not None:
        bearing_key = f"iforest_b{bearing_id}"
        if bearing_key in thresholds:
            return thresholds[bearing_key]
        logger.info("No per-bearing threshold for bearing %d; falling back to global.", bearing_id)
    return thresholds.get("iforest", float("inf"))


def _score_signal(
    model: IForestDetector,
    signal: list[float],
    fs: int,
    feature_order: list[str] | None,
) -> float:
    arr = np.array(signal, dtype=np.float64)
    feats = extract_all(arr, fs=fs)
    if feature_order is not None:
        # Explicit ordering by name: defends against feature dict reordering or
        # the model being trained with a different column order.
        missing = [k for k in feature_order if k not in feats]
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Feature mismatch: model expects {missing} but extract_all did not produce them.",
            )
        values = [feats[k] for k in feature_order]
    else:
        values = list(feats.values())
    X = np.array(values, dtype=np.float64).reshape(1, -1)
    return float(model.score(X)[0])


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    if not _MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {_MODEL_PATH}. Run 'make train' first.")
    app.state.model = IForestDetector.load(_MODEL_PATH)
    sidecar = _load_sidecar()
    app.state.thresholds = sidecar["thresholds"]
    app.state.feature_order = sidecar["feature_order"]
    logger.info(
        "IForestDetector loaded. global_threshold=%.4f bearings=%s feature_order=%s",
        app.state.thresholds.get("iforest", float("inf")),
        sorted(k for k in app.state.thresholds if k.startswith("iforest_b")),
        "set" if app.state.feature_order else "unset (legacy)",
    )
    yield


app = FastAPI(
    title="Industrial Anomaly Detection API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: allow any origin for the demo. Tighten before any real deployment that
# carries authenticated traffic.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    thresholds = getattr(app.state, "thresholds", {"iforest": float("inf")})
    return {
        "status": "ok",
        "model": "iforest",
        "threshold": thresholds.get("iforest", float("inf")),
        "per_bearing_thresholds": {
            k.removeprefix("iforest_b"): v
            for k, v in thresholds.items()
            if k.startswith("iforest_b")
        },
    }


@app.post("/score", response_model=ScoreResponse)
def score(body: ScoreRequest) -> ScoreResponse:
    model: IForestDetector = app.state.model
    threshold = _resolve_threshold(app.state.thresholds, body.bearing_id)
    s = _score_signal(model, body.signal, body.fs, app.state.feature_order)
    return ScoreResponse(
        score=s,
        is_anomaly=s >= threshold,
        threshold=threshold,
        bearing_id=body.bearing_id,
    )


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """Score a continuous stream of windows.

    Each client message must match :class:`ScoreRequest`. Malformed messages
    are reported back as ``{"error": "..."}`` instead of dropping the
    connection silently.
    """
    await ws.accept()
    model: IForestDetector = app.state.model
    try:
        while True:
            try:
                msg = await ws.receive_json()
                request = ScoreRequest.model_validate(msg)
            except ValidationError as exc:
                await ws.send_json({"error": "validation_error", "detail": exc.errors()})
                continue
            except (ValueError, KeyError) as exc:
                await ws.send_json({"error": "bad_message", "detail": str(exc)})
                continue

            threshold = _resolve_threshold(app.state.thresholds, request.bearing_id)
            try:
                s = _score_signal(model, request.signal, request.fs, app.state.feature_order)
            except HTTPException as exc:
                await ws.send_json({"error": "scoring_failed", "detail": exc.detail})
                continue

            await ws.send_json(
                {
                    "score": s,
                    "is_anomaly": s >= threshold,
                    "threshold": threshold,
                    "bearing_id": request.bearing_id,
                }
            )
    except WebSocketDisconnect:
        pass
