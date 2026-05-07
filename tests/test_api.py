"""Tests for src/api.py — FastAPI endpoints with a synthetic trained model.

All tests use a monkeypatched _MODEL_PATH pointing to a tiny IForestDetector
trained on synthetic data, so no real dataset pipeline run is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Guard: skip if fastapi or httpx are not installed (api extras).
fastapi = pytest.importorskip(
    "fastapi", reason="fastapi not installed — run: pip install -e '.[api]'"
)
pytest.importorskip("httpx", reason="httpx not installed — run: pip install -e '.[dev]'")

from fastapi.testclient import TestClient  # noqa: E402

from src.models.iforest import IForestDetector  # noqa: E402


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tiny trained IForestDetector and threshold file in a temp directory."""
    d = tmp_path_factory.mktemp("model")
    rng = np.random.default_rng(42)
    X_healthy = rng.standard_normal((100, 11))
    model = IForestDetector()
    model.fit(X_healthy)
    model_path = d / "iforest_model.joblib"
    model.save(model_path)

    threshold_path = d / "threshold.json"
    # New sidecar format: explicit thresholds dict + feature_order + legacy flat keys.
    feature_order = [
        "rms",
        "peak",
        "crest_factor",
        "kurtosis",
        "skewness",
        "std",
        "p2p",
        "band_0_500",
        "band_500_2000",
        "band_2000_5000",
        "band_5000_10000",
    ]
    sidecar = {
        "thresholds": {"iforest": 0.5, "iforest_b1": 0.40, "iforest_b2": 0.55},
        "feature_order": feature_order,
        "iforest": 0.5,
        "iforest_b1": 0.40,
        "iforest_b2": 0.55,
    }
    threshold_path.write_text(json.dumps(sidecar))

    return d


@pytest.fixture()
def client(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with patched model/threshold paths and fresh app state."""
    import src.api as api_mod

    # monkeypatch updates the module-level names that lifespan reads at startup.
    # Do NOT reload — that would reset the patched values.
    monkeypatch.setattr(api_mod, "_MODEL_PATH", model_dir / "iforest_model.joblib")
    monkeypatch.setattr(api_mod, "_THRESHOLD_PATH", model_dir / "threshold.json")

    with TestClient(api_mod.app) as c:
        yield c


def _make_signal(length: int = 2048) -> list[float]:
    rng = np.random.default_rng(0)
    return rng.standard_normal(length).tolist()


def test_health_endpoint(client: TestClient) -> None:
    """GET /health returns 200 with status, model, and threshold fields."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "iforest"
    assert "threshold" in body


def test_score_endpoint_returns_valid_schema(client: TestClient) -> None:
    """POST /score with a valid signal returns a ScoreResponse-shaped payload."""
    payload = {"signal": _make_signal(2048), "fs": 12000}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "score" in body
    assert "is_anomaly" in body
    assert "threshold" in body
    assert isinstance(body["score"], float)
    assert isinstance(body["is_anomaly"], bool)


def test_score_endpoint_validation_rejects_short_signal(client: TestClient) -> None:
    """POST /score with a signal shorter than 512 samples → 422 Unprocessable Entity."""
    payload = {"signal": [0.0] * 10}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 422


def test_websocket_stream(client: TestClient) -> None:
    """WS /stream: send 3 windows, receive 3 responses each with score and is_anomaly."""
    with client.websocket_connect("/stream") as ws:
        for _ in range(3):
            ws.send_json({"signal": _make_signal(2048), "fs": 12000})
            data = ws.receive_json()
            assert "score" in data
            assert "is_anomaly" in data
            assert isinstance(data["score"], float)
            assert isinstance(data["is_anomaly"], bool)


def test_score_with_bearing_id_uses_per_bearing_threshold(client: TestClient) -> None:
    """POST /score with bearing_id=1 returns the per-bearing threshold (0.40)."""
    payload = {"signal": _make_signal(2048), "fs": 20000, "bearing_id": 1}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["bearing_id"] == 1
    assert body["threshold"] == pytest.approx(0.40)


def test_score_falls_back_to_global_for_unknown_bearing(client: TestClient) -> None:
    """POST /score with an unknown bearing_id falls back to the global threshold."""
    payload = {"signal": _make_signal(2048), "fs": 20000, "bearing_id": 99}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["bearing_id"] == 99
    assert body["threshold"] == pytest.approx(0.5)


def test_health_exposes_per_bearing_thresholds(client: TestClient) -> None:
    """GET /health surfaces per-bearing thresholds calibrated at training time."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "per_bearing_thresholds" in body
    assert body["per_bearing_thresholds"] == {"1": pytest.approx(0.40), "2": pytest.approx(0.55)}


def test_websocket_returns_validation_error_for_short_signal(client: TestClient) -> None:
    """WS /stream replies with an error envelope on validation failure, no disconnect."""
    with client.websocket_connect("/stream") as ws:
        ws.send_json({"signal": [0.0] * 10, "fs": 20000})  # too short
        data = ws.receive_json()
        assert data["error"] == "validation_error"
        # Connection still alive — send a valid window next.
        ws.send_json({"signal": _make_signal(2048), "fs": 20000})
        data = ws.receive_json()
        assert "score" in data


def test_cors_headers_present(client: TestClient) -> None:
    """OPTIONS preflight returns Access-Control-Allow-Origin header."""
    resp = client.options(
        "/score",
        headers={
            "origin": "http://localhost:8501",
            "access-control-request-method": "POST",
        },
    )
    # Starlette returns 200 for the preflight when CORS middleware is installed.
    assert resp.status_code == 200
    assert "access-control-allow-origin" in {k.lower() for k in resp.headers}
