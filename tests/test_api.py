"""Tests for src/api.py — FastAPI endpoints with a synthetic trained model.

All tests use a monkeypatched _MODEL_PATH pointing to a tiny IForestDetector
trained on synthetic data, so no CWRU pipeline run is required.
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
    threshold_path.write_text(json.dumps({"iforest": 0.5}))

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
