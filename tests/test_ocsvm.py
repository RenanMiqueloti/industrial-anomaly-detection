"""Tests for src.models.ocsvm — OCSVMDetector."""

from __future__ import annotations

import numpy as np

from src.models.base import BaseDetector
from src.models.ocsvm import OCSVMDetector


def test_ocsvm_is_base_detector() -> None:
    assert issubclass(OCSVMDetector, BaseDetector)


def test_fit_score_anomaly_higher() -> None:
    """Anomalous samples (shifted 5σ) score higher than in-distribution samples."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 11))
    X_normal = rng.standard_normal((50, 11))
    X_anomaly = rng.standard_normal((50, 11)) + 5.0

    model = OCSVMDetector()
    model.fit(X_train)

    assert np.median(model.score(X_anomaly)) > np.median(model.score(X_normal))


def test_save_load_roundtrip(tmp_path) -> None:
    """Scores are identical after save/load cycle."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 11))

    model = OCSVMDetector()
    model.fit(X)

    path = tmp_path / "ocsvm.joblib"
    model.save(path)
    loaded = OCSVMDetector.load(path)

    np.testing.assert_allclose(model.score(X), loaded.score(X))
