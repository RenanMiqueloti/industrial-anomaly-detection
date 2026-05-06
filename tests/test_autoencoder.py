"""Tests for src.models.autoencoder — AutoEncoderDetector.

Uses max_epochs=20 for speed (CI < 30 s per test).
"""

from __future__ import annotations

import numpy as np

from src.models.autoencoder import AutoEncoderDetector
from src.models.base import BaseDetector


def test_autoencoder_is_base_detector() -> None:
    assert issubclass(AutoEncoderDetector, BaseDetector)


def test_fit_score_anomaly_higher() -> None:
    """Reconstruction error is higher for out-of-distribution samples (smoke test)."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 11)).astype(np.float32)
    X_normal = rng.standard_normal((50, 11)).astype(np.float32)
    X_anomaly = (rng.standard_normal((50, 11)) + 5.0).astype(np.float32)

    model = AutoEncoderDetector(max_epochs=20, patience=5, random_state=42)
    model.fit(X_train)

    assert np.median(model.score(X_anomaly)) > np.median(model.score(X_normal))


def test_save_load_roundtrip(tmp_path) -> None:
    """Scores are identical after save/load cycle."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 11)).astype(np.float32)

    model = AutoEncoderDetector(max_epochs=20, patience=5, random_state=42)
    model.fit(X)

    path = tmp_path / "ae.joblib"
    model.save(path)
    loaded = AutoEncoderDetector.load(path)

    np.testing.assert_allclose(model.score(X), loaded.score(X), rtol=1e-5)


def test_reproducible(tmp_path) -> None:
    """Two successive fits with the same seed produce the same scores on held-out data."""
    rng = np.random.default_rng(99)
    X_train = rng.standard_normal((150, 11)).astype(np.float32)
    X_test = rng.standard_normal((30, 11)).astype(np.float32)

    m1 = AutoEncoderDetector(max_epochs=20, patience=5, random_state=42)
    m1.fit(X_train)

    m2 = AutoEncoderDetector(max_epochs=20, patience=5, random_state=42)
    m2.fit(X_train)

    np.testing.assert_allclose(m1.score(X_test), m2.score(X_test), rtol=1e-5)
