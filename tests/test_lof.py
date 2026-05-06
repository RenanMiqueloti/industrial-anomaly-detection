"""Tests for src.models.lof — LOFDetector."""

from __future__ import annotations

import numpy as np

from src.models.base import BaseDetector
from src.models.lof import LOFDetector


def test_lof_is_base_detector() -> None:
    assert issubclass(LOFDetector, BaseDetector)


def test_fit_score_anomaly_higher() -> None:
    """Anomalous samples (shifted 5σ) score higher than in-distribution samples."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 11))
    X_normal = rng.standard_normal((50, 11))
    X_anomaly = rng.standard_normal((50, 11)) + 5.0

    model = LOFDetector()
    model.fit(X_train)

    assert np.median(model.score(X_anomaly)) > np.median(model.score(X_normal))


def test_save_load_roundtrip(tmp_path) -> None:
    """Scores are identical after save/load cycle."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 11))

    model = LOFDetector()
    model.fit(X)

    path = tmp_path / "lof.joblib"
    model.save(path)
    loaded = LOFDetector.load(path)

    np.testing.assert_allclose(model.score(X), loaded.score(X))


def test_novelty_true_allows_scoring_new_data() -> None:
    """LOFDetector must work on data not seen during fit (novelty=True contract)."""
    rng = np.random.default_rng(7)
    X_train = rng.standard_normal((100, 11))
    X_new = rng.standard_normal((20, 11)) + 10.0

    model = LOFDetector()
    model.fit(X_train)
    scores = model.score(X_new)
    assert scores.shape == (20,)
