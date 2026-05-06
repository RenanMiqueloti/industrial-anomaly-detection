"""Tests for src.evaluate — bootstrap CI and ROC plotting."""

from __future__ import annotations

import numpy as np

from src.evaluate import bootstrap_ci, plot_roc


def test_perfect_separation() -> None:
    """Perfect classifier yields ROC-AUC bootstrap mean ≈ 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    result = bootstrap_ci(y_true, scores)
    assert abs(result["roc_auc"][0] - 1.0) < 0.05
    assert result["roc_auc"][1] > 0.9  # lower CI bound also high


def test_random() -> None:
    """Random classifier CI for ROC-AUC should straddle 0.5."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    result = bootstrap_ci(y_true, scores)
    low, high = result["roc_auc"][1], result["roc_auc"][2]
    assert low <= 0.5 <= high


def test_plot_roc_creates_file(tmp_path) -> None:
    """plot_roc saves a non-empty PNG file."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    out = tmp_path / "roc.png"
    plot_roc(y_true, scores, out)
    assert out.exists()
    assert out.stat().st_size > 0
