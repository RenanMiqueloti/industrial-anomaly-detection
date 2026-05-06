"""Tests for src.compare — run_comparison with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.compare import run_comparison


def test_run_comparison_synthetic(tmp_path) -> None:
    """run_comparison returns a 4-row DataFrame with all required columns."""
    rng = np.random.default_rng(42)

    # 50 healthy + 50 anomalous test windows
    X_normal = rng.standard_normal((50, 11))
    X_anomaly = rng.standard_normal((50, 11)) + 5.0
    X_test = np.vstack([X_normal, X_anomaly])
    y_test = np.array([0] * 50 + [1] * 50)

    # Healthy training set (larger, in-distribution)
    X_train_healthy = rng.standard_normal((200, 11))

    # Save as .npy in tmp_path
    X_test_path = tmp_path / "X_test.npy"
    y_test_path = tmp_path / "y_test.npy"
    X_train_path = tmp_path / "X_train_healthy.npy"
    np.save(X_test_path, X_test)
    np.save(y_test_path, y_test)
    np.save(X_train_path, X_train_healthy)

    results = run_comparison(
        X_test_path=X_test_path,
        y_test_path=y_test_path,
        X_train_path=X_train_path,
        out_dir=tmp_path,
    )

    # Shape: exactly 4 models
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 4

    expected_cols = {
        "model",
        "roc_auc_mean",
        "roc_auc_low",
        "roc_auc_high",
        "f1_mean",
        "f1_low",
        "f1_high",
        "train_seconds",
    }
    assert expected_cols.issubset(results.columns)

    # All 4 models present
    assert set(results["model"]) == {"IsolationForest", "OC-SVM", "LOF", "AutoEncoder"}

    # Parquet and figure saved
    assert (tmp_path / "comparison.parquet").exists()
    assert (tmp_path / "figures" / "model_comparison.png").exists()

    # CI bounds are sane
    assert (results["roc_auc_low"] <= results["roc_auc_mean"]).all()
    assert (results["roc_auc_mean"] <= results["roc_auc_high"]).all()
