"""Tests for src.precompute — full-dataset score caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.compare import run_comparison
from src.precompute import precompute_scores


def _build_synthetic_features(tmp_path: Path) -> Path:
    """Return a features parquet that looks like data/features/features.parquet."""
    rng = np.random.default_rng(0)
    n = 80
    X = rng.standard_normal((n, 11))
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(11)])
    df["_meta_y"] = np.array([0] * 60 + [1] * 20)
    df["_meta_bearing_id"] = np.array([1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path, index=False)
    return features_path


def _train_three_models(tmp_path: Path) -> None:
    """Use run_comparison to materialise the three model .joblib files."""
    rng = np.random.default_rng(1)
    X_test = np.vstack([rng.standard_normal((40, 11)), rng.standard_normal((40, 11)) + 4.0])
    y_test = np.array([0] * 40 + [1] * 40)
    X_train_healthy = rng.standard_normal((120, 11))

    np.save(tmp_path / "X_test.npy", X_test)
    np.save(tmp_path / "y_test.npy", y_test)
    np.save(tmp_path / "X_train_healthy.npy", X_train_healthy)

    run_comparison(
        X_test_path=tmp_path / "X_test.npy",
        y_test_path=tmp_path / "y_test.npy",
        X_train_path=tmp_path / "X_train_healthy.npy",
        out_dir=tmp_path,
    )


def test_precompute_writes_one_column_per_trained_model(tmp_path: Path) -> None:
    features_path = _build_synthetic_features(tmp_path)
    _train_three_models(tmp_path)

    out_path = precompute_scores(features_path=features_path, results_dir=tmp_path)

    assert out_path == tmp_path / "full_dataset_scores.parquet"
    df = pd.read_parquet(out_path)
    # One row per feature parquet row.
    assert len(df) == 80
    # Score column per trained model + metadata.
    assert set(df.columns) >= {"iforest", "ocsvm", "ae", "_meta_y", "_meta_bearing_id"}
    # Scores are finite floats.
    for col in ("iforest", "ocsvm", "ae"):
        assert df[col].notna().all()
        assert np.isfinite(df[col]).all()


def test_precompute_skips_missing_models(tmp_path: Path) -> None:
    """Only writes columns for models whose joblib file exists."""
    features_path = _build_synthetic_features(tmp_path)
    _train_three_models(tmp_path)

    # Simulate a deploy where only IsolationForest is shipped.
    (tmp_path / "ocsvm_model.joblib").unlink()
    (tmp_path / "ae_model.joblib").unlink()

    out_path = precompute_scores(features_path=features_path, results_dir=tmp_path)
    df = pd.read_parquet(out_path)

    assert "iforest" in df.columns
    assert "ocsvm" not in df.columns
    assert "ae" not in df.columns


def test_precompute_raises_when_no_models_present(tmp_path: Path) -> None:
    features_path = _build_synthetic_features(tmp_path)
    # No models in tmp_path.
    with pytest.raises(FileNotFoundError, match="No trained models"):
        precompute_scores(features_path=features_path, results_dir=tmp_path)


def test_precompute_raises_when_features_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Feature parquet"):
        precompute_scores(features_path=tmp_path / "missing.parquet", results_dir=tmp_path)
