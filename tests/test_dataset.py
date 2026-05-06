"""Tests for src.dataset — feature matrix construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io

from src.dataset import build_feature_matrix


def _save_mfpt_mat(path, signal: np.ndarray, sr: float = 97_656.0) -> None:
    dt = np.dtype([("gs", object), ("sr", object), ("load", object), ("rate", object)])
    bearing = np.zeros((1,), dtype=dt)
    bearing["gs"][0] = signal.reshape(-1, 1)
    bearing["sr"][0] = np.array([[sr]])
    bearing["load"][0] = np.array([[270.0]])
    bearing["rate"][0] = np.array([[25.0]])
    scipy.io.savemat(str(path), {"bearing": bearing})


def test_build_feature_matrix_synthetic(tmp_path) -> None:
    """Two synthetic MFPT .mat files (1 normal, 1 OR) produce valid X, y, and parquet."""
    rng = np.random.default_rng(0)

    _save_mfpt_mat(tmp_path / "baseline_1.mat", rng.standard_normal(4096), sr=97_656.0)
    _save_mfpt_mat(tmp_path / "outer_race_1.mat", rng.standard_normal(4096), sr=97_656.0)

    out = tmp_path / "features.parquet"
    X, y, meta = build_feature_matrix(tmp_path, out)

    # Shape checks
    assert X.ndim == 2
    assert X.shape[0] > 0
    assert X.shape[1] == 11  # 7 time-domain + 4 band-energy features
    assert y.shape == (X.shape[0],)
    assert len(meta) == X.shape[0]

    # Binary labels
    assert set(np.unique(y)).issubset({0, 1})

    # Both classes represented
    assert 0 in np.unique(y), "expected at least one normal window"
    assert 1 in np.unique(y), "expected at least one faulty window"

    # Parquet saved
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == X.shape[0]
    assert "_meta_class" in df.columns
    assert "_meta_y" in df.columns
