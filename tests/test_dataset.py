"""Tests for src.dataset — feature matrix construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io

from src.dataset import build_feature_matrix


def test_build_feature_matrix_synthetic(tmp_path) -> None:
    """Two synthetic .mat files (1 normal, 1 IR) produce valid X, y, and parquet."""
    rng = np.random.default_rng(0)

    # Normal: file ID 97 → class 'normal'
    scipy.io.savemat(
        str(tmp_path / "97.mat"),
        {"X097_DE_time": rng.standard_normal(4096).reshape(-1, 1)},
    )
    # IR fault: file ID 105 → class 'IR'
    scipy.io.savemat(
        str(tmp_path / "105.mat"),
        {"X105_DE_time": rng.standard_normal(4096).reshape(-1, 1)},
    )

    out = tmp_path / "features.parquet"
    X, y, meta = build_feature_matrix(tmp_path, out)

    # Shape checks
    assert X.ndim == 2
    assert X.shape[0] > 0
    assert X.shape[1] == 11  # 7 time-domain + 4 band-energy (DEFAULT_BANDS)
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
