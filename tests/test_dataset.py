"""Tests for src.dataset — IMS feature matrix construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import build_ims_features
from src.ingest import IMS_ROWS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_snapshot(path: Path, n_bearings: int = 4, seed: int = 0) -> None:
    """Write a synthetic IMS snapshot file (tab-separated)."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((IMS_ROWS, n_bearings))
    np.savetxt(str(path), data, delimiter="\t")


def _make_run_dir(tmp_path: Path, n_snapshots: int = 5, n_bearings: int = 4) -> Path:
    """Create a synthetic IMS run directory with n_snapshots files."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    base_minute = 32
    for i in range(n_snapshots):
        minutes = base_minute + i * 10
        hour = 10 + minutes // 60
        minute = minutes % 60
        ts = f"2004.02.12.{hour:02d}.{minute:02d}.39"
        _write_snapshot(run_dir / ts, n_bearings=n_bearings, seed=i)
    return run_dir


# ---------------------------------------------------------------------------
# build_ims_features
# ---------------------------------------------------------------------------


def test_build_ims_features_basic_shape(tmp_path: Path) -> None:
    """5 snapshots x 4 bearings = 20 rows, 11 features."""
    run_dir = _make_run_dir(tmp_path, n_snapshots=5, n_bearings=4)
    out = tmp_path / "features.parquet"

    X, y, meta = build_ims_features(run_dir, out)

    assert X.ndim == 2
    assert X.shape[0] == 20  # 5 snapshots x 4 bearings
    assert X.shape[1] == 11  # 7 time-domain + 4 band-energy features
    assert y.shape == (20,)
    assert len(meta) == 20


def test_build_ims_features_labels(tmp_path: Path) -> None:
    """First healthy_frac fraction labelled 0; rest labelled 1 — on B1 only by default."""
    run_dir = _make_run_dir(tmp_path, n_snapshots=10, n_bearings=1)
    out = tmp_path / "features.parquet"

    _X, y, _meta = build_ims_features(run_dir, out, bearing_ids=[1], healthy_frac=0.40)

    # 4 healthy (floor(10*0.4)=4), 6 degraded
    assert int((y == 0).sum()) == 4
    assert int((y == 1).sum()) == 6


def test_build_ims_features_non_failure_bearing_is_all_healthy(tmp_path: Path) -> None:
    """Bearings outside documented_failure_bearings stay y=0 throughout (paper ground truth)."""
    run_dir = _make_run_dir(tmp_path, n_snapshots=10, n_bearings=1)
    out = tmp_path / "features.parquet"

    # B1 included but NOT in the documented failure set → all healthy.
    _, y, _ = build_ims_features(
        run_dir,
        out,
        bearing_ids=[1],
        healthy_frac=0.40,
        documented_failure_bearings=(),
    )
    assert int((y == 0).sum()) == 10
    assert int((y == 1).sum()) == 0


def test_build_ims_features_mixed_bearings_only_failure_bearing_gets_y1(tmp_path: Path) -> None:
    """With 4 bearings and default ``documented_failure_bearings=(1,)``,
    only B1 receives y=1 in its late period; B2/B3/B4 remain y=0 throughout.
    """
    run_dir = _make_run_dir(tmp_path, n_snapshots=10, n_bearings=4)
    out = tmp_path / "features.parquet"

    _, y, meta = build_ims_features(run_dir, out, healthy_frac=0.40)

    # Each bearing has 10 snapshots: floor(10*0.4)=4 healthy, 6 degraded — but
    # only B1 is in the documented failure set, so total y=1 = 6 (B1 only).
    assert int((y == 1).sum()) == 6
    assert int((y == 0).sum()) == 34  # 4 (B1 healthy) + 30 (B2/3/4 fully healthy)

    # Cross-check via meta: every y=1 row belongs to B1.
    y_series = pd.Series(y, name="y")
    failing = meta.assign(y=y_series)[y_series == 1]
    assert set(failing["bearing_id"].unique()) == {1}


def test_build_ims_features_parquet_saved(tmp_path: Path) -> None:
    """Parquet is written with meta columns."""
    run_dir = _make_run_dir(tmp_path, n_snapshots=3, n_bearings=2)
    out = tmp_path / "features.parquet"

    build_ims_features(run_dir, out, bearing_ids=[1, 2])

    assert out.exists()
    df = pd.read_parquet(out)
    assert "_meta_timestamp" in df.columns
    assert "_meta_bearing_id" in df.columns
    assert "_meta_filename" in df.columns
    assert "_meta_y" in df.columns
    assert "_meta_snapshot_idx" in df.columns


def test_build_ims_features_binary_labels(tmp_path: Path) -> None:
    """Labels are strictly {0, 1}."""
    run_dir = _make_run_dir(tmp_path, n_snapshots=5, n_bearings=1)
    out = tmp_path / "features.parquet"

    _, y, _ = build_ims_features(run_dir, out, bearing_ids=[1])

    assert set(np.unique(y)).issubset({0, 1})


def test_build_ims_features_bearing_filter(tmp_path: Path) -> None:
    """bearing_ids parameter restricts which bearings are included."""
    run_dir = _make_run_dir(tmp_path, n_snapshots=4, n_bearings=4)
    out = tmp_path / "features.parquet"

    X, _y, meta = build_ims_features(run_dir, out, bearing_ids=[1, 2])

    assert X.shape[0] == 8  # 4 snapshots x 2 bearings
    assert set(meta["bearing_id"].unique()) == {1, 2}


def test_build_ims_features_parquet_row_count(tmp_path: Path) -> None:
    """Parquet row count matches X row count."""
    run_dir = _make_run_dir(tmp_path, n_snapshots=3, n_bearings=3)
    out = tmp_path / "features.parquet"

    X, _, _ = build_ims_features(run_dir, out)

    df = pd.read_parquet(out)
    assert len(df) == X.shape[0]


def test_build_ims_features_no_nan(tmp_path: Path) -> None:
    """Feature matrix contains no NaN or infinite values."""
    run_dir = _make_run_dir(tmp_path, n_snapshots=5, n_bearings=2)
    out = tmp_path / "features.parquet"

    X, _, _ = build_ims_features(run_dir, out)

    assert not np.any(np.isnan(X))
    assert not np.any(np.isinf(X))
