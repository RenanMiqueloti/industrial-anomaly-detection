"""Tests for src.ingest — IMS snapshot loading and windowing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.ingest import IMS_ROWS, _parse_timestamp, load_ims_run, window

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_snapshot(
    path: Path,
    n_bearings: int = 4,
    n_rows: int = IMS_ROWS,
    seed: int = 0,
) -> None:
    """Write a synthetic IMS snapshot file (tab-separated, no header)."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_bearings))
    np.savetxt(str(path), data, delimiter="\t")


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------


def test_parse_timestamp_valid() -> None:
    ts = _parse_timestamp("2004.02.12.10.32.39")
    assert ts == datetime(2004, 2, 12, 10, 32, 39)


def test_parse_timestamp_valid_boundary() -> None:
    ts = _parse_timestamp("2003.10.22.12.06.24")
    assert ts == datetime(2003, 10, 22, 12, 6, 24)


def test_parse_timestamp_invalid_mfpt_name() -> None:
    assert _parse_timestamp("baseline_1") is None


def test_parse_timestamp_invalid_random() -> None:
    assert _parse_timestamp("not_a_timestamp") is None
    assert _parse_timestamp("2004.02.12") is None  # too short
    assert _parse_timestamp("") is None


# ---------------------------------------------------------------------------
# load_ims_run — basic cases
# ---------------------------------------------------------------------------


def test_load_ims_run_single_snapshot(tmp_path: Path) -> None:
    """One snapshot with 4 bearings loads all 4 rows."""
    _write_snapshot(tmp_path / "2004.02.12.10.32.39", n_bearings=4)

    df = load_ims_run(tmp_path)

    assert len(df) == 4
    assert set(df.columns) >= {"timestamp", "bearing_id", "signal", "filename"}
    assert set(df["bearing_id"]) == {1, 2, 3, 4}
    assert df["signal"].iloc[0].shape == (IMS_ROWS,)
    assert df["signal"].iloc[0].dtype == np.float64
    assert df["timestamp"].iloc[0] == datetime(2004, 2, 12, 10, 32, 39)


def test_load_ims_run_bearing_filter(tmp_path: Path) -> None:
    """bearing_ids filter restricts which columns are returned."""
    _write_snapshot(tmp_path / "2004.02.12.10.32.39", n_bearings=4)

    df = load_ims_run(tmp_path, bearing_ids=[1, 3])

    assert len(df) == 2
    assert set(df["bearing_id"]) == {1, 3}


def test_load_ims_run_single_bearing(tmp_path: Path) -> None:
    _write_snapshot(tmp_path / "2004.02.12.10.32.39", n_bearings=4)

    df = load_ims_run(tmp_path, bearing_ids=[2])

    assert len(df) == 1
    assert df["bearing_id"].iloc[0] == 2


def test_load_ims_run_multiple_snapshots_sorted(tmp_path: Path) -> None:
    """Multiple snapshots for one bearing are sorted chronologically."""
    ts_list = [
        "2004.02.12.10.52.39",
        "2004.02.12.10.32.39",  # intentionally out of order on disk
        "2004.02.12.10.42.39",
    ]
    for ts in ts_list:
        _write_snapshot(tmp_path / ts, n_bearings=1, seed=hash(ts) % 2**32)

    df = load_ims_run(tmp_path, bearing_ids=[1])

    assert len(df) == 3
    timestamps = list(df["timestamp"])
    assert timestamps == sorted(timestamps), "Rows must be sorted chronologically"


def test_load_ims_run_multiple_snapshots_all_bearings(tmp_path: Path) -> None:
    """3 snapshots × 4 bearings = 12 rows total."""
    for ts in ["2004.02.12.10.32.39", "2004.02.12.10.42.39", "2004.02.12.10.52.39"]:
        _write_snapshot(tmp_path / ts, n_bearings=4)

    df = load_ims_run(tmp_path)

    assert len(df) == 12
    assert df["bearing_id"].nunique() == 4


def test_load_ims_run_ignores_non_timestamp_files(tmp_path: Path) -> None:
    """Files whose names are not valid timestamps are silently skipped."""
    _write_snapshot(tmp_path / "2004.02.12.10.32.39", n_bearings=2)
    (tmp_path / "README.txt").write_text("not a snapshot")
    (tmp_path / "baseline_1.mat").write_text("")

    df = load_ims_run(tmp_path)

    assert len(df) == 2


def test_load_ims_run_empty_dir_raises(tmp_path: Path) -> None:
    """Empty directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No IMS snapshot files"):
        load_ims_run(tmp_path)


def test_load_ims_run_no_matching_bearing_raises(tmp_path: Path) -> None:
    """Requesting bearing not present in data raises FileNotFoundError."""
    _write_snapshot(tmp_path / "2004.02.12.10.32.39", n_bearings=4)

    with pytest.raises(FileNotFoundError, match="No valid snapshots"):
        load_ims_run(tmp_path, bearing_ids=[9])


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------


def test_window_basic() -> None:
    signal = np.arange(5000, dtype=float)
    wins = list(window(signal, length=2048, hop=2048))
    assert len(wins) == 2
    assert wins[0].shape == (2048,)


def test_window_overlap() -> None:
    signal = np.arange(5000, dtype=float)
    wins = list(window(signal, length=2048, hop=1024))
    assert len(wins) == 3


def test_window_discards_tail() -> None:
    signal = np.zeros(2049)
    wins = list(window(signal, length=2048, hop=2048))
    assert len(wins) == 1


def test_window_full_snapshot() -> None:
    """IMS snapshots (20 480 samples) used as a single window."""
    signal = np.zeros(IMS_ROWS)
    wins = list(window(signal, length=IMS_ROWS, hop=IMS_ROWS))
    assert len(wins) == 1
    assert wins[0].shape == (IMS_ROWS,)
