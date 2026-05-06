"""Tests for src.ingest — windowing and MFPT .mat loading."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from src.ingest import _infer_mfpt_class, load_mfpt, window

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_mfpt_mat(path, signal: np.ndarray, sr: float = 97_656.0, load: float = 270.0) -> None:
    """Write a minimal MFPT-style .mat file with a 'bearing' struct."""
    dt = np.dtype([("gs", object), ("sr", object), ("load", object), ("rate", object)])
    bearing = np.zeros((1,), dtype=dt)
    bearing["gs"][0] = signal.reshape(-1, 1)
    bearing["sr"][0] = np.array([[sr]])
    bearing["load"][0] = np.array([[load]])
    bearing["rate"][0] = np.array([[25.0]])
    scipy.io.savemat(str(path), {"bearing": bearing})


# ---------------------------------------------------------------------------
# window()
# ---------------------------------------------------------------------------


def test_window_basic() -> None:
    """5000 samples with len=hop=2048 yields exactly 2 full windows."""
    signal = np.arange(5000, dtype=float)
    wins = list(window(signal, length=2048, hop=2048))
    assert len(wins) == 2
    assert wins[0].shape == (2048,)
    assert wins[1].shape == (2048,)


def test_window_overlap() -> None:
    """5000 samples with length=2048, hop=1024 yields 3 windows."""
    signal = np.arange(5000, dtype=float)
    wins = list(window(signal, length=2048, hop=1024))
    assert len(wins) == 3


def test_window_discards_tail() -> None:
    """If the last window would be incomplete it is not yielded."""
    signal = np.zeros(2049)
    wins = list(window(signal, length=2048, hop=2048))
    assert len(wins) == 1


# ---------------------------------------------------------------------------
# _infer_mfpt_class()
# ---------------------------------------------------------------------------


def test_infer_class_baseline() -> None:
    assert _infer_mfpt_class("baseline_1") == "normal"
    assert _infer_mfpt_class("Baseline_3") == "normal"


def test_infer_class_outer_race() -> None:
    assert _infer_mfpt_class("outer_race_1") == "OR"
    assert _infer_mfpt_class("Outer_Race_7") == "OR"


def test_infer_class_inner_race() -> None:
    assert _infer_mfpt_class("inner_race_4") == "IR"
    assert _infer_mfpt_class("Inner_Race_1") == "IR"


def test_infer_class_unknown() -> None:
    assert _infer_mfpt_class("unknown_file") == "unknown"
    assert _infer_mfpt_class("real_world_1") == "unknown"


# ---------------------------------------------------------------------------
# load_mfpt()
# ---------------------------------------------------------------------------


def test_load_mfpt_baseline(tmp_path) -> None:
    """A synthetic baseline .mat loads as class='normal' with correct shape and sr."""
    _save_mfpt_mat(tmp_path / "baseline_1.mat", np.zeros(4096), sr=97_656.0)

    df = load_mfpt(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["class"] == "normal"
    assert row["signal"].shape == (4096,)
    assert row["signal"].dtype == np.float64
    assert row["sr"] == pytest.approx(97_656.0)


def test_load_mfpt_outer_race(tmp_path) -> None:
    """outer_race_1.mat infers class='OR'."""
    _save_mfpt_mat(tmp_path / "outer_race_1.mat", np.ones(4096), sr=97_656.0)
    df = load_mfpt(tmp_path)
    assert df.iloc[0]["class"] == "OR"


def test_load_mfpt_inner_race(tmp_path) -> None:
    """inner_race_1.mat infers class='IR' with 48 828 Hz sampling rate."""
    _save_mfpt_mat(tmp_path / "inner_race_1.mat", np.ones(4096), sr=48_828.0)
    df = load_mfpt(tmp_path)
    row = df.iloc[0]
    assert row["class"] == "IR"
    assert row["sr"] == pytest.approx(48_828.0)


def test_load_mfpt_multi_file(tmp_path) -> None:
    """Multiple files in subdirectories are all discovered."""
    sub = tmp_path / "sub"
    sub.mkdir()
    _save_mfpt_mat(tmp_path / "baseline_1.mat", np.zeros(4096))
    _save_mfpt_mat(sub / "outer_race_1.mat", np.zeros(4096))
    _save_mfpt_mat(sub / "inner_race_1.mat", np.zeros(4096))

    df = load_mfpt(tmp_path)
    assert len(df) == 3
    assert set(df["class"]) == {"normal", "OR", "IR"}


def test_load_mfpt_empty_raises(tmp_path) -> None:
    """Empty directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_mfpt(tmp_path)
