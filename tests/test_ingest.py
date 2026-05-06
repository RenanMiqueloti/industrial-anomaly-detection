"""Tests for src.ingest — windowing and CWRU .mat loading."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from src.ingest import load_cwru, window


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
    assert len(wins) == 1  # second window would need indices [2048:4096] but signal ends at 2049


def test_load_cwru_synthetic(tmp_path) -> None:
    """A synthetic .mat with X097_DE_time loads as class='normal', shape preserved."""
    mat_data = {"X097_DE_time": np.zeros(120_000).reshape(-1, 1)}
    scipy.io.savemat(str(tmp_path / "97.mat"), mat_data)

    df = load_cwru(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["class"] == "normal"
    assert row["signal"].shape == (120_000,)
    assert row["signal"].dtype == np.float64


def test_load_cwru_ir_file(tmp_path) -> None:
    """File 105.mat with X105_DE_time infers class='IR'."""
    mat_data = {"X105_DE_time": np.ones(4096).reshape(-1, 1)}
    scipy.io.savemat(str(tmp_path / "105.mat"), mat_data)

    df = load_cwru(tmp_path)
    assert df.iloc[0]["class"] == "IR"


def test_load_cwru_empty_raises(tmp_path) -> None:
    """Empty directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_cwru(tmp_path)
