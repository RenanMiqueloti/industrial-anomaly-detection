"""Smoke tests for the pure helper functions in src/dashboard.py.

These functions don't touch Streamlit globals — they're pure numpy/pandas
logic — so they can be exercised in unit-test isolation. The Streamlit-
rendering paths (_fig_*, _hero, _kpi_row, etc.) remain integration-tested
via manual dashboard launches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Streamlit is required for `import src.dashboard` to even succeed; skip
# gracefully if a stripped-down environment doesn't have it (e.g. minimal
# api-only deploys).
pytest.importorskip("streamlit", reason="streamlit not installed")

from src import dashboard as dash


# ---------------------------------------------------------------------------
# _safe_auc
# ---------------------------------------------------------------------------
def test_safe_auc_returns_value_for_valid_input() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.standard_normal(50), rng.standard_normal(50) + 3])
    auc = dash._safe_auc(y, scores)
    assert auc is not None
    assert 0.5 < auc <= 1.0


def test_safe_auc_returns_none_for_single_class() -> None:
    """AUC is undefined when y has only one class — helper returns None instead of raising."""
    y = np.zeros(20, dtype=int)
    scores = np.linspace(0, 1, 20)
    assert dash._safe_auc(y, scores) is None


def test_safe_auc_returns_none_for_nan_scores() -> None:
    """NaN-laden scores must not crash the dashboard — caught and logged."""
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, np.nan, 0.9, np.nan])
    assert dash._safe_auc(y, scores) is None


# ---------------------------------------------------------------------------
# _default_threshold (file-based + fallback paths)
# ---------------------------------------------------------------------------
def test_default_threshold_uses_global_json_when_file_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threshold_file = tmp_path / "threshold.json"
    threshold_file.write_text(json.dumps({"iforest": 0.42}))
    monkeypatch.setattr(dash, "_THRESHOLD_JSON", threshold_file)

    scores = np.linspace(0, 1, 100)
    y_test = np.array([0] * 50 + [1] * 50)
    assert dash._default_threshold(scores, y_test, "IsolationForest") == pytest.approx(0.42)


def test_default_threshold_uses_per_bearing_when_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threshold_file = tmp_path / "threshold.json"
    threshold_file.write_text(json.dumps({"iforest": 0.5, "iforest_b1": 0.30}))
    monkeypatch.setattr(dash, "_THRESHOLD_JSON", threshold_file)

    scores = np.linspace(0, 1, 100)
    y_test = np.array([0] * 50 + [1] * 50)
    assert dash._default_threshold(
        scores, y_test, "IsolationForest", bearing_id=1
    ) == pytest.approx(0.30)


def test_default_threshold_falls_back_to_p99_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dash, "_THRESHOLD_JSON", tmp_path / "nonexistent.json")

    scores = np.arange(100, dtype=float)
    y_test = np.array([0] * 100)
    # p99 of 0..99 is 98.01
    result = dash._default_threshold(scores, y_test, "IsolationForest")
    assert result == pytest.approx(98.01, rel=1e-3)


def test_default_threshold_falls_back_to_p50_when_no_normal_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dash, "_THRESHOLD_JSON", tmp_path / "nonexistent.json")
    scores = np.arange(100, dtype=float)
    y_test = np.ones(100, dtype=int)  # no y==0 windows
    assert dash._default_threshold(scores, y_test, "IsolationForest") == pytest.approx(49.5)


def test_default_threshold_handles_corrupt_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_file = tmp_path / "threshold.json"
    bad_file.write_text("{this is not valid json")
    monkeypatch.setattr(dash, "_THRESHOLD_JSON", bad_file)

    scores = np.arange(100, dtype=float)
    y_test = np.array([0] * 100)
    # Falls through to data-derived p99 instead of crashing.
    assert dash._default_threshold(scores, y_test, "IsolationForest") == pytest.approx(
        98.01, rel=1e-3
    )


# ---------------------------------------------------------------------------
# _load_all_thresholds
# ---------------------------------------------------------------------------
def test_load_all_thresholds_returns_per_bearing_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threshold_file = tmp_path / "threshold.json"
    threshold_file.write_text(
        json.dumps(
            {
                "iforest": 0.5,
                "iforest_b1": 0.30,
                "iforest_b2": 0.55,
                "iforest_b3": 0.65,
                # b4 missing on purpose
                "ocsvm_b1": 0.10,  # different model, must be ignored
            }
        )
    )
    monkeypatch.setattr(dash, "_THRESHOLD_JSON", threshold_file)

    result = dash._load_all_thresholds("IsolationForest")
    assert result == {1: pytest.approx(0.30), 2: pytest.approx(0.55), 3: pytest.approx(0.65)}


def test_load_all_thresholds_returns_empty_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dash, "_THRESHOLD_JSON", tmp_path / "nonexistent.json")
    assert dash._load_all_thresholds("IsolationForest") == {}


def test_load_all_thresholds_returns_empty_on_corrupt_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_file = tmp_path / "threshold.json"
    bad_file.write_text("not json")
    monkeypatch.setattr(dash, "_THRESHOLD_JSON", bad_file)
    assert dash._load_all_thresholds("IsolationForest") == {}


# ---------------------------------------------------------------------------
# _get_timestamps
# ---------------------------------------------------------------------------
def test_get_timestamps_returns_indexed_when_meta_present() -> None:
    meta = pd.DataFrame(
        {
            "_meta_timestamp": pd.date_range("2004-02-12", periods=3, freq="10min"),
        }
    )
    ts = dash._get_timestamps(meta, n=3)
    assert len(ts) == 3
    assert isinstance(ts, pd.DatetimeIndex)


def test_get_timestamps_returns_empty_when_meta_missing() -> None:
    meta = pd.DataFrame({"unrelated": [1, 2, 3]})
    ts = dash._get_timestamps(meta, n=3)
    assert len(ts) == 0


# ---------------------------------------------------------------------------
# _predict_failure
# ---------------------------------------------------------------------------
def test_predict_failure_returns_none_for_short_series() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    timestamps = pd.DatetimeIndex(pd.date_range("2004-02-12", periods=3, freq="10min"))
    assert dash._predict_failure(scores, timestamps, threshold=1.0) is None


def test_predict_failure_returns_none_for_flat_trend() -> None:
    """Negative or zero slope means scores aren't rising — no failure projection."""
    scores = np.full(50, 0.3)
    timestamps = pd.DatetimeIndex(pd.date_range("2004-02-12", periods=50, freq="10min"))
    assert dash._predict_failure(scores, timestamps, threshold=1.0) is None


def test_predict_failure_suppresses_noise_drift_far_below_threshold() -> None:
    """Noisy scores well below threshold should not produce a projection.

    Regression on a healthy bearing's trailing window can yield a tiny positive
    slope from noise. Without the capture guard, that drift would render a
    misleading "failure predicted in Xh" card on a clearly healthy bearing.
    """
    rng = np.random.default_rng(42)
    n = 100
    # Mean ~0.30 with mild upward drift, threshold at 0.60 — capture stays at 0
    scores = rng.normal(0.30, 0.02, n) + np.linspace(0, 0.05, n)
    timestamps = pd.DatetimeIndex(pd.date_range("2004-02-12", periods=n, freq="10min"))

    assert dash._predict_failure(scores, timestamps, threshold=0.60) is None
