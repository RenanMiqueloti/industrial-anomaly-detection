"""Tests for src/drift.py — PSI computation and drift flagging."""

from __future__ import annotations

import numpy as np
import pytest

from src.drift import PSI_ALERT_THRESHOLD, compute_psi, compute_psi_per_feature, flag_drift


def test_psi_zero_when_identical() -> None:
    """PSI between a distribution and itself should be ~0."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(1000)
    psi = compute_psi(data, data)
    assert psi == pytest.approx(0.0, abs=0.05), f"Expected PSI ≈ 0 for identical data, got {psi}"


def test_psi_high_when_shifted() -> None:
    """PSI should be large when current distribution is shifted far from reference."""
    rng = np.random.default_rng(1)
    reference = rng.standard_normal(1000)
    current = rng.standard_normal(1000) + 3.0
    psi = compute_psi(reference, current)
    assert psi > 0.5, f"Expected PSI > 0.5 for heavily shifted distribution, got {psi}"


def test_psi_per_feature_keys_match() -> None:
    """compute_psi_per_feature returns a dict with exactly the given feature names."""
    rng = np.random.default_rng(2)
    n_features = 5
    feature_names = [f"feat_{i}" for i in range(n_features)]
    reference = rng.standard_normal((200, n_features))
    current = rng.standard_normal((100, n_features))
    result = compute_psi_per_feature(reference, current, feature_names)
    assert set(result.keys()) == set(feature_names)
    assert len(result) == n_features
    for v in result.values():
        assert np.isfinite(v), "All PSI values must be finite"


def test_flag_drift_threshold() -> None:
    """flag_drift returns only features with PSI strictly above the threshold."""
    psi_dict = {"a": 0.1, "b": 0.3, "c": 0.05}
    flagged = flag_drift(psi_dict, threshold=PSI_ALERT_THRESHOLD)
    assert flagged == ["b"], f"Expected only 'b' to be flagged, got {flagged}"
