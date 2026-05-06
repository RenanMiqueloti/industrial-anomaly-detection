"""Tests for src.explain — SHAP explanations.

Physical-truth design: test_sanity_only_rms_differs validates that the SHAP
pipeline correctly attributes anomaly scores to the single shifted feature.
"""

from __future__ import annotations

import numpy as np
import pytest
import shap

from src.explain import explain
from src.models.base import BaseDetector
from src.models.iforest import IForestDetector
from src.models.ocsvm import OCSVMDetector

# Shared fixture data dimensions
N_FEATURES = 11
FEATURE_NAMES = [
    "rms",
    "peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "std",
    "p2p",
    "band_0_500",
    "band_500_1500",
    "band_1500_3000",
    "band_3000_6000",
]


@pytest.fixture(scope="module")
def fitted_iforest():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((300, N_FEATURES))
    model = IForestDetector(n_estimators=50, random_state=42)
    model.fit(X_train)
    return model, X_train


@pytest.fixture(scope="module")
def fitted_ocsvm():
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((100, N_FEATURES))
    model = OCSVMDetector()
    model.fit(X_train)
    return model, X_train


def test_explain_iforest_uses_treeexplainer(fitted_iforest) -> None:
    """IForestDetector explanation has correct shape and feature names."""
    model, _X_train = fitted_iforest
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, N_FEATURES))

    exp = explain(model, X, FEATURE_NAMES, eval_size=20)

    assert isinstance(exp, shap.Explanation)
    assert exp.values.shape == (20, N_FEATURES)
    assert exp.data.shape == (20, N_FEATURES)
    assert exp.feature_names == FEATURE_NAMES


def test_explain_ocsvm_uses_kernelexplainer(fitted_ocsvm) -> None:
    """OCSVMDetector explanation via KernelExplainer has correct shape."""
    model, X_train = fitted_ocsvm
    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, N_FEATURES))

    exp = explain(model, X, FEATURE_NAMES, X_background=X_train, bg_size=10, eval_size=5)

    assert isinstance(exp, shap.Explanation)
    assert exp.values.shape == (5, N_FEATURES)
    assert exp.feature_names == FEATURE_NAMES


def test_sanity_only_rms_differs(fitted_iforest) -> None:
    """Physical sanity: when only rms is anomalous, SHAP must attribute to rms.

    Constructs a synthetic dataset where all 11 features are N(0,1) for healthy
    windows, but only feature 0 (rms) is shifted +5 in anomalous windows.
    TreeExplainer on IForest must identify rms as the top-attributed feature
    (largest mean |SHAP value| across the anomalous eval set).
    """
    idx_rms = FEATURE_NAMES.index("rms")  # = 0

    rng = np.random.default_rng(42)
    X_healthy = rng.standard_normal((300, N_FEATURES))
    X_anomalous = rng.standard_normal((50, N_FEATURES))
    X_anomalous[:, idx_rms] += 5.0  # only rms is different

    model = IForestDetector(n_estimators=100, random_state=42)
    model.fit(X_healthy)

    exp = explain(model, X_anomalous, FEATURE_NAMES, eval_size=None)

    mean_abs = np.abs(exp.values).mean(axis=0)
    top_feature = int(np.argmax(mean_abs))
    assert top_feature == idx_rms, (
        f"Expected top SHAP feature to be 'rms' (idx {idx_rms}), "
        f"got idx {top_feature} ('{FEATURE_NAMES[top_feature]}'). "
        f"mean |SHAP|: {dict(zip(FEATURE_NAMES, mean_abs.round(4), strict=True))}"
    )


def test_explain_handles_eval_size_subsampling(fitted_iforest) -> None:
    """explain() with eval_size=20 returns exactly 20 rows from a 100-row input."""
    model, _ = fitted_iforest
    rng = np.random.default_rng(5)
    X = rng.standard_normal((100, N_FEATURES))

    exp = explain(model, X, FEATURE_NAMES, eval_size=20)

    assert exp.values.shape[0] == 20
    assert exp.data.shape[0] == 20


def test_explain_unknown_model_raises() -> None:
    """A BaseDetector subclass that isn't one of the 4 supported types raises ValueError."""

    class _UnknownDetector(BaseDetector):
        def fit(self, X_healthy):
            return self

        def score(self, X):
            return np.zeros(len(X))

        def save(self, path):
            pass

        @classmethod
        def load(cls, path):
            return cls()

    model = _UnknownDetector()
    X = np.random.default_rng(6).standard_normal((10, N_FEATURES))

    with pytest.raises(ValueError, match="Unsupported model type"):
        explain(model, X, FEATURE_NAMES)
