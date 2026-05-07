"""SHAP explanations for the 4 anomaly detectors.

Public API
----------
    explain(model, X, feature_names, ...) -> shap.Explanation
    save_summary_plot(explanation, out_path)

Dispatch logic
--------------
    IForestDetector  → TreeExplainer  (exact, O(TLD²), fast)
    OCSVMDetector
    LOFDetector       → KernelExplainer  (model-agnostic, slower)
    AutoEncoderDetector

For KernelExplainer, ``X_background`` controls the reference distribution.
``bg_size`` and ``eval_size`` cap the computation; defaults fit a single-core
CI run in under 60 s on IMS-sized feature matrices (11 features).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.autoencoder import AutoEncoderDetector
from src.models.base import BaseDetector
from src.models.iforest import IForestDetector
from src.models.lof import LOFDetector
from src.models.ocsvm import OCSVMDetector

DEFAULT_BG_SIZE: int = 50
DEFAULT_EVAL_SIZE: int = 100


def explain(
    model: BaseDetector,
    X: np.ndarray,
    feature_names: list[str],
    X_background: np.ndarray | None = None,
    bg_size: int = DEFAULT_BG_SIZE,
    eval_size: int | None = DEFAULT_EVAL_SIZE,
    random_state: int = 42,
) -> shap.Explanation:
    """Generate SHAP values for ``X`` using the appropriate explainer.

    Parameters
    ----------
    model:
        A fitted ``BaseDetector`` subclass.
    X:
        Feature matrix to explain, shape ``(n_samples, n_features)``.
    feature_names:
        Names matching columns of ``X``.
    X_background:
        Reference data for ``KernelExplainer``.  Required for non-tree models.
        When ``None``, a subsample of ``X`` is used as a fallback (not ideal
        for small datasets — prefer passing the training healthy set).
    bg_size:
        Maximum number of background samples for ``KernelExplainer``.
    eval_size:
        Number of rows of ``X`` to explain.  ``None`` explains all rows.
        Subsampled deterministically with ``random_state``.
    random_state:
        Seed for deterministic subsampling.

    Returns
    -------
    ``shap.Explanation`` with ``.values`` shape ``(n_eval, n_features)``,
    ``.data`` set to the input features (scaled for IForest, raw for others),
    and ``.feature_names`` set.

    Raises
    ------
    ValueError
        If ``model`` is not one of the four supported detector types.
    """
    if not isinstance(model, (IForestDetector, OCSVMDetector, LOFDetector, AutoEncoderDetector)):
        raise ValueError(
            f"Unsupported model type '{type(model).__name__}'. "
            "Expected one of: IForestDetector, OCSVMDetector, LOFDetector, AutoEncoderDetector."
        )

    # Deterministic subsample of eval rows
    rng = np.random.default_rng(random_state)
    if eval_size is not None and len(X) > eval_size:
        idx = rng.choice(len(X), eval_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    if isinstance(model, IForestDetector):
        return _explain_tree(model, X_eval, feature_names)

    return _explain_kernel(model, X_eval, feature_names, X_background, bg_size, random_state)


def _explain_tree(
    model: IForestDetector,
    X_eval: np.ndarray,
    feature_names: list[str],
) -> shap.Explanation:
    """TreeExplainer path — exact SHAP for IsolationForest."""
    X_scaled = model._scaler.transform(X_eval)
    explainer = shap.TreeExplainer(model._clf)
    exp = explainer(X_scaled)
    # Flatten to 2-D if the model returns a list/3-D (multi-output edge case)
    if isinstance(exp.values, list):
        exp = shap.Explanation(
            values=exp.values[0],
            base_values=exp.base_values[0] if exp.base_values is not None else None,
            data=X_scaled,
            feature_names=feature_names,
        )
    else:
        exp.feature_names = feature_names
    return exp


def _explain_kernel(
    model: BaseDetector,
    X_eval: np.ndarray,
    feature_names: list[str],
    X_background: np.ndarray | None,
    bg_size: int,
    random_state: int,
) -> shap.Explanation:
    """KernelExplainer path — model-agnostic SHAP for OC-SVM, LOF, AE."""
    if X_background is None:
        X_background = X_eval  # fallback; caller should prefer training healthy set

    # Subsample background for speed (KernelExplainer is O(2^F × n_bg))
    if len(X_background) > bg_size:
        rng_bg = np.random.default_rng(random_state + 1)
        bg_idx = rng_bg.choice(len(X_background), bg_size, replace=False)
        background = X_background[bg_idx]
    else:
        background = X_background

    explainer = shap.KernelExplainer(model.score, background)
    sv = explainer.shap_values(X_eval, silent=True)

    # shap_values returns ndarray (n_eval, n_features) for single-output models
    sv_arr = np.array(sv)

    return shap.Explanation(
        values=sv_arr,
        base_values=np.full(len(X_eval), float(explainer.expected_value)),
        data=X_eval,
        feature_names=feature_names,
    )


def save_summary_plot(
    explanation: shap.Explanation,
    out_path: Path,
) -> None:
    """Save a SHAP summary plot (beeswarm) as PNG.

    Uses matplotlib Agg backend; never calls plt.show().
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=explanation.feature_names,
        show=False,
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close("all")
