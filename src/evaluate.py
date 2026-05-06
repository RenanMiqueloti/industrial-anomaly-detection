"""Evaluation utilities: bootstrap CI and ROC curve plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap confidence intervals for ROC-AUC and F1.

    F1 threshold is the median of *scores* on the full test set — a heuristic
    chosen because we have no labelled positives at train time (unsupervised).

    Returns
    -------
    ``{'roc_auc': (mean, low, high), 'f1': (mean, low, high)}``
    where *low*/*high* are the percentile bounds of the ``ci`` interval.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    threshold = float(np.median(scores))

    roc_aucs: list[float] = []
    f1s: list[float] = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        roc_aucs.append(float(roc_auc_score(y_b, s_b)))
        y_pred = (s_b >= threshold).astype(int)
        f1s.append(float(f1_score(y_b, y_pred, zero_division=0)))

    alpha = 1.0 - ci
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0

    def _stats(vals: list[float]) -> tuple[float, float, float]:
        arr = np.array(vals)
        return float(arr.mean()), float(np.quantile(arr, lo_q)), float(np.quantile(arr, hi_q))

    return {"roc_auc": _stats(roc_aucs), "f1": _stats(f1s)}


def plot_roc(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    """Save ROC curve PNG to *out_path* (no plt.show)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"IForest (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — IsolationForest")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
