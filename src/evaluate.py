"""Evaluation utilities: bootstrap CI, ROC curve, and model comparison plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
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


def plot_comparison(
    results: pd.DataFrame,
    out_path: Path = Path("results/figures/model_comparison.png"),
) -> None:
    """Bar chart comparing ROC-AUC and F1 across models, with bootstrap CI error bars.

    Two groups (ROC-AUC, F1), four bars each (one per model).
    Error bars span the 95% bootstrap CI (low, high).
    Saves PNG to *out_path*; no plt.show().
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = results["model"].tolist()
    n = len(models)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, metric, label in [
        (axes[0], "roc_auc", "ROC-AUC"),
        (axes[1], "f1", "F1"),
    ]:
        means = results[f"{metric}_mean"].values
        lows = results[f"{metric}_low"].values
        highs = results[f"{metric}_high"].values
        yerr = np.array([means - lows, highs - means])

        ax.bar(x, means, width=width * 2, yerr=yerr, capsize=4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label} — 95% bootstrap CI")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="k", lw=0.8, linestyle="--", label="random")
        ax.legend(fontsize=8)

    fig.suptitle("Model Comparison — IMS/NASA Bearing Anomaly Detection", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
