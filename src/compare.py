"""4-model comparison: IForest, OC-SVM, LOF, AutoEncoder.

Trains each model on healthy windows, evaluates on the held-out test set,
and returns a summary DataFrame with bootstrap CI metrics.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import bootstrap_ci, plot_comparison
from src.models.autoencoder import AutoEncoderDetector
from src.models.iforest import IForestDetector
from src.models.lof import LOFDetector
from src.models.ocsvm import OCSVMDetector

# Default artifact directory; overridable via IAD_RESULTS_DIR for deployments
# that mount results elsewhere (e.g. /var/lib/iad/results).
_RESULTS_DIR = Path(os.getenv("IAD_RESULTS_DIR", "results"))

_MODELS = {
    "IsolationForest": IForestDetector,
    "OC-SVM": OCSVMDetector,
    "LOF": LOFDetector,
    "AutoEncoder": AutoEncoderDetector,
}


def run_comparison(
    X_test_path: Path | None = None,
    y_test_path: Path | None = None,
    X_train_path: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Train all 4 detectors on healthy windows, evaluate with bootstrap CI.

    Uses the same temporal split produced by ``make train`` so all four models
    train on identical healthy windows and score the same held-out test set —
    critical for a fair comparison.

    Parameters
    ----------
    X_test_path, y_test_path:
        Paths to the .npy files saved by ``src.cli train``.
    X_train_path:
        Path to a .npy of healthy training windows. When ``None``, defaults to
        ``out_dir / "X_train_healthy.npy"`` (saved by ``cli train``).
    out_dir:
        Directory for ``comparison.parquet`` and the comparison figure.

    Returns
    -------
    pd.DataFrame with columns:
        model, roc_auc_mean, roc_auc_low, roc_auc_high,
        f1_mean, f1_low, f1_high, train_seconds.
    """
    if out_dir is None:
        out_dir = _RESULTS_DIR
    if X_test_path is None:
        X_test_path = out_dir / "X_test.npy"
    if y_test_path is None:
        y_test_path = out_dir / "y_test.npy"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    if X_train_path is None:
        X_train_path = out_dir / "X_train_healthy.npy"
    if not Path(X_train_path).exists():
        raise FileNotFoundError(
            f"Healthy training set not found at {X_train_path}. "
            "Run 'make train' first — it saves X_train_healthy.npy alongside X_test.npy."
        )
    X_healthy = np.load(X_train_path)

    _MODEL_SAVE_NAMES = {
        "IsolationForest": "iforest_model.joblib",
        "OC-SVM": "ocsvm_model.joblib",
        "LOF": "lof_model.joblib",
        "AutoEncoder": "ae_model.joblib",
    }

    rows = []
    for name, ModelCls in _MODELS.items():
        model = ModelCls()
        t0 = time.perf_counter()
        model.fit(X_healthy)
        train_s = time.perf_counter() - t0

        model.save(out_dir / _MODEL_SAVE_NAMES[name])

        scores = model.score(X_test)
        ci = bootstrap_ci(y_test, scores)

        rows.append(
            {
                "model": name,
                "roc_auc_mean": ci["roc_auc"][0],
                "roc_auc_low": ci["roc_auc"][1],
                "roc_auc_high": ci["roc_auc"][2],
                "f1_mean": ci["f1"][0],
                "f1_low": ci["f1"][1],
                "f1_high": ci["f1"][2],
                "train_seconds": round(train_s, 2),
            }
        )

    results = pd.DataFrame(rows)
    out_parquet = out_dir / "comparison.parquet"
    results.to_parquet(out_parquet, index=False)

    fig_path = out_dir / "figures" / "model_comparison.png"
    plot_comparison(results, fig_path)

    return results
