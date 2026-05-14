"""3-model comparison: IForest, OC-SVM, AutoEncoder.

Trains each model on healthy windows, evaluates on the held-out test set,
and returns a summary DataFrame with bootstrap CI metrics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import bootstrap_ci, plot_comparison
from src.models.autoencoder import AutoEncoderDetector
from src.models.base import BaseDetector
from src.models.iforest import IForestDetector
from src.models.ocsvm import OCSVMDetector

logger = logging.getLogger(__name__)

# Default artifact directory; overridable via IAD_RESULTS_DIR for deployments
# that mount results elsewhere (e.g. /var/lib/iad/results).
_RESULTS_DIR = Path(os.getenv("IAD_RESULTS_DIR", "results"))

_MODELS = {
    "IsolationForest": IForestDetector,
    "OC-SVM": OCSVMDetector,
    "AutoEncoder": AutoEncoderDetector,
}

_MODEL_THRESHOLD_KEY = {
    "IsolationForest": "iforest",
    "OC-SVM": "ocsvm",
    "AutoEncoder": "ae",
}


def run_comparison(
    X_test_path: Path | None = None,
    y_test_path: Path | None = None,
    X_train_path: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Train all 3 detectors on healthy windows, evaluate with bootstrap CI.

    Uses the same temporal split produced by ``make train`` so every model
    trains on identical healthy windows and scores the same held-out test set —
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
        "AutoEncoder": "ae_model.joblib",
    }

    # Per-bearing healthy-set bearing IDs — written by `cli train`. When
    # present we compute per-bearing p99 thresholds for each model, matching
    # what IForest already gets and avoiding silent fallback to test-set p99
    # in the dashboard.
    bid_path = out_dir / "bid_train_healthy.npy"
    bid_train_healthy = np.load(bid_path) if bid_path.exists() else None

    rows = []
    fitted_models: dict[str, BaseDetector] = {}
    for name, ModelCls in _MODELS.items():
        model = ModelCls()
        t0 = time.perf_counter()
        model.fit(X_healthy)
        train_s = time.perf_counter() - t0

        model.save(out_dir / _MODEL_SAVE_NAMES[name])
        fitted_models[name] = model

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

    if bid_train_healthy is not None:
        _persist_per_bearing_thresholds(
            out_dir / "threshold.json", fitted_models, X_healthy, bid_train_healthy
        )

    results = pd.DataFrame(rows)
    out_parquet = out_dir / "comparison.parquet"
    results.to_parquet(out_parquet, index=False)

    fig_path = out_dir / "figures" / "model_comparison.png"
    plot_comparison(results, fig_path)

    return results


def _persist_per_bearing_thresholds(
    threshold_path: Path,
    fitted_models: dict[str, BaseDetector],
    X_healthy: np.ndarray,
    bid_train_healthy: np.ndarray,
    min_samples_per_bearing: int = 5,
) -> None:
    """Compute per-bearing p99 thresholds for each fitted model and merge into
    the shared ``threshold.json``.

    Keeps existing keys (e.g. ``iforest_b1`` already written by ``cli train``)
    and adds ``ocsvm_bN`` / ``ae_bN`` so the dashboard's per-bearing slider
    works the same for every model.
    """
    data = json.loads(threshold_path.read_text()) if threshold_path.exists() else {}

    bearing_ids = sorted({int(b) for b in np.unique(bid_train_healthy)})
    for name, model in fitted_models.items():
        key = _MODEL_THRESHOLD_KEY.get(name, name.lower())
        # Global p99 (legacy flat key) for back-compat with anything that read
        # the JSON before per-bearing keys existed.
        global_score = model.score(X_healthy)
        data[key] = float(np.percentile(global_score, 99))
        for bid in bearing_ids:
            mask = bid_train_healthy == bid
            if int(mask.sum()) < min_samples_per_bearing:
                continue
            thr_bid = float(np.percentile(model.score(X_healthy[mask]), 99))
            data[f"{key}_b{bid}"] = thr_bid
            logger.info("Threshold %s bearing %d: %.6f", name, bid, thr_bid)

    threshold_path.write_text(json.dumps(data, indent=2))
