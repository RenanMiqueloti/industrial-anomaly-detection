"""4-model comparison: IForest, OC-SVM, LOF, AutoEncoder.

Trains each model on healthy windows, evaluates on the held-out test set,
and returns a summary DataFrame with bootstrap CI metrics.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import bootstrap_ci, plot_comparison
from src.models.autoencoder import AutoEncoderDetector
from src.models.iforest import IForestDetector
from src.models.lof import LOFDetector
from src.models.ocsvm import OCSVMDetector

_MODELS = {
    "IsolationForest": IForestDetector,
    "OC-SVM": OCSVMDetector,
    "LOF": LOFDetector,
    "AutoEncoder": AutoEncoderDetector,
}


def run_comparison(
    X_test_path: Path = Path("results/X_test.npy"),
    y_test_path: Path = Path("results/y_test.npy"),
    X_train_path: Path | None = None,
    out_dir: Path = Path("results"),
) -> pd.DataFrame:
    """Train all 4 detectors on healthy windows, evaluate with bootstrap CI.

    Uses the same test split produced by ``make train`` (Sprint 1) so results
    are directly comparable across models.

    Parameters
    ----------
    X_test_path, y_test_path:
        Paths to the .npy files saved by ``src.cli train``.
    X_train_path:
        Path to a .npy of healthy training windows.  When ``None``, derived
        from the feature parquet at ``data/features/features.parquet``.
    out_dir:
        Directory for ``comparison.parquet`` and the comparison figure.

    Returns
    -------
    pd.DataFrame with columns:
        model, roc_auc_mean, roc_auc_low, roc_auc_high,
        f1_mean, f1_low, f1_high, train_seconds.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    if X_train_path is not None:
        X_healthy = np.load(X_train_path)
    else:
        _parquet = Path("data/features/features.parquet")
        if not _parquet.exists():
            raise FileNotFoundError(
                f"Feature parquet not found at {_parquet}. Run 'make features train' first."
            )
        df = pd.read_parquet(_parquet)
        feature_cols = [c for c in df.columns if not c.startswith("_meta_")]
        X_all = df[feature_cols].values.astype(np.float64)
        y_all = df["_meta_y"].values
        # Use only healthy windows from the training portion (y==0).
        # We mirror the 70/30 stratified split from cli.py to avoid data leakage.
        from sklearn.model_selection import train_test_split

        X_tr, _, y_tr, _ = train_test_split(
            X_all, y_all, test_size=0.30, stratify=y_all, random_state=42
        )
        X_healthy = X_tr[y_tr == 0]

    rows = []
    for name, ModelCls in _MODELS.items():
        model = ModelCls()
        t0 = time.perf_counter()
        model.fit(X_healthy)
        train_s = time.perf_counter() - t0

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
