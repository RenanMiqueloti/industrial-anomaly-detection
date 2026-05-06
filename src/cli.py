"""CLI entry point: download | features | train | eval | compare."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DATA_RAW = Path("data/raw")
_DATA_FEATURES = Path("data/features/features.parquet")
_RESULTS = Path("results")
_MODEL_PATH = _RESULTS / "iforest_model.joblib"
_METRICS_PATH = _RESULTS / "iforest_metrics.json"
_ROC_PATH = _RESULTS / "figures" / "iforest_roc.png"


def _cmd_download(_args: argparse.Namespace) -> None:
    import subprocess
    import sys

    _DATA_RAW.mkdir(parents=True, exist_ok=True)
    mirror = "https://github.com/jpdias/cwru-bearing-dataset"
    logger.info("Cloning CWRU mirror: %s", mirror)
    result = subprocess.run(
        ["git", "clone", "--depth=1", mirror, str(_DATA_RAW / "cwru-mirror")],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Mirror cloned to %s", _DATA_RAW / "cwru-mirror")
    else:
        logger.warning("Mirror clone failed: %s", result.stderr.strip())
        logger.warning(
            "Place .mat files under data/raw/ manually, then re-run: make features train eval"
        )
        sys.exit(1)


def _cmd_features(_args: argparse.Namespace) -> None:
    from src.dataset import build_feature_matrix

    logger.info("Extracting features from %s …", _DATA_RAW)
    X, y, meta = build_feature_matrix(_DATA_RAW, _DATA_FEATURES)
    logger.info(
        "Feature matrix: X=%s  y=%s  classes=%s",
        X.shape,
        y.shape,
        meta["class"].value_counts().to_dict(),
    )


def _cmd_train(_args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.models.iforest import IForestDetector

    if not _DATA_FEATURES.exists():
        raise FileNotFoundError(f"Run 'make features' first. Missing: {_DATA_FEATURES}")

    df = pd.read_parquet(_DATA_FEATURES)
    feature_cols = [c for c in df.columns if not c.startswith("_meta_")]
    X = df[feature_cols].values.astype(np.float64)
    y = df["_meta_y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_healthy = X_train[y_train == 0]

    logger.info("Fitting IForest on %d healthy windows …", len(X_healthy))
    model = IForestDetector()
    model.fit(X_healthy)
    model.save(_MODEL_PATH)
    logger.info("Model saved → %s", _MODEL_PATH)

    _RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(_RESULTS / "X_test.npy", X_test)
    np.save(_RESULTS / "y_test.npy", y_test)
    logger.info("Test split saved → %s", _RESULTS)


def _cmd_eval(_args: argparse.Namespace) -> None:
    import numpy as np

    from src.evaluate import bootstrap_ci, plot_roc
    from src.models.iforest import IForestDetector

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Run 'make train' first. Missing: {_MODEL_PATH}")

    X_test = np.load(_RESULTS / "X_test.npy")
    y_test = np.load(_RESULTS / "y_test.npy")

    model = IForestDetector.load(_MODEL_PATH)
    scores = model.score(X_test)

    metrics = bootstrap_ci(y_test, scores)
    logger.info("ROC-AUC: %.3f  [%.3f, %.3f]", *metrics["roc_auc"])
    logger.info("F1:      %.3f  [%.3f, %.3f]", *metrics["f1"])

    _RESULTS.mkdir(parents=True, exist_ok=True)
    _METRICS_PATH.write_text(
        json.dumps(
            {k: {"mean": v[0], "low": v[1], "high": v[2]} for k, v in metrics.items()},
            indent=2,
        )
    )
    logger.info("Metrics → %s", _METRICS_PATH)

    plot_roc(y_test, scores, _ROC_PATH)
    logger.info("ROC curve → %s", _ROC_PATH)


def _cmd_compare(_args: argparse.Namespace) -> None:
    from src.compare import run_comparison

    if not (_RESULTS / "X_test.npy").exists():
        raise FileNotFoundError("Run 'make train' first to generate results/X_test.npy")

    results = run_comparison(
        X_test_path=_RESULTS / "X_test.npy",
        y_test_path=_RESULTS / "y_test.npy",
        out_dir=_RESULTS,
    )
    logger.info("\n%s", results.to_string(index=False))
    logger.info("Comparison saved → %s", _RESULTS / "comparison.parquet")
    logger.info("Figure → %s", _RESULTS / "figures" / "model_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="industrial-anomaly-detection pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download CWRU bearing data to data/raw/")
    sub.add_parser("features", help="Extract features → data/features/features.parquet")
    sub.add_parser("train", help="Fit IsolationForest on healthy windows")
    sub.add_parser("eval", help="Evaluate with bootstrap CI and save results")
    sub.add_parser("compare", help="Run all 4 models and save comparison table + figure")

    args = parser.parse_args()
    dispatch = {
        "download": _cmd_download,
        "features": _cmd_features,
        "train": _cmd_train,
        "eval": _cmd_eval,
        "compare": _cmd_compare,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
