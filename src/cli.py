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
    import urllib.request
    import zipfile

    from tqdm import tqdm

    _DATA_RAW.mkdir(parents=True, exist_ok=True)
    # Original mfpt.org URL is dead (domain redirected to asnt.org).
    # Figshare is the canonical stable mirror for the dataset.
    url = "https://ndownloader.figshare.com/files/53038079"
    zip_path = _DATA_RAW / "mfpt.zip"

    logger.info("Downloading MFPT dataset from %s …", url)

    class _Progress(tqdm):  # type: ignore[type-arg]
        def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with _Progress(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="MFPT") as t:
        urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)

    logger.info("Extracting to %s …", _DATA_RAW)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(_DATA_RAW)
    zip_path.unlink()

    mat_count = len(list(_DATA_RAW.rglob("*.mat")))
    logger.info("Done — %d .mat files extracted to %s", mat_count, _DATA_RAW)


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

    # Split on integer indices so we can carry meta columns alongside X/y.
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(indices, test_size=0.30, stratify=y, random_state=42)
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

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

    # Persist class metadata for the test split (needed by explain CLI for per-fault SHAP).
    meta_cols = [c for c in df.columns if c.startswith("_meta_")]
    meta_test = df.iloc[idx_test][meta_cols].reset_index(drop=True)
    meta_test.to_parquet(_RESULTS / "meta_test.parquet", index=False)
    logger.info("meta_test.parquet saved → %s", _RESULTS / "meta_test.parquet")


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


def _cmd_explain(_args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    import shap as _shap

    from src.explain import DEFAULT_EVAL_SIZE, explain, save_summary_plot
    from src.models.iforest import IForestDetector

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Run 'make train' first. Missing: {_MODEL_PATH}")
    if not (_RESULTS / "X_test.npy").exists():
        raise FileNotFoundError(f"Run 'make train' first. Missing: {_RESULTS / 'X_test.npy'}")
    if not _DATA_FEATURES.exists():
        raise FileNotFoundError(f"Run 'make features' first. Missing: {_DATA_FEATURES}")

    model = IForestDetector.load(_MODEL_PATH)
    X_test = np.load(_RESULTS / "X_test.npy")

    df = pd.read_parquet(_DATA_FEATURES)
    feature_names = [c for c in df.columns if not c.startswith("_meta_")]

    # Subsample deterministically so meta rows align with the explained rows.
    if len(X_test) > DEFAULT_EVAL_SIZE:
        rng = np.random.default_rng(42)
        eval_idx = rng.choice(len(X_test), DEFAULT_EVAL_SIZE, replace=False)
        X_eval = X_test[eval_idx]
    else:
        eval_idx = np.arange(len(X_test))
        X_eval = X_test

    logger.info("Running TreeExplainer on %d windows …", len(X_eval))
    exp = explain(model, X_eval, feature_names, eval_size=None)

    summary_path = _RESULTS / "figures" / "shap_summary.png"
    save_summary_plot(exp, summary_path)
    logger.info("SHAP summary plot → %s", summary_path)

    # Per-fault plots require meta_test.parquet (saved by _cmd_train).
    meta_path = _RESULTS / "meta_test.parquet"
    if not meta_path.exists():
        logger.warning(
            "meta_test.parquet not found — re-run 'make train' to enable per-fault SHAP plots."
        )
        return

    meta_test = pd.read_parquet(meta_path).reset_index(drop=True)
    meta_eval = meta_test.iloc[eval_idx].reset_index(drop=True)

    for cls in sorted(meta_eval["_meta_class"].unique()):
        if cls == "normal":
            continue
        mask = (meta_eval["_meta_class"] == cls).values
        if mask.sum() < 5:
            logger.warning("Skipping per-fault SHAP for '%s' — only %d windows.", cls, mask.sum())
            continue
        exp_cls = _shap.Explanation(
            values=exp.values[mask],
            base_values=exp.base_values[mask] if exp.base_values is not None else None,
            data=exp.data[mask],
            feature_names=feature_names,
        )
        fault_path = _RESULTS / "figures" / f"shap_per_fault_{cls}.png"
        save_summary_plot(exp_cls, fault_path)
        logger.info("Per-fault SHAP (%s) → %s", cls, fault_path)


def _cmd_api(_args: argparse.Namespace) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        ["uvicorn", "src.api:app", "--port", "8000", "--reload"],
        check=False,
    )
    sys.exit(result.returncode)


def _cmd_drift(_args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    from src.drift import PSI_ALERT_THRESHOLD, compute_psi_per_feature, flag_drift

    if not _DATA_FEATURES.exists():
        raise FileNotFoundError(f"Run 'make features' first. Missing: {_DATA_FEATURES}")
    if not (_RESULTS / "X_test.npy").exists():
        raise FileNotFoundError(f"Run 'make train' first. Missing: {_RESULTS / 'X_test.npy'}")

    df = pd.read_parquet(_DATA_FEATURES)
    feature_cols = [c for c in df.columns if not c.startswith("_meta_")]

    reference = df.loc[df["_meta_class"] == "normal", feature_cols].values.astype(np.float64)
    current = np.load(_RESULTS / "X_test.npy")

    if reference.shape[1] != current.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: reference={reference.shape[1]}, "
            f"current={current.shape[1]}"
        )

    psi_dict = compute_psi_per_feature(reference, current, feature_cols)
    flagged = flag_drift(psi_dict, threshold=PSI_ALERT_THRESHOLD)

    report = {
        "psi_per_feature": psi_dict,
        "flagged_features": flagged,
        "threshold": PSI_ALERT_THRESHOLD,
    }
    _RESULTS.mkdir(parents=True, exist_ok=True)
    (_RESULTS / "drift_report.json").write_text(json.dumps(report, indent=2))

    if flagged:
        logger.warning("Drift detected in %d features: %s", len(flagged), flagged)
    else:
        logger.info("No drift detected (all PSI < %.1f)", PSI_ALERT_THRESHOLD)
    logger.info("Drift report → %s", _RESULTS / "drift_report.json")


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
    sub.add_parser("download", help="Download MFPT bearing dataset to data/raw/")
    sub.add_parser("features", help="Extract features → data/features/features.parquet")
    sub.add_parser("train", help="Fit IsolationForest on healthy windows")
    sub.add_parser("eval", help="Evaluate with bootstrap CI and save results")
    sub.add_parser("compare", help="Run all 4 models and save comparison table + figure")
    sub.add_parser("explain", help="Generate SHAP explanations → results/figures/shap_*.png")
    sub.add_parser("api", help="Launch FastAPI dev server at http://localhost:8000")
    sub.add_parser("drift", help="Compute PSI drift report → results/drift_report.json")

    args = parser.parse_args()
    dispatch = {
        "download": _cmd_download,
        "features": _cmd_features,
        "train": _cmd_train,
        "eval": _cmd_eval,
        "compare": _cmd_compare,
        "explain": _cmd_explain,
        "api": _cmd_api,
        "drift": _cmd_drift,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
