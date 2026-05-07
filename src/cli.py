"""CLI entry point: download | features | train | eval | compare | explain | api | drift."""

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

# IMS default run directory (Run 2 — Bearing 1 outer-race failure)
_IMS_RUN_DIR = _DATA_RAW / "2nd_test"


def _find_ims_run_dir() -> Path:
    """Locate the IMS run directory, trying common download layouts.

    Verifies that the directory contains at least one IMS timestamp file
    (named YYYY.MM.DD.HH.MM.SS) to avoid returning a parent directory.
    """
    import re as _re

    _ts_re = _re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")

    candidates = [
        _IMS_RUN_DIR,
        _DATA_RAW / "2nd_test" / "2nd_test",  # Kaggle zip nests one extra level
        _DATA_RAW / "IMS" / "2nd_test",
        _DATA_RAW / "bearing-dataset" / "2nd_test",
    ]
    for c in candidates:
        if c.exists() and any(_ts_re.match(f.name) for f in c.iterdir() if f.is_file()):
            return c
    raise FileNotFoundError(
        "IMS Run 2 directory not found. Expected one of:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\n\nRun 'make download' or see README for manual download instructions."
    )


def _cmd_download(_args: argparse.Namespace) -> None:
    """Download the IMS/NASA bearing dataset via Kaggle CLI.

    Requires:
        pip install kaggle
        Configure ~/.kaggle/kaggle.json with your API key.
        See: https://www.kaggle.com/docs/api

    Dataset: vinayak123tyagi/bearing-dataset (IMS University of Cincinnati)
    """
    import subprocess
    import sys

    _DATA_RAW.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading IMS/NASA bearing dataset from Kaggle …")
    logger.info("Dataset: vinayak123tyagi/bearing-dataset")

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "vinayak123tyagi/bearing-dataset",
                "-p",
                str(_DATA_RAW),
                "--unzip",
            ],
            check=True,
        )
    except FileNotFoundError:
        logger.error(
            "Kaggle CLI not found.\n"
            "Install with: pip install kaggle\n"
            "Then add ~/.kaggle/kaggle.json with your API key.\n"
            "Alternative: download manually from:\n"
            "  https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset\n"
            "  and extract to data/raw/ so that data/raw/2nd_test/ exists."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        logger.error("Kaggle download failed (exit %d). Check credentials.", exc.returncode)
        sys.exit(1)

    run2 = _find_ims_run_dir()
    n_files = sum(1 for _ in run2.iterdir() if _.is_file())
    logger.info("Done — %d snapshot files in %s", n_files, run2)


def _cmd_features(_args: argparse.Namespace) -> None:
    from src.dataset import build_ims_features

    run_dir = _find_ims_run_dir()
    logger.info("Extracting features from IMS Run 2 (%s) …", run_dir)
    X, y, meta = build_ims_features(run_dir, _DATA_FEATURES)
    n_bearings = meta["bearing_id"].nunique() if "bearing_id" in meta.columns else "?"
    logger.info(
        "Feature matrix: X=%s  y=%s  bearings=%s  healthy=%d  degraded=%d",
        X.shape,
        y.shape,
        n_bearings,
        int((y == 0).sum()),
        int((y == 1).sum()),
    )


def _cmd_train(_args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    from src.models.iforest import IForestDetector

    if not _DATA_FEATURES.exists():
        raise FileNotFoundError(f"Run 'make features' first. Missing: {_DATA_FEATURES}")

    df = pd.read_parquet(_DATA_FEATURES)
    feature_cols = [c for c in df.columns if not c.startswith("_meta_")]
    X = df[feature_cols].values.astype(np.float64)
    y = df["_meta_y"].values

    # For IMS: split by unique timestamps so all bearings appear in both sets.
    # Splitting by row index would put different bearings entirely in train vs test.
    if "_meta_timestamp" in df.columns:
        all_ts = pd.to_datetime(df["_meta_timestamp"])
        unique_ts = np.sort(all_ts.unique())
        split_ts = unique_ts[int(len(unique_ts) * 0.70)]
        idx_train = np.where(all_ts < split_ts)[0]
        idx_test = np.where(all_ts >= split_ts)[0]
        logger.info(
            "Temporal split by timestamp: %d train rows, %d test rows "
            "(cutoff: %s, bearings in test: %s)",
            len(idx_train),
            len(idx_test),
            pd.Timestamp(split_ts).strftime("%Y-%m-%d %H:%M"),
            sorted(df.iloc[idx_test]["_meta_bearing_id"].unique().tolist())
            if "_meta_bearing_id" in df.columns
            else "?",
        )
    else:
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(df))
        idx_train, idx_test = train_test_split(indices, test_size=0.30, stratify=y, random_state=42)

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    X_healthy = X_train[y_train == 0]

    logger.info("Fitting IForest on %d healthy snapshots …", len(X_healthy))
    model = IForestDetector()
    model.fit(X_healthy)
    model.save(_MODEL_PATH)
    logger.info("Model saved → %s", _MODEL_PATH)

    _RESULTS.mkdir(parents=True, exist_ok=True)

    # Global threshold: p99 of all healthy training scores.
    healthy_scores = model.score(X_healthy)
    thr_global = float(np.percentile(healthy_scores, 99))

    # Per-bearing thresholds: p99 of each bearing's own healthy training scores.
    # Bearings differ in baseline vibration level — a single threshold
    # over-penalises naturally "noisier" bearings and under-flags quieter ones.
    thresholds: dict = {"iforest": thr_global}
    if "_meta_bearing_id" in df.columns:
        for bid in sorted(df.iloc[idx_train]["_meta_bearing_id"].unique()):
            bid_mask = (df.iloc[idx_train]["_meta_bearing_id"] == bid).values
            bid_healthy = bid_mask & (y_train == 0)
            if bid_healthy.sum() < 5:
                continue
            thr_bid = float(np.percentile(model.score(X_train[bid_healthy]), 99))
            thresholds[f"iforest_b{bid}"] = thr_bid
            logger.info("Threshold bearing %d (p99 healthy): %.4f", bid, thr_bid)

    # Serialise the exact feature column order the model was trained on, so
    # downstream consumers (api.py, dashboard) can build feature vectors by
    # name lookup instead of relying on dict iteration order.
    sidecar = {
        "thresholds": thresholds,
        "feature_order": feature_cols,
        # Keep the legacy flat threshold keys at the top level for backward
        # compatibility with anything that read threshold.json before this
        # change. New code should use thresholds[...] / feature_order.
        **thresholds,
    }
    threshold_path = _RESULTS / "threshold.json"
    threshold_path.write_text(json.dumps(sidecar, indent=2))
    logger.info("Threshold global (p99 healthy train): %.4f  → %s", thr_global, threshold_path)

    np.save(_RESULTS / "X_test.npy", X_test)
    np.save(_RESULTS / "y_test.npy", y_test)
    # Save healthy training rows so compare.py uses the same temporal split
    # as IForest (avoids leakage between IForest's test set and the other
    # models' training data).
    np.save(_RESULTS / "X_train_healthy.npy", X_healthy)
    logger.info("Test split saved → %s", _RESULTS)

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

    n_classes = len(np.unique(y_test))
    if n_classes >= 2:
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
    else:
        logger.warning(
            "Test set has only one class (y=%s) — skipping AUC/F1. "
            "Use a split that includes both healthy and degraded windows for eval metrics.",
            np.unique(y_test),
        )

    # Threshold: use saved value from training (set by _cmd_train), else p99 of normal test.
    _thr_path = _RESULTS / "threshold.json"
    normal_mask = y_test == 0
    if not _thr_path.exists():
        thr = (
            float(np.percentile(scores[normal_mask], 99))
            if normal_mask.any()
            else float(np.median(scores))
        )
        _thr_path.write_text(json.dumps({"iforest": thr}, indent=2))
        logger.info("Threshold (p99 healthy) → %s  (%.4f)", _thr_path, thr)
    else:
        logger.info("Threshold already saved by 'make train' → %s", _thr_path)

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

    if len(X_test) > DEFAULT_EVAL_SIZE:
        rng = np.random.default_rng(42)
        eval_idx = rng.choice(len(X_test), DEFAULT_EVAL_SIZE, replace=False)
        X_eval = X_test[eval_idx]
    else:
        eval_idx = np.arange(len(X_test))
        X_eval = X_test

    logger.info("Running TreeExplainer on %d snapshots …", len(X_eval))
    exp = explain(model, X_eval, feature_names, eval_size=None)

    summary_path = _RESULTS / "figures" / "shap_summary.png"
    save_summary_plot(exp, summary_path)
    logger.info("SHAP summary plot → %s", summary_path)

    meta_path = _RESULTS / "meta_test.parquet"
    if not meta_path.exists():
        logger.warning(
            "meta_test.parquet not found — re-run 'make train' to enable per-segment SHAP."
        )
        return

    meta_test = pd.read_parquet(meta_path).reset_index(drop=True)
    meta_eval = meta_test.iloc[eval_idx].reset_index(drop=True)

    # Per-bearing SHAP (IMS has bearing_id, not fault class)
    if "_meta_bearing_id" in meta_eval.columns:
        for bid in sorted(meta_eval["_meta_bearing_id"].unique()):
            mask = (meta_eval["_meta_bearing_id"] == bid).values
            if mask.sum() < 5:
                logger.warning(
                    "Skipping per-bearing SHAP for bearing %d — only %d rows.", bid, mask.sum()
                )
                continue
            exp_bid = _shap.Explanation(
                values=exp.values[mask],
                base_values=exp.base_values[mask] if exp.base_values is not None else None,
                data=exp.data[mask],
                feature_names=feature_names,
            )
            fault_path = _RESULTS / "figures" / f"shap_bearing_{bid}.png"
            save_summary_plot(exp_bid, fault_path)
            logger.info("Per-bearing SHAP (bearing %d) → %s", bid, fault_path)
    elif "_meta_class" in meta_eval.columns:
        for cls in sorted(meta_eval["_meta_class"].unique()):
            if cls == "normal":
                continue
            mask = (meta_eval["_meta_class"] == cls).values
            if mask.sum() < 5:
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

    reference = df.loc[df["_meta_y"] == 0, feature_cols].values.astype(np.float64)
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
        description="industrial-anomaly-detection pipeline (IMS/NASA bearing dataset)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download IMS/NASA dataset via Kaggle CLI → data/raw/")
    sub.add_parser("features", help="Extract features → data/features/features.parquet")
    sub.add_parser("train", help="Fit IsolationForest on healthy snapshots")
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
