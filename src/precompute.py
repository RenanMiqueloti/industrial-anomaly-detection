"""Pre-compute full-dataset anomaly scores for the dashboard.

The dashboard scores the entire feature parquet for every model the user
toggles between. Doing that on the Streamlit worker turned into the second
biggest contributor to cold-start latency after package imports — running
``OC-SVM.score`` over ~4k rows on a shared CPU is slow enough to be felt.

This module scores the full feature parquet with every available model
and writes a single parquet that the dashboard just reads back. It's
intended to run as the last step of the offline pipeline, after
``make compare`` has produced all three model files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maps dashboard model labels → (filename, output column).
_MODELS: dict[str, tuple[str, str]] = {
    "IsolationForest": ("iforest_model.joblib", "iforest"),
    "OC-SVM": ("ocsvm_model.joblib", "ocsvm"),
    "AutoEncoder": ("ae_model.joblib", "ae"),
}

_DEFAULT_FEATURES = Path("data/features/features.parquet")
_DEFAULT_RESULTS = Path("results")
_OUTPUT_NAME = "full_dataset_scores.parquet"


def _load_detector(model_name: str, path: Path):
    # Imports are local so callers that only need IsolationForest don't pay
    # torch's import cost.
    if model_name == "IsolationForest":
        from src.models.iforest import IForestDetector

        return IForestDetector.load(path)
    if model_name == "OC-SVM":
        from src.models.ocsvm import OCSVMDetector

        return OCSVMDetector.load(path)
    if model_name == "AutoEncoder":
        from src.models.autoencoder import AutoEncoderDetector

        return AutoEncoderDetector.load(path)
    raise KeyError(model_name)


def precompute_scores(
    features_path: Path = _DEFAULT_FEATURES,
    results_dir: Path = _DEFAULT_RESULTS,
) -> Path:
    """Score the entire feature parquet with every model present in ``results_dir``.

    Skips models whose ``.joblib`` file is missing — the dashboard's fallback
    will recompute on the fly for those.

    Returns the path of the written parquet.
    """
    features_path = Path(features_path)
    results_dir = Path(results_dir)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature parquet not found at {features_path}. Run 'make features' first."
        )

    df = pd.read_parquet(features_path)
    feat_cols = [c for c in df.columns if not c.startswith("_meta_")]
    X = df[feat_cols].values

    out = pd.DataFrame()
    if "_meta_y" in df.columns:
        out["_meta_y"] = df["_meta_y"].values
    if "_meta_bearing_id" in df.columns:
        out["_meta_bearing_id"] = df["_meta_bearing_id"].astype(int).values

    scored_any = False
    for label, (filename, column) in _MODELS.items():
        model_path = results_dir / filename
        if not model_path.exists():
            logger.warning("Skipping %s — %s not found.", label, model_path)
            continue
        logger.info("Scoring %d rows with %s …", len(X), label)
        detector = _load_detector(label, model_path)
        out[column] = detector.score(X).astype(np.float64)
        scored_any = True

    if not scored_any:
        raise FileNotFoundError(
            f"No trained models found in {results_dir}. "
            "Run 'make train' (for IsolationForest) or 'make compare' (for all 3) first."
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / _OUTPUT_NAME
    out.to_parquet(out_path, index=False)
    logger.info(
        "Pre-computed scores → %s  (%d rows, %d columns)", out_path, len(out), len(out.columns)
    )
    return out_path
