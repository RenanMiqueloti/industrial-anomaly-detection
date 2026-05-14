"""Build feature matrix from raw IMS/NASA bearing signals.

One feature vector per snapshot per bearing.  The entire 20 480-sample snapshot
is used as a single "window" — no sub-windowing — so each row in the output
corresponds to one ~10-minute measurement point in time.

Label strategy (matches the IMS dataset ground truth):
    For bearings listed in ``documented_failure_bearings``:
        y = 0  for the first ``healthy_frac`` of chronologically-sorted snapshots
        y = 1  for the remaining snapshots (degradation region)
    For all other bearings:
        y = 0  throughout — the paper does not document a failure on them

Run 2 (the default IMS test): only Bearing 1 fails (outer race). B2/B3/B4 stay
healthy until the end. Labelling B2/B3/B4 as y=1 in the late period — as a
purely temporal rule would do — contradicts the dataset's documented ground
truth and corrupts every downstream metric (AUC, F1, recall) on those bearings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import extract_all
from src.ingest import IMS_FS, load_ims_run

# Run 2 ground truth from the IMS documentation: Bearing 1 outer-race failure.
IMS_RUN2_FAILURE_BEARINGS: tuple[int, ...] = (1,)


def build_ims_features(
    run_dir: Path | str,
    out_path: Path | str,
    bearing_ids: list[int] | None = None,
    healthy_frac: float = 0.40,
    fs: int = IMS_FS,
    documented_failure_bearings: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract one feature vector per IMS snapshot per bearing.

    Parameters
    ----------
    run_dir:      IMS run directory (e.g. ``data/raw/2nd_test/``).
    out_path:     destination parquet path; parent dirs are created automatically.
    bearing_ids:  which bearings to include (1-indexed). ``None`` → all columns.
    healthy_frac: fraction of chronologically-first snapshots labelled y=0
                  *for bearings with a documented failure*.
    fs:           sampling rate in Hz (default 20 000 for IMS).
    documented_failure_bearings:
        Bearings whose late period is labelled y=1. Bearings outside this set
        get y=0 throughout. ``None`` defaults to ``IMS_RUN2_FAILURE_BEARINGS``
        (Bearing 1 — the only documented failure in Run 2).

    Returns
    -------
    X:    ``(n_snapshots * n_bearings, n_features)`` float64 array.
    y:    ``(n_rows,)`` int array — 0 = healthy, 1 = documented degradation.
    meta: DataFrame with snapshot metadata (timestamp, bearing_id, filename).

    Parquet columns: feature columns + ``_meta_timestamp``, ``_meta_bearing_id``,
    ``_meta_filename``, ``_meta_snapshot_idx`` (per-bearing sequential index),
    ``_meta_y``.
    """
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if documented_failure_bearings is None:
        documented_failure_bearings = IMS_RUN2_FAILURE_BEARINGS
    fail_set = {int(b) for b in documented_failure_bearings}

    df_raw = load_ims_run(run_dir, bearing_ids=bearing_ids)

    feature_rows: list[dict] = []
    meta_rows: list[dict] = []

    for bid, grp in df_raw.groupby("bearing_id"):
        grp_sorted = grp.sort_values("timestamp").reset_index(drop=True)
        n_snapshots = len(grp_sorted)
        cutoff = int(np.floor(n_snapshots * healthy_frac))
        bearing_fails = int(bid) in fail_set

        for snap_idx, row in grp_sorted.iterrows():
            signal = row["signal"]
            feats = extract_all(signal, fs=fs)
            feature_rows.append(feats)
            label = 1 if (bearing_fails and snap_idx >= cutoff) else 0
            meta_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "bearing_id": int(bid),
                    "filename": row["filename"],
                    "snapshot_idx": int(snap_idx),
                    "y": label,
                }
            )

    meta = pd.DataFrame(meta_rows)
    feat_df = pd.DataFrame(feature_rows)

    X = feat_df.values.astype(np.float64)
    y = meta["y"].values.astype(int)

    out_df = feat_df.copy()
    out_df["_meta_timestamp"] = meta["timestamp"].values
    out_df["_meta_bearing_id"] = meta["bearing_id"].values
    out_df["_meta_filename"] = meta["filename"].values
    out_df["_meta_snapshot_idx"] = meta["snapshot_idx"].values
    out_df["_meta_y"] = y
    out_df.to_parquet(out_path, index=False)

    return X, y, meta
