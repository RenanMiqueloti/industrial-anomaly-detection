"""Build feature matrix from raw CWRU signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import extract_all
from src.ingest import load_cwru, window


def build_feature_matrix(
    raw_dir: Path | str,
    out_path: Path | str,
    window_len: int = 2048,
    hop: int = 2048,
    fs: int = 12_000,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Window every raw signal and extract features.

    Loads all .mat files from *raw_dir*, filters out ``class='unknown'``
    rows, extracts features from each window, saves a parquet file to
    *out_path*, and returns ``(X, y, meta)``.

    Parameters
    ----------
    raw_dir:  directory containing (possibly nested) .mat files.
    out_path: destination parquet path (parent dirs created as needed).
    window_len, hop: windowing parameters (samples).
    fs: sampling rate in Hz passed to :func:`~src.features.extract_all`.

    Returns
    -------
    X:    ``(n_windows, n_features)`` float64 array.
    y:    ``(n_windows,)`` int array — 0 = normal, 1 = faulty.
    meta: DataFrame with columns ``filename``, ``class``, ``window_idx``.
    """
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = load_cwru(raw_dir)
    df_raw = df_raw[df_raw["class"] != "unknown"].reset_index(drop=True)

    feature_rows: list[dict] = []
    meta_rows: list[dict] = []

    for _, row in df_raw.iterrows():
        for win_idx, win in enumerate(window(row["signal"], length=window_len, hop=hop)):
            feats = extract_all(win, fs=fs)
            feature_rows.append(feats)
            meta_rows.append(
                {"filename": row["filename"], "class": row["class"], "window_idx": win_idx}
            )

    meta = pd.DataFrame(meta_rows)
    feat_df = pd.DataFrame(feature_rows)

    X = feat_df.values.astype(np.float64)
    y = (meta["class"] != "normal").astype(int).values

    # Embed metadata columns (prefixed with _meta_ to avoid PyArrow reserved names).
    out_df = feat_df.copy()
    out_df["_meta_filename"] = meta["filename"].values
    out_df["_meta_class"] = meta["class"].values
    out_df["_meta_window_idx"] = meta["window_idx"].values
    out_df["_meta_y"] = y
    out_df.to_parquet(out_path, index=False)

    return X, y, meta
