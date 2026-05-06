"""Raw data ingestion for IMS/NASA bearing dataset.

Loads snapshot files from an IMS (Intelligent Maintenance Systems) run directory.
Each file is named by timestamp (YYYY.MM.DD.HH.MM.SS), contains 20 480 rows at
20 kHz (1 second of data) and N columns (one per bearing channel).

Reference
---------
Lee, J., Qiu, H., Yu, G., & Lin, J. (2007). Bearing Data Set.
IMS, University of Cincinnati. NASA Prognostics Data Repository.
https://data.nasa.gov/dataset/IMS-University-of-Cincinnati-Bearing-Dataset/3yud-nd96

Dataset layout
--------------
1st_test/ — Oct–Nov 2003, ~34 days, 8 columns (4 bearings × 2 sensors)
2nd_test/ — Feb 12–19 2004, ~7 days, 4 columns; Bearing 1 outer-race failure ← default
3rd_test/ — Mar–Apr 2004, ~31 days, 4 columns; Bearing 3 OR + Bearing 4 ball failure

Each snapshot file: 20 480 rows × N columns, tab-separated, no header.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")

IMS_FS: int = 20_000       # sampling rate in Hz (all runs)
IMS_ROWS: int = 20_480     # samples per snapshot (1 s at 20 kHz)


def _parse_timestamp(stem: str) -> datetime | None:
    """Parse an IMS filename stem into a datetime. Returns None if not a valid timestamp."""
    if not _TIMESTAMP_RE.match(stem):
        return None
    try:
        return datetime.strptime(stem, "%Y.%m.%d.%H.%M.%S")
    except ValueError:
        return None


def load_ims_run(
    run_dir: Path,
    bearing_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Load all snapshots from an IMS run directory.

    Parameters
    ----------
    run_dir:     directory containing timestamped snapshot files.
    bearing_ids: which bearings (1-indexed) to include. ``None`` loads all columns.

    Returns
    -------
    pd.DataFrame with columns:

    - ``timestamp`` : datetime (parsed from filename)
    - ``bearing_id``: int (1-indexed column position)
    - ``signal``    : np.ndarray of shape (N_rows,) float64
    - ``filename``  : str (original filename)

    For Run 1 (8-column), each bearing occupies 2 adjacent columns; this loader
    treats each column independently — bearing_id maps directly to column index + 1.

    Raises
    ------
    FileNotFoundError if no valid snapshot files are found under *run_dir*.
    """
    run_dir = Path(run_dir)
    snapshot_files = sorted(
        f for f in run_dir.iterdir()
        if f.is_file() and _parse_timestamp(f.name) is not None
    )
    if not snapshot_files:
        raise FileNotFoundError(
            f"No IMS snapshot files found in {run_dir}. "
            "Files must be named YYYY.MM.DD.HH.MM.SS (no extension or any extension)."
        )

    rows: list[dict] = []
    for path in snapshot_files:
        ts = _parse_timestamp(path.name)
        try:
            data = np.loadtxt(str(path), delimiter="\t", dtype=np.float64)
        except ValueError:
            try:
                data = np.loadtxt(str(path), dtype=np.float64)
            except ValueError:
                logger.warning("Skipping %s: cannot parse as numeric data", path.name)
                continue

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_cols = data.shape[1]
        for col in range(n_cols):
            bid = col + 1
            if bearing_ids is not None and bid not in bearing_ids:
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "bearing_id": bid,
                    "signal": data[:, col].astype(np.float64),
                    "filename": path.name,
                }
            )

    if not rows:
        raise FileNotFoundError(
            f"No valid snapshots loaded from {run_dir} "
            f"(requested bearing_ids={bearing_ids})."
        )

    df = pd.DataFrame(rows, columns=["timestamp", "bearing_id", "signal", "filename"])
    return df.sort_values(["bearing_id", "timestamp"]).reset_index(drop=True)


def window(
    signal: np.ndarray,
    length: int = 2048,
    hop: int = 2048,
) -> Iterator[np.ndarray]:
    """Yield consecutive windows of *length* samples, stepping by *hop*.

    Incomplete trailing samples are discarded.
    """
    n = len(signal)
    start = 0
    while start + length <= n:
        yield signal[start : start + length]
        start += hop
