"""Raw data ingestion for MFPT bearing dataset.

Loads the 23 .mat files from the Machinery Failure Prevention Technology Society
(MFPT) bearing fault dataset. Each file contains a MATLAB struct ``bearing`` with
fields ``gs`` (accelerometer signal, g-units), ``sr`` (sampling rate, Hz),
``load`` (load weight, lbs), and ``rate`` (shaft speed).

Reference
---------
Bechhoefer, E. (2013). *Condition Based Maintenance Fault Database for Testing
Diagnostics and Prognostic Algorithms*. Machinery Failure Prevention Technology
Society. https://www.mfpt.org/fault-data-sets/

Dataset layout (23 files total)
---------------------------------
baseline_1-3.mat      : healthy bearing, 97 656 Hz, 270 lbs constant load
outer_race_1-10.mat   : outer-race fault, 97 656 Hz (1-3) / 48 828 Hz (4-10)
inner_race_1-7.mat    : inner-race fault, 48 828 Hz, variable load 0-300 lbs
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

logger = logging.getLogger(__name__)


def _infer_mfpt_class(stem: str) -> str:
    """Infer fault class from the MFPT filename stem (case-insensitive).

    ``baseline_*`` → ``"normal"``
    ``outer_race_*`` → ``"OR"``
    ``inner_race_*`` → ``"IR"``
    anything else   → ``"unknown"``
    """
    lower = stem.lower()
    if lower.startswith("baseline"):
        return "normal"
    if lower.startswith("outer"):
        return "OR"
    if lower.startswith("inner"):
        return "IR"
    return "unknown"


def load_mfpt(root: Path) -> pd.DataFrame:
    """Load all MFPT bearing .mat files from *root* (recursive).

    Reads the ``bearing`` MATLAB struct from each ``.mat`` file and extracts:
    - ``gs``:   1-D accelerometer signal (float64, g-units)
    - ``sr``:   sampling rate in Hz (97 656 for baseline; 48 828 for fault files)

    Returns
    -------
    pd.DataFrame with columns: ``filename``, ``signal`` (np.ndarray),
    ``class`` (str), ``sr`` (float).

    Raises
    ------
    FileNotFoundError if no .mat files are found under *root*.
    """
    mat_files = sorted(root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {root}")

    rows = []
    for path in mat_files:
        try:
            mat = scipy.io.loadmat(str(path))
            bearing = mat["bearing"][0, 0]
            signal = np.array(bearing["gs"]).squeeze().astype(np.float64)
            sr = float(np.array(bearing["sr"]).squeeze())
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue

        cls = _infer_mfpt_class(path.stem)
        if cls == "unknown":
            logger.warning("Could not infer class for %s", path.name)

        rows.append({"filename": path.name, "signal": signal, "class": cls, "sr": sr})

    return pd.DataFrame(rows, columns=["filename", "signal", "class", "sr"])


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
