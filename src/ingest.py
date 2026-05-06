"""Raw data ingestion for CWRU bearing dataset.

Loads .mat files, discovers the drive-end (or front-end) time-series key,
infers the fault class from the filename/subfolder, and yields fixed-length
windows for downstream feature extraction.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

logger = logging.getLogger(__name__)

_KEY_PATTERNS = [
    re.compile(r"^X\d+_DE_time$"),
    re.compile(r"^X\d+_FE_time$"),
]

# CWRU numeric file-ID → fault class mapping
_FILE_CLASS_RANGES: list[tuple[range, str]] = [
    (range(97, 101), "normal"),
    (range(105, 119), "IR"),
    (range(119, 133), "B"),
    (range(133, 201), "OR"),
]


def _infer_class(stem: str) -> str:
    """Infer fault class from filename stem (case-insensitive keyword + numeric ID)."""
    lower = stem.lower()
    if "normal" in lower:
        return "normal"
    if "ir" in lower or "inner" in lower:
        return "IR"
    if re.search(r"\bb\b", lower) or lower.startswith("b_") or "_b_" in lower:
        return "B"
    if "or" in lower or "outer" in lower:
        return "OR"

    m = re.search(r"(\d+)", stem)
    if m:
        file_id = int(m.group(1))
        for id_range, cls in _FILE_CLASS_RANGES:
            if file_id in id_range:
                return cls

    return "unknown"


def load_cwru(root: Path) -> pd.DataFrame:
    """Load all .mat bearing files from *root* (recursive).

    For each file, discovers the drive-end or front-end time-series key
    (pattern ``X<N>_DE_time`` with ``X<N>_FE_time`` as fallback), infers
    the fault class, and stores the raw 1-D signal.

    Returns
    -------
    pd.DataFrame with columns: ``filename``, ``signal`` (np.ndarray), ``class``.

    Raises
    ------
    FileNotFoundError if no .mat files are found under *root*.
    """
    mat_files = sorted(root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {root}")

    rows = []
    for path in mat_files:
        mat = scipy.io.loadmat(str(path))
        signal_key: str | None = None
        for pattern in _KEY_PATTERNS:
            for k in mat:
                if pattern.match(k):
                    signal_key = k
                    break
            if signal_key:
                break

        if signal_key is None:
            logger.warning(
                "No matching key in %s — keys: %s",
                path.name,
                [k for k in mat if not k.startswith("_")],
            )
            continue

        signal = mat[signal_key].squeeze().astype(np.float64)
        cls = _infer_class(path.stem)
        if cls == "unknown":
            logger.warning("Could not infer class for %s", path.name)

        rows.append({"filename": path.name, "signal": signal, "class": cls})

    return pd.DataFrame(rows, columns=["filename", "signal", "class"])


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
