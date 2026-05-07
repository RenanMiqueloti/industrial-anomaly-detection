"""End-to-end synthetic pipeline test.

Runs entirely in-memory: no file I/O, no real dataset download required.
Must complete in < 30 s on a modern laptop.
"""

from __future__ import annotations

import numpy as np

from src.evaluate import bootstrap_ci
from src.features import extract_all
from src.models.iforest import IForestDetector

N_WINDOWS = 50
WIN_LEN = 2048


def _make_windows(rng: np.random.Generator, n: int, anomalous: bool) -> list[np.ndarray]:
    windows = []
    for _ in range(n):
        w = rng.standard_normal(WIN_LEN)
        if anomalous:
            positions = rng.integers(0, WIN_LEN, size=5)
            w[positions] += 5.0
        windows.append(w)
    return windows


def test_end_to_end_synthetic_pipeline() -> None:
    """IForest on synthetic impulse-injected windows should achieve ROC-AUC > 0.7."""
    rng = np.random.default_rng(42)

    normal_wins = _make_windows(rng, N_WINDOWS, anomalous=False)
    anomaly_wins = _make_windows(rng, N_WINDOWS, anomalous=True)

    X_normal = np.array([list(extract_all(w).values()) for w in normal_wins])
    X_anomaly = np.array([list(extract_all(w).values()) for w in anomaly_wins])

    X_all = np.vstack([X_normal, X_anomaly])
    y_all = np.array([0] * N_WINDOWS + [1] * N_WINDOWS)

    model = IForestDetector()
    model.fit(X_normal)

    scores = model.score(X_all)
    result = bootstrap_ci(y_all, scores)

    mean_auc = result["roc_auc"][0]
    assert mean_auc > 0.7, (
        f"Synthetic pipeline ROC-AUC {mean_auc:.3f} < 0.7 — "
        "check feature extraction or IForest configuration"
    )
