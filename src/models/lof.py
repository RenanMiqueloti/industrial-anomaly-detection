"""Local Outlier Factor anomaly detector."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from src.models.base import BaseDetector


class LOFDetector(BaseDetector):
    """LocalOutlierFactor with novelty=True (required for score_samples on new data)."""

    def __init__(self, n_neighbors: int = 20, contamination: str | float = "auto") -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self._scaler = RobustScaler()
        # novelty=True is mandatory — without it, .score_samples raises on unseen data.
        self._clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            contamination=contamination,
        )

    def fit(self, X_healthy: np.ndarray) -> LOFDetector:
        X_scaled = self._scaler.fit_transform(X_healthy)
        self._clf.fit(X_scaled)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        X_scaled = self._scaler.transform(X)
        return -self._clf.score_samples(X_scaled)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self._scaler,
                "clf": self._clf,
                "n_neighbors": self.n_neighbors,
                "contamination": self.contamination,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LOFDetector:
        data = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._scaler = data["scaler"]
        obj._clf = data["clf"]
        obj.n_neighbors = data["n_neighbors"]
        obj.contamination = data["contamination"]
        return obj
