"""IsolationForest anomaly detector with RobustScaler preprocessing."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class IForestDetector:
    """Wraps sklearn IsolationForest with a RobustScaler fit on healthy data."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._scaler = RobustScaler()
        self._clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_healthy: np.ndarray) -> IForestDetector:
        X_scaled = self._scaler.fit_transform(X_healthy)
        self._clf.fit(X_scaled)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores; higher value = more anomalous."""
        X_scaled = self._scaler.transform(X)
        return -self._clf.score_samples(X_scaled)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "clf": self._clf}, path)

    @classmethod
    def load(cls, path: Path) -> IForestDetector:
        data = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._scaler = data["scaler"]
        obj._clf = data["clf"]
        obj.n_estimators = obj._clf.n_estimators
        obj.contamination = obj._clf.contamination
        obj.random_state = obj._clf.random_state
        obj.n_jobs = obj._clf.n_jobs
        return obj
