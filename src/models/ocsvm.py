"""One-Class SVM anomaly detector."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from src.models.base import BaseDetector


class OCSVMDetector(BaseDetector):
    """sklearn OneClassSVM (RBF) wrapped behind BaseDetector."""

    def __init__(self, nu: float = 0.1, gamma: str | float = "scale") -> None:
        self.nu = nu
        self.gamma = gamma
        self._scaler = RobustScaler()
        self._clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

    def fit(self, X_healthy: np.ndarray) -> OCSVMDetector:
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
            {"scaler": self._scaler, "clf": self._clf, "nu": self.nu, "gamma": self.gamma}, path
        )

    @classmethod
    def load(cls, path: Path) -> OCSVMDetector:
        data = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._scaler = data["scaler"]
        obj._clf = data["clf"]
        obj.nu = data["nu"]
        obj.gamma = data["gamma"]
        return obj
