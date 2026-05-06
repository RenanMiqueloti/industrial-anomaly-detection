"""Abstract base class for unsupervised anomaly detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseDetector(ABC):
    """Common interface for all anomaly detectors.

    Convention: ``score(X)`` returns a 1-D array where **higher = more anomalous**.
    ``fit()`` receives only healthy (normal) windows.
    """

    @abstractmethod
    def fit(self, X_healthy: np.ndarray) -> BaseDetector: ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> BaseDetector: ...
