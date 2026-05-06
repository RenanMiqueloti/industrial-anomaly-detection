"""Tests for the BaseDetector ABC."""

from __future__ import annotations

import pytest

from src.models.base import BaseDetector


def test_basedetector_is_abstract() -> None:
    """BaseDetector cannot be instantiated — all 4 abstract methods must be implemented."""
    with pytest.raises(TypeError):
        BaseDetector()  # type: ignore[abstract]


def test_basedetector_requires_all_methods() -> None:
    """A subclass missing any abstract method also cannot be instantiated."""

    class Partial(BaseDetector):
        def fit(self, X_healthy):
            return self

        def score(self, X):
            return X[:, 0]

        # save and load intentionally omitted

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_concrete_subclass_instantiates() -> None:
    """A fully implemented subclass can be instantiated and used."""
    from pathlib import Path

    import numpy as np

    class Trivial(BaseDetector):
        def fit(self, X_healthy):
            return self

        def score(self, X):
            return np.zeros(len(X))

        def save(self, path: Path) -> None:
            pass

        @classmethod
        def load(cls, path: Path) -> Trivial:
            return cls()

    t = Trivial()
    t.fit(np.zeros((10, 5)))
    scores = t.score(np.zeros((5, 5)))
    assert scores.shape == (5,)
