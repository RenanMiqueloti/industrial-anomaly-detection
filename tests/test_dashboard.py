"""Tests for src/dashboard.py — import-only checks, no Streamlit server launch."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

# Guard: skip entire module if streamlit is unavailable.
st = pytest.importorskip("streamlit", reason="streamlit not installed")


def test_dashboard_module_imports() -> None:
    """src.dashboard can be imported and exposes the expected public callables.

    We patch st.set_page_config to avoid 'must be first Streamlit call' errors
    when running inside pytest's process.
    """
    # Patch set_page_config so the module-level call doesn't raise.
    with patch("streamlit.set_page_config"):
        if "src.dashboard" in sys.modules:
            del sys.modules["src.dashboard"]
        mod = importlib.import_module("src.dashboard")

    assert callable(getattr(mod, "load_test_data", None)), "load_test_data must be defined"
    assert callable(getattr(mod, "load_model", None)), "load_model must be defined"
    assert callable(getattr(mod, "compute_scores", None)), "compute_scores must be defined"
    assert callable(getattr(mod, "main", None)), "main must be defined"


def test_dashboard_handles_missing_artifacts(tmp_path, monkeypatch) -> None:
    """load_test_data() returns None when results/X_test.npy is absent."""
    # Redirect the path constants inside the dashboard module to tmp_path
    # so it looks for files there (guaranteed not to exist).
    with patch("streamlit.set_page_config"):
        if "src.dashboard" in sys.modules:
            del sys.modules["src.dashboard"]
        mod = importlib.import_module("src.dashboard")

    # Override the private path constants used by load_test_data.
    monkeypatch.setattr(mod, "_RESULTS", tmp_path / "results")
    monkeypatch.setattr(mod, "_DATA_FEATURES", tmp_path / "data" / "features.parquet")

    # Clear cache so the patched paths take effect.
    mod.load_test_data.clear()

    result = mod.load_test_data()
    assert result is None, (
        f"load_test_data() must return None when artifacts are absent, got: {result}"
    )
