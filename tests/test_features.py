"""Physical-truth tests for the feature extraction module.

Each assertion encodes a property that follows from signal-processing theory,
not from a re-implementation of the function under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.features import (
    DEFAULT_BANDS,
    extract_all,
    fft_band_energy,
    time_domain_features,
)


def _sine(amp: float, freq: float, fs: int, duration: float) -> np.ndarray:
    t = np.arange(int(fs * duration)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def test_rms_of_sine_is_amplitude_over_sqrt2() -> None:
    """RMS of a pure sine of amplitude A is A / sqrt(2) (textbook identity)."""
    fs = 12_000
    x = _sine(amp=2.0, freq=60.0, fs=fs, duration=1.0)
    feats = time_domain_features(x)
    assert math.isclose(feats.rms, 2.0 / math.sqrt(2), rel_tol=1e-3)


def test_crest_factor_of_sine_is_sqrt2() -> None:
    """Crest factor (peak / RMS) of a sine is exactly sqrt(2)."""
    x = _sine(amp=1.5, freq=120.0, fs=12_000, duration=1.0)
    feats = time_domain_features(x)
    assert math.isclose(feats.crest_factor, math.sqrt(2), rel_tol=1e-3)


def test_kurtosis_of_gaussian_noise_is_close_to_zero() -> None:
    """Excess kurtosis (Fisher) of N(0, 1) → 0 for large samples."""
    rng = np.random.default_rng(seed=42)
    x = rng.standard_normal(50_000)
    feats = time_domain_features(x)
    # Tolerance generous because finite sample kurtosis fluctuates.
    assert abs(feats.kurtosis) < 0.1


def test_extract_all_returns_expected_keys() -> None:
    """Time-domain (7) + band-energy (len(bands)) = 11 keys with defaults."""
    x = _sine(amp=1.0, freq=60.0, fs=12_000, duration=1.0)
    feats = extract_all(x)
    time_keys = {"rms", "peak", "crest_factor", "kurtosis", "skewness", "std", "p2p"}
    assert time_keys.issubset(feats.keys())
    band_keys = {f"band_{int(lo)}_{int(hi)}" for lo, hi in DEFAULT_BANDS}
    assert band_keys.issubset(feats.keys())
    assert len(feats) == len(time_keys) + len(band_keys)


def test_band_energy_concentrates_in_correct_band() -> None:
    """A 1 kHz sine puts almost all its energy in the 500–2000 Hz band (IMS bands)."""
    fs = 20_000
    x = _sine(amp=1.0, freq=1000.0, fs=fs, duration=1.0)
    energies = fft_band_energy(x, fs=fs, bands=DEFAULT_BANDS)
    target = energies["band_500_2000"]
    others = sum(v for k, v in energies.items() if k != "band_500_2000")
    assert target > 10 * max(others, 1e-12)


def test_features_reject_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        time_domain_features(np.zeros((10, 10)))


def test_features_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        time_domain_features(np.array([]))


def test_extract_all_accepts_custom_bands() -> None:
    """Custom bands override the default and all are present in the output."""
    x = _sine(amp=1.0, freq=60.0, fs=12_000, duration=1.0)
    custom = ((0.0, 100.0), (100.0, 500.0))
    feats = extract_all(x, bands=custom)
    assert "band_0_100" in feats
    assert "band_100_500" in feats
    # Default bands should NOT be present when an override is provided.
    assert "band_500_2000" not in feats
