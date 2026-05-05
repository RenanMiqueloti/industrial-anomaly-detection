"""Feature extraction for vibration time-series.

Implements the handcrafted features that out-perform deep learning on bearing
vibration data with limited samples:

- time-domain statistics (RMS, peak, crest factor, kurtosis, skewness, std,
  peak-to-peak)
- frequency-domain band energy via Welch's PSD over configurable bands

These features encode the physics of rolling-element bearings — peak amplitude
and crest factor track impulsive defects, kurtosis distinguishes incipient
faults from healthy noise, and band energy isolates the BPFO/BPFI/BSF/FTF
characteristic frequencies into coarser buckets.

Reference signal model: zero-mean acceleration in g, sampled at fs (default
12 kHz for the CWRU bearing dataset).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as sig
from scipy import stats

DEFAULT_FS: int = 12_000  # CWRU drive-end bearing sampling rate (Hz)
DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 500.0),
    (500.0, 1500.0),
    (1500.0, 3000.0),
    (3000.0, 6000.0),
)


@dataclass(frozen=True)
class TimeDomainFeatures:
    """Container for time-domain statistics of a vibration window.

    All values are unitless floats except where noted; the input signal is
    expected to be zero-mean acceleration in g.
    """

    rms: float
    peak: float
    crest_factor: float
    kurtosis: float
    skewness: float
    std: float
    p2p: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate_signal(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"signal must be 1-D, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError("signal must be non-empty")
    return arr


def time_domain_features(x: np.ndarray) -> TimeDomainFeatures:
    """Compute time-domain summary statistics of a 1-D signal.

    Args:
        x: 1-D numpy array (vibration window in g, assumed zero-mean).

    Returns:
        ``TimeDomainFeatures`` with rms, peak, crest_factor, kurtosis,
        skewness, std and peak-to-peak.

    Raises:
        ValueError: if ``x`` is not 1-D or is empty.
    """
    arr = _validate_signal(x)
    rms = float(np.sqrt(np.mean(arr**2)))
    peak = float(np.max(np.abs(arr)))
    # Guard against silence: crest factor is ill-defined for rms ~ 0.
    crest = float(peak / rms) if rms > 0 else 0.0
    return TimeDomainFeatures(
        rms=rms,
        peak=peak,
        crest_factor=crest,
        kurtosis=float(stats.kurtosis(arr, fisher=True, bias=False)),
        skewness=float(stats.skew(arr, bias=False)),
        std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        p2p=float(np.ptp(arr)),
    )


def fft_band_energy(
    x: np.ndarray,
    fs: int = DEFAULT_FS,
    bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
) -> dict[str, float]:
    """Integrated PSD energy in each frequency band via Welch's method.

    Each band ``(low, high)`` becomes a key ``"band_<low>_<high>"`` mapping
    to the trapezoidal integral of the PSD over that range, in g²/Hz · Hz.

    Args:
        x: 1-D vibration window.
        fs: sampling rate in Hz.
        bands: iterable of (low_hz, high_hz) pairs. Bands above Nyquist are
            clipped; bands fully above Nyquist contribute zero.

    Returns:
        Dict mapping band label to integrated energy.
    """
    arr = _validate_signal(x)
    nperseg = min(1024, arr.size)
    freqs, psd = sig.welch(arr, fs=fs, nperseg=nperseg)
    nyquist = fs / 2.0

    result: dict[str, float] = {}
    for low, high in bands:
        key = f"band_{int(low)}_{int(high)}"
        hi_clipped = min(high, nyquist)
        if low >= hi_clipped:
            result[key] = 0.0
            continue
        mask = (freqs >= low) & (freqs <= hi_clipped)
        if mask.sum() < 2:
            result[key] = 0.0
            continue
        result[key] = float(np.trapezoid(psd[mask], freqs[mask]))
    return result


def extract_all(
    x: np.ndarray,
    fs: int = DEFAULT_FS,
    bands: tuple[tuple[float, float], ...] | None = None,
) -> dict[str, float]:
    """Combine time- and frequency-domain features into a single feature vector.

    Args:
        x: 1-D vibration window.
        fs: sampling rate in Hz (default 12 kHz, CWRU drive-end).
        bands: optional override for frequency bands. ``None`` uses
            :data:`DEFAULT_BANDS`.

    Returns:
        Dict with 7 time-domain keys + ``len(bands)`` band-energy keys.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    feats = time_domain_features(x).as_dict()
    feats.update(fft_band_energy(x, fs=fs, bands=bands))
    return feats
