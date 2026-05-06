"""Population Stability Index (PSI) drift detection over feature distributions."""

from __future__ import annotations

import numpy as np

PSI_ALERT_THRESHOLD: float = 0.2


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Single-column PSI between reference and current samples.

    Bins reference into n_bins quantiles, then computes
        PSI = sum( (p_curr - p_ref) * ln(p_curr / p_ref) )
    over those bins. Adds 1e-6 to empty bins to avoid log(0).
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    # Ensure unique edges (degenerate distributions can produce duplicates).
    bin_edges = np.unique(bin_edges)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    eps = 1e-6
    p_ref = ref_counts / (ref_counts.sum() + eps) + eps
    p_cur = cur_counts / (cur_counts.sum() + eps) + eps

    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def compute_psi_per_feature(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list[str],
    n_bins: int = 10,
) -> dict[str, float]:
    """PSI per column. Returns {feature_name: psi}."""
    if reference.ndim != 2 or current.ndim != 2:
        raise ValueError("reference and current must be 2-D arrays (n_samples, n_features)")
    if reference.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match number of columns")
    return {
        name: compute_psi(reference[:, i], current[:, i], n_bins=n_bins)
        for i, name in enumerate(feature_names)
    }


def flag_drift(
    psi_dict: dict[str, float],
    threshold: float = PSI_ALERT_THRESHOLD,
) -> list[str]:
    """Returns the list of feature names whose PSI exceeds threshold."""
    return [name for name, psi in psi_dict.items() if psi > threshold]
