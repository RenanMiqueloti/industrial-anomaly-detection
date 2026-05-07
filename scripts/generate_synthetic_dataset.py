"""Generate a tiny synthetic IMS-like bearing dataset for offline demos.

Creates a directory of 60 timestamped snapshot files mimicking IMS Run 2 layout:

    data/raw/2nd_test/2nd_test/
        2004.02.12.10.32.39
        2004.02.12.10.42.39
        ...

Each file is 20 480 rows × 4 columns (one per bearing), tab-separated, no header.
The first 70% of snapshots simulate healthy operation: zero-mean Gaussian noise.
The last 30% simulate Bearing 1 with progressive outer-race degradation:
    - rising RMS amplitude
    - rising kurtosis (impulsive content via injected impacts)
    - rising 2-5 kHz band energy (BPFO characteristic frequency)

This is **not** a substitute for the real IMS/NASA dataset — vibration physics
and exact failure progression differ. It exists so that anyone can run the
full pipeline end-to-end (`make demo features train`) without a Kaggle account
to verify the code works and inspect the dashboard with the synthetic data.

Run:
    python scripts/generate_synthetic_dataset.py
    # or
    make demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FS: int = 20_000
ROWS_PER_SNAPSHOT: int = 20_480
N_BEARINGS: int = 4
DEFAULT_OUT_DIR = Path("data/raw/2nd_test/2nd_test")
DEFAULT_N_SNAPSHOTS = 60
DEFAULT_HEALTHY_FRAC = 0.70
DEFAULT_SNAPSHOT_INTERVAL_MIN = 10  # IMS Run 2 sampled every ~10 minutes
DEFAULT_START = datetime(2004, 2, 12, 10, 32, 39)


def _healthy_signal(rng: np.random.Generator, n: int) -> np.ndarray:
    """Baseline healthy vibration: low-amplitude Gaussian noise."""
    return rng.normal(loc=0.0, scale=0.05, size=n).astype(np.float64)


def _degraded_signal(
    rng: np.random.Generator,
    n: int,
    severity: float,
) -> np.ndarray:
    """Healthy baseline + progressive outer-race fault signature.

    Parameters
    ----------
    severity : float in [0, 1]
        0 → indistinguishable from healthy. 1 → fully developed fault.

    The fault signature combines:
    - Higher broadband noise floor (RMS rises).
    - Periodic impulses at ~230 Hz (typical BPFO range for medium-RPM bearings)
      with amplitude growing in severity.
    - Resonance ringing in the 2-5 kHz band excited by each impulse.
    """
    base = rng.normal(loc=0.0, scale=0.05 + 0.05 * severity, size=n)

    bpfo_hz = 230.0
    impulse_period_samples = int(FS / bpfo_hz)
    impulse_train = np.zeros(n)
    impulse_train[::impulse_period_samples] = 1.0

    resonance_hz = 3500.0
    decay_samples = int(FS * 0.001)
    decay_kernel = np.exp(-np.arange(decay_samples) / (decay_samples * 0.3))
    resonance = np.sin(2 * np.pi * resonance_hz * np.arange(decay_samples) / FS)
    impact = decay_kernel * resonance
    fault = np.convolve(impulse_train, impact, mode="same")

    fault_amplitude = 0.4 * severity
    return (base + fault_amplitude * fault).astype(np.float64)


def _make_snapshot(
    rng: np.random.Generator,
    n_rows: int,
    n_bearings: int,
    bearing1_severity: float,
) -> np.ndarray:
    """Build one snapshot: n_rows × n_bearings array.

    Bearing 1 carries the simulated fault progression; bearings 2-4 stay healthy.
    """
    cols = []
    for bearing_idx in range(n_bearings):
        if bearing_idx == 0:
            cols.append(_degraded_signal(rng, n_rows, severity=bearing1_severity))
        else:
            cols.append(_healthy_signal(rng, n_rows))
    return np.column_stack(cols)


def generate(
    out_dir: Path = DEFAULT_OUT_DIR,
    n_snapshots: int = DEFAULT_N_SNAPSHOTS,
    healthy_frac: float = DEFAULT_HEALTHY_FRAC,
    interval_minutes: int = DEFAULT_SNAPSHOT_INTERVAL_MIN,
    start_time: datetime = DEFAULT_START,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Generate a synthetic IMS-like run and return the output directory."""
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        logger.warning(
            "Output dir %s is not empty; pass --overwrite to regenerate. Skipping.",
            out_dir,
        )
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_healthy = round(n_snapshots * healthy_frac)

    logger.info(
        "Generating %d snapshots (%d healthy + %d degraded) into %s",
        n_snapshots,
        n_healthy,
        n_snapshots - n_healthy,
        out_dir,
    )

    for i in range(n_snapshots):
        if i < n_healthy:
            severity = 0.0
        else:
            progression = (i - n_healthy + 1) / max(1, n_snapshots - n_healthy)
            severity = progression  # linear ramp 0 → 1

        timestamp = start_time + timedelta(minutes=i * interval_minutes)
        filename = timestamp.strftime("%Y.%m.%d.%H.%M.%S")

        data = _make_snapshot(rng, ROWS_PER_SNAPSHOT, N_BEARINGS, severity)
        np.savetxt(out_dir / filename, data, delimiter="\t", fmt="%.6f")

        if (i + 1) % 10 == 0 or i + 1 == n_snapshots:
            logger.info("  wrote %d/%d snapshots", i + 1, n_snapshots)

    logger.info("Done. Run 'make features train' to process the synthetic data.")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-snapshots", type=int, default=DEFAULT_N_SNAPSHOTS)
    parser.add_argument("--healthy-frac", type=float, default=DEFAULT_HEALTHY_FRAC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true", help="regenerate even if out-dir is non-empty"
    )
    args = parser.parse_args(argv)

    generate(
        out_dir=args.out_dir,
        n_snapshots=args.n_snapshots,
        healthy_frac=args.healthy_frac,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
