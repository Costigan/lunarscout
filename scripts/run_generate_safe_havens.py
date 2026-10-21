#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_generate_safe_havens.py

Writes the three safe-haven screening products (hibernation, joint operating
total, joint operating longest run) into OUTPUT_DIRECTORY from one pass.

The default vector settings use SPICE Sun/Earth positions and the configured
default kernels. Horizons must already exist for the DEM and observer height.
Explicit backend="cuda" requires a compatible NVIDIA device and driver;
backend="auto" may use CPU instead. Importing this script starts no calculation.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import lunarscout as ls  # noqa: E402
import numpy as np  # noqa: E402


# Edit these paths before running. These match the neighboring mosaic scripts.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
HORIZONS_PATH = Path("/e/lunar_analyst_scenarios/70south/lighting/horizons")
OUTPUT_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
# Output files are <NAME_PREFIX>_hibernation_hours.tif and so on. This prefix differs from the
# earlier run (without the degenerate-outage rule, prefix "safe_haven") so the two never
# overwrite each other. Set the same value as STAGE1_PREFIX in the stage-2 scripts.
NAME_PREFIX = "safe_haven_degenerate"

# UTC timestamps; STOP_UTC is inclusive when it falls on the sampling grid.
# A stop at midnight on the first day of another month also creates a band
# for that partially sampled month.
START_UTC = "2024-01-01T00:00:00Z"
STOP_UTC = "2031-12-31T18:00:00Z"
STEP_HOURS = 6.0

# Outage / low-Sun conditions use strict less-than comparisons.
EARTH_ELEVATION_THRESHOLD_DEG = 2.0
SUNLIGHT_FRACTION_THRESHOLD = 0.2
OPERATING_SUNLIGHT_FRACTION = 1.0  # Sun fraction at or above which the rover operates.
MERGE_GAP_HOURS = 24.0  # Shadow runs closer than this are one run (0 disables).
# An Earth cycle in which Earth never drops below the threshold has no outage; it gets a
# one-sample outage at Earth's lowest elevation if Earth stays above the threshold for this
# many hours either side (0 disables). Pass 1 of the window analysis must use the same value.
DEGENERATE_OUTAGE_WINDOW_HOURS = 24.0
MAX_SLOPE_DEG = 20.0  # Steeper pixels are NODATA in all three products (None disables).
OBSERVER_HEIGHT_M = 0.0  # Must match the precomputed horizon tiles.
BACKEND = "auto"  # "auto", "cpu", or "cuda"; "cpu" never probes CUDA.

# Optional finite Moon-ME position arrays in metres, shape (time_count, 3).
# None generates that body's positions with SPICE. Supply both arrays to
# avoid SPICE import and kernel loading for this calculation.
SUN_VECTORS_M = None
EARTH_VECTORS_M = None

# Store durations as unsigned bytes, each count representing two hours:
# byte = clamp(ceil(float32_hours / 2), 0, 255). Thus 255 saturates at 510 hours.
# This applies to all three products. Undefined monthly results are NaN before
# conversion. NODATA must be an integer
# in [0, 255] for byte output; its value also fills missing-horizon patches.
NODATA = 255


def hours_to_two_hour_bytes(hours: np.ndarray) -> np.ndarray:
    """Round durations up to two-hour counts, saturate, and encode NaNs."""
    counts = np.clip(np.ceil(hours / 2.0), 0, 255)
    return np.where(np.isnan(counts), NODATA, counts).astype(np.uint8)


OUTPUT_TRANSFORM = hours_to_two_hour_bytes
OUTPUT_DTYPE = np.uint8  # GDAL GDT_Byte.
OUTPUT_TRANSFORM_ID = "safe-haven-2h-ceil-clip-uint8-v2"

COMPRESS = True
OVERWRITE = False  # Permit replacement of a completed product.
START_FRESH = False  # Discard staging; otherwise resume compatible staged work.
VERBOSE = True


def main() -> int:
    if not np.isfinite(NODATA) or not 0 <= NODATA <= 255 or int(NODATA) != NODATA:
        raise ValueError("Set NODATA to an integer in [0, 255] for byte output.")
    times = ls.times(START_UTC, STOP_UTC, step_hours=STEP_HOURS)
    print(f"Generating safe havens from {times.time_count} UTC samples...", flush=True)
    result = ls.generate_safe_haven_products(
        DEM_PATH,
        HORIZONS_PATH,
        OUTPUT_DIRECTORY,
        name_prefix=NAME_PREFIX,
        times=times,
        sun_vectors_m=SUN_VECTORS_M,
        earth_vectors_m=EARTH_VECTORS_M,
        earth_elevation_threshold_deg=EARTH_ELEVATION_THRESHOLD_DEG,
        sunlight_fraction_threshold=SUNLIGHT_FRACTION_THRESHOLD,
        operating_sunlight_fraction=OPERATING_SUNLIGHT_FRACTION,
        merge_gap_hours=MERGE_GAP_HOURS,
        degenerate_outage_window_hours=DEGENERATE_OUTAGE_WINDOW_HOURS,
        max_slope_deg=MAX_SLOPE_DEG,
        backend=BACKEND,
        observer_height_m=OBSERVER_HEIGHT_M,
        nodata=NODATA,
        output_transform=OUTPUT_TRANSFORM,
        output_dtype=OUTPUT_DTYPE,
        output_transform_id=OUTPUT_TRANSFORM_ID,
        compress=COMPRESS,
        overwrite=OVERWRITE,
        start_fresh=START_FRESH,
        verbose=VERBOSE,
    )
    for path in result:
        print(f"Safe-haven product written to: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
