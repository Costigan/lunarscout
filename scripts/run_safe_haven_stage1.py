#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage1.py [--max-slope-deg 15 --overwrite ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; `--help` lists them. run_stage1(...) takes the same values as keyword arguments, so another
script can call it (scripts/run_safe_haven.py runs the whole safe-haven sequence that way).

Stage 1 of the safe-haven analysis. Writes the three safe-haven screening products (hibernation, joint operating
total, joint operating longest run) into OUTPUT_DIRECTORY from one pass.

With LIGHT_STORE_PATH set (the default) the products are reduced from the stage-0
light store made by scripts/run_light_store.py: no horizon is read and no
Sun position is needed, so the run is much faster. The samples must then be the
store's 2-hour grid and the thresholds below the store's. Set LIGHT_STORE_PATH to
None to light from the horizons instead (HORIZONS_PATH must then exist for the DEM
and observer height).

Earth positions come from SPICE and the configured default kernels (the Sun's too
when lighting from the horizons). Explicit backend="cuda" requires a compatible
NVIDIA device and driver; backend="auto" may use CPU instead. Importing this script
starts no calculation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import lunarscout as ls  # noqa: E402
import numpy as np  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, defaults_from, options_from, run_with_logging,
)

LOG = logging.getLogger("lunarscout.scripts.stage1")


# Edit these paths before running. These match the neighboring mosaic scripts.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
HORIZONS_PATH = Path("/e/lunar_analyst_scenarios/70south/lighting/horizons")   # used only without a light store
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")      # stage 0; None lights from the horizons
OUTPUT_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
# Output files are <NAME_PREFIX>_hibernation_hours.tif and so on. Every stage names its outputs
# safe_haven_stage_<n>_<description>. This prefix differs from the earlier runs (prefixes "safe_haven"
# and "safe_haven_degenerate": 6-hour samples) so the products never overwrite each other. Set the same
# value as STAGE1_PREFIX in the stage-2 script.
NAME_PREFIX = "safe_haven_stage_1"

# UTC timestamps; STOP_UTC is inclusive when it falls on the sampling grid.
# A stop at midnight on the first day of another month also creates a band
# for that partially sampled month.
# With a light store these must be on the store's grid and inside its range (2-hour steps from
# 2024-01-01 to 2032-01-01 exclusive, so the last sample is 22:00 on 2031-12-31).
START_UTC = "2024-01-01T00:00:00Z"
STOP_UTC = "2031-12-31T22:00:00Z"
STEP_HOURS = 2.0

# Outage / low-Sun conditions use strict less-than comparisons.
EARTH_ELEVATION_THRESHOLD_DEG = 2.0
SUNLIGHT_FRACTION_THRESHOLD = 0.2
OPERATING_SUNLIGHT_FRACTION = 1.0  # Sun fraction at or above which the rover operates.
# The two windows below are in hours and become whole samples at STEP_HOURS (12 at 2 hours).
MERGE_GAP_HOURS = 24.0  # Shadow runs closer than this are one run (0 disables).
# An Earth cycle in which Earth never drops below the threshold has no outage; it gets a
# one-sample outage at Earth's lowest elevation if Earth stays above the threshold for this
# many hours either side (0 disables). Pass 1 of the window analysis must use the same value.
DEGENERATE_OUTAGE_WINDOW_HOURS = 24.0
MAX_SLOPE_DEG = 15.0  # Steeper pixels are NODATA in all three products (None disables). The rover's limit in stages 3 and 4.
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


def hours_to_two_hour_bytes(hours: np.ndarray, nodata: int = NODATA) -> np.ndarray:
    """Round durations up to two-hour counts, saturate, and encode NaNs as ``nodata``."""
    counts = np.clip(np.ceil(hours / 2.0), 0, 255)
    return np.where(np.isnan(counts), nodata, counts).astype(np.uint8)


OUTPUT_DTYPE = np.uint8  # GDAL GDT_Byte.
OUTPUT_TRANSFORM_ID = "safe-haven-2h-ceil-clip-uint8-v2"

COMPRESS = True
OVERWRITE = False  # Permit replacement of a completed product.
START_FRESH = False  # Discard staging; otherwise resume compatible staged work.
VERBOSE = True

# Console and log file. LOG_PATH None writes <OUTPUT_DIRECTORY>/<NAME_PREFIX>.log. COLOR is "auto", "always" or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF"),
    Option("HORIZONS_PATH", "path", "horizon tiles (used only without a light store)"),
    Option("LIGHT_STORE_PATH", "optional_path", "the stage-0 light store; none lights from the horizons"),
    Option("OUTPUT_DIRECTORY", "path", "where the three products are written"),
    Option("NAME_PREFIX", "str", "output file prefix (safe_haven_stage_1)"),
    Option("START_UTC", "str", "first sample (UTC)"),
    Option("STOP_UTC", "str", "last sample (UTC), inclusive when it is on the grid"),
    Option("STEP_HOURS", "float", "sample step in hours"),
    Option("EARTH_ELEVATION_THRESHOLD_DEG", "float", "Earth below this elevation is an outage"),
    Option("SUNLIGHT_FRACTION_THRESHOLD", "float", "a Sun fraction below this is shadow"),
    Option("OPERATING_SUNLIGHT_FRACTION", "float", "Sun fraction at or above which the rover operates"),
    Option("MERGE_GAP_HOURS", "float", "shadow runs closer than this are one run (0 disables)"),
    Option("DEGENERATE_OUTAGE_WINDOW_HOURS", "float", "the one-sample outage rule where Earth never sets (0 disables)"),
    Option("MAX_SLOPE_DEG", "optional_float", "steeper pixels are NODATA in all three products (none disables)"),
    Option("OBSERVER_HEIGHT_M", "float", "observer height (m); must match the horizon tiles"),
    Option("BACKEND", "str", "auto, cpu or cuda", choices=("auto", "cpu", "cuda")),
    Option("NODATA", "int", "byte value of undefined pixels (0 to 255)"),
    Option("COMPRESS", "bool", "compress the GeoTIFFs"),
    Option("OVERWRITE", "bool", "permit replacement of a completed product"),
    Option("START_FRESH", "bool", "discard staging instead of resuming staged work"),
    Option("VERBOSE", "bool", "print progress"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now)."""
    return defaults_from(OPTIONS, globals())


def run_stage1(*, dem_path, horizons_path, light_store_path, output_directory, name_prefix, start_utc, stop_utc,
               step_hours, earth_elevation_threshold_deg, sunlight_fraction_threshold, operating_sunlight_fraction,
               merge_gap_hours, degenerate_outage_window_hours, max_slope_deg, observer_height_m, backend, nodata,
               compress, overwrite, start_fresh, verbose) -> int:
    """Write the three stage-1 products; returns 0."""
    if not np.isfinite(nodata) or not 0 <= nodata <= 255 or int(nodata) != nodata:
        raise ValueError("Set NODATA to an integer in [0, 255] for byte output.")

    def to_bytes(hours: np.ndarray) -> np.ndarray:
        return hours_to_two_hour_bytes(hours, nodata)

    times = ls.times(start_utc, stop_utc, step_hours=step_hours)
    LOG.info("Generating safe havens from %d UTC samples...", times.time_count)
    result = ls.generate_safe_haven_products(
        dem_path,
        None if light_store_path is not None else horizons_path,
        output_directory,
        light_store_path=light_store_path,
        name_prefix=name_prefix,
        times=times,
        sun_vectors_m=SUN_VECTORS_M,
        earth_vectors_m=EARTH_VECTORS_M,
        earth_elevation_threshold_deg=earth_elevation_threshold_deg,
        sunlight_fraction_threshold=sunlight_fraction_threshold,
        operating_sunlight_fraction=operating_sunlight_fraction,
        merge_gap_hours=merge_gap_hours,
        degenerate_outage_window_hours=degenerate_outage_window_hours,
        max_slope_deg=max_slope_deg,
        backend=backend,
        observer_height_m=observer_height_m,
        nodata=nodata,
        output_transform=to_bytes,
        output_dtype=OUTPUT_DTYPE,
        output_transform_id=OUTPUT_TRANSFORM_ID,
        compress=compress,
        overwrite=overwrite,
        start_fresh=start_fresh,
        verbose=verbose,
    )
    for path in result:
        LOG.info("Safe-haven product written to: %s", path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: the three safe-haven screening products. Every option's default is the constant of the "
                    "same name at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the products."""
    args = build_parser().parse_args(list(argv))
    options = options_from(args, OPTIONS)
    default_log = Path(options["output_directory"]) / f"{options['name_prefix']}.log"
    return run_with_logging(args, default_log_path=default_log, logger=LOG, label="Stage 1",
                            procedure=lambda: run_stage1(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
