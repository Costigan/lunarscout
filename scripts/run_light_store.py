#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_light_store.py [--lighting cpu --patches 0,0 2,1 ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; `--help` lists them. run_light_store(...) takes the same values as keyword arguments, so
another script can call it (scripts/run_safe_haven.py runs the whole safe-haven sequence that way).

Stage 0 of the safe-haven analysis: light every 128 x 128 patch once, for the whole time range, and
store the three thresholded planes (Sun fraction at least 0.2, at least 1.0, and Earth above the outage
elevation) in one compressed file per patch under OUTPUT_DIRECTORY. Stages 1 and 3 read those files
instead of lighting from the horizons again (src/lunarscout/_numba_horizon/safe_haven_stage0.py and
safe_haven_lightstore.py). At 2-hour steps over 8 years a patch is about 215 MB raw and about 4 MB
stored (measured on eight patches with scripts/measure_light_store.py).

Lighting runs on the NVIDIA GPU when LIGHTING is "auto" and a device can be set up, or "cuda" (a
compatible NVIDIA device and driver are required and there is no fallback); with "cpu", or an "auto"
fallback, it runs on all CPU cores and is far slower (about 5 s per patch against a fraction of a second
on the GPU). The run is restartable: run it again after a kill, a crash or Ctrl-C and it skips the
patches already stored. A first Ctrl-C stops starting patches and lets the ones in flight finish; a
second stops at once. A different setting, time range or DEM needs a new OUTPUT_DIRECTORY (the
manifest refuses the old one).

Set PATCHES for a trial before running every patch. Progress is logged to the console (colored with
rich on a terminal when `pip install "lunarscout[console]"` is installed, plain when piped) and to a
plain-text log file next to the output. The store needs `pip install "lunarscout[lightstore]"` for
zstandard. Importing this file starts nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.file_format import HorizonTileStore  # noqa: E402
from lunarscout._numba_horizon.safe_haven_lightstore import LightStoreSettings  # noqa: E402
from lunarscout._numba_horizon.safe_haven_stage0 import Stage0Settings, generate_light_store  # noqa: E402
from lunarscout._numba_horizon.safe_haven_windows import spice_vector_source  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, defaults_from, options_from, run_with_logging,
)
from lunarscout.products import _load_dem  # noqa: E402

LOG = logging.getLogger("lunarscout.scripts.run_light_store")

# Edit these paths before running. They match run_safe_haven_stage2.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
HORIZONS_PATH = Path("/e/lunar_analyst_scenarios/70south/lighting/horizons")
OUTPUT_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/light_store")     # about 0.2 to 0.45 TB for every patch
OBSERVER_HEIGHT_M = 0.0

# The time grid. The step divides 24 hours; stage 1 and stage 3 both read this grid.
START_UTC = "2024-01-01T00:00:00"
STOP_UTC = "2032-01-01T00:00:00"        # exclusive
STEP_MINUTES = 120

# The thresholds are those of stage 1 (run_safe_haven_stage1.py).
SHADOW_FRACTION = 0.2                   # a sample below this Sun fraction is power-negative shadow
OPERATING_FRACTION = 1.0                # at or above this the rover operates (with the stage-1 fuzz of 1e-6)
EARTH_ELEVATION_DEG = 2.0               # Earth below this above the pixel's own horizon is an outage

# How the files are laid out: independently compressed chunks of this many steps (96 = 8 days at 2 hours).
STEPS_PER_CHUNK = 96
CODEC = "zstd-3"

# None lights every patch of the DEM that has a horizon tile; a list of (patch_row, patch_column)
# lights only those (a trial). The raster is 237 x 237 patches; the pole is near (118, 118).
PATCHES = None

# "auto" uses the GPU when one is available, "cuda" requires it, "cpu" never touches CUDA.
LIGHTING = "auto"
GPU_DEVICE_ID = 0
GPU_CHUNKS_PER_BATCH = 4                # chunks lit per GPU batch: STEPS_PER_CHUNK x this many steps
WORKERS = 8                             # patches in flight: horizons decoded and files compressed meanwhile
STATUS_SECONDS = 60.0

# Console and log file. LOG_PATH None writes next to the output directory. COLOR is "auto",
# "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF"),
    Option("HORIZONS_PATH", "path", "the horizon tile store"),
    Option("OUTPUT_DIRECTORY", "path", "where the store is written"),
    Option("OBSERVER_HEIGHT_M", "float", "observer height above the terrain (m); must match the horizon tiles"),
    Option("START_UTC", "str", "first step (UTC)"),
    Option("STOP_UTC", "str", "end of the range, exclusive (UTC)"),
    Option("STEP_MINUTES", "int", "the step; it divides 24 hours"),
    Option("SHADOW_FRACTION", "float", "a sample below this Sun fraction is power-negative shadow"),
    Option("OPERATING_FRACTION", "float", "at or above this Sun fraction the rover operates"),
    Option("EARTH_ELEVATION_DEG", "float", "Earth below this above the pixel's own horizon is an outage"),
    Option("STEPS_PER_CHUNK", "int", "steps per independently compressed chunk (96 = 8 days at 2 hours)"),
    Option("CODEC", "str", "chunk codec"),
    Option("PATCHES", "pairs", "light only these patches (row,column ...), a trial"),
    Option("LIGHTING", "str", "auto uses the GPU when there is one, cuda requires it, cpu never touches CUDA",
           choices=("auto", "cuda", "cpu")),
    Option("GPU_DEVICE_ID", "int", "which GPU"),
    Option("GPU_CHUNKS_PER_BATCH", "int", "chunks lit per GPU batch"),
    Option("WORKERS", "int", "patches in flight"),
    Option("STATUS_SECONDS", "float", "seconds between status lines"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now)."""
    return defaults_from(OPTIONS, globals())


def run_light_store(*, dem_path, horizons_path, output_directory, observer_height_m, start_utc, stop_utc, step_minutes,
                    shadow_fraction, operating_fraction, earth_elevation_deg, steps_per_chunk, codec, patches, lighting,
                    gpu_device_id, gpu_chunks_per_batch, workers, status_seconds) -> int:
    """Write the light store; returns 0 when it is complete and 1 when the run was interrupted (run it again)."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    dem, _georef = _load_dem(dem_path)
    settings = Stage0Settings(
        store=LightStoreSettings(
            step_minutes=step_minutes, steps_per_chunk=steps_per_chunk, shadow_fraction=shadow_fraction,
            operating_fraction=operating_fraction, earth_elevation_deg=earth_elevation_deg,
            observer_height_m=observer_height_m, codec=codec,
        ),
        start_utc=start_utc, stop_utc=stop_utc, lighting=lighting, gpu_device_id=gpu_device_id,
        gpu_chunks_per_batch=gpu_chunks_per_batch, workers=workers, status_seconds=status_seconds,
    )
    report = generate_light_store(
        output_directory, dem=dem, horizon_store=HorizonTileStore(horizons_path), settings=settings,
        sun_vectors_at=spice_vector_source("sun"), earth_vectors_at=spice_vector_source("earth"),
        patches=patches,
    )
    LOG.info("Store written to: %s", output_directory)
    if report.status != "completed":
        LOG.warning("The run is %s; run this script again to continue.", report.status)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 0: the thresholded-lighting store. Every option's default is the constant of the same name "
                    "at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the store."""
    args = build_parser().parse_args(list(argv))
    options = options_from(args, OPTIONS)
    default_log = options["output_directory"].with_name(options["output_directory"].name + ".log")
    return run_with_logging(args, default_log_path=default_log, logger=LOG, label="Stage 0",
                            procedure=lambda: run_light_store(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
