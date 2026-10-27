#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage0.py

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

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.file_format import HorizonTileStore  # noqa: E402
from lunarscout._numba_horizon.safe_haven_lightstore import LightStoreSettings  # noqa: E402
from lunarscout._numba_horizon.safe_haven_stage0 import Stage0Settings, generate_light_store  # noqa: E402
from lunarscout._numba_horizon.safe_haven_windows import spice_vector_source  # noqa: E402
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402
from lunarscout.products import _load_dem  # noqa: E402

LOG = logging.getLogger("lunarscout.scripts.run_safe_haven_stage0")

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


def _run() -> int:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    dem, _georef = _load_dem(DEM_PATH)
    settings = Stage0Settings(
        store=LightStoreSettings(
            step_minutes=STEP_MINUTES, steps_per_chunk=STEPS_PER_CHUNK, shadow_fraction=SHADOW_FRACTION,
            operating_fraction=OPERATING_FRACTION, earth_elevation_deg=EARTH_ELEVATION_DEG,
            observer_height_m=OBSERVER_HEIGHT_M, codec=CODEC,
        ),
        start_utc=START_UTC, stop_utc=STOP_UTC, lighting=LIGHTING, gpu_device_id=GPU_DEVICE_ID,
        gpu_chunks_per_batch=GPU_CHUNKS_PER_BATCH, workers=WORKERS, status_seconds=STATUS_SECONDS,
    )
    report = generate_light_store(
        OUTPUT_DIRECTORY, dem=dem, horizon_store=HorizonTileStore(HORIZONS_PATH), settings=settings,
        sun_vectors_at=spice_vector_source("sun"), earth_vectors_at=spice_vector_source("earth"),
        patches=PATCHES,
    )
    LOG.info("Store written to: %s", OUTPUT_DIRECTORY)
    if report.status != "completed":
        LOG.warning("The run is %s; run this script again to continue.", report.status)
        return 1
    return 0


def main() -> int:
    log_path = LOG_PATH if LOG_PATH is not None else OUTPUT_DIRECTORY.with_name(OUTPUT_DIRECTORY.name + ".log")
    setup_script_logging(log_path, color=COLOR, level=LOG_LEVEL)
    try:
        return _run()
    except Exception as error:
        LOG.error("Stage 0 stopped: %s: %s", type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
