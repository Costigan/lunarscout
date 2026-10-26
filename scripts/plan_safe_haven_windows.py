#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/plan_safe_haven_windows.py

Stage 3, pass 1 of the safe-haven window analysis (docs/safe-haven-windows-spec.md), the
seed-planning half of scripts/run_safe_haven_stage3.py run on its own for a trial:
for every 5.12 km cell and month with a haven, pick up to five seeds, recompute each
seed's hibernation from the horizons, find its flood boundaries, and write the plan
as JSON lines. It prints the counts that size the later lighting work. No flood runs
and no patch lighting is calculated.

The stage-1 products must come from scripts/run_generate_safe_havens.py with the
same times, thresholds, observer height and vectors. Pass 1 recomputes each seed's
hibernation score and stops with an error if it differs from the stage-1 raster.
SPICE supplies Sun and Earth positions unless you replace the two vector sources.
This is CPU-only and needs no NVIDIA device. Importing it starts no calculation.

Progress and status messages are logged: to the console, colored with rich on a terminal when
`pip install "lunarscout[console]"` is installed and plain when piped, and to a plain-text log
file next to the output (LOG_PATH, COLOR and LOG_LEVEL below), so `| tee` is not needed.
"""

from __future__ import annotations

import json
import logging
import sys
import time as clock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import lunarscout as ls  # noqa: E402
from lunarscout._numba_horizon.file_format import HorizonTileStore  # noqa: E402
from lunarscout._numba_horizon.safe_haven_windows import (  # noqa: E402
    Stage1Rasters,
    WindowParameters,
    run_pass_one,
    spice_vector_source,
)
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402
from lunarscout.products import _load_dem  # noqa: E402


# Edit these paths before running. They match run_generate_safe_havens.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
HORIZONS_PATH = Path("/e/lunar_analyst_scenarios/70south/lighting/horizons")
STAGE1_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
STAGE1_PREFIX = "safe_haven_2h"  # must equal NAME_PREFIX in run_generate_safe_havens.py
PLAN_PATH = Path("/e/lunar_analyst_scenarios/70south/safe_havens/safe_haven_2h_window_plan.jsonl")

# Must equal the stage-1 run.
START_UTC = "2024-01-01T00:00:00Z"
STOP_UTC = "2031-12-31T18:00:00Z"
STEP_HOURS = 6.0
EARTH_ELEVATION_THRESHOLD_DEG = 2.0
SUNLIGHT_FRACTION_THRESHOLD = 0.2
MERGE_GAP_HOURS = 24.0
OBSERVER_HEIGHT_M = 0.0
# The stage-1 products are bytes of two-hour counts with NODATA 255. Set both to
# None for float products in hours.
HOURS_PER_COUNT = 2.0
STAGE1_NODATA = 255

# Stage-2 parameters (the specification, section 2). Operating fraction must equal
# the stage-1 value.
PARAMETERS = WindowParameters(
    survival_limit_hours=72.0,
    operational_hours=48.0,
    rover_speed_m_s=0.01,
    connectivity=8,
    max_seeds_per_cell_month=5,
    seed_separation_m=250.0,
    seed_timing_tolerance_hours=6.0,
    haven_neighbourhood_m=100.0,
    min_nearby_haven_fraction=0.05,
    boundary_search_hours=24.0,
    operating_sunlight_fraction=1.0,
    cell_patches=2,
    degenerate_outage_window_hours=24.0,    # must equal the stage-1 run's
)

# None plans every cell. For a trial, list (cell_row, cell_column) pairs.
CELLS = None
PROGRESS_EVERY_CELLS = 50

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the output file. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.plan")


def _run() -> int:
    dem, _georef = _load_dem(DEM_PATH)
    stage1 = Stage1Rasters(
        STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_hibernation_hours.tif",
        STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_joint_operating_total_hours.tif",
        hours_per_count=HOURS_PER_COUNT,
        nodata=STAGE1_NODATA,
    )
    times = ls.times(START_UTC, STOP_UTC, step_hours=STEP_HOURS)
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % PROGRESS_EVERY_CELLS and done != total:
            return
        elapsed = clock.monotonic() - started
        remaining = elapsed / done * (total - done)
        LOG.info(
            "Pass 1: %d/%d cells, %.1f min elapsed, about %.1f min left (rate from this run only)",
            done, total, elapsed / 60, remaining / 60,
        )

    report = run_pass_one(
        plan_path=PLAN_PATH,
        dem=dem,
        horizon_store=HorizonTileStore(HORIZONS_PATH),
        stage1=stage1,
        times=times.values,
        sun_vectors_at=spice_vector_source("sun"),
        earth_vectors_at=spice_vector_source("earth"),
        parameters=PARAMETERS,
        observer_height_m=OBSERVER_HEIGHT_M,
        earth_elevation_threshold_deg=EARTH_ELEVATION_THRESHOLD_DEG,
        sunlight_fraction_threshold=SUNLIGHT_FRACTION_THRESHOLD,
        merge_gap_hours=MERGE_GAP_HOURS,
        cells=CELLS,
        progress=progress,
    )
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("Plan written to: %s", PLAN_PATH)
    return 0


def main() -> int:
    log_path = LOG_PATH if LOG_PATH is not None else PLAN_PATH.with_name(PLAN_PATH.name + ".log")
    setup_script_logging(log_path, color=COLOR, level=LOG_LEVEL)
    try:
        return _run()
    except Exception as error:
        LOG.error("Pass 1 stopped: %s: %s", type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
