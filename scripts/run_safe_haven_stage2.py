#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage2.py

Stage 2 of the rover safe-haven chain: choose the plan seeds. For every 5.12 km cell and month with a
haven pixel in the stage-1 hibernation product (hibernation at most SURVIVAL_LIMIT_HOURS), pick up to
five seed pixels, recompute each seed's hibernation from the stage-0 light store (it must equal the
stage-1 value) and write one row per seed to a CSV table: its cell, month, pixel, hibernation, the
kind of its outage, its Earth cycle, the outage's start and end, and the flood-grid steps of the
first lit step after the outage (where stage 3 floods from) and the last lit step before it (where a
rover coming from an earlier seed must arrive; blank when the seed is not lit before its outage).

There is no Earth gate: a seed need not have Earth available for 48 hours on either side. The outage
kinds are earth_never_sets (the one-sample outage defined where Earth never sets), outage_sun_lit
(a real outage no shadow touches) and outage_in_shadow. The Earth cycles are region-wide: the minima
of Earth's elevation above a flat horizon at the raster's centre, numbered from 0, about 27 days
apart; the "next" seeds of a seed are the seeds of the next cycle. A JSON report with the counts is
written next to the table.

The stage-1 products must come from scripts/run_safe_haven_stage1.py, and the light store from
scripts/run_light_store.py, with the same times, thresholds and degenerate-outage window. SPICE
supplies Earth's positions. This is CPU only (PLAN_WORKERS processes) and needs no NVIDIA device. It
is not resumable: it writes the table at the end, so run it again after an interruption. Importing it
starts nothing.

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
from lunarscout._numba_horizon.safe_haven_plan import plan_seeds  # noqa: E402
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory  # noqa: E402
from lunarscout._numba_horizon.safe_haven_windows import (  # noqa: E402
    Stage1Rasters,
    WindowParameters,
    spice_vector_source,
)
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402
from lunarscout.products import _load_dem  # noqa: E402


# Edit these paths before running. They match run_safe_haven_stage1.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")      # stage 0
STAGE1_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
STAGE1_PREFIX = "safe_haven_2h"  # must equal NAME_PREFIX in run_safe_haven_stage1.py
# The seed table. An existing table (or report) is not replaced unless OVERWRITE is True.
OUTPUT_PATH = STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_plan_seeds.csv"
OVERWRITE = False

# Must equal the stage-1 run.
START_UTC = "2024-01-01T00:00:00Z"
STOP_UTC = "2031-12-31T22:00:00Z"
STEP_HOURS = 2.0
EARTH_ELEVATION_THRESHOLD_DEG = 2.0
SUNLIGHT_FRACTION_THRESHOLD = 0.2
MERGE_GAP_HOURS = 24.0
# The stage-1 products are bytes of two-hour counts with NODATA 255. Set both to
# None for float products in hours.
HOURS_PER_COUNT = 2.0
STAGE1_NODATA = 255

# The seed rules (docs/safe-haven-windows-spec.md, section 2). The operating fraction and the
# degenerate-outage window must equal the stage-1 run's; forward_only must stay True.
PARAMETERS = WindowParameters(
    survival_limit_hours=72.0,
    operational_hours=48.0,
    flood_step_minutes=120,    # the light store's step
    moves_per_step=4,
    max_seeds_per_cell_month=5,
    seed_separation_m=250.0,
    seed_timing_tolerance_hours=6.0,
    haven_neighbourhood_m=100.0,
    min_nearby_haven_fraction=0.05,
    boundary_search_hours=24.0,
    operating_sunlight_fraction=1.0,
    cell_patches=2,
    degenerate_outage_window_hours=24.0,
    forward_only=True,
)

# None plans every cell. For a trial, list (cell_row, cell_column) pairs.
CELLS = None
PLAN_WORKERS = 16
PROGRESS_EVERY_CELLS = 100

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the output file. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.stage2")


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
        LOG.info("Stage 2: %d/%d cells, %.1f min elapsed, about %.1f min left (rate from this run only)",
                 done, total, elapsed / 60, elapsed / done * (total - done) / 60)

    report = plan_seeds(
        OUTPUT_PATH,
        dem=dem,
        stage1=stage1,
        light_store=LightStoreDirectory(LIGHT_STORE_PATH, dem),
        times=times.values,
        earth_vectors_at=spice_vector_source("earth"),
        parameters=PARAMETERS,
        earth_elevation_threshold_deg=EARTH_ELEVATION_THRESHOLD_DEG,
        sunlight_fraction_threshold=SUNLIGHT_FRACTION_THRESHOLD,
        merge_gap_hours=MERGE_GAP_HOURS,
        cells=CELLS,
        workers=PLAN_WORKERS,
        overwrite=OVERWRITE,
        progress=progress,
    )
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d seeds in %d cells over %d Earth cycles; seed table written to: %s",
             report.seeds, report.cells_with_seeds, report.earth_cycles, OUTPUT_PATH)
    return 0


def main() -> int:
    log_path = LOG_PATH if LOG_PATH is not None else OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".log")
    setup_script_logging(log_path, color=COLOR, level=LOG_LEVEL)
    try:
        return _run()
    except Exception as error:
        LOG.error("Stage 2 stopped: %s: %s", type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
