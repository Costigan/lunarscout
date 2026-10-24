#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage3.py

Stage 3 of the safe-haven analysis (docs/safe-haven-windows-spec.md): for every 5.12 km cell
and month with a haven, pick up to five seeds, flood each one forward and backward from its
hibernation over the lit terrain, and write one table row per reported seed (CSV). The cells
come from the stage-2 list (scripts/run_safe_haven_stage2.py), which must be made first, or
from CELLS below for a trial. The pole is processed in blocks in snake order; after each block
the script appends to a checkpoint file, so running it again with the same settings resumes at
the next block. Run it again after a kill, a crash or Ctrl-C. A different setting, including
BLOCK_PATCHES, needs a new OUTPUT_PATH or the checkpoint deleted.

The stage-1 products must come from scripts/run_generate_safe_havens.py with the same
times, thresholds, observer height and degenerate-outage window, and the horizons must
be the ones stage 1 used. SPICE supplies Sun and Earth positions unless you replace the
two vector sources.

This is the CPU reference route: CPU lightmap, CPU packed-bit flood, horizons held in
host memory. It needs no NVIDIA device. It loads the CPU heavily for many hours, and it reads
the horizon tiles from disk throughout: watch scripts/monitor_temperatures.py in a second
terminal for the first 10 to 20 minutes. A block of 8 patches holds up to 10 x 10
patches of float32 horizons (about 9.4 GB); lower BLOCK_PATCHES (a multiple of
CELL_PATCHES) if the host has less memory. Set CELLS for a trial before running every cell
(scripts/list_safe_haven_cells.py prints a CELLS line). Importing this file starts nothing.

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
from lunarscout._numba_horizon.safe_haven_blocks import run_pass_two  # noqa: E402
from lunarscout._numba_horizon.safe_haven_cells import read_cell_list  # noqa: E402
from lunarscout._numba_horizon.safe_haven_windows import (  # noqa: E402
    Stage1Rasters,
    WindowParameters,
    spice_vector_source,
)
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402
from lunarscout.products import _load_dem  # noqa: E402


# Edit these paths before running. They match run_generate_safe_havens.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
HORIZONS_PATH = Path("/e/lunar_analyst_scenarios/70south/lighting/horizons")
STAGE1_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
STAGE1_PREFIX = "safe_haven_degenerate"  # must equal NAME_PREFIX in run_generate_safe_havens.py
# The stage-2 list of cells with havens (scripts/run_safe_haven_stage2.py). It is refused if the
# stage-1 files, the survival limit or the cell size have changed since it was made.
CELL_LIST_PATH = STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_haven_cells.json"
# The table. The checkpoint is OUTPUT_PATH + ".checkpoint.jsonl" next to it and is
# removed when the table is complete. An existing OUTPUT_PATH is not replaced unless
# OVERWRITE is True.
OUTPUT_PATH = Path("/e/lunar_analyst_scenarios/70south/safe_havens/safe_haven_degenerate_windows.csv")
OVERWRITE = False

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
# the stage-1 value, and the degenerate-outage window the stage-1 run's.
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
    block_patches=8,                        # a multiple of cell_patches; memory scales with it
    degenerate_outage_window_hours=24.0,
)

# None runs every cell in the stage-2 list. For a trial, list (cell_row, cell_column) pairs, for
# example the CELLS line printed by scripts/list_safe_haven_cells.py; the stage-2 list is then
# not needed.
CELLS = None

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the output file. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.stage3")


def _run() -> int:
    dem, _georef = _load_dem(DEM_PATH)
    stage1 = Stage1Rasters(
        STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_hibernation_hours.tif",
        STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_joint_operating_total_hours.tif",
        hours_per_count=HOURS_PER_COUNT,
        nodata=STAGE1_NODATA,
    )
    cells = CELLS
    if cells is None:
        if not CELL_LIST_PATH.exists():
            raise ValueError(f"{CELL_LIST_PATH} does not exist; run scripts/run_safe_haven_stage2.py first "
                             "or set CELLS for a trial")
        found = read_cell_list(
            CELL_LIST_PATH, stage1, survival_limit_hours=PARAMETERS.survival_limit_hours,
            cell_size=PARAMETERS.cell_patches * 128,
        )
        cells = [(entry["cell_row"], entry["cell_col"]) for entry in found]
        LOG.info("Stage 3: %d cells from %s", len(cells), CELL_LIST_PATH)
    times = ls.times(START_UTC, STOP_UTC, step_hours=STEP_HOURS)
    started = clock.monotonic()
    baseline: list[int] = []

    def progress(done: int, total: int) -> None:
        if not baseline:
            baseline.append(done)
            LOG.info("Stage 3: %d/%d blocks already in the checkpoint.", done, total)
            return
        ran = done - baseline[0]
        elapsed = clock.monotonic() - started
        remaining = elapsed / ran * (total - done) if ran else float("nan")
        LOG.info(
            "Stage 3: %d/%d blocks, %.1f min elapsed, about %.1f min left (rate from this run only)",
            done, total, elapsed / 60, remaining / 60,
        )

    report = run_pass_two(
        dem=dem,
        horizon_store=HorizonTileStore(HORIZONS_PATH),
        stage1=stage1,
        times=times.values,
        sun_vectors_at=spice_vector_source("sun"),
        earth_vectors_at=spice_vector_source("earth"),
        output_path=OUTPUT_PATH,
        parameters=PARAMETERS,
        cells=cells,
        overwrite=OVERWRITE,
        observer_height_m=OBSERVER_HEIGHT_M,
        earth_elevation_threshold_deg=EARTH_ELEVATION_THRESHOLD_DEG,
        sunlight_fraction_threshold=SUNLIGHT_FRACTION_THRESHOLD,
        merge_gap_hours=MERGE_GAP_HOURS,
        progress=progress,
    )
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("Table written to: %s", OUTPUT_PATH)
    return 0


def main() -> int:
    log_path = LOG_PATH if LOG_PATH is not None else OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".log")
    setup_script_logging(log_path, color=COLOR, level=LOG_LEVEL)
    try:
        return _run()
    except Exception as error:
        LOG.error("Stage 3 stopped: %s: %s", type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
