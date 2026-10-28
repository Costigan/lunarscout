#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage2.py [--plan-workers 8 --survival-limit-hours 72 ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; the seed rules of PARAMETERS are one flag per field; `--help` lists them. run_stage2(...) takes
the same values as keyword arguments, so another script can call it (scripts/run_safe_haven.py runs the whole
safe-haven sequence that way).

Stage 2 of the rover safe-haven chain: choose the plan seeds. For every 5.12 km cell and month with a
haven pixel in the stage-1 hibernation product (hibernation at most SURVIVAL_LIMIT_HOURS), pick up to
five seed pixels, recompute each seed's hibernation from the stage-0 light store (it must equal the
stage-1 value) and write one row per seed to a CSV table: its cell, month, pixel (raster row and
column, and x_m and y_m, the pixel centre in the DEM's CRS), hibernation, the kind of its outage, its Earth
cycle, the outage's start and end, and the flood-grid steps of the first lit step after the outage (where stage 3 floods from) and the last lit step before it (where a
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

import argparse
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
from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, add_window_parameters, defaults_from, options_from, run_with_logging,
    window_parameters_from,
)
from lunarscout.products import _load_dem  # noqa: E402


# Edit these paths before running. They match run_safe_haven_stage1.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")      # stage 0
STAGE1_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
STAGE1_PREFIX = "safe_haven_stage_1"  # must equal NAME_PREFIX in run_safe_haven_stage1.py
# The seed table. An existing table (or report) is not replaced unless OVERWRITE is True.
OUTPUT_PATH = STAGE1_DIRECTORY / "safe_haven_stage_2_plan_seeds.csv"
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


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF"),
    Option("LIGHT_STORE_PATH", "path", "the stage-0 light store"),
    Option("STAGE1_DIRECTORY", "path", "where the stage-1 products are"),
    Option("STAGE1_PREFIX", "str", "the stage-1 file prefix (NAME_PREFIX of stage 1)"),
    Option("OUTPUT_PATH", "path", "the seed table"),
    Option("OVERWRITE", "bool", "replace an existing table"),
    Option("START_UTC", "str", "first sample (UTC); must equal the stage-1 run"),
    Option("STOP_UTC", "str", "last sample (UTC); must equal the stage-1 run"),
    Option("STEP_HOURS", "float", "sample step in hours"),
    Option("EARTH_ELEVATION_THRESHOLD_DEG", "float", "Earth below this elevation is an outage"),
    Option("SUNLIGHT_FRACTION_THRESHOLD", "float", "a Sun fraction below this is shadow"),
    Option("MERGE_GAP_HOURS", "float", "shadow runs closer than this are one run"),
    Option("HOURS_PER_COUNT", "optional_float", "the stage-1 products' unit (2 hours per count); none: hours"),
    Option("STAGE1_NODATA", "optional_int", "the stage-1 products' nodata byte; none: float products"),
    Option("CELLS", "pairs", "plan only these cells (cell_row,cell_column ...), a trial"),
    Option("PLAN_WORKERS", "int", "processes planning cells in parallel"),
    Option("PROGRESS_EVERY_CELLS", "int", "log progress every this many cells"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now), the seed rules as ``parameters``."""
    return {**defaults_from(OPTIONS, globals()), "parameters": PARAMETERS}


def run_stage2(*, dem_path, light_store_path, stage1_directory, stage1_prefix, output_path, overwrite, start_utc,
               stop_utc, step_hours, earth_elevation_threshold_deg, sunlight_fraction_threshold, merge_gap_hours,
               hours_per_count, stage1_nodata, cells, plan_workers, progress_every_cells, parameters) -> int:
    """Write the seed table (and its report); returns 0. ``parameters`` is a ``WindowParameters`` with
    ``forward_only`` True."""
    dem, _georef = _load_dem(dem_path)
    stage1_directory = Path(stage1_directory)
    stage1 = Stage1Rasters(
        stage1_directory / f"{stage1_prefix}_hibernation_hours.tif",
        stage1_directory / f"{stage1_prefix}_joint_operating_total_hours.tif",
        hours_per_count=hours_per_count,
        nodata=stage1_nodata,
    )
    times = ls.times(start_utc, stop_utc, step_hours=step_hours)
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % progress_every_cells and done != total:
            return
        elapsed = clock.monotonic() - started
        LOG.info("Stage 2: %d/%d cells, %.1f min elapsed, about %.1f min left (rate from this run only)",
                 done, total, elapsed / 60, elapsed / done * (total - done) / 60)

    report = plan_seeds(
        output_path,
        dem=dem,
        stage1=stage1,
        light_store=LightStoreDirectory(light_store_path, dem),
        times=times.values,
        earth_vectors_at=spice_vector_source("earth"),
        parameters=parameters,
        earth_elevation_threshold_deg=earth_elevation_threshold_deg,
        sunlight_fraction_threshold=sunlight_fraction_threshold,
        merge_gap_hours=merge_gap_hours,
        cells=cells,
        workers=plan_workers,
        overwrite=overwrite,
        progress=progress,
    )
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d seeds in %d cells over %d Earth cycles; seed table written to: %s",
             report.seeds, report.cells_with_seeds, report.earth_cycles, output_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2: the plan seeds. Every option's default is the constant of the same name at the top of "
                    "this file.")
    add_options(parser, OPTIONS, default_options())
    add_window_parameters(parser, PARAMETERS)
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the seed table."""
    args = build_parser().parse_args(list(argv))
    options = {**options_from(args, OPTIONS), "parameters": window_parameters_from(args, PARAMETERS)}
    output_path = Path(options["output_path"])
    return run_with_logging(args, default_log_path=output_path.with_name(output_path.name + ".log"), logger=LOG,
                            label="Stage 2", procedure=lambda: run_stage2(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
