#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage4.py [--flood-workers 8 --max-slope-deg 15 ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; the flood settings are one flag per field; `--help` lists them. run_stage4(...) takes the same values as keyword arguments, so
another script can call it (scripts/run_safe_haven.py runs the whole safe-haven sequence that way).

Stage 4 of the rover safe-haven chain: chain the plan seeds over the edges of stage 3
(scripts/run_safe_haven_stage3.py). An edge A -> B says a rover that wakes at seed A can be on seed B's pixel
when B's outage begins, hibernates through it and wakes at B's after boundary. Following edges gives a chain of
seeds and a mission that lasts from the first seed's wake-up to the end of the last seed's lit endurance.

Every seed gets one chain label, (root, root's operations start): a seed with no incoming edge is its own root;
otherwise it takes the label of the incoming edge whose root starts earliest, ties broken by the lowest root id,
and among edges with that same root the longer chain, then the lowest predecessor id. The labels are made in
time order, so the result does not depend on the order of the tables. The PSR counts need the rover's routes, which
stage 3 did not keep: the flood of every predecessor seed is run again from the light store with its frontiers kept
and traced back from the successor seed (see run_safe_haven_chain_maps.py). That takes a minute or two with 16
workers; the rest reads three tables and takes seconds. CPU only, no NVIDIA device.

One row per seed is written to OUTPUT_PATH, the final table: the stage-2 columns (cell, month,
pixel, x_m and y_m: the pixel centre in the DEM's CRS, made from DEM_PATH; hibernation, outage kind, Earth
cycle, outage start and end, before and after steps), the stage-3 columns (seed_passable, lit_endurance_hours,
reach48_pixels, the PSR touches) and

* root_id, root_start_utc: the first seed of the chain that reaches this seed and its operations start;
* predecessor_id: the seed before this one in that chain (0 for a root);
* chain_length: the seeds in the chain, this one included (1 for a root);
* mission_hours: the hours from the root's operations start to the end of this seed's lit endurance (a lower
  bound where the last seed's flood hit the cap);
* previous_seeds, next_seeds: the numbers of edges into and out of the seed;
* chain_psr_small, chain_psr_medium, chain_psr_large: the number of distinct permanently shadowed regions (PSRs) of
  each size class that the chain's routes (root to this seed) pass next to, a PSR pixel being among the eight
  neighbours of a route pixel. Small is up to 400 m2 (one pixel at 20 m), medium up to 1,000,000 m2, large above (the
  limits of FLOOD_SETTINGS, which must be stage 3's).

A JSON report (chain length counts, mission percentiles and the longest chains) is written next to the table.
Importing this file starts nothing.

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

from lunarscout._numba_horizon.pixel_coordinates import dem_header  # noqa: E402
from lunarscout._numba_horizon.safe_haven_chains import RouteInputs, run_chains  # noqa: E402
from lunarscout._numba_horizon.safe_haven_floods import FloodSettings  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_flood_settings, add_logging_options, add_options, defaults_from, flood_settings_from, options_from,
    run_with_logging,
)


# Edit these paths before running. They match run_safe_haven_stage3.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")                 # for the seeds' x_m and y_m
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
SEEDS_PATH = STAGE_DIRECTORY / "safe_haven_stage_2_plan_seeds.csv"             # stage 2
FLOODS_PATH = STAGE_DIRECTORY / "safe_haven_stage_3_floods.csv"                # stage 3
EDGES_PATH = STAGE_DIRECTORY / "safe_haven_stage_3_edges.csv"                  # stage 3
OUTPUT_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
STEP_MINUTES = 120      # the light store's step (the unit of the steps in the seed table)
# For the PSR counts: the routes are reconstructed from the light store and the passable mask stage 3 cached, so
# FLOOD_SETTINGS must be the settings stage 3 ran with (run_safe_haven_stage3.py).
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")
PASSABLE_PATH = STAGE_DIRECTORY / "safe_haven_stage_3_passable_mask.npy"
PSR_IDS_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_ids.tif")             # scripts/generate_psr_regions.py
PSR_TABLE_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_regions.csv")
FLOOD_SETTINGS = FloodSettings(max_days=28.0, moves_per_step=4, connectivity=8, max_slope_deg=15.0,
                               psr_small_max_m2=400.0, psr_medium_max_m2=1_000_000.0)
ROUTE_WORKERS = 16
PROGRESS_EVERY_SEEDS = 500
OVERWRITE = True        # an existing table is replaced when the run completes; False refuses to replace it

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the output file. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.stage4")


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF (for the pixel coordinates and the grid)"),
    Option("SEEDS_PATH", "path", "the stage-2 seed table"),
    Option("FLOODS_PATH", "path", "the stage-3 flood table"),
    Option("EDGES_PATH", "path", "the stage-3 edge table"),
    Option("OUTPUT_PATH", "path", "the final per-seed table"),
    Option("STEP_MINUTES", "int", "the light store's step (the unit of the steps in the seed table)"),
    Option("LIGHT_STORE_PATH", "path", "the stage-0 light store (for the routes)"),
    Option("PASSABLE_PATH", "path", "the passable mask stage 3 cached"),
    Option("PSR_IDS_PATH", "path", "the PSR id raster"),
    Option("PSR_TABLE_PATH", "path", "the PSR region table"),
    Option("ROUTE_WORKERS", "int", "processes reconstructing routes in parallel"),
    Option("PROGRESS_EVERY_SEEDS", "int", "log progress every this many predecessor seeds"),
    Option("OVERWRITE", "bool", "replace an existing table when the run completes"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now), the flood settings as ``flood_settings``."""
    return {**defaults_from(OPTIONS, globals()), "flood_settings": FLOOD_SETTINGS}


def run_stage4(*, dem_path, seeds_path, floods_path, edges_path, output_path, step_minutes, light_store_path,
               passable_path, psr_ids_path, psr_table_path, route_workers, progress_every_seeds, overwrite,
               flood_settings) -> int:
    """Chain the seeds and write the final table; returns 0. ``flood_settings`` must be stage 3's."""
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % progress_every_seeds and done != total:
            return
        elapsed = clock.monotonic() - started
        LOG.info("Routes: %d/%d predecessor seeds, %.1f min elapsed, about %.1f min left (rate from this run only)",
                 done, total, elapsed / 60, elapsed / done * (total - done) / 60)

    transform, width, height, _crs = dem_header(dem_path)
    routes = RouteInputs(
        light_store_path=light_store_path, passable_path=passable_path, psr_ids_path=psr_ids_path,
        psr_table_path=psr_table_path, dem_shape=(height, width), flood_settings=flood_settings,
        workers=route_workers, progress=progress)
    report = run_chains(output_path, seeds_path=seeds_path, floods_path=floods_path, edges_path=edges_path,
                        geo_transform=transform, step_minutes=step_minutes, overwrite=overwrite, routes=routes)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d seeds in chains over %d edges, longest chain %d seeds, longest mission %.0f h; table written to: %s",
             report.seeds_in_chains, report.edges, report.longest_chain, report.longest_mission_hours, output_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 4: the chains. Every option's default is the constant of the same name at the top of this "
                    "file.")
    add_options(parser, OPTIONS, default_options())
    add_flood_settings(parser, FLOOD_SETTINGS)
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the final table."""
    args = build_parser().parse_args(list(argv))
    options = {**options_from(args, OPTIONS), "flood_settings": flood_settings_from(args, FLOOD_SETTINGS)}
    output_path = Path(options["output_path"])
    return run_with_logging(args, default_log_path=output_path.with_name(output_path.name + ".log"), logger=LOG,
                            label="Stage 4", procedure=lambda: run_stage4(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
