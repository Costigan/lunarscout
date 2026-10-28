#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage4.py

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
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402


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
FLOOD_SETTINGS = FloodSettings(max_days=28.0, moves_per_step=4, connectivity=8, max_slope_deg=20.0,
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


def _run() -> int:
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % PROGRESS_EVERY_SEEDS and done != total:
            return
        elapsed = clock.monotonic() - started
        LOG.info("Routes: %d/%d predecessor seeds, %.1f min elapsed, about %.1f min left (rate from this run only)",
                 done, total, elapsed / 60, elapsed / done * (total - done) / 60)

    transform, width, height, _crs = dem_header(DEM_PATH)
    routes = RouteInputs(
        light_store_path=LIGHT_STORE_PATH, passable_path=PASSABLE_PATH, psr_ids_path=PSR_IDS_PATH,
        psr_table_path=PSR_TABLE_PATH, dem_shape=(height, width), flood_settings=FLOOD_SETTINGS,
        workers=ROUTE_WORKERS, progress=progress)
    report = run_chains(OUTPUT_PATH, seeds_path=SEEDS_PATH, floods_path=FLOODS_PATH, edges_path=EDGES_PATH,
                        geo_transform=transform, step_minutes=STEP_MINUTES, overwrite=OVERWRITE, routes=routes)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d seeds in chains over %d edges, longest chain %d seeds, longest mission %.0f h; table written to: %s",
             report.seeds_in_chains, report.edges, report.longest_chain, report.longest_mission_hours, OUTPUT_PATH)
    return 0


def main() -> int:
    log_path = LOG_PATH if LOG_PATH is not None else OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".log")
    setup_script_logging(log_path, color=COLOR, level=LOG_LEVEL)
    try:
        return _run()
    except Exception as error:
        LOG.error("Stage 4 stopped: %s: %s", type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
