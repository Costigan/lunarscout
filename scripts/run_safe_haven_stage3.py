#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage3.py

Stage 3 of the rover safe-haven chain: flood forward in time from every plan seed of stage 2
(scripts/run_safe_haven_stage2.py). The rover starts at the seed pixel on the seed's after boundary and, in each
2-hour step, makes 4 one-pixel moves, each into a passable pixel that is lit in that step (or stays on its pixel
if that is lit). The flood ends when no lit pixel can be reached or after MAX_DAYS. Lit pixels are read from the
stage-0 light store (scripts/run_light_store.py), 24 hours of one patch at a time, as the flood needs them.

One row per seed is written to OUTPUT_PATH:

* lit_endurance_hours: how long the rover can stay on lit terrain from the after boundary, at most MAX_DAYS
  (a value at the cap is a lower bound);
* reach48_pixels: the pixels the rover can be on 48 hours after the boundary;
* touch_small / touch_medium / touch_large: the hours after the boundary at which a passable pixel of a small,
  medium or large permanently shadowed region (PSR) is next to the frontier, and that region's id (from
  scripts/generate_psr_regions.py); blank if none is reached. Small is under PSR_SMALL_MAX_M2, medium up to
  PSR_MEDIUM_MAX_M2, large above;
* seed_passable: 0 if the seed pixel is not passable terrain (a seed on a slope over MAX_SLOPE_DEG). The rover
  still starts there and moves off it, but it cannot stay on that pixel.

EDGES_PATH lists the seeds a flood arrives at (seed_id, target_id, arrival_hours): the seeds of the next Earth
cycle (EDGE_RULE) whose pixel is in the frontier at the target's before boundary. Stage 4 chains them.

The passable mask (valid height, slope at most MAX_SLOPE_DEG, a light-store file for the patch) is made once
and kept at PASSABLE_PATH (about 115 MB); it is remade when the slope limit or the store changes. The run
checkpoints to OUTPUT_PATH + ".checkpoint.jsonl" after every cell-month, so run it again after a kill, a crash
or Ctrl-C with the same settings to resume (a different setting needs the checkpoint deleted). Memory is
about UNIT_CACHE_MIB per worker plus the PSR id raster (3.7 GB, shared). It is CPU only, reads the store's
files throughout and needs no NVIDIA device. Importing it starts nothing.

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

from lunarscout._numba_horizon.safe_haven_floods import FloodSettings, run_floods  # noqa: E402
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory  # noqa: E402
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402
from lunarscout.products import _load_dem  # noqa: E402


# Edit these paths before running. They match run_safe_haven_stage2.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")      # stage 0
STAGE1_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
SEEDS_PATH = STAGE1_DIRECTORY / "safe_haven_stage_2_plan_seeds.csv"           # stage 2
# The PSR products (scripts/generate_psr_regions.py).
PSR_IDS_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_ids.tif")
PSR_TABLE_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_regions.csv")
PASSABLE_PATH = STAGE1_DIRECTORY / "safe_haven_stage_3_passable_mask.npy"
OUTPUT_PATH = STAGE1_DIRECTORY / "safe_haven_stage_3_floods.csv"
EDGES_PATH = STAGE1_DIRECTORY / "safe_haven_stage_3_edges.csv"
OVERWRITE = False       # existing tables are not replaced unless this is True

SETTINGS = FloodSettings(
    max_days=28.0,              # the flood and the endurance end here
    moves_per_step=4,           # 4 pixels per 2-hour step: 0.0111 m/s at 20 m
    connectivity=8,
    max_slope_deg=20.0,
    psr_small_max_m2=400.0,     # small: up to 400 m2 (one pixel at 20 m); medium: up to 1,000,000 m2; large above
    psr_medium_max_m2=1_000_000.0,
    reach_hours=48.0,
    edge_rule="next_cycle",     # or "any_later"
    unit_cache_mib=1024,        # per worker
)

SEED_IDS = None         # None floods every seed. For a trial, list seed ids.
FLOOD_WORKERS = 16
PROGRESS_EVERY_GROUPS = 200

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the output file. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.stage3")


def _run() -> int:
    dem, georef = _load_dem(DEM_PATH)
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % PROGRESS_EVERY_GROUPS and done != total:
            return
        elapsed = clock.monotonic() - started
        LOG.info("Stage 3: %d/%d cell-months, %.1f min elapsed, about %.1f min left (rate from this run only)",
                 done, total, elapsed / 60, elapsed / done * (total - done) / 60)

    report = run_floods(
        OUTPUT_PATH, EDGES_PATH,
        seeds_path=SEEDS_PATH, dem=dem, georef=georef, directory=LightStoreDirectory(LIGHT_STORE_PATH, dem),
        psr_ids_path=PSR_IDS_PATH, psr_table_path=PSR_TABLE_PATH, passable_path=PASSABLE_PATH,
        settings=SETTINGS, seed_ids=SEED_IDS, workers=FLOOD_WORKERS, overwrite=OVERWRITE, progress=progress)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d seeds flooded, %d edges; tables written to: %s and %s", report.seeds, report.edges,
             OUTPUT_PATH, EDGES_PATH)
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
