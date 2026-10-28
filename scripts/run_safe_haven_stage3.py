#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage3.py [--flood-workers 8 --max-slope-deg 15 ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; the rover's flood settings are one flag per field; `--help` lists them. run_stage3(...) takes the same values as keyword arguments, so
another script can call it (scripts/run_safe_haven.py runs the whole safe-haven sequence that way).

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

import argparse
import json
import logging
import sys
import time as clock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.safe_haven_floods import FloodSettings, run_floods  # noqa: E402
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_flood_settings, add_logging_options, add_options, defaults_from, flood_settings_from, options_from,
    run_with_logging,
)
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
OVERWRITE = True        # existing tables are replaced when the run completes; False refuses to replace them

SETTINGS = FloodSettings(
    max_days=28.0,              # the flood and the endurance end here
    moves_per_step=4,           # 4 pixels per 2-hour step: 0.0111 m/s at 20 m
    connectivity=8,
    max_slope_deg=15.0,         # the Horn slope of a pixel the rover may enter or stay on
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


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF"),
    Option("LIGHT_STORE_PATH", "path", "the stage-0 light store"),
    Option("SEEDS_PATH", "path", "the stage-2 seed table"),
    Option("PSR_IDS_PATH", "path", "the PSR id raster (generate_psr_regions.py)"),
    Option("PSR_TABLE_PATH", "path", "the PSR region table (generate_psr_regions.py)"),
    Option("PASSABLE_PATH", "path", "the cached passable mask (made if missing or for other settings)"),
    Option("OUTPUT_PATH", "path", "the per-seed flood table"),
    Option("EDGES_PATH", "path", "the seed-to-seed edge table"),
    Option("OVERWRITE", "bool", "replace existing tables when the run completes"),
    Option("SEED_IDS", "ids", "flood only these seeds, a trial"),
    Option("FLOOD_WORKERS", "int", "processes flooding seeds in parallel"),
    Option("PROGRESS_EVERY_GROUPS", "int", "log progress every this many cell-months"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now), the flood settings as ``settings``."""
    return {**defaults_from(OPTIONS, globals()), "settings": SETTINGS}


def run_stage3(*, dem_path, light_store_path, seeds_path, psr_ids_path, psr_table_path, passable_path, output_path,
               edges_path, overwrite, seed_ids, flood_workers, progress_every_groups, settings) -> int:
    """Flood every plan seed and write the flood and edge tables; returns 0. ``settings`` is a ``FloodSettings``."""
    dem, georef = _load_dem(dem_path)
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % progress_every_groups and done != total:
            return
        elapsed = clock.monotonic() - started
        LOG.info("Stage 3: %d/%d cell-months, %.1f min elapsed, about %.1f min left (rate from this run only)",
                 done, total, elapsed / 60, elapsed / done * (total - done) / 60)

    report = run_floods(
        output_path, edges_path,
        seeds_path=seeds_path, dem=dem, georef=georef, directory=LightStoreDirectory(light_store_path, dem),
        psr_ids_path=psr_ids_path, psr_table_path=psr_table_path, passable_path=passable_path,
        settings=settings, seed_ids=seed_ids, workers=flood_workers, overwrite=overwrite, progress=progress)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d seeds flooded, %d edges; tables written to: %s and %s", report.seeds, report.edges,
             output_path, edges_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 3: the forward floods. Every option's default is the constant of the same name at the top "
                    "of this file.")
    add_options(parser, OPTIONS, default_options())
    add_flood_settings(parser, SETTINGS)
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and run the floods."""
    args = build_parser().parse_args(list(argv))
    options = {**options_from(args, OPTIONS), "settings": flood_settings_from(args, SETTINGS)}
    output_path = Path(options["output_path"])
    return run_with_logging(args, default_log_path=output_path.with_name(output_path.name + ".log"), logger=LOG,
                            label="Stage 3", procedure=lambda: run_stage3(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
