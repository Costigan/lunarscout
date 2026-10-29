#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_monthly_chain_maps.py [--route-workers 8 ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; the flood settings are one flag per field; `--help` lists them. make_monthly_chain_maps(...)
takes the same values as keyword arguments, so another script can call it (scripts/run_safe_haven.py runs the whole
safe-haven sequence that way).

The pixels of the longest-chain image of each month, for the pole viewer (scripts/view_safe_haven.py):
safe_haven_stage_4_monthly_chain_pixels.npz in OUTPUT_DIRECTORY. It is the yearly image
(scripts/run_safe_haven_yearly_chain_maps.py) by month: a chain belongs to the UTC month in which its root starts
(`root_start_utc` in the stage-4 table) and stays whole in it, the havens drawn (white) are the seeds of that month's
chains, isolated ones too, and the routes, aqua PSR passes and length colours are the same, so the pixels of the months
of a year are the pixels of that year's image. A month is stored as only the pixels drawn (a few thousand to some tens of
thousands): their row, column and colour class, ordered by class so that drawing them in order puts the longer chains,
then aqua, then the havens on top, and the colour table and the number of chains of each month. Every month from the
first to the last in which a chain starts is in the file; a month without chains has no pixels.

The lines are the rover's routes (RASTER_ROUTES), found by flooding every predecessor seed again from the light store
(the flood settings must be those stage 3 ran with). With RASTER_ROUTES False the lines are straight and no light store is
needed. The file is written through a staging file, so a failure leaves any earlier file untouched; an existing file is
replaced as its new version is complete (OVERWRITE True; False refuses to replace it). The routes take about 35 s on the
real table and each month then a few seconds (the 96 months of the standard window take a few minutes); about 6 GB of
memory is used. It is CPU only and needs no NVIDIA device. Importing it starts nothing.

Progress and status messages are logged: to the console, colored with rich on a terminal when
`pip install "lunarscout[console]"` is installed and plain when piped, and to a plain-text log
file in OUTPUT_DIRECTORY (LOG_PATH, COLOR and LOG_LEVEL below).
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

from lunarscout._numba_horizon.safe_haven_chain_maps import run_monthly_chain_maps  # noqa: E402
from lunarscout._numba_horizon.safe_haven_floods import FloodSettings  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_flood_settings, add_logging_options, add_options, defaults_from, flood_settings_from, options_from,
    run_with_logging,
)


# Edit these paths before running. They match run_safe_haven_chain_maps.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
CHAINS_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")
PSR_IDS_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_ids.tif")        # None marks no PSR (scripts/generate_psr_regions.py)
PASSABLE_PATH = STAGE_DIRECTORY / "safe_haven_stage_3_passable_mask.npy"      # cached by stage 3
# Must be the settings stage 3 ran with (run_safe_haven_stage3.py).
FLOOD_SETTINGS = FloodSettings(max_days=28.0, moves_per_step=4, connectivity=8, max_slope_deg=15.0)
RASTER_ROUTES = True    # False: straight lines (quick, no light store needed)
ROUTE_WORKERS = 16
OUTPUT_DIRECTORY = STAGE_DIRECTORY
OVERWRITE = True        # existing files are replaced as the new ones are complete; False refuses to replace them
PROGRESS_EVERY_SEEDS = 500

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the outputs. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.monthly_chain_maps")


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF (its grid and CRS)"),
    Option("CHAINS_PATH", "path", "the stage-4 table"),
    Option("OUTPUT_DIRECTORY", "path", "where the monthly pixel file is written"),
    Option("OVERWRITE", "bool", "replace existing files as the new ones are complete"),
    Option("LIGHT_STORE_PATH", "path", "the stage-0 light store (for the routes)"),
    Option("PSR_IDS_PATH", "optional_path", "the PSR id raster; none marks no PSR"),
    Option("PASSABLE_PATH", "path", "the passable mask stage 3 cached"),
    Option("RASTER_ROUTES", "bool", "the images follow the rover's routes; false draws straight lines"),
    Option("ROUTE_WORKERS", "int", "processes reconstructing routes in parallel"),
    Option("PROGRESS_EVERY_SEEDS", "int", "log progress every this many predecessor seeds"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now), the flood settings as ``flood_settings``."""
    return {**defaults_from(OPTIONS, globals()), "flood_settings": FLOOD_SETTINGS}


def make_monthly_chain_maps(*, dem_path, chains_path, output_directory, overwrite, light_store_path,
                           psr_ids_path, passable_path, raster_routes, route_workers, progress_every_seeds,
                           flood_settings) -> int:
    """Write the monthly chain pixel file; returns 0. ``flood_settings`` must be stage 3's."""
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % progress_every_seeds and done != total:
            return
        elapsed = clock.monotonic() - started
        LOG.info("Routes: %d/%d predecessor seeds, %.1f min elapsed, about %.1f min left (rate from this run only)",
                 done, total, elapsed / 60, elapsed / done * (total - done) / 60)

    report = run_monthly_chain_maps(
        output_directory, chains_path=chains_path, dem_path=dem_path, overwrite=overwrite,
        light_store_path=light_store_path if raster_routes else None, passable_path=passable_path,
        flood_settings=flood_settings, workers=route_workers, progress=progress, psr_ids_path=psr_ids_path)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d months, %s pixels written to: %s (%d chains in all; raster lines: %s)", len(report.months),
             f"{report.pixels:,}", report.file, sum(m["chains"] for m in report.months.values()), report.raster_lines)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The pixels of each root's longest-chain map, by the month the chain starts. Every option's default is the "
                    "constant of the same name at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_flood_settings(parser, FLOOD_SETTINGS)
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the monthly chain pixels."""
    args = build_parser().parse_args(list(argv))
    options = {**options_from(args, OPTIONS), "flood_settings": flood_settings_from(args, FLOOD_SETTINGS)}
    output_directory = Path(options["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    return run_with_logging(args, default_log_path=output_directory / "safe_haven_stage_4_monthly_chain_maps.log",
                            logger=LOG, label="Monthly chain maps", procedure=lambda: make_monthly_chain_maps(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
