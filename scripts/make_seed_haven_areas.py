#!/usr/bin/env python3
"""Write the seed haven areas of every month: the maps and pixel lists of the havens near each chain start.

    .venv/bin/python scripts/make_seed_haven_areas.py [--area-radius-meters 200 --no-overwrite ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; `--help` lists them. make_seed_haven_areas(...) takes the same values as keyword arguments, so
another script can call it (scripts/run_safe_haven.py runs the whole safe-haven sequence that way, as its
seed-haven-areas step). The work is done by the public function ``lunarscout.generate_seed_haven_areas``.

The area of a month is the current start of each chain drawn in the month (its first seed that is not passed, the pixels
marked as chain starts in the monthly chain maps), every haven of the month, a seed or not, whose centre is within
AREA_RADIUS_METERS of one of them (200 m, 10 pixels on the 20 m DEM), and each lone last seed as its pixel alone. The other
seeds, isolated ones, passed ones and later seeds of a chain, are not part of an area; the rules are in
docs/safe-haven-monthly-products.md. The havens are read from the monthly haven images (scripts/threshold_safe_havens.py, step
haven-months) in HAVEN_MONTHS_DIRECTORY; a month without an image has its chain starts and lone seeds alone.

Two products are written into OUTPUT_DIRECTORY: safe_haven_stage_4_seed_haven_areas/ (one GeoTIFF per month on the DEM's
grid: hibernation in counts of HOURS_PER_COUNT hours, 255 NODATA, a palette in the viewer's shadow-period classes, minimum
overviews) and safe_haven_stage_4_seed_haven_areas.npz (the same pixels as lists, for scripts/view_safe_haven.py, which
draws them as "Seed haven areas (monthly)"). Both are written through staging files; OVERWRITE False refuses to replace
existing products. It takes about 4 minutes on the real table (96 full-size images, 150 MB). CPU only; no NVIDIA device is used. Importing it starts nothing.

Progress and status messages are logged: to the console, colored with rich on a terminal when
`pip install "lunarscout[console]"` is installed and plain when piped, and to a plain-text log
file in OUTPUT_DIRECTORY (LOG_PATH, COLOR and LOG_LEVEL below).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout.safe_haven_areas import generate_seed_haven_areas  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, defaults_from, options_from, run_with_logging,
)


# Edit these paths before running. They match run_safe_haven_chain_maps.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
CHAINS_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
HAVEN_MONTHS_DIRECTORY = STAGE_DIRECTORY / "safe_haven_stage_1_haven_months"      # scripts/threshold_safe_havens.py
OUTPUT_DIRECTORY = STAGE_DIRECTORY
AREA_RADIUS_METERS = 200.0     # a haven belongs to an area when it is this close to a chain start (metres)
HOURS_PER_COUNT = 2.0          # hours per stored hibernation count; the stage-1 product's
OVERWRITE = True               # existing products are replaced as the new ones are complete; False refuses to replace them

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the outputs. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.seed_haven_areas")


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF (its grid, CRS and pixel size)"),
    Option("CHAINS_PATH", "path", "the final stage-4 table"),
    Option("HAVEN_MONTHS_DIRECTORY", "optional_path", "the monthly haven images; none: safe_haven_stage_1_haven_months beside the table"),
    Option("OUTPUT_DIRECTORY", "path", "where the areas' GeoTIFFs and pixel file are written"),
    Option("AREA_RADIUS_METERS", "float", "a haven belongs to an area within this distance (m) of a chain start"),
    Option("HOURS_PER_COUNT", "float", "hours per stored hibernation count (the images')"),
    Option("OVERWRITE", "bool", "replace existing products as the new ones are complete"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now)."""
    return defaults_from(OPTIONS, globals())


def make_seed_haven_areas(*, dem_path, chains_path, haven_months_directory, output_directory, area_radius_meters,
                          hours_per_count, overwrite) -> int:
    """Write the seed haven areas of every month; returns 0."""
    report = generate_seed_haven_areas(
        chains_path, dem_path, output_directory, haven_months_directory=haven_months_directory,
        radius_meters=area_radius_meters, hours_per_count=hours_per_count, overwrite=overwrite, progress=LOG.info)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d months, %d chain starts and %d lone seeds, %s haven pixels (%d months with a haven image) within %g m, "
             "written to %s and %s", report.months, report.chain_starts, report.lone_seeds, f"{report.pixels:,}",
             report.months_with_images, report.radius_meters, report.directory, report.file)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The havens near the start of each chain, month by month. Every option's default is the constant of "
                    "the same name at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the seed haven areas."""
    args = build_parser().parse_args(list(argv))
    options = options_from(args, OPTIONS)
    output_directory = Path(options["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    return run_with_logging(args, default_log_path=output_directory / "safe_haven_stage_4_seed_haven_areas.log",
                            logger=LOG, label="Seed haven areas", procedure=lambda: make_seed_haven_areas(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
