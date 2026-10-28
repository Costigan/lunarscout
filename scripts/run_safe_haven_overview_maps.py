#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_overview_maps.py [--years 2026 2027 --no-overwrite ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; `--help` lists them. make_overview_maps(...) takes the same values as keyword arguments, so
another script can call it (scripts/run_safe_haven.py runs the whole safe-haven sequence that way).

Overview maps of where the safe havens are around the pole in each year. The persistence of a 5.12 km cell in a year is
the number of months of the year (0 to 12) in which the cell has at least one valid seed in the stage-2 table
(scripts/run_safe_haven_stage2.py). It needs only that table and the DEM's header and pixels (for the backdrop): no
lighting, floods or routes. A count of seeds would mostly show stage 2's cap of five seeds per cell and month, so the
maps show how many months a cell has havens instead.

For each year in YEARS (None: every year in the table), in OUTPUT_DIRECTORY:

* safe_haven_stage_2_persistence_<year>.tif: the data, a one-band uint8 GeoTIFF on the cell grid (value = months, 0
  = none) in the DEM's CRS, for QGIS;
* safe_haven_stage_2_persistence_<year>.png: the picture for people: a dark hillshade of the DEM, the PSRs, a polar
  graticule (parallels every 2 degrees, meridians every 45) and a square per cell in one blue hue, brighter for more
  months, with the numbers of cells and seeds;

and, for all the years together, safe_haven_stage_2_persistence_contact_sheet.png (small multiples with the same colour
scale in every panel) and safe_haven_stage_2_persistence.report.json (per year: seeds, cells with seeds, the number of
cells with each number of months). CELL_PATCHES must be the value stage 2 ran with (a cell is CELL_PATCHES * 128
pixels; 2 gives 5.12 km); OVERVIEW_PIXELS_PER_CELL is the size of a cell in the PNGs and must divide it (16 gives
PNGs of about 1,900 pixels across the pole). The PSR outlines need PSR_IDS_PATH (none: no outlines).

The run reads the DEM once (about a minute), then draws each year in seconds, and writes the files through staging
files, so a failure leaves any earlier file untouched. An existing file is replaced as its new version is complete
(OVERWRITE True; False refuses to replace files). It needs matplotlib (`pip install "lunarscout[plot]"`), is CPU only
and needs no NVIDIA device. Importing it starts nothing.

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

from lunarscout._numba_horizon.safe_haven_overview import run_overview_maps  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, defaults_from, options_from, run_with_logging,
)


# Edit these paths before running. They match run_safe_haven_stage2.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
SEEDS_PATH = STAGE_DIRECTORY / "safe_haven_stage_2_plan_seeds.csv"
PSR_IDS_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_ids.tif")        # None: no PSR outlines (scripts/generate_psr_regions.py)
OUTPUT_DIRECTORY = STAGE_DIRECTORY
YEARS = None                     # UTC years to draw; None: every year in the stage-2 table
CELL_PATCHES = 2                 # stage 2's cell size in patches of 128 pixels (2 = 5.12 km)
OVERVIEW_PIXELS_PER_CELL = 16    # the size of a cell in the PNGs; must divide CELL_PATCHES * 128
OVERWRITE = True                 # existing files are replaced as the new ones are complete; False refuses to replace them

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the outputs. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.overview_maps")


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF (its grid, CRS and terrain)"),
    Option("SEEDS_PATH", "path", "the stage-2 seed table"),
    Option("PSR_IDS_PATH", "optional_path", "the PSR id raster; none: no PSR outlines"),
    Option("OUTPUT_DIRECTORY", "path", "where the maps are written"),
    Option("YEARS", "ids", "UTC years to draw; give no value for every year in the table"),
    Option("CELL_PATCHES", "int", "stage 2's cell size in patches of 128 pixels"),
    Option("OVERVIEW_PIXELS_PER_CELL", "int", "a cell's size in the PNGs (a divisor of the cell size in pixels)"),
    Option("OVERWRITE", "bool", "replace existing files as the new ones are complete"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now)."""
    return defaults_from(OPTIONS, globals())


def make_overview_maps(*, dem_path, seeds_path, psr_ids_path, output_directory, years, cell_patches,
                       overview_pixels_per_cell, overwrite) -> int:
    """Write the persistence maps; returns 0. ``cell_patches`` must be stage 2's."""
    started = clock.monotonic()
    report = run_overview_maps(
        output_directory, seeds_path=seeds_path, dem_path=dem_path, psr_ids_path=psr_ids_path, years=years,
        cell_patches=cell_patches, overview_pixels_per_cell=overview_pixels_per_cell, overwrite=overwrite)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    for year, entry in report.years.items():
        LOG.info("%s: %s seeds, %s of %s cells have seeds in at least one month (mean %.1f months)", year,
                 f"{entry['seeds']:,}", f"{entry['cells_with_seeds']:,}", f"{report.cell_shape[0] * report.cell_shape[1]:,}",
                 entry["mean_months_of_cells_with_seeds"])
    LOG.info("%d years written to: %s in %.1f s", len(report.years), output_directory, clock.monotonic() - started)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overview maps of where the safe havens are around the pole in each year: the months of the year "
                    "in which each 5.12 km cell has a seed. Every option's default is the constant of the same name "
                    "at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the overview maps."""
    args = build_parser().parse_args(list(argv))
    options = options_from(args, OPTIONS)
    output_directory = Path(options["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    return run_with_logging(args, default_log_path=output_directory / "safe_haven_stage_2_persistence.log", logger=LOG,
                            label="Overview maps", procedure=lambda: make_overview_maps(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
