#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/generate_psr_regions.py [--psr-path psr.tif --overwrite ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; `--help` lists them. run_psr_regions(...) takes the same values as keyword arguments, so another
script can call it (scripts/run_safe_haven.py runs the whole safe-haven sequence that way).

Gives every connected permanently shadowed region (PSR) of a PSR map its own id, and writes two products
that later stages reuse (src/lunarscout/_numba_horizon/safe_haven_psr.py):

* IDS_PATH, an int32 GeoTIFF on the PSR map's grid: id 1, 2, 3, ... on every pixel of each region (numbered
  in raster-scan order of its first pixel) and 0 on every pixel that is not a PSR;
* TABLE_PATH, a CSV with one row per id: the region's size in pixels and in square metres, its bounding
  box in pixels and its centroid in the map's coordinates.

Only PSR pixels are labelled: the map must hold only 0 (not PSR) and PSR_VALUE (PSR), with PSR_VALUE
marking under half of the raster, or the script stops instead of labelling the wrong pixels. Regions are
connected through eight neighbours, the way the safe-haven flood moves. It needs a few GB of memory for
a 30,000 x 30,000 raster and takes a minute or two on one core; no GPU. Importing it starts nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time as clock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.safe_haven_psr import PSR_VALUE, make_psr_products, read_region_table  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, defaults_from, options_from, run_with_logging,
)

# Edit these paths before running.
PSR_PATH = Path("/e/lunar_analyst_scenarios/70south/psr.tif")           # 255 = PSR, 0 = not PSR
IDS_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_ids.tif")
TABLE_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_regions.csv")
PSR_VALUE_IN_MAP = PSR_VALUE
CONNECTIVITY = 8            # 4 or 8
OVERWRITE = False

LOG_PATH = None             # None writes next to TABLE_PATH
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.psr_regions")


OPTIONS = (
    Option("PSR_PATH", "path", "the PSR map: 255 = PSR, 0 = not PSR"),
    Option("IDS_PATH", "path", "the id raster to write"),
    Option("TABLE_PATH", "path", "the region table to write"),
    Option("PSR_VALUE_IN_MAP", "int", "the value that marks a PSR pixel in the map"),
    Option("CONNECTIVITY", "int", "4 or 8 neighbours connect a region", choices=(4, 8)),
    Option("OVERWRITE", "bool", "replace existing products"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now)."""
    return defaults_from(OPTIONS, globals())


def run_psr_regions(*, psr_path, ids_path, table_path, psr_value_in_map, connectivity, overwrite) -> int:
    """Label the PSRs and write the id raster and the region table; returns 0."""
    started = clock.monotonic()
    regions, psr_pixels = make_psr_products(
        psr_path, ids_path, table_path, psr_value=psr_value_in_map, connectivity=connectivity, overwrite=overwrite)
    table = read_region_table(table_path)
    LOG.info("%d PSR regions from %d PSR pixels (%.1f km2 in all), largest %.2f km2, median %.0f m2; %.0f s",
             regions, psr_pixels, table.area_m2.sum() / 1e6, table.area_m2.max() / 1e6,
             float(sorted(table.area_m2)[len(table) // 2]), clock.monotonic() - started)
    LOG.info("Ids written to: %s", ids_path)
    LOG.info("Region table written to: %s", table_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label the permanently shadowed regions. Every option's default is the constant of the same name "
                    "at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the PSR products."""
    args = build_parser().parse_args(list(argv))
    options = options_from(args, OPTIONS)
    table_path = Path(options["table_path"])
    return run_with_logging(args, default_log_path=table_path.with_name(table_path.name + ".log"), logger=LOG,
                            label="PSR regions", procedure=lambda: run_psr_regions(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
