#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_cluster_maps.py [--years 2026 2027 --no-overwrite ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; `--help` lists them. make_cluster_maps(...) takes the same values as keyword arguments, so
another script can call it (scripts/run_safe_haven.py runs the whole safe-haven sequence that way).

Bounding boxes of the groups of connected safe havens in each year. A connected haven is a seed that is joined to
another seed by a chain; a seed that is alone is dropped. The connected havens are clustered by connection (two are in
one cluster when a chain of any length joins them) and each cluster gets the bounding box of its havens' pixels. The
boxes are drawn on a raster map, coloured by the number of connected havens in the cluster (classes: 2, 3, 4, 5-6, 7-9,
10-19, 20-49, 50+; one aqua hue, brighter for more).

CONNECT says what a chain is:

* "chains": the chains of stage 4 (scripts/run_safe_haven_stage4.py). A seed has one predecessor at most, so a
  cluster is one root's whole tree: the root and all that descend from it.
* "edges": every stage-3 edge (EDGES_PATH, scripts/run_safe_haven_stage3.py), kept by stage 4 or not. A seed that
  several chains could reach joins their trees, so there are fewer, larger clusters.

A cluster belongs to the UTC year its root's operations start in, as in the yearly chain maps; with "edges" an edge
between seeds whose roots start in different years is not used (the count is in the report). Seeds are the rows of the
final stage-4 table, so the maps need no lighting, floods or routes.

For each year in YEARS (None: every year in which a chain starts), in OUTPUT_DIRECTORY:

* safe_haven_stage_4_cluster_boxes_<year>.tif: an RGBA GeoTIFF on the DEM's grid with the outline of every box,
  OUTLINE_PIXELS wide just outside the box (so the havens inside stay visible) and transparent elsewhere, for QGIS.
  At 20 m a pixel is small: zoom in, or raise OUTLINE_PIXELS, to see the lines on a whole-pole view;
* safe_haven_stage_4_cluster_boxes_<year>.png: the picture for people: the dark hillshade, PSRs and polar graticule of
  the other overview maps with the boxes on top (a box under 4 overview pixels is drawn at 4 so it can be seen);

and, for all the years together, safe_haven_stage_4_cluster_boxes_contact_sheet.png (small multiples with one colour
scale), safe_haven_stage_4_cluster_boxes.csv (one row per cluster: year, havens, the first seed, pixel bounds, CRS
bounds, longest mission; the data behind the pictures) and safe_haven_stage_4_cluster_boxes.report.json. CELL_PATCHES
must be the value stage 2 ran with (2 gives 5.12 km); it and OVERVIEW_PIXELS_PER_CELL set only the grid of the PNGs'
backdrop. The PSR outlines need PSR_IDS_PATH (none: no outlines).

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

from lunarscout._numba_horizon.safe_haven_cluster_maps import run_cluster_maps  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, defaults_from, options_from, run_with_logging,
)


# Edit these paths before running. They match run_safe_haven_stage4.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
CHAINS_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
EDGES_PATH = STAGE_DIRECTORY / "safe_haven_stage_3_edges.csv"     # needed only for CONNECT = "edges"
PSR_IDS_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_ids.tif")        # None: no PSR outlines (scripts/generate_psr_regions.py)
OUTPUT_DIRECTORY = STAGE_DIRECTORY
CONNECT = "chains"              # what joins two havens: "chains" (stage 4's chains) or "edges" (every stage-3 edge)
YEARS = None                     # UTC years to draw; None: every year in the stage-2 table
CELL_PATCHES = 2                 # stage 2's cell size in patches of 128 pixels (2 = 5.12 km)
OVERVIEW_PIXELS_PER_CELL = 16    # the size of a cell in the PNGs; must divide CELL_PATCHES * 128
OUTLINE_PIXELS = 6              # the width of the raster outlines, in DEM pixels
OVERWRITE = True                 # existing files are replaced as the new ones are complete; False refuses to replace them

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the outputs. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.cluster_maps")


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF (its grid, CRS and terrain)"),
    Option("CHAINS_PATH", "path", "the final stage-4 table"),
    Option("EDGES_PATH", "path", "the stage-3 edge table (used when CONNECT is edges)"),
    Option("PSR_IDS_PATH", "optional_path", "the PSR id raster; none: no PSR outlines"),
    Option("OUTPUT_DIRECTORY", "path", "where the maps are written"),
    Option("CONNECT", "str", "what joins two havens: chains or edges", choices=("chains", "edges")),
    Option("YEARS", "ids", "UTC years to draw; give no value for every year in the table"),
    Option("CELL_PATCHES", "int", "stage 2's cell size in patches of 128 pixels"),
    Option("OVERVIEW_PIXELS_PER_CELL", "int", "a cell's size in the PNGs (a divisor of the cell size in pixels)"),
    Option("OUTLINE_PIXELS", "int", "the width of the raster outlines in DEM pixels"),
    Option("OVERWRITE", "bool", "replace existing files as the new ones are complete"),
)


def default_options() -> dict:
    """The options with this file's constants as their values (read now)."""
    return defaults_from(OPTIONS, globals())


def make_cluster_maps(*, dem_path, chains_path, edges_path, psr_ids_path, output_directory, connect, years, cell_patches,
                      overview_pixels_per_cell, outline_pixels, overwrite) -> int:
    """Write the cluster-box maps; returns 0. ``cell_patches`` must be stage 2's."""
    started = clock.monotonic()
    report = run_cluster_maps(
        output_directory, chains_path=chains_path, dem_path=dem_path, edges_path=edges_path, psr_ids_path=psr_ids_path,
        connect=connect, years=years, cell_patches=cell_patches, overview_pixels_per_cell=overview_pixels_per_cell,
        outline_pixels=outline_pixels, overwrite=overwrite)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%s clusters of %s connected havens (of %s seeds), joined by %s", f"{report.clusters:,}",
             f"{report.connected_havens:,}", f"{report.seeds:,}", connect)
    for year, entry in report.years.items():
        LOG.info("%s: %s clusters, %s connected havens, the largest cluster has %s", year, f"{entry['clusters']:,}",
                 f"{entry['connected_havens']:,}", f"{entry['largest_cluster']:,}")
    LOG.info("%d years written to: %s in %.1f s", len(report.years), output_directory, clock.monotonic() - started)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounding boxes of the groups of connected safe havens in each year, coloured by the number of "
                    "havens in the group. Every option's default is the constant of the same name "
                    "at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the cluster-box maps."""
    args = build_parser().parse_args(list(argv))
    options = options_from(args, OPTIONS)
    output_directory = Path(options["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    return run_with_logging(args, default_log_path=output_directory / "safe_haven_stage_4_cluster_boxes.log", logger=LOG,
                            label="Cluster maps", procedure=lambda: make_cluster_maps(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
