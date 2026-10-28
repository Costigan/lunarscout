#!/usr/bin/env python3
"""Edit the constants below or give them on the command line, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_chain_maps.py [--route-workers 8 --no-raster-routes ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; the flood settings are one flag per field; `--help` lists them. make_chain_maps(...) takes the
same values as keyword arguments, so another script can call it (scripts/run_safe_haven.py runs the whole safe-haven
sequence that way).

Maps of the stage-4 chains (scripts/run_safe_haven_stage4.py): two chain sets, each as a colour image on the DEM's
grid, a copy with dilated lines and a GeoPackage, six files in OUTPUT_DIRECTORY:

* safe_haven_stage_4_predecessor_edges{,_dilated}.tif and .gpkg: one segment from every seed that has a predecessor
  to the seed (every edge of the stage-4 label forest, once), coloured by the longest chain that passes through it,
  so a chain keeps one colour along its length.
* safe_haven_stage_4_longest_chain_per_root{,_dilated}.tif and .gpkg: for every root with descendants, its longest
  chain (the most seeds; ties go to the longer mission, then the lowest seed id), coloured by its length.

The images are for people, not data (the table is the data): RGBA GeoTIFFs with a transparent background. Chain
lengths fall in intervals: chains of 2 seeds, then four intervals cut at the quartiles of the lengths of the chains of
3 or more (the same cut points for both sets). Each interval is a distinct, saturated hue and brighter for longer
chains (violet, rose, orange, yellow, green), so the lines stand out on black or a dark hillshade. Every haven (every
seed, isolated ones too) is a white pixel and route pixels that pass next to a permanently shadowed region (PSR; a PSR
pixel among the eight neighbours, from PSR_IDS_PATH) are aqua. Paint order, bottom to top: routes (longer chains over
shorter ones), aqua, havens. The dilated copy stamps a square of 2 * DILATE_PIXELS + 1 pixels along every segment
(havens one pixel larger); we may drop it after looking at it. The GeoPackages are not dilated and use the same
colours: `predecessor_edges` (one segment per edge) and `chains` (one line through the seeds of each chain, with the
numbers of small, medium and large PSRs it passes) with attributes, and `isolated_seeds` (white points). A GeoPackage
segment is a straight line between the centres of two seed pixels.

In the images every segment is the rover's route (RASTER_ROUTES): stage 3 keeps frontiers, not paths, so the flood
of every predecessor seed is run again from the light store with its frontiers kept (about 100 MB per 28-day flood
in each of ROUTE_WORKERS processes) and the route is traced back from the successor seed's pixel at its before
boundary to the predecessor's pixel at its after boundary. A route is one feasible, time-respecting walk (at most 4
one-pixel moves per 2-hour step, into pixels that are passable and lit), not a shortest one. The flood settings must
be those stage 3 ran with and the passable mask is the one it cached. With RASTER_ROUTES False the images use straight
lines like the GeoPackages, which is quick and needs no light store. The GeoPackages always hold straight lines (a
route would be thousands of short segments).

The run takes minutes with routes (seconds without), uses about 6 GB of memory for a full-resolution image plus the
flood buffers, and writes the files through staging files, so a failure leaves any earlier file untouched. An existing
file is replaced as its new version is complete (OVERWRITE True; False refuses to replace files). It is CPU only and
needs no NVIDIA device. Importing it starts nothing.

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

from lunarscout._numba_horizon.safe_haven_chain_maps import run_chain_maps  # noqa: E402
from lunarscout._numba_horizon.safe_haven_floods import FloodSettings  # noqa: E402
from lunarscout._script_cli import (  # noqa: E402
    Option, add_flood_settings, add_logging_options, add_options, defaults_from, flood_settings_from, options_from,
    run_with_logging,
)


# Edit these paths before running. They match run_safe_haven_stage4.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
CHAINS_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")
PSR_IDS_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_ids.tif")        # None marks no PSR (scripts/generate_psr_regions.py)
PASSABLE_PATH = STAGE_DIRECTORY / "safe_haven_stage_3_passable_mask.npy"      # cached by stage 3
# Must be the settings stage 3 ran with (run_safe_haven_stage3.py).
FLOOD_SETTINGS = FloodSettings(max_days=28.0, moves_per_step=4, connectivity=8, max_slope_deg=15.0)
RASTER_ROUTES = True    # False: the rasters use straight lines (quick, no light store needed)
ROUTE_WORKERS = 16
OUTPUT_DIRECTORY = STAGE_DIRECTORY
DILATE_PIXELS = 2       # the dilated rasters' lines are 2 * DILATE_PIXELS + 1 pixels wide
OVERWRITE = True        # existing files are replaced as the new ones are complete; False refuses to replace them
PROGRESS_EVERY_SEEDS = 500

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the outputs. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.chain_maps")


OPTIONS = (
    Option("DEM_PATH", "path", "the DEM GeoTIFF (its grid and CRS)"),
    Option("CHAINS_PATH", "path", "the stage-4 table"),
    Option("OUTPUT_DIRECTORY", "path", "where the six files are written"),
    Option("DILATE_PIXELS", "int", "the dilated images' lines are 2 * this + 1 pixels wide"),
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


def make_chain_maps(*, dem_path, chains_path, output_directory, dilate_pixels, overwrite, light_store_path,
                    psr_ids_path, passable_path, raster_routes, route_workers, progress_every_seeds,
                    flood_settings) -> int:
    """Write the six chain-map files; returns 0. ``flood_settings`` must be stage 3's."""
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % progress_every_seeds and done != total:
            return
        elapsed = clock.monotonic() - started
        LOG.info("Routes: %d/%d predecessor seeds, %.1f min elapsed, about %.1f min left (rate from this run only)",
                 done, total, elapsed / 60, elapsed / done * (total - done) / 60)

    report = run_chain_maps(
        output_directory, chains_path=chains_path, dem_path=dem_path, dilate_pixels=dilate_pixels, overwrite=overwrite,
        light_store_path=light_store_path if raster_routes else None, passable_path=passable_path,
        flood_settings=flood_settings, workers=route_workers, progress=progress, psr_ids_path=psr_ids_path)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d predecessor edges, %d longest chains, %d isolated seeds; six files written to: %s (raster lines: %s)",
             report.predecessor_edges, report.chains, report.isolated_seeds, output_directory, report.raster_lines)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The chain maps: colour images and GeoPackages of the stage-4 chains. Every option's default is "
                    "the constant of the same name at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_flood_settings(parser, FLOOD_SETTINGS)
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the chain maps."""
    args = build_parser().parse_args(list(argv))
    options = {**options_from(args, OPTIONS), "flood_settings": flood_settings_from(args, FLOOD_SETTINGS)}
    output_directory = Path(options["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    return run_with_logging(args, default_log_path=output_directory / "safe_haven_stage_4_chain_maps.log", logger=LOG,
                            label="Chain maps", procedure=lambda: make_chain_maps(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
