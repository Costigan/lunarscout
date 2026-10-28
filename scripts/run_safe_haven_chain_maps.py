#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_chain_maps.py

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

import json
import logging
import sys
import time as clock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.safe_haven_chain_maps import run_chain_maps  # noqa: E402
from lunarscout._numba_horizon.safe_haven_floods import FloodSettings  # noqa: E402
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402


# Edit these paths before running. They match run_safe_haven_stage4.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
CHAINS_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
LIGHT_STORE_PATH = Path("/e/lunar_analyst_scenarios/70south/light_store")
PSR_IDS_PATH = Path("/e/lunar_analyst_scenarios/70south/psr_ids.tif")        # None marks no PSR (scripts/generate_psr_regions.py)
PASSABLE_PATH = STAGE_DIRECTORY / "safe_haven_stage_3_passable_mask.npy"      # cached by stage 3
# Must be the settings stage 3 ran with (run_safe_haven_stage3.py).
FLOOD_SETTINGS = FloodSettings(max_days=28.0, moves_per_step=4, connectivity=8, max_slope_deg=20.0)
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


def _run() -> int:
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % PROGRESS_EVERY_SEEDS and done != total:
            return
        elapsed = clock.monotonic() - started
        LOG.info("Routes: %d/%d predecessor seeds, %.1f min elapsed, about %.1f min left (rate from this run only)",
                 done, total, elapsed / 60, elapsed / done * (total - done) / 60)

    report = run_chain_maps(
        OUTPUT_DIRECTORY, chains_path=CHAINS_PATH, dem_path=DEM_PATH, dilate_pixels=DILATE_PIXELS, overwrite=OVERWRITE,
        light_store_path=LIGHT_STORE_PATH if RASTER_ROUTES else None, passable_path=PASSABLE_PATH,
        flood_settings=FLOOD_SETTINGS, workers=ROUTE_WORKERS, progress=progress, psr_ids_path=PSR_IDS_PATH)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d predecessor edges, %d longest chains, %d isolated seeds; six files written to: %s (raster lines: %s)",
             report.predecessor_edges, report.chains, report.isolated_seeds, OUTPUT_DIRECTORY, report.raster_lines)
    return 0


def main() -> int:
    log_path = LOG_PATH if LOG_PATH is not None else OUTPUT_DIRECTORY / "safe_haven_stage_4_chain_maps.log"
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    setup_script_logging(log_path, color=COLOR, level=LOG_LEVEL)
    try:
        return _run()
    except Exception as error:
        LOG.error("Chain maps stopped: %s: %s", type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
