#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_chain_maps.py

Maps of the stage-4 chains (scripts/run_safe_haven_stage4.py): two chain sets, each as a raster on the DEM's grid,
a raster copy with dilated lines and a GeoPackage, six files in OUTPUT_DIRECTORY:

* safe_haven_stage_4_predecessor_edges{,_dilated}.tif and .gpkg: one straight segment from every seed that has a
  predecessor to the seed (every edge of the stage-4 label forest, once). An edge takes the length of the longest
  chain that passes through it, so a chain keeps one colour along its length.
* safe_haven_stage_4_longest_chain_per_root{,_dilated}.tif and .gpkg: for every root with descendants, its longest
  chain (the most seeds; ties go to the longer mission, then the lowest seed id), all of it at the chain's length.
  Seeds that are alone (a root with no descendants) are dots with the value 1, also as a point layer.

The rasters are three uint16 bands: 1, the chain length in seeds (0 is background; brighter is longer); 2, the month
in which the chain's root starts operations (1 is the first month of the table, 2024-01); 3, the chain's mission
hours (the longest mission at or below an edge for the predecessor-edge raster). Segments are drawn shortest chain
first, so where they cross all three bands come from the longer one. The dilated copy stamps a square of
2 * DILATE_PIXELS + 1 pixels along every segment (dots one pixel larger); the GeoPackage layers are not dilated:
`predecessor_edges` (one segment per edge) and `chains` (one line through the seeds of each chain) with attributes
(seed, root, chain length, mission hours, root start) and `isolated_seeds` (points). A segment is a straight line
between the centres of two seed pixels: a schematic, not the rover's path, and it may cross terrain the rover
cannot.

The run takes a few minutes, uses about 6 GB of memory for a full-resolution raster and writes the files through
staging files, so a failure leaves any earlier file untouched. An existing file is not replaced unless OVERWRITE is
True. It is CPU only and needs no NVIDIA device. Importing it starts nothing.

Progress and status messages are logged: to the console, colored with rich on a terminal when
`pip install "lunarscout[console]"` is installed and plain when piped, and to a plain-text log
file in OUTPUT_DIRECTORY (LOG_PATH, COLOR and LOG_LEVEL below).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.safe_haven_chain_maps import run_chain_maps  # noqa: E402
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402


# Edit these paths before running. They match run_safe_haven_stage4.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
CHAINS_PATH = STAGE_DIRECTORY / "safe_haven_stage_4_chains.csv"
OUTPUT_DIRECTORY = STAGE_DIRECTORY
DILATE_PIXELS = 2       # the dilated rasters' lines are 2 * DILATE_PIXELS + 1 pixels wide
OVERWRITE = False       # existing files are not replaced unless this is True

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the outputs. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.chain_maps")


def _run() -> int:
    report = run_chain_maps(OUTPUT_DIRECTORY, chains_path=CHAINS_PATH, dem_path=DEM_PATH,
                            dilate_pixels=DILATE_PIXELS, overwrite=OVERWRITE)
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    LOG.info("%d predecessor edges, %d longest chains, %d isolated seeds; six files written to: %s",
             report.predecessor_edges, report.chains, report.isolated_seeds, OUTPUT_DIRECTORY)
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
