#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/generate_psr_regions.py

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

import logging
import sys
import time as clock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.safe_haven_psr import PSR_VALUE, make_psr_products, read_region_table  # noqa: E402
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402

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


def _run() -> int:
    started = clock.monotonic()
    regions, psr_pixels = make_psr_products(
        PSR_PATH, IDS_PATH, TABLE_PATH, psr_value=PSR_VALUE_IN_MAP, connectivity=CONNECTIVITY, overwrite=OVERWRITE)
    table = read_region_table(TABLE_PATH)
    LOG.info("%d PSR regions from %d PSR pixels (%.1f km2 in all), largest %.2f km2, median %.0f m2; %.0f s",
             regions, psr_pixels, table.area_m2.sum() / 1e6, table.area_m2.max() / 1e6,
             float(sorted(table.area_m2)[len(table) // 2]), clock.monotonic() - started)
    LOG.info("Ids written to: %s", IDS_PATH)
    LOG.info("Region table written to: %s", TABLE_PATH)
    return 0


def main() -> int:
    log_path = LOG_PATH if LOG_PATH is not None else TABLE_PATH.with_name(TABLE_PATH.name + ".log")
    setup_script_logging(log_path, color=COLOR, level=LOG_LEVEL)
    try:
        return _run()
    except Exception as error:
        LOG.error("PSR regions stopped: %s: %s", type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
