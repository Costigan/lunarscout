#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/add_pixel_coordinates.py

Adds the columns x_m and y_m, the centre of the pixel in the DEM's CRS (projected metres), to an existing CSV
table that names pixels by raster `row` and `column`: the stage-2 seed table
(scripts/run_safe_haven_stage2.py) or the final stage-4 table (scripts/run_safe_haven_stage4.py) of tables made
before they carried the coordinates. The new columns go right after `column`. A table that already has x_m and
y_m gets them overwritten in place, so it is safe to run again, for example after the DEM was re-georeferenced.
Every other column is copied unchanged and the table is replaced only when the whole new table is written (the
original stays if the run fails). With OUTPUT_PATH set the table is written there and TABLE_PATH is left alone.
DEM_PATH supplies only the geotransform (its header is read, not its pixels), so this takes seconds, uses the CPU
only and needs no NVIDIA device. Importing it starts nothing.

Progress and status messages are logged: to the console, colored with rich on a terminal when
`pip install "lunarscout[console]"` is installed and plain when piped, and to a plain-text log
file next to the table (LOG_PATH, COLOR and LOG_LEVEL below).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.pixel_coordinates import add_coordinate_columns, dem_geo_transform  # noqa: E402
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402


# Edit these before running. The DEM must be the one the table's rows and columns refer to.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
TABLE_PATH = Path("/e/lunar_analyst_scenarios/70south/safe_havens/safe_haven_stage_4_chains.csv")
OUTPUT_PATH = None      # None replaces TABLE_PATH; set a path to write a new table instead

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file. LOG_PATH None writes next to the table.
# COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.add_pixel_coordinates")


def _run() -> int:
    transform = dem_geo_transform(DEM_PATH)
    target = TABLE_PATH if OUTPUT_PATH is None else OUTPUT_PATH
    rows = add_coordinate_columns(TABLE_PATH, transform, output_path=OUTPUT_PATH)
    LOG.info("x_m and y_m written for %d rows of %s", rows, target)
    return 0


def main() -> int:
    log_path = LOG_PATH if LOG_PATH is not None else TABLE_PATH.with_name(TABLE_PATH.name + ".coordinates.log")
    setup_script_logging(log_path, color=COLOR, level=LOG_LEVEL)
    try:
        return _run()
    except Exception as error:
        LOG.error("Adding coordinates stopped: %s: %s", type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
