#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/list_safe_haven_cells.py

Lists the 256 x 256 pixel cells (5.12 km at 20 m) that hold at least one safe-haven
pixel in the stage-1 hibernation product, so you can choose cells for a trial of
scripts/plan_safe_haven_windows.py or scripts/run_safe_haven_stage3.py (the full list for
stage 3 comes from scripts/run_safe_haven_stage2.py, which scans much faster). A haven pixel has a hibernation value at or
below SURVIVAL_LIMIT_HOURS in some month. The script prints a table and a CELLS
line to paste into the planning script. CPU only; it needs the stage-1 products
from scripts/run_safe_haven_stage1.py. Importing it starts no calculation.
"""

from __future__ import annotations

import sys
import time as clock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.safe_haven_windows import (  # noqa: E402
    Stage1Rasters,
    cell_of_coordinate,
    cells_around,
    haven_cells,
)
from lunarscout.products import _load_dem  # noqa: E402


# Edit these before running. They match plan_safe_haven_windows.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
STAGE1_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
STAGE1_PREFIX = "safe_haven_2h"  # must equal NAME_PREFIX in run_safe_haven_stage1.py
HOURS_PER_COUNT = 2.0   # None, with STAGE1_NODATA None, for float products in hours
STAGE1_NODATA = 255
SURVIVAL_LIMIT_HOURS = 72.0
CELL_SIZE = 256         # cell_patches * 128

# To look only near a place, give its (northing, easting) in the DEM's projection, in
# metres, and a radius in cells (1 gives the 3 x 3 cells around it). Set
# AROUND_NORTHING_EASTING_M to None to scan every cell, which reads the whole product.
AROUND_NORTHING_EASTING_M = (117030.0, 71534.0)
AROUND_RADIUS_CELLS = 1
TOP = 30                # rows of the table to print


def main() -> int:
    dem, _georef = _load_dem(DEM_PATH)
    stage1 = Stage1Rasters(
        STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_hibernation_hours.tif",
        STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_joint_operating_total_hours.tif",
        hours_per_count=HOURS_PER_COUNT,
        nodata=STAGE1_NODATA,
    )
    cells = None
    if AROUND_NORTHING_EASTING_M is not None:
        northing, easting = AROUND_NORTHING_EASTING_M
        centre = cell_of_coordinate(dem.geo_transform, easting, northing, CELL_SIZE)
        cells = cells_around(
            centre, AROUND_RADIUS_CELLS, height=dem.height, width=dem.width, cell_size=CELL_SIZE
        )
        print(
            f"Northing, easting {AROUND_NORTHING_EASTING_M} is in cell {centre}; "
            f"scanning {len(cells)} cells."
        )
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % 500 == 0:
            elapsed = clock.monotonic() - started
            print(f"{done}/{total} cells, about {elapsed / done * (total - done) / 60:.1f} min left", flush=True)

    found = haven_cells(
        stage1, survival_limit_hours=SURVIVAL_LIMIT_HOURS, cell_size=CELL_SIZE,
        cells=cells, progress=progress,
    )
    print(f"{len(found)} cells have a haven pixel at or under {SURVIVAL_LIMIT_HOURS:g} h.")
    print("cell_row cell_col months_with_haven haven_pixels")
    for entry in found[:TOP]:
        print(f"{entry['cell_row']:8d} {entry['cell_col']:8d} {entry['months_with_haven']:17d} {entry['haven_pixels']:12d}")
    chosen = [(e["cell_row"], e["cell_col"]) for e in found[:TOP]]
    print(f"CELLS = {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
