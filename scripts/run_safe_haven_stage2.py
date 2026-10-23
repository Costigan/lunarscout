#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/run_safe_haven_stage2.py

Stage 2 of the safe-haven analysis: scan the stage-1 hibernation product once and write the
list of 5.12 km cells that have at least one haven pixel (hibernation at most
SURVIVAL_LIMIT_HOURS in some month), with the number of months that have one and the number of
haven pixels. Stage 3 (scripts/run_safe_haven_stage3.py) reads this list instead of scanning
the product again. The list is a small JSON file tied to the stage-1 files by size and
modification time, the survival limit and the cell size; stage 3 refuses it if any of them
has changed, and then this script must be run again.

The scan reads the product one strip of cell rows at a time (about 0.75 GB of memory) and is
mostly drive reads at modest CPU load. It is not resumable; if it is interrupted, run it
again. It needs no NVIDIA device. Importing it starts nothing.
"""

from __future__ import annotations

import sys
import time as clock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon.safe_haven_cells import (  # noqa: E402
    scan_cells,
    write_cell_list,
)
from lunarscout._numba_horizon.safe_haven_windows import Stage1Rasters  # noqa: E402


# Edit these paths before running. They match run_generate_safe_havens.py.
STAGE1_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
STAGE1_PREFIX = "safe_haven_degenerate"  # must equal NAME_PREFIX in run_generate_safe_havens.py
# The stage-1 products are bytes of two-hour counts with NODATA 255. Set both to None for
# float products in hours.
HOURS_PER_COUNT = 2.0
STAGE1_NODATA = 255

# Must equal the values stage 3 uses (PARAMETERS.survival_limit_hours and
# PARAMETERS.cell_patches * 128).
SURVIVAL_LIMIT_HOURS = 72.0
CELL_SIZE = 256

CELL_LIST_PATH = STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_haven_cells.json"
OVERWRITE = False       # an existing CELL_LIST_PATH is not replaced unless this is True
TOP = 20                # rows of the table to print


def main() -> int:
    if CELL_LIST_PATH.exists() and not OVERWRITE:
        raise ValueError(f"{CELL_LIST_PATH} exists; set OVERWRITE = True to replace it")
    stage1 = Stage1Rasters(
        STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_hibernation_hours.tif",
        STAGE1_DIRECTORY / f"{STAGE1_PREFIX}_joint_operating_total_hours.tif",
        hours_per_count=HOURS_PER_COUNT,
        nodata=STAGE1_NODATA,
    )
    started = clock.monotonic()

    def progress(done: int, total: int) -> None:
        if done % 10 and done != total:
            return
        elapsed = clock.monotonic() - started
        print(f"Stage 2: {done}/{total} cell rows, {elapsed / 60:.1f} min elapsed, "
              f"about {elapsed / done * (total - done) / 60:.1f} min left", flush=True)

    found = scan_cells(stage1, survival_limit_hours=SURVIVAL_LIMIT_HOURS, cell_size=CELL_SIZE,
                       progress=progress)
    write_cell_list(CELL_LIST_PATH, found, stage1, survival_limit_hours=SURVIVAL_LIMIT_HOURS,
                    cell_size=CELL_SIZE, overwrite=OVERWRITE)
    total_cells = (-(-stage1.height // CELL_SIZE)) * (-(-stage1.width // CELL_SIZE))
    months = sum(entry["months_with_haven"] for entry in found)
    print(f"{len(found)} of {total_cells} cells have a haven pixel at or under "
          f"{SURVIVAL_LIMIT_HOURS:g} h; {months} cell-months in all.", flush=True)
    print("cell_row cell_col months_with_haven haven_pixels")
    for entry in found[:TOP]:
        print(f"{entry['cell_row']:8d} {entry['cell_col']:8d} {entry['months_with_haven']:17d} "
              f"{entry['haven_pixels']:12d}")
    print(f"Cell list written to: {CELL_LIST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
