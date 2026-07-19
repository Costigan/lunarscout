#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import lunarscout as ls  # noqa: E402


# Edit these paths before running.
OUTPUT_DIR = Path("/tmp/lunarscout_horizons")
DEM_PATHS = [
    Path("/e/lunar_analyst_scenarios/polar_mosaic/dem.tif"),
    Path("/d/viper/maps/lola/LDEM_80S_20M-2017-06-15-processed.tif"),
]

OBSERVER_HEIGHT_METERS = 0.0
COMPRESS_HORIZONS = True


def progress(event: ls.ProgressEvent) -> None:
    tile_text = (
        ""
        if event.tile_x is None
        else f" tile=({event.tile_x},{event.tile_y})"
    )
    print(
        f"{event.fraction:6.1%} "
        f"{event.completed}/{event.total} "
        f"{event.stage}: {event.message}{tile_text}",
        flush=True,
    )


def main() -> int:
    status = ls.cuda.status()
    print(f"CUDA available: {status.available}")
    if not status.available:
        print(f"CUDA unavailable: {status.reason}")
        return 2

    print()
    print("Generating horizons...")
    result = ls.generate_horizons(
        OUTPUT_DIR,
        DEM_PATHS,
        observer_height_m=OBSERVER_HEIGHT_METERS,
        compress=COMPRESS_HORIZONS,
        progress_event_callback=progress,
    )
    print()
    print(f"Horizon files written to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
