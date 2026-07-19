#!/usr/bin/env python3
"""Generate public Python/Numba horizon tiles on a real NVIDIA GPU."""

from __future__ import annotations

from pathlib import Path

import lunarscout as ls


# Edit these paths before running. The first DEM defines the output grid;
# following DEMs extend terrain coverage from local to regional scale.
OUTPUT_DIRECTORY = Path("/tmp/lunarscout_horizons")
DEM_PATHS = [
    Path("/path/to/local/dem.tif"),
    Path("/path/to/regional/dem.tif"),
]


def progress(event: ls.ProgressEvent) -> None:
    file_text = "" if event.tile_x is None else f" tile=({event.tile_x},{event.tile_y})"
    print(
        f"{event.fraction:6.1%} {event.completed}/{event.total} "
        f"{event.stage}{file_text}",
        flush=True,
    )


def main() -> int:
    if not ls.cuda.is_available():
        status = ls.cuda.status()
        raise SystemExit(f"A compatible NVIDIA CUDA device is required: {status.reason}")
    result = ls.generate_horizons(
        OUTPUT_DIRECTORY,
        DEM_PATHS,
        compress=True,
        progress_event_callback=progress,
    )
    print(f"Horizons written to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
