#!/usr/bin/env python3
"""Generate terrain horizon tiles from one or more DEMs.

Requires a compatible NVIDIA GPU.  The first DEM defines the output grid;
additional DEMs extend terrain coverage.  Horizons are written as compressed
.cbin tiles and existing structurally valid tiles are skipped.

Example:
  python examples/15_generate_horizons.py \\
      --primary-dem /data/dem.tif \\
      --output /data/horizons
"""

from __future__ import annotations

import sys
from pathlib import Path

import lunarscout as ls


def _progress(event: ls.ProgressEvent) -> None:
    file_text = ""
    if event.tile_x is not None:
        file_text = f" tile=({event.tile_x},{event.tile_y})"
    print(
        f"  {event.fraction:6.1%} {event.completed}/{event.total} "
        f"{event.stage}{file_text}",
        flush=True,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-dem",
        type=Path,
        required=True,
        help="Primary DEM; defines the output grid.",
    )
    parser.add_argument(
        "--surrounding-dem",
        type=Path,
        action="append",
        default=[],
        help="Additional DEMs for extended terrain coverage (repeatable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for horizon tiles.",
    )
    parser.add_argument(
        "--observer-height-m",
        type=float,
        default=0.0,
        help="Observer height above terrain in metres (default: 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate all tiles even if structurally valid files exist.",
    )
    args = parser.parse_args()

    if not ls.cuda.is_available():
        status = ls.cuda.status()
        print(
            f"ERROR: A compatible NVIDIA CUDA device is required.\n"
            f"       {status.reason}\n"
            f"       Install lunarscout[cuda] on a machine with a supported GPU.",
            file=sys.stderr,
        )
        return 1

    dem_paths = [str(args.primary_dem.resolve())] + [
        str(d.resolve()) for d in args.surrounding_dem
    ]
    for p in dem_paths:
        if not Path(p).is_file():
            print(f"ERROR: DEM not found: {p}", file=sys.stderr)
            return 1

    output = str(args.output.resolve())
    print(f"Generating horizons -> {output}")
    print(f"  DEMs: {dem_paths}")

    result = ls.generate_horizons(
        output,
        dem_paths,
        observer_height_m=args.observer_height_m,
        compress=True,
        overwrite=args.overwrite,
        progress_event_callback=_progress,
    )
    print(f"Horizons written to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
