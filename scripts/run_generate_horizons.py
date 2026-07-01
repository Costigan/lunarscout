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

OBSERVER_ELEVATION_METERS = 0.0
SKIP_EXISTING = True
COMPRESS_HORIZONS = False


def progress(event: ls.NativeHorizonProgress) -> None:
    file_text = f" {event.file_name}" if event.file_name else ""
    print(
        f"{event.percent:6.2f}% "
        f"{event.processed_patches}/{event.total_patches} "
        f"{event.stage}: {event.message}{file_text}",
        flush=True,
    )


def main() -> int:
    print("Native status:")
    status = ls.native.status()
    for name, component in status["components"].items():
        print(f"  {name}: available={component.get('available')}")
    if not status["available"]:
        print()
        print("Native runtime is not fully available. Build moonlib first, for example:")
        print("  dotnet build native/moonlib/moonlib.csproj")
        return 2

    print()
    print("Generating horizons...")
    result = ls.GenerateHorizons(
        OUTPUT_DIR,
        DEM_PATHS,
        observer_elevation=OBSERVER_ELEVATION_METERS,
        skip_existing=SKIP_EXISTING,
        compress_horizons=COMPRESS_HORIZONS,
        progress_callback=progress,
    )
    print()
    print(f"Horizon files written to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
