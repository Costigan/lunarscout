#!/usr/bin/env python3
"""Generate horizons and a PSR map for the VIPER landing site.

Generates terrain horizon tiles from three concentric DEMs (inner, middle,
outer) and produces a permanent-shadow-region GeoTIFF for the inner DEM.

Requires a compatible NVIDIA GPU.

Example:
  python examples/viper_psr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lunarscout as ls

INNER_DEM = Path("/e/projects/renderscout-data/phase1/dems/viper_v71.tif")
MIDDLE_DEM = Path("/e/projects/renderscout-data/phase1/dems/Site20v2_final_adj_5mpp_surf.tif")
OUTER_DEM = Path("/e/projects/renderscout-data/phase1/dems/ldem_80s_20m_float.tif")
HORIZONS_DIR = Path("/tmp/viper-site/horizons")
PSR_OUTPUT = Path("/tmp/viper-site/viper_psr.tif")


def main() -> int:
    if not ls.cuda.is_available():
        status = ls.cuda.status()
        print(
            f"ERROR: A compatible NVIDIA CUDA device is required.\n"
            f"       {status.reason}\n"
            f"       Install lunarscout[cuda] on a machine with a supported GPU.",
            file=sys.stderr,
        )
        return 1

    dem_paths = [
        str(INNER_DEM.expanduser().resolve()),
        str(MIDDLE_DEM.expanduser().resolve()),
        str(OUTER_DEM.expanduser().resolve()),
    ]
    for p in dem_paths:
        if not Path(p).is_file():
            print(f"ERROR: DEM not found: {p}", file=sys.stderr)
            return 1

    horizons = str(HORIZONS_DIR.expanduser().resolve())
    print(f"Generating horizons -> {horizons}")
    print(f"  DEMs: {dem_paths}")

    ls.generate_horizons(
        horizons,
        dem_paths,
        observer_height_m=0.0,
        overwrite=False,
        verbose=True,
    )
    print(f"Horizons written to {horizons}")

    psr_output = str(PSR_OUTPUT.expanduser().resolve())
    times = ls.times(
        "1970-01-01T00:00:00Z",
        "2044-01-01T00:00:00Z",
        step_hours=6,
    )
    print(f"Generating PSR -> {psr_output}")

    result = ls.generate_psr(
        dem_paths[0],
        horizons,
        psr_output,
        times=times,
        backend="auto",
        overwrite=False,
        verbose=True,
    )
    print(f"PSR written to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
