#!/usr/bin/env python3
"""Generate the synthetic 256×256 DEM and its horizon tiles.

Creates a south-polar stereographic DEM with a bowl, cone, bump, and many
small craters.  Generates four 128×128 compressed horizon tiles.  Requires a GPU.

Output: dem.tif and horizons/ under the output directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import lunarscout as ls
import numpy as np


_WKT = (
    'PROJCS["ESRI:103878",'
    'GEOGCS["Moon_2000",DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["Meter",1]]'
)
_PROJ4 = (
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+R=1737400 +units=m +no_defs"
)

WIDTH = 256
HEIGHT = 256
PIXEL_SIZE = 10.0
ORIGIN_X = -1280.0
ORIGIN_Y = 1280.0
OBSERVER_HEIGHT_M = 0.0


def _add_crater(dem, rows, cols, cy, cx, radius, depth, rng=None):
    """Subtract a crater depression (Gaussian dip or parabolic bowl)."""
    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    if rng is not None and rng.uniform() < 0.5:
        profile = np.exp(-0.5 * (dist / (radius * 0.5)) ** 2)
    else:
        profile = np.maximum(0.0, 1.0 - dist / radius) ** 2
    dem -= depth * profile


def _add_cone(dem, rows, cols, cy, cx, radius, height):
    """Add a parabolic peak."""
    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    profile = np.maximum(0.0, 1.0 - dist / radius) ** 2
    dem += height * profile


def build_synthetic_dem() -> np.ndarray:
    rows, cols = np.indices((HEIGHT, WIDTH), dtype=np.float64)

    dem = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

    # Bowl (large crater) in the upper-left quadrant -- Gaussian dip
    dist_bowl = np.sqrt((rows - 48) ** 2 + (cols - 48) ** 2)
    dem -= 40.0 * np.exp(-0.5 * (dist_bowl / 30.0) ** 2)

    # Cone/peak in lower-right quadrant
    _add_cone(dem, rows, cols, cy=180, cx=190, radius=15, height=50)

    # Small bump near top-right
    _add_cone(dem, rows, cols, cy=40, cx=200, radius=20, height=25)

    # Scattered small craters for realistic lightmap shadows
    rng = np.random.Generator(np.random.PCG64(seed=42))
    margin = 7
    for _ in range(200):
        cy_c = rng.uniform(margin, HEIGHT - margin)
        cx_c = rng.uniform(margin, WIDTH - margin)
        radius = rng.uniform(2.0, 10.0)
        depth = rng.uniform(1.0, 3.0)
        _add_crater(dem, rows, cols, cy=cy_c, cx=cx_c,
                    radius=radius, depth=depth, rng=rng)

    dem[0, 0] = -9999.0

    return dem.astype(np.float32)


def build_georef() -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4=_PROJ4,
        affine_transform=(ORIGIN_X, PIXEL_SIZE, 0.0, ORIGIN_Y, 0.0, -PIXEL_SIZE),
        width=WIDTH,
        height=HEIGHT,
        pixel_size_x=PIXEL_SIZE,
        pixel_size_y=-PIXEL_SIZE,
        nodata=-9999.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write dem.tif and horizons/.",
    )
    parser.add_argument(
        "--compress", action="store_true", default=True,
        help="Generate compressed .cbin horizons (default).",
    )
    args = parser.parse_args()

    out = args.output_dir.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    dem = build_synthetic_dem()
    georef = build_georef()
    dem_path = out / "dem.tif"

    print(f"Writing DEM: {dem_path}  ({WIDTH}×{HEIGHT} {dem.dtype})")
    ls.write_geotiff(str(dem_path), dem, georef)
    print(f"  min={dem[dem != -9999.0].min():.1f}  max={dem.max():.1f}")

    horizons_dir = out / "horizons"
    print(f"\nGenerating horizons -> {horizons_dir}")

    ls.generate_horizons(
        str(horizons_dir),
        [str(dem_path)],
        observer_height_m=OBSERVER_HEIGHT_M,
        compress=args.compress,
        verbose=True,
    )

    # Verify
    scenario = ls.open_scenario(out)
    tiles = sorted(horizons_dir.rglob("*.cbin"))
    print(f"\nGenerated {len(tiles)} horizon tile(s):")
    for tile in tiles:
        size = tile.stat().st_size
        print(f"  {tile.relative_to(out)}  ({size:,} bytes)")

    # Read and validate the first tile
    horizon = scenario.horizon_for_pixel(x=0, y=0, observer_height_decimeters=0)
    if horizon is not None:
        print(f"\nFirst tile horizon shape: {horizon.shape}, dtype={horizon.dtype}")
        print(f"  min={horizon.min():.3f}  max={horizon.max():.3f}")
        assert horizon.shape == (1440,), f"Expected (1440,), got {horizon.shape}"
        print("  Validation: OK")
    else:
        print("\nWARNING: Could not read horizon at (0,0)")


if __name__ == "__main__":
    sys.exit(main() or 0)
