#!/usr/bin/env python3
"""Generate the synthetic 256×256 DEM and its horizon tiles.

Creates a south-polar stereographic DEM with a bowl, ridges, and a plateau.
Then generates four 128×128 compressed horizon tiles.  Requires a GPU.

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


def build_synthetic_dem() -> np.ndarray:
    rows, cols = np.indices((HEIGHT, WIDTH), dtype=np.float64)

    # Centre coordinates in pixels
    cy, cx = HEIGHT / 2, WIDTH / 2

    # Base flat terrain
    dem = np.full((HEIGHT, WIDTH), 100.0, dtype=np.float64)

    # Bowl (crater) in the upper-left quadrant
    dist_bowl = np.sqrt((rows - 48) ** 2 + (cols - 48) ** 2)
    bowl_depth = 40.0 * np.exp(-0.5 * (dist_bowl / 30.0) ** 2)
    dem -= bowl_depth

    # Ridge running diagonally through the centre
    ridge_dist = np.abs(
        (rows - cy) * np.cos(np.radians(35))
        + (cols - cx) * np.sin(np.radians(35))
    )
    ridge_height = 30.0 * np.exp(-0.5 * (ridge_dist / 15.0) ** 2)
    # Taper the ridge near edges
    edge_taper = np.exp(
        -0.5 * ((rows - cy) ** 2 + (cols - cx) ** 2) / (HEIGHT * 0.6) ** 2
    )
    dem += ridge_height * edge_taper

    # Cone/peak in lower-right quadrant
    dist_cone = np.sqrt((rows - 180) ** 2 + (cols - 190) ** 2)
    cone_height = 50.0 * np.maximum(0.0, 1.0 - dist_cone / 35.0) ** 2
    dem += cone_height

    # Small bump near top-right
    dist_bump = np.sqrt((rows - 40) ** 2 + (cols - 200) ** 2)
    bump_height = 25.0 * np.exp(-0.5 * (dist_bump / 20.0) ** 2)
    dem += bump_height

    # Medium-frequency roughness
    roughness = (
        4.0 * np.sin(rows * 0.15) * np.cos(cols * 0.12)
        + 3.0 * np.cos(rows * 0.09) * np.sin(cols * 0.11)
        + 2.0 * np.sin((rows + cols) * 0.07)
    )
    dem += roughness

    # Set nodata in one corner pixel for mask exercise
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
