"""A multi-patch synthetic scene for the block-schedule tests: a 300 x 300 DEM (3 x 3
patches, the last row and column partial) with four 8 x 8 clusters of low horizon, one of
them straddling a cell boundary.  The Sun and Earth curves are those of ``windows_scene``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import lunarscout as ls
import windows_scene as base
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, PATCH_SIZE, HorizonTileStore

SIZE = 300
#: (row0, column0) of each 8 x 8 cluster.  The second straddles the boundary of patch
#: rows 0 and 1 (rows 124..131); the last lies in the partial last patch row.
CLUSTERS = [(60, 60), (124, 20), (200, 200), (280, 130)]


def horizon_map() -> np.ndarray:
    horizon = np.zeros((SIZE, SIZE), dtype=np.float32)
    for row, column in CLUSTERS:
        horizon[row:row + 8, column:column + 8] = base.LOW_HORIZON
    return horizon


def georef():
    return ls.GeoReference(
        projection_wkt=base.WKT, projection_proj4=base.PROJ4,
        affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
        width=SIZE, height=SIZE, pixel_size_x=1.0, pixel_size_y=-1.0, nodata=None,
    )


def write_inputs(directory: Path) -> tuple[Path, Path]:
    dem_path = directory / "dem.tif"
    ls.write_geotiff(dem_path, np.zeros((SIZE, SIZE), dtype=np.float32), georef())
    horizon = horizon_map()
    store = HorizonTileStore(directory / "horizons")
    for tile_y in range(0, SIZE, PATCH_SIZE):
        for tile_x in range(0, SIZE, PATCH_SIZE):
            patch = horizon[tile_y:tile_y + PATCH_SIZE, tile_x:tile_x + PATCH_SIZE]
            height, width = patch.shape
            columns = np.repeat(patch.reshape(-1, 1), AZIMUTH_COUNT, axis=1)
            store.write(tile_y, tile_x, 0.0, columns, compress=True, valid_width=width, valid_height=height)
    return dem_path, directory / "horizons"


def stage_one(directory: Path):
    dem_path, horizons_path = write_inputs(directory)
    times = ls.times(base.START, "2027-03-31T18:00:00Z", step_hours=base.STEP_HOURS)
    stamps = times.values
    products = ls.generate_safe_haven_products(
        dem_path, horizons_path, directory / "stage1", times=times,
        sun_vectors_m=base.sun_vectors(stamps), earth_vectors_m=base.earth_vectors(stamps), backend="cpu",
    )
    return dem_path, horizons_path, products, stamps
