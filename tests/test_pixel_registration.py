"""Pixel-is-area registration of horizon and product grids (review finding M-D)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer

import lunarscout as ls
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout.products import _load_dem


_WKT = 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'
_PROJ4 = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m"
_RADIUS_M = 1_737_400.0


def _georef(origin_x: float, origin_y: float, resolution: float, size: int):
    return ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4=_PROJ4,
        affine_transform=(origin_x, resolution, 0.0, origin_y, 0.0, -resolution),
        width=size,
        height=size,
        pixel_size_x=resolution,
        pixel_size_y=-resolution,
        nodata=None,
    )


def _write(path: Path, values: np.ndarray, georef) -> Path:
    return ls.write_geotiff(path, values.astype(np.float32), georef)


def test_horizon_grid_places_values_at_public_cell_centres(tmp_path: Path) -> None:
    georef = _georef(1000.0, -1000.0, 20.0, 4)
    dem = _load_dem(_write(tmp_path / "dem.tif", np.zeros((4, 4)), georef))[0]

    for column, row in ((0, 0), (3, 1), (2.25, 3.5)):
        expected = georef.pixel_to_projected(column, row)
        np.testing.assert_allclose(dem.pixel_to_crs(column, row), expected, atol=1e-9)
        np.testing.assert_allclose(dem.crs_to_pixel(*expected), (column, row), atol=1e-9)


def test_mixed_resolution_dems_sample_the_same_surface(tmp_path: Path) -> None:
    # A plane sampled at each DEM's cell centres; bilinear interpolation
    # reproduces a plane exactly, so both DEMs must agree everywhere inside.
    def plane(x, y):
        return 0.5 * x - 0.25 * y

    dems = []
    for resolution, size in ((20.0, 100), (100.0, 20)):
        georef = _georef(0.0, 0.0, resolution, size)
        centres = (np.arange(size) + 0.5) * resolution
        x, y = np.meshgrid(centres, -centres)
        dems.append(_load_dem(_write(tmp_path / f"{size}.tif", plane(x, y), georef))[0])

    rng = np.random.default_rng(3)
    for x, y in rng.uniform(60.0, 1940.0, size=(50, 2)) * (1.0, -1.0):
        heights = [dem.elevation(*dem.crs_to_pixel(x, y)) for dem in dems]
        np.testing.assert_allclose(heights, plane(x, y), atol=1e-3)


def _sun_vector_and_corner_elevation(georef, elevation_deg: float):
    """Sun vector at ``elevation_deg`` due north of pixel (0, 0)'s centre.

    Built with pyproj, independently of Lunarscout's frame code.  Also returns
    the sun's elevation as seen from the pixel's corner, the old convention.
    """
    to_lon_lat = Transformer.from_crs(_PROJ4, f"+proj=longlat +R={_RADIUS_M}", always_xy=True)

    def frame(x, y):
        longitude, latitude = np.deg2rad(to_lon_lat.transform(x, y))
        up = np.array((
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ))
        north = np.array((
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        ))
        return up * _RADIUS_M, up, north

    position, up, north = frame(*georef.pixel_to_projected(0, 0))
    elevation = np.deg2rad(elevation_deg)
    sun = position + 1.5e11 * (np.cos(elevation) * north + np.sin(elevation) * up)
    corner, corner_up, _ = frame(*georef.pixel_to_projected(0, 0, anchor="corner"))
    direction = (sun - corner) / np.linalg.norm(sun - corner)
    return sun, np.rad2deg(np.arcsin(direction @ corner_up))


def _sun_elevation(tmp_path: Path, backend: str) -> tuple[float, float]:
    # 20 m pixels north-east of the pole: the half-pixel offset of the cell
    # centre points due north, so it changes the sun's elevation by 4.7e-4 deg.
    georef = _georef(1000.0, -1000.0, 20.0, 1)
    dem_path = _write(tmp_path / "dem.tif", np.zeros((1, 1)), georef)
    horizons = tmp_path / "horizons"
    HorizonTileStore(horizons).write(
        0, 0, 0.0, np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True, valid_width=1, valid_height=1,
    )
    sun, corner_elevation = _sun_vector_and_corner_elevation(georef, 10.0)

    output = ls.generate_sun_elevation(
        dem_path,
        horizons,
        tmp_path / f"sun-{backend}.tif",
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=sun[None, :],
        backend=backend,
    )

    with rasterio.open(output) as dataset:
        return float(dataset.read(1)[0, 0]), corner_elevation


def test_sun_elevation_is_computed_at_the_cell_centre(tmp_path: Path) -> None:
    value, corner_elevation = _sun_elevation(tmp_path, "cpu")

    assert abs(corner_elevation - 10.0) > 4e-4
    assert abs(value - 10.0) < 2e-5


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_cuda_sun_elevation_is_computed_at_the_cell_centre(tmp_path: Path) -> None:
    value, _ = _sun_elevation(tmp_path, "cuda")

    assert abs(value - 10.0) < 2e-5
