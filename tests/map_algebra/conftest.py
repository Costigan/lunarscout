from __future__ import annotations

import numpy as np
import pytest
from pyproj import CRS

from lunarscout.georeference import GeoReference

MOON_WKT = (
    'PROJCS["Moon_South_Pole_Stereographic",'
    'GEOGCS["Moon 2000",'
    'DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1]]'
)

MOON_PROJ4 = (
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+R=1737400 +units=m +no_defs +type=crs"
)

MOON_NORTH_WKT = (
    'PROJCS["Moon_North_Pole_Stereographic",'
    'GEOGCS["Moon 2000",'
    'DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",90],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1]]'
)

MOON_NORTH_PROJ4 = (
    "+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+R=1737400 +units=m +no_defs +type=crs"
)


def _georef(
    width: int = 10,
    height: int = 8,
    pixel_x: float = 20.0,
    pixel_y: float = -20.0,
    origin_x: float = 1000.0,
    origin_y: float = 2000.0,
    rotation_x: float = 0.0,
    rotation_y: float = 0.0,
    nodata: int | float | None = None,
    wkt: str | None = None,
    proj4: str | None = None,
) -> GeoReference:
    return GeoReference(
        projection_wkt=wkt or MOON_WKT,
        projection_proj4=proj4 or MOON_PROJ4,
        affine_transform=(origin_x, pixel_x, rotation_x, origin_y, rotation_y, pixel_y),
        width=width,
        height=height,
        pixel_size_x=pixel_x,
        pixel_size_y=pixel_y,
        nodata=nodata,
    )


@pytest.fixture
def north_up_georef() -> GeoReference:
    return _georef(width=10, height=8, pixel_x=20.0, pixel_y=-20.0)


@pytest.fixture
def anisotropic_georef() -> GeoReference:
    return _georef(width=10, height=8, pixel_x=20.0, pixel_y=-10.0)


@pytest.fixture
def rotated_georef() -> GeoReference:
    return _georef(width=10, height=8, pixel_x=20.0, pixel_y=-20.0, rotation_x=2.0, rotation_y=0.0)


@pytest.fixture
def shifted_georef() -> GeoReference:
    return _georef(width=10, height=8, pixel_x=20.0, pixel_y=-20.0, origin_x=1100.0, origin_y=2100.0)


@pytest.fixture
def small_georef() -> GeoReference:
    return _georef(width=4, height=3, pixel_x=20.0, pixel_y=-20.0)


@pytest.fixture
def nodata_georef() -> GeoReference:
    return _georef(width=10, height=8, pixel_x=20.0, pixel_y=-20.0, nodata=-9999.0)


@pytest.fixture
def differing_crs_georef() -> GeoReference:
    return _georef(
        width=10,
        height=8,
        pixel_x=20.0,
        pixel_y=-20.0,
        wkt=MOON_NORTH_WKT,
        proj4=MOON_NORTH_PROJ4,
    )


@pytest.fixture
def partial_coverage_georef() -> GeoReference:
    return _georef(width=6, height=6, pixel_x=20.0, pixel_y=-20.0)


@pytest.fixture
def float_raster_values(north_up_georef: GeoReference) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 100.0, size=(north_up_georef.height, north_up_georef.width)).astype(np.float32)


@pytest.fixture
def int_raster_values(north_up_georef: GeoReference) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(north_up_georef.height, north_up_georef.width), dtype=np.uint8)


@pytest.fixture
def bool_raster_values(north_up_georef: GeoReference) -> np.ndarray:
    rng = np.random.default_rng(43)
    return rng.random(size=(north_up_georef.height, north_up_georef.width)) > 0.5


@pytest.fixture
def all_valid_mask(north_up_georef: GeoReference) -> np.ndarray:
    return np.ones((north_up_georef.height, north_up_georef.width), dtype=np.bool_)


@pytest.fixture
def sparse_valid_mask(north_up_georef: GeoReference) -> np.ndarray:
    mask = np.zeros((north_up_georef.height, north_up_georef.width), dtype=np.bool_)
    mask[2:6, 3:7] = True
    return mask


@pytest.fixture
def masked_float_raster(north_up_georef: GeoReference) -> np.ma.MaskedArray:
    data = np.arange(8 * 10, dtype=np.float32).reshape(8, 10)
    data[0, 0] = -9999.0
    mask = np.zeros((8, 10), dtype=np.bool_)
    mask[0, 1] = True
    mask[1, 2] = True
    return np.ma.array(data, mask=mask)
