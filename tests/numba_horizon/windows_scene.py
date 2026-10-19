"""A small synthetic scene for the stage-2 tests: a 20 x 20 DEM whose pixels differ
only by horizon, with Sun and Earth elevation curves defined as functions of time.

Pixels with a -3 degree horizon (the cluster and one outlier) see the Sun through
most of each night, so their shadow is shorter than the 72 hour survival limit;
the rest of the DEM, at 0 degrees, sits in shadow for about 93 hours.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import lunarscout as ls
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.psr import _pixel_frame

WKT = 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'
PROJ4 = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m"
SIZE = 20
EPOCH = np.datetime64("2027-01-01T00:00:00", "us")
START, STOP, STEP_HOURS = "2027-01-01T00:00:00Z", "2027-04-30T18:00:00Z", 6.0
CLUSTER = (slice(4, 12), slice(4, 12))
OUTLIER = (17, 17)
LOW_HORIZON = -3.0


def georef():
    return ls.GeoReference(
        projection_wkt=WKT, projection_proj4=PROJ4,
        affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
        width=SIZE, height=SIZE, pixel_size_x=1.0, pixel_size_y=-1.0, nodata=None,
    )


def horizon_map() -> np.ndarray:
    horizon = np.zeros((SIZE, SIZE), dtype=np.float32)
    horizon[CLUSTER] = LOW_HORIZON
    horizon[OUTLIER] = LOW_HORIZON
    return horizon


def write_inputs(directory: Path) -> tuple[Path, Path]:
    dem_path = directory / "dem.tif"
    ls.write_geotiff(dem_path, np.zeros((SIZE, SIZE), dtype=np.float32), georef())
    horizons = np.repeat(horizon_map().reshape(-1, 1), AZIMUTH_COUNT, axis=1)
    horizons_path = directory / "horizons"
    HorizonTileStore(horizons_path).write(
        0, 0, 0.0, horizons, compress=True, valid_width=SIZE, valid_height=SIZE
    )
    return dem_path, horizons_path


def days(stamps) -> np.ndarray:
    return (np.asarray(stamps).astype("datetime64[us]") - EPOCH) / np.timedelta64(1, "D")


def sun_elevation(stamps) -> np.ndarray:
    """Dips below the horizon every 29.5 days: -4 degrees at the bottom."""
    t = days(stamps)
    elevation = np.full(t.shape, 5.0)
    for k in range(-1, 6):
        elevation -= 9.0 * np.exp(-(((t - (16.5 + 29.5 * k)) / 2.5) ** 2))
    return elevation


def earth_elevation(stamps) -> np.ndarray:
    return 6.0 + 8.0 * np.sin(2.0 * np.pi * days(stamps) / 27.3 + 0.9)


def _body(elevation_deg: float) -> np.ndarray:
    dem = DemGrid(
        np.zeros((1, 1), dtype=np.float32),
        np.asarray((1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0)),
        ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0),
    )
    rotation, translation = _pixel_frame(dem, 0, 0)
    angle = np.deg2rad(elevation_deg)
    local = np.asarray((0.0, np.cos(angle), np.sin(angle)))
    return (local * 150_000_000_000.0 - translation) @ rotation.T


def sun_vectors(stamps) -> np.ndarray:
    return np.stack([_body(value) for value in sun_elevation(stamps)])


def earth_vectors(stamps) -> np.ndarray:
    return np.stack([_body(value) for value in earth_elevation(stamps)])


def stage_one(directory: Path, **options):
    """Write the scene and its stage-1 products; return (dem path, horizons path, products, times)."""
    dem_path, horizons_path = write_inputs(directory)
    times = ls.times(START, STOP, step_hours=STEP_HOURS)
    stamps = times.values
    products = ls.generate_safe_haven_products(
        dem_path, horizons_path, directory / "stage1",
        times=times, sun_vectors_m=sun_vectors(stamps), earth_vectors_m=earth_vectors(stamps),
        backend="cpu", **options,
    )
    return dem_path, horizons_path, products, stamps
