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
#: Optional Earth-side corner: a -9 degree horizon keeps Earth 7 degrees or more above it
#: through every cycle, so these pixels never have a real outage.
EARTHSIDE = (slice(14, 20), slice(0, 6))
EARTHSIDE_HORIZON = -9.0


def georef():
    return ls.GeoReference(
        projection_wkt=WKT, projection_proj4=PROJ4,
        affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
        width=SIZE, height=SIZE, pixel_size_x=1.0, pixel_size_y=-1.0, nodata=None,
    )


#: A ridge seen to the north-east (azimuth 15 degrees) at the Earth-side corner, used with
#: ``earth_vectors_ridge``: Earth's azimuth swings +-20 degrees through each cycle, so it
#: crosses the ridge twice, with the margin dipping there (to about 3.3 degrees, never below
#: the 2 degree threshold) while Earth is still high.  The margin's minimum is then at the
#: ridge, but the minimum of Earth's elevation above a flat horizon is about 2.5 days later.
RIDGE_AZIMUTH_DEG = 15.0
RIDGE_HEIGHT_DEG = 5.0
RIDGE_WIDTH_DEG = 6.0
AZIMUTH_SWING_DEG = 20.0
AZIMUTH_PHASE = -0.9


def ridge_horizon() -> np.ndarray:
    azimuth = np.arange(AZIMUTH_COUNT) * 0.25
    offset = (azimuth - RIDGE_AZIMUTH_DEG + 180.0) % 360.0 - 180.0
    return (EARTHSIDE_HORIZON + RIDGE_HEIGHT_DEG * np.exp(-((offset / RIDGE_WIDTH_DEG) ** 2))).astype(np.float32)


def horizon_map(earthside: bool = False) -> np.ndarray:
    horizon = np.zeros((SIZE, SIZE), dtype=np.float32)
    horizon[CLUSTER] = LOW_HORIZON
    horizon[OUTLIER] = LOW_HORIZON
    if earthside:
        horizon[EARTHSIDE] = EARTHSIDE_HORIZON
    return horizon


def write_inputs(directory: Path, *, earthside: bool = False, ridge: bool = False) -> tuple[Path, Path]:
    dem_path = directory / "dem.tif"
    ls.write_geotiff(dem_path, np.zeros((SIZE, SIZE), dtype=np.float32), georef())
    horizons = np.repeat(horizon_map(earthside or ridge).reshape(-1, 1), AZIMUTH_COUNT, axis=1)
    if ridge:
        corner = np.zeros((SIZE, SIZE), dtype=bool)
        corner[EARTHSIDE] = True
        horizons[corner.reshape(-1)] = ridge_horizon()
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


def _body(elevation_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    dem = DemGrid(
        np.zeros((1, 1), dtype=np.float32),
        np.asarray((1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0)),
        ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0),
    )
    rotation, translation = _pixel_frame(dem, 0, 0)
    angle = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)
    local = np.asarray((np.sin(azimuth) * np.cos(angle), np.cos(azimuth) * np.cos(angle), np.sin(angle)))
    return (local * 150_000_000_000.0 - translation) @ rotation.T


def sun_vectors(stamps) -> np.ndarray:
    return np.stack([_body(value) for value in sun_elevation(stamps)])


def earth_vectors(stamps) -> np.ndarray:
    return np.stack([_body(value) for value in earth_elevation(stamps)])


def earth_azimuth(stamps) -> np.ndarray:
    return AZIMUTH_SWING_DEG * np.sin(2.0 * np.pi * days(stamps) / 27.3 + AZIMUTH_PHASE)


def earth_vectors_ridge(stamps) -> np.ndarray:
    """Earth as ``earth_vectors`` has it, with its azimuth swinging through each cycle."""
    return np.stack([
        _body(elevation, azimuth)
        for elevation, azimuth in zip(earth_elevation(stamps), earth_azimuth(stamps))
    ])


def stage_one(directory: Path, *, earthside: bool = False, ridge: bool = False, backend="cpu", **options):
    """Write the scene and its stage-1 products; return (dem path, horizons path, products, times)."""
    dem_path, horizons_path = write_inputs(directory, earthside=earthside, ridge=ridge)
    times = ls.times(START, STOP, step_hours=STEP_HOURS)
    stamps = times.values
    earth = earth_vectors_ridge if ridge else earth_vectors
    products = ls.generate_safe_haven_products(
        dem_path, horizons_path, directory / "stage1",
        times=times, sun_vectors_m=sun_vectors(stamps), earth_vectors_m=earth(stamps),
        backend=backend, **options,
    )
    return dem_path, horizons_path, products, stamps
