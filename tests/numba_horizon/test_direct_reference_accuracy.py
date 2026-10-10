"""Production horizons against the exact-spherical direct reference (review LOW).

``direct_reference_trace`` marches a straight ray from the observer through the
same bilinear terrain and returns exact spherical slopes, with no fitted ray
segment, hierarchy or near-field shortcut.  On rough synthetic terrain the
production GPU horizons measured mean |d| 0.0062 deg, p95 0.038 deg, max 0.092
deg and mean signed -0.0056 deg (2026-10-10); the bounds below leave about 2x
margin while still catching errors the size of review findings H-A and H-B
(0.3 deg mean).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import lunarscout as ls
from lunarscout._numba_horizon.direct_reference import direct_reference_trace
from lunarscout._numba_horizon.file_format import HorizonTileStore, read_horizon_tile
from lunarscout._numba_horizon.geometry import (
    enu_to_moon_matrix,
    inverse_stereographic,
    lat_lon_to_vector,
)
from lunarscout.products import _load_dem

pytestmark = pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)

_WKT = 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'


def test_production_horizons_match_the_exact_spherical_reference(tmp_path: Path) -> None:
    rng = np.random.default_rng(20261010)
    x, y = np.meshgrid(np.arange(256) * 20.0, np.arange(256) * 20.0)
    elevation = 0.05 * x
    for _ in range(60):
        cx, cy = rng.uniform(0.0, 5120.0, 2)
        sigma = rng.uniform(60.0, 500.0)
        elevation += rng.uniform(-150.0, 300.0) * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2)
        )
    georef = ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=(100_000.0, 20.0, 0.0, -100_000.0, 0.0, -20.0),
        width=256,
        height=256,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )
    dem_path = ls.write_geotiff(tmp_path / "dem.tif", elevation.astype(np.float32), georef)
    horizons = ls.generate_horizons(tmp_path / "horizons", [dem_path])
    dem = _load_dem(dem_path)[0]
    tile = read_horizon_tile(HorizonTileStore(horizons).find_existing_path(128, 128, 0.0))

    differences = []
    for column, row in rng.integers(128, 192, size=(30, 2)):
        latitude, longitude = inverse_stereographic(
            *dem.pixel_to_crs(column, row), dem.projection
        )
        observer = lat_lon_to_vector(
            latitude, longitude, dem.projection.radius_m + float(dem.elevation_m[row, column])
        )
        rotation = enu_to_moon_matrix(latitude, longitude)
        for azimuth_index in range(0, 1440, 60):
            azimuth = np.deg2rad(azimuth_index * 0.25)
            direction = np.array((np.sin(azimuth), np.cos(azimuth), 0.0)) @ rotation
            trace = direct_reference_trace(
                dem, observer, direction / np.linalg.norm(direction),
                np.arange(1.0, 8000.0, 5.0),  # quarter-pixel steps past the DEM edge
            )
            reference = np.degrees(np.arctan(trace[:, 4].max()))
            differences.append(tile[row - 128, column - 128, azimuth_index] - reference)

    differences = np.asarray(differences)
    magnitude = np.abs(differences)
    assert magnitude.mean() < 0.015
    assert np.percentile(magnitude, 95) < 0.08
    assert magnitude.max() < 0.25
    assert differences.mean() > -0.015
