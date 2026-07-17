from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.lightmap import compute_lightmap_patch_reference
from lunarscout._numba_horizon.lightmap_pipeline import run_lightmap_product
from lunarscout._numba_horizon.psr import _pixel_frame
from lunarscout.georeference import GeoReference


def _dem(width: int = 1, height: int = 1) -> DemGrid:
    return DemGrid(
        np.zeros((height, width), dtype=np.float32),
        np.asarray((1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0), dtype=np.float64),
        ProjectionParameters(
            radius_m=1_737_400.0,
            latitude_origin_rad=-np.pi / 2.0,
            longitude_origin_rad=0.0,
            scale=1.0,
            false_easting_m=0.0,
            false_northing_m=0.0,
        ),
    )


def _georef(width: int, height: int) -> GeoReference:
    return GeoReference(
        projection_wkt='PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
        width=width,
        height=height,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=None,
    )


def _position_for_local_angle(
    dem: DemGrid, azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    rotation, translation = _pixel_frame(dem, 0, 0)
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    local = np.asarray(
        (
            np.sin(azimuth) * np.cos(elevation),
            np.cos(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ),
        dtype=np.float64,
    )
    return (local * 150_000_000_000.0 - translation) @ rotation.T


def test_lightmap_reference_encodes_full_half_and_zero_solar_disk() -> None:
    dem = _dem()
    horizons = np.zeros((1, 1, AZIMUTH_COUNT), dtype=np.float32)
    vectors = np.stack(
        (
            _position_for_local_angle(dem, 0.0, 1.0),
            _position_for_local_angle(dem, 0.0, 0.0),
            _position_for_local_angle(dem, 0.0, -1.0),
        )
    )

    actual = compute_lightmap_patch_reference(
        dem,
        horizons,
        vectors,
        tile_y=0,
        tile_x=0,
        valid_height=1,
        valid_width=1,
    )

    np.testing.assert_array_equal(actual[:, 0, 0], (255, 127, 0))


def test_lightmap_pipeline_writes_timestamp_bands_and_invalid_patch(tmp_path: Path) -> None:
    dem = _dem(width=129, height=1)
    horizon_store = HorizonTileStore(tmp_path / "horizons")
    horizon_store.write(
        0,
        0,
        0.0,
        np.zeros((128, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=128,
        valid_height=1,
    )
    times = (
        datetime(2027, 1, 1, tzinfo=timezone.utc),
        datetime(2027, 1, 1, 6, tzinfo=timezone.utc),
    )
    vectors = np.stack(
        (
            _position_for_local_angle(dem, 0.0, 1.0),
            _position_for_local_angle(dem, 0.0, -1.0),
        )
    )
    output = tmp_path / "lightmap.tif"

    result = run_lightmap_product(
        dem=dem,
        georef=_georef(129, 1),
        horizon_store=horizon_store,
        output_path=output,
        times_utc=times,
        sun_vectors_m=vectors,
        invalid_value=7,
    )

    assert result == output
    with rasterio.open(output) as dataset:
        assert dataset.count == 2
        assert dataset.profile["interleave"] == "band"
        assert dataset.tags(1)["TIMESTAMP_UTC"] == "2027-01-01T00:00:00.000000Z"
        assert dataset.tags(2)["TIMESTAMP_UTC"] == "2027-01-01T06:00:00.000000Z"
        np.testing.assert_array_equal(dataset.read(1)[0, :128], 255)
        np.testing.assert_array_equal(dataset.read(2)[0, :128], 0)
        np.testing.assert_array_equal(dataset.read()[:, 0, 128], (7, 7))
        np.testing.assert_array_equal(dataset.dataset_mask()[0, :128], 255)
        assert dataset.dataset_mask()[0, 128] == 0
