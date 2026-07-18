from __future__ import annotations

from datetime import datetime, timezone
import base64
import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest
import rasterio

from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.cuda_backend import CudaBackendError
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.lightmap import (
    _sun_fraction_reference,
    compute_lightmap_patch_reference,
)
from lunarscout._numba_horizon.lightmap_pipeline import (
    LightmapPipelineCancelled,
    run_lightmap_product,
)
from lunarscout._numba_horizon.lightmap_cuda import LightmapCudaSession
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
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


def test_sun_fraction_reference_matches_csharp_builder_oracle() -> None:
    fixture_path = (
        Path(__file__).parents[1]
        / "data"
        / "numba_horizon"
        / "phase6b_lightmap_csharp.json"
    )
    artifact = json.loads(fixture_path.read_text(encoding="utf-8"))
    for case in artifact["cases"]:
        horizons = np.frombuffer(
            base64.b64decode(case["horizons_float32_base64"]), dtype="<f4"
        )
        for sample in case["samples"]:
            fraction = _sun_fraction_reference(
                horizons, sample["azimuth_deg"], sample["elevation_deg"]
            )
            encoded = np.uint8(np.float32(255.0) * fraction)
            assert int(encoded) == sample["encoded_byte"], (
                case["name"],
                sample,
                float(fraction),
            )


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_lightmap_cuda_batches_match_cpu_reference() -> None:
    dem = _dem(width=3, height=2)
    pixel = np.arange(128 * 128, dtype=np.float32)[:, None]
    azimuth = np.arange(AZIMUTH_COUNT, dtype=np.float32)[None, :]
    horizons = (
        np.float32(0.1)
        + np.float32(0.01) * (pixel % np.float32(17.0))
        + np.float32(0.0001) * (azimuth % np.float32(13.0))
    ).reshape(128, 128, AZIMUTH_COUNT)
    vectors = np.stack(
        tuple(
            _position_for_local_angle(dem, azimuth_deg, elevation_deg)
            for azimuth_deg, elevation_deg in (
                (0.02, 0.18),
                (359.98, 0.18),
                (10.12, 0.45),
                (275.249, 0.52),
                (123.456, -0.2),
            )
        )
    )
    expected = compute_lightmap_patch_reference(
        dem,
        horizons,
        vectors,
        tile_y=0,
        tile_x=0,
        valid_height=2,
        valid_width=3,
    )

    actual = np.stack(
        tuple(
            LightmapCudaSession(time_batch_size=2).iter_patch_tiles(
                dem,
                horizons,
                vectors,
                tile_y=0,
                tile_x=0,
                valid_height=2,
                valid_width=3,
            )
        )
    )

    np.testing.assert_array_equal(actual, expected)


def test_compiled_cpu_batches_match_reference() -> None:
    dem = _dem(width=2, height=2)
    horizons = np.zeros((128, 128, AZIMUTH_COUNT), dtype=np.float32)
    vectors = np.stack(
        tuple(
            _position_for_local_angle(dem, 359.98, elevation)
            for elevation in (1.0, 0.18, 0.0, -0.18, -1.0)
        )
    )
    expected = compute_lightmap_patch_reference(
        dem,
        horizons,
        vectors,
        tile_y=0,
        tile_x=0,
        valid_height=2,
        valid_width=2,
    )

    actual = np.stack(
        tuple(
            LightmapCpuSession(time_batch_size=2).iter_patch_tiles(
                dem,
                horizons,
                vectors,
                tile_y=0,
                tile_x=0,
                valid_height=2,
                valid_width=2,
            )
        )
    )

    np.testing.assert_array_equal(actual, expected)


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
        backend="cpu",
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


def test_auto_backend_falls_back_to_compiled_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import lightmap_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(lightmap_cuda, "LightmapCudaSession", UnavailableCudaSession)
    dem = _dem()
    horizon_store = HorizonTileStore(tmp_path / "horizons")
    horizon_store.write(
        0,
        0,
        0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )
    result = run_lightmap_product(
        dem=dem,
        georef=_georef(1, 1),
        horizon_store=horizon_store,
        output_path=tmp_path / "auto.tif",
        times_utc=("2027-01-01T00:00:00Z",),
        sun_vectors_m=_position_for_local_angle(dem, 0.0, 1.0)[None, :],
        backend="auto",
    )

    with rasterio.open(result) as dataset:
        assert dataset.read(1).item() == 255


def test_interrupted_patch_recomputes_every_band_on_resume(tmp_path: Path) -> None:
    dem = _dem()
    horizon_store = HorizonTileStore(tmp_path / "horizons")
    horizon_store.write(
        0,
        0,
        0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )
    cancelled = threading.Event()

    def interrupted_calculator(*_args, **_kwargs):
        yield np.asarray([[5]], dtype=np.uint8)
        cancelled.set()
        yield np.asarray([[6]], dtype=np.uint8)

    common = dict(
        dem=dem,
        georef=_georef(1, 1),
        horizon_store=horizon_store,
        output_path=tmp_path / "restart.tif",
        times_utc=("2027-01-01T00:00:00Z", "2027-01-01T06:00:00Z"),
        sun_vectors_m=np.stack(
            (
                _position_for_local_angle(dem, 0.0, 1.0),
                _position_for_local_angle(dem, 0.0, -1.0),
            )
        ),
    )
    with pytest.raises(LightmapPipelineCancelled):
        run_lightmap_product(
            **common,
            patch_calculator=interrupted_calculator,
            cancellation_requested=cancelled.is_set,
        )

    journal = json.loads(
        (tmp_path / ".restart.tif.lunarscout-partial.journal.json").read_text()
    )
    assert journal["completed_patches"] == {}

    def resumed_calculator(*_args, **_kwargs):
        yield np.asarray([[9]], dtype=np.uint8)
        yield np.asarray([[10]], dtype=np.uint8)

    result = run_lightmap_product(
        **common,
        patch_calculator=resumed_calculator,
    )
    with rasterio.open(result) as dataset:
        np.testing.assert_array_equal(dataset.read()[:, 0, 0], (9, 10))
