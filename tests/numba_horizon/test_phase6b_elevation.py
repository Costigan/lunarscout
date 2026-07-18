from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import threading

import numpy as np
import pytest
import rasterio

from lunarscout._numba_horizon.cuda_backend import CudaBackendError
from lunarscout._numba_horizon.elevation_pipeline import (
    BodyElevationPipelineCancelled,
    run_earth_elevation_product,
    run_sun_elevation_product,
)
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.psr import _pixel_frame
from lunarscout.georeference import GeoReference


UTC = timezone.utc


def _dem() -> DemGrid:
    return DemGrid(
        np.zeros((1, 1), dtype=np.float32),
        np.asarray((1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0), dtype=np.float64),
        ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0),
    )


def _georef() -> GeoReference:
    return GeoReference(
        projection_wkt='PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
        width=1,
        height=1,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=None,
    )


def _position(dem: DemGrid, elevation_deg: float) -> np.ndarray:
    rotation, translation = _pixel_frame(dem, 0, 0)
    elevation = np.deg2rad(elevation_deg)
    local = np.asarray((0.0, np.cos(elevation), np.sin(elevation)))
    return (local * 150_000_000_000.0 - translation) @ rotation.T


def _store(tmp_path: Path, horizon_deg: float = 0.25) -> HorizonTileStore:
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(
        0,
        0,
        0.0,
        np.full((1, AZIMUTH_COUNT), horizon_deg, dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )
    return store


def _times() -> tuple[datetime, datetime]:
    return (
        datetime(2027, 1, 1, tzinfo=UTC),
        datetime(2027, 1, 1, 6, tzinfo=UTC),
    )


def test_sun_elevation_product_writes_float32_timestamped_margin_bands(
    tmp_path: Path,
) -> None:
    dem = _dem()
    times = _times()
    output = run_sun_elevation_product(
        dem=dem,
        georef=_georef(),
        horizon_store=_store(tmp_path),
        output_path=tmp_path / "sun-elevation.tif",
        times_utc=times,
        sun_vectors_m=np.stack((_position(dem, 1.25), _position(dem, -0.75))),
        backend="cpu",
        time_batch_size=1,
    )

    with rasterio.open(output) as dataset:
        assert dataset.dtypes == ("float32", "float32")
        np.testing.assert_allclose(dataset.read()[:, 0, 0], (1.0, -1.0), atol=3e-4)
        assert dataset.tags(1)["TIMESTAMP_UTC"] == "2027-01-01T00:00:00.000000Z"
        assert dataset.tags(2)["TIMESTAMP_UTC"] == "2027-01-01T06:00:00.000000Z"
        assert dataset.dataset_mask().item() == 255


def test_earth_elevation_auto_backend_falls_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import lightmap_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(lightmap_cuda, "LightmapCudaSession", UnavailableCudaSession)
    dem = _dem()
    times = _times()
    output = run_earth_elevation_product(
        dem=dem,
        georef=_georef(),
        horizon_store=_store(tmp_path),
        output_path=tmp_path / "earth-elevation.tif",
        times_utc=times,
        earth_vectors_m=np.stack((_position(dem, 0.25), _position(dem, 1.25))),
        backend="auto",
    )

    with rasterio.open(output) as dataset:
        np.testing.assert_allclose(dataset.read()[:, 0, 0], (0.0, 1.0), atol=3e-4)


def test_body_elevation_cancellation_resumes_the_whole_patch(tmp_path: Path) -> None:
    dem = _dem()
    times = _times()
    cancelled = threading.Event()

    def interrupted(*_args, **_kwargs):
        yield np.asarray([[1.0]], dtype=np.float32)
        cancelled.set()
        yield np.asarray([[2.0]], dtype=np.float32)

    common = dict(
        dem=dem,
        georef=_georef(),
        horizon_store=_store(tmp_path),
        output_path=tmp_path / "resumed-elevation.tif",
        times_utc=times,
        sun_vectors_m=np.stack((_position(dem, 1.0), _position(dem, 2.0))),
    )
    with pytest.raises(BodyElevationPipelineCancelled):
        run_sun_elevation_product(
            **common,
            _margin_calculator=interrupted,
            cancellation_requested=cancelled.is_set,
        )

    assert not common["output_path"].exists()

    def completed(*_args, **_kwargs):
        yield np.asarray([[1.0]], dtype=np.float32)
        yield np.asarray([[2.0]], dtype=np.float32)

    output = run_sun_elevation_product(
        **common,
        _margin_calculator=completed,
    )
    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read()[:, 0, 0], (1.0, 2.0))


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_cpu_and_cuda_elevation_products_agree(tmp_path: Path) -> None:
    dem = _dem()
    times = _times()
    vectors = np.stack((_position(dem, 1.25), _position(dem, -0.75)))
    common = dict(
        dem=dem,
        georef=_georef(),
        horizon_store=_store(tmp_path),
        times_utc=times,
        sun_vectors_m=vectors,
        time_batch_size=1,
    )
    cpu = run_sun_elevation_product(
        **common, output_path=tmp_path / "cpu.tif", backend="cpu"
    )
    cuda = run_sun_elevation_product(
        **common, output_path=tmp_path / "cuda.tif", backend="cuda"
    )

    with rasterio.open(cpu) as cpu_dataset, rasterio.open(cuda) as cuda_dataset:
        np.testing.assert_allclose(
            cpu_dataset.read(), cuda_dataset.read(), atol=2e-5
        )
        np.testing.assert_array_equal(
            cpu_dataset.dataset_mask(), cuda_dataset.dataset_mask()
        )
