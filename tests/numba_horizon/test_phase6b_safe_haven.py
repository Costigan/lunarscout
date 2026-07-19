from __future__ import annotations

from io import StringIO
import numpy as np
import os
import pytest
import rasterio
from pathlib import Path
import threading

from lunarscout._numba_horizon.cuda_backend import CudaBackendError
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.psr import _pixel_frame
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.lightmap_cuda import LightmapCudaSession
from lunarscout._numba_horizon.safe_haven import (
    EarthOutage,
    find_earth_outages,
    reduce_safe_haven_patch,
    reduce_safe_haven_patch_stream,
)
from lunarscout._numba_horizon.safe_haven_pipeline import (
    SafeHavenPipelineCancelled,
    run_safe_haven_product,
    run_safe_haven_product_cpu,
)
from lunarscout.georeference import GeoReference


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


def test_earth_outages_are_half_open_and_include_edge_regions() -> None:
    elevations = np.asarray((-1.0, -2.0, 3.0, 1.0, 0.0, 3.0, -4.0))

    outages = find_earth_outages(elevations, threshold_deg=2.0)

    assert outages == (
        EarthOutage(0, 2, 1),
        EarthOutage(3, 5, 4),
        EarthOutage(6, 7, 6),
    )


def test_safe_haven_duration_extends_outside_outage_and_preserves_float_hours() -> None:
    fractions = np.asarray(
        (
            ((0.1, 0.9),),
            ((0.1, 0.1),),
            ((0.9, 0.1),),
            ((0.1, 0.1),),
            ((0.1, 0.9),),
        ),
        dtype=np.float32,
    )
    outages = (EarthOutage(0, 3, 1), EarthOutage(3, 5, 3))

    actual = reduce_safe_haven_patch(
        fractions,
        outages,
        sunlight_threshold=0.2,
        time_step_hours=2.5,
    )

    np.testing.assert_array_equal(
        actual,
        np.asarray((((5.0, 7.5),), ((5.0, 7.5),)), dtype=np.float32),
    )

    streamed = reduce_safe_haven_patch_stream(
        iter(fractions),
        len(fractions),
        outages,
        sunlight_threshold=0.2,
        time_step_hours=2.5,
    )
    np.testing.assert_array_equal(np.stack(streamed), actual)


def test_cpu_safe_haven_pipeline_writes_float_duration_bands(tmp_path: Path) -> None:
    dem = _dem()
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(
        0,
        0,
        0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )
    times = tuple(f"2027-01-01T{hour:02d}:00:00Z" for hour in (0, 6, 12, 18, 23))
    sun_vectors = np.stack(tuple(_position(dem, 1.0) for _ in times))
    earth_vectors = np.stack(
        tuple(_position(dem, value) for value in (-1.0, -2.0, 3.0, 0.0, 3.0))
    )

    def fractions(*_args, **_kwargs):
        for value in (0.1, 0.1, 0.9, 0.1, 0.9):
            yield np.asarray([[value]], dtype=np.float32)

    output = run_safe_haven_product_cpu(
        dem=dem,
        georef=_georef(),
        horizon_store=store,
        output_path=tmp_path / "safe-haven.tif",
        times_utc=times,
        sun_vectors_m=sun_vectors,
        earth_vectors_m=earth_vectors,
        time_step_hours=2.5,
        earth_threshold_deg=2.0,
        sunlight_threshold=0.2,
        fraction_calculator=fractions,
    )

    with rasterio.open(output) as dataset:
        assert dataset.dtypes == ("float32", "float32")
        np.testing.assert_array_equal(dataset.read()[:, 0, 0], (5.0, 2.5))
        assert dataset.tags(1)["TIMESTAMP_UTC"] == "2027-01-01T06:00:00.000000Z"
        assert dataset.tags(2)["TIMESTAMP_UTC"] == "2027-01-01T18:00:00.000000Z"


def test_safe_haven_auto_backend_falls_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import lightmap_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(lightmap_cuda, "LightmapCudaSession", UnavailableCudaSession)
    dem = _dem()
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(
        0,
        0,
        0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )
    times = ("2027-01-01T00:00:00Z", "2027-01-01T06:00:00Z")

    output = run_safe_haven_product(
        dem=dem,
        georef=_georef(),
        horizon_store=store,
        output_path=tmp_path / "auto-safe-haven.tif",
        times_utc=times,
        sun_vectors_m=np.stack((_position(dem, -1.0), _position(dem, -1.0))),
        earth_vectors_m=np.stack((_position(dem, 0.0), _position(dem, 0.0))),
        time_step_hours=6.0,
        backend="auto",
    )

    with rasterio.open(output) as dataset:
        assert dataset.read(1).item() == 12.0


def test_safe_haven_cancellation_is_checked_during_time_stream_and_resumes(
    tmp_path: Path,
) -> None:
    dem = _dem()
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(
        0,
        0,
        0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )
    times = ("2027-01-01T00:00:00Z", "2027-01-01T06:00:00Z")
    cancelled = threading.Event()

    def interrupted_fractions(*_args, **_kwargs):
        yield np.asarray([[0.1]], dtype=np.float32)
        cancelled.set()
        yield np.asarray([[0.1]], dtype=np.float32)

    common = dict(
        dem=dem,
        georef=_georef(),
        horizon_store=store,
        output_path=tmp_path / "resumed-safe-haven.tif",
        times_utc=times,
        sun_vectors_m=np.stack((_position(dem, -1.0), _position(dem, -1.0))),
        earth_vectors_m=np.stack((_position(dem, 0.0), _position(dem, 0.0))),
        time_step_hours=6.0,
    )
    with pytest.raises(SafeHavenPipelineCancelled):
        run_safe_haven_product(
            **common,
            fraction_calculator=interrupted_fractions,
            cancellation_requested=cancelled.is_set,
        )

    assert not common["output_path"].exists()
    progress_stream = StringIO()
    progress = []

    def completed_fractions(*_args, **_kwargs):
        yield np.asarray([[0.1]], dtype=np.float32)
        yield np.asarray([[0.1]], dtype=np.float32)

    result = run_safe_haven_product(
        **common,
        fraction_calculator=completed_fractions,
        progress_callback=progress.append,
        progress_stream=progress_stream,
    )

    assert result == common["output_path"]
    assert [event.state for event in progress] == [
        "start",
        "read",
        "calculate",
        "write",
        "valid",
        "complete",
    ]
    assert progress_stream.getvalue().endswith(
        "Safe haven complete: 1/1 patches\n"
    )


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_cuda_and_cpu_fraction_streams_give_same_safe_haven_duration() -> None:
    dem = _dem()
    horizons = np.zeros((128, 128, AZIMUTH_COUNT), dtype=np.float32)
    vectors = np.stack(
        tuple(_position(dem, value) for value in (-1.0, -1.0, 1.0, -1.0))
    )
    outage = (EarthOutage(0, 4, 0),)
    cpu = LightmapCpuSession(time_batch_size=2).iter_patch_fraction_tiles(
        dem,
        horizons,
        vectors,
        tile_y=0,
        tile_x=0,
        valid_height=1,
        valid_width=1,
    )
    gpu = LightmapCudaSession(time_batch_size=2).iter_patch_fraction_tiles(
        dem,
        horizons,
        vectors,
        tile_y=0,
        tile_x=0,
        valid_height=1,
        valid_width=1,
    )

    cpu_duration = reduce_safe_haven_patch_stream(
        cpu, 4, outage, sunlight_threshold=0.2, time_step_hours=2.5
    )
    gpu_duration = reduce_safe_haven_patch_stream(
        gpu, 4, outage, sunlight_threshold=0.2, time_step_hours=2.5
    )

    np.testing.assert_array_equal(cpu_duration, gpu_duration)
    assert cpu_duration[0].item() == 5.0
