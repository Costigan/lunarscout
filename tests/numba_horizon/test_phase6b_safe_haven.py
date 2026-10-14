from __future__ import annotations

from io import StringIO
import numpy as np
import os
import pytest
import rasterio
from pathlib import Path
import threading
from datetime import datetime, timezone

from lunarscout._numba_horizon.cuda_backend import CudaBackendError
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.psr import _pixel_frame
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.lightmap_cuda import LightmapCudaSession
from lunarscout._numba_horizon.safe_haven import (
    build_month_bands,
    _month_indices_map,
    reduce_safe_haven_products_patch_stream,
)
from lunarscout._numba_horizon.safe_haven_pipeline import (
    SafeHavenPipelineCancelled,
    run_safe_haven_products,
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


_JAN_2027 = datetime(2027, 1, 1, tzinfo=timezone.utc)


def _paths(directory: Path, name: str = "sh") -> tuple[Path, Path, Path]:
    return (
        directory / f"{name}-hibernation.tif",
        directory / f"{name}-joint-total.tif",
        directory / f"{name}-joint-longest.tif",
    )


def run_safe_haven_products_cpu(**kwargs):
    return run_safe_haven_products(**kwargs, backend="cpu")


def test_build_month_bands_single_month() -> None:
    times = ["2027-01-01T00:00:00Z", "2027-01-15T00:00:00Z", "2027-01-31T00:00:00Z"]
    bands = build_month_bands(times)
    assert len(bands) == 1
    start, stop = bands[0]
    assert start.year == 2027 and start.month == 1
    assert stop.year == 2027 and stop.month == 2


def test_build_month_bands_spans_two_months() -> None:
    times = ["2027-01-15T00:00:00Z", "2027-02-15T00:00:00Z"]
    bands = build_month_bands(times)
    assert len(bands) == 2
    assert bands[0][0].month == 1
    assert bands[1][0].month == 2


def test_month_indices_map() -> None:
    times = ["2027-01-01T00:00:00Z", "2027-02-01T00:00:00Z"]
    bands = build_month_bands(times)
    mapping = _month_indices_map(times, bands)
    assert mapping[0] == 0
    assert mapping[1] == 1


def test_cpu_safe_haven_pipeline_writes_monthly_float_duration_bands(tmp_path: Path) -> None:
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

    def fractions(*_args, **_kwargs):
        for value in (0.1, 0.1, 0.1, 0.1, 0.1):
            yield np.asarray([[value]], dtype=np.float32)

    def elevations(*_args, **_kwargs):
        # The hibernation starts at t1 (below threshold at t1-t2, above at t0, t3-t4).
        for value in (3.0, 1.0, 1.0, 3.0, 3.0):
            yield np.asarray([[value]], dtype=np.float32)

    hibernation, joint_total, joint_longest = run_safe_haven_products_cpu(
        dem=dem,
        georef=_georef(),
        horizon_store=store,
        output_paths=_paths(tmp_path),
        times_utc=times,
        sun_vectors_m=np.stack(tuple(_position(dem, 1.0) for _ in times)),
        earth_vectors_m=np.stack(tuple(_position(dem, -1.0) for _ in times)),
        time_step_hours=2.5,
        earth_threshold_deg=2.0,
        sunlight_threshold=0.2,
        fraction_calculator=fractions,
        elevation_calculator=elevations,
    )

    # Shadow throughout: the one run spans all five samples, 12.5 h.
    with rasterio.open(hibernation) as dataset:
        assert dataset.dtypes == ("float32",)
        assert dataset.read(1).item() == pytest.approx(12.5)
    # The Sun is never fully lit, so there is no joint operating time.
    for path in (joint_total, joint_longest):
        with rasterio.open(path) as dataset:
            assert dataset.dtypes == ("float32",)
            assert dataset.read(1).item() == 0.0


def test_safe_haven_auto_backend_falls_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import safe_haven_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(safe_haven_cuda, "SafeHavenCudaSession", UnavailableCudaSession)
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
    times = ("2027-01-01T00:00:00Z", "2027-01-01T12:00:00Z")

    outputs = run_safe_haven_products(
        dem=dem,
        georef=_georef(),
        horizon_store=store,
        output_paths=_paths(tmp_path, "auto"),
        times_utc=times,
        sun_vectors_m=np.stack((_position(dem, -1.0), _position(dem, -1.0))),
        # Earth is available at t0 and below the threshold from t1: a hibernation
        # starts at t1, inside shadow that began at t0.
        earth_vectors_m=np.stack((_position(dem, 10.0), _position(dem, -1.0))),
        time_step_hours=12.0,
        backend="auto",
    )

    with rasterio.open(outputs[0]) as dataset:
        assert dataset.read(1).item() == pytest.approx(24.0)


def test_safe_haven_partial_injection_uses_reference_path(tmp_path: Path) -> None:
    dem = _dem()
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(0, 0, 0.0, np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
                compress=True, valid_width=1, valid_height=1)
    times = ("2027-01-01T00:00:00Z", "2027-01-01T12:00:00Z")

    def fractions(*_args, **_kwargs):
        yield np.asarray([[0.1]], dtype=np.float32)
        yield np.asarray([[0.1]], dtype=np.float32)

    outputs = run_safe_haven_products(
        dem=dem, georef=_georef(), horizon_store=store,
        output_paths=_paths(tmp_path, "partial"), times_utc=times,
        sun_vectors_m=np.stack((_position(dem, -1.0), _position(dem, -1.0))),
        earth_vectors_m=np.stack((_position(dem, 10.0), _position(dem, -1.0))),
        time_step_hours=12.0, fraction_calculator=fractions, backend="auto",
    )
    with rasterio.open(outputs[0]) as dataset:
        assert np.isfinite(dataset.read(1).item())


def test_cuda_production_dispatch_does_not_use_host_iterators_or_reducer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import lightmap_cpu, safe_haven_cuda, safe_haven_pipeline

    dem = _dem()
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(0, 0, 0.0, np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
                compress=True, valid_width=1, valid_height=1)
    calls = []

    class FakeCuda:
        def __init__(self, **_kwargs):
            pass
        def reduce_patch(self, *_args, **_kwargs):
            calls.append("reduce")
            layer = (np.asarray([[2.0]], dtype=np.float32),)
            return layer, layer, layer

    monkeypatch.setattr(safe_haven_cuda, "SafeHavenCudaSession", FakeCuda)
    monkeypatch.setattr(safe_haven_pipeline, "reduce_safe_haven_products_patch_stream",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("host reducer")))
    monkeypatch.setattr(lightmap_cpu.LightmapCpuSession, "iter_patch_fraction_tiles",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("host fraction")))
    monkeypatch.setattr(lightmap_cpu.LightmapCpuSession, "iter_patch_margin_tiles",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("host margin")))
    outputs = run_safe_haven_products(
        dem=dem, georef=_georef(), horizon_store=store, output_paths=_paths(tmp_path, "gpu"),
        times_utc=("2027-01-01T00:00:00Z",),
        sun_vectors_m=np.asarray([_position(dem, -1.0)]),
        earth_vectors_m=np.asarray([_position(dem, -1.0)]), time_step_hours=2.0,
        backend="cuda",
    )
    assert calls == ["reduce"]
    for output in outputs:
        with rasterio.open(output) as dataset:
            assert dataset.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cuda"]'
            assert dataset.read(1).item() == 2.0


def test_explicit_cpu_does_not_construct_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lunarscout._numba_horizon import safe_haven_cuda
    monkeypatch.setattr(safe_haven_cuda, "SafeHavenCudaSession",
                        lambda **_k: (_ for _ in ()).throw(AssertionError("CUDA probe")))
    dem = _dem()
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(0, 0, 0.0, np.zeros((1, AZIMUTH_COUNT), dtype=np.float32), compress=True,
                valid_width=1, valid_height=1)
    outputs = run_safe_haven_products_cpu(
        dem=dem, georef=_georef(), horizon_store=store, output_paths=_paths(tmp_path, "cpu"),
        times_utc=("2027-01-01T00:00:00Z", "2027-01-01T12:00:00Z"),
        sun_vectors_m=np.stack((_position(dem, -1.0),) * 2),
        earth_vectors_m=np.stack((_position(dem, -1.0), _position(dem, 10.0))),
        time_step_hours=12.0,
    )
    assert all(output.exists() for output in outputs)


def test_explicit_cuda_unavailable_and_auto_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lunarscout._numba_horizon import safe_haven_cuda
    class Unavailable:
        def __init__(self, **_k):
            raise CudaBackendError("unavailable")
    monkeypatch.setattr(safe_haven_cuda, "SafeHavenCudaSession", Unavailable)
    dem = _dem()
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(0, 0, 0.0, np.zeros((1, AZIMUTH_COUNT), dtype=np.float32), compress=True,
                valid_width=1, valid_height=1)
    kwargs = dict(dem=dem, georef=_georef(), horizon_store=store,
                  times_utc=("2027-01-01T00:00:00Z", "2027-01-01T12:00:00Z"),
                  sun_vectors_m=np.stack((_position(dem, -1.0),) * 2),
                  earth_vectors_m=np.stack((_position(dem, -1.0), _position(dem, 10.0))),
                  time_step_hours=12.0)
    with pytest.raises(CudaBackendError):
        run_safe_haven_products(output_paths=_paths(tmp_path, "explicit"), backend="cuda", **kwargs)
    auto = run_safe_haven_products(output_paths=_paths(tmp_path, "auto"), backend="auto", **kwargs)
    assert all(path.exists() for path in auto)


def test_fake_cuda_typed_cancellation_resumes_unjournaled_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import safe_haven_cuda
    calls = {"count": 0}

    class FakeCuda:
        def __init__(self, **_kwargs):
            pass
        def reduce_patch(self, *_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise safe_haven_cuda.SafeHavenCudaCancelled("cancelled")
            layer = (np.asarray([[1.0]], dtype=np.float32),)
            return layer, layer, layer

    monkeypatch.setattr(safe_haven_cuda, "SafeHavenCudaSession", FakeCuda)
    dem = _dem()
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(0, 0, 0.0, np.zeros((1, AZIMUTH_COUNT), dtype=np.float32), compress=True,
                valid_width=1, valid_height=1)
    kwargs = dict(dem=dem, georef=_georef(), horizon_store=store,
                  times_utc=("2027-01-01T00:00:00Z",),
                  sun_vectors_m=np.asarray([_position(dem, -1.0)]),
                  earth_vectors_m=np.asarray([_position(dem, -1.0)]), time_step_hours=1.0,
                  backend="cuda", output_paths=_paths(tmp_path, "resume"))
    with pytest.raises(SafeHavenPipelineCancelled):
        run_safe_haven_products(**kwargs)
    assert not any(path.exists() for path in kwargs["output_paths"])
    assert all(path.exists() for path in run_safe_haven_products(**kwargs))


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
    times = ("2027-01-01T00:00:00Z", "2027-01-01T12:00:00Z")
    cancelled = threading.Event()

    def interrupted_fractions(*_args, **_kwargs):
        yield np.asarray([[0.1]], dtype=np.float32)
        cancelled.set()
        yield np.asarray([[0.1]], dtype=np.float32)

    def elevations(*_args, **_kwargs):
        yield np.asarray([[1.0]], dtype=np.float32)
        yield np.asarray([[1.0]], dtype=np.float32)

    common = dict(
        dem=dem,
        georef=_georef(),
        horizon_store=store,
        output_paths=_paths(tmp_path, "resumed"),
        times_utc=times,
        sun_vectors_m=np.stack((_position(dem, -1.0), _position(dem, -1.0))),
        earth_vectors_m=np.stack((_position(dem, -1.0), _position(dem, -1.0))),
        time_step_hours=12.0,
    )
    with pytest.raises(SafeHavenPipelineCancelled):
        run_safe_haven_products(
            **common,
            fraction_calculator=interrupted_fractions,
            elevation_calculator=elevations,
            cancellation_requested=cancelled.is_set,
        )

    assert not any(path.exists() for path in common["output_paths"])
    progress_stream = StringIO()
    progress = []

    def completed_fractions(*_args, **_kwargs):
        yield np.asarray([[0.1]], dtype=np.float32)
        yield np.asarray([[0.1]], dtype=np.float32)

    def completed_elevations(*_args, **_kwargs):
        yield np.asarray([[1.0]], dtype=np.float32)
        yield np.asarray([[1.0]], dtype=np.float32)

    result = run_safe_haven_products(
        **common,
        fraction_calculator=completed_fractions,
        elevation_calculator=completed_elevations,
        progress_callback=progress.append,
        progress_stream=progress_stream,
    )

    assert result == common["output_paths"]
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
    times = [_JAN_2027.replace(hour=h) for h in (0, 6, 12, 18)]
    month_bands = build_month_bands(times)

    cpu_frac = LightmapCpuSession(time_batch_size=2).iter_patch_fraction_tiles(
        dem, horizons, vectors, tile_y=0, tile_x=0, valid_height=1, valid_width=1,
    )
    gpu_frac = LightmapCudaSession(time_batch_size=2).iter_patch_fraction_tiles(
        dem, horizons, vectors, tile_y=0, tile_x=0, valid_height=1, valid_width=1,
    )
    cpu_margin = LightmapCpuSession(time_batch_size=2).iter_patch_margin_tiles(
        dem, horizons, vectors, tile_y=0, tile_x=0, valid_height=1, valid_width=1,
    )
    gpu_margin = LightmapCudaSession(time_batch_size=2).iter_patch_margin_tiles(
        dem, horizons, vectors, tile_y=0, tile_x=0, valid_height=1, valid_width=1,
    )

    month_idx = _month_indices_map(times, month_bands)

    options = dict(
        month_index_of=month_idx, sunlight_threshold=0.2, earth_threshold_deg=2.0,
        operating_sunlight_fraction=1.0, merge_gap_hours=24.0, time_step_hours=2.5,
    )
    cpu_duration = reduce_safe_haven_products_patch_stream(
        cpu_frac, cpu_margin, 4, month_bands, **options
    )
    gpu_duration = reduce_safe_haven_products_patch_stream(
        gpu_frac, gpu_margin, 4, month_bands, **options
    )

    np.testing.assert_array_equal(cpu_duration, gpu_duration)
    # Earth crossing may not occur with these test vectors — that's fine,
    # the important thing is CPU and CUDA produce identical outputs.
