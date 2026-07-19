from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio

import lunarscout as ls
from lunarscout._numba_horizon.cuda_backend import CudaBackendError
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.psr import _pixel_frame


_WKT = 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'


def _georef() -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
        width=1,
        height=1,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=None,
    )


def _dem() -> DemGrid:
    return DemGrid(
        np.zeros((1, 1), dtype=np.float32),
        np.asarray(_georef().affine_transform, dtype=np.float64),
        ProjectionParameters(
            radius_m=1_737_400.0,
            latitude_origin_rad=-np.pi / 2.0,
            longitude_origin_rad=0.0,
            scale=1.0,
            false_easting_m=0.0,
            false_northing_m=0.0,
        ),
    )


def _sun_vector(elevation_deg: float) -> np.ndarray:
    dem = _dem()
    rotation, translation = _pixel_frame(dem, 0, 0)
    elevation = np.deg2rad(elevation_deg)
    local = np.asarray((0.0, np.cos(elevation), np.sin(elevation)))
    return (local * 150_000_000_000.0 - translation) @ rotation.T


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    dem_path = tmp_path / "dem.tif"
    ls.write_geotiff(dem_path, np.zeros((1, 1), dtype=np.float32), _georef())
    horizons_path = tmp_path / "horizons"
    HorizonTileStore(horizons_path).write(
        0,
        0,
        0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )
    return dem_path, horizons_path


def test_public_cpu_lightmap_uses_explicit_vectors_without_cuda_or_spice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    monkeypatch.setitem(sys.modules, "spiceypy", None)
    monkeypatch.setitem(
        sys.modules, "lunarscout._numba_horizon.lightmap_cuda", None
    )
    cuda_modules_before = {
        name for name in sys.modules if name == "numba.cuda" or name.startswith("numba.cuda.")
    }
    fractions: list[float] = []
    events: list[ls.ProgressEvent] = []

    output = ls.generate_lightmap(
        dem_path,
        horizons_path,
        tmp_path / "lightmap.tif",
        times=(datetime(2027, 1, 1, tzinfo=timezone.utc),),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        backend="cpu",
        verbose=True,
        progress_callback=fractions.append,
        progress_event_callback=events.append,
    )

    assert output == tmp_path / "lightmap.tif"
    assert fractions == [0.0, 1.0]
    assert events[0].backend == "cpu"
    assert events[-1].stage == "complete"
    assert "lightmap: using cpu backend" in capsys.readouterr().out
    cuda_modules_after = {
        name for name in sys.modules if name == "numba.cuda" or name.startswith("numba.cuda.")
    }
    assert cuda_modules_after == cuda_modules_before
    with rasterio.open(output) as dataset:
        assert dataset.read(1).item() == 255
        assert dataset.dataset_mask().item() == 255
        assert dataset.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'


def test_public_lightmap_default_auto_reports_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import lightmap_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(lightmap_cuda, "LightmapCudaSession", UnavailableCudaSession)
    dem_path, horizons_path = _inputs(tmp_path)
    events: list[ls.ProgressEvent] = []

    output = ls.generate_lightmap(
        dem_path,
        horizons_path,
        tmp_path / "auto.tif",
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        progress_event_callback=events.append,
    )

    assert events[0].backend == "cpu"
    with rasterio.open(output) as dataset:
        assert dataset.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'


def test_public_explicit_cuda_failure_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import lightmap_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(lightmap_cuda, "LightmapCudaSession", UnavailableCudaSession)
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "cuda.tif"

    with pytest.raises(ls.CudaError) as raised:
        ls.generate_lightmap(
            dem_path,
            horizons_path,
            output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            backend="cuda",
        )

    assert raised.value.code == "cuda_lightmap_unavailable"
    assert not output.exists()


def test_public_cuda_jit_failure_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import lightmap_pipeline

    class FakeCudaJitError(RuntimeError):
        pass

    FakeCudaJitError.__module__ = "numba.cuda.cudadrv.driver"

    def fail_jit(**_kwargs):
        raise FakeCudaJitError("PTX compilation failed")

    monkeypatch.setattr(lightmap_pipeline, "run_lightmap_product", fail_jit)
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "cuda-jit.tif"

    with pytest.raises(ls.CudaError) as raised:
        ls.generate_lightmap(
            dem_path,
            horizons_path,
            output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            backend="cuda",
        )

    assert raised.value.code == "cuda_lightmap_execution_failed"
    assert raised.value.details["error"] == "PTX compilation failed"
    assert not output.exists()


def test_public_lightmap_validates_vectors_before_product_creation(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "invalid.tif"

    with pytest.raises(ls.VectorError) as raised:
        ls.generate_lightmap(
            dem_path,
            horizons_path,
            output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=np.zeros((2, 3)),
            backend="cpu",
        )

    assert raised.value.code == "product_vectors_invalid"
    assert not output.exists()


def test_public_lightmap_propagates_callback_exception(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    expected = ValueError("callback failed")

    def fail(_event: ls.ProgressEvent) -> None:
        raise expected

    with pytest.raises(ValueError) as raised:
        ls.generate_lightmap(
            dem_path,
            horizons_path,
            tmp_path / "callback.tif",
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            backend="cpu",
            progress_event_callback=fail,
        )

    assert raised.value is expected


def test_scenario_lightmap_resolves_canonical_paths(tmp_path: Path) -> None:
    dem_path, _horizons_path = _inputs(tmp_path)
    assert dem_path == tmp_path / "dem.tif"
    scenario = ls.open_scenario(tmp_path)

    output = scenario.lightmap(
        "analysis/lightmap.tif",
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        backend="cpu",
    )

    assert output == tmp_path / "analysis/lightmap.tif"


def test_public_cpu_psr_returns_path_values_mask_and_backend_metadata(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    fractions: list[float] = []
    events: list[ls.ProgressEvent] = []

    output = ls.generate_psr(
        dem_path,
        horizons_path,
        tmp_path / "psr.tif",
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        backend="cpu",
        progress_callback=fractions.append,
        progress_event_callback=events.append,
    )

    assert output == tmp_path / "psr.tif"
    assert fractions == [0.0, 1.0]
    assert events[0].backend == "cpu"
    with rasterio.open(output) as dataset:
        assert dataset.read(1).item() == 0
        assert dataset.dataset_mask().item() == 255
        assert dataset.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'


def test_scenario_psr_routes_to_public_python_product(tmp_path: Path) -> None:
    _inputs(tmp_path)
    scenario = ls.open_scenario(tmp_path)

    output = scenario.psr(
        "analysis/psr.tif",
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        backend="cpu",
    )

    assert output == tmp_path / "analysis/psr.tif"


def test_public_psr_default_auto_reports_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import psr_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(psr_cuda, "PsrCudaSession", UnavailableCudaSession)
    dem_path, horizons_path = _inputs(tmp_path)
    events: list[ls.ProgressEvent] = []

    output = ls.generate_psr(
        dem_path,
        horizons_path,
        tmp_path / "auto-psr.tif",
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        progress_event_callback=events.append,
    )

    assert events[0].backend == "cpu"
    with rasterio.open(output) as dataset:
        assert dataset.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'


def test_public_psr_explicit_cuda_failure_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import psr_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(psr_cuda, "PsrCudaSession", UnavailableCudaSession)
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "cuda-psr.tif"

    with pytest.raises(ls.CudaError) as raised:
        ls.generate_psr(
            dem_path,
            horizons_path,
            output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            backend="cuda",
        )

    assert raised.value.code == "cuda_psr_unavailable"
    assert not output.exists()


@pytest.mark.parametrize(
    ("function_name", "vector_name", "expected"),
    [
        ("generate_sun_elevation", "sun_vectors_m", 1.0),
        ("generate_earth_elevation", "earth_vectors_m", -0.75),
    ],
)
def test_public_cpu_body_elevation_products(
    tmp_path: Path,
    function_name: str,
    vector_name: str,
    expected: float,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    function = getattr(ls, function_name)
    output = function(
        dem_path,
        horizons_path,
        tmp_path / f"{function_name}.tif",
        times=("2027-01-01T00:00:00Z",),
        backend="cpu",
        **{vector_name: _sun_vector(expected)[None, :]},
    )

    with rasterio.open(output) as dataset:
        np.testing.assert_allclose(dataset.read(1).item(), expected, atol=1e-4)
        assert dataset.dataset_mask().item() == 255
        assert dataset.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'


def test_public_cpu_safe_havens_use_hours_and_strict_thresholds(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = (
        "2027-01-01T00:00:00Z",
        "2027-01-01T06:00:00Z",
    )
    output = ls.generate_safe_havens(
        dem_path,
        horizons_path,
        tmp_path / "safe-havens.tif",
        times=times,
        sun_vectors_m=np.stack((_sun_vector(-1.0), _sun_vector(-1.0))),
        earth_vectors_m=np.stack((_sun_vector(0.0), _sun_vector(0.0))),
        earth_elevation_threshold_deg=2.0,
        sunlight_fraction_threshold=0.2,
        backend="cpu",
    )

    with rasterio.open(output) as dataset:
        assert dataset.count == 1
        assert dataset.read(1).item() == 12.0
        assert dataset.dataset_mask().item() == 255
        assert dataset.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'


def test_public_safe_havens_reject_nonuniform_samples_before_output(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "nonuniform.tif"
    times = (
        "2027-01-01T00:00:00Z",
        "2027-01-01T06:00:00Z",
        "2027-01-01T13:00:00Z",
    )
    vectors = np.stack(tuple(_sun_vector(-1.0) for _ in times))

    with pytest.raises(ls.ProductTimeError) as raised:
        ls.generate_safe_havens(
            dem_path,
            horizons_path,
            output,
            times=times,
            sun_vectors_m=vectors,
            earth_vectors_m=vectors,
            backend="cpu",
        )

    assert raised.value.code == "safe_haven_times_not_uniform"
    assert not output.exists()


@pytest.mark.parametrize(
    ("function_name", "thresholds", "needs_earth"),
    [
        ("mission_duration_from_sunlight", {"sunlight_fraction_threshold": 0.5}, False),
        ("mission_duration_from_sun_elevation", {"sun_elevation_threshold_deg": 0.0}, False),
        (
            "mission_duration_from_sunlight_and_earth",
            {"sunlight_fraction_threshold": 0.5, "earth_elevation_threshold_deg": 0.0},
            True,
        ),
        (
            "mission_duration_from_sun_and_earth_elevation",
            {"sun_elevation_threshold_deg": 0.0, "earth_elevation_threshold_deg": 0.0},
            True,
        ),
    ],
)
def test_public_cpu_mission_duration_families(
    tmp_path: Path,
    function_name: str,
    thresholds: dict[str, float],
    needs_earth: bool,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = (
        "2027-01-01T00:00:00Z",
        "2027-01-02T00:00:00Z",
    )
    kwargs = {
        "times": times,
        "evaluation_start": times[0],
        "evaluation_stop": times[1],
        "candidate_start_intervals": ((times[0], times[1]),),
        "sun_vectors_m": np.stack((_sun_vector(1.0), _sun_vector(1.0))),
        "output_unit": "days",
        "backend": "cpu",
        **thresholds,
    }
    if needs_earth:
        kwargs["earth_vectors_m"] = np.stack(
            (_sun_vector(1.0), _sun_vector(1.0))
        )
    output = getattr(ls, function_name)(
        dem_path,
        horizons_path,
        tmp_path / f"{function_name}.tif",
        **kwargs,
    )

    with rasterio.open(output) as dataset:
        assert dataset.read(1).item() == 1.0
        assert dataset.tags(1)["DURATION_UNIT"] == "days"
        assert dataset.dataset_mask().item() == 255
        assert dataset.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the public CPU/CUDA matrix",
)
def test_all_public_downstream_cpu_and_cuda_products_agree(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = (
        "2027-01-01T00:00:00Z",
        "2027-01-02T00:00:00Z",
    )
    sun_high = np.stack((_sun_vector(1.0), _sun_vector(1.0)))
    sun_low = np.stack((_sun_vector(-1.0), _sun_vector(-1.0)))
    earth_high = np.stack((_sun_vector(1.0), _sun_vector(1.0)))
    mission_common = {
        "times": times,
        "evaluation_start": times[0],
        "evaluation_stop": times[1],
        "candidate_start_intervals": ((times[0], times[1]),),
        "sun_vectors_m": sun_high,
        "output_unit": "hours",
    }
    operations = (
        ("lightmap", ls.generate_lightmap, {"times": times, "sun_vectors_m": sun_high}),
        ("psr", ls.generate_psr, {"times": times, "sun_vectors_m": sun_high}),
        (
            "sun_elevation",
            ls.generate_sun_elevation,
            {"times": times, "sun_vectors_m": sun_high},
        ),
        (
            "earth_elevation",
            ls.generate_earth_elevation,
            {"times": times, "earth_vectors_m": earth_high},
        ),
        (
            "safe_havens",
            ls.generate_safe_havens,
            {
                "times": times,
                "sun_vectors_m": sun_low,
                "earth_vectors_m": earth_high,
                "earth_elevation_threshold_deg": 2.0,
                "sunlight_fraction_threshold": 0.2,
            },
        ),
        (
            "mission_sunlight",
            ls.mission_duration_from_sunlight,
            {**mission_common, "sunlight_fraction_threshold": 0.5},
        ),
        (
            "mission_sun_elevation",
            ls.mission_duration_from_sun_elevation,
            {**mission_common, "sun_elevation_threshold_deg": 0.0},
        ),
        (
            "mission_sunlight_earth",
            ls.mission_duration_from_sunlight_and_earth,
            {
                **mission_common,
                "earth_vectors_m": earth_high,
                "sunlight_fraction_threshold": 0.5,
                "earth_elevation_threshold_deg": 0.0,
            },
        ),
        (
            "mission_sun_earth_elevation",
            ls.mission_duration_from_sun_and_earth_elevation,
            {
                **mission_common,
                "earth_vectors_m": earth_high,
                "sun_elevation_threshold_deg": 0.0,
                "earth_elevation_threshold_deg": 0.0,
            },
        ),
    )

    for name, function, kwargs in operations:
        cpu_path = function(
            dem_path,
            horizons_path,
            tmp_path / f"{name}-cpu.tif",
            backend="cpu",
            **kwargs,
        )
        cuda_path = function(
            dem_path,
            horizons_path,
            tmp_path / f"{name}-cuda.tif",
            backend="cuda",
            **kwargs,
        )
        with rasterio.open(cpu_path) as cpu, rasterio.open(cuda_path) as cuda:
            assert cpu.count == cuda.count
            assert cpu.dtypes == cuda.dtypes
            if np.issubdtype(np.dtype(cpu.dtypes[0]), np.integer):
                np.testing.assert_array_equal(cpu.read(), cuda.read())
            else:
                np.testing.assert_allclose(cpu.read(), cuda.read(), atol=1e-4)
            np.testing.assert_array_equal(cpu.dataset_mask(), cuda.dataset_mask())
            assert cpu.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'
            assert cuda.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cuda"]'
