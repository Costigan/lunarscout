from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect
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


def test_public_product_storage_defaults_are_explicit() -> None:
    byte_products = (ls.generate_lightmap, ls.generate_psr)
    float_products = (
        ls.generate_sun_elevation,
        ls.generate_earth_elevation,
        ls.generate_safe_havens,
        ls.mission_duration_from_sunlight,
        ls.mission_duration_from_sun_elevation,
        ls.mission_duration_from_sunlight_and_earth_elevation,
        ls.mission_duration_from_sun_and_earth_elevation,
    )
    for function in (*byte_products, *float_products):
        parameters = inspect.signature(function).parameters
        assert parameters["compress"].default is True
        assert parameters["output_transform"].default is None
        assert parameters["output_dtype"].default is None
        assert parameters["output_transform_id"].default is None
    for function in float_products:
        assert np.isnan(inspect.signature(function).parameters["nodata"].default)


def test_output_conversion_is_validated_before_file_access(tmp_path: Path) -> None:
    with pytest.raises(ls.InputError) as raised:
        ls.generate_lightmap(
            tmp_path / "missing-dem.tif",
            tmp_path / "missing-horizons",
            tmp_path / "output.tif",
            times=ls.times(
                "2027-01-01T00:00:00Z",
                "2027-01-02T00:00:00Z",
                step_hours=24,
            ),
            output_transform=lambda values: values,
        )
    assert raised.value.code == "product_output_conversion_invalid"


def test_public_lightmap_converts_patch_to_requested_dtype(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = ls.times(
        "2027-01-01T00:00:00Z",
        "2027-01-02T00:00:00Z",
        step_hours=24,
    )
    output = ls.generate_lightmap(
        dem_path,
        horizons_path,
        tmp_path / "converted-lightmap.tif",
        times=times,
        sun_vectors_m=np.stack((_sun_vector(1.0), _sun_vector(1.0))),
        backend="cpu",
        output_transform=lambda values: values.astype(np.uint16) * 2,
        output_dtype="uint16",
    )
    with rasterio.open(output) as dataset:
        assert dataset.dtypes == ("uint16", "uint16")
        assert np.all(dataset.read() == 510)


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


def test_public_lightmap_can_disable_tile_compression(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = ls.generate_lightmap(
        dem_path,
        horizons_path,
        tmp_path / "uncompressed.tif",
        times=ls.times(
            "2027-01-01T00:00:00Z",
            "2027-01-01T00:00:00Z",
            step_hours=1,
        ),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        backend="cpu",
        compress=False,
    )

    with rasterio.open(output) as dataset:
        assert dataset.profile["tiled"] is True
        assert dataset.block_shapes == [(128, 128)]
        assert dataset.compression is None


def test_existing_output_is_rejected_before_dem_spice_or_cuda(
    tmp_path: Path,
) -> None:
    horizons = tmp_path / "horizons"
    horizons.mkdir()
    output = tmp_path / "existing.tif"
    original = b"completed-output"
    output.write_bytes(original)

    with pytest.raises(ls.ProductStorageError) as raised:
        ls.generate_lightmap(
            tmp_path / "missing-dem.tif",
            horizons,
            output,
            times=ls.times(
                "2027-01-01T00:00:00Z",
                "2027-01-01T00:00:00Z",
                step_hours=1,
            ),
        )

    assert raised.value.code == "product_output_exists"
    assert output.read_bytes() == original
    assert "spiceypy" not in sys.modules


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
        "2027-01-01T12:00:00Z",
        "2027-01-01T18:00:00Z",
    )
    output = ls.generate_safe_havens(
        dem_path,
        horizons_path,
        tmp_path / "safe-havens.tif",
        times=times,
        # Sun always low, Earth crosses threshold: below at t0-t1, above at t2-t3
        sun_vectors_m=np.stack(4 * (_sun_vector(-1.0),)),
        earth_vectors_m=np.stack(
            (_sun_vector(-1.0), _sun_vector(-1.0), _sun_vector(10.0), _sun_vector(10.0))
        ),
        earth_elevation_threshold_deg=2.0,
        sunlight_fraction_threshold=0.2,
        backend="cpu",
    )

    with rasterio.open(output) as dataset:
        assert dataset.count == 1
        duration = dataset.read(1).item()
        assert np.isfinite(duration) and duration > 0.0
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
            "mission_duration_from_sunlight_and_earth_elevation",
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
        "evaluation_start": times[0],
        "evaluation_stop": times[1],
        "step": timedelta(days=1),
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
    times = ls.times(
        "2027-01-01T00:00:00Z",
        "2027-01-02T00:00:00Z",
        step_hours=24,
    )
    sun_high = np.stack((_sun_vector(1.0), _sun_vector(1.0)))
    sun_low = np.stack((_sun_vector(-1.0), _sun_vector(-1.0)))
    earth_high = np.stack((_sun_vector(1.0), _sun_vector(1.0)))
    mission_common = {
        "evaluation_start": times.start,
        "evaluation_stop": times.stop,
        "step": timedelta(days=1),
        "candidate_start_intervals": ((times.start, times.stop),),
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
            ls.mission_duration_from_sunlight_and_earth_elevation,
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


# ---------------------------------------------------------------------------
# Structured exception class, code, and details tests
# ---------------------------------------------------------------------------


def test_output_exists_is_rejected_with_stable_code_and_details(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "lightmap.tif"
    output.write_text("guard")

    with pytest.raises(ls.ProductStorageError) as exc_info:
        ls.generate_lightmap(
            dem_path,
            horizons_path,
            output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            overwrite=False,
            backend="cpu",
        )

    assert exc_info.value.code == "product_output_exists"
    assert isinstance(exc_info.value.details, dict)
    assert str(output) in exc_info.value.details.get("path", "")
    assert output.read_text() == "guard"


def test_dem_not_georeferenced_raises_with_stable_code(tmp_path: Path) -> None:
    dem_path = tmp_path / "no_crs.tif"
    with rasterio.open(
        dem_path, "w",
        driver="GTiff",
        width=1, height=1,
        count=1,
        dtype=np.float32,
    ) as ds:
        ds.write(np.zeros((1, 1), dtype=np.float32), 1)
    horizons = tmp_path / "horizons"
    HorizonTileStore(horizons).write(
        0, 0, 0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True, valid_width=1, valid_height=1,
    )

    with pytest.raises(ls.GridError) as exc_info:
        ls.generate_lightmap(
            dem_path, horizons, tmp_path / "out.tif",
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            backend="cpu",
        )

    assert exc_info.value.code == "product_dem_not_georeferenced"


def test_backend_invalid_raises_with_code_before_dem(tmp_path: Path) -> None:
    with pytest.raises(ls.InputError) as exc_info:
        ls.generate_lightmap(
            tmp_path / "noop.tif",
            tmp_path / "noop",
            tmp_path / "out.tif",
            times=("2027-01-01T00:00:00Z",),
            backend="gpu",  # type: ignore[arg-type]
        )

    assert exc_info.value.code == "product_backend_invalid"


def test_out_of_range_observer_height_is_an_input_error(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)

    with pytest.raises(ls.ProductCalculationError):
        ls.generate_lightmap(
            dem_path, horizons_path, tmp_path / "out.tif",
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            observer_height_m=-1.0,
            backend="cpu",
        )


def test_mission_duration_unit_invalid_raises_with_code(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = ("2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z")

    with pytest.raises(ls.InputError) as exc_info:
        ls.mission_duration_from_sunlight(
            dem_path, horizons_path, tmp_path / "out.tif",
            evaluation_start=times[0],
            evaluation_stop=times[1],
            step=timedelta(days=1),
            candidate_start_intervals=((times[0], times[1]),),
            sunlight_fraction_threshold=0.5,
            output_unit="seconds",  # type: ignore[arg-type]
            backend="cpu",
        )

    assert exc_info.value.code == "mission_duration_unit_invalid"


def test_output_transform_missing_dtype_raises_with_code(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)

    with pytest.raises(ls.InputError) as exc_info:
        ls.generate_lightmap(
            dem_path, horizons_path, tmp_path / "out.tif",
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            output_transform=lambda x: x.astype(np.uint16),
            backend="cpu",
        )

    assert exc_info.value.code == "product_output_conversion_invalid"


def test_safe_haven_output_preserved_on_overwrite_false(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "safe.tif"

    ls.generate_safe_havens(
        dem_path, horizons_path, output,
        times=("2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z"),
        sun_vectors_m=np.stack((_sun_vector(-1.0), _sun_vector(-1.0))),
        earth_vectors_m=np.stack((_sun_vector(1.0), _sun_vector(1.0))),
        backend="cpu",
    )
    original = output.read_bytes()

    with pytest.raises(ls.ProductStorageError):
        ls.generate_safe_havens(
            dem_path, horizons_path, output,
            times=("2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z"),
            sun_vectors_m=np.stack((_sun_vector(-1.0), _sun_vector(-1.0))),
            earth_vectors_m=np.stack((_sun_vector(1.0), _sun_vector(1.0))),
            overwrite=False,
            backend="cpu",
        )

    assert output.read_bytes() == original


# ---------------------------------------------------------------------------
# Safe-haven boundary and edge-case tests
# ---------------------------------------------------------------------------


def test_safe_haven_no_earth_outage_produces_nodata_band(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "safe.tif"
    times = (
        "2027-01-01T00:00:00Z",
        "2027-01-02T00:00:00Z",
        "2027-01-03T00:00:00Z",
        "2027-01-04T00:00:00Z",
    )
    sun = np.stack(4 * (_sun_vector(-1.0),))

    result = ls.generate_safe_havens(
        dem_path, horizons_path, output,
        times=times,
        sun_vectors_m=sun,
        earth_vectors_m=np.stack(4 * (_sun_vector(10.0),)),
        earth_elevation_threshold_deg=2.0,
        sunlight_fraction_threshold=0.2,
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        assert np.isnan(ds.read(1).item())


def test_safe_haven_whole_interval_outage_produces_nodata_band(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "safe.tif"
    times = (
        "2027-01-01T00:00:00Z",
        "2027-01-02T00:00:00Z",
        "2027-01-03T00:00:00Z",
    )
    sun = np.stack(3 * (_sun_vector(-1.0),))

    result = ls.generate_safe_havens(
        dem_path, horizons_path, output,
        times=times,
        sun_vectors_m=sun,
        earth_vectors_m=np.stack(3 * (_sun_vector(-5.0),)),
        earth_elevation_threshold_deg=2.0,
        sunlight_fraction_threshold=0.2,
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        assert np.isnan(ds.read(1).item())
        assert ds.dataset_mask().item() == 255


def test_safe_haven_adjacent_outages_produce_two_bands(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "safe.tif"
    times = (
        "2027-01-01T00:00:00Z",
        "2027-01-01T01:00:00Z",
        "2027-01-01T02:00:00Z",
        "2027-01-01T03:00:00Z",
        "2027-01-01T04:00:00Z",
        "2027-01-01T05:00:00Z",
    )

    outage_earth = _sun_vector(-5.0)
    above_earth = _sun_vector(10.0)

    earth = np.stack(
        (outage_earth, outage_earth, above_earth, outage_earth, outage_earth, above_earth),
    )
    sun = np.stack(6 * (_sun_vector(-1.0),))

    result = ls.generate_safe_havens(
        dem_path, horizons_path, output,
        times=times,
        sun_vectors_m=sun,
        earth_vectors_m=earth,
        earth_elevation_threshold_deg=2.0,
        sunlight_fraction_threshold=0.2,
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1  # single month band
        assert ds.dataset_mask().item() == 255


def test_safe_haven_above_threshold_earth_yields_nodata_band(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "safe.tif"
    times = (
        "2027-01-01T00:00:00Z",
        "2027-01-02T00:00:00Z",
    )
    sun = np.stack((_sun_vector(-1.0), _sun_vector(-1.0)))

    result = ls.generate_safe_havens(
        dem_path, horizons_path, output,
        times=times,
        sun_vectors_m=sun,
        earth_vectors_m=np.stack((_sun_vector(10.0), _sun_vector(10.0))),
        earth_elevation_threshold_deg=2.0,
        sunlight_fraction_threshold=0.2,
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        assert np.isnan(ds.read(1).item())


def test_safe_haven_missing_horizon_returns_nodata_and_invalid_mask(
    tmp_path: Path,
) -> None:
    dem, georef = _dem(), _georef()
    ls.write_geotiff(
        tmp_path / "dem.tif",
        np.zeros((128, 128), dtype=np.float32),
        ls.GeoReference(
            projection_wkt=georef.projection_wkt,
            projection_proj4=georef.projection_proj4,
            affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
            width=128, height=128,
            pixel_size_x=1.0, pixel_size_y=-1.0,
            nodata=None,
        ),
    )
    (tmp_path / "horizons").mkdir()
    output = tmp_path / "safe_missing.tif"
    times = ("2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z")
    sun = np.stack((_sun_vector(-1.0), _sun_vector(-1.0)))
    earth = np.stack((_sun_vector(-5.0), _sun_vector(-5.0)))

    result = ls.generate_safe_havens(
        tmp_path / "dem.tif",
        tmp_path / "horizons",
        output,
        times=times,
        sun_vectors_m=sun,
        earth_vectors_m=earth,
        earth_elevation_threshold_deg=2.0,
        sunlight_fraction_threshold=0.2,
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        data = ds.read(1)
        mask = ds.dataset_mask()
        assert np.all(np.isnan(data))
        assert np.all(mask == 0)


# ---------------------------------------------------------------------------
# Mission-duration edge-case tests
# ---------------------------------------------------------------------------


def test_mission_duration_inclusive_threshold_at_exact_boundary(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = ("2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z")
    output = tmp_path / "md.tif"

    result = ls.mission_duration_from_sunlight(
        dem_path, horizons_path, output,
        evaluation_start=times[0],
        evaluation_stop=times[1],
        step=timedelta(days=1),
        candidate_start_intervals=((times[0], times[1]),),
        sunlight_fraction_threshold=1.0,
        sun_vectors_m=np.stack((_sun_vector(1.0), _sun_vector(1.0))),
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        duration = ds.read(1).item()
        assert np.isfinite(duration) and duration >= 0.0
        assert ds.dataset_mask().item() == 255


def test_mission_duration_no_feasible_start_returns_zero_and_valid_mask(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = ("2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z")
    output = tmp_path / "md.tif"

    result = ls.mission_duration_from_sunlight(
        dem_path, horizons_path, output,
        evaluation_start=times[0],
        evaluation_stop=times[1],
        step=timedelta(days=1),
        candidate_start_intervals=((times[0], times[1]),),
        sunlight_fraction_threshold=1.0,
        sun_vectors_m=np.stack((_sun_vector(-1.0), _sun_vector(-1.0))),
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        duration = ds.read(1).item()
        assert duration == 0.0
        assert ds.dataset_mask().item() == 255


def test_mission_duration_respects_evaluation_start(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "md.tif"

    result = ls.mission_duration_from_sunlight(
        dem_path, horizons_path, output,
        evaluation_start="2027-01-01T00:00:00Z",
        evaluation_stop="2027-01-01T12:00:00Z",
        step=timedelta(hours=6),
        candidate_start_intervals=(
            ("2027-01-01T00:00:00Z", "2027-01-01T12:00:00Z"),
        ),
        sunlight_fraction_threshold=1.0,
        sun_vectors_m=np.stack(
            (_sun_vector(1.0), _sun_vector(1.0), _sun_vector(1.0))
        ),
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        assert ds.dataset_mask().item() == 255


def test_mission_duration_days_vs_hours_unit_conversion(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = ("2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z")
    sun = np.stack((_sun_vector(1.0), _sun_vector(1.0)))

    hours_out = tmp_path / "md_hours.tif"
    ls.mission_duration_from_sunlight(
        dem_path, horizons_path, hours_out,
        evaluation_start=times[0], evaluation_stop=times[1],
        step=timedelta(days=1),
        candidate_start_intervals=((times[0], times[1]),),
        sunlight_fraction_threshold=0.5,
        sun_vectors_m=sun,
        output_unit="hours",
        backend="cpu",
    )

    days_out = tmp_path / "md_days.tif"
    ls.mission_duration_from_sunlight(
        dem_path, horizons_path, days_out,
        evaluation_start=times[0], evaluation_stop=times[1],
        step=timedelta(days=1),
        candidate_start_intervals=((times[0], times[1]),),
        sunlight_fraction_threshold=0.5,
        sun_vectors_m=sun,
        output_unit="days",
        backend="cpu",
    )

    with rasterio.open(hours_out) as hs, rasterio.open(days_out) as ds:
        hrs = hs.read(1).item()
        dys = ds.read(1).item()
        assert hrs == pytest.approx(24.0 * dys, abs=0.01)
        assert hs.tags(1)["DURATION_UNIT"] == "hours"
        assert ds.tags(1)["DURATION_UNIT"] == "days"


def test_mission_duration_evaluation_stop_between_samples_is_clipped(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    sun = np.stack(
        (_sun_vector(1.0), _sun_vector(1.0)),
    ).astype(np.float64)
    output = tmp_path / "md.tif"

    result = ls.mission_duration_from_sunlight(
        dem_path, horizons_path, output,
        evaluation_start="2027-01-01T00:00:00Z",
        evaluation_stop="2027-01-01T12:00:00Z",
        step=timedelta(hours=12),
        candidate_start_intervals=(
            ("2027-01-01T00:00:00Z", "2027-01-01T12:00:00Z"),
        ),
        sunlight_fraction_threshold=0.5,
        sun_vectors_m=sun,
        output_unit="hours",
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        duration = ds.read(1).item()
        assert np.isfinite(duration) and 0.0 <= duration <= 12.0
        assert ds.dataset_mask().item() == 255


def test_mission_duration_multiple_candidate_intervals_produce_multiple_bands(
    tmp_path: Path,
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = ("2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z")
    sun = np.stack((_sun_vector(1.0), _sun_vector(1.0)))
    output = tmp_path / "md.tif"

    result = ls.mission_duration_from_sunlight(
        dem_path, horizons_path, output,
        evaluation_start=times[0],
        evaluation_stop=times[1],
        step=timedelta(days=1),
        candidate_start_intervals=(
            ("2027-01-01T00:00:00Z", "2027-01-01T12:00:00Z"),
            ("2027-01-01T12:00:00Z", "2027-01-02T00:00:00Z"),
        ),
        sunlight_fraction_threshold=0.5,
        sun_vectors_m=sun,
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 2


# ---------------------------------------------------------------------------
# Public cancellation, restart, and overwrite behaviour
# ---------------------------------------------------------------------------


def test_progress_event_cancellation_leaves_no_output(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "cancel.tif"

    def cancel_after_first(_event: ls.ProgressEvent) -> None:
        raise RuntimeError("user-cancelled")

    with pytest.raises(RuntimeError):
        ls.generate_lightmap(
            dem_path, horizons_path, output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            backend="cpu",
            progress_event_callback=cancel_after_first,
        )

    assert not output.exists()


def test_cancellation_check_prevents_output_publication(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "cancel.tif"

    with pytest.raises(ls.OperationCancelledError) as exc_info:
        ls.generate_lightmap(
            dem_path, horizons_path, output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            backend="cpu",
            cancellation_requested=lambda: True,
        )

    assert exc_info.value.code == "lightmap_cancelled"
    assert not output.exists()


def test_resume_after_cancellation_completes(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "resume.tif"

    first = True

    def cancel_only_first() -> bool:
        nonlocal first
        if first:
            first = False
            return True
        return False

    with pytest.raises(ls.OperationCancelledError):
        ls.generate_lightmap(
            dem_path, horizons_path, output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            backend="cpu",
            cancellation_requested=cancel_only_first,
        )

    result = ls.generate_lightmap(
        dem_path, horizons_path, output,
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        assert ds.dataset_mask().item() == 255
        assert ds.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'


def test_start_fresh_discards_staged_state(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "fresh.tif"

    first = True

    def cancel_once() -> bool:
        nonlocal first
        if first:
            first = False
            return True
        return False

    with pytest.raises(ls.OperationCancelledError):
        ls.generate_lightmap(
            dem_path, horizons_path, output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            backend="cpu",
            cancellation_requested=cancel_once,
        )

    result = ls.generate_lightmap(
        dem_path, horizons_path, output,
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        start_fresh=True,
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        assert ds.dataset_mask().item() == 255


def test_failed_overwrite_preserves_original(tmp_path: Path) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    output = tmp_path / "overwrite.tif"

    ls.generate_lightmap(
        dem_path, horizons_path, output,
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=_sun_vector(1.0)[None, :],
        backend="cpu",
    )
    original = output.read_bytes()

    with pytest.raises(ls.OperationCancelledError):
        ls.generate_lightmap(
            dem_path, horizons_path, output,
            times=("2027-01-01T00:00:00Z",),
            sun_vectors_m=_sun_vector(1.0)[None, :],
            overwrite=True,
            backend="cpu",
            cancellation_requested=lambda: True,
        )

    assert output.read_bytes() == original


def test_invalid_tile_is_journaled_as_completed(tmp_path: Path) -> None:
    dem, georef = _dem(), _georef()
    ls.write_geotiff(
        tmp_path / "dem.tif",
        np.zeros((128, 128), dtype=np.float32),
        ls.GeoReference(
            projection_wkt=georef.projection_wkt,
            projection_proj4=georef.projection_proj4,
            affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
            width=128, height=128,
            pixel_size_x=1.0, pixel_size_y=-1.0,
            nodata=None,
        ),
    )
    (tmp_path / "horizons").mkdir()
    output = tmp_path / "invalid_tile.tif"

    result = ls.generate_lightmap(
        tmp_path / "dem.tif",
        tmp_path / "horizons",
        output,
        times=("2027-01-01T00:00:00Z",),
        sun_vectors_m=np.ascontiguousarray(
            _sun_vector(1.0)[None, :], dtype=np.float64
        ),
        backend="cpu",
    )

    with rasterio.open(result) as ds:
        assert ds.count == 1
        full_mask = ds.dataset_mask()
        assert full_mask.shape == (128, 128)
        assert np.all(full_mask == 0)


# ---------------------------------------------------------------------------
# Band-count limit test
# ---------------------------------------------------------------------------


def test_product_job_rejects_band_count_exceeding_65535(tmp_path: Path) -> None:
    from lunarscout._numba_horizon.product_store import ProductJob

    job = ProductJob(
        georef=_georef(),
        dtype=np.uint8,
        band_count=65537,
        algorithm="test",
        configuration={"test": True},
        horizon_inventory_identity="test-identity",
    )

    with pytest.raises(ValueError) as exc_info:
        job.manifest()

    assert "65535" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Final GeoTIFF metadata compatibility tests
# ---------------------------------------------------------------------------


_METADATA_FIELDS = frozenset(
    (
        "LUNARSCOUT_TIMESTAMPS_UTC",
        "LUNARSCOUT_COMPUTE_BACKENDS",
    )
)


@pytest.mark.parametrize(
    "generator_name,extra_kwargs",
    [
        ("generate_lightmap", {}),
        ("generate_psr", {}),
        ("generate_sun_elevation", {}),
        ("generate_earth_elevation", {}),
        ("generate_safe_havens", {}),
    ],
)
def test_downstream_products_emit_timestamps_and_backend_tags(
    tmp_path: Path,
    generator_name: str,
    extra_kwargs: dict[str, object],
) -> None:
    dem_path, horizons_path = _inputs(tmp_path)
    times = ("2027-01-01T00:00:00Z", "2027-01-02T00:00:00Z")
    sun = np.stack((_sun_vector(1.0), _sun_vector(1.0)))
    earth = np.stack((_sun_vector(1.0), _sun_vector(1.0)))
    output = tmp_path / f"{generator_name}.tif"

    kwargs: dict[str, object] = {
        "times": times,
        "backend": "cpu",
        **extra_kwargs,
    }

    if generator_name == "generate_earth_elevation":
        kwargs["earth_vectors_m"] = earth
    elif generator_name == "generate_safe_havens":
        kwargs["sun_vectors_m"] = sun
        kwargs["earth_vectors_m"] = earth
        kwargs["earth_elevation_threshold_deg"] = 2.0
        kwargs["sunlight_fraction_threshold"] = 0.2
    else:
        kwargs["sun_vectors_m"] = sun

    gen_func = getattr(ls, generator_name)
    result = gen_func(dem_path, horizons_path, output, **kwargs)  # type: ignore[arg-type]

    with rasterio.open(result) as ds:
        tags = ds.tags()
        assert "LUNARSCOUT_TIMESTAMPS_UTC" in tags
        assert "LUNARSCOUT_COMPUTE_BACKENDS" in tags
        assert tags["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cpu"]'
