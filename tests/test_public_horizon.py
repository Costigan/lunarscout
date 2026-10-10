from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import lunarscout as ls
import lunarscout.horizon as horizon_module
from lunarscout._numba_horizon.cuda_backend import CudaBackendError
from lunarscout._numba_horizon.file_format import HorizonTileStore, read_horizon_tile
from lunarscout._numba_horizon.pipeline import (
    HorizonPipelineCancelled,
    HorizonProgress,
)


_WKT = 'PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'


def _write_dem(path: Path, *, width: int = 1, height: int = 1) -> Path:
    georef = ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=(1000.0, 20.0, 0.0, -1000.0, 0.0, -20.0),
        width=width,
        height=height,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )
    return ls.write_geotiff(
        path,
        np.zeros((height, width), dtype=np.float32),
        georef,
    )


def test_public_horizon_facade_reports_cuda_and_returns_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    primary = _write_dem(tmp_path / "primary.tif")
    outer = _write_dem(tmp_path / "outer.tif")
    output = tmp_path / "horizons"
    calls: list[tuple[object, ...]] = []

    def fake_run(dems, output_directory, **kwargs):
        calls.append((dems, output_directory, kwargs))
        callback = kwargs["progress_callback"]
        callback(HorizonProgress(0, 100, 10.0, "prepare_patches", "Preparing."))
        callback(HorizonProgress(0, 100, 15.0, "process_patches", "Starting."))
        callback(HorizonProgress(1, 100, 1.0, "process_patches", "Generated."))
        callback(
            HorizonProgress(
                100,
                100,
                100.0,
                "complete",
                "Complete.",
                "horizon_00000_00128_000.cbin",
            )
        )

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", fake_run)
    fractions: list[float] = []
    events: list[ls.ProgressEvent] = []

    actual = ls.generate_horizons(
        output,
        [primary, outer],
        observer_height_m=1.5,
        compress=False,
        overwrite=True,
        verbose=True,
        progress_callback=fractions.append,
        progress_event_callback=events.append,
    )

    assert actual == output
    assert len(calls[0][0]) == 2
    assert calls[0][1] == output
    assert calls[0][2]["observer_height_m"] == 1.5
    assert calls[0][2]["compress"] is False
    assert calls[0][2]["overwrite"] is True
    assert calls[0][2]["verbose"] is True
    assert fractions == pytest.approx([0.1, 0.15, 0.1585, 1.0])
    assert events[0].backend == "cuda"
    assert (events[-1].tile_y, events[-1].tile_x) == (0, 128)
    verbose_output = capsys.readouterr().out.splitlines()
    assert verbose_output[0] == "horizons: using cuda backend"
    progress_lines = verbose_output[1:]
    assert len({line.index("|") for line in progress_lines}) == 1
    assert progress_lines[0].startswith("horizons: prepare_patches")
    assert "--.-- s/patch | ETA -------------------" in progress_lines[0]
    assert progress_lines[0].endswith("| Preparing.")
    assert re.search(
        r"process_patches\s+1/100\s+\|\s+\d+\.\d{2} s/patch \| "
        r"ETA \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| Generated\.$",
        progress_lines[2],
    )


@pytest.mark.parametrize("verbose", [False, True])
def test_public_horizon_routes_verbose_timings_to_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verbose: bool,
) -> None:
    from lunarscout._numba_horizon import cuda_backend, pipeline

    dem = _write_dem(tmp_path / "dem.tif")
    calls = []
    monkeypatch.setattr(
        cuda_backend, "CudaSession",
        lambda **_kwargs: SimpleNamespace(build_max_pyramid=lambda _dem: object()),
    )
    # No pending patches avoids CUDA warmup while exercising the real public
    # facade and its production pipeline binding.
    monkeypatch.setattr(pipeline, "enumerate_patches", lambda *_args: [])
    monkeypatch.setattr(
        pipeline,
        "run_bounded_pipeline",
        lambda *_args, **kwargs: calls.append(kwargs) or SimpleNamespace(failed_patches=()),
    )
    ls.generate_horizons(tmp_path / "horizons", [dem], verbose=verbose)
    assert len(calls) == 1
    assert calls[0]["timing_stream"] is (sys.stdout if verbose else None)
    assert "progress_stream" not in calls[0]


def test_public_horizon_cuda_failure_is_structured_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dem = _write_dem(tmp_path / "dem.tif")
    output = tmp_path / "horizons"

    def unavailable(*_args, **_kwargs):
        raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", unavailable)

    with pytest.raises(ls.CudaError) as raised:
        ls.generate_horizons(output, [dem])

    assert raised.value.code == "cuda_horizon_unavailable"
    assert not output.exists()


def test_public_horizon_missing_cuda_runtime_has_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dem = _write_dem(tmp_path / "dem.tif")

    def unavailable(*_args, **_kwargs):
        raise CudaBackendError(
            'The Lunarscout CUDA runtime is not installed. Install the CUDA '
            'runtime with: pip install "lunarscout[cuda]"'
        )

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", unavailable)

    with pytest.raises(ls.CudaError) as raised:
        ls.generate_horizons(tmp_path / "horizons", [dem])

    assert raised.value.code == "cuda_horizon_unavailable"
    assert raised.value.details["error"].endswith(
        'pip install "lunarscout[cuda]"'
    )


def test_public_horizon_step_limit_error_keeps_its_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dem = _write_dem(tmp_path / "dem.tif")
    details = {"tile_column": 0, "tile_row": 0, "ray_count": 3}

    def step_limit(*_args, **_kwargs):
        raise ls.HorizonGenerationError(
            "Horizon rays exceeded the ray-march step limit.",
            code="horizon_step_limit_exceeded",
            details=details,
        )

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", step_limit)

    with pytest.raises(ls.HorizonGenerationError) as raised:
        ls.generate_horizons(tmp_path / "horizons", [dem])

    assert raised.value.code == "horizon_step_limit_exceeded"
    assert raised.value.details == details


def test_public_horizon_cuda_jit_failure_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dem = _write_dem(tmp_path / "dem.tif")

    class FakeCudaJitError(RuntimeError):
        pass

    FakeCudaJitError.__module__ = "numba.cuda.cudadrv.driver"

    def fail_jit(*_args, **_kwargs):
        raise FakeCudaJitError("PTX compilation failed")

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", fail_jit)

    with pytest.raises(ls.CudaError) as raised:
        ls.generate_horizons(tmp_path / "horizons", [dem])

    assert raised.value.code == "cuda_horizon_execution_failed"
    assert raised.value.details["error"] == "PTX compilation failed"


def test_public_horizon_validates_paths_before_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", fail_if_called)

    with pytest.raises(ls.InputError) as raised:
        ls.generate_horizons(tmp_path / "horizons", [tmp_path / "missing.tif"])

    assert raised.value.code == "horizon_dem_not_found"
    assert called is False


def test_public_horizon_rejects_dems_on_different_spheres(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", fail_if_called)
    primary = _write_dem(tmp_path / "primary.tif", width=4, height=4)
    georef = ls.GeoReference(
        projection_wkt=_WKT.replace("1737400", "1737150"),
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737150 +units=m",
        affine_transform=(0.0, 100.0, 0.0, 0.0, 0.0, -100.0),
        width=4,
        height=4,
        pixel_size_x=100.0,
        pixel_size_y=-100.0,
        nodata=None,
    )
    outer = ls.write_geotiff(
        tmp_path / "outer.tif", np.zeros((4, 4), dtype=np.float32), georef
    )

    with pytest.raises(ls.GridError) as raised:
        ls.generate_horizons(tmp_path / "horizons", [primary, outer])

    assert raised.value.code == "horizon_dem_radius_mismatch"
    assert raised.value.details["radii_m"] == [1737400.0, 1737150.0]
    assert called is False


def test_public_horizon_accepts_dems_of_different_extent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received = []
    monkeypatch.setattr(
        horizon_module,
        "_run_horizon_pipeline",
        lambda dems, *_args, **_kwargs: received.append(dems),
    )
    primary = _write_dem(tmp_path / "primary.tif", width=4, height=4)
    outer = _write_polar_dem(
        tmp_path / "outer.tif", np.zeros((8, 6)), origin_x=-500.0, origin_y=900.0
    )

    ls.generate_horizons(tmp_path / "horizons", [primary, outer])

    assert [dem.width for dem in received[0]] == [4, 6]


def test_public_horizon_cancellation_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dem = _write_dem(tmp_path / "dem.tif")

    def cancelled(*_args, **_kwargs):
        raise HorizonPipelineCancelled("cancelled")

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", cancelled)
    with pytest.raises(ls.OperationCancelledError) as raised:
        ls.generate_horizons(tmp_path / "horizons", [dem])

    assert raised.value.code == "horizon_generation_cancelled"


def test_public_horizon_callback_exception_propagates_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dem = _write_dem(tmp_path / "dem.tif")
    expected = RuntimeError("caller callback failed")

    def fake_run(_dems, _output, **kwargs):
        kwargs["progress_callback"](
            HorizonProgress(0, 1, 10.0, "prepare_patches", "Preparing.")
        )

    def callback(_fraction: float) -> None:
        raise expected

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", fake_run)
    with pytest.raises(RuntimeError) as raised:
        ls.generate_horizons(
            tmp_path / "horizons", [dem], progress_callback=callback
        )

    assert raised.value is expected


def test_public_horizon_validates_patch_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dem = _write_dem(tmp_path / "dem.tif")
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(horizon_module, "_run_horizon_pipeline", fail_if_called)

    with pytest.raises(ls.InputError) as raised:
        ls.generate_horizons(tmp_path / "h", [dem], patch_stride=0)
    assert raised.value.code == "horizon_patch_stride_invalid"

    with pytest.raises(ls.InputError) as raised:
        ls.generate_horizons(tmp_path / "h", [dem], patch_stride=2, patch_offset=2)
    assert raised.value.code == "horizon_patch_offset_invalid"

    with pytest.raises(ls.InputError) as raised:
        ls.generate_horizons(tmp_path / "h", [dem], patch_stride=2, patch_offset=-1)
    assert raised.value.code == "horizon_patch_offset_invalid"

    assert called is False


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_public_horizon_executes_real_cuda_kernel(tmp_path: Path) -> None:
    primary = _write_dem(tmp_path / "primary.tif")
    surrounding = _write_dem(tmp_path / "surrounding.tif")

    output = ls.generate_horizons(
        tmp_path / "horizons", [primary, surrounding]
    )

    path = HorizonTileStore(output).find_existing_path(0, 0, 0.0)
    assert path is not None
    assert HorizonTileStore.is_complete(path)
    values = read_horizon_tile(path)
    assert values.shape == (128, 128, 1440)
    assert np.all(np.isfinite(values[0, 0]))
    assert not any(
        name == "clr"
        or name == "pythonnet"
        or name.startswith("pythonnet.")
        or name == "moonlib"
        or name.startswith("moonlib.")
        for name in sys.modules
    )


def _write_polar_dem(
    path: Path, elevation: np.ndarray, *, origin_x: float, origin_y: float
) -> Path:
    height, width = elevation.shape
    georef = ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=(origin_x, 20.0, 0.0, origin_y, 0.0, -20.0),
        width=width,
        height=height,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )
    return ls.write_geotiff(path, elevation.astype(np.float32), georef)


def test_pole_patch_selection_covers_patches_near_the_pole(tmp_path: Path) -> None:
    from lunarscout._numba_horizon.pipeline import enumerate_patches
    from lunarscout.products import _load_dem

    # The south pole (projection origin) lies at pixel (128, 384).
    path = _write_polar_dem(
        tmp_path / "dem.tif",
        np.zeros((1024, 1024)),
        origin_x=-128.5 * 20.0,
        origin_y=384.5 * 20.0,
    )
    dem = _load_dem(path)[0]
    patches = enumerate_patches(dem.width, dem.height)

    selected = horizon_module._pole_patch_origins(dem, patches)

    # Patches whose nearest pixel is within 160 px of (128, 384).
    expected = {(x, y) for x in (0, 128, 256) for y in (256, 384)}
    expected |= {(x, y) for x in (0, 128) for y in (128, 512)}
    assert selected == expected


def test_pole_patch_selection_includes_pole_just_outside_the_dem(
    tmp_path: Path,
) -> None:
    from lunarscout._numba_horizon.pipeline import enumerate_patches
    from lunarscout.products import _load_dem

    # The pole lies 100 px left of column 0 and level with row 64; the DEM is
    # narrower than a patch, so the edge patch is partial.
    path = _write_polar_dem(
        tmp_path / "dem.tif",
        np.zeros((300, 100)),
        origin_x=99.5 * 20.0,
        origin_y=64.5 * 20.0,
    )
    dem = _load_dem(path)[0]
    patches = enumerate_patches(dem.width, dem.height)

    assert horizon_module._pole_patch_origins(dem, patches) == {(0, 0), (0, 128)}
    assert horizon_module._pole_patch_origins(dem, patches, radius_px=50) == set()


def test_pole_patch_selection_is_empty_far_from_the_pole(tmp_path: Path) -> None:
    from lunarscout._numba_horizon.pipeline import enumerate_patches
    from lunarscout.products import _load_dem

    path = _write_polar_dem(
        tmp_path / "dem.tif", np.zeros((256, 256)), origin_x=50_000.0, origin_y=-50_000.0
    )
    dem = _load_dem(path)[0]

    assert horizon_module._pole_patch_origins(
        dem, enumerate_patches(dem.width, dem.height)
    ) == set()


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_public_horizon_is_exact_around_the_pole(tmp_path: Path) -> None:
    # A bowl centred on the south pole: z = h (r / rim)^2 inside the rim, flat
    # at h outside.  Azimuths are measured from true north, so every pixel at
    # the same distance from the pole sees the same horizon profile, and the
    # pole itself sees the rim at the same angle in every direction.  The DEM
    # is three patches wide; the last patch is too far from the pole for
    # exact rays, so one run mixes both segment layouts.
    rim_m, rim_height_m, radius_m = 1000.0, 200.0, 1737400.0
    x, y = np.meshgrid((np.arange(384) - 64) * 20.0, -(np.arange(128) - 64) * 20.0)
    r = np.minimum(np.hypot(x, y), rim_m)
    path = _write_polar_dem(
        tmp_path / "bowl.tif",
        rim_height_m * (r / rim_m) ** 2,
        origin_x=-64.5 * 20.0,
        origin_y=64.5 * 20.0,
    )

    output = ls.generate_horizons(tmp_path / "horizons", [path], compress=False)

    store = HorizonTileStore(output)
    far_tile = read_horizon_tile(store.find_existing_path(0, 256, 0.0))
    assert not np.any(np.isnan(far_tile))
    tile = read_horizon_tile(store.find_existing_path(0, 0, 0.0))
    pole = tile[64, 64]
    expected = np.degrees(np.arctan(rim_height_m / rim_m - rim_m / (2.0 * radius_m)))
    assert np.max(np.abs(pole - expected)) < 0.15
    assert np.ptp(pole) < 0.1
    for k in (1, 4, 16):
        profiles = np.stack(
            (tile[64, 64 + k], tile[64, 64 - k], tile[64 + k, 64], tile[64 - k, 64])
        )
        assert np.max(np.ptp(profiles, axis=0)) < 0.05, k


def _fake_gpu_failing_at(monkeypatch: pytest.MonkeyPatch, failing_tile_x: int, error):
    from lunarscout._numba_horizon import cuda_backend, generator

    monkeypatch.setattr(
        cuda_backend, "CudaSession",
        lambda **_kwargs: SimpleNamespace(build_max_pyramid=lambda _dem: object()),
    )

    def fake_generate(_session, _segments, _pyramids, *, tile_column, tile_row, **_kwargs):
        if tile_column == failing_tile_x:
            raise error
        return SimpleNamespace(slopes=np.zeros((128 * 128, 1440), dtype=np.float32))

    monkeypatch.setattr(generator, "generate_patch_horizons", fake_generate)


def test_public_horizon_can_finish_other_patches_after_a_patch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_gpu_failing_at(monkeypatch, 128, ValueError("bad terrain"))
    dem = _write_polar_dem(
        tmp_path / "dem.tif", np.zeros((16, 300)), origin_x=50_000.0, origin_y=-50_000.0
    )

    with pytest.raises(ls.HorizonGenerationError) as raised:
        ls.generate_horizons(tmp_path / "out", [dem], skip_failed_patches=True)

    assert raised.value.code == "horizon_patches_failed"
    assert raised.value.details["patches"] == [
        {"tile_x": 128, "tile_y": 0, "error": "ValueError: bad terrain"}
    ]
    store = HorizonTileStore(tmp_path / "out")
    assert store.list_existing_tiles(0.0) == {(0, 0), (0, 256)}

    _fake_gpu_failing_at(monkeypatch, -1, ValueError("unused"))
    ls.generate_horizons(tmp_path / "out", [dem], skip_failed_patches=True)
    assert store.list_existing_tiles(0.0) == {(0, 0), (0, 128), (0, 256)}


def test_public_horizon_stops_at_the_first_failure_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_gpu_failing_at(monkeypatch, 0, ValueError("bad terrain"))
    dem = _write_polar_dem(
        tmp_path / "dem.tif", np.zeros((16, 300)), origin_x=50_000.0, origin_y=-50_000.0
    )

    with pytest.raises(ls.HorizonGenerationError) as raised:
        ls.generate_horizons(tmp_path / "out", [dem])

    assert raised.value.code != "horizon_patches_failed"


def test_public_horizon_never_skips_cuda_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCudaApiError(Exception):
        pass

    FakeCudaApiError.__module__ = "numba.cuda.cudadrv.driver"
    _fake_gpu_failing_at(monkeypatch, 128, FakeCudaApiError("illegal address"))
    dem = _write_polar_dem(
        tmp_path / "dem.tif", np.zeros((16, 300)), origin_x=50_000.0, origin_y=-50_000.0
    )

    with pytest.raises(ls.CudaError) as raised:
        ls.generate_horizons(tmp_path / "out", [dem], skip_failed_patches=True)

    assert raised.value.code == "cuda_horizon_execution_failed"
