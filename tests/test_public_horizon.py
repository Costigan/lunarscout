from __future__ import annotations

import os
from pathlib import Path
import sys

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
    assert fractions == pytest.approx([0.1, 0.15, 0.1585, 1.0])
    assert events[0].backend == "cuda"
    assert (events[-1].tile_y, events[-1].tile_x) == (0, 128)
    assert "horizons: using cuda backend" in capsys.readouterr().out


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
