from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lunarscout as ls
import rasterio
import lunarscout.native as native


def _georef() -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt='PROJCS["test",GEOGCS["g",DATUM["d",SPHEROID["s",1,0]],PRIMEM["p",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=eqc +R=1 +units=m +no_defs",
        affine_transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
        width=2,
        height=2,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=None,
    )


def _scenario(tmp_path: Path) -> ls.Scenario:
    root = tmp_path / "scenario"
    (root / "horizons").mkdir(parents=True)
    ls.write_geotiff(
        root / "dem.tif",
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        _georef(),
    )
    return ls.open_scenario(root)


class _Bridge:
    def __init__(
        self,
        georef: ls.GeoReference,
        *,
        values: np.ndarray | None = None,
        validity: np.ndarray | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.georef = georef
        self.values = (
            np.asarray([[255, 0], [0, 255]], dtype=np.uint8)
            if values is None
            else values
        )
        self.validity = validity
        self.failure = failure
        self.calls: list[tuple[object, ...]] = []

    def GeneratePermanentShadowMap(
        self,
        scenario_root: str,
        dem_path: str,
        surrounding_dem_paths,
        horizons_path: str,
        output_path: str,
        progress_callback,
        cancellation_callback,
    ) -> None:
        self.calls.append(
            (
                scenario_root,
                dem_path,
                list(surrounding_dem_paths),
                horizons_path,
                output_path,
            )
        )
        progress_callback(
            SimpleNamespace(
                Stage="native_execution",
                Percent=55.0,
                Message="Fake native PSR progress.",
            )
        )
        if self.failure is not None:
            raise self.failure
        if cancellation_callback():
            raise RuntimeError("cancelled")
        ls.write_geotiff(output_path, self.values, self.georef)
        if self.validity is not None:
            with rasterio.open(output_path, "r+") as dataset:
                dataset.write_mask(np.asarray(self.validity, dtype=np.uint8))


class _OutOfOrderProgressBridge(_Bridge):
    def GeneratePermanentShadowMap(
        self,
        scenario_root: str,
        dem_path: str,
        surrounding_dem_paths,
        horizons_path: str,
        output_path: str,
        progress_callback,
        cancellation_callback,
    ) -> None:
        for percent in (60.0, 40.0, 80.0):
            progress_callback(
                SimpleNamespace(
                    Stage="native_execution",
                    Percent=percent,
                    Message="Fake native PSR progress.",
                )
            )
        assert cancellation_callback() is False
        ls.write_geotiff(output_path, self.values, self.georef)


def test_scenario_psr_generates_valid_atomic_byte_mask(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    bridge = _Bridge(_georef())
    progress: list[ls.native.NativeProductProgress] = []

    output = scenario.psr(
        "analysis/psr.tif",
        progress_callback=progress.append,
        _bridge=bridge,
    )

    values, georef = ls.read_geotiff(output)
    assert output == scenario.root / "analysis" / "psr.tif"
    assert values.dtype == np.uint8
    np.testing.assert_array_equal(values, [[255, 0], [0, 255]])
    assert georef is not None
    assert ls.same_grid(georef, _georef())
    assert [item.stage for item in progress] == [
        "preflight",
        "native_execution",
        "validate_output",
        "complete",
    ]
    assert progress[-1].percent == 100.0
    call = bridge.calls[0]
    assert call[0] == str(scenario.root)
    assert call[1] == str(scenario.dem_path())
    assert call[2] == []
    assert call[3] == str(scenario.horizons_path())
    assert Path(str(call[4])).name.startswith(".psr.tif.staging-")
    assert not list(output.parent.glob(".psr.tif.staging-*"))


def test_psr_constructs_explicit_pythonnet_callback_delegates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _scenario(tmp_path)
    bridge = _Bridge(_georef())
    constructed: list[str] = []

    def progress_delegate(callback):
        constructed.append("progress")
        return callback

    def cancellation_delegate(callback):
        constructed.append("cancellation")
        return callback

    moonlib = SimpleNamespace(
        PsrProgressCallback=progress_delegate,
        PsrCancellationCallback=cancellation_delegate,
    )
    bootstrap = SimpleNamespace(import_moonlib=lambda **_kwargs: moonlib)
    monkeypatch.setattr(native, "_create_moonlib_bridge", lambda **_kwargs: bridge)
    monkeypatch.setattr(native, "_bootstrap_module", lambda: bootstrap)

    scenario.psr("analysis/psr.tif")

    assert constructed == ["progress", "cancellation"]


def test_psr_accepts_internal_validity_mask_for_unknown_pixels(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    bridge = _Bridge(
        _georef(),
        values=np.asarray([[255, 0], [0, 255]], dtype=np.uint8),
        validity=np.asarray([[255, 0], [255, 255]], dtype=np.uint8),
    )

    output = scenario.psr("analysis/psr.tif", _bridge=bridge)

    with rasterio.open(output) as dataset:
        assert dataset.nodata is None
        np.testing.assert_array_equal(
            dataset.read_masks(1),
            [[255, 0], [255, 255]],
        )


def test_psr_normalizes_out_of_order_native_progress(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    progress: list[ls.native.NativeProductProgress] = []

    scenario.psr(
        "analysis/psr.tif",
        progress_callback=progress.append,
        _bridge=_OutOfOrderProgressBridge(_georef()),
    )

    percentages = [item.percent for item in progress]
    assert percentages == sorted(percentages)
    assert percentages == [1.0, 60.0, 60.0, 80.0, 96.0, 100.0]


def test_psr_rejects_existing_output_without_native_start(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    output = scenario.output_path("analysis/psr.tif")
    output.parent.mkdir()
    output.write_bytes(b"existing")
    bridge = _Bridge(_georef())

    with pytest.raises(ls.NativeInputError) as raised:
        scenario.psr("analysis/psr.tif", _bridge=bridge)

    assert raised.value.code == "native_psr_output_exists"
    assert output.read_bytes() == b"existing"
    assert bridge.calls == []


def test_failed_psr_overwrite_preserves_existing_output(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    output = scenario.output_path("analysis/psr.tif")
    output.parent.mkdir()
    output.write_bytes(b"existing")
    bridge = _Bridge(_georef(), failure=RuntimeError("native failure"))

    with pytest.raises(ls.NativeProductError) as raised:
        scenario.psr("analysis/psr.tif", overwrite=True, _bridge=bridge)

    assert raised.value.code == "native_psr_generation_failed"
    assert raised.value.details["error"] == "native failure"
    assert output.read_bytes() == b"existing"
    assert not list(output.parent.glob(".psr.tif.staging-*"))


def test_cancelled_psr_cleans_staging_and_preserves_existing(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    output = scenario.output_path("analysis/psr.tif")
    output.parent.mkdir()
    output.write_bytes(b"existing")
    bridge = _Bridge(_georef())
    cancelled = False

    def progress(event: ls.native.NativeProductProgress) -> None:
        nonlocal cancelled
        if event.stage == "native_execution":
            cancelled = True

    with pytest.raises(ls.NativeProductError) as raised:
        scenario.psr(
            "analysis/psr.tif",
            overwrite=True,
            progress_callback=progress,
            cancellation_requested=lambda: cancelled,
            _bridge=bridge,
        )

    assert raised.value.code == "native_psr_cancelled"
    assert output.read_bytes() == b"existing"
    assert not list(output.parent.glob(".psr.tif.staging-*"))


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[1, 0], [0, 255]], dtype=np.uint8),
        np.asarray([[1.0, 0.0], [0.0, 255.0]], dtype=np.float32),
    ],
)
def test_psr_rejects_invalid_native_mask(tmp_path: Path, values: np.ndarray) -> None:
    scenario = _scenario(tmp_path)
    bridge = _Bridge(_georef(), values=values)

    with pytest.raises(ls.NativeProductError) as raised:
        scenario.psr("analysis/psr.tif", _bridge=bridge)

    assert raised.value.code == "native_psr_values_invalid"
    assert not scenario.output_path("analysis/psr.tif").exists()
    assert not list((scenario.root / "analysis").glob(".psr.tif.staging-*"))


def test_psr_uses_scenario_path_safety(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)

    with pytest.raises(ls.ScenarioPathError):
        scenario.psr("../psr.tif", _bridge=_Bridge(_georef()))
