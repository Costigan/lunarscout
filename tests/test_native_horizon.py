from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lunarscout as ls
import lunarscout.native as native


class _FakeDotNetList(list):
    @property
    def Count(self) -> int:
        return len(self)


class _Task:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def GetAwaiter(self):
        self.calls.append("GetAwaiter")
        return self

    def GetResult(self) -> None:
        self.calls.append("GetResult")


class _ElevationMap:
    loaded_paths: list[str] = []

    def __init__(self, path: str) -> None:
        self.path = path
        self.__class__.loaded_paths.append(path)


class _QuadTreeHorizonGenerator:
    generated_patch_list: _FakeDotNetList | None = None
    removed_patch_list: _FakeDotNetList | None = None
    instances: list["_QuadTreeHorizonGenerator"] = []

    def __init__(self, disable_hierarchy: bool) -> None:
        self.disable_hierarchy = disable_hierarchy
        self.calls: list[tuple[object, ...]] = []
        self.await_calls: list[str] = []
        self.disposed = False
        self.__class__.instances.append(self)

    @staticmethod
    def GeneratePatchList(primary_dem: _ElevationMap):
        _QuadTreeHorizonGenerator.generated_patch_list = _FakeDotNetList(
            [SimpleNamespace(Index=0)]
        )
        return _QuadTreeHorizonGenerator.generated_patch_list

    @staticmethod
    def RemoveCompletedPatches(patches, directory: str, observer_elevation: float):
        _QuadTreeHorizonGenerator.removed_patch_list = _FakeDotNetList(list(patches))
        _QuadTreeHorizonGenerator.removed_patch_list.remove_args = (
            patches,
            directory,
            observer_elevation,
        )
        return _QuadTreeHorizonGenerator.removed_patch_list

    def GenerateHorizonsForPatches(
        self,
        output_directory: str,
        dems,
        patches,
        observer_elevation: float,
        compress_horizons: bool,
        progress_callback,
        cancellation_callback,
    ):
        self.calls.append(
            (
                output_directory,
                list(dems),
                patches,
                observer_elevation,
                compress_horizons,
                progress_callback,
                cancellation_callback,
            )
        )
        if progress_callback is not None:
            progress_callback(
                SimpleNamespace(
                    ProcessedPatches=1,
                    TotalPatches=1,
                    Percent=100.0,
                    Stage="complete",
                    Message="done",
                    FileName="horizon_0_0.bin",
                )
            )
        return _Task(self.await_calls)

    def Dispose(self) -> None:
        self.disposed = True


_ORIGINAL_REMOVE_COMPLETED = _QuadTreeHorizonGenerator.RemoveCompletedPatches


def _fake_modules():
    _ElevationMap.loaded_paths = []
    _QuadTreeHorizonGenerator.generated_patch_list = None
    _QuadTreeHorizonGenerator.removed_patch_list = None
    _QuadTreeHorizonGenerator.instances = []
    _QuadTreeHorizonGenerator.RemoveCompletedPatches = _ORIGINAL_REMOVE_COMPLETED
    moonlib = SimpleNamespace(
        HorizonProgressCallback=lambda callback: callback,
        HorizonCancellationCallback=lambda callback: callback,
    )
    horizon = SimpleNamespace(
        ElevationMap=_ElevationMap,
        QuadTreeHorizonGenerator=_QuadTreeHorizonGenerator,
    )
    return moonlib, horizon


def test_generate_horizons_is_available_from_root_and_native_namespace() -> None:
    assert ls.GenerateHorizons is native.GenerateHorizons
    assert ls.NativeHorizonProgress is native.NativeHorizonProgress


def test_generate_horizons_calls_direct_quadtree_pipeline(tmp_path: Path) -> None:
    primary = tmp_path / "primary.tif"
    surrounding = tmp_path / "surrounding.tif"
    primary.write_bytes(b"dem")
    surrounding.write_bytes(b"dem")
    output = tmp_path / "horizons"
    moonlib, horizon = _fake_modules()
    progress: list[native.NativeHorizonProgress] = []

    result = native.GenerateHorizons(
        output,
        [primary, surrounding],
        observer_elevation=2.5,
        compress_horizons=True,
        progress_callback=progress.append,
        cancellation_requested=lambda: False,
        _moonlib=moonlib,
        _horizon_module=horizon,
    )

    assert result == output.resolve()
    assert output.is_dir()
    assert _ElevationMap.loaded_paths == [
        str(primary.resolve()),
        str(surrounding.resolve()),
    ]
    instance = _QuadTreeHorizonGenerator.instances[0]
    assert instance.disable_hierarchy is False
    assert instance.await_calls == ["GetAwaiter", "GetResult"]
    assert instance.disposed is True
    remove_args = _QuadTreeHorizonGenerator.removed_patch_list.remove_args
    assert remove_args == (
        _QuadTreeHorizonGenerator.generated_patch_list,
        str(output.resolve()),
        2.5,
    )
    call = instance.calls[0]
    assert call[0] == str(output.resolve())
    assert [dem.path for dem in call[1]] == [
        str(primary.resolve()),
        str(surrounding.resolve()),
    ]
    assert call[2] is _QuadTreeHorizonGenerator.removed_patch_list
    assert call[3] == 2.5
    assert call[4] is True
    assert progress == [
        native.NativeHorizonProgress(
            processed_patches=1,
            total_patches=1,
            percent=100.0,
            stage="complete",
            message="done",
            file_name="horizon_0_0.bin",
        )
    ]


def test_generate_horizons_skips_native_generator_when_no_patches(tmp_path: Path) -> None:
    primary = tmp_path / "primary.tif"
    primary.write_bytes(b"dem")
    output = tmp_path / "horizons"
    moonlib, horizon = _fake_modules()

    def skip_existing_patches(_patches, _directory: str, _observer_elevation: float):
        return _FakeDotNetList()

    horizon.QuadTreeHorizonGenerator.RemoveCompletedPatches = staticmethod(
        skip_existing_patches
    )

    result = native.GenerateHorizons(
        output,
        [primary],
        _moonlib=moonlib,
        _horizon_module=horizon,
    )

    assert result == output.resolve()
    assert _QuadTreeHorizonGenerator.instances == []


def test_generate_horizons_can_process_all_patches_without_skip_existing(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.tif"
    primary.write_bytes(b"dem")
    output = tmp_path / "horizons"
    moonlib, horizon = _fake_modules()

    def fail_if_called(_patches, _directory: str, _observer_elevation: float):
        raise AssertionError("RemoveCompletedPatches should not be called")

    horizon.QuadTreeHorizonGenerator.RemoveCompletedPatches = staticmethod(
        fail_if_called
    )

    native.GenerateHorizons(
        output,
        [primary],
        skip_existing=False,
        _moonlib=moonlib,
        _horizon_module=horizon,
    )

    instance = _QuadTreeHorizonGenerator.instances[0]
    assert instance.calls[0][2] is _QuadTreeHorizonGenerator.generated_patch_list


def test_generate_horizons_rejects_missing_dem(tmp_path: Path) -> None:
    with pytest.raises(ls.NativeInputError) as raised:
        native.GenerateHorizons(tmp_path / "horizons", [tmp_path / "missing.tif"])

    assert raised.value.code == "native_horizons_input_missing"
