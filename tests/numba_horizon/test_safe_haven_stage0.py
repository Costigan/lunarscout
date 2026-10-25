"""Stage 0: the driver that lights every patch once and the reader of its directory."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil

import numpy as np
import pytest

import blocks_scene as bs
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_lightstore as lightstore
from lunarscout._numba_horizon import safe_haven_stage0 as stage0
from lunarscout._numba_horizon.cuda_backend import CudaBackendError
from lunarscout._numba_horizon.file_format import PATCH_SIZE, HorizonTileStore, read_patch_horizons
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.safe_haven_lightstore import LightStoreSettings
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory, Stage0Settings
from lunarscout.errors import InputError, ProductStorageError
from lunarscout.products import _load_dem

START, STOP = "2027-01-10T00:00:00", "2027-01-20T00:00:00"        # 120 steps of two hours
ALL_PATCHES = [(row, column) for row in range(3) for column in range(3)]


def settings_for(**kwargs) -> Stage0Settings:
    store = LightStoreSettings(steps_per_chunk=kwargs.pop("steps_per_chunk", 48), codec=kwargs.pop("codec", "zlib-1"))
    kwargs.setdefault("lighting", "cpu")
    kwargs.setdefault("workers", 2)
    return Stage0Settings(store=store, start_utc=START, stop_utc=STOP, **kwargs)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("stage0_scene")
    dem_path, horizons_path = bs.write_inputs(directory)
    dem, _georef = _load_dem(dem_path)
    return dict(dem=dem, store=HorizonTileStore(horizons_path))


def run(inputs, directory, settings=None, **options):
    return stage0.generate_light_store(
        directory, dem=inputs["dem"], horizon_store=inputs["store"], settings=settings or settings_for(),
        sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors, **options)


def expected_planes(inputs, patch, settings):
    grid_steps = np.arange(settings.first_step, settings.first_step + settings.step_count)
    stamps = stage0.FloodGrid(settings.store.step_minutes).times(grid_steps)
    horizons = read_patch_horizons(inputs["store"], patch[0] * PATCH_SIZE, patch[1] * PATCH_SIZE, 0.0)
    return lightstore.light_patch_planes(LightmapCpuSession(), inputs["dem"], horizons, patch[0], patch[1],
                                         scene.sun_vectors(stamps), scene.earth_vectors(stamps), settings.store)


@pytest.fixture
def made(inputs, tmp_path):
    settings = settings_for()
    report = run(inputs, tmp_path, settings)
    return tmp_path, settings, report


# --- a full run -------------------------------------------------------------------------------------


def test_every_patch_is_lit_and_stored_exactly(inputs, made) -> None:
    directory, settings, report = made
    assert settings.step_count == 120
    assert (report.status, report.patches_total, report.patches_written, report.patches_already_done) == ("completed", 9, 9, 0)
    assert report.patches_without_horizons == [] and report.lighting == "cpu" and report.bytes_written > 0
    store = LightStoreDirectory(directory, inputs["dem"])
    assert (store.first_step, store.step_count, store.patch_grid) == (settings.first_step, 120, (3, 3))
    seen_set = seen_clear = False
    for patch in ALL_PATCHES:
        with store.open_patch(*patch) as opened:
            planes = opened.read_steps(store.first_step, store.stop_step)
            assert np.array_equal(planes, expected_planes(inputs, patch, settings)), patch
            assert np.array_equal(opened.read_steps(store.first_step + 50, store.first_step + 101), planes[:, 50:101])
        seen_set |= bool(planes.any())
        seen_clear |= bool((~planes).any())
    assert seen_set and seen_clear
    written = json.loads((directory / stage0.REPORT_NAME).read_text())
    assert written["format"] == stage0.REPORT_FORMAT and written["patches_written"] == 9
    assert written["settings"]["store"]["codec"] == "zlib-1"
    manifest = json.loads((directory / stage0.MANIFEST_NAME).read_text())
    assert manifest["patch_grid"] == [3, 3] and manifest["dem"]["shape"] == [300, 300]
    assert sorted(entry.name for entry in directory.iterdir()) == [
        "000", "001", "002", stage0.MANIFEST_NAME, stage0.REPORT_NAME]               # no staging files


def test_the_files_do_not_depend_on_the_number_of_workers(inputs, tmp_path) -> None:
    one, four = tmp_path / "one", tmp_path / "four"
    one.mkdir(), four.mkdir()
    run(inputs, one, settings_for(workers=1))
    run(inputs, four, settings_for(workers=4))
    for patch in ALL_PATCHES:
        assert stage0.patch_file(one, *patch).read_bytes() == stage0.patch_file(four, *patch).read_bytes()


def test_a_rerun_lights_nothing(inputs, made, monkeypatch) -> None:
    directory, settings, _ = made
    monkeypatch.setattr(lightstore, "light_patch_planes", lambda *a, **k: pytest.fail("a stored patch was lit again"))
    before = {path: path.read_bytes() for path in directory.glob("*/*.lsl")}
    report = run(inputs, directory, settings)
    assert (report.status, report.patches_written, report.patches_already_done) == ("completed", 0, 9)
    assert {path: path.read_bytes() for path in directory.glob("*/*.lsl")} == before


# --- resuming ---------------------------------------------------------------------------------------


def test_a_failure_keeps_finished_patches_and_the_rerun_finishes(inputs, tmp_path, monkeypatch) -> None:
    real = lightstore.light_patch_planes
    calls = []

    def flaky(session, dem, horizons, row, column, *args, **kwargs):
        calls.append((row, column))
        if len(calls) == 4:
            raise RuntimeError("the lighting failed")
        return real(session, dem, horizons, row, column, *args, **kwargs)

    monkeypatch.setattr(lightstore, "light_patch_planes", flaky)
    with pytest.raises(RuntimeError, match="the lighting failed"):
        run(inputs, tmp_path, settings_for(workers=1))
    failed = json.loads((tmp_path / stage0.REPORT_NAME).read_text())
    assert failed["status"] == "failed" and "the lighting failed" in failed["error"]
    stored = list(tmp_path.glob("*/*.lsl"))
    assert 1 <= len(stored) < 9 and not list(tmp_path.glob("**/*.partial-*"))
    monkeypatch.setattr(lightstore, "light_patch_planes", real)
    report = run(inputs, tmp_path, settings_for(workers=1))
    assert report.status == "completed" and report.patches_already_done == len(stored)
    assert report.patches_written == 9 - len(stored) and len(list(tmp_path.glob("*/*.lsl"))) == 9


def test_an_interrupt_finishes_what_is_running_and_the_rerun_resumes(inputs, tmp_path, monkeypatch) -> None:
    interrupts = []

    @contextlib.contextmanager
    def fake_interrupt():
        yield interrupts

    real = lightstore.light_patch_planes
    calls = []

    def interrupting(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            interrupts.append(2)                                  # SIGINT arrives while the second patch is lit
        return real(*args, **kwargs)

    monkeypatch.setattr(stage0, "_graceful_interrupt", fake_interrupt)
    monkeypatch.setattr(lightstore, "light_patch_planes", interrupting)
    first = run(inputs, tmp_path, settings_for(workers=1))
    assert first.status == "interrupted" and 2 <= first.patches_written < 9
    monkeypatch.setattr(lightstore, "light_patch_planes", real)
    interrupts.clear()
    second = run(inputs, tmp_path, settings_for(workers=1))
    assert second.status == "completed" and second.patches_already_done == first.patches_written
    assert first.patches_written + second.patches_written == 9


def test_unusable_and_misplaced_files_are_replaced(inputs, made, caplog) -> None:
    directory, settings, _ = made
    truncated, misplaced = stage0.patch_file(directory, 0, 1), stage0.patch_file(directory, 2, 2)
    good = truncated.read_bytes()
    truncated.write_bytes(good[:-9])
    shutil.copy(stage0.patch_file(directory, 0, 0), misplaced)          # a valid file for another patch
    with caplog.at_level(logging.WARNING, logger=stage0.logger.name):
        report = run(inputs, directory, settings)
    assert report.patches_written == 2 and report.patches_already_done == 7
    assert caplog.text.count("replacing the unusable") == 2
    assert truncated.read_bytes() == good
    with LightStoreDirectory(directory).open_patch(2, 2) as opened:
        assert opened.patch == (2, 2)


def test_staging_files_from_a_killed_run_are_removed(inputs, made) -> None:
    directory, settings, _ = made
    leftover = directory / "001" / "light_001_001.lsl.partial-4242"
    leftover.write_bytes(b"x")
    (directory / "light_store.json.partial-4242").write_text("{")
    run(inputs, directory, settings)
    assert not leftover.exists() and not (directory / "light_store.json.partial-4242").exists()


# --- refusing a different store ---------------------------------------------------------------------


@pytest.mark.parametrize("change", [
    dict(codec="zstd-3"), dict(steps_per_chunk=24), dict(start_utc="2027-01-11T00:00:00"),
])
def test_a_store_made_with_other_settings_is_refused(inputs, made, change) -> None:
    directory, settings, _ = made
    if "start_utc" in change:
        other = Stage0Settings(store=settings.store, start_utc=change["start_utc"], stop_utc=STOP, lighting="cpu")
    else:
        other = settings_for(**change)
    with pytest.raises(ProductStorageError) as error:
        run(inputs, directory, other)
    assert error.value.code == "light_store_manifest_mismatch" and error.value.details["differing"]
    assert len(list(directory.glob("*/*.lsl"))) == 9                  # nothing was touched


def test_a_store_made_for_another_dem_is_refused(inputs, made) -> None:
    directory, settings, _ = made
    heights = inputs["dem"].elevation_m
    changed = type(inputs["dem"])(heights + np.float32(1.0), inputs["dem"].geo_transform, inputs["dem"].projection)
    with pytest.raises(ProductStorageError) as error:
        stage0.generate_light_store(directory, dem=changed, horizon_store=inputs["store"], settings=settings,
                                    sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors)
    assert error.value.details["differing"] == ["dem"]
    with pytest.raises(ProductStorageError) as reader:
        LightStoreDirectory(directory, changed)
    assert reader.value.code == "light_store_manifest_mismatch"


def test_a_damaged_manifest_is_reported(inputs, made) -> None:
    directory, settings, _ = made
    (directory / stage0.MANIFEST_NAME).write_text("{not json")
    with pytest.raises(ProductStorageError) as error:
        run(inputs, directory, settings)
    assert error.value.code == "light_store_manifest_unreadable"
    with pytest.raises(ProductStorageError):
        LightStoreDirectory(directory)


# --- patches ----------------------------------------------------------------------------------------


def test_a_subset_of_patches_and_bad_patch_lists(inputs, tmp_path) -> None:
    report = run(inputs, tmp_path, patches=[(2, 1), (0, 0)])
    assert (report.patches_total, report.patches_written) == (2, 2)
    store = LightStoreDirectory(tmp_path)
    assert store.open_patch(1, 1) is None                              # not lit: no file
    with store.open_patch(2, 1) as opened:
        assert opened.patch == (2, 1)
    with pytest.raises(InputError) as outside:
        run(inputs, tmp_path, patches=[(3, 0)])
    assert outside.value.code == "light_store_patch_outside"
    with pytest.raises(InputError) as twice:
        run(inputs, tmp_path, patches=[(0, 0), (0, 0)])
    assert twice.value.code == "light_store_patch_duplicate"


def test_patches_without_horizon_tiles_are_recorded_and_not_stored(inputs, tmp_path) -> None:
    empty = HorizonTileStore(tmp_path / "no_horizons")
    out = tmp_path / "store"
    out.mkdir()
    report = stage0.generate_light_store(out, dem=inputs["dem"], horizon_store=empty, settings=settings_for(),
                                         sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors)
    assert report.status == "completed" and report.patches_written == 0 and report.bytes_written == 0
    assert sorted(map(tuple, report.patches_without_horizons)) == ALL_PATCHES
    assert not list(out.glob("*/*.lsl"))
    assert all(LightStoreDirectory(out).open_patch(*patch) is None for patch in ALL_PATCHES)


def test_a_missing_directory_is_an_error(inputs, tmp_path) -> None:
    with pytest.raises(ProductStorageError) as error:
        run(inputs, tmp_path / "absent")
    assert error.value.code == "light_store_directory_missing"


# --- choosing the lighting --------------------------------------------------------------------------


class FakeGpu:
    """Stands in for the CUDA session: computes the planes on the CPU, so the driver's GPU route runs."""

    def __init__(self) -> None:
        self.calls = 0

    def light_patch_planes(self, dem, horizons, sun, earth, settings, *, tile_y, tile_x, valid_height, valid_width):
        self.calls += 1
        return lightstore.light_patch_planes(
            LightmapCpuSession(), dem, horizons, tile_y // PATCH_SIZE, tile_x // PATCH_SIZE, sun, earth, settings)


def test_cpu_lighting_never_touches_cuda(inputs, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(stage0, "_make_gpu_session", lambda settings: pytest.fail("CUDA was probed"))
    assert run(inputs, tmp_path, settings_for(lighting="cpu")).lighting == "cpu"


def test_the_gpu_session_is_used_when_it_is_there(inputs, tmp_path, monkeypatch) -> None:
    gpu = FakeGpu()
    monkeypatch.setattr(stage0, "_make_gpu_session", lambda settings: gpu)
    for lighting in ("auto", "cuda"):
        directory = tmp_path / lighting
        directory.mkdir()
        report = run(inputs, directory, settings_for(lighting=lighting))
        assert report.lighting == "cuda"
    assert gpu.calls == 18
    for patch in ALL_PATCHES:                                          # and it gives the same files
        assert stage0.patch_file(tmp_path / "auto", *patch).read_bytes() == stage0.patch_file(tmp_path / "cuda", *patch).read_bytes()


def test_cuda_never_falls_back_but_auto_does(inputs, tmp_path, monkeypatch, caplog) -> None:
    def broken(settings):
        raise CudaBackendError("no device")

    monkeypatch.setattr(stage0, "_make_gpu_session", broken)
    forced = tmp_path / "forced"
    forced.mkdir()
    with pytest.raises(CudaBackendError):
        run(inputs, forced, settings_for(lighting="cuda"))
    assert not list(forced.glob("*/*.lsl")) and json.loads((forced / stage0.REPORT_NAME).read_text())["status"] == "failed"
    fallback = tmp_path / "auto"
    fallback.mkdir()
    with caplog.at_level(logging.WARNING, logger=stage0.logger.name):
        report = run(inputs, fallback, settings_for(lighting="auto"))
    assert report.lighting == "cpu" and "CUDA lighting is not available" in caplog.text


@pytest.mark.parametrize("bad", [
    dict(lighting="gpu"), dict(workers=0), dict(gpu_chunks_per_batch=0), dict(status_seconds=0.0),
    dict(start_utc=STOP, stop_utc=START),
])
def test_invalid_settings(bad) -> None:
    kwargs = dict(start_utc=START, stop_utc=STOP)
    kwargs.update(bad)
    with pytest.raises(InputError):
        Stage0Settings(**kwargs)
