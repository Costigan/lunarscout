"""Lander continuous Sun: the longest continuous Sun from a landing in each pixel-month, from the light store."""

from __future__ import annotations

import contextlib
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio

import windows_scene as scene
from lunarscout._numba_horizon import lander_sun_duration as sun
from lunarscout._numba_horizon import safe_haven_stage0 as stage0
from lunarscout._numba_horizon.file_format import PATCH_SIZE
from lunarscout._numba_horizon.safe_haven_flood_packed import pack_bits, unpack_bits
from lunarscout._numba_horizon.safe_haven_lightstore import LightStore, LightStoreSettings, write_patch
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory
from lunarscout._numba_horizon.safe_haven_windows import FloodGrid
from lunarscout.errors import InputError
from lunarscout.products import _load_dem
from test_safe_haven_stage3_store import prepared            # the block scene and a 2-hour store of it

STORE_SETTINGS = LightStoreSettings(steps_per_chunk=12, codec="zlib-1")
GRID = FloodGrid(120)


def step_of(text: str) -> int:
    return int(np.datetime64(text, "us").astype(np.int64)) // GRID.step_us


# --- the rule, on hand-built bits --------------------------------------------------------------------------

FIRST = step_of("2027-01-30T00:00:00")          # step 0 is 2027-01-30 00:00; step 24 is 2027-02-01 00:00
STEPS = 60                                      # to 2027-02-04 (so the last step is 2027-02-03 22:00)


def hand_planes():
    """Plane 1 (operating) with runs at chosen pixels of patch (0, 0); plane 0 is lit everywhere, plane 2 empty."""
    lit = np.zeros((STEPS, PATCH_SIZE, PATCH_SIZE), dtype=bool)

    def run(row, column, *spans):
        for first, last in spans:
            lit[first:last + 1, row, column] = True

    run(0, 0, (0, 4))               # under way at the first step: its remainder from step 0 is known, 5 steps
    run(0, 1, (10, 13))             # 4 steps in January
    run(0, 2, (22, 30))             # starts in January (step 22 is Jan 31 20:00), 9 steps, runs into February
    run(0, 3, (5, 5), (7, 8))       # one step out of Sun ends a run: 1 and 2, never merged
    run(0, 4, (14, 16), (30, 35), (40, 41))    # January 3; February 6 and 2
    run(0, 5, (55, 59))             # still under way at the last step: credited, a lower bound (5 steps in February)
    run(0, 6, (0, STEPS - 1))       # Sun the whole time: 60 steps from January's first, 36 from February's
    run(0, 7)                       # never in Sun
    run(0, 8, (59, 59))             # lit only at the last step
    run(0, 70, (10, 12))            # the second word of the row
    run(127, 127, (24, 27))         # the first step of February, in the last corner
    run(5, 5, (23, 23), (24, 24))   # one run across midnight of Feb 1: 2 steps from January's last step, 1 from February's first
    planes = np.zeros((3, STEPS, PATCH_SIZE, 2), dtype=np.uint64)
    planes[0] = pack_bits(np.ones((STEPS, PATCH_SIZE, PATCH_SIZE), dtype=bool))
    planes[1] = pack_bits(lit)
    return planes


@pytest.fixture(scope="module")
def hand_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("hand") / "light_000_000.lsl"
    write_patch(path, hand_planes(), STORE_SETTINGS, patch_row=0, patch_column=0, first_step=FIRST)
    return path


@pytest.mark.parametrize("compiled", [True, False])
def test_the_longest_run_from_a_landing_in_each_month_on_hand_built_cases(hand_file, compiled) -> None:
    month_of, starts = sun.month_bands(FIRST, FIRST + STEPS, 120)
    assert list(starts.astype(str)) == ["2027-01", "2027-02"] and month_of[23] == 0 and month_of[24] == 1
    with LightStore(hand_file) as opened:
        out = sun.reduce_patch(opened, 1, FIRST, FIRST + STEPS, month_of, 2, PATCH_SIZE, PATCH_SIZE, compiled=compiled)
    assert out.shape == (2, PATCH_SIZE, PATCH_SIZE) and out.dtype == np.uint16
    january, february = out[0], out[1]
    assert (january[0, 0], february[0, 0]) == (5, 0)             # under way at the first step: known remainder
    assert (january[0, 1], february[0, 1]) == (4, 0)
    assert (january[0, 2], february[0, 2]) == (9, 7)             # 9 steps from step 22; 7 remain at February's first step (24..30)
    assert (january[0, 3], february[0, 3]) == (2, 0)             # no gap bridging
    assert (january[0, 4], february[0, 4]) == (3, 6)
    assert (january[0, 5], february[0, 5]) == (0, 5)             # truncated at the end, credited
    assert (january[0, 6], february[0, 6]) == (60, 36)           # a period decays by a month per band, never 0 inside it
    assert (january[0, 7], february[0, 7]) == (0, 0)
    assert (january[0, 8], february[0, 8]) == (0, 1)
    assert (january[0, 70], february[0, 70]) == (3, 0)           # a pixel in the second word
    assert (january[127, 127], february[127, 127]) == (0, 4)
    assert (january[5, 5], february[5, 5]) == (2, 1)
    untouched = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    for row, column in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 70), (127, 127), (5, 5)]:
        untouched[row, column] = False
    assert not out[:, untouched].any()


def test_the_plane_decides_what_sun_is(hand_file) -> None:
    month_of, _ = sun.month_bands(FIRST, FIRST + STEPS, 120)
    with LightStore(hand_file) as opened:
        partial = sun.reduce_patch(opened, 0, FIRST, FIRST + STEPS, month_of, 2, PATCH_SIZE, PATCH_SIZE)
    assert (partial == np.array([60, 36], dtype=np.uint16)[:, None, None]).all()      # plane 0 is lit at every step


def test_a_valid_window_crops_the_tile(hand_file) -> None:
    month_of, _ = sun.month_bands(FIRST, FIRST + STEPS, 120)
    with LightStore(hand_file) as opened:
        out = sun.reduce_patch(opened, 1, FIRST, FIRST + STEPS, month_of, 2, 6, 10)
    assert out.shape == (2, 6, 10) and out[0, 0, 1] == 4 and out[1, 0, 4] == 6


def test_a_run_longer_than_the_count_saturates() -> None:
    planes = pack_bits(np.ones((10, 1, 128), dtype=bool)).reshape(10, 1, 2)
    run_start = np.full(128, -1, np.int32)
    best = np.zeros((1, 128), np.uint16)
    month_of = np.zeros(10, dtype=np.int16)
    month_first = np.zeros(1, dtype=np.int32)
    sun.advance_python(planes, 0, month_of, month_first, run_start, best, 3)
    sun.finish_python(10, month_of, month_first, run_start, best, 3)
    assert (best == 3).all()


def test_the_month_bands_name_every_calendar_month_of_the_range() -> None:
    first, stop = step_of("2027-11-20T00:00:00"), step_of("2028-03-02T00:00:00")
    month_of, starts = sun.month_bands(first, stop, 120)
    assert list(starts.astype(str)) == ["2027-11", "2027-12", "2028-01", "2028-02", "2028-03"]
    assert month_of[0] == 0 and month_of[-1] == 4 and month_of.dtype == np.int16
    assert (np.diff(month_of) >= 0).all()


def test_the_threshold_must_be_one_the_store_holds() -> None:
    assert sun.plane_for_threshold(STORE_SETTINGS, 1.0) == 1 and sun.plane_for_threshold(STORE_SETTINGS, 0.2) == 0
    with pytest.raises(InputError) as caught:
        sun.plane_for_threshold(STORE_SETTINGS, 0.5)
    assert caught.value.code == "lander_sun_threshold_unavailable"
    custom = LightStoreSettings(operating_fraction=0.99, codec="zlib-1")
    assert sun.plane_for_threshold(custom, 0.99) == 1
    for bad in (-0.1, 1.5):
        with pytest.raises(InputError):
            sun.SunDurationSettings(threshold=bad)
    for name in ("workers", "checkpoint_patches"):
        with pytest.raises(InputError):
            sun.SunDurationSettings(**{name: 0})


# --- the product on the scene -----------------------------------------------------------------------------


def reference(prepared, first, stop, plane):
    """The product by an independent route: unpack every patch and, for every step in Sun, take the rest of its run."""
    month_of, starts = sun.month_bands(first, stop, 120)
    out = np.zeros((len(starts), prepared.dem.height, prepared.dem.width), dtype=np.uint16)
    for pr in range(3):
        for pc in range(3):
            with prepared.light_store.open_patch(pr, pc) as opened:
                lit = unpack_bits(opened.read_steps(first, stop)[plane], PATCH_SIZE)
            height = min(PATCH_SIZE, prepared.dem.height - pr * PATCH_SIZE)
            width = min(PATCH_SIZE, prepared.dem.width - pc * PATCH_SIZE)
            for r in range(height):
                for c in range(width):
                    padded = np.concatenate(([False], lit[:, r, c], [False]))
                    begins = np.flatnonzero(~padded[:-1] & padded[1:])
                    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
                    for begin, end in zip(begins, ends):
                        for step in range(begin, end):            # land at every step in Sun: the rest of the run
                            cell = out[month_of[step], pr * PATCH_SIZE + r, pc * PATCH_SIZE + c]
                            out[month_of[step], pr * PATCH_SIZE + r, pc * PATCH_SIZE + c] = max(cell, end - step)
    return out


def georef_of(prepared):
    return _load_dem(prepared.directory / "dem.tif")[1]


def run_product(prepared, name, **options):
    settings = options.pop("settings", sun.SunDurationSettings(workers=3, checkpoint_patches=4, status_seconds=0))
    path = prepared.directory / f"{name}.tif"
    report = sun.generate_sun_duration(path, directory=prepared.light_store, dem=prepared.dem,
                                       georef=georef_of(prepared), settings=settings, overwrite=True, **options)
    return path, report


def read_all(path):
    """``(values, info)``: the bands, and the metadata read while the file is open."""
    with rasterio.open(path) as dataset:
        info = types.SimpleNamespace(nodata=dataset.nodata, tags=dataset.tags(), band_tags=dataset.tags(1),
                                     masks=dataset.read_masks(1), count=dataset.count, dtypes=dataset.dtypes)
        return dataset.read(), info


@pytest.mark.parametrize("threshold, plane", [(1.0, 1), (0.2, 0)])
def test_the_product_equals_the_reference_on_the_scene(prepared, threshold, plane) -> None:
    settings = sun.SunDurationSettings(threshold=threshold, start_utc="2026-12-20T00:00:00", stop_utc="2027-04-10T00:00:00",
                                       workers=3, checkpoint_patches=4, status_seconds=0)
    path, report = run_product(prepared, f"scene_{plane}", settings=settings)
    first, stop = step_of("2026-12-20T00:00:00"), step_of("2027-04-10T00:00:00")
    values, dataset = read_all(path)
    assert (report.status, report.patches_total, report.patches_written, report.bands) == ("completed", 9, 9, 5)
    assert values.dtype == np.uint16 and values.shape == (5, prepared.dem.height, prepared.dem.width)
    expected = reference(prepared, first, stop, plane)
    assert np.array_equal(values, expected)
    assert values.max() > 12 and len(np.unique(values)) > 2                      # runs of more than a day, and varied values
    assert dataset.nodata == sun.NODATA and not (values == sun.NODATA).any()
    stamps = dataset.tags["LUNARSCOUT_TIMESTAMPS_UTC"]
    assert stamps.count("T00:00:00") == 5 and "2026-12-01" in stamps and "2027-04-01" in stamps
    assert dataset.band_tags["TIMESTAMP_UTC"].startswith("2026-12-01") and dataset.count == 5 and set(dataset.dtypes) == {"uint16"}


def test_the_whole_store_range_is_the_default(prepared) -> None:
    path, report = run_product(prepared, "whole")
    assert report.bands == 5 and report.first_step == prepared.light_store.first_step
    assert report.step_count == prepared.light_store.step_count
    assert np.array_equal(read_all(path)[0], reference(prepared, prepared.light_store.first_step,
                                                       prepared.light_store.stop_step, 1))


def test_workers_do_not_change_the_product(prepared) -> None:
    one, _ = run_product(prepared, "w1", settings=sun.SunDurationSettings(workers=1, checkpoint_patches=100, status_seconds=0))
    many, _ = run_product(prepared, "w5", settings=sun.SunDurationSettings(workers=5, checkpoint_patches=1, status_seconds=0))
    assert np.array_equal(read_all(one)[0], read_all(many)[0])


def test_a_missing_store_file_and_an_invalid_dem_node_are_nodata(prepared, tmp_path) -> None:
    copy = tmp_path / "store"
    shutil.copytree(prepared.store_directory, copy)
    stage0.patch_file(copy, 1, 1).unlink()
    dem, georef = _load_dem(prepared.directory / "dem.tif")
    dem.elevation_m[5, 7] = np.nan
    directory = LightStoreDirectory(copy)
    path = tmp_path / "nodata.tif"
    report = sun.generate_sun_duration(path, directory=directory, dem=dem, georef=georef,
                                       settings=sun.SunDurationSettings(workers=2, status_seconds=0))
    values, dataset = read_all(path)
    assert report.patches_without_store_file == 1
    assert (values[:, 128:256, 128:256] == sun.NODATA).all()                  # the patch without a file
    assert (values[:, 5, 7] == sun.NODATA).all()                              # the invalid DEM node
    good = values[:, 0:128, 0:128].copy()
    good[:, 5, 7] = 0
    assert (good != sun.NODATA).all()
    assert dataset.masks[5, 7] == 0 and dataset.masks[10, 10] == 255


def test_an_interrupted_run_resumes_to_the_same_product(prepared, monkeypatch) -> None:
    whole, _ = run_product(prepared, "uninterrupted")
    interrupts: list = []

    @contextlib.contextmanager
    def fake_interrupt():
        yield interrupts

    original = sun.reduce_patch
    calls: list = []

    def interrupting(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            interrupts.append(2)                                  # a Ctrl-C arrives while the third patch is reduced
        return original(*args, **kwargs)

    monkeypatch.setattr(sun, "_graceful_interrupt", fake_interrupt)
    monkeypatch.setattr(sun, "reduce_patch", interrupting)
    path = prepared.directory / "resumed.tif"
    settings = sun.SunDurationSettings(workers=1, checkpoint_patches=1, status_seconds=0)
    first = sun.generate_sun_duration(path, directory=prepared.light_store, dem=prepared.dem, georef=georef_of(prepared),
                                      settings=settings, overwrite=True)
    assert first.status == "interrupted" and 3 <= first.patches_written < 9 and not path.exists()
    monkeypatch.setattr(sun, "reduce_patch", original)
    interrupts.clear()
    second = sun.generate_sun_duration(path, directory=prepared.light_store, dem=prepared.dem, georef=georef_of(prepared),
                                       settings=settings, overwrite=True)
    assert second.status == "completed" and second.patches_already_done == first.patches_written
    assert first.patches_written + second.patches_written == 9
    assert np.array_equal(read_all(path)[0], read_all(whole)[0])


def test_a_subset_of_patches_is_not_finalized(prepared, tmp_path) -> None:
    path = tmp_path / "subset.tif"
    report = sun.generate_sun_duration(path, directory=prepared.light_store, dem=prepared.dem, georef=georef_of(prepared),
                                       settings=sun.SunDurationSettings(workers=1, status_seconds=0), patches=[(0, 0), (2, 1)])
    assert report.patches_written == 2 and not path.exists() and report.notes
    assert (tmp_path / ".subset.tif.lunarscout-partial.tif").exists()


def test_refusals(prepared, tmp_path) -> None:
    kwargs = dict(directory=prepared.light_store, dem=prepared.dem, georef=georef_of(prepared))
    cases = [
        (sun.SunDurationSettings(threshold=0.5), "lander_sun_threshold_unavailable"),
        (sun.SunDurationSettings(start_utc="2026-12-20T01:00:00"), "lander_sun_time_misaligned"),
        (sun.SunDurationSettings(start_utc="2026-11-01T00:00:00"), "lander_sun_time_range"),
        (sun.SunDurationSettings(stop_utc="2027-06-01T00:00:00"), "lander_sun_time_range"),
        (sun.SunDurationSettings(start_utc="2027-03-01T00:00:00", stop_utc="2027-02-01T00:00:00"), "lander_sun_time_range"),
    ]
    for settings, code in cases:
        with pytest.raises(InputError) as caught:
            sun.generate_sun_duration(tmp_path / "x.tif", settings=settings, **kwargs)
        assert caught.value.code == code
    assert not list(tmp_path.glob("*x.tif*"))
    existing, _ = run_product(prepared, "exists")
    with pytest.raises(Exception, match="exists"):
        sun.generate_sun_duration(existing, settings=sun.SunDurationSettings(status_seconds=0), **kwargs)


def test_a_different_threshold_or_range_is_not_resumed(prepared, monkeypatch) -> None:
    path = prepared.directory / "mismatch.tif"
    kwargs = dict(directory=prepared.light_store, dem=prepared.dem, georef=georef_of(prepared))
    monkeypatch.setattr(sun, "_graceful_interrupt", lambda: contextlib.nullcontext([1]))        # stop after the first patches
    first = sun.generate_sun_duration(path, settings=sun.SunDurationSettings(workers=1, status_seconds=0), overwrite=True, **kwargs)
    assert first.status == "interrupted" or first.patches_written == 0
    monkeypatch.undo()
    with pytest.raises(Exception):
        sun.generate_sun_duration(path, settings=sun.SunDurationSettings(threshold=0.2, status_seconds=0), overwrite=True, **kwargs)


# --- the script -------------------------------------------------------------------------------------------


def test_the_script_writes_the_product_and_importing_it_starts_nothing(prepared, tmp_path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_lander_continuous_sun", Path(__file__).resolve().parents[2] / "scripts" / "run_lander_continuous_sun.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.THRESHOLD == 1.0 and module.LIGHT_STORE_PATH == load_stage0_output() and module.OVERWRITE is True      # the rerun replaces the old product
    out = tmp_path / "lander.tif"
    for name, value in dict(DEM_PATH=prepared.directory / "dem.tif", LIGHT_STORE_PATH=prepared.store_directory,
                            OUTPUT_PATH=out, WORKERS=2, CHECKPOINT_PATCHES=3, STATUS_SECONDS=0.0, OVERWRITE=False).items():
        monkeypatch.setattr(module, name, value)
    assert module.main() == 0
    assert np.array_equal(read_all(out)[0], reference(prepared, prepared.light_store.first_step, prepared.light_store.stop_step, 1))
    assert "Lander sun duration" in (tmp_path / "lander.tif.log").read_text()
    with pytest.raises(Exception):
        module.main()                                                  # the output exists


def load_stage0_output():
    spec = importlib.util.spec_from_file_location(
        "run_light_store", Path(__file__).resolve().parents[2] / "scripts" / "run_light_store.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OUTPUT_DIRECTORY
