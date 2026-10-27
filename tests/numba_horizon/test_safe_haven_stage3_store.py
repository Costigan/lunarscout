"""Stage 3 fed from the stage-0 light store: the table the horizon route gives, without a horizon.

The scene is the block scene at the production step: stage 1 at 2-hour samples, a store on that grid,
the flood on a 2-hour step with 4 moves. The horizon route (the CPU lightmap of each patch, the
reference) and the store route must write the same table.
"""

from __future__ import annotations

import csv
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest

import blocks_scene as bs
import lunarscout as ls
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_blocks as blocks
from lunarscout._numba_horizon import safe_haven_stage0 as stage0
from lunarscout._numba_horizon import safe_haven_stage3_store as from_store
from lunarscout._numba_horizon.file_format import PATCH_SIZE, HorizonTileStore, read_patch_horizons
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.safe_haven_lightstore import LightStoreSettings, patch_geometry
from lunarscout._numba_horizon.safe_haven_stage1_store import flat_earth_elevation
from lunarscout._numba_horizon.safe_haven_litstore import LitStore, fill_lit_store
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory, Stage0Settings
from lunarscout._numba_horizon.safe_haven_stage3 import PipelineSettings
from lunarscout._numba_horizon.safe_haven_windows import (
    FloodGrid, PixelSignals, Stage1Rasters, WindowParameters, spice_vector_source)
from lunarscout.errors import InputError
from lunarscout.products import _load_dem

FIRST, LAST = "2027-01-01T00:00:00Z", "2027-03-31T22:00:00Z"          # stage 1: 1,080 samples of 2 hours
STORE_START, STORE_STOP = "2026-12-01T00:00:00", "2027-05-01T00:00:00"  # the store reaches past it, so a flood window never leaves it
OPERATIONAL_HOURS = 24.0
FAST = dict(plan_workers=2, light_workers=3, numba_threads_per_light_worker=1, flood_workers=2)


def parameters(block_patches=1, **overrides):
    settings = dict(cell_patches=1, block_patches=block_patches, haven_neighbourhood_m=3.0, seed_separation_m=4.0,
                    operational_hours=OPERATIONAL_HOURS)               # the production step: 120 minutes, 4 moves
    settings.update(overrides)
    return WindowParameters(**settings)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("stage3_store_scene")
    dem_path, horizons_path = bs.write_inputs(directory)
    times = ls.times(FIRST, LAST, step_hours=2.0)
    stamps = times.values
    products = ls.generate_safe_haven_products(
        dem_path, horizons_path, directory / "stage1", times=times, backend="cpu",
        sun_vectors_m=scene.sun_vectors(stamps), earth_vectors_m=scene.earth_vectors(stamps))
    dem, _ = _load_dem(dem_path)
    store_directory = directory / "store"
    store_directory.mkdir()
    stage0.generate_light_store(
        store_directory, dem=dem, horizon_store=HorizonTileStore(horizons_path),
        settings=Stage0Settings(store=LightStoreSettings(steps_per_chunk=48, codec="zlib-1"), start_utc=STORE_START,
                                stop_utc=STORE_STOP, lighting="cpu", workers=2),
        sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors)
    return SimpleNamespace(
        directory=directory, dem=dem, horizons=HorizonTileStore(horizons_path), stamps=stamps,
        stage1=Stage1Rasters(products.hibernation, products.joint_total),
        light_store=LightStoreDirectory(store_directory, dem), store_directory=store_directory)


def kwargs(prepared, name, *, light_store, block_patches=1, **options):
    common = dict(dem=prepared.dem, stage1=prepared.stage1, times=prepared.stamps, earth_vectors_at=scene.earth_vectors,
                  output_path=prepared.directory / f"{name}.csv", parameters=parameters(block_patches), overwrite=True)
    if light_store:
        common["light_store"] = prepared.light_store
    else:
        common.update(horizon_store=prepared.horizons, sun_vectors_at=scene.sun_vectors)
    common.update(options)
    return common


@pytest.fixture(scope="module")
def reference(prepared):
    """The horizon route's table and report: the reference the store route must reproduce."""
    report = blocks.run_pass_two(**kwargs(prepared, "horizons", light_store=False))
    return prepared.directory / "horizons.csv", report


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- the table --------------------------------------------------------------------------------


def test_the_reference_scene_has_havens_and_floods_that_reach_across_cells(reference) -> None:
    path, report = reference
    rows = read_rows(path)
    assert report.rows == len(rows) > 5 and report.flood_step_minutes == 120
    assert all(int(r["lit_reachable_pixels_before"]) > 1 and int(r["lit_reachable_pixels_after"]) > 1 for r in rows)
    assert {r["month"] for r in rows} >= {"2027-01", "2027-02"}


def test_the_store_route_writes_the_horizon_routes_table(prepared, reference) -> None:
    path, report = reference
    stored_report = blocks.run_pass_two(**kwargs(prepared, "store", light_store=True))
    assert (prepared.directory / "store.csv").read_bytes() == path.read_bytes()
    assert (stored_report.rows, stored_report.plan, stored_report.flood) == (report.rows, report.plan, report.flood)
    assert stored_report.lighting == "light store" and stored_report.horizon_loads == 0
    assert stored_report.flood.seeds_boundary_mismatch == 0


def test_stage_three_reproduces_the_hibernations_of_a_stage_one_made_from_the_same_store(prepared, reference) -> None:
    """The whole chain on the store: stage 1 reads it, stage 3 recomputes each seed's hibernation from it,
    and every one agrees (no `invalid_stage1_mismatch`)."""
    times = ls.times(FIRST, LAST, step_hours=2.0)
    products = ls.generate_safe_haven_products(
        prepared.directory / "dem.tif", None, prepared.directory / "stage1_from_store", times=times,
        light_store_path=prepared.store_directory, backend="cpu", earth_vectors_m=scene.earth_vectors(times.values))
    options = kwargs(prepared, "chain", light_store=True)
    options["stage1"] = Stage1Rasters(products.hibernation, products.joint_total)
    report = blocks.run_pass_two(**options)
    assert report.plan.invalid_stage1_mismatch == 0 and report.plan.timelines_evaluated > 0
    assert (prepared.directory / "chain.csv").read_bytes() == reference[0].read_bytes()


@pytest.mark.parametrize("block_patches", [2, 3])
def test_the_table_is_the_same_for_any_block_size(prepared, reference, block_patches) -> None:
    blocks.run_pass_two(**kwargs(prepared, f"store_b{block_patches}", light_store=True, block_patches=block_patches))
    assert (prepared.directory / f"store_b{block_patches}.csv").read_bytes() == reference[0].read_bytes()


@pytest.mark.parametrize("label, settings", [
    ("minimal", dict(plan_workers=1, light_workers=1, numba_threads_per_light_worker=1, flood_workers=1)),
    ("fast", FAST),
    ("one_cell_at_a_time", dict(FAST, cells_in_flight=1, store_budget_bytes=1)),
])
def test_the_pipeline_reads_units_from_the_store_and_writes_the_same_table(prepared, reference, label, settings) -> None:
    report = blocks.run_pass_two(**kwargs(prepared, f"pipe_{label}", light_store=True, pipeline=PipelineSettings(**settings)))
    assert (prepared.directory / f"pipe_{label}.csv").read_bytes() == reference[0].read_bytes()
    assert report.lighting == "light store" and report.horizon_loads == 0
    assert report.flood.lit_units_computed + report.flood.lit_units_reused == (
        reference[1].flood.lit_units_computed + reference[1].flood.lit_units_reused)
    assert multiprocessing.active_children() == []


def test_the_pipeline_never_probes_cuda_for_a_store(prepared, reference, monkeypatch) -> None:
    from lunarscout._numba_horizon import safe_haven_stage3 as stage3

    monkeypatch.setattr(stage3, "select_lighting", lambda *a, **k: pytest.fail("a store needs no lighting"))
    blocks.run_pass_two(**kwargs(prepared, "pipe_nocuda", light_store=True,
                                 pipeline=PipelineSettings(**FAST, lighting="cuda")))
    assert (prepared.directory / "pipe_nocuda.csv").read_bytes() == reference[0].read_bytes()


def test_worker_processes_read_the_store_too(prepared, reference) -> None:
    report = blocks.run_pass_two(**kwargs(prepared, "store_workers", light_store=True, workers=2))
    assert (prepared.directory / "store_workers.csv").read_bytes() == reference[0].read_bytes()
    assert report.horizon_loads == 0


def test_no_horizon_is_read_by_the_store_route(prepared, monkeypatch) -> None:
    import lunarscout._numba_horizon.safe_haven_blocks as run_module
    import lunarscout._numba_horizon.safe_haven_stage3 as pipeline_module
    import lunarscout._numba_horizon.safe_haven_windows as windows_module

    def forbidden(*args, **kwargs):
        pytest.fail("a horizon was read")

    for module in (run_module, pipeline_module, windows_module):
        monkeypatch.setattr(module, "read_patch_horizons", forbidden)
    monkeypatch.setattr(HorizonTileStore, "read", forbidden, raising=False)
    blocks.run_pass_two(**kwargs(prepared, "nohorizon", light_store=True))
    blocks.run_pass_two(**kwargs(prepared, "nohorizon_pipe", light_store=True, pipeline=PipelineSettings(**FAST)))


# --- the checkpoint and the refusals ----------------------------------------------------------


def test_a_checkpoint_made_from_horizons_is_not_resumed_from_a_store(prepared) -> None:
    horizons = kwargs(prepared, "switch", light_store=False)
    assert blocks.run_pass_two(stop_after_blocks=1, **horizons) is None
    with pytest.raises(ValueError, match="different configuration"):
        blocks.run_pass_two(**kwargs(prepared, "switch", light_store=True))


@pytest.mark.parametrize("override, code", [
    (dict(parameters=parameters(flood_step_minutes=30, moves_per_step=1)), "safe_haven_store_step_mismatch"),
    (dict(sunlight_fraction_threshold=0.3), "safe_haven_store_thresholds"),
    (dict(earth_elevation_threshold_deg=3.0), "safe_haven_store_thresholds"),
    (dict(parameters=parameters(operating_sunlight_fraction=0.9)), "safe_haven_store_thresholds"),
])
def test_a_store_that_cannot_answer_the_run_is_refused(prepared, override, code) -> None:
    with pytest.raises(InputError) as caught:
        blocks.run_pass_two(**kwargs(prepared, "refused", light_store=True, **override))
    assert caught.value.code == code
    assert not (prepared.directory / "refused.csv").exists()


def test_exactly_one_lighting_source_is_required(prepared) -> None:
    both = kwargs(prepared, "both", light_store=True, horizon_store=prepared.horizons, sun_vectors_at=scene.sun_vectors)
    neither = kwargs(prepared, "neither", light_store=False)
    del neither["horizon_store"], neither["sun_vectors_at"]
    for call in (both, neither):
        with pytest.raises(ValueError, match="exactly one"):
            blocks.run_pass_two(**call)


def test_a_store_made_for_another_dem_is_refused(prepared, tmp_path) -> None:
    other = _load_dem(prepared.directory / "dem.tif")[0]
    other.elevation_m[0, 0] += 1.0
    with pytest.raises(Exception) as caught:
        LightStoreDirectory(prepared.store_directory, other)
    assert getattr(caught.value, "code", "") == "light_store_manifest_mismatch"


def test_stage_one_samples_outside_the_store_are_refused(prepared, tmp_path) -> None:
    short = ls.times("2026-11-01T00:00:00Z", "2026-11-30T22:00:00Z", step_hours=2.0)
    with pytest.raises(InputError) as caught:
        from_store.check_run(prepared.light_store, flood_step_minutes=120, times=short.values, stage1_step_hours=2.0,
                             sunlight_threshold=0.2, operating_sunlight_fraction=1.0, earth_threshold_deg=2.0)
    assert caught.value.code == "safe_haven_store_time_range"


# --- the signals of one pixel -----------------------------------------------------------------


def test_store_signals_carry_what_pixel_signals_say_across_the_thresholds(prepared) -> None:
    store_signals = from_store.StoreSignals(prepared.light_store, prepared.dem)
    horizon_signals = PixelSignals(prepared.dem, prepared.horizons)
    step = 120 * 60_000_000
    first = int(np.datetime64("2027-01-10T00:00:00", "us").astype(np.int64)) // step
    steps = np.arange(first, first + 700)
    times_us = steps * step
    stamps = times_us.astype("datetime64[us]")
    sun, earth = scene.sun_vectors(stamps), scene.earth_vectors(stamps)
    rng = np.random.default_rng(3)
    pixels = [(62, 62), (124, 22), (203, 203), (283, 133), (5, 5)] + [
        (int(r), int(c)) for r, c in zip(rng.integers(0, 300, 6), rng.integers(0, 300, 6))]
    for row, column in pixels:
        want_sun = horizon_signals.series(row, column, sun, margin=False)
        want_earth = horizon_signals.series(row, column, earth, margin=True)
        got_sun = store_signals.series(row, column, None, margin=False, times_us=times_us)
        got_earth = store_signals.series(row, column, None, margin=True, times_us=times_us)
        np.testing.assert_array_equal(got_sun < 0.2, want_sun < 0.2)
        np.testing.assert_array_equal(got_sun >= 1.0 - 1e-6, want_sun >= 1.0 - 1e-6)
        np.testing.assert_array_equal(got_earth < 2.0, want_earth < 2.0)
        # the flat-horizon series is stage 1's: the one computed at the patch's centre pixel
        tile_y, tile_x, height, width = patch_geometry(prepared.dem, row // PATCH_SIZE, column // PATCH_SIZE)
        np.testing.assert_array_equal(
            store_signals.series(row, column, earth, margin=True, flat_elevation=True, times_us=times_us),
            flat_earth_elevation(prepared.dem, earth, height, width, tile_y, tile_x, LightmapCpuSession()))
    assert any((store_signals.series(r, c, None, margin=False, times_us=times_us) >= 1.0).any() for r, c in pixels)
    assert any((store_signals.series(r, c, None, margin=False, times_us=times_us) < 0.2).any() for r, c in pixels)
    store_signals.close()


def test_store_signals_edges(prepared, tmp_path) -> None:
    import shutil

    copy = tmp_path / "store"
    shutil.copytree(prepared.store_directory, copy)
    stage0.patch_file(copy, 1, 1).unlink()
    signals = from_store.StoreSignals(LightStoreDirectory(copy, prepared.dem), prepared.dem)
    step = 120 * 60_000_000
    first = int(np.datetime64(STORE_START, "us").astype(np.int64)) // step
    ok = signals.series(10, 10, None, margin=False, times_us=np.array([first + 3, first + 4]) * step)
    assert ok.shape == (2,) and ok.dtype == np.float32
    assert signals.series(150, 150, None, margin=False, times_us=np.array([first]) * step) is None       # no file
    assert signals.series(-1, 5, None, margin=False, times_us=np.array([first]) * step) is None          # outside the DEM
    assert signals.series(300, 5, None, margin=True, times_us=np.array([first]) * step) is None
    before = signals.series(10, 10, None, margin=False, times_us=np.array([first - 5, first - 2]) * step)
    assert not before.any()                                                                                # outside the range: dark
    margin = signals.series(10, 10, None, margin=True, times_us=np.array([first - 5, first - 2]) * step)
    assert (margin == -from_store.CLEAR_MARGIN_DEG).all()
    straddle = signals.series(10, 10, None, margin=True, times_us=np.array([first - 2, first + 1]) * step)
    assert straddle[0] == -from_store.CLEAR_MARGIN_DEG and straddle.shape == (2,)
    with pytest.raises(InputError) as caught:
        signals.series(10, 10, None, margin=False, times_us=np.array([first * step + 1]))
    assert caught.value.code == "safe_haven_store_time_misaligned"
    with pytest.raises(ValueError):
        signals.series(10, 10, None, margin=False)
    with pytest.raises(ValueError):
        signals.series(10, 10, None, margin=False, flat_elevation=True, times_us=np.array([first]) * step)
    signals.close()


# --- the lit units ----------------------------------------------------------------------------


def test_units_from_the_store_equal_the_lit_units_of_the_horizons(prepared) -> None:
    grid = FloodGrid(120)
    shape = (prepared.dem.height, prepared.dem.width)
    patches = [(0, 0), (1, 0), (1, 1), (2, 1)]
    first_chunk = int(np.datetime64("2027-01-05", "us").astype(np.int64)) // (grid.step_us * grid.steps_per_chunk)
    chunks = [first_chunk, first_chunk + 1, first_chunk + 20]
    want, got = LitStore(grid, shape), LitStore(grid, shape)
    fill_lit_store(want, dem=prepared.dem, horizon_store=prepared.horizons, patches=patches, chunks=chunks,
                   sun_vectors_at=scene.sun_vectors)
    counts = from_store.fill_lit_store_from_light_store(got, directory=prepared.light_store, patches=patches, chunks=chunks)
    assert (counts.computed, counts.reused) == (12, 0) and len(got) == len(want) == 12
    for patch in patches:
        for chunk in chunks:
            assert np.array_equal(got._chunks[(*patch, chunk)], want._chunks[(*patch, chunk)]), (patch, chunk)
    assert any(chunk.any() for chunk in got._chunks.values()) and any(not chunk.all() for chunk in got._chunks.values())
    again = from_store.fill_lit_store_from_light_store(got, directory=prepared.light_store, patches=patches, chunks=chunks)
    assert (again.computed, again.reused) == (0, 12)
    assert from_store.fill_lit_store_from_light_store(got, directory=prepared.light_store, patches=[(5, 5)], chunks=chunks).computed == 0


def test_units_outside_the_stored_range_are_dark_and_a_partial_one_is_clipped(prepared, tmp_path) -> None:
    opened = prepared.light_store.open_patch(0, 0)
    spc = 12
    first_chunk = opened.first_step // spc
    stop_chunk = -(-opened.stop_step // spc)
    with opened:
        assert not from_store.unit_bits(opened, first_chunk - 3, spc).any()
        assert not from_store.unit_bits(opened, stop_chunk + 2, spc).any()
        inside = from_store.unit_bits(opened, first_chunk + 40, spc)
        assert inside.shape == (spc, PATCH_SIZE, 2) and inside.dtype == np.uint64 and inside.any()
        # a chunk that starts before the stored range holds the stored steps and zeros before them
        shifted = opened.first_step % spc
        if shifted:
            edge = from_store.unit_bits(opened, first_chunk, spc)
            assert not edge[: spc - shifted].any()


def test_a_missing_patch_file_is_an_all_dark_unit(prepared, tmp_path) -> None:
    import shutil

    copy = tmp_path / "store"
    shutil.copytree(prepared.store_directory, copy)
    stage0.patch_file(copy, 2, 2).unlink()
    grid = FloodGrid(120)
    store = LitStore(grid, (prepared.dem.height, prepared.dem.width))
    counts = from_store.fill_lit_store_from_light_store(
        store, directory=LightStoreDirectory(copy, prepared.dem), patches=[(2, 2)], chunks=[14000, 14001])
    assert counts.patches_without_horizons == 1 and counts.computed == 2
    assert all(not unit.any() for unit in store._chunks.values())


# --- the scripts ------------------------------------------------------------------------------
