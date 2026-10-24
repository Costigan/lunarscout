"""The pass-2 block schedule and CPU driver (spec section 8) on a multi-patch scene."""

from __future__ import annotations

import csv
import json
import multiprocessing
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import blocks_scene as bs
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_blocks as blocks
from lunarscout._numba_horizon import safe_haven_flood as oracle
from lunarscout._numba_horizon.file_format import PATCH_SIZE, HorizonTileStore
from lunarscout._numba_horizon.safe_haven_windows import FloodGrid, Stage1Rasters, WindowParameters
from lunarscout.products import _load_dem

OPERATIONAL_HOURS = 24.0


def parameters(block_patches, **overrides):
    settings = dict(
        cell_patches=1, block_patches=block_patches, rover_speed_m_s=1.0 / 1800.0,
        haven_neighbourhood_m=3.0, seed_separation_m=4.0, operational_hours=OPERATIONAL_HOURS,
    )
    settings.update(overrides)
    return WindowParameters(**settings)


# --- geometry, order and the load plan --------------------------------------------------------


def test_blocks_are_scanned_in_a_snake() -> None:
    assert blocks.block_order(3, 5, 2) == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    assert blocks.block_order(1, 1, 4) == [(0, 0)]
    assert blocks.block_of_cell((5, 7), 4) == (1, 1)


def test_a_cell_window_is_the_cell_plus_a_halo_clipped_to_the_raster() -> None:
    kwargs = dict(cell_patches=2, halo_patches=1, patch_rows=237, patch_columns=237)
    interior = blocks.cell_patches_with_halo((10, 10), **kwargs)
    assert len(interior) == 4 * 4 and (19, 19) in interior and (20, 20) in interior and (18, 18) not in interior
    corner = blocks.cell_patches_with_halo((0, 0), **kwargs)
    assert corner == {(r, c) for r in range(3) for c in range(3)}
    last = blocks.cell_patches_with_halo((118, 118), **kwargs)
    assert last == {(r, c) for r in range(235, 237) for c in range(235, 237)}


def block_patch_sets(patches, block_patches, cell_patches=2, halo=1):
    cells = -(-patches // cell_patches)
    block_cells = block_patches // cell_patches
    sets = []
    for block in blocks.block_order(cells, cells, block_cells):
        members = [(r, c) for r in range(block[0] * block_cells, min(cells, (block[0] + 1) * block_cells))
                   for c in range(block[1] * block_cells, min(cells, (block[1] + 1) * block_cells))]
        union = set()
        for cell in members:
            union |= blocks.cell_patches_with_halo(
                cell, cell_patches=cell_patches, halo_patches=halo, patch_rows=patches, patch_columns=patches)
        sets.append(union)
    return sets


def test_the_load_plan_on_the_pole_matches_the_specification_budget() -> None:
    """Spec section 8: blocks of 8 load each patch about 1.55 times, 1.24 with the snake
    carry-over; a resident block of 100 patches of float32 horizons is 9.4 GB."""
    plan = blocks.plan_horizon_loads(block_patch_sets(237, 8), carry_over=False)
    carried = blocks.plan_horizon_loads(block_patch_sets(237, 8))
    assert plan.distinct_patches == carried.distinct_patches == 237 * 237
    assert plan.loads_per_patch == pytest.approx(1.549, abs=0.005)
    assert carried.loads_per_patch == pytest.approx(1.237, abs=0.005)
    assert carried.peak_resident == 100
    assert carried.peak_resident * PATCH_SIZE * PATCH_SIZE * 1440 * 4 / 1e9 == pytest.approx(9.44, abs=0.01)
    big = blocks.plan_horizon_loads(block_patch_sets(237, 10))
    assert big.peak_resident == 144
    assert big.peak_resident * PATCH_SIZE * PATCH_SIZE * 1440 * 4 / 1e9 == pytest.approx(13.59, abs=0.01)


def test_the_snake_loads_fewer_patches_than_row_by_row() -> None:
    snake = block_patch_sets(40, 8)
    row_by_row = sorted(snake, key=lambda s: (min(r for r, _ in s), min(c for _, c in s)))
    assert blocks.plan_horizon_loads(snake).loads < blocks.plan_horizon_loads(row_by_row).loads


def test_residency_drops_before_it_loads_and_counts_reads() -> None:
    reads = []
    residency = blocks.HorizonResidency(lambda patch: reads.append(patch) or np.zeros(1, np.float32))
    residency.set_resident({(0, 0), (0, 1), (1, 0)})
    residency.set_resident({(0, 1), (1, 0), (1, 1)})
    assert residency.loads == 4 and sorted(reads) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert residency.peak_resident == 3 and len(residency) == 3
    with pytest.raises(KeyError):
        residency.get((0, 0))
    residency.set_resident({(0, 1), (1, 0), (1, 1)})
    assert residency.loads == 4                       # nothing new to read


def test_a_block_must_be_a_whole_number_of_cells() -> None:
    with pytest.raises(ValueError, match="block_patches"):
        WindowParameters(cell_patches=2, block_patches=3)


# --- the driver on the scene -----------------------------------------------------------------


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("blocks_scene")
    dem_path, horizons_path, products, stamps = bs.stage_one(directory)
    dem, _georef = _load_dem(dem_path)
    return SimpleNamespace(
        directory=directory, dem=dem, store=HorizonTileStore(horizons_path), stamps=stamps,
        stage1=Stage1Rasters(products.hibernation, products.joint_total),
    )


def run(prepared, name, block_patches, **options):
    path = prepared.directory / f"{name}.csv"
    report = blocks.run_pass_two(
        dem=prepared.dem, horizon_store=prepared.store, stage1=prepared.stage1, times=prepared.stamps,
        sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors, output_path=path,
        parameters=parameters(block_patches), overwrite=True, **options,
    )
    return path, report


@pytest.fixture(scope="module")
def tables(prepared):
    return {size: run(prepared, f"block{size}", size) for size in (1, 2, 3)}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_table_is_identical_for_any_block_size(tables) -> None:
    texts = {size: path.read_bytes() for size, (path, _report) in tables.items()}
    assert texts[1] == texts[2] == texts[3]
    rows = read_rows(tables[1][0])
    assert tuple(rows[0].keys()) == blocks.TABLE_COLUMNS
    # the clusters lie in cells (0, 0), (1, 1) and (2, 1), and one straddles cells (0, 0) and (1, 0)
    assert {(r["cell_row"], r["cell_col"]) for r in rows} == {("0", "0"), ("1", "0"), ("1", "1"), ("2", "1")}
    reports = {size: report for size, (_path, report) in tables.items()}
    assert [reports[s].blocks_total for s in (1, 2, 3)] == [4, 2, 1]
    assert reports[3].blocks_total == 1 and reports[3].horizon_loads == 9 and reports[3].peak_resident_patches == 9
    for report in reports.values():
        assert report.rows == len(rows) > 0
        assert report.flood.rows == report.rows
        assert report.flood.seeds_boundary_mismatch == 0


def test_a_row_belongs_to_the_month_its_outage_starts_in(tables) -> None:
    rows = read_rows(tables[1][0])
    assert {r["month"] for r in rows} == {"2027-01", "2027-02", "2027-03"}
    for row in rows:
        assert row["outage_start_utc"][:7] == row["month"]


def test_rows_are_ranked_within_their_cell_month(tables) -> None:
    rows = read_rows(tables[1][0])
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["cell_row"], row["cell_col"], row["month"]), []).append(row)
    for group in groups.values():
        assert [int(r["rank"]) for r in group] == list(range(1, len(group) + 1))
        keys = [(-int(r["score"]), int(r["seed_row"]), int(r["seed_col"])) for r in group]
        assert keys == sorted(keys)
        assert all(int(r["valid_seeds"]) == len(group) <= int(r["candidate_seeds"]) <= 5 for r in group)
        assert all(int(r["score"]) == int(r["count_before"]) + int(r["count_after"]) for r in group)
        assert all(int(r["count_before"]) > 0 and int(r["count_after"]) > 0 for r in group)


def test_cell_bounds_follow_the_geotransform_including_the_partial_last_cell(tables) -> None:
    rows = read_rows(tables[1][0])
    last = next(r for r in rows if (r["cell_row"], r["cell_col"]) == ("2", "1"))
    assert (float(last["x_min_m"]), float(last["x_max_m"])) == (1128.0, 1256.0)
    assert (float(last["y_max_m"]), float(last["y_min_m"])) == (-1256.0, -1300.0)
    seed_row, seed_col = int(last["seed_row"]), int(last["seed_col"])
    assert float(last["seed_x_m"]) == 1000.0 + seed_col + 0.5
    assert float(last["seed_y_m"]) == -1000.0 - seed_row - 0.5


def test_each_flood_count_equals_the_oracle_on_the_real_lit_state(prepared, tables) -> None:
    rows = read_rows(tables[2][0])
    straddling = [r for r in rows if r["cell_row"] == "1" and r["cell_col"] == "0" and r["month"] == "2027-01"]
    assert straddling
    grid = FloodGrid(30)
    steps = grid.step_count(OPERATIONAL_HOURS)
    signals = __import__("lunarscout._numba_horizon.safe_haven_windows", fromlist=["x"]).PixelSignals(
        prepared.dem, prepared.store)

    def step_of(text):
        return int(np.datetime64(text.rstrip("Z"), "us").astype(np.int64)) // grid.step_us

    first_row = straddling[0]
    before, after = step_of(first_row["boundary_before_utc"]), step_of(first_row["boundary_after_utc"])
    cube = oracle.lit_reference_cube(
        signals, scene.sun_vectors, grid, (0, 0), (bs.SIZE, bs.SIZE), before - steps, after + steps)
    hibernation = prepared.stage1.read_hibernation_window(0, bs.SIZE, 0, bs.SIZE)[0]
    passable = np.isfinite(hibernation)
    for row in straddling[:2]:
        assert step_of(row["boundary_before_utc"]) == before and step_of(row["boundary_after_utc"]) == after
        expected = oracle.flood_seed_reference(
            lambda step: cube[step - (before - steps)], passable, (int(row["seed_row"]), int(row["seed_col"])),
            before, after, steps=steps)
        assert (int(row["count_before"]), int(row["count_after"])) == (expected.count_before, expected.count_after)


def test_the_driver_reads_horizons_as_the_load_plan_says(prepared, tables) -> None:
    for size, (_path, report) in tables.items():
        cells = sorted({(int(r["cell_row"]), int(r["cell_col"])) for r in read_rows(tables[size][0])})
        order = [b for b in blocks.block_order(3, 3, size) if any(blocks.block_of_cell(c, size) == b for c in cells)]
        sets = []
        for block in order:
            members = [c for c in cells if blocks.block_of_cell(c, size) == block]
            union = set()
            for cell in members:
                union |= blocks.cell_patches_with_halo(cell, cell_patches=1, halo_patches=1, patch_rows=3, patch_columns=3)
            sets.append(union)
        assert report.horizon_loads == blocks.plan_horizon_loads(sets).loads


def test_a_run_limited_to_some_cells_gives_those_cells_rows(prepared, tables) -> None:
    path, report = run(prepared, "one_cell", 2, cells=[(1, 0)])
    rows = read_rows(path)
    full = [r for r in read_rows(tables[2][0]) if (r["cell_row"], r["cell_col"]) == ("1", "0")]
    assert rows == full and rows


# --- checkpoint and resume ----------------------------------------------------------------------


def test_a_run_resumes_from_its_checkpoint_to_the_same_table(prepared, tables) -> None:
    path = prepared.directory / "resumed.csv"
    checkpoint = path.with_name(path.name + ".checkpoint.jsonl")
    kwargs = dict(dem=prepared.dem, horizon_store=prepared.store, stage1=prepared.stage1, times=prepared.stamps,
                  sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors,
                  output_path=path, parameters=parameters(1), overwrite=True)
    assert blocks.run_pass_two(stop_after_blocks=2, **kwargs) is None
    lines = checkpoint.read_text().splitlines()
    assert json.loads(lines[0])["type"] == "header" and len(lines) == 3 and not path.exists()
    with checkpoint.open("a") as handle:
        handle.write('{"type": "block", "block": [1')       # a torn final line is ignored
    report = blocks.run_pass_two(**kwargs)
    assert report.blocks_resumed == 2 and report.blocks_total == tables[1][1].blocks_total
    assert path.read_bytes() == tables[1][0].read_bytes()
    assert not checkpoint.exists()
    assert report.rows == tables[1][1].rows and report.flood.rows == report.rows


def test_a_checkpoint_from_another_configuration_is_refused(prepared) -> None:
    path = prepared.directory / "refused.csv"
    kwargs = dict(dem=prepared.dem, horizon_store=prepared.store, stage1=prepared.stage1, times=prepared.stamps,
                  sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors, output_path=path,
                  overwrite=True)
    assert blocks.run_pass_two(stop_after_blocks=1, parameters=parameters(1), **kwargs) is None
    with pytest.raises(ValueError, match="different configuration"):
        blocks.run_pass_two(parameters=parameters(2), **kwargs)
    with pytest.raises(ValueError, match="different configuration"):
        blocks.run_pass_two(parameters=parameters(1, max_seeds_per_cell_month=3), **kwargs)


def test_existing_output_is_not_replaced_without_overwrite_and_a_missing_directory_is_refused(prepared) -> None:
    kwargs = dict(dem=prepared.dem, horizon_store=prepared.store, stage1=prepared.stage1, times=prepared.stamps,
                  sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors,
                  parameters=parameters(3))
    existing = prepared.directory / "existing.csv"
    existing.write_text("keep me")
    with pytest.raises(ValueError, match="exists"):
        blocks.run_pass_two(output_path=existing, **kwargs)
    assert existing.read_text() == "keep me"
    with pytest.raises(ValueError, match="directory"):
        blocks.run_pass_two(output_path=prepared.directory / "nowhere" / "out.csv", **kwargs)


# --- NODATA walls and rejected seeds ----------------------------------------------------------------


def test_nodata_pixels_block_the_flood(prepared, tables, tmp_path) -> None:
    """Columns 70-71 (rows 0-129) are NODATA in the hibernation product: a wall just east of
    the cluster at rows and columns 60..67.  The counts must equal the oracle's with the wall."""
    import shutil

    import rasterio

    wall = (slice(0, 130), slice(70, 72))
    hibernation = tmp_path / "hibernation_wall.tif"
    shutil.copy(prepared.stage1.hibernation_path, hibernation)
    with rasterio.open(hibernation, "r+") as dataset:
        data = dataset.read()
        data[:, wall[0], wall[1]] = np.nan
        dataset.write(data)
    walled = SimpleNamespace(**{**vars(prepared), "stage1": Stage1Rasters(hibernation, prepared.stage1.joint_total_path)})
    path, _report = run(walled, "walled", 3, cells=[(0, 0)])
    plain = [r for r in read_rows(tables[3][0]) if (r["cell_row"], r["cell_col"]) == ("0", "0")]
    rows = [r for r in read_rows(path) if r["month"] == "2027-01"]
    assert rows and plain
    grid = FloodGrid(30)
    steps = grid.step_count(OPERATIONAL_HOURS)
    signals = __import__("lunarscout._numba_horizon.safe_haven_windows", fromlist=["x"]).PixelSignals(
        prepared.dem, prepared.store)

    def step_of(text):
        return int(np.datetime64(text.rstrip("Z"), "us").astype(np.int64)) // grid.step_us

    first = rows[0]
    before, after = step_of(first["boundary_before_utc"]), step_of(first["boundary_after_utc"])
    cube = oracle.lit_reference_cube(
        signals, scene.sun_vectors, grid, (0, 0), (bs.SIZE, bs.SIZE), before - steps, after + steps)
    passable = np.isfinite(walled.stage1.read_hibernation_window(0, bs.SIZE, 0, bs.SIZE)[0])
    assert not passable[wall].any()
    seen_difference = False
    for row in rows[:2]:
        expected = oracle.flood_seed_reference(
            lambda step: cube[step - (before - steps)], passable, (int(row["seed_row"]), int(row["seed_col"])),
            before, after, steps=steps)
        assert (int(row["count_before"]), int(row["count_after"])) == (expected.count_before, expected.count_after)
        match = next(r for r in plain if r["month"] == "2027-01" and r["seed_row"] == row["seed_row"]
                     and r["seed_col"] == row["seed_col"])
        seen_difference |= int(match["score"]) != int(row["score"])
    assert seen_difference                                    # the wall cut the flood


def test_seeds_that_fail_the_flood_are_counted_and_leave_no_row(prepared, monkeypatch) -> None:
    from lunarscout._numba_horizon.safe_haven_flood import FloodResult

    outcomes = iter([None, FloodResult(5, 0), FloodResult(0, 3)])
    seen = []

    def stub(*args, **kwargs):
        seen.append(args[3])
        return next(outcomes, FloodResult(4, 6))

    monkeypatch.setattr(blocks, "flood_seed_from_store", stub)
    path, report = run(prepared, "stubbed", 3, cells=[(0, 0)])
    counts = report.flood
    assert counts.seeds_boundary_mismatch == 1 and counts.seeds_without_both_sides == 2
    rows = read_rows(path)
    assert all(r["score"] == "10" and r["count_before"] == "4" for r in rows)
    first = next(r for r in rows if r["month"] == "2027-01")
    assert int(first["valid_seeds"]) == int(first["candidate_seeds"]) - 3
    monkeypatch.setattr(blocks, "flood_seed_from_store", lambda *a, **k: FloodResult(0, 0))
    path, report = run(prepared, "all_invalid", 3, cells=[(0, 0)])
    assert read_rows(path) == [] and report.flood.cell_months_without_valid_seed == report.flood.cell_months_flooded > 0


# --- progress and cached vectors ---------------------------------------------------------------------


def test_progress_starts_at_the_blocks_a_checkpoint_already_holds(prepared) -> None:
    path = prepared.directory / "progress.csv"
    kwargs = dict(dem=prepared.dem, horizon_store=prepared.store, stage1=prepared.stage1, times=prepared.stamps,
                  sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors,
                  output_path=path, parameters=parameters(1), overwrite=True)
    calls = []
    blocks.run_pass_two(stop_after_blocks=1, progress=lambda done, total: calls.append((done, total)), **kwargs)
    assert calls == [(0, 4), (1, 4)]
    calls.clear()
    blocks.run_pass_two(progress=lambda done, total: calls.append((done, total)), **kwargs)
    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_each_set_of_times_is_sent_to_the_vector_source_once(prepared, tables) -> None:
    asked = []

    def sun(stamps):
        asked.append(np.asarray(stamps).astype("datetime64[us]").tobytes())
        return scene.sun_vectors(stamps)

    path = prepared.directory / "cached.csv"
    blocks.run_pass_two(
        dem=prepared.dem, horizon_store=prepared.store, stage1=prepared.stage1, times=prepared.stamps,
        sun_vectors_at=sun, earth_vectors_at=scene.earth_vectors, output_path=path,
        parameters=parameters(3), overwrite=True, cells=[(0, 0)],
    )
    assert len(asked) == len(set(asked)) > 5
    expected = [r for r in read_rows(tables[3][0]) if (r['cell_row'], r['cell_col']) == ('0', '0')]
    assert read_rows(path) == expected and expected


def test_the_vector_cache_is_bounded_and_returns_equal_answers() -> None:
    cache = blocks.CachedVectors(lambda stamps: np.arange(len(stamps) * 3, dtype=float).reshape(-1, 3), limit=2)
    first = np.array(["2027-01-01T00:00", "2027-01-01T06:00"], dtype="datetime64[us]")
    assert np.array_equal(cache(first), cache(first)) and (cache.calls, cache.hits) == (1, 1)
    for hour in (1, 2, 3):
        cache(first + np.timedelta64(hour, "h"))
    assert len(cache._answers) == 2


# --- a resume does not rescan the stage-1 raster -------------------------------------------------------


def test_a_resume_takes_its_cells_from_the_checkpoint_instead_of_scanning(prepared, tables, monkeypatch) -> None:
    path = prepared.directory / "noscan.csv"
    kwargs = dict(dem=prepared.dem, horizon_store=prepared.store, stage1=prepared.stage1, times=prepared.stamps,
                  sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors,
                  output_path=path, parameters=parameters(1), overwrite=True)
    scans = []
    real = blocks.scan_cells
    monkeypatch.setattr(blocks, "scan_cells", lambda *a, **k: scans.append(1) or real(*a, **k))
    assert blocks.run_pass_two(stop_after_blocks=1, **kwargs) is None
    assert len(scans) == 1                                    # the first launch has to find the cells
    report = blocks.run_pass_two(**kwargs)
    assert len(scans) == 1 and report.blocks_resumed == 1     # the resume did not
    assert path.read_bytes() == tables[1][0].read_bytes()


def test_a_checkpoint_is_refused_when_a_stage_1_file_has_changed(prepared, tmp_path) -> None:
    import os
    import shutil

    hibernation, joint = tmp_path / "hibernation.tif", tmp_path / "joint.tif"
    shutil.copy(prepared.stage1.hibernation_path, hibernation)
    shutil.copy(prepared.stage1.joint_total_path, joint)
    stage1 = Stage1Rasters(hibernation, joint)
    kwargs = dict(dem=prepared.dem, horizon_store=prepared.store, stage1=stage1, times=prepared.stamps,
                  sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors,
                  output_path=tmp_path / "out.csv", parameters=parameters(1), overwrite=True)
    assert blocks.run_pass_two(stop_after_blocks=1, **kwargs) is None
    stat = hibernation.stat()
    os.utime(hibernation, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))      # rewritten, same size
    with pytest.raises(ValueError, match="different configuration"):
        blocks.run_pass_two(**kwargs)
    os.utime(hibernation, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert blocks.run_pass_two(**kwargs).blocks_resumed == 1


def test_a_torn_or_missing_checkpoint_header_falls_back_to_a_scan(tmp_path) -> None:
    missing = tmp_path / "none.jsonl"
    assert blocks._checkpoint_cells(missing) is None
    torn = tmp_path / "torn.jsonl"
    torn.write_text('{"type": "header", "config": {"cel')
    assert blocks._checkpoint_cells(torn) is None
    torn.write_text(json.dumps({"type": "header", "config": {"cells": [[1, 2], [3, 4]]}}) + "\n")
    assert blocks._checkpoint_cells(torn) == [(1, 2), (3, 4)]
    torn.write_text(json.dumps({"type": "block", "config": {"cells": [[1, 2]]}}) + "\n")
    assert blocks._checkpoint_cells(torn) is None              # not a header


# --- blocks in worker processes ------------------------------------------------------------------------


def parallel_kwargs(prepared, name, block_patches=1, **options):
    return dict(
        dem=prepared.dem, horizon_store=prepared.store, stage1=prepared.stage1, times=prepared.stamps,
        sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors,
        output_path=prepared.directory / f"{name}.csv", parameters=parameters(block_patches),
        overwrite=True, **options,
    )


@pytest.mark.parametrize("workers, block_patches", [(2, 1), (3, 1), (2, 2)])
def test_the_table_is_identical_for_any_number_of_workers(prepared, tables, workers, block_patches) -> None:
    report = blocks.run_pass_two(**parallel_kwargs(prepared, f"w{workers}b{block_patches}", block_patches,
                                                   workers=workers))
    path = prepared.directory / f"w{workers}b{block_patches}.csv"
    assert path.read_bytes() == tables[block_patches if block_patches in tables else 1][0].read_bytes()
    sequential = tables[block_patches][1]
    assert report.workers == workers and report.rows == sequential.rows
    assert report.plan == sequential.plan and report.flood == sequential.flood
    assert report.blocks_total == sequential.blocks_total and report.blocks_resumed == 0
    assert report.peak_resident_patches == sequential.peak_resident_patches
    # every worker loads its own horizons, so the total lies between the 9 distinct patches and the
    # 4 blocks' windows read without any sharing (at most 9 patches each)
    assert 9 <= report.horizon_loads <= 36
    assert multiprocessing.active_children() == []                  # no worker is left running


def test_a_checkpoint_made_with_some_workers_resumes_with_any_other(prepared, tables) -> None:
    path = prepared.directory / "mixed.csv"
    kwargs = parallel_kwargs(prepared, "mixed")
    assert blocks.run_pass_two(stop_after_blocks=2, workers=2, **kwargs) is None
    checkpoint = path.with_name(path.name + ".checkpoint.jsonl")
    lines = checkpoint.read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(line)["type"] in ("header", "block") for line in lines)
    report = blocks.run_pass_two(workers=1, **kwargs)                 # finish sequentially
    assert report.blocks_resumed == 2 and path.read_bytes() == tables[1][0].read_bytes()
    assert blocks.run_pass_two(stop_after_blocks=1, workers=1, **kwargs) is None
    report = blocks.run_pass_two(workers=3, **kwargs)                 # and in parallel the other way round
    assert report.blocks_resumed == 1 and path.read_bytes() == tables[1][0].read_bytes()


def test_workers_must_be_a_positive_integer(prepared) -> None:
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError, match="workers"):
            blocks.run_pass_two(**parallel_kwargs(prepared, "bad"), workers=bad)


def test_progress_counts_every_block_in_the_order_they_finish(prepared) -> None:
    calls = []
    blocks.run_pass_two(**parallel_kwargs(prepared, "progress_parallel"), workers=2,
                        progress=lambda done, total: calls.append((done, total)))
    assert calls == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]


def test_the_first_interrupt_lets_running_blocks_finish_and_the_run_resumes(prepared, tables) -> None:
    import os
    import signal

    path = prepared.directory / "interrupted.csv"
    kwargs = parallel_kwargs(prepared, "interrupted")
    sent = []

    def progress(done, total):
        if done == 1 and not sent:                     # the first block has finished: interrupt
            sent.append(1)
            os.kill(os.getpid(), signal.SIGINT)

    previous = signal.getsignal(signal.SIGINT)
    assert blocks.run_pass_two(workers=2, progress=progress, **kwargs) is None
    assert signal.getsignal(signal.SIGINT) == previous                # the handler is put back
    checkpoint = path.with_name(path.name + ".checkpoint.jsonl")
    finished = [json.loads(line) for line in checkpoint.read_text().splitlines()[1:]]
    assert 1 <= len(finished) < 4 and not path.exists()               # no more blocks were handed out
    assert multiprocessing.active_children() == []
    report = blocks.run_pass_two(workers=2, **kwargs)
    assert report.blocks_resumed == len(finished)
    assert path.read_bytes() == tables[1][0].read_bytes()


def test_a_sequential_run_also_stops_cleanly_on_an_interrupt(prepared, tables) -> None:
    import os
    import signal

    path = prepared.directory / "interrupted_sequential.csv"
    kwargs = parallel_kwargs(prepared, "interrupted_sequential")
    sent = []

    def progress(done, total):
        if done == 1 and not sent:
            sent.append(1)
            os.kill(os.getpid(), signal.SIGINT)

    previous = signal.getsignal(signal.SIGINT)
    assert blocks.run_pass_two(workers=1, progress=progress, **kwargs) is None
    assert signal.getsignal(signal.SIGINT) == previous
    checkpoint = path.with_name(path.name + ".checkpoint.jsonl")
    assert len(checkpoint.read_text().splitlines()) == 2 and not path.exists()      # header + the one finished block
    assert blocks.run_pass_two(workers=1, **kwargs).blocks_resumed == 1
    assert path.read_bytes() == tables[1][0].read_bytes()


@pytest.mark.parametrize("workers", [1, 2])
def test_an_interrupt_during_the_last_block_still_writes_the_table(prepared, tables, workers) -> None:
    import os
    import signal

    path = prepared.directory / f"last_block_{workers}.csv"
    sent = []

    def progress(done, total):
        if done == total and not sent:                    # the last block has just finished
            sent.append(1)
            os.kill(os.getpid(), signal.SIGINT)

    report = blocks.run_pass_two(workers=workers, progress=progress, **parallel_kwargs(prepared, f"last_block_{workers}"))
    assert report is not None and report.blocks_resumed == 0
    assert path.read_bytes() == tables[1][0].read_bytes()


def test_a_second_interrupt_stops_at_once(prepared) -> None:
    import os
    import signal

    kwargs = parallel_kwargs(prepared, "double_interrupt")
    sent = []

    def progress(done, total):
        if done == 1 and not sent:
            sent.extend([1, 2])
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)

    with pytest.raises(KeyboardInterrupt):
        blocks.run_pass_two(workers=1, progress=progress, **kwargs)
    assert multiprocessing.active_children() == []


def test_a_failing_block_is_raised_after_the_finished_blocks_are_recorded(prepared, monkeypatch) -> None:
    path = prepared.directory / "failing.csv"
    kwargs = parallel_kwargs(prepared, "failing")
    real = blocks._run_block

    def sometimes_fails(cells_in_block, **arguments):
        if (1, 1) in cells_in_block:
            raise RuntimeError("block (1, 1) is broken")
        return real(cells_in_block, **arguments)

    monkeypatch.setattr(blocks, "_run_block", sometimes_fails)        # inherited by the forked workers
    with pytest.raises(RuntimeError, match="block .1, 1. is broken"):
        blocks.run_pass_two(workers=2, **kwargs)
    checkpoint = path.with_name(path.name + ".checkpoint.jsonl")
    recorded = [json.loads(line)["block"] for line in checkpoint.read_text().splitlines()[1:]]
    assert [1, 1] not in recorded and not path.exists()
    assert multiprocessing.active_children() == []
    monkeypatch.setattr(blocks, "_run_block", real)
    assert blocks.run_pass_two(workers=2, **kwargs).blocks_resumed == len(recorded)


def test_the_cpu_is_divided_among_the_workers(monkeypatch) -> None:
    monkeypatch.setattr(blocks.os, "cpu_count", lambda: 24)
    assert [blocks.threads_per_worker(w) for w in (1, 5, 24, 48)] == [24, 4, 1, 1]
    assert blocks.threads_per_worker(5, 2) == 2
    monkeypatch.setattr(blocks.os, "cpu_count", lambda: None)
    assert blocks.threads_per_worker(4) == 1
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError, match="numba_threads_per_worker"):
            blocks.threads_per_worker(2, bad)


def test_a_worker_limits_its_numba_threads_and_ignores_interrupts() -> None:
    import multiprocessing
    import signal

    import numba

    context = multiprocessing.get_context("fork")
    answers = context.Queue()

    def child():
        blocks._WORKER.update(arguments={"horizon_store": None, "observer_height_m": 0.0})
        blocks._worker_init(2, [])
        answers.put((numba.get_num_threads(), signal.getsignal(signal.SIGINT) == signal.SIG_IGN))

    process = context.Process(target=child)
    process.start()
    answer = answers.get(timeout=60)
    process.join(10)
    assert answer == (2, True) and process.exitcode == 0
    assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN           # this process is untouched


def test_workers_run_after_the_parent_has_used_a_parallel_numba_kernel(prepared, tables) -> None:
    import numba
    import numpy as np

    @numba.njit(parallel=True)
    def warm(values):
        total = 0.0
        for index in numba.prange(values.shape[0]):
            total += values[index]
        return total

    warm(np.arange(1000.0))                                             # the parent has used the threading layer
    blocks.run_pass_two(**parallel_kwargs(prepared, "afternumba"), workers=2, numba_threads_per_worker=1)
    assert (prepared.directory / "afternumba.csv").read_bytes() == tables[1][0].read_bytes()


def test_the_memory_a_worker_needs_is_the_block_with_its_halo() -> None:
    assert blocks.estimate_block_memory_bytes(8) == 100 * 128 * 128 * 1440 * 4
    assert blocks.estimate_block_memory_bytes(4) == 36 * 128 * 128 * 1440 * 4
    assert blocks.estimate_block_memory_bytes(2, halo_patches=1) / 1e9 == pytest.approx(1.51, abs=0.01)


# --- SPICE in forked workers -----------------------------------------------------------------------------


def test_the_loaded_spice_kernels_are_listed_without_meta_kernels_and_reloaded_from_scratch(monkeypatch) -> None:
    from types import SimpleNamespace

    calls = []
    entries = [("meta.tm", "META"), ("naif.tls", "TEXT"), ("de440.bsp", "SPK"), ("moon.bpc", "PCK")]
    fake = SimpleNamespace(
        ktotal=lambda kind: len(entries),
        kdata=lambda index, kind: (entries[index][0], entries[index][1], "", index),
        kclear=lambda: calls.append("kclear"),
        furnsh=lambda name: calls.append(name),
    )
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    files = blocks._loaded_spice_kernels()
    assert files == ["naif.tls", "de440.bsp", "moon.bpc"]
    blocks._refurnish_spice(files)
    assert calls == ["kclear", "naif.tls", "de440.bsp", "moon.bpc"]
    calls.clear()
    blocks._refurnish_spice([])                                          # nothing loaded: SPICE is left alone
    assert calls == []
    monkeypatch.delitem(sys.modules, "spiceypy")
    assert blocks._loaded_spice_kernels() == []                          # SPICE was never imported


def test_workers_read_the_same_spice_kernels_without_corrupting_each_other() -> None:
    """Forked workers share the parent's open kernel files. Without the re-load in the worker
    initializer, five workers reading different times gave silently wrong Sun vectors and
    SPICE(INVALIDRADIUS) errors (25 wrong results and 39 errors in 3,000 calls). It runs in a
    fresh process so that importing SPICE does not leak into the other tests."""
    import subprocess
    from pathlib import Path

    here = Path(__file__).resolve().parent
    root = here.parents[1]
    done = subprocess.run(
        [sys.executable, str(here / "spice_fork_check.py")], capture_output=True, text=True, timeout=300,
        env={"PYTHONPATH": f"{root / 'src'}", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    if done.returncode == 3:
        pytest.skip(done.stdout.strip())                  # SPICE or its default kernels are not installed
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result == {"wrong": 0, "errors": []}
