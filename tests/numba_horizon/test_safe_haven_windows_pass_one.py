"""Stage-2 pass 1 on a small synthetic scene, and its selection rules on stubs.

See docs/safe-haven-windows-spec.md, sections 5 and 8.  The scene is in
``windows_scene.py``: stage-1 products are written by the real stage-1 code, so a
pass-1 score that disagreed with them raises ``PlanConsistencyError``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_windows as windows
from lunarscout._numba_horizon.file_format import HorizonTileStore, read_patch_horizons
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.safe_haven_windows import (
    FloodGrid,
    PixelRecord,
    PlanConsistencyError,
    Stage1Rasters,
    WindowParameters,
    run_pass_one,
)
from lunarscout.products import _load_dem

PARAMETERS = WindowParameters(
    cell_patches=1, rover_speed_m_s=1.0 / 1800.0, haven_neighbourhood_m=3.0,
    seed_separation_m=4.0,
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("scene")
    dem_path, horizons_path, products, stamps = scene.stage_one(directory)
    dem, _georef = _load_dem(dem_path)
    return SimpleNamespace(
        directory=directory, dem=dem, products=products, stamps=stamps,
        store=HorizonTileStore(horizons_path),
        stage1=Stage1Rasters(products.hibernation, products.joint_total),
    )


def options(prepared, **overrides):
    values = dict(
        dem=prepared.dem, horizon_store=prepared.store, stage1=prepared.stage1,
        times=prepared.stamps, sun_vectors_at=scene.sun_vectors,
        earth_vectors_at=scene.earth_vectors, parameters=PARAMETERS,
    )
    values.update(overrides)
    return values


@pytest.fixture(scope="module")
def run(prepared):
    report: list = []
    records = list(windows.iter_cell_months(report=report, **options(prepared)))
    return records, report[0]


def test_only_the_months_with_a_usable_haven_are_planned(run) -> None:
    records, report = run
    # January and February: shadow of 42 h in the cluster; March: no shadow in the
    # outage.  April: the cluster scores 0 too, but the Sun sets again right after
    # the outage ends, so no seed is lit within the 24 h search and the month is dropped.
    assert [record["month"] for record in records] == [0, 1, 2]
    counts = report.counts
    assert (counts.cell_months_examined, counts.cell_months_with_haven,
            counts.cell_months_kept) == (4, 4, 3)
    assert counts.invalid_no_lit_boundary == 5
    assert counts.seeds_selected == 20 and counts.seeds_valid == 15
    assert (report.flood_step_minutes, report.flood_steps) == (30, 96)


def test_the_isolated_haven_pixel_never_becomes_a_seed(run) -> None:
    records, report = run
    # The outlier is a haven in January, February and March: 1 of 29 pixels in the disk.
    assert report.counts.dropped_by_nearby_fraction == 3
    for record in records:
        for seed in record["seeds"]:
            assert (seed["row"], seed["column"]) != scene.OUTLIER
            assert seed["nearby_haven_fraction"] >= PARAMETERS.min_nearby_haven_fraction
            assert seed["nearby_pixel_count"] <= 29


def test_seeds_are_cluster_pixels_that_keep_their_distance(run, prepared) -> None:
    records, _report = run
    cluster_rows = range(4, 12)
    for record in records:
        seeds = record["seeds"]
        assert 1 < len(seeds) <= PARAMETERS.max_seeds_per_cell_month
        for seed in seeds:
            assert seed["row"] in cluster_rows and seed["column"] in cluster_rows
            assert seed["hibernation_hours"] == (42.0 if record["month"] < 2 else 0.0)
        for index, first in enumerate(seeds):
            for second in seeds[index + 1:]:
                distance = np.hypot(first["row"] - second["row"], first["column"] - second["column"])
                # identical timing, so the 4 pixel separation applies
                assert distance >= PARAMETERS.seed_separation_m


def test_the_nearby_fraction_counts_cluster_pixels_in_the_disk(run) -> None:
    records, _report = run
    january = {(s["row"], s["column"]): s for s in records[0]["seeds"]}
    corner = january[(4, 4)]
    # Disk of radius 3 pixels around (4, 4): 29 pixels, all inside the 20 x 20 DEM;
    # the cluster covers rows and columns 4..11, which leaves the 3 by 3 quadrant
    # offsets (0..3, 0..3) inside the disk: 4 + 3 + 3 + 2 = 11 pixels.
    assert corner["nearby_pixel_count"] == 29
    assert corner["nearby_haven_count"] == 11
    assert corner["nearby_haven_fraction"] == pytest.approx(11 / 29)


def test_boundaries_are_the_first_lit_step_away_from_the_hibernation(run, prepared) -> None:
    records, _report = run
    grid = FloodGrid(30)
    cap = 48
    horizons = read_patch_horizons(prepared.store, 0, 0, 0.0)
    for record in records:
        for seed in record["seeds"]:
            nominal_before = grid.floor_index(seed["outage_start_us"])
            nominal_after = grid.ceil_index(seed["outage_end_us"])
            before = np.arange(seed["boundary_before_step"], nominal_before + 1)
            after = np.arange(nominal_after, seed["boundary_after_step"] + 1)
            steps = np.concatenate((before, after))
            session = LightmapCpuSession(time_batch_size=len(steps))
            (batch,) = list(session.iter_patch_fraction_batches(
                prepared.dem, horizons, scene.sun_vectors(grid.times(steps)),
                tile_y=0, tile_x=0, valid_height=scene.SIZE, valid_width=scene.SIZE,
            ))
            lit = batch[:, seed["row"], seed["column"]].astype(np.float64) >= 1.0 - 1e-6
            before_lit, after_lit = lit[: before.size], lit[before.size:]
            assert before_lit[0] and not before_lit[1:].any()    # lit only at the boundary
            assert after_lit[-1] and not after_lit[:-1].any()
            assert nominal_before - seed["boundary_before_step"] <= cap
            assert seed["boundary_after_step"] - nominal_after <= cap


def test_the_lighting_ranges_and_chunks_follow_the_boundaries(run) -> None:
    records, _report = run
    grid = FloodGrid(30)
    for record in records:
        before = [s["boundary_before_step"] for s in record["seeds"]]
        after = [s["boundary_after_step"] for s in record["seeds"]]
        assert record["before_range_steps"] == [min(before) - 96, max(before)]
        assert record["after_range_steps"] == [min(after), max(after) + 96]
        low = record["before_range_steps"][0] // grid.steps_per_chunk
        high = record["after_range_steps"][1] // grid.steps_per_chunk
        assert record["chunks"][0] == low and record["chunks"][-1] == high
        # the outage between the two ranges is not lit: the chunks have a hole
        assert len(record["chunks"]) < high - low + 1


def test_the_report_counts_the_lighting_tasks(run) -> None:
    records, report = run
    # One patch holds the whole scene; every cell-month uses its own chunks.
    assert report.counts.tasks_unshared == sum(len(r["chunks"]) for r in records)
    assert report.counts.tasks_shared == len({c for r in records for c in r["chunks"]})
    assert report.counts.tasks_shared <= report.counts.tasks_unshared


def test_the_plan_is_written_as_json_lines_with_a_report(prepared, tmp_path) -> None:
    path = tmp_path / "plan.jsonl"
    report = run_pass_one(plan_path=path, **options(prepared))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["month"] for line in lines] == [0, 1, 2]
    assert lines[0]["seeds"][0]["outage_start_utc"].startswith("2027-01-")
    assert report.as_dict()["counts"]["cell_months_kept"] == 3


def test_two_hour_byte_products_give_the_same_plan(prepared, tmp_path) -> None:
    import lunarscout as ls

    def to_bytes(hours):
        counts = np.clip(np.ceil(hours / 2.0), 0, 255)
        return np.where(np.isnan(counts), 255, counts).astype(np.uint8)

    dem_path, horizons_path = scene.write_inputs(tmp_path)
    times = ls.times(scene.START, scene.STOP, step_hours=scene.STEP_HOURS)
    products = ls.generate_safe_haven_products(
        dem_path, horizons_path, tmp_path / "bytes", times=times,
        sun_vectors_m=scene.sun_vectors(times.values), earth_vectors_m=scene.earth_vectors(times.values),
        backend="cpu", nodata=255, output_transform=to_bytes, output_dtype=np.uint8,
        output_transform_id="safe-haven-2h-ceil-clip-uint8-v2",
    )
    stage1 = Stage1Rasters(products.hibernation, products.joint_total, hours_per_count=2.0, nodata=255)

    byte_records = list(windows.iter_cell_months(**options(prepared, stage1=stage1)))
    float_records = list(windows.iter_cell_months(**options(prepared)))

    assert byte_records == float_records


def test_cells_can_be_planned_one_at_a_time(prepared) -> None:
    whole = list(windows.iter_cell_months(**options(prepared)))
    chosen = list(windows.iter_cell_months(**options(prepared, cells=[(0, 0)])))
    assert chosen == whole
    assert list(windows.iter_cell_months(**options(prepared, cells=[]))) == []


def test_a_disagreement_with_stage_1_is_an_error_not_a_silent_difference(prepared) -> None:
    def shifted_sun(stamps):
        return scene.sun_vectors(np.asarray(stamps).astype("datetime64[us]") + np.timedelta64(3, "D"))

    with pytest.raises(PlanConsistencyError):
        list(windows.iter_cell_months(**options(prepared, sun_vectors_at=shifted_sun)))


def test_months_must_match_the_stage_1_bands(prepared) -> None:
    with pytest.raises(ValueError, match="months"):
        list(windows.iter_cell_months(**options(prepared, times=prepared.stamps[:200])))


# --- the selection rules, on stub timelines -----------------------------------------


US_PER_HOUR = 3_600_000_000


class StubEvaluator:
    """Returns a prepared record for each pixel and counts the evaluations."""

    def __init__(self, parameters, timing, scores=None):
        self.parameters = parameters
        self.time = SimpleNamespace(sun_vectors_at=None)
        self.signals = None
        self.timing = timing          # (row, column) -> outage start hours
        self.scores = scores or {}
        self.calls = []

    def evaluate(self, row, column, month):
        self.calls.append((row, column))
        start = self.timing(row, column)
        if start is None:
            return "no_record"
        return PixelRecord(
            row=row, column=column, score_hours=self.scores.get((row, column), 12.0),
            outage_first=0, outage_last=1, run_first=-1, run_last=-1,
            outage_start_us=int(start * US_PER_HOUR), outage_end_us=int((start + 100) * US_PER_HOUR),
            earth_visible_before_hours=48.0, earth_visible_after_hours=48.0,
        )


def stub_select(monkeypatch, hibernation, joint, timing, *, scores=None, **overrides):
    settings = dict(
        cell_patches=1, seed_separation_m=5.0, haven_neighbourhood_m=2.0,
        min_nearby_haven_fraction=0.0, rover_speed_m_s=1.0 / 1800.0,
    )
    settings.update(overrides)
    parameters = WindowParameters(**settings)
    evaluator = StubEvaluator(parameters, timing, scores)
    monkeypatch.setattr(windows, "find_boundaries", lambda *args: (10, 20))
    counts = windows.PlanCounts()
    seeds = windows.select_seeds(
        hibernation_hours=hibernation, joint_total_hours=joint, window_origin=(100, 200),
        core=(0, hibernation.shape[0], 0, hibernation.shape[1]), month=0, evaluator=evaluator,
        grid=FloodGrid(30), pixel_size_m=1.0, expected=lambda hours: hours,
        counts=counts,
    )
    return seeds, evaluator, counts


def havens(shape=(30, 30), score=12.0):
    return np.full(shape, score)


def test_seeds_are_taken_in_descending_proxy_order_and_keep_their_separation(monkeypatch) -> None:
    hibernation = havens()
    joint = np.zeros((30, 30))
    joint[3, 3], joint[20, 20] = 9.0, 7.0
    # A half-hour flood is one step, so the proxy is a 3 x 3 box sum.
    seeds, evaluator, _counts = stub_select(
        monkeypatch, hibernation, joint, lambda r, c: 0.0,
        operational_hours=0.5, max_seeds_per_cell_month=2,
    )
    # (2, 2) is the first of the nine pixels that see the 9; its eight neighbours are
    # probed and found to be duplicates before the pixels around the 7 are reached.
    assert [(s.row, s.column) for s in seeds] == [(102, 202), (119, 219)]
    assert len(evaluator.calls) == 10


def test_a_nearby_pixel_with_a_different_timing_is_a_separate_seed(monkeypatch) -> None:
    hibernation = np.full((12, 12), np.nan)
    hibernation[5, 5] = hibernation[5, 6] = 12.0
    joint = np.zeros((12, 12))
    timing = lambda row, column: 0.0 if column == 205 else 20.0     # 20 h apart, tolerance 6 h
    seeds, _evaluator, _counts = stub_select(monkeypatch, hibernation, joint, timing)
    assert sorted((s.row, s.column) for s in seeds) == [(105, 205), (105, 206)]


def test_a_nearby_pixel_with_the_same_timing_is_a_duplicate(monkeypatch) -> None:
    hibernation = np.full((12, 12), np.nan)
    hibernation[5, 5] = hibernation[5, 6] = 12.0
    joint = np.zeros((12, 12))
    timing = lambda row, column: 0.0 if column == 205 else 5.0     # 5 h apart is within 6 h
    seeds, _evaluator, _counts = stub_select(monkeypatch, hibernation, joint, timing)
    assert [(s.row, s.column) for s in seeds] == [(105, 205)]


def test_timing_is_compared_at_both_ends_of_the_outage(monkeypatch) -> None:
    hibernation = np.full((12, 12), np.nan)
    hibernation[5, 5] = hibernation[5, 6] = 12.0
    joint = np.zeros((12, 12))

    class EndsDiffer(StubEvaluator):
        def evaluate(self, row, column, month):
            record = super().evaluate(row, column, month)
            if column == 6:     # same start, end 30 hours later (window origin is 0 here)
                record = PixelRecord(**{**record.__dict__, "outage_end_us": record.outage_end_us + 30 * US_PER_HOUR})
            return record

    parameters = WindowParameters(cell_patches=1, seed_separation_m=5.0, haven_neighbourhood_m=2.0,
                                  min_nearby_haven_fraction=0.0, rover_speed_m_s=1.0 / 1800.0)
    evaluator = EndsDiffer(parameters, lambda r, c: 0.0)
    monkeypatch.setattr(windows, "find_boundaries", lambda *args: (10, 20))
    seeds = windows.select_seeds(
        hibernation_hours=hibernation, joint_total_hours=joint, window_origin=(0, 0),
        core=(0, 12, 0, 12), month=0, evaluator=evaluator, grid=FloodGrid(30), pixel_size_m=1.0,
        expected=lambda hours: hours, counts=windows.PlanCounts(),
    )
    assert len(seeds) == 2


def test_probes_of_nearby_pixels_are_limited_per_seed(monkeypatch) -> None:
    hibernation = havens((12, 12))
    joint = np.zeros((12, 12))
    seeds, evaluator, counts = stub_select(
        monkeypatch, hibernation, joint, lambda r, c: 0.0, timing_probes_per_seed=3,
        max_seeds_per_cell_month=1,
    )
    assert len(seeds) == 1 and len(evaluator.calls) == 1        # the limit stops at one seed
    seeds, evaluator, counts = stub_select(
        monkeypatch, hibernation, joint, lambda r, c: 0.0, timing_probes_per_seed=3,
        max_seeds_per_cell_month=2,
    )
    # one pixel accepted, then 3 near probes (all duplicates), then the far pixels
    assert len(seeds) == 2
    near = [call for call in evaluator.calls[1:]
            if np.hypot(call[0] - evaluator.calls[0][0], call[1] - evaluator.calls[0][1]) < 5.0]
    assert len(near) <= 3


def test_pixels_without_a_record_are_counted_and_skipped(monkeypatch) -> None:
    hibernation = np.full((12, 12), np.nan)
    hibernation[2, 2] = hibernation[9, 9] = 12.0
    joint = np.zeros((12, 12))
    seeds, _evaluator, counts = stub_select(
        monkeypatch, hibernation, joint, lambda r, c: None if (r, c) == (102, 202) else 0.0,
    )
    assert [(s.row, s.column) for s in seeds] == [(109, 209)]
    assert counts.invalid_no_record == 1 and counts.timelines_evaluated == 2


def test_a_wrong_recomputed_score_raises(monkeypatch) -> None:
    hibernation = np.full((12, 12), np.nan)
    hibernation[5, 5] = 12.0
    with pytest.raises(PlanConsistencyError):
        stub_select(monkeypatch, hibernation, np.zeros((12, 12)), lambda r, c: 0.0,
                    scores={(105, 205): 18.0})


def test_the_cell_core_decides_which_pixels_can_be_seeds(monkeypatch) -> None:
    hibernation = havens((12, 12))
    parameters = WindowParameters(cell_patches=1, seed_separation_m=5.0, haven_neighbourhood_m=2.0,
                                  min_nearby_haven_fraction=0.0, rover_speed_m_s=1.0 / 1800.0)
    evaluator = StubEvaluator(parameters, lambda r, c: 0.0)
    monkeypatch.setattr(windows, "find_boundaries", lambda *args: (10, 20))
    seeds = windows.select_seeds(
        hibernation_hours=hibernation, joint_total_hours=np.zeros((12, 12)), window_origin=(0, 0),
        core=(4, 8, 4, 8), month=0, evaluator=evaluator, grid=FloodGrid(30), pixel_size_m=1.0,
        expected=lambda hours: hours, counts=windows.PlanCounts(),
    )
    assert seeds and all(4 <= s.row < 8 and 4 <= s.column < 8 for s in seeds)


# --- the candidate evaluator on the scene --------------------------------------------


def evaluator_for(prepared, stamps, **parameters):
    time = windows.TimeContext.build(stamps, scene.sun_vectors, scene.earth_vectors)
    settings = WindowParameters(cell_patches=1, rover_speed_m_s=1.0 / 1800.0, **parameters)
    signals = windows.PixelSignals(prepared.dem, prepared.store)
    return windows.CandidateEvaluator(signals, time, settings)


def contiguous(flags):
    flags = list(flags)
    return flags.index(False) if False in flags else len(flags)


def test_earth_visibility_beside_the_hibernation_is_measured_and_capped(prepared) -> None:
    evaluator = evaluator_for(prepared, prepared.stamps, operational_hours=720.0)
    available = scene.earth_elevation(prepared.stamps) - scene.LOW_HORIZON >= 2.0
    for month in (0, 1):
        record = evaluator.evaluate(5, 5, month)
        first, last = record.outage_first, record.outage_last
        assert not available[first] and available[first - 1]
        assert not available[last] and available[last + 1]
        before = contiguous(available[:first][::-1])
        after = contiguous(available[last + 1:])
        assert record.earth_visible_before_hours == min(720.0, before * 6.0)
        assert record.earth_visible_after_hours == min(720.0, after * 6.0)
    # January's visibility before the outage is cut short by the start of the series.
    january = evaluator.evaluate(5, 5, 0)
    assert january.earth_visible_before_hours == january.outage_first * 6.0 < 720.0
    # The default 48 hours is a cap.
    capped = evaluator_for(prepared, prepared.stamps).evaluate(5, 5, 0)
    assert (capped.earth_visible_before_hours, capped.earth_visible_after_hours) == (48.0, 48.0)


def test_a_month_without_a_start_has_no_record(prepared) -> None:
    stamps = prepared.stamps[(prepared.stamps >= np.datetime64("2027-01-20"))
                             & (prepared.stamps < np.datetime64("2027-01-30"))]
    assert evaluator_for(prepared, stamps).evaluate(5, 5, 0) == "no_record"


def test_an_outage_that_never_ends_in_the_series_is_reported(prepared) -> None:
    stamps = prepared.stamps[prepared.stamps < np.datetime64("2027-01-17")]
    assert evaluator_for(prepared, stamps).evaluate(5, 5, 0) == "outage_unclosed"


def test_a_pixel_without_a_horizon_tile_is_reported(prepared, tmp_path) -> None:
    time = windows.TimeContext.build(prepared.stamps, scene.sun_vectors, scene.earth_vectors)
    signals = windows.PixelSignals(prepared.dem, HorizonTileStore(tmp_path / "empty"))
    evaluator = windows.CandidateEvaluator(signals, time, PARAMETERS)
    assert evaluator.evaluate(5, 5, 0) == "no_horizon"


def test_the_evaluator_scores_the_scene_as_stage_1_does(prepared) -> None:
    import rasterio

    with rasterio.open(prepared.products.hibernation) as dataset:
        stage_one = dataset.read()
    evaluator = evaluator_for(prepared, prepared.stamps)
    for month in range(4):
        for row, column in ((5, 5), (0, 0), (17, 17)):
            record = evaluator.evaluate(row, column, month)
            if isinstance(record, str):
                assert np.isnan(stage_one[month, row, column])
            else:
                assert record.score_hours == stage_one[month, row, column]


# --- boundaries of the rules, on stubs --------------------------------------------------


def test_timings_exactly_one_tolerance_apart_are_duplicates(monkeypatch) -> None:
    hibernation = np.full((12, 12), np.nan)
    hibernation[5, 5] = hibernation[5, 6] = 12.0
    timing = lambda row, column: 0.0 if column == 205 else 6.0      # exactly the 6 h tolerance
    seeds, _evaluator, _counts = stub_select(monkeypatch, hibernation, np.zeros((12, 12)), timing)
    assert len(seeds) == 1
    timing = lambda row, column: 0.0 if column == 205 else 6.001
    seeds, _evaluator, _counts = stub_select(monkeypatch, hibernation, np.zeros((12, 12)), timing)
    assert len(seeds) == 2


def test_a_fraction_exactly_at_the_threshold_is_kept_and_a_score_exactly_at_the_limit_is_a_haven(
    monkeypatch,
) -> None:
    hibernation = np.full((12, 12), np.nan)
    hibernation[5, 5] = hibernation[5, 6] = 72.0       # exactly the survival limit
    # Each pixel sees 2 haven pixels among the 13 in a 2 pixel radius disk.
    seeds, _evaluator, counts = stub_select(
        monkeypatch, hibernation, np.zeros((12, 12)), lambda r, c: 0.0,
        min_nearby_haven_fraction=2 / 13, scores={(105, 205): 72.0, (105, 206): 72.0},
    )
    assert len(seeds) == 1 and counts.dropped_by_nearby_fraction == 0
    seeds, _evaluator, counts = stub_select(
        monkeypatch, hibernation, np.zeros((12, 12)), lambda r, c: 0.0,
        min_nearby_haven_fraction=2 / 13 + 1e-6, scores={(105, 205): 72.0, (105, 206): 72.0},
    )
    assert seeds == [] and counts.dropped_by_nearby_fraction == 2
    hibernation[5, 5] = hibernation[5, 6] = 72.001     # just over the limit
    seeds, _evaluator, counts = stub_select(
        monkeypatch, hibernation, np.zeros((12, 12)), lambda r, c: 0.0, scores={(105, 205): 72.001},
    )
    assert seeds == [] and counts.haven_pixels == 0


def test_stage_1_values_are_expected_as_stored(prepared) -> None:
    floats = Stage1Rasters(prepared.products.hibernation, prepared.products.joint_total)
    counts = Stage1Rasters(prepared.products.hibernation, prepared.products.joint_total,
                           hours_per_count=2.0, nodata=255)
    assert floats.expected(7.0) == 7.0
    assert (counts.expected(7.0), counts.expected(6.0), counts.expected(0.0)) == (8.0, 6.0, 0.0)
    assert counts.expected(5000.0) == 510.0                      # saturated at 255 counts
    with pytest.raises(ValueError):
        Stage1Rasters(prepared.products.hibernation, prepared.products.joint_total, hours_per_count=2.0)


# --- flood boundaries on stub signals ------------------------------------------------------


class StepSignals:
    """Sun fractions chosen per flood step: vectors carry the step index in column 0."""

    def __init__(self, fraction_of_step):
        self.fraction_of_step = fraction_of_step

    def series(self, row, column, vectors, *, margin):
        steps = np.asarray(vectors)[:, 0].astype(np.int64)
        return np.array([self.fraction_of_step(step) for step in steps], dtype=np.float32)


def step_time_context():
    grid = FloodGrid(30)

    def sun_vectors_at(stamps):
        steps = np.asarray(stamps).astype("datetime64[us]").astype(np.int64) // grid.step_us
        return np.stack([steps, np.zeros_like(steps), np.zeros_like(steps)], axis=1).astype(float)

    return grid, SimpleNamespace(sun_vectors_at=sun_vectors_at)


def record_at(start_us, end_us):
    return PixelRecord(
        row=0, column=0, score_hours=0.0, outage_first=0, outage_last=0, run_first=-1, run_last=-1,
        outage_start_us=start_us, outage_end_us=end_us,
        earth_visible_before_hours=0.0, earth_visible_after_hours=0.0,
    )


def boundaries(fraction_of_step, start_us, end_us, **parameters):
    grid, time = step_time_context()
    return windows.find_boundaries(
        StepSignals(fraction_of_step), time, grid, record_at(start_us, end_us),
        WindowParameters(**parameters),
    )


def test_unlit_outage_times_are_rounded_toward_the_hibernation_before_the_search() -> None:
    step = FloodGrid(30).step_us
    # Always lit: the boundaries are the nominal ones, floor of the start and ceil of the end.
    result = boundaries(lambda k: 1.0, 100 * step + step // 2, 200 * step + 1)
    assert result == (100, 201)
    result = boundaries(lambda k: 1.0, 100 * step, 200 * step)
    assert result == (100, 200)


def test_the_boundary_moves_away_from_the_hibernation_to_the_first_lit_step() -> None:
    step = FloodGrid(30).step_us
    # Dark from step 90 to 100 before, and from 200 to 205 after.
    lit = lambda k: 0.0 if 90 <= k <= 100 or 200 <= k <= 205 else 1.0
    assert boundaries(lit, 100 * step, 200 * step) == (89, 206)


def test_the_search_stops_at_the_cap_and_no_further() -> None:
    step = FloodGrid(30).step_us          # a 24 h cap is 48 steps
    start, end = 1000 * step, 2000 * step
    only = lambda before, after: (lambda k: 1.0 if k in (before, after) else 0.0)
    assert boundaries(only(1000 - 48, 2000 + 48), start, end) == (952, 2048)
    assert boundaries(only(1000 - 49, 2000 + 48), start, end) is None
    assert boundaries(only(1000 - 48, 2000 + 49), start, end) is None
    assert boundaries(only(1000 - 12, 2000 + 12), start, end, boundary_search_hours=6.0) == (988, 2012)
    assert boundaries(only(1000 - 13, 2000 + 12), start, end, boundary_search_hours=6.0) is None


def test_the_lit_test_has_a_rounding_tolerance_and_follows_the_operating_fraction() -> None:
    step = FloodGrid(30).step_us
    assert boundaries(lambda k: 1.0 - 1e-7, 100 * step, 200 * step) == (100, 200)
    assert boundaries(lambda k: 1.0 - 1e-4, 100 * step, 200 * step) is None
    assert boundaries(lambda k: 0.99, 100 * step, 200 * step, operating_sunlight_fraction=0.99) == (100, 200)


# --- other pure rules -------------------------------------------------------------------------


def test_ranges_and_chunks_span_the_earliest_and_latest_boundaries() -> None:
    def seed(before, after):
        return windows.Seed(0, 0, 0.0, 0, 0, before, after, 1, 1, 1.0, 0.0, 0.0)

    grid = FloodGrid(30)
    seeds = [seed(1000, 1500), seed(1100, 1700), seed(1040, 1600)]
    assert windows.lighting_ranges(seeds, 96) == ((1000 - 96, 1100), (1500, 1700 + 96))
    # 48 steps per chunk: 904..1100 touches chunks 18..22; 1500..1796 touches 31..37.
    assert windows.lighting_chunks(seeds, grid, 96) == [18, 19, 20, 21, 22, 31, 32, 33, 34, 35, 36, 37]


def test_the_window_around_a_cell_is_clipped_to_the_raster() -> None:
    window, core = windows.cell_window(1, 0, cell_size=256, halo=128, height=600, width=300)
    assert window == (128, 600, 0, 300) and core == (128, 384, 0, 256)
    window, core = windows.cell_window(2, 1, cell_size=256, halo=128, height=600, width=300)
    assert window == (384, 600, 128, 300) and core == (128, 216, 128, 172)
    assert windows.halo_pixels(96) == 128 and windows.halo_pixels(128) == 128
    assert windows.halo_pixels(129) == 256


def test_equal_proxies_are_taken_in_row_then_column_order(monkeypatch) -> None:
    hibernation = np.full((12, 12), np.nan)
    hibernation[5, 9] = hibernation[7, 2] = 12.0
    seeds, _evaluator, _counts = stub_select(
        monkeypatch, hibernation, np.zeros((12, 12)), lambda r, c: 0.0, max_seeds_per_cell_month=1,
    )
    assert [(s.row, s.column) for s in seeds] == [(105, 209)]


# --- strict thresholds on stub signals --------------------------------------------------------


class TimelineSignals:
    """Fraction and Earth margin by sample index; vectors carry the index in column 0."""

    def __init__(self, fraction, margin):
        self.fraction, self.margin = np.asarray(fraction, np.float32), np.asarray(margin, np.float32)

    def series(self, row, column, vectors, *, margin, flat_elevation=False):
        indices = np.asarray(vectors)[:, 0].astype(int)
        return (self.margin if margin else self.fraction)[indices]


def test_equality_is_neither_shadow_nor_outage() -> None:
    count = 40
    stamps = np.datetime64("2027-01-01T00:00:00", "us") + np.arange(count) * np.timedelta64(6, "h")

    def indexed(values):
        steps = (np.asarray(values).astype("datetime64[us]") - stamps[0]) // np.timedelta64(6, "h")
        return np.stack([steps, 0 * steps, 0 * steps], axis=1).astype(float)

    fraction = np.full(count, 1.0)
    margin = np.full(count, 3.0)
    margin[5] = 2.0               # equal to the threshold: Earth is available
    margin[6:10] = 1.0            # the outage, samples 6..9
    fraction[7] = 0.1             # shadow
    fraction[8] = 0.25            # equal to the threshold: not shadow
    time = windows.TimeContext.build(stamps, indexed, indexed)
    parameters = WindowParameters(
        cell_patches=1, rover_speed_m_s=1.0 / 1800.0, degenerate_outage_window_hours=0.0
    )
    evaluator = windows.CandidateEvaluator(
        TimelineSignals(fraction, margin), time, parameters,
        earth_elevation_threshold_deg=2.0, sunlight_fraction_threshold=0.25, merge_gap_hours=0.0,
    )

    record = evaluator.evaluate(0, 0, 0)

    assert (record.outage_first, record.outage_last) == (6, 9)
    assert record.score_hours == 6.0              # one shadow sample (7), not two
    assert (record.run_first, record.run_last) == (7, 7)


# --- progress and the SPICE adapter --------------------------------------------------------


def test_progress_is_reported_after_every_cell(prepared) -> None:
    calls = []
    list(windows.iter_cell_months(**options(prepared, progress=lambda done, total: calls.append((done, total)))))
    assert calls == [(1, 1)]
    calls.clear()
    list(windows.iter_cell_months(**options(prepared, cells=[(0, 0), (0, 0)], progress=lambda d, t: calls.append((d, t)))))
    assert calls == [(1, 2), (2, 2)]


def test_the_spice_adapter_passes_utc_times_to_the_vector_generator(monkeypatch) -> None:
    seen = {}

    def fake(body, moments):
        seen["body"], seen["moments"] = body, moments
        return SimpleNamespace(vectors_m=np.zeros((len(moments), 3)))

    from lunarscout._numba_horizon import product_vectors

    monkeypatch.setattr(product_vectors, "generate_moon_me_vectors", fake)
    stamps = np.array(["2027-01-01T00:00:00", "2027-01-01T00:30:00"], dtype="datetime64[us]")
    result = windows.spice_vector_source("earth")(stamps)
    assert result.shape == (2, 3) and seen["body"] == "earth"
    assert [m.isoformat() for m in seen["moments"]] == ["2027-01-01T00:00:00+00:00", "2027-01-01T00:30:00+00:00"]


def test_haven_cells_lists_cells_with_a_haven_most_months_first(prepared) -> None:
    found = windows.haven_cells(prepared.stage1, survival_limit_hours=72.0, cell_size=128)
    # One cell. Months 0 and 1 have the 8 x 8 cluster and the outlier (65 pixels); month 2
    # the same; month 3 every pixel (400). The limit decides: at 30 h nothing in months 0-1.
    assert found == [{"cell_row": 0, "cell_col": 0, "months_with_haven": 4, "haven_pixels": 3 * 65 + 400}]
    assert windows.haven_cells(prepared.stage1, survival_limit_hours=30.0, cell_size=128)[0][
        "months_with_haven"] == 2
    assert windows.haven_cells(prepared.stage1, survival_limit_hours=72.0, cell_size=128, cells=[]) == []
    calls = []
    windows.haven_cells(prepared.stage1, survival_limit_hours=72.0, cell_size=10,
                        progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]
    ordered = windows.haven_cells(prepared.stage1, survival_limit_hours=72.0, cell_size=10)
    assert [(e["months_with_haven"], e["haven_pixels"]) for e in ordered] == sorted(
        [(e["months_with_haven"], e["haven_pixels"]) for e in ordered], reverse=True)


# --- a month in which every pixel fails ------------------------------------------------------


def test_a_cell_month_where_every_timeline_fails_is_abandoned_after_the_limit(monkeypatch) -> None:
    hibernation = havens((30, 30))
    seeds, evaluator, counts = stub_select(
        monkeypatch, hibernation, np.zeros((30, 30)), lambda r, c: None, max_consecutive_failures=7,
    )
    assert seeds == [] and len(evaluator.calls) == 7
    assert (counts.cell_months_abandoned, counts.invalid_no_record) == (1, 7)


def test_a_success_resets_the_failure_count(monkeypatch) -> None:
    hibernation = np.full((30, 30), np.nan)
    hibernation[0, :10] = 12.0                 # ten haven pixels in row order
    # pixels 0..4 fail, pixel 5 works, pixels 6..9 fail: never 6 failures in a row
    working = {205}
    timing = lambda row, column: 0.0 if column in working else None
    seeds, evaluator, counts = stub_select(
        monkeypatch, hibernation, np.zeros((30, 30)), timing, max_consecutive_failures=6,
        seed_separation_m=1.0, max_seeds_per_cell_month=3,
    )
    assert [(s.row, s.column) for s in seeds] == [(100, 205)]
    assert len(evaluator.calls) == 10 and counts.cell_months_abandoned == 0


def test_seeds_found_before_abandoning_are_kept(monkeypatch) -> None:
    hibernation = np.full((30, 30), np.nan)
    hibernation[0, :20] = 12.0
    working = {200}
    timing = lambda row, column: 0.0 if column in working else None
    seeds, _evaluator, counts = stub_select(
        monkeypatch, hibernation, np.zeros((30, 30)), timing, max_consecutive_failures=5,
        seed_separation_m=1.0,
    )
    assert [(s.row, s.column) for s in seeds] == [(100, 200)] and counts.cell_months_abandoned == 1


def test_numpy_integer_coordinates_are_accepted_by_the_horizon_reader(prepared) -> None:
    signals = windows.PixelSignals(prepared.dem, prepared.store)
    column = signals.horizon_column(np.int64(5), np.int64(5))
    assert column.shape == (1440,)
    assert signals.series(np.int64(5), np.int64(5), np.zeros((1, 3)) + [0.0, 3.8e8, 1e5], margin=False).shape == (1,)


def test_the_synthetic_scene_never_abandons_a_cell_month(run) -> None:
    assert run[1].counts.cell_months_abandoned == 0
