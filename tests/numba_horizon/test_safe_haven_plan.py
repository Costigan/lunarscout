"""Stage 2 of the rover chain: the plan seeds, from the stage-1 products and the light store."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lunarscout as ls
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_plan as plan
from lunarscout._numba_horizon.safe_haven_stage3_store import StoreSignals
from lunarscout._numba_horizon.safe_haven_windows import (
    CandidateEvaluator,
    FloodGrid,
    PixelRecord,
    Stage1Rasters,
    TimeContext,
    WindowParameters,
    find_boundaries,
)
from lunarscout.errors import InputError
from test_safe_haven_stage1_store import earthside, make_store, products as stage1_products  # noqa: F401
from test_safe_haven_stage3_store import prepared  # noqa: F401  (the block scene and a 2-hour store of it)

DAY_US = 86_400_000_000
STEP_US = 120 * 60_000_000


def parameters(**overrides):
    settings = dict(cell_patches=1, haven_neighbourhood_m=3.0, seed_separation_m=4.0, operational_hours=24.0,
                    forward_only=True)
    settings.update(overrides)
    return WindowParameters(**settings)


def run_plan(prepared, path, **overrides):
    options = dict(dem=prepared.dem, stage1=prepared.stage1, light_store=prepared.light_store,
                   times=prepared.stamps, earth_vectors_at=scene.earth_vectors, parameters=parameters())
    options.update(overrides)
    return plan.plan_seeds(path, **options)


@pytest.fixture(scope="module")
def planned(prepared, tmp_path_factory):
    path = tmp_path_factory.mktemp("plan") / "seeds.csv"
    report = run_plan(prepared, path)
    return path, report, plan.read_seed_table(path)


# --- the table ---------------------------------------------------------------------------------------


def test_the_table_lists_every_seed_in_month_row_column_order_with_ids_from_one(planned) -> None:
    path, report, rows = planned
    assert path.read_text().splitlines()[0] == ",".join(plan.SEED_COLUMNS)
    assert report.seeds == len(rows) == 54
    assert [r["seed_id"] for r in rows] == list(range(1, 55))
    keys = [(r["month"], r["row"], r["column"]) for r in rows]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert {r["month"] for r in rows} == {"2027-01", "2027-02", "2027-03"}
    assert report.cells == 9 and report.cells_with_seeds == 4 and report.cell_months_with_seeds == 12
    assert report.counts.invalid_stage1_mismatch == 0 and report.counts.invalid_earth_window == 0


def test_the_report_is_written_next_to_the_table(planned) -> None:
    path, report, _rows = planned
    written = json.loads(path.with_name(path.name + ".report.json").read_text())
    assert written["seeds"] == 54 and written["by_kind"] == report.by_kind
    assert written["parameters"]["forward_only"] is True and written["earth_cycles"] == 3


def test_each_seeds_hibernation_is_the_stage_one_value(planned, prepared) -> None:
    _path, _report, rows = planned
    hibernation = prepared.stage1.read_hibernation_window(0, prepared.dem.height, 0, prepared.dem.width)
    month_index = {"2027-01": 0, "2027-02": 1, "2027-03": 2}
    for row in rows:
        assert row["hibernation_hours"] == hibernation[month_index[row["month"]], row["row"], row["column"]]
        assert row["hibernation_hours"] <= 72.0


def test_boundaries_are_the_first_lit_steps_either_side_of_the_outage(planned, prepared) -> None:
    _path, _report, rows = planned
    signals = StoreSignals(prepared.light_store, prepared.dem)
    cap = 12                                                    # the 24 h search, in 2-hour steps
    for row in rows:
        start = int(np.datetime64(row["outage_start_utc"].rstrip("Z"), "us").astype(np.int64)) // STEP_US
        end = -(-int(np.datetime64(row["outage_end_utc"].rstrip("Z"), "us").astype(np.int64)) // STEP_US)
        steps = np.arange(start - cap, end + cap + 1)
        lit = signals.series(row["row"], row["column"], None, margin=False,
                             times_us=steps * STEP_US) >= 1.0 - 1e-6
        after = steps[end - steps[0]:][lit[end - steps[0]:]]
        assert row["after_step"] == after[0]
        before = steps[:start - steps[0] + 1][lit[:start - steps[0] + 1]]
        assert row["before_step"] == before[-1]
    signals.close()


def test_the_outage_kinds_and_the_earth_cycles_of_the_scene(planned) -> None:
    _path, report, rows = planned
    # January and February: a shadow touches the outage; March: the Sun is up through it (score 0).
    for row in rows:
        assert row["outage_kind"] == ("outage_sun_lit" if row["hibernation_hours"] == 0 else "outage_in_shadow")
        assert row["earth_cycle"] == {"2027-01": 0, "2027-02": 1, "2027-03": 2}[row["month"]]
    assert report.by_kind == {"outage_in_shadow": 36, "outage_sun_lit": 18}
    assert report.seeds_without_before_boundary == 0 and report.seeds_without_cycle == 0
    assert report.earth_cycles == 3


def test_the_number_of_workers_does_not_change_the_table(prepared, planned, tmp_path) -> None:
    run_plan(prepared, tmp_path / "workers.csv", workers=2)
    assert (tmp_path / "workers.csv").read_bytes() == planned[0].read_bytes()


def test_a_subset_of_cells_gives_those_cells_seeds(prepared, planned, tmp_path) -> None:
    run_plan(prepared, tmp_path / "subset.csv", cells=[(1, 0)])
    subset = plan.read_seed_table(tmp_path / "subset.csv")
    wanted = [r for r in planned[2] if (r["cell_row"], r["cell_col"]) == (1, 0)]
    assert subset and len(subset) == len(wanted)
    for got, want in zip(subset, wanted):
        assert {k: v for k, v in got.items() if k != "seed_id"} == {k: v for k, v in want.items() if k != "seed_id"}


# --- no Earth gate; the before boundary is optional --------------------------------------------------


def test_there_is_no_earth_gate_in_forward_only_planning(prepared) -> None:
    """With 1,000 hours of Earth required either side no seed passes the gate; forward-only has no gate."""
    time = TimeContext.build(prepared.stamps, None, scene.earth_vectors)
    row, column, month = 60, 60, 0

    def evaluate(**overrides):
        settings = parameters(operational_hours=1000.0, **overrides)
        signals = StoreSignals(prepared.light_store, prepared.dem)
        try:
            return CandidateEvaluator(signals, time, settings).evaluate(row, column, month)
        finally:
            signals.close()

    assert evaluate(forward_only=False) == "earth_window"
    record = evaluate()
    assert isinstance(record, PixelRecord) and record.outage_kind == "outage_in_shadow"
    assert record.score_hours == 38.0


def lit_at(steps_lit):
    """A stand-in for the signals of one pixel: lit exactly at the given flood-grid steps."""
    lit = set(steps_lit)

    def series(row, column, vectors, *, margin, times_us=None, flat_elevation=False):
        steps = np.asarray(times_us) // STEP_US
        return np.array([1.0 if int(step) in lit else 0.0 for step in steps], dtype=np.float32)

    return SimpleNamespace(series=series)


def a_record(start_step, end_step):
    return PixelRecord(row=0, column=0, score_hours=0.0, outage_first=0, outage_last=0, run_first=-1, run_last=-1,
                       outage_start_us=start_step * STEP_US, outage_end_us=end_step * STEP_US,
                       earth_visible_before_hours=0.0, earth_visible_after_hours=0.0)


def test_a_seed_that_is_not_lit_before_its_outage_keeps_no_before_boundary() -> None:
    grid, time = FloodGrid(120), SimpleNamespace(sun_vectors_at=None)
    record = a_record(1000, 1010)
    only_after = lit_at([1013, 1020])
    assert find_boundaries(only_after, time, grid, record, parameters(forward_only=False)) is None
    assert find_boundaries(only_after, time, grid, record, parameters()) == (None, 1013)
    both = lit_at([995, 998, 1013])
    assert find_boundaries(both, time, grid, record, parameters()) == (998, 1013)
    assert find_boundaries(both, time, grid, record, parameters(forward_only=False)) == (998, 1013)
    # the after boundary is required either way
    assert find_boundaries(lit_at([995, 998]), time, grid, record, parameters()) is None


def test_a_seed_with_no_before_boundary_has_a_blank_before_step(tmp_path) -> None:
    rows = [dict(month="2027-01", cell_row=0, cell_col=0, row=1, column=2, hibernation_hours=0.0,
                 outage_kind="outage_sun_lit", earth_cycle=0, outage_start_utc="a", outage_end_utc="b",
                 before_step=None, after_step=7)]
    plan._write_table(tmp_path / "t.csv", rows, (1000.0, 20.0, 0.0, -500.0, 0.0, -20.0))
    assert tmp_path.joinpath("t.csv").read_text().splitlines()[1].endswith(",,7")
    assert plan.read_seed_table(tmp_path / "t.csv")[0]["before_step"] is None


# --- Earth never sets --------------------------------------------------------------------------------


def test_a_corner_where_earth_never_sets_gets_one_sample_outages(earthside, tmp_path) -> None:
    dem_path, _horizons, store = earthside
    from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory
    from lunarscout.products import _load_dem

    dem, _ = _load_dem(dem_path)
    made = stage1_products(tmp_path, "stage1", dem_path, light_store_path=store, degenerate_outage_window_hours=24.0)
    stamps = ls.times("2027-01-01T00:00:00Z", "2027-03-31T22:00:00Z", step_hours=2.0).values
    out = tmp_path / "earthside.csv"
    plan.plan_seeds(out, dem=dem, stage1=Stage1Rasters(made.hibernation, made.joint_total),
                    light_store=LightStoreDirectory(store, dem), times=stamps,
                    earth_vectors_at=scene.earth_vectors, parameters=parameters(degenerate_outage_window_hours=24.0, max_seeds_per_cell_month=30))
    rows = plan.read_seed_table(out)
    corner = [r for r in rows if r["row"] >= 14 and r["column"] < 6]
    cluster = [r for r in rows if 4 <= r["row"] < 12 and 4 <= r["column"] < 12]
    assert corner and cluster
    for row in corner:
        assert row["outage_kind"] == "earth_never_sets"
        start = np.datetime64(row["outage_start_utc"].rstrip("Z"), "us")
        end = np.datetime64(row["outage_end_utc"].rstrip("Z"), "us")
        assert end - start == np.timedelta64(2, "h")                     # one sample
    assert {r["outage_kind"] for r in cluster} <= {"outage_in_shadow", "outage_sun_lit"}
    assert {r["earth_cycle"] for r in corner} <= {0, 1, 2}


# --- Earth cycles ------------------------------------------------------------------------------------


def sinusoid(days=120.0, period=27.3, wiggle=0.0, step_hours=2.0, phase=0.9):
    times = np.arange(0, int(days * 24 / step_hours)) * int(step_hours * 3_600_000_000)
    t = times / DAY_US
    return times, 6.0 + 8.0 * np.sin(2 * np.pi * t / period + phase) + wiggle * np.sin(2 * np.pi * t / 3.1)


def test_the_cycle_minima_are_one_per_cycle_and_ignore_small_wiggles() -> None:
    for wiggle, tolerance in ((0.0, 0.2), (0.6, 1.5)):          # a wiggle moves a minimum, never adds one
        times, elevation = sinusoid(wiggle=wiggle)
        minima = plan.earth_minima(elevation, times) / DAY_US
        expected = (3 * np.pi / 2 - 0.9) / (2 * np.pi) * 27.3 + 27.3 * np.arange(4)
        assert len(minima) == 4
        np.testing.assert_allclose(minima, expected, atol=tolerance)


def test_the_cycle_of_a_time_is_the_nearest_minimum_or_minus_one() -> None:
    minima = np.array([10, 37, 64], dtype=np.int64) * DAY_US
    assert [plan.earth_cycle_of(minima, int(d * DAY_US)) for d in (0, 10, 22, 24, 51, 64, 79)] == [0, 0, 0, 1, 2, 2, 2]
    assert plan.earth_cycle_of(minima, int(80 * DAY_US)) == -1
    assert plan.earth_cycle_of(minima, -int(6 * DAY_US)) == -1


@pytest.mark.parametrize("days, period, code", [
    (30.0, 27.3, "plan_earth_cycles"),         # one minimum: no cycles to number
    (200.0, 45.0, "plan_earth_cycles"),        # cycles 45 days apart are not Earth's
])
def test_series_without_regular_cycles_are_refused(days, period, code) -> None:
    times, elevation = sinusoid(days=days, period=period, phase=0.0)
    with pytest.raises(InputError) as caught:
        plan.earth_minima(elevation, times)
    assert caught.value.code == code


def test_the_reference_elevation_is_earths_flat_horizon_elevation_at_the_raster_centre(prepared) -> None:
    vectors = scene.earth_vectors(prepared.stamps)
    series = plan.reference_earth_elevation(prepared.dem, vectors)
    assert series.shape == (len(prepared.stamps),) and np.isfinite(series).all()
    np.testing.assert_allclose(series, scene.earth_elevation(prepared.stamps), atol=0.01)


# --- refusals ----------------------------------------------------------------------------------------


def test_an_existing_table_is_not_replaced_without_overwrite(prepared, planned, tmp_path) -> None:
    before = planned[0].read_bytes()
    with pytest.raises(InputError) as caught:
        run_plan(prepared, planned[0])
    assert caught.value.code == "plan_output_exists" and planned[0].read_bytes() == before
    run_plan(prepared, planned[0], overwrite=True)
    assert planned[0].read_bytes() == before


@pytest.mark.parametrize("override, code", [
    (dict(parameters=parameters(forward_only=False)), "plan_not_forward_only"),
    (dict(sunlight_fraction_threshold=0.3), "safe_haven_store_thresholds"),
    (dict(earth_elevation_threshold_deg=3.0), "safe_haven_store_thresholds"),
    (dict(parameters=parameters(flood_step_minutes=30, moves_per_step=1)), "safe_haven_store_step_mismatch"),
    (dict(times=ls.times("2027-01-01T00:00:00Z", "2027-02-27T22:00:00Z", step_hours=2.0).values), "plan_grid_mismatch"),
])
def test_runs_that_the_inputs_cannot_support_are_refused_before_writing(prepared, tmp_path, override, code) -> None:
    with pytest.raises(InputError) as caught:
        run_plan(prepared, tmp_path / "refused.csv", **override)
    assert caught.value.code == code
    assert not (tmp_path / "refused.csv").exists()


def test_a_table_of_something_else_is_not_read(tmp_path) -> None:
    (tmp_path / "other.csv").write_text("a,b\n1,2\n")
    with pytest.raises(InputError) as caught:
        plan.read_seed_table(tmp_path / "other.csv")
    assert caught.value.code == "plan_table_unreadable"


# --- the script --------------------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven_stage2.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_safe_haven_stage2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_importing_the_script_starts_nothing_and_its_defaults_match_stage_one() -> None:
    module = load_script()
    stage1 = importlib.util.spec_from_file_location("run_safe_haven_stage1", SCRIPT.parent / "run_safe_haven_stage1.py")
    one = importlib.util.module_from_spec(stage1)
    stage1.loader.exec_module(one)
    assert module.STAGE1_PREFIX == one.NAME_PREFIX and module.LIGHT_STORE_PATH == one.LIGHT_STORE_PATH
    assert module.OUTPUT_PATH.name == "safe_haven_stage_2_plan_seeds.csv" and module.OVERWRITE is False
    assert module.PARAMETERS.forward_only is True and module.PARAMETERS.max_seeds_per_cell_month == 5
    assert module.PARAMETERS.survival_limit_hours == 72.0 and module.CELLS is None


def test_the_script_writes_the_seed_table_and_prints_the_report(prepared, planned, tmp_path, monkeypatch, capsys) -> None:
    module = load_script()
    out = tmp_path / "seeds.csv"
    for name, value in dict(
        DEM_PATH=prepared.directory / "dem.tif", LIGHT_STORE_PATH=prepared.store_directory,
        STAGE1_DIRECTORY=prepared.stage1.hibernation_path.parent, STAGE1_PREFIX="safe_haven", OUTPUT_PATH=out,
        START_UTC="2027-01-01T00:00:00Z", STOP_UTC="2027-03-31T22:00:00Z", STEP_HOURS=2.0,
        HOURS_PER_COUNT=None, STAGE1_NODATA=None, PLAN_WORKERS=1, PROGRESS_EVERY_CELLS=1, PARAMETERS=parameters(),
    ).items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "spice_vector_source", lambda body: scene.earth_vectors)

    assert module.main() == 0

    printed = capsys.readouterr().out
    assert '"seeds": 54' in printed and "Stage 2: 9/9 cells" in printed
    assert out.read_bytes() == planned[0].read_bytes()
    with pytest.raises(InputError, match="exists"):
        module.main()
