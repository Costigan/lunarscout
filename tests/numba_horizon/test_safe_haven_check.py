"""The safe-haven lighting check: the traced reference horizon and the comparison of stored lighting with it."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_check as check
from lunarscout._numba_horizon import safe_haven_stage0 as stage0
from lunarscout._numba_horizon.direct_reference import direct_reference_trace
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters, enu_to_moon_matrix, vector_to_lat_lon
from lunarscout._numba_horizon.safe_haven_lightstore import LightStoreSettings
from lunarscout._numba_horizon.safe_haven_plan import month_labels
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory, Stage0Settings
from lunarscout._numba_horizon.safe_haven_stage3_store import StoreSignals
from lunarscout._numba_horizon.safe_haven_windows import CandidateEvaluator, TimeContext, WindowParameters
from lunarscout.errors import InputError

import lunarscout as ls

PROJECTION = ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0)
FAST_TRACE = check.TraceSettings(chunk_samples=256)          # small scenes: do not sample 16,000 distances per ray


def make_dem(elevation, transform) -> DemGrid:
    return DemGrid(np.ascontiguousarray(elevation, dtype=np.float32), np.asarray(transform, dtype=np.float64), PROJECTION)


def rough_dem() -> DemGrid:
    rng = np.random.default_rng(7)
    x, y = np.meshgrid(np.arange(256) * 20.0, np.arange(256) * 20.0)
    elevation = 0.05 * x
    for _ in range(40):
        cx, cy = rng.uniform(0.0, 5120.0, 2)
        elevation += rng.uniform(-150.0, 300.0) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * rng.uniform(60.0, 500.0) ** 2))
    return make_dem(elevation, (100_000.0, 20.0, 0.0, -100_000.0, 0.0, -20.0))


# --- the ray trace ---------------------------------------------------------------------------------


def test_the_vectorised_trace_equals_the_direct_reference_sample_by_sample() -> None:
    dem = rough_dem()
    rng = np.random.default_rng(3)
    for column, row in rng.integers(20, 236, size=(4, 2)):
        observer = check.observer_vector(dem, int(row), int(column))
        for azimuth_index in (0, 97, 360, 700, 1100, 1439):
            azimuth = np.deg2rad(azimuth_index * 0.25)
            rotation = enu_to_moon_matrix(*vector_to_lat_lon(observer))
            direction = np.array((np.sin(azimuth), np.cos(azimuth), 0.0)) @ rotation
            distances = np.arange(1.0, 9000.0, 5.0)                   # runs past the DEM edge
            expected = direct_reference_trace(dem, observer, direction, distances)
            last_node = (expected[:, 1] >= dem.width - 1) | (expected[:, 2] >= dem.height - 1)
            if last_node.any():                                       # the trace stops at the last node, the direct one does not
                expected = expected[:int(np.argmax(last_node))]
            actual = check.trace_ray(dem, observer, direction, distances)
            assert actual.shape == expected.shape and expected.shape[0] > 0
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def test_a_trace_ends_at_the_dem_edge_and_an_invalid_node_has_no_slope() -> None:
    elevation = np.zeros((40, 40), dtype=np.float32)
    elevation[:, 24:27] = -30000.0                                    # bands of nodata nodes either side,
    elevation[:, 13:16] = -30000.0                                    # so the diagonal ray meets one
    dem = make_dem(elevation, (1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0))
    observer = check.observer_vector(dem, 20, 20)
    east_enu = np.array((1.0, 0.0, 0.0))
    direction = east_enu @ enu_to_moon_matrix(*vector_to_lat_lon(observer))
    samples = check.trace_ray(dem, observer, direction / np.linalg.norm(direction), np.arange(1.0, 200.0, 0.5))
    assert 0 < samples.shape[0] < 398                                  # left the DEM
    assert np.isnan(samples[:, 4]).any() and np.isfinite(samples[:, 4]).any()
    assert np.isnan(samples[:, 4][samples[:, 3] < -1000.0]).all()


def test_the_last_row_and_column_strip_is_not_sampled_as_zero_elevation() -> None:
    """DemGrid.elevation is 0 between the last node and the edge; a trace must stop at the last node instead of reading
    a wall there. On a floor 400 m below the sphere no ray to the edge has a positive slope."""
    dem = make_dem(np.full((60, 60), -400.0), (1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0))
    assert dem.elevation(59.5, 30.0) == 0.0                           # the strip the scalar function gets wrong
    horizon = check.reference_horizon([dem], 58, 58, settings=FAST_TRACE)       # one pixel from the corner
    assert horizon.max() < 0.0
    observer = check.observer_vector(dem, 30, 30)
    rotation = enu_to_moon_matrix(*vector_to_lat_lon(observer))
    for azimuth_index in (0, 360, 720, 1080):
        azimuth = np.deg2rad(azimuth_index * 0.25)
        direction = np.array((np.sin(azimuth), np.cos(azimuth), 0.0)) @ rotation
        samples = check.trace_ray(dem, observer, direction, np.arange(1.0, 200.0, 0.5), rotation)
        assert (samples[:, 1] < 59).all() and (samples[:, 2] < 59).all()


def test_the_distance_schedule_is_dense_near_coarse_far_in_bounded_chunks() -> None:
    settings = check.TraceSettings(near_range_m=100.0, near_step_fraction=0.25, far_step_fraction=0.5, chunk_samples=300,
                                   max_range_m=1000.0)
    chunks = list(check._distance_chunks(20.0, settings))
    distances = np.concatenate(chunks)
    assert max(len(c) for c in chunks) <= 300 and distances[0] == 1.0 and distances[-1] <= 1000.0
    assert np.all(np.diff(distances) > 0)
    near, far = np.diff(distances[distances < 100.0]), np.diff(distances[distances >= 100.0])
    assert np.allclose(near, 5.0) and np.allclose(far, 10.0) and distances[distances < 100.0][-1] + 5.0 == pytest.approx(100.0, abs=5.0)
    assert 1000.0 - distances[-1] < 10.0                              # runs out to the range


def test_trace_settings_must_be_positive() -> None:
    with pytest.raises(InputError, match="positive"):
        check.TraceSettings(far_step_fraction=0.0)


# --- the reference horizon -------------------------------------------------------------------------


def inner_dem(elevation=None) -> DemGrid:
    return make_dem(np.zeros((128, 128)) if elevation is None else elevation, (1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0))


def test_flat_terrain_has_a_horizon_at_the_curvature_dip_only() -> None:
    horizon = check.reference_horizon([inner_dem()], 64, 64, settings=FAST_TRACE)
    assert horizon.shape == (AZIMUTH_COUNT,) and horizon.dtype == np.float32
    assert horizon.max() <= 0.0 and horizon.min() > -0.001


def test_a_pillar_lowers_the_horizon_by_the_angle_to_the_ground_below() -> None:
    elevation = np.zeros((128, 128))
    elevation[64, 64] = 10.0
    dem = inner_dem(elevation)
    horizon = check.reference_horizon([dem], 64, 64, settings=FAST_TRACE)
    north = horizon[0]                                                # azimuth 0 runs along the diagonal, 64 * sqrt(2) m
    assert north == pytest.approx(-np.degrees(np.arctan(10.0 / (64.0 * np.sqrt(2.0)))), abs=0.35)
    assert horizon.max() < -0.3 and horizon.min() > -50.0


def test_a_later_dem_extends_the_terrain_with_a_distant_hill() -> None:
    outer_elevation = np.zeros((100, 100))
    outer_elevation[74:77, 74:77] = 300.0                             # a plateau whose near node is at x = 1490, y = -1490
    outer = make_dem(outer_elevation, (0.0, 20.0, 0.0, 0.0, 0.0, -20.0))
    inner = inner_dem()
    alone = check.reference_horizon([inner], 64, 64, azimuth_indices=[0, 720], settings=FAST_TRACE)
    both = check.reference_horizon([inner, outer], 64, 64, azimuth_indices=[0, 720], settings=FAST_TRACE)
    expected = np.degrees(np.arctan(300.0 / np.hypot(1490.0 - 1064.5, 1490.0 - 1064.5)))
    assert alone[0] < 0.01 and both[0] == pytest.approx(expected, abs=1.0)     # the samples are 5 m apart on a 28 m diagonal
    assert both[1] == pytest.approx(alone[1], abs=1e-6)               # the opposite way sees nothing new


def test_a_later_dem_is_used_only_beyond_where_the_ray_left_the_one_before() -> None:
    """The outer DEM stands 5 m above the inner one at the observer; sampled from 1 m it would put a 78 degree slope
    in every bin. Production starts it where the inner DEM's ray ends, so the horizon is the inner DEM's."""
    outer = make_dem(np.full((100, 100), 5.0), (0.0, 20.0, 0.0, 0.0, 0.0, -20.0))
    alone = check.reference_horizon([inner_dem()], 64, 64, azimuth_indices=[0, 400, 900], settings=FAST_TRACE)
    both = check.reference_horizon([inner_dem(), outer], 64, 64, azimuth_indices=[0, 400, 900], settings=FAST_TRACE)
    edge = 63.5 * np.sqrt(2.0)                                         # north: along the diagonal to the inner DEM's edge
    assert both[0] == pytest.approx(np.degrees(np.arctan(5.0 / edge)), abs=0.2)      # the plateau beyond it, 5 m up
    assert both.max() < 10.0 and (both > alone).all()


def test_the_ray_limit_of_a_dem_is_one_and_a_fifth_of_its_smaller_side() -> None:
    dem = make_dem(np.zeros((50, 80)), (0.0, 10.0, 0.0, 0.0, 0.0, -10.0))
    assert check.dem_ray_limit_m(dem, check.TraceSettings()) == pytest.approx(600.0)
    assert check.dem_ray_limit_m(dem, check.TraceSettings(max_range_m=400.0)) == 400.0


def test_the_horizon_is_clamped_to_the_stored_range_and_a_pixel_outside_is_refused() -> None:
    elevation = np.zeros((128, 128))
    elevation[64, 65] = 5000.0                                         # a tower one pixel away
    horizon = check.reference_horizon([inner_dem(elevation)], 64, 64, settings=FAST_TRACE)
    assert horizon.max() == np.float32(50.0)
    with pytest.raises(InputError, match="outside"):
        check.reference_horizon([inner_dem()], 200, 5)


# --- comparing lighting ----------------------------------------------------------------------------------


STORE = LightStoreSettings(codec="zlib-1")


def series(fraction, margin):
    return np.asarray(fraction, dtype=np.float32), np.asarray(margin, dtype=np.float32)


def test_plane_bits_use_the_store_thresholds() -> None:
    bits = check.plane_bits([0.19999, 0.2, 0.999999, 1.0, 0.9999989], [1.99, 2.0, 2.01, -90, 90], STORE)
    assert bits[0].tolist() == [False, True, True, True, True]               # shadow threshold 0.2, inclusive
    assert bits[1].tolist() == [False, False, True, True, False]             # operating 1 - 1e-6
    assert bits[2].tolist() == [False, True, True, False, True]              # Earth 2 degrees, inclusive


def test_a_bit_the_shifted_horizon_reproduces_is_explained_and_one_it_does_not_is_not() -> None:
    stamps = np.arange(4) * 7_200_000_000
    flat = series([1.0, 0.5, 0.1, 0.0], [10, 10, 10, 10])
    # step 1: the reference (0.5) is lit-partial, stored says not; the raised horizon (0.1) agrees with stored: explained.
    # step 2: stored says lit-partial while even the lowered horizon (0.15) is below 0.2: unexplained.
    stored = np.array([[True, False, True, False], [True, False, False, False], [True, True, True, True]])
    low = series([1.0, 0.9, 0.15, 0.0], [10, 10, 10, 10])
    high = series([1.0, 0.1, 0.0, 0.0], [10, 10, 10, 10])
    planes = check.compare_lighting(stored, flat, None, low, high, STORE, times_us=stamps)
    partial = planes[0]
    assert (partial.stored_vs_reference, partial.unexplained_vs_reference) == (2, 1)
    assert partial.worst[0]["time"] == "1970-01-01T04:00:00.000000Z" and partial.worst[0]["stored"] is True
    assert not partial.ok and planes[1].ok and planes[2].ok


def test_a_difference_from_the_production_tile_counts_unless_it_is_a_tie_at_the_threshold() -> None:
    stamps = np.arange(3) * 7_200_000_000
    reference = series([1.0, 0.2000005, 0.5], [10, 10, 10])
    production = series([1.0, 0.1999995, 0.1], [10, 10, 10])          # step 1 within 1e-3 of 0.2, step 2 is far
    stored = np.array([[True, True, True], [True, False, False], [True, True, True]])
    planes = check.compare_lighting(stored, reference, production, reference, reference, STORE, times_us=stamps)
    partial = planes[0]
    assert partial.stored_vs_production == 2 and partial.unexplained_vs_production == 1
    assert partial.production_vs_reference == 2


# --- judging a score -----------------------------------------------------------------------------------------


def outcome(score, start="2027-01-03T16:00:00.000000Z", end="2027-01-03T20:00:00.000000Z", kind="outage_in_shadow"):
    return check.Outcome(score_hours=score, outage_start_utc=start, outage_end_utc=end, kind=kind)


def test_a_score_is_judged_match_within_tolerance_or_differs() -> None:
    expected = outcome(40.0)
    judge = lambda ref, low, high, **kw: check.judge_outcome(expected, ref, low, high, 2.0, 72.0, haven=True, **kw)
    assert judge(outcome(40.0), outcome(36.0), outcome(44.0)) == "match"
    assert judge(outcome(42.0), outcome(38.0), outcome(46.0)) == "within_tolerance"
    assert judge(outcome(80.0), outcome(30.0), outcome(90.0)) == "within_tolerance"      # the lowered horizon gives 30
    assert judge(outcome(80.0), outcome(50.0), outcome(90.0)) == "differs"               # even that is above the table's 40
    assert judge(outcome(44.0), outcome(42.0), outcome(46.0)) == "within_tolerance"      # one 2-hour step below the envelope
    assert judge(outcome(46.0), outcome(44.0), outcome(48.0)) == "differs"               # two steps
    none = check.Outcome(reason="no_record")
    assert judge(none, none, none) == "differs"
    assert judge(none, none, outcome(40.0)) == "within_tolerance"                        # a shifted horizon reproduces it
    shifted = outcome(40.0, start="2027-01-09T16:00:00.000000Z")
    assert judge(shifted, shifted, shifted) == "differs"               # the same score at another time is not the outage


def test_a_non_haven_is_judged_by_whether_a_shifted_horizon_would_make_it_a_haven() -> None:
    judge = lambda ref, low, high: check.judge_outcome(check.Outcome(), ref, low, high, 2.0, 72.0, haven=False)
    assert judge(outcome(93.0), outcome(90.0), outcome(96.0)) == "match"
    assert judge(outcome(93.0), outcome(70.0), outcome(96.0)) == "within_tolerance"
    assert judge(outcome(60.0), outcome(50.0), outcome(80.0)) == "within_tolerance"
    assert judge(outcome(60.0), outcome(50.0), outcome(70.0)) == "differs"
    assert judge(check.Outcome(reason="no_record"), check.Outcome(reason="no_record"), check.Outcome(reason="no_record")) == "match"


# --- choosing pixels ----------------------------------------------------------------------------------------------


def table(count=400):
    rng = np.random.default_rng(1)
    rows = []
    for number in range(count):
        score = float(rng.choice((0.0, 6.0, 30.0, 70.0)))
        rows.append({"seed_id": number + 1, "month": f"{2024 + number % 8}-{number % 12 + 1:02d}", "row": number, "column": 7,
                     "hibernation_hours": score, "outage_kind": ("earth_never_sets" if number % 9 == 0 else "outage_in_shadow"),
                     "outage_start_utc": "", "outage_end_utc": "", "chain_length": int(rng.choice((1, 3, 12)))})
    return rows


def test_the_sample_is_spread_over_every_stratum_and_repeatable() -> None:
    rows = table()
    sample = check.stratified_sample(rows, 24, random_seed=5)
    assert len(sample) == 24 and len({r["seed_id"] for r in sample}) == 24
    strata = [check.strata_of(r) for r in sample]
    assert {s["kind"] for s in strata} == {"earth_never_sets", "outage_in_shadow"}
    assert {s["score"] for s in strata} == {"0", "1-24", ">24"} and {s["chain"] for s in strata} == {"1", "2-5", "6+"}
    assert len({s["year"] for s in strata}) == 8
    assert [r["seed_id"] for r in check.stratified_sample(rows, 24, random_seed=5)] == [r["seed_id"] for r in sample]
    assert [r["seed_id"] for r in check.stratified_sample(rows, 24, random_seed=6)] != [r["seed_id"] for r in sample]
    assert len(check.stratified_sample(rows[:5], 40)) == 5 and check.stratified_sample([], 3) == []


def test_the_region_is_the_upper_right_quadrant_near_the_top_or_right_edge() -> None:
    inside = check.region_filter("upper_right_edges", 1000, 800, 100)
    assert inside(0, 799) and inside(99, 400) and inside(499, 700) and inside(300, 799) and inside(0, 400)
    assert not inside(100, 400)                    # the top band is 100 rows (0 to 99)
    assert not inside(300, 699)                    # the right band is 100 columns (700 to 799)
    assert not inside(500, 799)                    # the lower half of the rows is out
    assert not inside(50, 399)                     # the left half of the columns is out
    assert all(check.region_filter("all", 10, 10)(r, c) for r in (0, 9) for c in (0, 9))
    for bad in (lambda: check.region_filter("middle", 10, 10), lambda: check.region_filter("all", 10, 10, 0)):
        with pytest.raises(InputError, match="region|band"):
            bad()


def test_the_chain_table_is_read_with_the_columns_the_check_needs(tmp_path) -> None:
    path = tmp_path / "chains.csv"
    header = list(check.CHAIN_COLUMNS) + ["extra"]
    path.write_text(",".join(header) + "\n1,2027-01,5,6,12,outage_in_shadow,2027-01-03T16:00:00.000000Z,2027-01-03T20:00:00.000000Z,3,x\n")
    rows = check.read_chain_table(path)
    assert rows == [{"seed_id": 1, "month": "2027-01", "row": 5, "column": 6, "hibernation_hours": 12.0,
                     "outage_kind": "outage_in_shadow", "outage_start_utc": "2027-01-03T16:00:00.000000Z",
                     "outage_end_utc": "2027-01-03T20:00:00.000000Z", "chain_length": 3}]
    path.write_text("seed_id,month\n1,2027-01\n")
    with pytest.raises(InputError, match="lacks the columns"):
        check.read_chain_table(path)


# --- a whole check on a synthetic scene ---------------------------------------------------------------------------------


FIRST, LAST = "2027-01-01T00:00:00Z", "2027-03-31T22:00:00Z"
STORE_START, STORE_STOP = "2026-12-01T00:00:00", "2027-05-01T00:00:00"
PILLARS = ((30, 25), (40, 70), (70, 40), (90, 90), (50, 105), (100, 30), (20, 100))   # no two on one north-south diagonal
PILLAR_HEIGHT_M = 10.0


def scene_parameters() -> WindowParameters:
    return WindowParameters(cell_patches=1, haven_neighbourhood_m=3.0, seed_separation_m=4.0, operational_hours=24.0,
                            forward_only=True)


def build_scene(directory, *, corrupted=None, corruption_deg=4.0):
    """A flat 128 x 128 DEM with a pillar at each of PILLARS. The horizon tile holds the traced horizon at the
    pillars (raised by 4 degrees at ``corrupted``) and 0 degrees elsewhere, and a light store is made from it."""
    elevation = np.zeros((128, 128), dtype=np.float32)
    for row, column in PILLARS:
        elevation[row, column] = PILLAR_HEIGHT_M
    dem = inner_dem(elevation)
    columns = np.zeros((128 * 128, AZIMUTH_COUNT), dtype=np.float32)
    traced = {}
    for row, column in PILLARS:
        traced[(row, column)] = check.reference_horizon([dem], row, column, settings=FAST_TRACE)
        columns[row * 128 + column] = traced[(row, column)] + (corruption_deg if (row, column) == corrupted else 0.0)
    horizon_directory = directory / "horizons"
    HorizonTileStore(horizon_directory).write(0, 0, 0.0, columns, compress=True, valid_width=128, valid_height=128)
    store_directory = directory / "store"
    store_directory.mkdir()
    stage0.generate_light_store(
        store_directory, dem=dem, horizon_store=HorizonTileStore(horizon_directory),
        settings=Stage0Settings(store=LightStoreSettings(steps_per_chunk=48, codec="zlib-1"), start_utc=STORE_START,
                                stop_utc=STORE_STOP, lighting="cpu", workers=1),
        sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors)
    stamps = ls.times(FIRST, LAST, step_hours=2.0).values
    return SimpleNamespace(dem=dem, horizons=HorizonTileStore(horizon_directory), stamps=stamps,
                           light_store=LightStoreDirectory(store_directory, dem), directory=directory)


def table_from_store(world) -> list[dict]:
    """What stage 2 would have put in its table: the stored evaluator's havens among the pillars."""
    time = TimeContext.build(world.stamps, None, scene.earth_vectors)
    signals = StoreSignals(world.light_store, world.dem)
    evaluator = CandidateEvaluator(signals, time, scene_parameters())
    months = month_labels(time)
    rows = []
    for row, column in PILLARS:
        for month in range(3):
            found = check.evaluate_outcome(evaluator, row, column, month)
            if found.is_haven(72.0):
                rows.append({"seed_id": len(rows) + 1, "month": months[month], "row": row, "column": column,
                             "hibernation_hours": found.score_hours, "outage_kind": found.kind,
                             "outage_start_utc": found.outage_start_utc, "outage_end_utc": found.outage_end_utc,
                             "chain_length": 1})
    signals.close()
    return rows


@pytest.fixture(scope="module")
def good_world(tmp_path_factory):
    world = build_scene(tmp_path_factory.mktemp("check_good"))
    world.rows = table_from_store(world)
    return world


def run(world, directory, **options):
    arguments = dict(dems=[world.dem], light_store=world.light_store, horizon_store=world.horizons, chain_rows=world.rows,
                     times=world.stamps, sun_vectors_at=scene.sun_vectors, earth_vectors_at=scene.earth_vectors,
                     parameters=scene_parameters(), seeds=len(world.rows), non_havens=3,
                     settings=check.CheckSettings(window_days=10.0, trace=FAST_TRACE))
    arguments.update(options)
    return check.run_check(directory, **arguments)


def test_the_scene_has_havens_to_check(good_world) -> None:
    assert len(good_world.rows) >= 8
    assert {r["row"] for r in good_world.rows} >= {30, 90}
    assert {r["outage_kind"] for r in good_world.rows} <= {"outage_in_shadow", "outage_sun_lit", "earth_never_sets"}


def test_a_store_made_from_the_horizons_the_reference_reproduces_passes(good_world, tmp_path) -> None:
    report = run(good_world, tmp_path / "out")
    assert report.passed and report.seeds_checked == len(good_world.rows) and report.non_havens_checked == 3
    assert report.lighting_failed == 0 and report.stored_not_reproducing_table == 0
    assert set(report.verdicts) <= {"haven:match", "haven:within_tolerance", "non_haven:match"}
    assert report.verdicts.get("haven:match", 0) >= len(good_world.rows) - 2
    for totals in report.plane_totals.values():
        assert totals["unexplained_vs_reference"] == 0 and totals["unexplained_vs_production"] == 0
        assert totals["steps"] > 0
    assert report.plane_totals["lit_partial"]["stored_set"] > 0
    assert report.horizon["pixels"] >= 1 and report.horizon["relevant_max_abs_deg"] < 0.1


def test_the_report_files_are_written_and_not_replaced(good_world, tmp_path) -> None:
    out = tmp_path / "out"
    run(good_world, out)
    document = json.loads((out / "lighting_check.json").read_text())
    assert document["format"] == check.REPORT_FORMAT and document["passed"] is True
    assert len(document["pixels"]) == len(good_world.rows) + 3 and all(p["ok"] for p in document["pixels"])
    assert document["pixels"][0]["planes"][0]["plane"] == "lit_partial"
    with (out / "lighting_check.csv").open(newline="") as handle:
        lines = list(csv.DictReader(handle))
    assert len(lines) == len(good_world.rows) + 3 and tuple(lines[0]) == check.CSV_COLUMNS
    assert {line["kind"] for line in lines} == {"haven", "non_haven"}
    assert "Result: **PASS**" in (out / "lighting_check.md").read_text()
    before = (out / "lighting_check.json").read_bytes()
    with pytest.raises(InputError, match="exists"):
        run(good_world, out)
    assert (out / "lighting_check.json").read_bytes() == before
    run(good_world, out, overwrite=True)


def test_a_sample_smaller_than_the_table_is_spread_not_taken_in_order(good_world, tmp_path) -> None:
    report = run(good_world, tmp_path / "out", seeds=4, non_havens=0)
    assert report.seeds_checked == 4 and report.non_havens_checked == 0 and len(report.pixels) == 4
    with pytest.raises(InputError, match="positive"):
        run(good_world, tmp_path / "bad", seeds=0)


def test_a_region_limits_the_seeds_and_the_non_havens(good_world, tmp_path) -> None:
    """On the 128 x 128 scene the upper-right quadrant is rows 0-63, columns 64-127; with a 40 pixel band it holds
    the pillars (20, 100) and (50, 105) but not (40, 70), whose row and column are both outside the strips."""
    report = run(good_world, tmp_path / "out", region="upper_right_edges", edge_band_pixels=40, non_havens=2,
                 seeds=100)
    wanted = {r["seed_id"] for r in good_world.rows if (r["row"], r["column"]) in {(20, 100), (50, 105)}}
    assert wanted and report.seeds_checked == len(wanted)
    haven_pixels = {(p["row"], p["column"]) for p in report.pixels if p["kind"] == "haven"}
    assert haven_pixels == {(20, 100), (50, 105)}
    inside = check.region_filter("upper_right_edges", 128, 128, 40)
    assert all(inside(p["row"], p["column"]) for p in report.pixels)               # the non-havens too
    assert report.non_havens_checked == 2
    document = json.loads((tmp_path / "out" / "lighting_check.json").read_text())
    assert document["settings"]["region"] == "upper_right_edges" and document["settings"]["seeds_in_region"] == len(wanted)
    with pytest.raises(InputError, match="no seed"):
        run(good_world, tmp_path / "none", region="upper_right_edges", edge_band_pixels=5)


def test_a_horizon_tile_that_is_wrong_is_reported_as_a_failure(good_world, tmp_path_factory) -> None:
    """The tile at one pillar is 4 degrees too high (its horizon is about -4 degrees, so Earth's margin falls below the
    2 degree threshold): the store is not what the reference gives, and the table's seed is not a haven of it."""
    corrupted = (30, 25)
    bad = build_scene(tmp_path_factory.mktemp("check_bad"), corrupted=corrupted)
    bad.rows = good_world.rows                                          # the table made from the good store
    report = run(bad, bad.directory / "out", non_havens=0)
    assert not report.passed and report.lighting_failed >= 1
    failed = [p for p in report.pixels if not p["ok"]]
    assert {(p["row"], p["column"]) for p in failed} == {corrupted}
    assert all(p["kind"] == "haven" for p in failed)
    assert sum(plane["unexplained_vs_reference"] for p in failed for plane in p["planes"]) > 0
    assert "FAIL" in (bad.directory / "out" / "lighting_check.md").read_text()
    # the pillars that are right still pass
    assert sum(p["ok"] for p in report.pixels) == len(good_world.rows) - len([p for p in report.pixels if (p["row"], p["column"]) == corrupted])


def test_a_small_horizon_error_is_reported_with_the_shift_that_explains_it(good_world, tmp_path_factory) -> None:
    """A tile 0.8 degrees too high at one pillar: beyond the 0.3 degree tolerance, so the lighting is not reproduced,
    but a horizon shifted by 1 degree explains every stored bit, and the report says so."""
    corrupted = (30, 25)
    bad = build_scene(tmp_path_factory.mktemp("check_small"), corrupted=corrupted, corruption_deg=0.8)
    bad.rows = good_world.rows
    report = run(bad, bad.directory / "out", non_havens=0)
    failed = [p for p in report.pixels if not p["ok"]]
    assert failed and {(p["row"], p["column"]) for p in failed} == {corrupted}
    assert {p["explaining_shift_deg"] for p in failed} == {1.0}
    assert report.explained_by_shift == {"<= 1 deg": len(failed)}
    assert "<= 1 deg" in (bad.directory / "out" / "lighting_check.md").read_text()
    tight = run(bad, bad.directory / "out_tight", non_havens=0,
                settings=check.CheckSettings(window_days=10.0, trace=FAST_TRACE, horizon_tolerance_deg=1.2))
    assert tight.lighting_failed == 0                                  # at 1.2 degrees the same tile is within tolerance


def test_a_stored_lighting_that_does_not_reproduce_the_table_is_reported(good_world, tmp_path) -> None:
    rows = [dict(r) for r in good_world.rows]
    rows[0]["hibernation_hours"] += 2.0                                  # the table says something the store does not
    report = run(good_world, tmp_path / "out", chain_rows=rows)
    assert report.stored_not_reproducing_table >= 1 and not report.passed
