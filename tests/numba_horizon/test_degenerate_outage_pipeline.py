"""The degenerate-outage rule through the public function, and through pass 1.

The Earth-side corner of the synthetic scene (``windows_scene.EARTHSIDE``) never has a
real outage, so without the rule its hibernation is NODATA in every month; with it,
each Earth cycle's elevation minimum is a one-sample outage in the Sun, which is up.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio

import lunarscout as ls
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_windows as windows
from lunarscout._numba_horizon.file_format import HorizonTileStore
from lunarscout._numba_horizon.safe_haven_windows import (
    Stage1Rasters,
    WindowParameters,
    degenerate_outage_flags,
)
from lunarscout.products import _load_dem

PIXEL = (16, 2)         # inside the Earth-side corner
PARAMETERS = WindowParameters(
    cell_patches=1, rover_speed_m_s=1.0 / 1800.0, haven_neighbourhood_m=3.0,
    seed_separation_m=4.0,
)


def hibernation_of(products):
    with rasterio.open(products.hibernation) as dataset:
        return dataset.read()


@pytest.fixture(scope="module")
def with_rule(tmp_path_factory):
    directory = tmp_path_factory.mktemp("with_rule")
    dem_path, horizons_path, products, stamps = scene.stage_one(directory, earthside=True)
    return SimpleNamespace(
        dem_path=dem_path, horizons_path=horizons_path, products=products, stamps=stamps,
        hibernation=hibernation_of(products), directory=directory,
    )


def test_the_default_window_gives_each_month_a_zero_hour_hibernation(with_rule) -> None:
    corner = with_rule.hibernation[:, PIXEL[0], PIXEL[1]]
    assert corner.tolist() == [0.0, 0.0, 0.0, 0.0]     # one Earth cycle's minimum per month
    for row in range(14, 20):
        for column in range(0, 6):
            assert with_rule.hibernation[:, row, column].tolist() == [0.0] * 4
    # pixels with real outages are untouched by the rule
    assert with_rule.hibernation[:2, 8, 8].tolist() == [42.0, 42.0]


def test_a_zero_window_leaves_the_corner_without_a_hibernation(tmp_path) -> None:
    _dem, _horizons, products, _stamps = scene.stage_one(
        tmp_path, earthside=True, degenerate_outage_window_hours=0.0
    )
    corner = hibernation_of(products)[:, 14:20, 0:6]
    assert np.isnan(corner).all()


def test_the_outage_sample_is_unavailable_for_the_joint_layers(with_rule, tmp_path) -> None:
    _dem, _horizons, without, _stamps = scene.stage_one(
        tmp_path, earthside=True, degenerate_outage_window_hours=0.0
    )
    with rasterio.open(with_rule.products.joint_total) as a, rasterio.open(without.joint_total) as b:
        with_rule_total, without_total = a.read(), b.read()
    row, column = PIXEL
    # One 6 hour sample per month is Earth-unavailable under the rule; the Sun is up
    # there, so each month loses exactly 6 hours of joint operating time.
    assert (without_total[:, row, column] - with_rule_total[:, row, column]).tolist() == [6.0] * 4


def test_a_window_that_is_not_a_whole_number_of_steps_is_rounded_and_bad_ones_rejected(tmp_path) -> None:
    dem_path, horizons_path = scene.write_inputs(tmp_path)
    times = ls.times(scene.START, scene.STOP, step_hours=scene.STEP_HOURS)
    common = dict(
        times=times, sun_vectors_m=scene.sun_vectors(times.values),
        earth_vectors_m=scene.earth_vectors(times.values), backend="cpu",
    )
    for bad, code in ((-1.0, "safe_haven_degenerate_window_invalid"),
                      (float("nan"), "safe_haven_degenerate_window_invalid"),
                      (2.0, "safe_haven_degenerate_window_invalid"),      # under half a step
                      (100.0, "safe_haven_degenerate_window_invalid")):   # over 15 steps
        with pytest.raises(ls.InputError) as error:
            ls.generate_safe_haven_products(
                dem_path, horizons_path, tmp_path / "bad", degenerate_outage_window_hours=bad, **common
            )
        assert error.value.code == code


def test_resuming_with_a_different_window_is_refused(tmp_path) -> None:
    dem_path, horizons_path = scene.write_inputs(tmp_path)
    times = ls.times(scene.START, scene.STOP, step_hours=scene.STEP_HOURS)
    common = dict(
        times=times, sun_vectors_m=scene.sun_vectors(times.values),
        earth_vectors_m=scene.earth_vectors(times.values), backend="cpu",
    )
    cancel_after = {"calls": 0}

    def cancel():
        cancel_after["calls"] += 1
        return cancel_after["calls"] > 5

    with pytest.raises(ls.OperationCancelledError):
        ls.generate_safe_haven_products(
            dem_path, horizons_path, tmp_path / "out", cancellation_requested=cancel, **common
        )
    with pytest.raises(ls.ProductStorageError):
        ls.generate_safe_haven_products(
            dem_path, horizons_path, tmp_path / "out", degenerate_outage_window_hours=12.0, **common
        )


# --- pass 1 uses the same rule --------------------------------------------------------------


def options(with_rule, **overrides):
    dem, _georef = _load_dem(with_rule.dem_path)
    values = dict(
        dem=dem, horizon_store=HorizonTileStore(with_rule.horizons_path),
        stage1=Stage1Rasters(with_rule.products.hibernation, with_rule.products.joint_total),
        times=with_rule.stamps, sun_vectors_at=scene.sun_vectors,
        earth_vectors_at=scene.earth_vectors, parameters=PARAMETERS,
    )
    values.update(overrides)
    return values


def test_pass_one_reproduces_the_degenerate_hibernation_of_the_corner(with_rule) -> None:
    opts = options(with_rule)
    time = windows.TimeContext.build(with_rule.stamps, scene.sun_vectors, scene.earth_vectors)
    evaluator = windows.CandidateEvaluator(
        windows.PixelSignals(opts["dem"], opts["horizon_store"]), time, PARAMETERS
    )
    for month in range(4):
        record = evaluator.evaluate(*PIXEL, month)
        assert (record.outage_first, record.outage_last) == (record.outage_first,) * 2
        assert record.score_hours == 0.0 and record.run_first == -1
        # the outage is one sample: T_e is the next sample, 6 hours after T_s
        assert record.outage_end_us - record.outage_start_us == 6 * 3_600_000_000
        # the sample is at Earth's lowest elevation within the cycle
        elevation = scene.earth_elevation(with_rule.stamps)
        assert elevation[record.outage_first] < elevation[record.outage_first - 1]
        assert elevation[record.outage_first] <= elevation[record.outage_first + 1]
        assert (record.earth_visible_before_hours, record.earth_visible_after_hours) == (48.0, 48.0)


def test_pass_one_selects_seeds_in_the_corner_and_checks_them_against_stage_1(with_rule) -> None:
    # A 30 hour limit leaves the cluster (42 hours in January and February) out, so the
    # corner, whose hibernations score 0, supplies every seed of those months.  Pass 1
    # raises PlanConsistencyError if its recomputed score ever differed from stage 1.
    parameters = WindowParameters(
        cell_patches=1, rover_speed_m_s=1.0 / 1800.0, haven_neighbourhood_m=3.0,
        seed_separation_m=4.0, survival_limit_hours=30.0,
    )
    records = list(windows.iter_cell_months(**options(with_rule, parameters=parameters)))
    by_month = {record["month"]: record["seeds"] for record in records}
    for month in (0, 1):
        seeds = by_month[month]
        assert seeds and all(
            14 <= seed["row"] < 20 and seed["column"] < 6 and seed["hibernation_hours"] == 0.0
            for seed in seeds
        )


def test_pass_one_without_the_rule_disagrees_with_stage_1_where_the_rule_matters(with_rule) -> None:
    # Stage 1 was run with the rule; a pass 1 that leaves it off recomputes NODATA there.
    parameters = WindowParameters(
        cell_patches=1, rover_speed_m_s=1.0 / 1800.0, haven_neighbourhood_m=3.0,
        seed_separation_m=4.0, degenerate_outage_window_hours=0.0,
    )
    time = windows.TimeContext.build(with_rule.stamps, scene.sun_vectors, scene.earth_vectors)
    opts = options(with_rule)
    evaluator = windows.CandidateEvaluator(
        windows.PixelSignals(opts["dem"], opts["horizon_store"]), time, parameters
    )
    assert evaluator.evaluate(*PIXEL, 0) == "no_record"


# --- the flags themselves ----------------------------------------------------------------------


def test_the_numpy_flags_match_the_rule_as_stated() -> None:
    from test_safe_haven_degenerate_outage import degenerate_outages

    rng = np.random.default_rng(4)
    for _ in range(300):
        count = int(rng.integers(1, 80))
        flat = np.round(rng.normal(5, 3, count) * 2) / 2          # many ties
        flat = flat.astype(np.float32)
        margin = (flat - rng.uniform(-1, 6)).astype(np.float32)
        window = int(rng.integers(1, 8))
        outage = margin < 2.0
        np.testing.assert_array_equal(
            degenerate_outage_flags(flat, outage, window),
            degenerate_outages(flat, margin, window),
        )
    assert not degenerate_outage_flags(np.zeros(5), np.zeros(5, bool), 0).any()


@pytest.mark.parametrize(
    ("hours", "step_hours", "expected"),
    [(None, 6.0, 4), (None, 1.0, 15), (None, 48.0, 1), (12.0, 6.0, 2), (0.0, 6.0, 0), (30.0, 6.0, 5)],
)
def test_the_window_in_hours_becomes_a_whole_number_of_steps(
    tmp_path, monkeypatch, hours, step_hours, expected
) -> None:
    from lunarscout._numba_horizon import safe_haven_pipeline

    seen = {}

    def capture(**arguments):
        seen.update(arguments)
        return tuple(tmp_path / name for name in ("a.tif", "b.tif", "c.tif"))

    monkeypatch.setattr(safe_haven_pipeline, "run_safe_haven_products", capture)
    dem_path, horizons_path = scene.write_inputs(tmp_path)
    times = ls.times(scene.START, "2027-02-10T00:00:00Z", step_hours=step_hours)
    ls.generate_safe_haven_products(
        dem_path, horizons_path, tmp_path / "out", times=times,
        sun_vectors_m=scene.sun_vectors(times.values), earth_vectors_m=scene.earth_vectors(times.values),
        backend="cpu", degenerate_outage_window_hours=hours,
    )
    assert seen["degenerate_window_samples"] == expected


# --- the whole CPU pipeline against the offline rule, with terrain in the margin -------------


def ridge_signals(directory, stamps):
    """Per-pixel Sun fraction, flat elevation and margin from the kernel, for the ridge scene."""
    dem, _georef = _load_dem(directory / "dem.tif")
    signals = windows.PixelSignals(dem, HorizonTileStore(directory / "horizons"))

    def series(row, column):
        sun = signals.series(row, column, scene.sun_vectors(stamps), margin=False)
        earth = scene.earth_vectors_ridge(stamps)
        margin = signals.series(row, column, earth, margin=True)
        flat = signals.series(row, column, earth, margin=True, flat_elevation=True)
        return sun, flat, margin

    return series


def test_the_cpu_pipeline_matches_the_offline_rule_when_the_ridge_moves_the_margin_minimum(tmp_path) -> None:
    from test_safe_haven_degenerate_outage import degenerate_outages, reference

    _dem, _horizons, products, stamps = scene.stage_one(tmp_path, ridge=True)
    series = ridge_signals(tmp_path, stamps)
    time = windows.TimeContext.build(stamps, scene.sun_vectors, scene.earth_vectors_ridge)
    layers = []
    for path in products:
        with rasterio.open(path) as dataset:
            layers.append(dataset.read())
    for row, column in ((16, 2), (19, 5), (8, 8), (0, 0), (17, 17)):
        sun, flat, margin = series(row, column)
        expected = reference(sun, flat, margin, time.sample_month, window=4, band_count=4)
        for layer, want in zip(layers, expected):
            np.testing.assert_array_equal(layer[:, row, column], want.astype(np.float32))
    # the corner has no real outage, and Earth's flat minimum is where the rule puts it:
    sun, flat, margin = series(16, 2)
    assert not (margin < 2.0).any()
    forced = np.flatnonzero(degenerate_outages(flat, margin, 4))
    assert len(forced) == 4 and (np.diff(forced) > 100).all()      # one per 27.3 day cycle (109 samples)
    # ... which is away from the margin's own minima (the ridge crossings)
    cycle = slice(forced[0] - 40, forced[0] + 40)
    assert abs(int(np.argmin(margin[cycle])) - 40) > 8


def test_pass_one_finds_the_degenerate_outage_at_the_flat_minimum_not_the_margin_minimum(tmp_path) -> None:
    _dem, _horizons, products, stamps = scene.stage_one(tmp_path, ridge=True)
    dem, _georef = _load_dem(tmp_path / "dem.tif")
    time = windows.TimeContext.build(stamps, scene.sun_vectors, scene.earth_vectors_ridge)
    evaluator = windows.CandidateEvaluator(
        windows.PixelSignals(dem, HorizonTileStore(tmp_path / "horizons")), time, PARAMETERS
    )
    elevation = scene.earth_elevation(stamps)
    for month in range(4):
        record = evaluator.evaluate(16, 2, month)
        assert record.outage_first == record.outage_last
        k = record.outage_first
        assert elevation[k] < elevation[k - 1] and elevation[k] <= elevation[k + 1]
        assert elevation[k] < 0.0                                   # the deep part of the cycle


@pytest.mark.skipif(
    __import__("os").environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the real-device comparison",
)
def test_the_cuda_backend_gives_the_cpu_products_with_the_rule_on_the_ridge_scene(tmp_path) -> None:
    (tmp_path / "cpu").mkdir()
    (tmp_path / "gpu").mkdir()
    cpu = scene.stage_one(tmp_path / "cpu", ridge=True)[2]
    gpu = scene.stage_one(tmp_path / "gpu", ridge=True, backend="cuda")[2]
    for cpu_path, gpu_path in zip(cpu, gpu):
        with rasterio.open(cpu_path) as expected, rasterio.open(gpu_path) as actual:
            np.testing.assert_array_equal(actual.read(), expected.read())
            assert actual.tags()["LUNARSCOUT_COMPUTE_BACKENDS"] == '["cuda"]'
    with rasterio.open(gpu.hibernation) as dataset:
        corner = dataset.read()[:, 14:20, 0:6]
    assert (corner == 0.0).all()        # the degenerate outages, on the device
