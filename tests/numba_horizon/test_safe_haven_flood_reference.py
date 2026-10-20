"""The stage-2 flood reference (spec section 6): hand-worked cases, an independent
path-existence oracle on random inputs, and the real lit state of the synthetic scene."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_flood as flood
from lunarscout._numba_horizon import safe_haven_windows as windows
from lunarscout._numba_horizon.file_format import HorizonTileStore
from lunarscout._numba_horizon.safe_haven_windows import FloodGrid, Stage1Rasters, WindowParameters
from lunarscout.products import _load_dem


def always_lit(shape):
    lit = np.ones(shape, dtype=bool)
    return lambda step: lit


# --- hand-worked cases ----------------------------------------------------------------


@pytest.mark.parametrize("connectivity, expected", [(8, 7 * 7), (4, 25)])
def test_open_ground_gives_a_square_or_a_diamond(connectivity, expected) -> None:
    passable = np.ones((11, 11), dtype=bool)
    for direction in (flood.BEFORE, flood.AFTER):
        reached = flood.flood_side_reference(
            always_lit(passable.shape), passable, (5, 5), 100,
            direction=direction, steps=3, connectivity=connectivity,
        )
        assert len(reached) == expected
    # Chebyshev radius 3 (8-neighbour) or Manhattan radius 3 (4-neighbour) around the seed.
    metric = (lambda dr, dc: max(abs(dr), abs(dc))) if connectivity == 8 else (lambda dr, dc: abs(dr) + abs(dc))
    assert reached == {(r, c) for r in range(11) for c in range(11) if metric(r - 5, c - 5) <= 3}


def test_zero_steps_is_the_seed_alone_and_the_seed_is_not_tested_at_the_boundary() -> None:
    passable = np.ones((3, 3), dtype=bool)
    dark = lambda step: np.zeros((3, 3), dtype=bool)
    assert flood.flood_side_reference(dark, passable, (1, 1), 0, direction=flood.AFTER, steps=0) == {(1, 1)}
    # Lit only after the boundary: the seed at the boundary needs no light of its own.
    assert flood.flood_side_reference(dark, passable, (1, 1), 0, direction=flood.AFTER, steps=1) == set()


def test_time_runs_backward_on_the_before_side() -> None:
    """Column j is lit at step ``t0 - j``: a runner moving one pixel per step backward in
    time gets through on the before side and never on the after side."""
    passable = np.ones((1, 6), dtype=bool)
    t0 = 50

    def runner_backward(step):
        lit = np.zeros((1, 6), dtype=bool)
        j = t0 - step
        if 0 <= j < 6:
            lit[0, j] = True
        return lit

    before = flood.flood_side_reference(runner_backward, passable, (0, 0), t0, direction=flood.BEFORE, steps=5)
    assert before == {(0, 5)}                       # the runner moves one pixel per step
    after = flood.flood_side_reference(runner_backward, passable, (0, 0), t0, direction=flood.AFTER, steps=5)
    assert after == set()                           # step t0 + j never lights column j


def test_the_before_and_after_sides_use_their_own_boundaries() -> None:
    passable = np.ones((1, 4), dtype=bool)
    lit_at = lambda step: np.array([[step % 2 == 0] * 4])
    result = flood.flood_seed_reference(lit_at, passable, (0, 0), 11, 20, steps=2)
    # before: step 10 is lit but step 9 is dark; after: step 21 is dark. Both end empty.
    assert (result.count_before, result.count_after) == (0, 0)
    result = flood.flood_seed_reference(lambda step: np.ones((1, 4), dtype=bool), passable, (0, 0), 11, 20, steps=2)
    assert (result.count_before, result.count_after) == (3, 3)


def test_an_emptied_frontier_stays_empty() -> None:
    passable = np.ones((1, 5), dtype=bool)

    def lit_at(step):
        # dark at the first step, lit ever after
        return np.full((1, 5), step != 99)

    assert flood.flood_side_reference(lit_at, passable, (0, 2), 100, direction=flood.BEFORE, steps=4) == set()


def test_a_diagonal_gap_passes_under_8_neighbours_only() -> None:
    passable = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=bool)
    lit_at = always_lit(passable.shape)
    eight = flood.flood_side_reference(lit_at, passable, (0, 0), 0, direction=flood.AFTER, steps=2, connectivity=8)
    four = flood.flood_side_reference(lit_at, passable, (0, 0), 0, direction=flood.AFTER, steps=2, connectivity=4)
    assert eight == {(0, 0), (1, 1), (2, 2)}
    assert four == {(0, 0)}


def test_impassable_pixels_and_the_window_edge_clip_the_flood() -> None:
    passable = np.ones((5, 5), dtype=bool)
    passable[:, 2] = False                           # a wall across the window
    reached = flood.flood_side_reference(
        always_lit(passable.shape), passable, (2, 0), 0, direction=flood.AFTER, steps=10,
    )
    assert reached == {(r, c) for r in range(5) for c in (0, 1)}


def test_the_score_needs_both_sides() -> None:
    assert flood.FloodResult(4, 5).score == 9 and flood.FloodResult(4, 5).valid
    assert not flood.FloodResult(4, 0).valid and not flood.FloodResult(0, 4).valid


def test_the_lit_rule_uses_the_stage_1_fuzz() -> None:
    values = [1.0, 1.0 - 5e-7, 1.0 - 2e-6, 0.99]
    assert flood.lit_threshold_test(values, 1.0).tolist() == [True, True, False, False]
    assert flood.lit_threshold_test(values, 0.99).tolist() == [True, True, True, True]


@pytest.mark.parametrize("call", [
    lambda: flood.neighbour_offsets(6),
    lambda: flood.flood_side_reference(always_lit((2, 2)), np.ones((2, 2), bool), (0, 0), 0, direction=0, steps=1),
    lambda: flood.flood_side_reference(always_lit((2, 2)), np.ones((2, 2), bool), (0, 0), 0, direction=1, steps=-1),
    lambda: flood.flood_side_reference(always_lit((3, 3)), np.ones((2, 2), bool), (0, 0), 0, direction=1, steps=1),
])
def test_invalid_arguments_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


# --- an independent oracle: does a time-ordered path exist? ---------------------------------


def path_count(lit_at, passable, seed, t0, direction, steps, connectivity):
    """|R_n| by asking, for each pixel, whether a path of length n from the seed exists."""
    moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        moves += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    rows, columns = passable.shape
    lit = {}

    def lit_state(j):
        if j not in lit:
            lit[j] = lit_at(t0 + direction * j)
        return lit[j]

    @lru_cache(maxsize=None)
    def reachable(j, r, c):
        if not (passable[r, c] and (j == 0 or lit_state(j)[r, c])):
            return j == 0 and (r, c) == seed
        if j == 0:
            return (r, c) == seed
        return any(
            0 <= r - dr < rows and 0 <= c - dc < columns and reachable(j - 1, r - dr, c - dc)
            for dr, dc in moves
        )

    return sum(reachable(steps, r, c) for r in range(rows) for c in range(columns))


@pytest.mark.parametrize("connectivity", [4, 8])
def test_the_set_flood_agrees_with_path_existence_on_random_inputs(connectivity) -> None:
    rng = np.random.default_rng(20271231 + connectivity)
    for _ in range(40):
        shape = (int(rng.integers(3, 8)), int(rng.integers(3, 8)))
        passable = rng.random(shape) < 0.85
        seed = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
        passable[seed] = True
        steps = int(rng.integers(0, 6))
        density = float(rng.uniform(0.3, 1.0))
        cube = rng.random((200, *shape)) < density
        t0 = 100
        for direction in (flood.BEFORE, flood.AFTER):
            reached = flood.flood_side_reference(
                lambda step: cube[step], passable, seed, t0,
                direction=direction, steps=steps, connectivity=connectivity,
            )
            expected = path_count(lambda step: cube[step], passable, seed, t0, direction, steps, connectivity)
            assert len(reached) == expected


# --- the synthetic scene --------------------------------------------------------------------


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flood_scene")
    dem_path, horizons_path, products, stamps = scene.stage_one(directory)
    dem, _georef = _load_dem(dem_path)
    return dict(
        dem=dem, store=HorizonTileStore(horizons_path), stamps=stamps,
        stage1=Stage1Rasters(products.hibernation, products.joint_total),
    )


@pytest.mark.parametrize("flood_steps", [None, 5, 12])
def test_the_flood_on_the_scene_lit_state_matches_path_existence(prepared, flood_steps) -> None:
    parameters = WindowParameters(
        cell_patches=1, rover_speed_m_s=1.0 / 1800.0, haven_neighbourhood_m=3.0, seed_separation_m=4.0,
    )
    records = list(windows.iter_cell_months(
        dem=prepared["dem"], horizon_store=prepared["store"], stage1=prepared["stage1"],
        times=prepared["stamps"], sun_vectors_at=scene.sun_vectors,
        earth_vectors_at=scene.earth_vectors, parameters=parameters,
    ))
    assert records
    grid = FloodGrid(30)
    steps = flood_steps or grid.step_count(parameters.operational_hours)
    signals = windows.PixelSignals(prepared["dem"], prepared["store"])
    shape = (scene.SIZE, scene.SIZE)
    checked = 0
    for record in records:
        month = record["month"]
        hibernation = prepared["stage1"].read_hibernation_window(0, scene.SIZE, 0, scene.SIZE)[month]
        passable = np.isfinite(hibernation)
        for seed in record["seeds"]:
            first = seed["boundary_before_step"] - steps
            last = seed["boundary_after_step"] + steps
            cube = flood.lit_reference_cube(signals, scene.sun_vectors, grid, (0, 0), shape, first, last)
            lit_at = lambda step, cube=cube, first=first: cube[step - first]
            position = (seed["row"], seed["column"])
            result = flood.flood_seed_reference(
                lit_at, passable, position, seed["boundary_before_step"], seed["boundary_after_step"],
                steps=steps, connectivity=parameters.connectivity,
            )
            assert result.count_before == path_count(
                lit_at, passable, position, seed["boundary_before_step"], flood.BEFORE, steps,
                parameters.connectivity)
            assert result.count_after == path_count(
                lit_at, passable, position, seed["boundary_after_step"], flood.AFTER, steps,
                parameters.connectivity)
            assert 0 <= result.count_before <= passable.sum() and 0 <= result.count_after <= passable.sum()
            checked += 1
            if flood_steps == 5:
                # Worked by hand: five steps reach a (2 * 5 + 1) square clipped to the 20 x 20
                # DEM.  Before the outage the whole DEM is lit.  After it, in months 1 and 2,
                # only the -3 degree cluster (rows and columns 4..11) is lit at the boundary
                # and the rest is still in shadow, so the flood stays inside the cluster.
                rows = range(max(0, position[0] - 5), min(scene.SIZE, position[0] + 6))
                columns = range(max(0, position[1] - 5), min(scene.SIZE, position[1] + 6))
                box = len(rows) * len(columns)
                in_cluster = (len([r for r in rows if 4 <= r < 12])
                              * len([c for c in columns if 4 <= c < 12]))
                assert result.count_before == box
                assert result.count_after == (box if month == 0 else in_cluster)
    assert checked >= 5
