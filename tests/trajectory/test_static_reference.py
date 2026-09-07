from __future__ import annotations

import numpy as np
import pytest

import lunarscout as ls


def _replay_path_cost(
    path: np.ndarray,
    georef: ls.GeoReference,
    traversable: np.ndarray,
    elevation: np.ndarray,
    model: ls.trajectory.StaticTravelModel,
) -> float:
    cost = 0.0
    gt = georef.affine_transform
    for left, right in zip(path, path[1:]):
        dx = int(right[0] - left[0])
        dy = int(right[1] - left[1])
        assert (dx, dy) != (0, 0)
        assert abs(dx) <= 1 and abs(dy) <= 1
        x0, y0 = map(int, left)
        x1, y1 = map(int, right)
        assert traversable[y0, x0] and traversable[y1, x1]
        if dx != 0 and dy != 0:
            assert traversable[y0, x1] and traversable[y1, x0]
        projected_dx = dx * gt[1] + dy * gt[2]
        projected_dy = dx * gt[4] + dy * gt[5]
        distance_m = float(np.hypot(projected_dx, projected_dy))
        factor = 1.0
        if model.slip is not None:
            factor = model.slip.factor((elevation[y1, x1] - elevation[y0, x0]) / distance_m)
        cost += distance_m / model.speed_m_per_h * factor
    return cost


def test_all_traversable_field_and_path_agree(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=4, height=3)
    traversable = np.ones((3, 4), dtype=bool)
    field = ls.trajectory.static_travel_time(traversable, georef, (0, 0))
    path = ls.trajectory.static_path(traversable, georef, (0, 0), (3, 2))

    assert path.reachable is True
    assert path.travel_time_hours == pytest.approx(field.travel_time_hours[2, 3])
    assert path.path is not None
    np.testing.assert_array_equal(path.path[0], [0, 0])
    np.testing.assert_array_equal(path.path[-1], [3, 2])
    assert np.all(field.reached)
    assert field.georef is georef
    assert field.start == (0, 0)
    assert not field.travel_time_hours.flags.writeable
    assert not field.reached.flags.writeable
    assert not path.path.flags.writeable


def test_barrier_returns_documented_unreachable_results(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=3, height=3)
    traversable = np.ones((3, 3), dtype=bool)
    traversable[:, 1] = False
    field = ls.trajectory.static_travel_time(traversable, georef, (0, 1))
    path = ls.trajectory.static_path(traversable, georef, (0, 1), (2, 1))

    assert path.reachable is False
    assert path.travel_time_hours is None
    assert path.path is None
    assert field.reached[1, 2] == np.bool_(False)
    assert field.travel_time_hours[1, 2] == np.inf


def test_isolated_start_exhausts_the_frontier(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=3, height=3)
    traversable = np.zeros((3, 3), dtype=bool)
    traversable[1, 1] = True
    result = ls.trajectory.static_travel_time(traversable, georef, (1, 1))

    assert result.reached.sum() == 1
    assert result.travel_time_hours[1, 1] == 0.0
    assert np.all(np.isinf(result.travel_time_hours[~result.reached]))


def test_start_equals_goal_has_one_cell_zero_cost(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=2, height=2)
    result = ls.trajectory.static_path(
        np.ones((2, 2), dtype=bool), georef, (1, 1), (1, 1)
    )

    assert result.reachable is True
    assert result.travel_time_hours == 0.0
    np.testing.assert_array_equal(result.path, np.asarray([[1, 1]], dtype=np.int64))


def test_diagonal_corner_cutting_is_forbidden(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=2, height=2)
    traversable = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    result = ls.trajectory.static_path(traversable, georef, (0, 0), (1, 1))

    assert result.reachable is False


def test_one_blocked_corner_requires_cardinal_detour(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=2, height=2)
    traversable = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
    model = ls.trajectory.StaticTravelModel(speed_m_per_h=10.0)
    result = ls.trajectory.static_path(
        traversable, georef, (0, 0), (1, 1), model=model
    )

    assert result.travel_time_hours == pytest.approx(2.0)
    np.testing.assert_array_equal(result.path, [[0, 0], [0, 1], [1, 1]])


def test_four_neighbor_model_uses_cardinal_path(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=2, height=2)
    model = ls.trajectory.StaticTravelModel(
        speed_m_per_h=10.0, include_diagonals=False
    )
    result = ls.trajectory.static_path(
        np.ones((2, 2), dtype=bool), georef, (0, 0), (1, 1), model=model
    )

    assert result.travel_time_hours == pytest.approx(2.0)
    assert result.path is not None and result.path.shape == (3, 2)


def test_signed_slope_changes_uphill_and_downhill_cost(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=2, height=1)
    slip = ls.trajectory.SlipFunction(
        (-1.0, 0.0, 1.0),
        (0.5, 1.0, 3.0),
        extrapolation="constant",
    )
    model = ls.trajectory.StaticTravelModel(speed_m_per_h=10.0, slip=slip)
    elevation = np.asarray([[0.0, 5.0]])
    uphill = ls.trajectory.static_path(
        np.ones((1, 2), dtype=bool),
        georef,
        (0, 0),
        (1, 0),
        elevation=elevation,
        model=model,
    )
    downhill = ls.trajectory.static_path(
        np.ones((1, 2), dtype=bool),
        georef,
        (1, 0),
        (0, 0),
        elevation=elevation,
        model=model,
    )

    assert uphill.travel_time_hours == pytest.approx(2.0)
    assert downhill.travel_time_hours == pytest.approx(0.75)


def test_slip_uses_slope_not_raw_elevation_difference(make_trajectory_georef) -> None:
    slip = ls.trajectory.SlipFunction(
        (-1.0, 0.0, 1.0), (2.0, 1.0, 3.0), extrapolation="constant"
    )
    model = ls.trajectory.StaticTravelModel(speed_m_per_h=10.0, slip=slip)
    times = []
    for resolution, rise in ((10.0, 5.0), (20.0, 10.0)):
        georef = make_trajectory_georef(
            width=2,
            height=1,
            affine=(0.0, resolution, 0.0, 0.0, 0.0, -resolution),
        )
        result = ls.trajectory.static_path(
            np.ones((1, 2), dtype=bool),
            georef,
            (0, 0),
            (1, 0),
            elevation=np.asarray([[0.0, rise]]),
            model=model,
        )
        times.append(result.travel_time_hours)

    assert times[0] is not None and times[1] == pytest.approx(times[0] * 2.0)


def test_infeasible_slip_edge_is_deliberately_unreachable(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=2, height=1)
    model = ls.trajectory.StaticTravelModel(
        slip=ls.trajectory.SlipFunction((-0.1, 0.1), (1.0, 1.0))
    )
    result = ls.trajectory.static_path(
        np.ones((1, 2), dtype=bool),
        georef,
        (0, 0),
        (1, 0),
        elevation=np.asarray([[0.0, 2.0]]),
        model=model,
    )

    assert result.reachable is False


def test_accumulated_cost_overflow_is_not_treated_as_unreachable(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=3, height=1)
    model = ls.trajectory.StaticTravelModel(
        speed_m_per_h=1.0e-307,
        include_diagonals=False,
    )

    with pytest.raises(ls.PlanningError) as captured:
        ls.trajectory.static_travel_time(
            np.ones((1, 3), dtype=bool), georef, (0, 0), model=model
        )

    assert captured.value.code == "trajectory_cost_overflow"


def test_randomized_astar_costs_match_dijkstra(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(
        width=9,
        height=7,
        affine=(100.0, 7.0, 1.5, 200.0, -0.75, -11.0),
    )
    generator = np.random.default_rng(20260907)
    for include_diagonals in (False, True):
        model = ls.trajectory.StaticTravelModel(
            speed_m_per_h=23.0,
            include_diagonals=include_diagonals,
            slip=ls.trajectory.SlipFunction(
                (-5.0, 0.0, 5.0),
                (1.4, 1.0, 2.0),
                extrapolation="constant",
            ),
        )
        for _ in range(8):
            traversable = generator.random((7, 9)) > 0.2
            traversable[0, 0] = True
            traversable[-1, -1] = True
            elevation = generator.normal(0.0, 3.0, size=(7, 9))
            field = ls.trajectory.static_travel_time(
                traversable,
                georef,
                (0, 0),
                elevation=elevation,
                model=model,
            )
            path = ls.trajectory.static_path(
                traversable,
                georef,
                (0, 0),
                (8, 6),
                elevation=elevation,
                model=model,
            )
            expected = field.travel_time_hours[6, 8]
            if np.isfinite(expected):
                assert path.reachable is True
                assert path.travel_time_hours == pytest.approx(expected, rel=1e-12)
                assert path.path is not None
                replayed = _replay_path_cost(
                    path.path, georef, traversable, elevation, model
                )
                assert path.travel_time_hours == pytest.approx(replayed, rel=1e-12)
            else:
                assert path.reachable is False
