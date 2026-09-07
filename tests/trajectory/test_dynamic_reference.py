from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.trajectory._dynamic_reference import (
    DynamicOccupancyTimeline,
    exact_dynamic_path,
)
from lunarscout.trajectory._validation import prepare_static_problem


_T0 = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _timeline(georef, allowed: np.ndarray, step_hours: float = 1.0):
    boundaries = tuple(
        _T0 + timedelta(hours=index * step_hours)
        for index in range(allowed.shape[0] + 1)
    )
    return DynamicOccupancyTimeline(boundaries, allowed, georef)


def _line_problem(make_trajectory_georef, length: int, *, duration_hours: float = 0.5):
    georef = make_trajectory_georef(width=length, height=1)
    model = ls.trajectory.StaticTravelModel(
        speed_m_per_h=10.0 / duration_hours,
        include_diagonals=False,
    )
    problem = prepare_static_problem(
        np.ones((1, length), dtype=bool),
        georef,
        (0, 0),
        goal=(length - 1, 0),
        model=model,
    )
    return georef, problem


def test_move_crosses_multiple_environment_intervals(make_trajectory_georef) -> None:
    georef, problem = _line_problem(
        make_trajectory_georef, 2, duration_hours=2.5
    )
    timeline = _timeline(georef, np.ones((4, 1, 2), dtype=bool))
    result = exact_dynamic_path(problem, timeline, _T0)

    assert result.reachable is True
    assert result.arrival_time == _T0 + timedelta(hours=2.5)
    assert result.departure_times == (_T0,)


def test_wait_before_departure_when_destination_becomes_allowed(
    make_trajectory_georef,
) -> None:
    georef, problem = _line_problem(make_trajectory_georef, 2)
    allowed = np.ones((3, 1, 2), dtype=bool)
    allowed[0, 0, 1] = False
    result = exact_dynamic_path(problem, _timeline(georef, allowed), _T0)

    assert result.arrival_time == _T0 + timedelta(hours=1.5)
    assert result.departure_times == (_T0 + timedelta(hours=1.0),)


def test_wait_after_partial_progress(make_trajectory_georef) -> None:
    georef, problem = _line_problem(make_trajectory_georef, 3)
    allowed = np.ones((4, 1, 3), dtype=bool)
    allowed[:2, 0, 2] = False
    result = exact_dynamic_path(problem, _timeline(georef, allowed), _T0)

    assert result.arrival_time == _T0 + timedelta(hours=2.5)
    assert result.arrival_times == (
        _T0,
        _T0 + timedelta(hours=0.5),
        _T0 + timedelta(hours=2.5),
    )
    assert result.departure_times == (
        _T0,
        _T0 + timedelta(hours=2.0),
    )


def test_source_cannot_wait_across_forbidden_interval(make_trajectory_georef) -> None:
    georef, problem = _line_problem(make_trajectory_georef, 2)
    allowed = np.ones((3, 1, 2), dtype=bool)
    allowed[0, 0, 1] = False
    allowed[1:, 0, 0] = False
    result = exact_dynamic_path(problem, _timeline(georef, allowed), _T0)

    assert result.reachable is False


def test_arrival_at_boundary_uses_new_destination_interval(
    make_trajectory_georef,
) -> None:
    georef, problem = _line_problem(
        make_trajectory_georef, 2, duration_hours=1.0
    )
    allowed = np.ones((2, 1, 2), dtype=bool)
    allowed[1, 0, 0] = False
    result = exact_dynamic_path(problem, _timeline(georef, allowed), _T0)

    assert result.reachable is True
    assert result.arrival_time == _T0 + timedelta(hours=1.0)

    allowed[1, 0, 1] = False
    blocked = exact_dynamic_path(problem, _timeline(georef, allowed), _T0)
    assert blocked.reachable is False


def test_forbidden_mid_edge_is_not_sampled_only_at_endpoints(
    make_trajectory_georef,
) -> None:
    georef, problem = _line_problem(
        make_trajectory_georef, 2, duration_hours=2.5
    )
    allowed = np.ones((5, 1, 2), dtype=bool)
    allowed[1, 0, 1] = False
    result = exact_dynamic_path(problem, _timeline(georef, allowed), _T0)

    assert result.arrival_time == _T0 + timedelta(hours=4.5)
    assert result.departure_times == (_T0 + timedelta(hours=2.0),)


def test_arrival_at_final_boundary_is_timeline_exhaustion(
    make_trajectory_georef,
) -> None:
    georef, problem = _line_problem(
        make_trajectory_georef, 2, duration_hours=1.0
    )
    timeline = _timeline(georef, np.ones((1, 1, 2), dtype=bool))
    result = exact_dynamic_path(problem, timeline, _T0)

    assert result.reachable is False


def test_dynamic_start_and_departure_are_validated(make_trajectory_georef) -> None:
    georef, problem = _line_problem(make_trajectory_georef, 2)
    allowed = np.ones((2, 1, 2), dtype=bool)
    allowed[0, 0, 0] = False
    timeline = _timeline(georef, allowed)
    with pytest.raises(ls.TrajectoryInputError) as unavailable:
        exact_dynamic_path(problem, timeline, _T0)
    with pytest.raises(ls.TrajectoryInputError) as outside:
        exact_dynamic_path(problem, timeline, _T0 + timedelta(hours=2))

    assert unavailable.value.code == "trajectory_dynamic_start_unavailable"
    assert outside.value.code == "trajectory_departure_outside_timeline"


def test_dynamic_timeline_rejects_naive_nonmonotonic_and_bad_shape(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=2, height=1)
    with pytest.raises(ls.TrajectoryInputError) as naive:
        DynamicOccupancyTimeline(
            (datetime(2030, 1, 1), datetime(2030, 1, 2)),
            np.ones((1, 1, 2), dtype=bool),
            georef,
        )
    with pytest.raises(ls.TrajectoryInputError) as order:
        DynamicOccupancyTimeline(
            (_T0, _T0), np.ones((1, 1, 2), dtype=bool), georef
        )
    with pytest.raises(ls.TrajectoryInputError) as shape:
        DynamicOccupancyTimeline(
            (_T0, _T0 + timedelta(hours=1)),
            np.ones((2, 1, 2), dtype=bool),
            georef,
        )

    assert naive.value.code == "trajectory_invalid_dynamic_time"
    assert order.value.code == "trajectory_invalid_dynamic_time"
    assert shape.value.code == "trajectory_invalid_dynamic_grid"


def test_dynamic_grid_must_match_static_problem(make_trajectory_georef) -> None:
    georef, problem = _line_problem(make_trajectory_georef, 2)
    other = make_trajectory_georef(
        width=2,
        height=1,
        affine=(1.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    )
    timeline = _timeline(other, np.ones((2, 1, 2), dtype=bool))

    with pytest.raises(ls.TrajectoryInputError) as captured:
        exact_dynamic_path(problem, timeline, _T0)

    assert captured.value.code == "trajectory_dynamic_grid_mismatch"


def _brute_force_half_hour_line(allowed: np.ndarray) -> int | None:
    """Return the earliest goal tick using an independent time-expanded BFS."""

    interval_count, _height, width = allowed.shape
    final_tick = interval_count * 2
    frontier = {(0, 0)}
    for tick in range(final_tick):
        next_frontier: set[tuple[int, int]] = set()
        for state_tick, x in frontier:
            if state_tick != tick:
                next_frontier.add((state_tick, x))
                continue
            if x == width - 1:
                return tick
            next_tick = tick + 1
            if next_tick >= final_tick:
                continue
            current_interval = tick // 2
            arrival_interval = next_tick // 2
            if (
                allowed[current_interval, 0, x]
                and allowed[arrival_interval, 0, x]
            ):
                next_frontier.add((next_tick, x))
            for destination in (x - 1, x + 1):
                if destination < 0 or destination >= width:
                    continue
                if (
                    allowed[current_interval, 0, x]
                    and allowed[current_interval, 0, destination]
                    and allowed[arrival_interval, 0, destination]
                ):
                    next_frontier.add((next_tick, destination))
        frontier = next_frontier
    return None


def test_exact_oracle_matches_independent_time_expansion(
    make_trajectory_georef,
) -> None:
    georef, problem = _line_problem(make_trajectory_georef, 4)
    generator = np.random.default_rng(20260907)
    for _ in range(50):
        allowed = generator.random((4, 1, 4)) > 0.3
        allowed[0, 0, 0] = True
        timeline = _timeline(georef, allowed)
        result = exact_dynamic_path(problem, timeline, _T0)
        expected_tick = _brute_force_half_hour_line(allowed)
        if expected_tick is None:
            assert result.reachable is False
        else:
            assert result.reachable is True
            assert result.arrival_time == _T0 + timedelta(hours=expected_tick / 2)


def test_exact_oracle_has_a_structured_small_problem_limit(
    make_trajectory_georef,
    monkeypatch,
) -> None:
    from lunarscout.trajectory import _dynamic_reference

    georef, problem = _line_problem(make_trajectory_georef, 2)
    timeline = _timeline(georef, np.ones((2, 1, 2), dtype=bool))
    monkeypatch.setattr(_dynamic_reference, "_MAX_EXACT_STATES", 3)

    with pytest.raises(ls.PlanningError) as captured:
        exact_dynamic_path(problem, timeline, _T0)

    assert captured.value.code == "trajectory_exact_dynamic_state_limit"
    assert captured.value.details == {"state_count": 4, "maximum": 3}
