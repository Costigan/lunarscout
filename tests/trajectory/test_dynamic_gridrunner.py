from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.trajectory._dynamic_gridrunner import (
    gridrunner_dynamic_path,
    gridrunner_dynamic_path_from_provider,
)
from lunarscout.trajectory._dynamic_mobility import compile_dynamic_travel_model
from lunarscout.trajectory._dynamic_reference import (
    DynamicOccupancyTimeline,
    exact_dynamic_path,
)
from lunarscout.trajectory._validation import prepare_static_problem


T0 = datetime(2037, 1, 1, tzinfo=timezone.utc)


def _case(make_trajectory_georef, allowed, *, start=(0, 0), goal=None):
    intervals, height, width = allowed.shape
    grid = make_trajectory_georef(width=width, height=height)
    if goal is None:
        goal = (width - 1, height - 1)
    problem = prepare_static_problem(
        np.any(allowed, axis=0),
        grid,
        start,
        goal=goal,
        model=ls.trajectory.StaticTravelModel(
            speed_m_per_h=20.0,
            include_diagonals=False,
        ),
    )
    boundaries = tuple(
        T0 + timedelta(hours=index) for index in range(intervals + 1)
    )
    timeline = DynamicOccupancyTimeline(boundaries, allowed, grid)
    return problem, timeline


def _assert_same(left, right) -> None:
    assert left.reachable == right.reachable
    assert left.arrival_time == right.arrival_time
    if left.reachable:
        assert left.cells is not None and right.cells is not None
        assert left.cells[0].tolist() == right.cells[0].tolist()
        assert left.cells[-1].tolist() == right.cells[-1].tolist()


def _assert_occupancy_feasible(result, timeline) -> None:
    if not result.reachable:
        return
    assert result.cells is not None
    assert result.arrival_times is not None
    assert result.departure_times is not None
    assert len(result.arrival_times) == len(result.cells)
    assert len(result.departure_times) == len(result.cells) - 1
    for index, departure_time in enumerate(result.departure_times):
        source_x, source_y = result.cells[index]
        destination_x, destination_y = result.cells[index + 1]
        source_arrival = timeline.hours_from_start(
            result.arrival_times[index], name="source_arrival"
        )
        departure = timeline.hours_from_start(departure_time, name="departure")
        arrival = timeline.hours_from_start(
            result.arrival_times[index + 1], name="arrival"
        )
        assert departure >= source_arrival
        assert abs(int(destination_x) - int(source_x)) + abs(
            int(destination_y) - int(source_y)
        ) == 1
        wait_first = timeline.interval_index(source_arrival)
        wait_last = int(
            np.searchsorted(timeline._boundary_hours, departure, side="left") - 1
        )
        if departure > source_arrival:
            assert wait_first is not None
            assert np.all(
                timeline.allowed[wait_first : wait_last + 1, source_y, source_x]
            )
        move_first = timeline.interval_index(departure)
        move_last = int(
            np.searchsorted(timeline._boundary_hours, arrival, side="left") - 1
        )
        destination_interval = timeline.interval_index(arrival)
        assert move_first is not None and destination_interval is not None
        assert np.all(
            timeline.allowed[move_first : move_last + 1, source_y, source_x]
        )
        assert np.all(
            timeline.allowed[
                move_first : move_last + 1, destination_y, destination_x
            ]
        )
        assert timeline.allowed[destination_interval, destination_y, destination_x]


def test_gridrunner_matches_exact_wait_and_move_case(make_trajectory_georef) -> None:
    allowed = np.ones((4, 1, 3), dtype=bool)
    allowed[:2, 0, 2] = False
    problem, timeline = _case(make_trajectory_georef, allowed)

    exact = exact_dynamic_path(problem, timeline, T0)
    blocked = gridrunner_dynamic_path(problem, timeline, T0, block_size=1)

    _assert_same(blocked.path, exact)
    assert blocked.diagnostics.blocks_processed > 0
    assert blocked.diagnostics.local_passes >= blocked.diagnostics.blocks_processed


def test_gridrunner_matches_exact_compiled_mobility(make_trajectory_georef) -> None:
    allowed = np.ones((5, 1, 3), dtype=bool)
    problem, timeline = _case(make_trajectory_georef, allowed)
    factors = np.ones((5, 4), dtype=np.float64)
    factors[0, 0] = 2.0
    factors[1, 0] = 0.5
    compiled = compile_dynamic_travel_model(
        problem, timeline, interval_edge_factors=factors
    )

    exact = exact_dynamic_path(problem, timeline, T0, travel_model=compiled)
    blocked = gridrunner_dynamic_path(
        problem, timeline, T0, travel_model=compiled, block_size=2
    )

    _assert_same(blocked.path, exact)


def test_gridrunner_matches_exact_randomized_small_cases(
    make_trajectory_georef,
) -> None:
    generator = np.random.default_rng(20260908)
    for _ in range(40):
        allowed = generator.random((4, 4, 5)) > 0.25
        allowed[0, 0, 0] = True
        allowed[:, 3, 4] = True
        static = np.any(allowed, axis=0)
        static[0, 0] = True
        static[3, 4] = True
        grid = make_trajectory_georef(width=5, height=4)
        problem = prepare_static_problem(
            static,
            grid,
            (0, 0),
            goal=(4, 3),
            model=ls.trajectory.StaticTravelModel(
                speed_m_per_h=20.0,
                include_diagonals=False,
            ),
        )
        timeline = DynamicOccupancyTimeline(
            tuple(T0 + timedelta(hours=index) for index in range(5)),
            allowed,
            grid,
        )

        exact = exact_dynamic_path(problem, timeline, T0)
        blocked = gridrunner_dynamic_path(problem, timeline, T0, block_size=2)

        _assert_same(blocked.path, exact)
        _assert_occupancy_feasible(blocked.path, timeline)


def test_gridrunner_validation_and_state_bound(
    make_trajectory_georef, monkeypatch
) -> None:
    from lunarscout.trajectory import _dynamic_gridrunner

    problem, timeline = _case(
        make_trajectory_georef, np.ones((2, 1, 2), dtype=bool)
    )
    with pytest.raises(ls.TrajectoryInputError) as block:
        gridrunner_dynamic_path(problem, timeline, T0, block_size=0)
    assert block.value.code == "trajectory_invalid_block_size"

    monkeypatch.setattr(_dynamic_gridrunner, "_MAX_GRIDRUNNER_STATES", 3)
    with pytest.raises(ls.PlanningError) as capacity:
        gridrunner_dynamic_path(problem, timeline, T0)
    assert capacity.value.code == "trajectory_gridrunner_state_limit"


def test_gridrunner_reactivates_a_previously_processed_block(
    make_trajectory_georef,
) -> None:
    allowed = np.asarray(
        [
            [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
            [[0, 1, 0], [1, 0, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            [[1, 0, 0], [0, 1, 0], [1, 1, 1]],
        ],
        dtype=bool,
    )
    problem, timeline = _case(make_trajectory_georef, allowed)

    exact = exact_dynamic_path(problem, timeline, T0)
    blocked = gridrunner_dynamic_path(problem, timeline, T0, block_size=2)

    _assert_same(blocked.path, exact)
    assert blocked.path.reachable
    assert blocked.diagnostics.block_reactivations >= 1


def test_gridrunner_reads_configuration_provider_in_bounded_windows(
    make_trajectory_georef,
) -> None:
    allowed = np.ones((3, 3, 5), dtype=bool)
    problem, timeline = _case(make_trajectory_georef, allowed)
    source = ls.trajectory.ArraySunlightProvider(
        timeline.boundaries,
        np.full(allowed.shape, 255, dtype=np.uint8),
        timeline.georef,
    )
    configuration = ls.trajectory.SunlightThresholdProvider(source, 1.0)

    result = gridrunner_dynamic_path_from_provider(
        problem,
        configuration,
        timeline.boundaries,
        T0,
        block_size=2,
    )
    exact = exact_dynamic_path(problem, timeline, T0)

    _assert_same(result.path, exact)
    assert all(key[3] <= 2 and key[4] <= 2 for key in source._cache.values.keys())
