from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.trajectory._dynamic_gridrunner import gridrunner_dynamic_path
from lunarscout.trajectory._dynamic_reference import (
    DynamicOccupancyTimeline,
    exact_dynamic_path,
)
from lunarscout.trajectory._safe_interval import (
    build_safe_intervals,
    safe_interval_dynamic_path,
    safe_interval_dynamic_path_from_provider,
)
from lunarscout.trajectory._validation import prepare_static_problem


T0 = datetime(2039, 1, 1, tzinfo=timezone.utc)


def _case(
    make_trajectory_georef,
    allowed,
    *,
    start=(0, 0),
    goal=None,
    speed=20.0,
):
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
            speed_m_per_h=speed,
            include_diagonals=False,
        ),
    )
    boundaries = tuple(
        T0 + timedelta(hours=index) for index in range(intervals + 1)
    )
    return problem, DynamicOccupancyTimeline(boundaries, allowed, grid)


def _assert_same(left, right) -> None:
    assert left.reachable == right.reachable
    assert left.arrival_time == right.arrival_time
    if left.reachable:
        assert left.cells is not None and right.cells is not None
        assert left.cells[0].tolist() == right.cells[0].tolist()
        assert left.cells[-1].tolist() == right.cells[-1].tolist()


def _assert_replay(result, problem, timeline) -> None:
    if not result.reachable:
        return
    assert result.cells is not None
    assert result.arrival_times is not None
    assert result.departure_times is not None
    for index, departure_time in enumerate(result.departure_times):
        source_x, source_y = (int(value) for value in result.cells[index])
        destination_x, destination_y = (
            int(value) for value in result.cells[index + 1]
        )
        source_arrival = timeline.hours_from_start(
            result.arrival_times[index], name="source_arrival"
        )
        departure = timeline.hours_from_start(departure_time, name="departure")
        arrival = timeline.hours_from_start(
            result.arrival_times[index + 1], name="arrival"
        )
        step = next(
            item
            for item in problem.steps
            if item.dx == destination_x - source_x
            and item.dy == destination_y - source_y
        )
        assert departure >= source_arrival
        assert arrival - departure == pytest.approx(
            problem.transition_time(source_x, source_y, step)
        )
        wait_first = timeline.interval_index(source_arrival)
        wait_last = int(
            np.searchsorted(timeline._boundary_hours, departure, side="left") - 1
        )
        if departure > source_arrival:
            assert wait_first is not None
            assert np.all(
                timeline.allowed[
                    wait_first : wait_last + 1,
                    source_y,
                    source_x,
                ]
            )
        move_first = timeline.interval_index(departure)
        move_last = int(
            np.searchsorted(timeline._boundary_hours, arrival, side="left") - 1
        )
        arrival_interval = timeline.interval_index(arrival)
        assert move_first is not None and arrival_interval is not None
        for y, x in (
            (source_y, source_x),
            (destination_y, destination_x),
        ):
            assert np.all(timeline.allowed[move_first : move_last + 1, y, x])
        assert timeline.allowed[arrival_interval, destination_y, destination_x]


def test_maximal_safe_interval_construction(make_trajectory_georef) -> None:
    allowed = np.asarray(
        [
            [[1, 0]],
            [[1, 1]],
            [[0, 1]],
            [[1, 0]],
            [[1, 1]],
        ],
        dtype=bool,
    )
    problem, timeline = _case(make_trajectory_georef, allowed)

    table = build_safe_intervals(timeline)

    assert problem.georef == timeline.georef
    assert [
        (item.start_hours, item.stop_hours, item.first_interval, item.stop_interval)
        for item in table.intervals(0, 0)
    ] == [(0.0, 2.0, 0, 2), (3.0, 5.0, 3, 5)]
    assert [
        (item.start_hours, item.stop_hours, item.first_interval, item.stop_interval)
        for item in table.intervals(1, 0)
    ] == [(1.0, 3.0, 1, 3), (4.0, 5.0, 4, 5)]
    assert table.safe_interval_count == 4


@pytest.mark.parametrize(
    ("allowed", "speed", "arrival_hours", "waits"),
    [
        (np.ones((4, 1, 3), dtype=bool), 20.0, 1.0, 0),
        (
            np.asarray(
                [
                    [[1, 1, 0]],
                    [[1, 1, 0]],
                    [[1, 1, 1]],
                    [[1, 1, 1]],
                ],
                dtype=bool,
            ),
            20.0,
            2.5,
            1,
        ),
        (np.ones((4, 1, 2), dtype=bool), 4.0, 2.5, 0),
    ],
)
def test_safe_interval_hand_authored_cases(
    make_trajectory_georef, allowed, speed, arrival_hours, waits
) -> None:
    problem, timeline = _case(
        make_trajectory_georef, allowed, goal=(allowed.shape[2] - 1, 0), speed=speed
    )

    exact = exact_dynamic_path(problem, timeline, T0)
    result = safe_interval_dynamic_path(problem, timeline, T0)

    _assert_same(result.path, exact)
    _assert_replay(result.path, problem, timeline)
    assert result.path.arrival_time == T0 + timedelta(hours=arrival_hours)
    assert result.path.arrival_times is not None
    assert result.path.departure_times is not None
    wait_count = sum(
        departure > arrival
        for arrival, departure in zip(
            result.path.arrival_times, result.path.departure_times
        )
    )
    assert wait_count == waits


def test_safe_interval_matches_both_exact_planners_on_random_cases(
    make_trajectory_georef,
) -> None:
    generator = np.random.default_rng(20260909)
    for _ in range(60):
        allowed = generator.random((5, 4, 5)) > 0.35
        allowed[0, 0, 0] = True
        allowed[:, 3, 4] = True
        problem, timeline = _case(make_trajectory_georef, allowed)

        exact = exact_dynamic_path(problem, timeline, T0)
        gridrunner = gridrunner_dynamic_path(problem, timeline, T0, block_size=2)
        safe = safe_interval_dynamic_path(problem, timeline, T0)

        _assert_same(safe.path, exact)
        _assert_same(safe.path, gridrunner.path)
        _assert_replay(safe.path, problem, timeline)


def test_safe_interval_matches_exact_on_nonuniform_timelines(
    make_trajectory_georef,
) -> None:
    generator = np.random.default_rng(20260910)
    for _ in range(30):
        intervals = 6
        allowed = generator.random((intervals, 3, 4)) > 0.3
        allowed[0, 0, 0] = True
        allowed[:, 2, 3] = True
        grid = make_trajectory_georef(width=4, height=3)
        problem = prepare_static_problem(
            np.any(allowed, axis=0),
            grid,
            (0, 0),
            goal=(3, 2),
            model=ls.trajectory.StaticTravelModel(
                speed_m_per_h=float(generator.uniform(12.0, 50.0)),
                include_diagonals=bool(generator.integers(0, 2)),
            ),
        )
        durations = generator.uniform(0.2, 1.3, size=intervals)
        offsets = np.concatenate(([0.0], np.cumsum(durations)))
        boundaries = tuple(T0 + timedelta(hours=float(value)) for value in offsets)
        timeline = DynamicOccupancyTimeline(boundaries, allowed, grid)

        exact = exact_dynamic_path(problem, timeline, T0)
        safe = safe_interval_dynamic_path(problem, timeline, T0)

        _assert_same(safe.path, exact)
        _assert_replay(safe.path, problem, timeline)


def test_safe_interval_provider_path_and_compression(
    make_trajectory_georef,
) -> None:
    allowed = np.ones((20, 2, 3), dtype=bool)
    problem, timeline = _case(make_trajectory_georef, allowed)
    source = ls.trajectory.ArraySunlightProvider(
        timeline.boundaries,
        np.full(allowed.shape, 255, dtype=np.uint8),
        timeline.georef,
    )
    provider = ls.trajectory.SunlightThresholdProvider(source, 1.0)

    result = safe_interval_dynamic_path_from_provider(
        problem,
        provider,
        timeline.boundaries,
        T0,
        read_block_size=2,
    )

    assert result.path.reachable
    assert result.diagnostics.safe_intervals == 6
    assert result.diagnostics.safe_intervals < allowed.size
    assert all(key[3] <= 2 and key[4] <= 2 for key in source._cache.values.keys())


def test_safe_interval_state_limit_precedes_provider_reads(
    make_trajectory_georef, monkeypatch
) -> None:
    from lunarscout.trajectory import _safe_interval

    allowed = np.ones((1, 1, 2), dtype=bool)
    problem, timeline = _case(make_trajectory_georef, allowed)

    class CountingProvider:
        georef = timeline.georef

        def __init__(self):
            self.calls = 0

        def read(self, x0, y0, width, height, time):
            self.calls += 1
            return np.ones((height, width), dtype=bool)

    provider = CountingProvider()
    monkeypatch.setattr(_safe_interval, "_MAX_SAFE_INTERVAL_TIMELINE_STATES", 1)

    with pytest.raises(ls.PlanningError) as capacity:
        safe_interval_dynamic_path_from_provider(
            problem, provider, timeline.boundaries, T0
        )
    assert capacity.value.code == "trajectory_safe_interval_state_limit"
    assert provider.calls == 0
