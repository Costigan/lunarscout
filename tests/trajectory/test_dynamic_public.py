from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls


T0 = datetime(2038, 1, 1, tzinfo=timezone.utc)


def _inputs(make_trajectory_georef):
    grid = make_trajectory_georef(width=3, height=1)
    boundaries = tuple(T0 + timedelta(hours=index) for index in range(5))
    allowed = np.ones((4, 1, 3), dtype=bool)
    allowed[:2, 0, 2] = False
    configuration = ls.trajectory.StaticConfigurationSpaceProvider(
        np.ones((1, 3), dtype=bool), grid
    )
    dynamic = ls.trajectory.ArraySunlightProvider(
        boundaries, allowed.astype(np.uint8) * 255, grid
    )
    configuration = ls.trajectory.AllOfConfigurationSpaceProvider(
        (
            configuration,
            ls.trajectory.SunlightThresholdProvider(dynamic, 1.0),
        )
    )
    return grid, boundaries, configuration


def test_public_dynamic_path_reports_waits_and_exact_arrival(
    make_trajectory_georef,
) -> None:
    grid, boundaries, configuration = _inputs(make_trajectory_georef)
    result = ls.trajectory.dynamic_path(
        np.ones((1, 3), dtype=bool),
        grid,
        (0, 0),
        (2, 0),
        boundaries,
        configuration,
        T0,
        model=ls.trajectory.StaticTravelModel(
            speed_m_per_h=20.0, include_diagonals=False
        ),
        backend="cpu",
    )

    assert isinstance(result, ls.trajectory.DynamicPathResult)
    assert result.reachable
    assert result.cells.tolist() == [[0, 0], [1, 0], [2, 0]]
    assert result.arrival_time == T0 + timedelta(hours=2.5)
    assert result.travel_time_hours == 2.5
    assert result.departure_times == (T0, T0 + timedelta(hours=2))
    assert result.wait_intervals == (
        (T0 + timedelta(minutes=30), T0 + timedelta(hours=2)),
    )
    assert result.cells.flags.writeable is False


def test_public_dynamic_unreachable_and_start_equals_goal(
    make_trajectory_georef,
) -> None:
    grid, boundaries, configuration = _inputs(make_trajectory_georef)
    unavailable_goal = ls.trajectory.ArraySunlightProvider(
        boundaries, np.asarray([[[255, 0, 0]]] * 4, dtype=np.uint8), grid
    )
    blocked = ls.trajectory.dynamic_path(
        np.ones((1, 3), dtype=bool),
        grid,
        (0, 0),
        (2, 0),
        boundaries,
        ls.trajectory.SunlightThresholdProvider(unavailable_goal, 1.0),
        T0,
    )
    same = ls.trajectory.dynamic_path(
        np.ones((1, 3), dtype=bool),
        grid,
        (0, 0),
        (0, 0),
        boundaries,
        configuration,
        T0,
    )

    assert blocked.reachable is False
    assert blocked.arrival_time is None and blocked.cells is None
    assert same.reachable and same.travel_time_hours == 0.0
    assert same.cells.tolist() == [[0, 0]]
    assert same.departure_times == () and same.wait_intervals == ()


def test_public_dynamic_auto_and_cpu_are_equivalent(make_trajectory_georef) -> None:
    grid, boundaries, configuration = _inputs(make_trajectory_georef)
    arguments = (
        np.ones((1, 3), dtype=bool),
        grid,
        (0, 0),
        (2, 0),
        boundaries,
        configuration,
        T0,
    )
    auto = ls.trajectory.dynamic_path(*arguments, backend="auto")
    cpu = ls.trajectory.dynamic_path(*arguments, backend="cpu")

    assert auto.arrival_time == cpu.arrival_time
    assert np.array_equal(auto.cells, cpu.cells)


def test_dispatch_fails_before_provider_reads(make_trajectory_georef) -> None:
    grid = make_trajectory_georef(width=1, height=1)

    class CountingProvider:
        georef = grid

        def __init__(self):
            self.calls = 0

        def read(self, x0, y0, width, height, time):
            self.calls += 1
            return np.ones((height, width), dtype=bool)

    provider = CountingProvider()
    arguments = (
        np.ones((1, 1), dtype=bool),
        grid,
        (0, 0),
        (0, 0),
        (T0, T0 + timedelta(hours=1)),
        provider,
        T0,
    )
    with pytest.raises(ls.TrajectoryInputError) as algorithm:
        ls.trajectory.dynamic_path(*arguments, algorithm="other")
    with pytest.raises(ls.TrajectoryInputError) as backend:
        ls.trajectory.dynamic_path(*arguments, backend="other")
    with pytest.raises(ls.PlanningError) as cuda:
        ls.trajectory.dynamic_path(*arguments, backend="cuda")

    assert algorithm.value.code == "trajectory_unknown_algorithm"
    assert backend.value.code == "trajectory_unknown_backend"
    assert cuda.value.code == "trajectory_backend_unavailable"
    assert provider.calls == 0


def test_grid_mismatch_fails_before_provider_read(make_trajectory_georef) -> None:
    grid = make_trajectory_georef(width=1, height=1)
    other = make_trajectory_georef(
        width=1,
        height=1,
        affine=(1.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    )

    class CountingProvider:
        georef = other

        def __init__(self):
            self.calls = 0

        def read(self, x0, y0, width, height, time):
            self.calls += 1
            return np.ones((height, width), dtype=bool)

    provider = CountingProvider()
    with pytest.raises(ls.TrajectoryInputError) as captured:
        ls.trajectory.dynamic_path(
            np.ones((1, 1), dtype=bool),
            grid,
            (0, 0),
            (0, 0),
            (T0, T0 + timedelta(hours=1)),
            provider,
            T0,
        )
    assert captured.value.code == "trajectory_provider_grid_mismatch"
    assert provider.calls == 0


def test_dynamic_result_validates_time_and_owns_cells() -> None:
    cells = np.asarray([[0, 0], [1, 0]], dtype=np.int64)
    result = ls.trajectory.DynamicPathResult(
        True,
        T0 + timedelta(hours=1),
        cells,
        (T0, T0 + timedelta(hours=1)),
        (T0,),
    )
    cells[:] = 9
    assert result.cells.tolist() == [[0, 0], [1, 0]]

    with pytest.raises(ls.TrajectoryInputError) as invalid:
        ls.trajectory.DynamicPathResult(
            True,
            T0 + timedelta(hours=1),
            np.asarray([[0, 0], [1, 0]]),
            (T0, T0 + timedelta(hours=1)),
            (T0 + timedelta(hours=2),),
        )
    assert invalid.value.code == "trajectory_invalid_dynamic_result"


def test_dynamic_result_validates_exact_uint64_limit() -> None:
    accepted = ls.trajectory.DynamicPathResult(
        True,
        T0,
        np.asarray([[np.iinfo(np.int64).max, 0]], dtype=np.uint64),
        (T0,),
        (),
    )
    assert accepted.cells.tolist() == [[np.iinfo(np.int64).max, 0]]

    with pytest.raises(ls.TrajectoryInputError) as invalid:
        ls.trajectory.DynamicPathResult(
            True,
            T0,
            np.asarray([[np.iinfo(np.uint64).max, 0]], dtype=np.uint64),
            (T0,),
            (),
        )
    assert invalid.value.code == "trajectory_invalid_dynamic_result"


def test_public_state_limit_fails_before_provider_reads(
    make_trajectory_georef, monkeypatch
) -> None:
    from lunarscout.trajectory import _dynamic_gridrunner

    grid = make_trajectory_georef(width=2, height=1)

    class CountingProvider:
        georef = grid

        def __init__(self):
            self.calls = 0

        def read(self, x0, y0, width, height, time):
            self.calls += 1
            return np.ones((height, width), dtype=bool)

    provider = CountingProvider()
    monkeypatch.setattr(_dynamic_gridrunner, "_MAX_GRIDRUNNER_STATES", 1)

    with pytest.raises(ls.PlanningError) as capacity:
        ls.trajectory.dynamic_path(
            np.ones((1, 2), dtype=bool),
            grid,
            (0, 0),
            (1, 0),
            (T0, T0 + timedelta(hours=1)),
            provider,
            T0,
        )
    assert capacity.value.code == "trajectory_gridrunner_state_limit"
    assert provider.calls == 0
