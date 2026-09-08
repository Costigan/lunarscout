from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.trajectory._dynamic_reference import DynamicOccupancyTimeline
from lunarscout.trajectory._power_accounting import PiecewiseSunlightTimeline
from lunarscout.trajectory._power_greedy import (
    _preferred,
    greedy_soc_path,
)
from lunarscout.trajectory._power_reference import _SocLabel, exact_soc_path
from lunarscout.trajectory._validation import prepare_static_problem


T0 = datetime(2041, 1, 1, tzinfo=timezone.utc)


def _models(
    *,
    capacity=500.0,
    initial=300.0,
    minimum=0.0,
    solar=100.0,
    drive=100.0,
    idle=0.0,
):
    return {
        "solar": ls.trajectory.SolarPowerModel(solar),
        "battery": ls.trajectory.BatteryModel(
            capacity, initial, minimum, 1.0, 1.0
        ),
        "rover": ls.trajectory.RoverPowerModel(drive, idle),
    }


def _case(
    make_trajectory_georef,
    allowed,
    sunlight,
    *,
    start=(0, 0),
    goal=None,
    edge_hours=1.0,
    interval_hours=1.0,
    available=None,
):
    intervals, height, width = allowed.shape
    grid = make_trajectory_georef(width=width, height=height)
    if goal is None:
        goal = (width - 1, height - 1)
    if available is None:
        available = np.ones((height, width), dtype=bool)
    problem = prepare_static_problem(
        available,
        grid,
        start,
        goal=goal,
        model=ls.trajectory.StaticTravelModel(
            speed_m_per_h=10.0 / edge_hours,
            include_diagonals=False,
        ),
    )
    boundaries = tuple(
        T0 + timedelta(hours=index * interval_hours)
        for index in range(intervals + 1)
    )
    return (
        problem,
        DynamicOccupancyTimeline(boundaries, allowed, grid),
        PiecewiseSunlightTimeline(boundaries, sunlight, grid),
    )


def test_greedy_matches_exact_when_one_label_per_state_is_sufficient(
    make_trajectory_georef,
) -> None:
    allowed = np.ones((4, 1, 3), dtype=bool)
    sunlight = np.asarray(
        [[[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]] * 2,
        dtype=np.float64,
    )
    problem, occupancy, power = _case(
        make_trajectory_georef,
        allowed,
        sunlight,
        goal=(2, 0),
        edge_hours=0.5,
    )
    models = _models(initial=250.0, drive=200.0, idle=50.0)
    exact = exact_soc_path(problem, occupancy, power, T0, **models)
    greedy = greedy_soc_path(problem, occupancy, power, T0, **models)

    assert exact.reachable and greedy.reachable
    assert greedy.path.arrival_time == exact.path.arrival_time
    assert greedy.path.cells is not None and exact.path.cells is not None
    assert greedy.path.cells.tolist() == exact.path.cells.tolist()
    assert greedy.energy == exact.energy


def test_greedy_can_miss_later_high_energy_label_in_same_interval(
    make_trajectory_georef,
) -> None:
    allowed = np.ones((1, 2, 3), dtype=bool)
    available = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=bool)
    sunlight = np.zeros((1, 2, 3), dtype=np.float64)
    sunlight[:, 1, :2] = 1.0
    problem, occupancy, power = _case(
        make_trajectory_georef,
        allowed,
        sunlight,
        goal=(2, 0),
        edge_hours=0.5,
        interval_hours=4.0,
        available=available,
    )
    models = _models(
        capacity=300.0,
        initial=300.0,
        minimum=0.0,
        solar=500.0,
        drive=400.0,
        idle=0.0,
    )
    exact = exact_soc_path(problem, occupancy, power, T0, **models)
    greedy = greedy_soc_path(problem, occupancy, power, T0, **models)

    assert exact.reachable
    assert exact.path.cells is not None
    assert exact.path.cells.tolist() == [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [2, 0],
    ]
    assert exact.energy is not None and exact.energy.final_energy_wh == 0.0
    assert not greedy.reachable
    assert greedy.diagnostics.candidates_discarded > 0


def test_greedy_preference_is_earliest_then_equal_time_energy() -> None:
    battery = _models()["battery"]
    incumbent = _SocLabel(0, 0, 1.0, 200.0, None, None)
    earlier_lower = _SocLabel(0, 0, 0.5, 100.0, None, None)
    later_higher = _SocLabel(0, 0, 1.5, 300.0, None, None)
    equal_higher = _SocLabel(0, 0, 1.0, 201.0, None, None)

    assert _preferred(earlier_lower, incumbent, battery)
    assert not _preferred(later_higher, incumbent, battery)
    assert _preferred(equal_higher, incumbent, battery)


def test_greedy_random_cases_are_checked_against_exact(
    make_trajectory_georef,
) -> None:
    generator = np.random.default_rng(20260908)
    models = _models(
        capacity=4.0,
        initial=2.0,
        minimum=0.0,
        solar=2.0,
        drive=2.0,
        idle=1.0,
    )
    for _ in range(40):
        allowed = generator.random((5, 1, 3)) > 0.2
        allowed[0, 0, 0] = True
        sunlight = generator.integers(0, 2, size=(5, 1, 3)).astype(np.float64)
        problem, occupancy, power = _case(
            make_trajectory_georef,
            allowed,
            sunlight,
            goal=(2, 0),
        )
        exact = exact_soc_path(problem, occupancy, power, T0, **models)
        greedy = greedy_soc_path(problem, occupancy, power, T0, **models)
        if greedy.reachable:
            assert exact.reachable
            assert exact.path.arrival_time <= greedy.path.arrival_time
            assert greedy.energy is not None and greedy.energy.feasible
        if not exact.reachable:
            assert not greedy.reachable


def test_greedy_label_and_state_bounds_are_structured(
    make_trajectory_georef, monkeypatch
) -> None:
    from lunarscout.trajectory import _power_greedy

    allowed = np.ones((3, 1, 2), dtype=bool)
    sunlight = np.zeros((3, 1, 2), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef, allowed, sunlight, goal=(1, 0)
    )
    with pytest.raises(ls.PlanningError) as labels:
        greedy_soc_path(
            problem,
            occupancy,
            power,
            T0,
            max_labels=1,
            **_models(),
        )
    monkeypatch.setattr(_power_greedy, "_MAX_GREEDY_STATES", 5)
    with pytest.raises(ls.PlanningError) as states:
        greedy_soc_path(problem, occupancy, power, T0, **_models())

    assert labels.value.code == "trajectory_greedy_soc_label_limit"
    assert labels.value.details == {"label_count": 1, "maximum": 1}
    assert states.value.code == "trajectory_greedy_soc_state_limit"
    assert states.value.details == {"state_count": 6, "maximum": 5}


def test_every_greedy_reachable_result_is_replayed(
    make_trajectory_georef, monkeypatch
) -> None:
    from lunarscout.trajectory import _power_greedy

    allowed = np.ones((3, 1, 2), dtype=bool)
    sunlight = np.ones((3, 1, 2), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef, allowed, sunlight, goal=(1, 0)
    )
    original = _power_greedy.replay_soc_path
    calls = []

    def recording_replay(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_power_greedy, "replay_soc_path", recording_replay)
    result = greedy_soc_path(problem, occupancy, power, T0, **_models())
    assert result.reachable and calls == [True]
