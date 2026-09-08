from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.trajectory._dynamic_reference import (
    DynamicOccupancyTimeline,
    ExactDynamicPathResult,
)
from lunarscout.trajectory._power_accounting import PiecewiseSunlightTimeline
from lunarscout.trajectory._power_reference import (
    _SocLabel,
    _dominates,
    exact_soc_path,
    replay_soc_path,
)
from lunarscout.trajectory._validation import prepare_static_problem


T0 = datetime(2040, 1, 1, tzinfo=timezone.utc)


def _case(
    make_trajectory_georef,
    allowed,
    sunlight,
    *,
    start=(0, 0),
    goal=None,
    edge_hours=1.0,
    available=None,
):
    intervals, height, width = allowed.shape
    grid = make_trajectory_georef(width=width, height=height)
    if goal is None:
        goal = (width - 1, height - 1)
    if available is None:
        available = np.any(allowed, axis=0)
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
        T0 + timedelta(hours=index) for index in range(intervals + 1)
    )
    occupancy = DynamicOccupancyTimeline(boundaries, allowed, grid)
    power = PiecewiseSunlightTimeline(boundaries, sunlight, grid)
    return problem, occupancy, power


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


def test_exact_soc_returns_replayed_energy_timeline(make_trajectory_georef) -> None:
    allowed = np.ones((4, 1, 3), dtype=bool)
    sunlight = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    problem, occupancy, power = _case(
        make_trajectory_georef,
        allowed,
        sunlight,
        goal=(2, 0),
        edge_hours=0.5,
    )
    models = _models(initial=250.0, drive=200.0, idle=50.0)
    result = exact_soc_path(problem, occupancy, power, T0, **models)

    assert result.reachable
    assert result.path.cells is not None
    assert result.path.cells.tolist() == [[0, 0], [1, 0], [2, 0]]
    assert result.path.arrival_time == T0 + timedelta(hours=1)
    assert result.energy is not None
    assert result.energy.final_energy_wh == 100.0
    assert [segment.mode for segment in result.energy.segments] == [
        "drive",
        "drive",
    ]
    replayed = replay_soc_path(
        problem, occupancy, power, T0, result.path, **models
    )
    assert replayed == result.energy


def test_non_round_edge_duration_survives_datetime_replay(
    make_trajectory_georef,
) -> None:
    allowed = np.ones((3, 1, 2), dtype=bool)
    sunlight = np.ones((3, 1, 2), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef,
        allowed,
        sunlight,
        goal=(1, 0),
        edge_hours=10.0 / 13.0,
    )
    result = exact_soc_path(
        problem, occupancy, power, T0, **_models(solar=100.0, drive=100.0)
    )
    assert result.reachable
    assert result.path.arrival_time == T0 + timedelta(hours=10.0 / 13.0)


def test_waiting_charges_before_an_otherwise_infeasible_drive(
    make_trajectory_georef,
) -> None:
    allowed = np.ones((3, 1, 2), dtype=bool)
    sunlight = np.asarray([[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
    problem, occupancy, power = _case(
        make_trajectory_georef,
        allowed,
        sunlight,
        goal=(1, 0),
        edge_hours=0.5,
    )
    models = _models(
        capacity=300.0,
        initial=100.0,
        minimum=60.0,
        solar=300.0,
        drive=400.0,
        idle=100.0,
    )
    result = exact_soc_path(problem, occupancy, power, T0, **models)

    assert result.reachable
    assert result.path.departure_times == (T0 + timedelta(hours=1),)
    assert result.path.arrival_time == T0 + timedelta(hours=1.5)
    assert result.energy is not None
    assert [segment.mode for segment in result.energy.segments] == [
        "idle",
        "drive",
    ]
    assert result.energy.final_energy_wh == 100.0


def test_waiting_in_darkness_consumes_energy(make_trajectory_georef) -> None:
    allowed = np.ones((3, 1, 2), dtype=bool)
    allowed[0, 0, 1] = False
    sunlight = np.zeros((3, 1, 2), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef,
        allowed,
        sunlight,
        goal=(1, 0),
        edge_hours=0.5,
    )
    models = _models(initial=200.0, drive=100.0, idle=50.0)
    result = exact_soc_path(problem, occupancy, power, T0, **models)

    assert result.reachable
    assert result.path.departure_times == (T0 + timedelta(hours=1),)
    assert result.energy is not None
    assert result.energy.final_energy_wh == 100.0
    assert result.energy.segments[0].mode == "idle"
    assert result.energy.segments[0].consumed_wh == 50.0


def test_earlier_lower_and_later_higher_energy_labels_survive_for_high_load(
    make_trajectory_georef,
) -> None:
    allowed = np.ones((5, 2, 4), dtype=bool)
    available = np.asarray(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    )
    sunlight = np.zeros((5, 2, 4), dtype=np.float64)
    sunlight[:, 1, :3] = 1.0
    problem, occupancy, power = _case(
        make_trajectory_georef,
        allowed,
        sunlight,
        goal=(3, 0),
        edge_hours=0.5,
        available=available,
    )
    models = _models(
        capacity=300.0,
        initial=200.0,
        minimum=0.0,
        solar=100.0,
        drive=100.0,
        idle=0.0,
    )
    # Make only the final source cell expensive by using a steep uphill edge.
    elevation = np.zeros((2, 4), dtype=np.float64)
    elevation[0, 3] = 10.0
    slip = ls.trajectory.SlipFunction(
        (-1.0, 0.0, 1.0),
        (1.0, 1.0, 3.0),
        extrapolation="constant",
    )
    problem = prepare_static_problem(
        available,
        problem.georef,
        (0, 0),
        goal=(3, 0),
        elevation=elevation,
        model=ls.trajectory.StaticTravelModel(
            speed_m_per_h=20.0,
            include_diagonals=False,
            slip=slip,
        ),
    )
    result = exact_soc_path(problem, occupancy, power, T0, **models)

    assert result.reachable
    assert result.path.cells is not None
    assert result.path.cells.tolist() == [
        [0, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [2, 0],
        [3, 0],
    ]
    assert result.energy is not None
    assert result.energy.final_energy_wh == 0.0
    assert result.diagnostics.labels_created > len(result.path.cells)


def test_catch_up_dominance_accounts_for_idle_charging(
    make_trajectory_georef,
) -> None:
    allowed = np.ones((3, 1, 1), dtype=bool)
    sunny = np.ones((3, 1, 1), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef,
        allowed,
        sunny,
        start=(0, 0),
        goal=(0, 0),
    )
    models = _models(capacity=300.0, initial=100.0, solar=100.0, idle=0.0)
    early = _SocLabel(0, 0, 0.0, 100.0, None, None)
    later = _SocLabel(0, 0, 1.0, 150.0, None, None)
    assert _dominates(
        early,
        later,
        occupancy=occupancy,
        sunlight=power,
        **models,
    )

    dark = PiecewiseSunlightTimeline(
        power.boundaries,
        np.zeros_like(power.fractions),
        problem.georef,
    )
    assert not _dominates(
        early,
        later,
        occupancy=occupancy,
        sunlight=dark,
        **models,
    )


def _exhaustive_integer_hour_arrival(
    allowed: np.ndarray,
    sunlight: np.ndarray,
    *,
    initial: float,
    capacity: float,
    minimum: float,
    solar_w: float,
    drive_w: float,
    idle_w: float,
) -> int | None:
    """Independent time-expanded enumeration without label dominance."""

    interval_count, _height, width = allowed.shape
    states = {(0, initial)}
    for tick in range(interval_count):
        if any(x == width - 1 for x, _energy in states):
            return tick
        if tick + 1 >= interval_count:
            break
        following: set[tuple[int, float]] = set()
        for x, energy in states:
            if allowed[tick, 0, x] and allowed[tick + 1, 0, x]:
                waited = min(
                    capacity,
                    energy + solar_w * sunlight[tick, 0, x] - idle_w,
                )
                if waited >= minimum:
                    following.add((x, waited))
            for destination in (x - 1, x + 1):
                if destination < 0 or destination >= width:
                    continue
                if not (
                    allowed[tick, 0, x]
                    and allowed[tick, 0, destination]
                    and allowed[tick + 1, 0, destination]
                ):
                    continue
                moved = min(
                    capacity,
                    energy + solar_w * sunlight[tick, 0, x] - drive_w,
                )
                if moved >= minimum:
                    following.add((destination, moved))
        states = following
    return None


def test_exact_soc_matches_independent_exhaustive_state_enumeration(
    make_trajectory_georef,
) -> None:
    generator = np.random.default_rng(20260907)
    model_values = {
        "capacity": 4.0,
        "initial": 2.0,
        "minimum": 0.0,
        "solar": 2.0,
        "drive": 2.0,
        "idle": 1.0,
    }
    for _ in range(40):
        allowed = generator.random((5, 1, 3)) > 0.2
        allowed[0, 0, 0] = True
        sunlight = generator.integers(0, 2, size=(5, 1, 3)).astype(np.float64)
        problem, occupancy, power = _case(
            make_trajectory_georef,
            allowed,
            sunlight,
            goal=(2, 0),
            edge_hours=1.0,
            available=np.ones((1, 3), dtype=bool),
        )
        models = _models(**model_values)
        result = exact_soc_path(problem, occupancy, power, T0, **models)
        expected = _exhaustive_integer_hour_arrival(
            allowed,
            sunlight,
            initial=model_values["initial"],
            capacity=model_values["capacity"],
            minimum=model_values["minimum"],
            solar_w=model_values["solar"],
            drive_w=model_values["drive"],
            idle_w=model_values["idle"],
        )
        if expected is None:
            assert not result.reachable
        else:
            assert result.reachable
            assert result.path.arrival_time == T0 + timedelta(hours=expected)


def test_exact_soc_label_bound_is_structured(make_trajectory_georef) -> None:
    allowed = np.ones((3, 1, 2), dtype=bool)
    sunlight = np.zeros((3, 1, 2), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef, allowed, sunlight, goal=(1, 0)
    )
    with pytest.raises(ls.PlanningError) as error:
        exact_soc_path(
            problem,
            occupancy,
            power,
            T0,
            max_labels=1,
            **_models(),
        )
    assert error.value.code == "trajectory_exact_soc_label_limit"
    assert error.value.details == {"label_count": 1, "maximum": 1}


def test_every_reachable_result_is_replayed(
    make_trajectory_georef, monkeypatch
) -> None:
    from lunarscout.trajectory import _power_reference

    allowed = np.ones((3, 1, 2), dtype=bool)
    sunlight = np.ones((3, 1, 2), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef, allowed, sunlight, goal=(1, 0)
    )
    original = _power_reference.replay_soc_path
    calls = []

    def recording_replay(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_power_reference, "replay_soc_path", recording_replay)
    result = exact_soc_path(problem, occupancy, power, T0, **_models())
    assert result.reachable and calls == [True]


def test_exact_soc_validates_timeline_and_model_contracts(
    make_trajectory_georef,
) -> None:
    allowed = np.ones((2, 1, 2), dtype=bool)
    sunlight = np.ones((2, 1, 2), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef, allowed, sunlight, goal=(1, 0)
    )
    shifted = PiecewiseSunlightTimeline(
        tuple(value + timedelta(minutes=1) for value in power.boundaries),
        sunlight,
        problem.georef,
    )
    with pytest.raises(ls.TrajectoryInputError) as timeline:
        exact_soc_path(problem, occupancy, shifted, T0, **_models())
    with pytest.raises(ls.TrajectoryInputError) as bound:
        exact_soc_path(
            problem, occupancy, power, T0, max_labels=True, **_models()
        )
    with pytest.raises(ls.TrajectoryInputError) as model:
        exact_soc_path(
            problem,
            occupancy,
            power,
            T0,
            solar=None,
            battery=_models()["battery"],
            rover=_models()["rover"],
        )
    assert timeline.value.code == "trajectory_soc_timeline_mismatch"
    assert bound.value.code == "trajectory_invalid_soc_label_limit"
    assert model.value.code == "trajectory_invalid_power_model"


def test_replay_rejects_a_corrupt_movement_duration(make_trajectory_georef) -> None:
    allowed = np.ones((3, 1, 2), dtype=bool)
    sunlight = np.ones((3, 1, 2), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef,
        allowed,
        sunlight,
        goal=(1, 0),
        edge_hours=1.0,
    )
    bad = ExactDynamicPathResult(
        True,
        T0 + timedelta(hours=0.5),
        np.asarray([[0, 0], [1, 0]], dtype=np.int64),
        (T0, T0 + timedelta(hours=0.5)),
        (T0,),
    )
    with pytest.raises(ls.PlanningError) as error:
        replay_soc_path(
            problem, occupancy, power, T0, bad, **_models()
        )
    assert error.value.code == "trajectory_invalid_soc_replay"


def test_start_equals_goal_and_unreachable_results(make_trajectory_georef) -> None:
    one = np.ones((2, 1, 1), dtype=bool)
    problem, occupancy, power = _case(
        make_trajectory_georef, one, one.astype(np.float64), goal=(0, 0)
    )
    same = exact_soc_path(problem, occupancy, power, T0, **_models())
    assert same.reachable
    assert same.energy is not None
    assert same.energy.segments == ()
    assert same.energy.final_energy_wh == 300.0

    allowed = np.ones((2, 1, 2), dtype=bool)
    dark = np.zeros((2, 1, 2), dtype=np.float64)
    problem, occupancy, power = _case(
        make_trajectory_georef, allowed, dark, goal=(1, 0)
    )
    unreachable = exact_soc_path(
        problem,
        occupancy,
        power,
        T0,
        **_models(initial=50.0, minimum=40.0, drive=100.0),
    )
    assert not unreachable.reachable
    assert unreachable.energy is None
