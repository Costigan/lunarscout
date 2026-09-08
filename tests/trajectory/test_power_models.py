from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.trajectory._power_accounting import (
    PiecewiseSunlightTimeline,
    integrate_energy,
    integrate_timeline_operation,
)


def _models():
    return (
        ls.trajectory.SolarPowerModel(rated_power_w=120.0),
        ls.trajectory.BatteryModel(
            capacity_wh=500.0,
            initial_energy_wh=300.0,
            minimum_energy_wh=100.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
        ),
        ls.trajectory.RoverPowerModel(drive_power_w=200.0, idle_power_w=40.0),
    )


def _timeline(make_trajectory_georef):
    time0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
    fractions = np.asarray(
        [
            [[0.0, 1.0]],
            [[1.0, 0.0]],
            [[0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    return PiecewiseSunlightTimeline(
        tuple(time0 + timedelta(hours=index) for index in range(4)),
        fractions,
        make_trajectory_georef(width=2, height=1),
    )


def test_orientation_independent_solar_output() -> None:
    solar, _, _ = _models()
    assert solar.watts_in(0.0) == 0.0
    assert solar.watts_in(0.25) == 30.0
    assert solar.watts_in(1.0) == 120.0


def test_drive_discharge_and_idle_charge() -> None:
    solar, battery, rover = _models()
    drive = integrate_energy(
        300.0, 0.5, 0.5, "drive", solar=solar, battery=battery, rover=rover
    )
    idle = integrate_energy(
        drive.energy_wh, 2.0, 1.0, "idle", solar=solar, battery=battery, rover=rover
    )
    assert drive.energy_wh == 230.0
    assert drive.generated_wh == 30.0
    assert drive.consumed_wh == 100.0
    assert drive.feasible
    assert idle.energy_wh == 390.0
    assert idle.feasible


def test_capacity_clipping_discards_excess_energy() -> None:
    solar, battery, rover = _models()
    result = integrate_energy(
        480.0, 1.0, 1.0, "idle", solar=solar, battery=battery, rover=rover
    )
    assert result.energy_wh == 500.0
    assert result.discarded_wh == 60.0


def test_depleted_infeasible_state_is_clipped_at_zero() -> None:
    solar, battery, rover = _models()
    result = integrate_energy(
        10.0, 1.0, 0.0, "drive", solar=solar, battery=battery, rover=rover
    )
    assert result.energy_wh == 0.0
    assert not result.feasible


def test_minimum_energy_equality_and_tolerance() -> None:
    solar, battery, rover = _models()
    exact = integrate_energy(
        200.0, 0.5, 0.0, "drive", solar=solar, battery=battery, rover=rover
    )
    near = integrate_energy(
        200.0 - 1e-11,
        0.5,
        0.0,
        "drive",
        solar=solar,
        battery=battery,
        rover=rover,
    )
    below = integrate_energy(
        199.0, 0.5, 0.0, "drive", solar=solar, battery=battery, rover=rover
    )
    assert exact.energy_wh == 100.0 and exact.feasible
    assert near.energy_wh == 100.0 and near.feasible
    assert not below.feasible and below.energy_wh == 99.0


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan, True])
def test_power_models_reject_invalid_numbers(bad) -> None:
    with pytest.raises(ls.TrajectoryInputError):
        ls.trajectory.SolarPowerModel(bad)
    with pytest.raises(ls.TrajectoryInputError):
        ls.trajectory.RoverPowerModel(bad, 1.0)


def test_battery_requires_explicit_ideal_efficiencies() -> None:
    with pytest.raises(TypeError):
        ls.trajectory.BatteryModel(500.0, 300.0, 100.0)
    with pytest.raises(ls.TrajectoryInputError) as unsupported:
        ls.trajectory.BatteryModel(500.0, 300.0, 100.0, 0.9, 1.0)
    assert unsupported.value.code == "trajectory_unsupported_battery_efficiency"


def test_invalid_fraction_mode_and_state_are_structured() -> None:
    solar, battery, rover = _models()
    with pytest.raises(ls.TrajectoryInputError) as fraction:
        solar.watts_in(1.01)
    with pytest.raises(ls.TrajectoryInputError) as mode:
        integrate_energy(
            300.0, 1.0, 1.0, "science", solar=solar, battery=battery, rover=rover
        )
    with pytest.raises(ls.TrajectoryInputError) as state:
        integrate_energy(
            501.0, 1.0, 1.0, "idle", solar=solar, battery=battery, rover=rover
        )
    assert fraction.value.code == "trajectory_invalid_sunlight_fraction"
    assert mode.value.code == "trajectory_invalid_power_mode"
    assert state.value.code == "trajectory_invalid_battery_state"


def test_operation_splits_at_steps_and_samples_source_cell(
    make_trajectory_georef,
) -> None:
    timeline = _timeline(make_trajectory_georef)
    solar = ls.trajectory.SolarPowerModel(100.0)
    battery = ls.trajectory.BatteryModel(500.0, 300.0, 100.0, 1.0, 1.0)
    rover = ls.trajectory.RoverPowerModel(100.0, 20.0)
    result = integrate_timeline_operation(
        300.0,
        timeline.boundaries[0] + timedelta(minutes=30),
        timeline.boundaries[2] + timedelta(minutes=30),
        (0, 0),
        "drive",
        timeline=timeline,
        solar=solar,
        battery=battery,
        rover=rover,
    )

    assert result.feasible
    assert result.final_energy_wh == 225.0
    assert [segment.sunlight_fraction for segment in result.segments] == [
        0.0,
        1.0,
        0.5,
    ]
    assert [segment.end_energy_wh for segment in result.segments] == [
        250.0,
        250.0,
        225.0,
    ]
    assert result.generated_wh == 125.0
    assert result.consumed_wh == 200.0
    assert result.discarded_wh == 0.0
    assert result.segments[0].cell == (0, 0)
    assert result.segments[0].stop_time == timeline.boundaries[1]
    assert result.segments[1].start_time == timeline.boundaries[1]
    # The other cell has the opposite signal. Its value is not blended into a
    # movement operation whose source cell is (0, 0).
    assert timeline.fractions[:, 0, 1].tolist() == [1.0, 0.0, 1.0]


def test_wait_uses_occupied_cell_and_boundary_start_uses_new_signal(
    make_trajectory_georef,
) -> None:
    timeline = _timeline(make_trajectory_georef)
    solar = ls.trajectory.SolarPowerModel(100.0)
    battery = ls.trajectory.BatteryModel(500.0, 300.0, 100.0, 1.0, 1.0)
    rover = ls.trajectory.RoverPowerModel(100.0, 20.0)
    result = integrate_timeline_operation(
        300.0,
        timeline.boundaries[1],
        timeline.boundaries[2],
        (0, 0),
        "idle",
        timeline=timeline,
        solar=solar,
        battery=battery,
        rover=rover,
    )

    assert len(result.segments) == 1
    assert result.segments[0].sunlight_fraction == 1.0
    assert result.segments[0].mode == "idle"
    assert result.final_energy_wh == 380.0


def test_timeline_operation_stops_at_first_infeasible_segment(
    make_trajectory_georef,
) -> None:
    timeline = _timeline(make_trajectory_georef)
    result = integrate_timeline_operation(
        150.0,
        timeline.boundaries[0],
        timeline.boundaries[2],
        (0, 0),
        "drive",
        timeline=timeline,
        solar=ls.trajectory.SolarPowerModel(0.0),
        battery=ls.trajectory.BatteryModel(500.0, 300.0, 100.0, 1.0, 1.0),
        rover=ls.trajectory.RoverPowerModel(200.0, 20.0),
    )

    assert not result.feasible
    assert len(result.segments) == 1
    assert result.final_energy_wh == 0.0
    assert result.segments[-1].stop_time == timeline.boundaries[1]


def test_zero_duration_operation_is_empty_but_still_validated(
    make_trajectory_georef,
) -> None:
    timeline = _timeline(make_trajectory_georef)
    solar, battery, rover = _models()
    result = integrate_timeline_operation(
        100.0,
        timeline.boundaries[-1],
        timeline.boundaries[-1],
        (0, 0),
        "idle",
        timeline=timeline,
        solar=solar,
        battery=battery,
        rover=rover,
    )
    assert result.feasible and result.segments == ()

    below_minimum = integrate_timeline_operation(
        99.0,
        timeline.boundaries[0],
        timeline.boundaries[0],
        (0, 0),
        "idle",
        timeline=timeline,
        solar=solar,
        battery=battery,
        rover=rover,
    )
    assert not below_minimum.feasible and below_minimum.segments == ()

    with pytest.raises(ls.TrajectoryInputError) as mode:
        integrate_timeline_operation(
            100.0,
            timeline.boundaries[0],
            timeline.boundaries[0],
            (0, 0),
            "science",
            timeline=timeline,
            solar=solar,
            battery=battery,
            rover=rover,
        )
    with pytest.raises(ls.TrajectoryInputError) as state:
        integrate_timeline_operation(
            501.0,
            timeline.boundaries[0],
            timeline.boundaries[0],
            (0, 0),
            "idle",
            timeline=timeline,
            solar=solar,
            battery=battery,
            rover=rover,
        )
    assert mode.value.code == "trajectory_invalid_power_mode"
    assert state.value.code == "trajectory_invalid_battery_state"


@pytest.mark.parametrize(
    "fractions",
    [
        np.ones((2, 1, 2), dtype=np.float64),
        np.ones((3, 1, 2), dtype=bool),
        np.ones((3, 1, 2), dtype=np.complex128),
        np.full((3, 1, 2), np.nan),
        np.full((3, 1, 2), 1.01),
        [[[0.0], [1.0, 0.0]]],
    ],
)
def test_sunlight_timeline_rejects_invalid_arrays(
    make_trajectory_georef, fractions
) -> None:
    time0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
    boundaries = tuple(time0 + timedelta(hours=index) for index in range(4))
    with pytest.raises(ls.TrajectoryInputError) as error:
        PiecewiseSunlightTimeline(
            boundaries,
            fractions,
            make_trajectory_georef(width=2, height=1),
        )
    assert error.value.code == "trajectory_invalid_power_timeline"


def test_sunlight_timeline_owns_read_only_float64_values(
    make_trajectory_georef,
) -> None:
    time0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
    source = np.zeros((1, 1, 1), dtype=np.float32)
    timeline = PiecewiseSunlightTimeline(
        (time0, time0 + timedelta(hours=1)),
        source,
        make_trajectory_georef(width=1, height=1),
    )
    source[0, 0, 0] = 1.0
    assert timeline.fractions.dtype == np.float64
    assert timeline.fractions[0, 0, 0] == 0.0
    assert not timeline.fractions.flags.writeable
    with pytest.raises(ValueError):
        timeline.fractions[0, 0, 0] = 1.0


def test_operation_rejects_time_cell_and_order_failures(
    make_trajectory_georef,
) -> None:
    timeline = _timeline(make_trajectory_georef)
    solar, battery, rover = _models()
    common = {
        "timeline": timeline,
        "solar": solar,
        "battery": battery,
        "rover": rover,
    }

    with pytest.raises(ls.TrajectoryInputError) as cell:
        integrate_timeline_operation(
            300.0,
            timeline.boundaries[0],
            timeline.boundaries[1],
            (2, 0),
            "idle",
            **common,
        )
    with pytest.raises(ls.TrajectoryInputError) as order:
        integrate_timeline_operation(
            300.0,
            timeline.boundaries[1],
            timeline.boundaries[0],
            (0, 0),
            "idle",
            **common,
        )
    with pytest.raises(ls.TrajectoryInputError) as range_error:
        integrate_timeline_operation(
            300.0,
            timeline.boundaries[0] - timedelta(seconds=1),
            timeline.boundaries[0],
            (0, 0),
            "idle",
            **common,
        )
    assert cell.value.code == "trajectory_invalid_power_cell"
    assert order.value.code == "trajectory_power_time_out_of_range"
    assert range_error.value.code == "trajectory_power_time_out_of_range"


def test_public_energy_records_normalize_and_validate_timeline() -> None:
    offset = timezone(timedelta(hours=-8))
    start = datetime(2030, 1, 1, tzinfo=offset)
    segment = ls.trajectory.EnergySegment(
        "drive",
        start,
        start + timedelta(hours=1),
        (np.int64(1), np.int64(2)),
        0.5,
        300.0,
        250.0,
        50.0,
        100.0,
        0.0,
    )
    source = [segment]
    result = ls.trajectory.EnergyTimelineResult(True, 300.0, 250.0, source)
    source.clear()

    assert segment.start_time.tzinfo is timezone.utc
    assert segment.cell == (1, 2)
    assert result.segments == (segment,)
    assert result.generated_wh == 50.0
    with pytest.raises(FrozenInstanceError):
        segment.mode = "idle"

    discontinuous = ls.trajectory.EnergySegment(
        "idle",
        segment.stop_time + timedelta(minutes=1),
        segment.stop_time + timedelta(hours=1),
        (1, 2),
        0.0,
        250.0,
        230.0,
        0.0,
        20.0,
        0.0,
    )
    with pytest.raises(ls.TrajectoryInputError) as continuity:
        ls.trajectory.EnergyTimelineResult(
            True, 300.0, 230.0, (segment, discontinuous)
        )
    with pytest.raises(ls.TrajectoryInputError) as final:
        ls.trajectory.EnergyTimelineResult(True, 300.0, 249.0, (segment,))
    assert continuity.value.code == "trajectory_invalid_energy_timeline_result"
    assert final.value.code == "trajectory_invalid_energy_timeline_result"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "science"},
        {"mode": []},
        {"sunlight_fraction": math.nan},
        {"sunlight_fraction": 1.1},
        {"cell": (True, 0)},
        {"stop_time": datetime(2030, 1, 1, tzinfo=timezone.utc)},
    ],
)
def test_public_energy_segment_rejects_invalid_fields(kwargs) -> None:
    values = {
        "mode": "idle",
        "start_time": datetime(2030, 1, 1, tzinfo=timezone.utc),
        "stop_time": datetime(2030, 1, 1, 1, tzinfo=timezone.utc),
        "cell": (0, 0),
        "sunlight_fraction": 0.5,
        "start_energy_wh": 100.0,
        "end_energy_wh": 100.0,
        "generated_wh": 10.0,
        "consumed_wh": 10.0,
        "discarded_wh": 0.0,
    }
    values.update(kwargs)
    with pytest.raises(ls.TrajectoryInputError) as error:
        ls.trajectory.EnergySegment(**values)
    assert error.value.code == "trajectory_invalid_energy_segment"
