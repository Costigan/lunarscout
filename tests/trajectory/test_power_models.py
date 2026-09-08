from __future__ import annotations

import math

import pytest

import lunarscout as ls
from lunarscout.trajectory._power_accounting import integrate_energy


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
