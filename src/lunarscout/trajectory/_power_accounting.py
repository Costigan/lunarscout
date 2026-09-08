from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..errors import TrajectoryInputError
from .power import BatteryModel, RoverPowerModel, SolarPowerModel, _finite


PowerMode = Literal["drive", "idle"]


@dataclass(frozen=True, slots=True)
class EnergyTransition:
    energy_wh: float
    generated_wh: float
    consumed_wh: float
    discarded_wh: float
    feasible: bool


def integrate_energy(
    energy_wh: float,
    duration_hours: float,
    sunlight_fraction: float,
    mode: PowerMode,
    *,
    solar: SolarPowerModel,
    battery: BatteryModel,
    rover: RoverPowerModel,
) -> EnergyTransition:
    """Integrate one constant-signal segment under the Phase 4A model."""

    energy = _finite("energy_wh", energy_wh, positive=False)
    duration = _finite("duration_hours", duration_hours, positive=False)
    if energy > battery.capacity_wh:
        raise TrajectoryInputError(
            "energy_wh cannot exceed battery capacity.",
            code="trajectory_invalid_battery_state",
            details={"energy_wh": energy, "capacity_wh": battery.capacity_wh},
        )
    if mode == "drive":
        load_w = rover.drive_power_w
    elif mode == "idle":
        load_w = rover.idle_power_w
    else:
        raise TrajectoryInputError(
            "mode must be 'drive' or 'idle'.",
            code="trajectory_invalid_power_mode",
            details={"mode": mode},
        )
    generated = solar.watts_in(sunlight_fraction) * duration
    consumed = load_w * duration
    if generated >= consumed:
        unconstrained = energy + (generated - consumed) * battery.charge_efficiency
    else:
        unconstrained = energy - (consumed - generated) / battery.discharge_efficiency
    stored = max(0.0, min(unconstrained, battery.capacity_wh))
    discarded = max(0.0, unconstrained - battery.capacity_wh)
    tolerance = 1.0e-12 * max(1.0, battery.capacity_wh)
    feasible = unconstrained >= battery.minimum_energy_wh - tolerance
    if feasible and stored < battery.minimum_energy_wh:
        stored = battery.minimum_energy_wh
    return EnergyTransition(
        energy_wh=float(stored),
        generated_wh=float(generated),
        consumed_wh=float(consumed),
        discarded_wh=float(discarded),
        feasible=feasible,
    )
