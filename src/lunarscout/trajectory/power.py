from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import TrajectoryInputError


def _finite(name: str, value: object, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        result = np.nan
    else:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            result = np.nan
    if not np.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise TrajectoryInputError(
            f"{name} must be a finite {qualifier} number.",
            code="trajectory_invalid_power_model",
            details={"name": name, "value": value},
        )
    return result


@dataclass(frozen=True, slots=True)
class SolarPowerModel:
    """Orientation-independent post-conversion solar output model."""

    rated_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rated_power_w",
            _finite("rated_power_w", self.rated_power_w, positive=False),
        )

    def watts_in(self, sunlight_fraction: float) -> float:
        fraction = _finite("sunlight_fraction", sunlight_fraction, positive=False)
        if fraction > 1.0:
            raise TrajectoryInputError(
                "sunlight_fraction must lie in [0, 1].",
                code="trajectory_invalid_sunlight_fraction",
                details={"sunlight_fraction": sunlight_fraction},
            )
        return self.rated_power_w * fraction


@dataclass(frozen=True, slots=True)
class BatteryModel:
    """Stored-energy battery model with capacity and minimum bounds in Wh."""

    capacity_wh: float
    initial_energy_wh: float
    minimum_energy_wh: float
    charge_efficiency: float
    discharge_efficiency: float

    def __post_init__(self) -> None:
        capacity = _finite("capacity_wh", self.capacity_wh, positive=True)
        initial = _finite("initial_energy_wh", self.initial_energy_wh, positive=False)
        minimum = _finite("minimum_energy_wh", self.minimum_energy_wh, positive=False)
        charge = _finite("charge_efficiency", self.charge_efficiency, positive=True)
        discharge = _finite(
            "discharge_efficiency", self.discharge_efficiency, positive=True
        )
        if initial > capacity or minimum > initial:
            raise TrajectoryInputError(
                "Battery energies must satisfy 0 <= minimum <= initial <= capacity.",
                code="trajectory_invalid_battery_model",
                details={
                    "capacity_wh": capacity,
                    "initial_energy_wh": initial,
                    "minimum_energy_wh": minimum,
                },
            )
        if charge != 1.0 or discharge != 1.0:
            raise TrajectoryInputError(
                "The initial battery model requires 100% charge and discharge efficiency.",
                code="trajectory_unsupported_battery_efficiency",
                details={
                    "charge_efficiency": charge,
                    "discharge_efficiency": discharge,
                },
            )
        object.__setattr__(self, "capacity_wh", capacity)
        object.__setattr__(self, "initial_energy_wh", initial)
        object.__setattr__(self, "minimum_energy_wh", minimum)
        object.__setattr__(self, "charge_efficiency", charge)
        object.__setattr__(self, "discharge_efficiency", discharge)


@dataclass(frozen=True, slots=True)
class RoverPowerModel:
    """Constant electrical loads for the initial drive and idle modes."""

    drive_power_w: float
    idle_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "drive_power_w",
            _finite("drive_power_w", self.drive_power_w, positive=False),
        )
        object.__setattr__(
            self,
            "idle_power_w",
            _finite("idle_power_w", self.idle_power_w, positive=False),
        )
