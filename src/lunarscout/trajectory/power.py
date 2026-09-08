from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias

import numpy as np

from ..errors import TrajectoryInputError
from ._time_contract import as_utc


PowerMode: TypeAlias = Literal["drive", "idle"]


def _finite(
    name: str,
    value: object,
    *,
    positive: bool,
    code: str = "trajectory_invalid_power_model",
) -> float:
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
            code=code,
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


@dataclass(frozen=True, slots=True)
class EnergySegment:
    """Energy accounting for one constant-sunlight temporal segment."""

    mode: PowerMode
    start_time: datetime
    stop_time: datetime
    cell: tuple[int, int]
    sunlight_fraction: float
    start_energy_wh: float
    end_energy_wh: float
    generated_wh: float
    consumed_wh: float
    discarded_wh: float

    def __post_init__(self) -> None:
        if self.mode not in ("drive", "idle"):
            raise TrajectoryInputError(
                "EnergySegment mode must be 'drive' or 'idle'.",
                code="trajectory_invalid_energy_segment",
            )
        start = as_utc(self.start_time, name="start_time")
        stop = as_utc(self.stop_time, name="stop_time")
        if stop <= start:
            raise TrajectoryInputError(
                "EnergySegment stop_time must be after start_time.",
                code="trajectory_invalid_energy_segment",
            )
        if (
            not isinstance(self.cell, tuple)
            or len(self.cell) != 2
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in self.cell
            )
        ):
            raise TrajectoryInputError(
                "EnergySegment cell must be an integer (x, y) tuple.",
                code="trajectory_invalid_energy_segment",
            )
        values = {
            name: _finite(
                name,
                getattr(self, name),
                positive=False,
                code="trajectory_invalid_energy_segment",
            )
            for name in (
                "sunlight_fraction",
                "start_energy_wh",
                "end_energy_wh",
                "generated_wh",
                "consumed_wh",
                "discarded_wh",
            )
        }
        if values["sunlight_fraction"] > 1.0:
            raise TrajectoryInputError(
                "EnergySegment sunlight_fraction must lie in [0, 1].",
                code="trajectory_invalid_energy_segment",
            )
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "stop_time", stop)
        object.__setattr__(self, "cell", (int(self.cell[0]), int(self.cell[1])))
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class EnergyTimelineResult:
    """Segments evaluated through completion or the first infeasible segment."""

    feasible: bool
    initial_energy_wh: float
    final_energy_wh: float
    segments: tuple[EnergySegment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, (bool, np.bool_)):
            raise TrajectoryInputError(
                "EnergyTimelineResult feasible must be Boolean.",
                code="trajectory_invalid_energy_timeline_result",
            )
        initial = _finite(
            "initial_energy_wh",
            self.initial_energy_wh,
            positive=False,
            code="trajectory_invalid_energy_timeline_result",
        )
        final = _finite(
            "final_energy_wh",
            self.final_energy_wh,
            positive=False,
            code="trajectory_invalid_energy_timeline_result",
        )
        try:
            segments = tuple(self.segments)
        except TypeError as exc:
            raise TrajectoryInputError(
                "EnergyTimelineResult segments must be iterable.",
                code="trajectory_invalid_energy_timeline_result",
            ) from exc
        if any(not isinstance(segment, EnergySegment) for segment in segments):
            raise TrajectoryInputError(
                "EnergyTimelineResult segments must be EnergySegment values.",
                code="trajectory_invalid_energy_timeline_result",
            )
        previous_energy = initial
        previous_stop = None
        for index, segment in enumerate(segments):
            if segment.start_energy_wh != previous_energy or (
                previous_stop is not None and segment.start_time != previous_stop
            ):
                raise TrajectoryInputError(
                    "EnergyTimelineResult segments must form a continuous timeline.",
                    code="trajectory_invalid_energy_timeline_result",
                    details={"segment": index},
                )
            previous_energy = segment.end_energy_wh
            previous_stop = segment.stop_time
        if final != previous_energy:
            raise TrajectoryInputError(
                "EnergyTimelineResult final energy must match its last segment.",
                code="trajectory_invalid_energy_timeline_result",
            )
        object.__setattr__(self, "feasible", bool(self.feasible))
        object.__setattr__(self, "initial_energy_wh", initial)
        object.__setattr__(self, "final_energy_wh", final)
        object.__setattr__(self, "segments", segments)

    @property
    def generated_wh(self) -> float:
        return sum((segment.generated_wh for segment in self.segments), 0.0)

    @property
    def consumed_wh(self) -> float:
        return sum((segment.consumed_wh for segment in self.segments), 0.0)

    @property
    def discarded_wh(self) -> float:
        return sum((segment.discarded_wh for segment in self.segments), 0.0)
