from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from ..errors import TrajectoryInputError
from ..georeference import GeoReference
from ._time_contract import IntervalTimeAxis
from .power import (
    BatteryModel,
    EnergySegment,
    EnergyTimelineResult,
    PowerMode,
    RoverPowerModel,
    SolarPowerModel,
    _finite,
)


@dataclass(frozen=True, slots=True)
class EnergyTransition:
    energy_wh: float
    generated_wh: float
    consumed_wh: float
    discarded_wh: float
    feasible: bool


@dataclass(frozen=True, slots=True, eq=False)
class PiecewiseSunlightTimeline:
    boundaries: tuple[datetime, ...]
    fractions: NDArray[np.float64]
    georef: GeoReference
    _axis: IntervalTimeAxis = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.georef, GeoReference):
            raise TrajectoryInputError(
                "Power sunlight timeline requires a GeoReference.",
                code="trajectory_invalid_power_timeline",
            )
        axis = IntervalTimeAxis(self.boundaries)
        try:
            values = np.asarray(self.fractions)
        except (TypeError, ValueError) as exc:
            raise TrajectoryInputError(
                "Power sunlight fractions must be a rectangular numeric array.",
                code="trajectory_invalid_power_timeline",
            ) from exc
        expected = (axis.interval_count, self.georef.height, self.georef.width)
        if (
            values.shape != expected
            or np.issubdtype(values.dtype, np.bool_)
            or np.issubdtype(values.dtype, np.complexfloating)
            or not np.issubdtype(values.dtype, np.number)
        ):
            raise TrajectoryInputError(
                "Power sunlight fractions must be a real (interval, y, x) array.",
                code="trajectory_invalid_power_timeline",
                details={"shape": list(values.shape), "expected": list(expected)},
            )
        fractions = np.array(values, dtype=np.float64, copy=True)
        if np.any(~np.isfinite(fractions)) or np.any(
            (fractions < 0.0) | (fractions > 1.0)
        ):
            raise TrajectoryInputError(
                "Power sunlight fractions must be finite and lie in [0, 1].",
                code="trajectory_invalid_power_timeline",
            )
        fractions.flags.writeable = False
        object.__setattr__(self, "boundaries", axis.boundaries)
        object.__setattr__(self, "fractions", fractions)
        object.__setattr__(self, "_axis", axis)


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


def integrate_timeline_operation(
    energy_wh: float,
    start_time: datetime,
    stop_time: datetime,
    cell: tuple[int, int],
    mode: PowerMode,
    *,
    timeline: PiecewiseSunlightTimeline,
    solar: SolarPowerModel,
    battery: BatteryModel,
    rover: RoverPowerModel,
) -> EnergyTimelineResult:
    """Split one source-cell operation at every sunlight step boundary."""

    energy = _finite("energy_wh", energy_wh, positive=False)
    if energy > battery.capacity_wh:
        raise TrajectoryInputError(
            "energy_wh cannot exceed battery capacity.",
            code="trajectory_invalid_battery_state",
            details={"energy_wh": energy, "capacity_wh": battery.capacity_wh},
        )
    if mode not in ("drive", "idle"):
        raise TrajectoryInputError(
            "mode must be 'drive' or 'idle'.",
            code="trajectory_invalid_power_mode",
            details={"mode": mode},
        )
    start = timeline._axis.snap_hour(
        timeline._axis.hours_from_start(start_time, name="start_time")
    )
    stop = timeline._axis.snap_hour(
        timeline._axis.hours_from_start(stop_time, name="stop_time")
    )
    if stop < start or start < 0.0 or stop > timeline._axis.duration_hours:
        raise TrajectoryInputError(
            "Power operation times must be ordered within timeline coverage.",
            code="trajectory_power_time_out_of_range",
        )
    if (
        not isinstance(cell, tuple)
        or len(cell) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in cell
        )
        or not timeline.georef.contains_pixel(int(cell[0]), int(cell[1]))
    ):
        raise TrajectoryInputError(
            "Power operation cell must be an in-bounds integer (x, y) tuple.",
            code="trajectory_invalid_power_cell",
        )
    normalized_cell = (int(cell[0]), int(cell[1]))
    segments = []
    cursor = start
    while cursor < stop:
        interval = timeline._axis.interval_index_from_hours(cursor)
        assert interval is not None
        segment_stop = min(stop, float(timeline._axis.boundary_hours[interval + 1]))
        fraction = float(
            timeline.fractions[interval, normalized_cell[1], normalized_cell[0]]
        )
        transition = integrate_energy(
            energy,
            segment_stop - cursor,
            fraction,
            mode,
            solar=solar,
            battery=battery,
            rover=rover,
        )
        segments.append(
            EnergySegment(
                mode=mode,
                start_time=timeline._axis.datetime_from_hours(cursor),
                stop_time=timeline._axis.datetime_from_hours(segment_stop),
                cell=normalized_cell,
                sunlight_fraction=fraction,
                start_energy_wh=energy,
                end_energy_wh=transition.energy_wh,
                generated_wh=transition.generated_wh,
                consumed_wh=transition.consumed_wh,
                discarded_wh=transition.discarded_wh,
            )
        )
        energy = transition.energy_wh
        cursor = segment_stop
        if not transition.feasible:
            return EnergyTimelineResult(False, energy_wh, energy, tuple(segments))
    feasible = energy >= battery.minimum_energy_wh - (
        1.0e-12 * max(1.0, battery.capacity_wh)
    )
    return EnergyTimelineResult(feasible, energy_wh, energy, tuple(segments))
