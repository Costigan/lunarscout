from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from ..alignment import same_grid
from ..errors import TrajectoryInputError
from ..georeference import GeoReference
from ._time_contract import BOUNDARY_SNAP_HOURS, IntervalTimeAxis
from ._validation import StaticProblem

if TYPE_CHECKING:
    from ._dynamic_reference import DynamicOccupancyTimeline


def _factor_array(
    values: npt.ArrayLike,
    expected: tuple[int, ...],
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values)
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TrajectoryInputError(
            f"{name} must use a real numeric dtype.",
            code="trajectory_invalid_dynamic_mobility",
            details={"name": name, "dtype": str(array.dtype)},
        )
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrajectoryInputError(
            f"{name} cannot be converted to float64.",
            code="trajectory_invalid_dynamic_mobility",
            details={"name": name},
        ) from exc
    if result.shape != expected:
        raise TrajectoryInputError(
            f"{name} has an invalid shape.",
            code="trajectory_invalid_dynamic_mobility",
            details={
                "name": name,
                "shape": list(result.shape),
                "expected": list(expected),
            },
        )
    if np.any(np.isnan(result)) or np.any(result <= 0.0):
        raise TrajectoryInputError(
            f"{name} factors must be positive finite values or positive infinity.",
            code="trajectory_invalid_dynamic_mobility",
            details={"name": name},
        )
    return result


@dataclass(frozen=True, slots=True, eq=False)
class CompiledDynamicTravelModel:
    """Private interval-specialized movement table for exact dynamic search."""

    georef: GeoReference
    boundaries: tuple[datetime, ...]
    available: NDArray[np.bool_]
    base_duration_hours: NDArray[np.float64]
    interval_factors: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not isinstance(self.georef, GeoReference):
            raise TrajectoryInputError(
                "Compiled mobility georef must be a GeoReference.",
                code="trajectory_invalid_dynamic_mobility",
            )
        height = self.georef.height
        width = self.georef.width
        axis = IntervalTimeAxis(self.boundaries)
        available = np.asarray(self.available)
        base = np.asarray(self.base_duration_hours)
        factors = np.asarray(self.interval_factors)
        if (
            available.shape != (height, width)
            or available.dtype != np.dtype(np.bool_)
        ):
            raise TrajectoryInputError(
                "Compiled mobility availability must be Boolean and match georef.",
                code="trajectory_invalid_dynamic_mobility",
            )
        if base.ndim != 3 or base.shape[:2] != (height, width):
            raise TrajectoryInputError(
                "Compiled base durations must have shape (y, x, direction).",
                code="trajectory_invalid_dynamic_mobility",
            )
        expected = (axis.interval_count, *base.shape)
        if factors.shape != expected:
            raise TrajectoryInputError(
                "Compiled interval factors have an invalid shape.",
                code="trajectory_invalid_dynamic_mobility",
                details={"shape": list(factors.shape), "expected": list(expected)},
            )
        if (
            base.dtype != np.dtype(np.float64)
            or factors.dtype != np.dtype(np.float64)
            or np.any(np.isnan(base))
            or np.any(base <= 0.0)
            or np.any(np.isnan(factors))
            or np.any(factors <= 0.0)
        ):
            raise TrajectoryInputError(
                "Compiled mobility values must be positive or positive infinity.",
                code="trajectory_invalid_dynamic_mobility",
            )
        available_copy = np.array(available, dtype=np.bool_, copy=True)
        base_copy = np.array(base, dtype=np.float64, copy=True, order="C")
        factor_copy = np.array(factors, dtype=np.float64, copy=True, order="C")
        available_copy.flags.writeable = False
        base_copy.flags.writeable = False
        factor_copy.flags.writeable = False
        object.__setattr__(self, "available", available_copy)
        object.__setattr__(self, "boundaries", axis.boundaries)
        object.__setattr__(self, "base_duration_hours", base_copy)
        object.__setattr__(self, "interval_factors", factor_copy)

    @property
    def direction_count(self) -> int:
        return int(self.base_duration_hours.shape[2])

    def require_compatible(
        self,
        problem: StaticProblem,
        timeline: DynamicOccupancyTimeline,
    ) -> None:
        try:
            grid_matches = same_grid(self.georef, problem.georef) and same_grid(
                self.georef, timeline.georef
            )
        except Exception as exc:
            raise TrajectoryInputError(
                "Unable to compare compiled mobility grids.",
                code="trajectory_dynamic_mobility_mismatch",
                details={"error": str(exc)},
            ) from exc
        if (
            not grid_matches
            or self.boundaries != timeline.boundaries
            or self.direction_count != len(problem.steps)
            or not np.array_equal(self.available, problem.available)
        ):
            raise TrajectoryInputError(
                "Compiled mobility does not match the problem and timeline.",
                code="trajectory_dynamic_mobility_mismatch",
            )

    def arrival_hours(
        self,
        timeline: DynamicOccupancyTimeline,
        x: int,
        y: int,
        direction: int,
        departure: float,
    ) -> float:
        base = float(self.base_duration_hours[y, x, direction])
        if not np.isfinite(base):
            return np.inf
        current = timeline.snap_hour(departure)
        interval = timeline.interval_index(current)
        if interval is None:
            return np.inf
        remaining = 1.0
        while interval < timeline.interval_count:
            factor = float(self.interval_factors[interval, y, x, direction])
            if not np.isfinite(factor):
                return np.inf
            full_duration = base * factor
            if not np.isfinite(full_duration) or full_duration <= 0.0:
                return np.inf
            boundary = float(timeline._boundary_hours[interval + 1])
            available_hours = boundary - current
            required_hours = remaining * full_duration
            if required_hours <= available_hours + BOUNDARY_SNAP_HOURS:
                arrival = timeline.snap_hour(current + required_hours)
                if arrival >= timeline.duration_hours:
                    return np.inf
                return arrival
            remaining -= available_hours / full_duration
            current = boundary
            interval += 1
        return np.inf


def compile_dynamic_travel_model(
    problem: StaticProblem,
    timeline: DynamicOccupancyTimeline,
    *,
    interval_edge_factors: npt.ArrayLike | None = None,
    hazard_factors: npt.ArrayLike | None = None,
) -> CompiledDynamicTravelModel:
    """Compile static signed-slope durations and explicit dynamic factors."""

    height = problem.georef.height
    width = problem.georef.width
    direction_count = len(problem.steps)
    base = np.full((height, width, direction_count), np.inf, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            for direction, step in enumerate(problem.steps):
                base[y, x, direction] = problem.transition_time(x, y, step)

    factor_shape = (timeline.interval_count, height, width, direction_count)
    if interval_edge_factors is None:
        factors = np.ones(factor_shape, dtype=np.float64)
    else:
        raw = np.asarray(interval_edge_factors)
        if raw.shape == (timeline.interval_count, direction_count):
            raw = np.broadcast_to(raw[:, None, None, :], factor_shape)
        factors = _factor_array(raw, factor_shape, name="interval_edge_factors")
        factors = np.array(factors, dtype=np.float64, copy=True, order="C")

    if hazard_factors is not None:
        hazards = _factor_array(
            hazard_factors,
            (height, width),
            name="hazard_factors",
        )
        for y in range(height):
            for x in range(width):
                for direction, step in enumerate(problem.steps):
                    destination_x = x + step.dx
                    destination_y = y + step.dy
                    if 0 <= destination_x < width and 0 <= destination_y < height:
                        try:
                            with np.errstate(over="raise", invalid="raise"):
                                factors[:, y, x, direction] *= hazards[
                                    destination_y, destination_x
                                ]
                        except FloatingPointError as exc:
                            raise TrajectoryInputError(
                                "Combined dynamic mobility factors overflowed.",
                                code="trajectory_dynamic_mobility_overflow",
                                details={
                                    "source": [x, y],
                                    "destination": [destination_x, destination_y],
                                },
                            ) from exc
    if np.any(np.isnan(factors)) or np.any(factors <= 0.0):
        raise TrajectoryInputError(
            "Combined dynamic mobility factors overflowed or became invalid.",
            code="trajectory_invalid_dynamic_mobility",
        )
    try:
        with np.errstate(over="raise", invalid="raise"):
            np.multiply(base[None, ...], factors)
    except FloatingPointError as exc:
        raise TrajectoryInputError(
            "Dynamic edge durations overflow float64.",
            code="trajectory_dynamic_mobility_overflow",
        ) from exc
    return CompiledDynamicTravelModel(
        georef=problem.georef,
        boundaries=timeline.boundaries,
        available=problem.available,
        base_duration_hours=base,
        interval_factors=factors,
    )
