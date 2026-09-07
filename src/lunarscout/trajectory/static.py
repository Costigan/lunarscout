from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import TrajectoryInputError
from ..georeference import GeoReference
from ..spice_geometry import LonLat
from ._static_reference import astar_path, dijkstra_field
from ._validation import prepare_static_problem


Cell: TypeAlias = tuple[int, int]
SlipExtrapolation: TypeAlias = Literal["infeasible", "constant"]


@dataclass(frozen=True, slots=True)
class SlipFunction:
    """Piecewise-linear signed-slope travel-time factors.

    ``signed_slopes`` must be strictly increasing and contain at least two
    values. Positive slope is uphill. Outside the knot domain, movement is
    either infeasible or uses the nearest endpoint factor.
    """

    signed_slopes: tuple[float, ...]
    factors: tuple[float, ...]
    extrapolation: SlipExtrapolation = "infeasible"

    def __post_init__(self) -> None:
        try:
            slopes = tuple(float(value) for value in self.signed_slopes)
            factors = tuple(float(value) for value in self.factors)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrajectoryInputError(
                "Slip-function knots and factors must be finite numbers.",
                code="trajectory_invalid_slip_function",
            ) from exc
        if len(slopes) < 2 or len(slopes) != len(factors):
            raise TrajectoryInputError(
                "SlipFunction requires equally sized slope and factor sequences "
                "with at least two values.",
                code="trajectory_invalid_slip_function",
                details={"slope_count": len(slopes), "factor_count": len(factors)},
            )
        if not all(np.isfinite(slopes)) or not all(np.isfinite(factors)):
            raise TrajectoryInputError(
                "Slip-function knots and factors must be finite.",
                code="trajectory_invalid_slip_function",
            )
        if any(right <= left for left, right in zip(slopes, slopes[1:])):
            raise TrajectoryInputError(
                "Slip-function signed slopes must be strictly increasing.",
                code="trajectory_invalid_slip_function",
            )
        if any(factor <= 0.0 for factor in factors):
            raise TrajectoryInputError(
                "Slip-function factors must be positive.",
                code="trajectory_invalid_slip_function",
            )
        if self.extrapolation not in {"infeasible", "constant"}:
            raise TrajectoryInputError(
                "Slip-function extrapolation must be 'infeasible' or 'constant'.",
                code="trajectory_invalid_slip_function",
                details={"extrapolation": self.extrapolation},
            )
        object.__setattr__(self, "signed_slopes", slopes)
        object.__setattr__(self, "factors", factors)

    @property
    def minimum_factor(self) -> float:
        """Return the global minimum feasible factor for A* lower bounds."""

        return min(self.factors)

    def factor(self, signed_slope: float) -> float:
        """Return a travel-time factor, or infinity for an infeasible slope."""

        try:
            slope = float(signed_slope)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrajectoryInputError(
                "Signed slope must be a finite number.",
                code="trajectory_invalid_signed_slope",
                details={"signed_slope": signed_slope},
            ) from exc
        if not np.isfinite(slope):
            raise TrajectoryInputError(
                "Signed slope must be a finite number.",
                code="trajectory_invalid_signed_slope",
                details={"signed_slope": signed_slope},
            )
        if slope < self.signed_slopes[0]:
            return self.factors[0] if self.extrapolation == "constant" else np.inf
        if slope > self.signed_slopes[-1]:
            return self.factors[-1] if self.extrapolation == "constant" else np.inf
        return float(np.interp(slope, self.signed_slopes, self.factors))


@dataclass(frozen=True, slots=True)
class StaticTravelModel:
    """Deterministic static rover travel-time model."""

    speed_m_per_h: float = 36.0
    include_diagonals: bool = True
    slip: SlipFunction | None = None

    def __post_init__(self) -> None:
        if isinstance(self.speed_m_per_h, (bool, np.bool_)):
            raise TrajectoryInputError(
                "speed_m_per_h must be a finite positive number.",
                code="trajectory_invalid_speed",
                details={"speed_m_per_h": self.speed_m_per_h},
            )
        try:
            speed = float(self.speed_m_per_h)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrajectoryInputError(
                "speed_m_per_h must be a finite positive number.",
                code="trajectory_invalid_speed",
                details={"speed_m_per_h": self.speed_m_per_h},
            ) from exc
        if not np.isfinite(speed) or speed <= 0.0:
            raise TrajectoryInputError(
                "speed_m_per_h must be a finite positive number.",
                code="trajectory_invalid_speed",
                details={"speed_m_per_h": self.speed_m_per_h},
            )
        if not isinstance(self.include_diagonals, (bool, np.bool_)):
            raise TrajectoryInputError(
                "include_diagonals must be Boolean.",
                code="trajectory_invalid_connectivity",
                details={"include_diagonals": self.include_diagonals},
            )
        if self.slip is not None and not isinstance(self.slip, SlipFunction):
            raise TrajectoryInputError(
                "slip must be a SlipFunction or None.",
                code="trajectory_invalid_slip_function",
                details={"type": type(self.slip).__name__},
            )
        object.__setattr__(self, "speed_m_per_h", speed)
        object.__setattr__(self, "include_diagonals", bool(self.include_diagonals))


@dataclass(frozen=True, slots=True, eq=False)
class PathResult:
    reachable: bool
    travel_time_hours: float | None
    path: NDArray[np.int64] | None

    def __post_init__(self) -> None:
        if not isinstance(self.reachable, (bool, np.bool_)):
            raise TrajectoryInputError(
                "PathResult reachable must be Boolean.",
                code="trajectory_invalid_path_result",
            )
        reachable = bool(self.reachable)
        if not reachable:
            if self.travel_time_hours is not None or self.path is not None:
                raise TrajectoryInputError(
                    "An unreachable PathResult must not contain time or path data.",
                    code="trajectory_invalid_path_result",
                )
            object.__setattr__(self, "reachable", False)
            return
        if self.travel_time_hours is None or self.path is None:
            raise TrajectoryInputError(
                "A reachable PathResult requires travel time and path data.",
                code="trajectory_invalid_path_result",
            )
        if isinstance(self.travel_time_hours, (bool, np.bool_)):
            raise TrajectoryInputError(
                "PathResult travel time must be finite and non-negative.",
                code="trajectory_invalid_path_result",
            )
        travel_time = float(self.travel_time_hours)
        if not np.isfinite(travel_time) or travel_time < 0.0:
            raise TrajectoryInputError(
                "PathResult travel time must be finite and non-negative.",
                code="trajectory_invalid_path_result",
            )
        path_input = np.asarray(self.path)
        if not np.issubdtype(path_input.dtype, np.integer) or np.issubdtype(
            path_input.dtype, np.bool_
        ):
            raise TrajectoryInputError(
                "PathResult path coordinates must use an integer dtype.",
                code="trajectory_invalid_path_result",
                details={"dtype": str(path_input.dtype)},
            )
        path = np.array(path_input, dtype=np.int64, copy=True)
        if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != 2:
            raise TrajectoryInputError(
                "PathResult path must have non-empty shape (N, 2).",
                code="trajectory_invalid_path_result",
                details={"shape": list(path.shape)},
            )
        path.flags.writeable = False
        object.__setattr__(self, "reachable", True)
        object.__setattr__(self, "travel_time_hours", travel_time)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True, eq=False)
class TravelTimeResult:
    travel_time_hours: NDArray[np.float64]
    reached: NDArray[np.bool_]
    georef: GeoReference
    start: Cell

    def __post_init__(self) -> None:
        if not isinstance(self.georef, GeoReference):
            raise TrajectoryInputError(
                "TravelTimeResult georef must be a GeoReference.",
                code="trajectory_invalid_travel_time_result",
            )
        expected = (self.georef.height, self.georef.width)
        travel_time_input = np.asarray(self.travel_time_hours)
        reached_input = np.asarray(self.reached)
        if not np.issubdtype(travel_time_input.dtype, np.number) or np.issubdtype(
            travel_time_input.dtype, np.complexfloating
        ):
            raise TrajectoryInputError(
                "TravelTimeResult times must use a real numeric dtype.",
                code="trajectory_invalid_travel_time_result",
            )
        if not np.issubdtype(reached_input.dtype, np.bool_):
            raise TrajectoryInputError(
                "TravelTimeResult reached must use a Boolean dtype.",
                code="trajectory_invalid_travel_time_result",
            )
        travel_time = np.array(travel_time_input, dtype=np.float64, copy=True)
        reached = np.array(reached_input, dtype=np.bool_, copy=True)
        if travel_time.shape != expected or reached.shape != expected:
            raise TrajectoryInputError(
                "TravelTimeResult arrays must match the GeoReference grid.",
                code="trajectory_invalid_travel_time_result",
                details={
                    "travel_time_shape": list(travel_time.shape),
                    "reached_shape": list(reached.shape),
                    "expected": list(expected),
                },
            )
        if np.any(np.isnan(travel_time)) or np.any(travel_time < 0.0):
            raise TrajectoryInputError(
                "TravelTimeResult times must be non-negative or positive infinity.",
                code="trajectory_invalid_travel_time_result",
            )
        if not np.array_equal(reached, np.isfinite(travel_time)):
            raise TrajectoryInputError(
                "TravelTimeResult reached must identify exactly the finite times.",
                code="trajectory_invalid_travel_time_result",
            )
        if (
            not isinstance(self.start, tuple)
            or len(self.start) != 2
            or any(
                isinstance(item, (bool, np.bool_))
                or not isinstance(item, (int, np.integer))
                for item in self.start
            )
        ):
            raise TrajectoryInputError(
                "TravelTimeResult start must be an integer (x, y) tuple.",
                code="trajectory_invalid_travel_time_result",
            )
        start = (int(self.start[0]), int(self.start[1]))
        if not self.georef.contains_pixel(*start) or not reached[start[1], start[0]]:
            raise TrajectoryInputError(
                "TravelTimeResult start must be an in-bounds reached cell.",
                code="trajectory_invalid_travel_time_result",
            )
        if travel_time[start[1], start[0]] != 0.0:
            raise TrajectoryInputError(
                "TravelTimeResult start must have zero travel time.",
                code="trajectory_invalid_travel_time_result",
            )
        travel_time.flags.writeable = False
        reached.flags.writeable = False
        object.__setattr__(self, "travel_time_hours", travel_time)
        object.__setattr__(self, "reached", reached)
        object.__setattr__(self, "start", start)


def static_travel_time(
    traversable: ArrayLike,
    georef: GeoReference,
    start: Cell | LonLat,
    *,
    valid: ArrayLike | None = None,
    elevation: ArrayLike | None = None,
    model: StaticTravelModel | None = None,
) -> TravelTimeResult:
    """Return minimum static travel time from start to every reachable cell."""

    problem = prepare_static_problem(
        traversable,
        georef,
        start,
        valid=valid,
        elevation=elevation,
        model=model,
    )
    travel_time = dijkstra_field(problem)
    reached = np.isfinite(travel_time)
    return TravelTimeResult(
        travel_time_hours=travel_time,
        reached=reached,
        georef=georef,
        start=problem.start,
    )


def static_path(
    traversable: ArrayLike,
    georef: GeoReference,
    start: Cell | LonLat,
    goal: Cell | LonLat,
    *,
    valid: ArrayLike | None = None,
    elevation: ArrayLike | None = None,
    model: StaticTravelModel | None = None,
) -> PathResult:
    """Return a minimum-travel-time static raster-cell path."""

    problem = prepare_static_problem(
        traversable,
        georef,
        start,
        goal=goal,
        valid=valid,
        elevation=elevation,
        model=model,
    )
    assert problem.goal is not None
    travel_time, path = astar_path(problem)
    if path is None:
        return PathResult(reachable=False, travel_time_hours=None, path=None)
    return PathResult(
        reachable=True,
        travel_time_hours=travel_time,
        path=path,
    )
