from __future__ import annotations

from dataclasses import dataclass
from math import floor, isclose
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import TrajectoryInputError
from ..georeference import GeoReference
from ..spice_geometry import LonLat
from ._geometry import NeighborStep, linear_units_to_metres, neighbor_steps

if TYPE_CHECKING:
    from .static import Cell, StaticTravelModel


# PROJ round trips near lunar projection singularities can differ by several
# 1e-7 pixels. Snap only at cell boundaries so the documented half-open extent
# is stable under a projected -> LonLat -> projected round trip.
_INTEGER_SNAP_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class StaticProblem:
    available: NDArray[np.bool_]
    elevation_m: NDArray[np.float64] | None
    georef: GeoReference
    start: tuple[int, int]
    goal: tuple[int, int] | None
    model: StaticTravelModel
    units_to_metres: float
    steps: tuple[NeighborStep, ...]

    def transition_time(
        self,
        x: int,
        y: int,
        step: NeighborStep,
    ) -> float:
        nx = x + step.dx
        ny = y + step.dy
        height, width = self.available.shape
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            return np.inf
        if not self.available[y, x] or not self.available[ny, nx]:
            return np.inf
        if step.dx != 0 and step.dy != 0:
            if not self.available[y, nx] or not self.available[ny, x]:
                return np.inf
        factor = 1.0
        if self.model.slip is not None:
            assert self.elevation_m is not None
            signed_slope = (
                self.elevation_m[ny, nx] - self.elevation_m[y, x]
            ) / step.distance_m
            factor = self.model.slip.factor(signed_slope)
        if not np.isfinite(factor):
            return np.inf
        duration = step.distance_m / self.model.speed_m_per_h * factor
        if not np.isfinite(duration) or duration <= 0.0:
            raise TrajectoryInputError(
                "The travel model produced an invalid edge duration.",
                code="trajectory_invalid_edge_duration",
                details={"from": [x, y], "to": [nx, ny], "duration": duration},
            )
        return duration

    @property
    def minimum_factor(self) -> float:
        if self.model.slip is None:
            return 1.0
        return self.model.slip.minimum_factor


def _array_shape_error(name: str, shape: tuple[int, ...], expected: tuple[int, int]):
    return TrajectoryInputError(
        f"{name} must have shape (height, width) matching georef.",
        code="trajectory_grid_shape_mismatch",
        details={"name": name, "shape": list(shape), "expected": list(expected)},
    )


def _normalize_arrays(
    traversable: ArrayLike,
    georef: GeoReference,
    *,
    valid: ArrayLike | None,
    elevation: ArrayLike | None,
    require_elevation: bool,
) -> tuple[NDArray[np.bool_], NDArray[np.float64] | None]:
    expected = (georef.height, georef.width)
    traversable_array = np.asarray(traversable)
    if traversable_array.shape != expected:
        raise _array_shape_error("traversable", traversable_array.shape, expected)
    if not (
        np.issubdtype(traversable_array.dtype, np.bool_)
        or np.issubdtype(traversable_array.dtype, np.integer)
    ):
        raise TrajectoryInputError(
            "traversable must use a Boolean or integer dtype.",
            code="trajectory_invalid_traversable_dtype",
            details={"dtype": str(traversable_array.dtype)},
        )
    if valid is None:
        valid_array = np.ones(expected, dtype=np.bool_)
    else:
        valid_array = np.asarray(valid)
        if valid_array.shape != expected:
            raise _array_shape_error("valid", valid_array.shape, expected)
        if not np.issubdtype(valid_array.dtype, np.bool_):
            raise TrajectoryInputError(
                "valid must use a Boolean dtype.",
                code="trajectory_invalid_valid_dtype",
                details={"dtype": str(valid_array.dtype)},
            )
    available = np.asarray(valid_array & (traversable_array != 0), dtype=np.bool_)

    if elevation is None:
        if require_elevation:
            raise TrajectoryInputError(
                "elevation is required when the travel model uses slip.",
                code="trajectory_elevation_required",
            )
        return available, None
    elevation_array = np.asarray(elevation)
    if elevation_array.shape != expected:
        raise _array_shape_error("elevation", elevation_array.shape, expected)
    if np.issubdtype(elevation_array.dtype, np.bool_) or not np.issubdtype(
        elevation_array.dtype, np.number
    ) or np.issubdtype(elevation_array.dtype, np.complexfloating):
        raise TrajectoryInputError(
            "elevation must use a real numeric dtype and contain metres.",
            code="trajectory_invalid_elevation_dtype",
            details={"dtype": str(elevation_array.dtype)},
        )
    elevation_m = np.asarray(elevation_array, dtype=np.float64)
    if np.any(~np.isfinite(elevation_m[available])):
        raise TrajectoryInputError(
            "Elevation must be finite at every available cell.",
            code="trajectory_nonfinite_elevation",
        )
    return available, elevation_m


def _snap_near_integer(value: float) -> float:
    nearest = round(value)
    if isclose(value, nearest, rel_tol=0.0, abs_tol=_INTEGER_SNAP_TOLERANCE):
        return float(nearest)
    return value


def normalize_cell(value: Cell | LonLat, georef: GeoReference, *, name: str) -> tuple[int, int]:
    if isinstance(value, LonLat):
        try:
            column, row = georef.lonlat_to_pixel(
                value.longitude,
                value.latitude,
                anchor="corner",
            )
            column = _snap_near_integer(float(column))
            row = _snap_near_integer(float(row))
        except Exception as exc:
            raise TrajectoryInputError(
                f"Unable to convert {name} from longitude/latitude to a raster cell.",
                code="trajectory_coordinate_transform_failed",
                details={"name": name, "longitude": value.longitude, "latitude": value.latitude},
            ) from exc
        cell = (floor(column), floor(row))
    else:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TrajectoryInputError(
                f"{name} must be an (x, y) integer tuple or LonLat.",
                code="trajectory_invalid_cell",
                details={"name": name, "type": type(value).__name__},
            )
        if any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            for item in value
        ):
            raise TrajectoryInputError(
                f"{name} pixel coordinates must be integers.",
                code="trajectory_invalid_cell",
                details={"name": name, "value": list(value)},
            )
        cell = (int(value[0]), int(value[1]))
    if not georef.contains_pixel(cell[0], cell[1]):
        raise TrajectoryInputError(
            f"{name} is outside the raster extent.",
            code="trajectory_cell_out_of_bounds",
            details={
                "name": name,
                "cell": list(cell),
                "width": georef.width,
                "height": georef.height,
            },
        )
    return cell


def _require_available(cell: tuple[int, int], available: NDArray[np.bool_], *, name: str) -> None:
    x, y = cell
    if not available[y, x]:
        raise TrajectoryInputError(
            f"{name} must be valid and traversable.",
            code="trajectory_cell_unavailable",
            details={"name": name, "cell": [x, y]},
        )


def prepare_static_problem(
    traversable: ArrayLike,
    georef: GeoReference,
    start: Cell | LonLat,
    *,
    goal: Cell | LonLat | None = None,
    valid: ArrayLike | None = None,
    elevation: ArrayLike | None = None,
    model: StaticTravelModel | None = None,
) -> StaticProblem:
    from .static import StaticTravelModel

    if not isinstance(georef, GeoReference):
        raise TrajectoryInputError(
            "georef must be a GeoReference.",
            code="trajectory_invalid_georef",
            details={"type": type(georef).__name__},
        )
    if model is None:
        model = StaticTravelModel()
    elif not isinstance(model, StaticTravelModel):
        raise TrajectoryInputError(
            "model must be a StaticTravelModel or None.",
            code="trajectory_invalid_model",
            details={"type": type(model).__name__},
        )
    units_to_metres = linear_units_to_metres(georef)
    available, elevation_m = _normalize_arrays(
        traversable,
        georef,
        valid=valid,
        elevation=elevation,
        require_elevation=model.slip is not None,
    )
    start_cell = normalize_cell(start, georef, name="start")
    _require_available(start_cell, available, name="start")
    goal_cell = None
    if goal is not None:
        goal_cell = normalize_cell(goal, georef, name="goal")
        _require_available(goal_cell, available, name="goal")
    steps = neighbor_steps(
        georef,
        include_diagonals=model.include_diagonals,
        units_to_metres=units_to_metres,
    )
    return StaticProblem(
        available=available,
        elevation_m=elevation_m,
        georef=georef,
        start=start_cell,
        goal=goal_cell,
        model=model,
        units_to_metres=units_to_metres,
        steps=steps,
    )
