from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isclose

import numpy as np
from pyproj import CRS

from ..errors import TrajectoryInputError
from ..georeference import GeoReference


_CARDINAL_OFFSETS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_DIAGONAL_OFFSETS = ((1, 1), (-1, 1), (-1, -1), (1, -1))


@dataclass(frozen=True, slots=True)
class NeighborStep:
    dx: int
    dy: int
    distance_m: float


def linear_units_to_metres(georef: GeoReference) -> float:
    try:
        crs = CRS.from_wkt(georef.projection_wkt)
    except Exception as exc:
        raise TrajectoryInputError(
            "The trajectory grid CRS cannot be parsed.",
            code="trajectory_invalid_crs",
            details={"error": str(exc)},
        ) from exc
    if not crs.is_projected:
        raise TrajectoryInputError(
            "Trajectory planning in physical units requires a projected CRS.",
            code="trajectory_crs_not_projected",
        )
    axes = tuple(crs.axis_info[:2])
    if len(axes) != 2:
        raise TrajectoryInputError(
            "The projected CRS must define two horizontal axes.",
            code="trajectory_crs_missing_linear_units",
        )
    try:
        factors = tuple(float(axis.unit_conversion_factor) for axis in axes)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrajectoryInputError(
            "The projected CRS has unusable linear-unit metadata.",
            code="trajectory_crs_missing_linear_units",
        ) from exc
    if any(not np.isfinite(value) or value <= 0.0 for value in factors):
        raise TrajectoryInputError(
            "The projected CRS has unusable linear-unit metadata.",
            code="trajectory_crs_missing_linear_units",
            details={"conversion_factors": list(factors)},
        )
    if not isclose(factors[0], factors[1], rel_tol=1e-12, abs_tol=0.0):
        raise TrajectoryInputError(
            "Trajectory planning requires matching horizontal-axis units.",
            code="trajectory_crs_mismatched_linear_units",
            details={"conversion_factors": list(factors)},
        )
    return factors[0]


def neighbor_steps(
    georef: GeoReference,
    *,
    include_diagonals: bool,
    units_to_metres: float,
) -> tuple[NeighborStep, ...]:
    gt = georef.affine_transform
    offsets = _CARDINAL_OFFSETS + (_DIAGONAL_OFFSETS if include_diagonals else ())
    result: list[NeighborStep] = []
    for dx, dy in offsets:
        projected_dx = dx * gt[1] + dy * gt[2]
        projected_dy = dx * gt[4] + dy * gt[5]
        distance_m = hypot(projected_dx, projected_dy) * units_to_metres
        if not np.isfinite(distance_m) or distance_m <= 0.0:
            raise TrajectoryInputError(
                "The affine transform produces an invalid neighbor distance.",
                code="trajectory_invalid_affine_step",
                details={"dx": dx, "dy": dy, "distance_m": distance_m},
            )
        result.append(NeighborStep(dx=dx, dy=dy, distance_m=distance_m))
    return tuple(result)


def cell_distance_m(
    georef: GeoReference,
    left: tuple[int, int],
    right: tuple[int, int],
    *,
    units_to_metres: float,
) -> float:
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    gt = georef.affine_transform
    return hypot(dx * gt[1] + dy * gt[2], dx * gt[4] + dy * gt[5]) * units_to_metres
