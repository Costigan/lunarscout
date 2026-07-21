from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MapAlgebraUnitError


def require_matching_units(
    *,
    units_a: str | None,
    units_b: str | None,
    allow_unknown: bool = False,
) -> None:
    if units_a is None and units_b is None:
        return
    if units_a is None or units_b is None:
        if not allow_unknown:
            raise MapAlgebraUnitError(
                "Both operands must have known units or use allow_unknown_units=True.",
                code="map_algebra_unknown_units",
                details={"left_units": units_a, "right_units": units_b},
            )
        return
    if units_a != units_b:
        raise MapAlgebraUnitError(
            "Raster units must match exactly for this operation.",
            code="map_algebra_unit_mismatch",
            details={"left_units": units_a, "right_units": units_b},
        )


def require_angle_units(units: str | None, *, argument: str) -> str:
    if units is None:
        raise MapAlgebraUnitError(
            f"Trigonometric operations require 'degrees' or 'radians' units. "
            f"Operand '{argument}' has unknown units.",
            code="map_algebra_missing_angle_units",
            details={"argument": argument, "units": units},
        )
    normalized = str(units).strip().lower()
    if normalized not in ("degrees", "radians"):
        raise MapAlgebraUnitError(
            f"Trigonometric operations require 'degrees' or 'radians' units. "
            f"Operand '{argument}' has units '{units}'.",
            code="map_algebra_invalid_angle_units",
            details={"argument": argument, "units": units},
        )
    return normalized


def multiply_units(
    *,
    units_a: str | None,
    units_b: str | None,
    output_units: str | None = None,
) -> str | None:
    if units_a is not None and units_b is not None:
        if output_units is None:
            raise MapAlgebraUnitError(
                "Multiplication of two unit-bearing rasters requires "
                "explicit output_units.",
                code="map_algebra_missing_output_units",
                details={"left_units": units_a, "right_units": units_b},
            )
        return output_units
    return units_a or units_b
