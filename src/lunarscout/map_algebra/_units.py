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


def normalize_output_units(output_units: str | None) -> str | None:
    if output_units is None:
        return None
    if not isinstance(output_units, str) or not output_units.strip():
        raise MapAlgebraUnitError(
            "output_units must be a non-empty string or None.",
            code="map_algebra_invalid_output_units",
            details={"output_units": repr(output_units)},
        )
    return output_units.strip()


def power_units(
    *,
    base_units: str | None,
    exponent_units: str | None,
    exponent_is_scalar: bool,
    exponent_value: Any = None,
    output_units: str | None = None,
    base_is_raster: bool = True,
) -> str | None:
    """Infer power units without assigning one unit to variable exponents."""
    normalized_output = normalize_output_units(output_units)

    if not exponent_is_scalar:
        if exponent_units is not None:
            raise MapAlgebraUnitError(
                "A raster exponent must carry no unit metadata under the "
                "current power contract.",
                code="map_algebra_dimensioned_exponent",
                details={"exponent_units": exponent_units},
            )
        if base_is_raster and base_units is not None:
            raise MapAlgebraUnitError(
                "A unit-bearing raster base requires a scalar exponent.",
                code="map_algebra_unitful_power_requires_scalar_exponent",
                details={"base_units": base_units},
            )
        if normalized_output is not None:
            raise MapAlgebraUnitError(
                "A raster exponent cannot produce one fixed output unit.",
                code="map_algebra_unexpected_output_units",
                details={"output_units": normalized_output},
            )
        return None

    if not base_is_raster:
        return None
    if base_units is None:
        return normalized_output
    if exponent_value == 1 and normalized_output is None:
        return base_units
    if normalized_output is None:
        raise MapAlgebraUnitError(
            "A unit-bearing raster raised to a power other than one requires "
            "explicit output_units.",
            code="map_algebra_missing_output_units",
            details={
                "base_units": base_units,
                "exponent": repr(exponent_value),
            },
        )
    return normalized_output
