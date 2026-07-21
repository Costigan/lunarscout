from __future__ import annotations

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


def result_units_for_arithmetic(
    *,
    units_a: str | None,
    units_b: str | None,
    operation: str,
    output_units: str | None = None,
) -> str | None:
    if operation in ("add", "subtract", "min", "max"):
        require_matching_units(units_a=units_a, units_b=units_b)
        return units_a
    if operation in ("multiply", "divide"):
        if units_a is not None and units_b is not None:
            if output_units is None:
                raise MapAlgebraUnitError(
                    "Multiplication or division of two unit-bearing rasters "
                    "requires explicit output_units.",
                    code="map_algebra_missing_output_units",
                    details={"left_units": units_a, "right_units": units_b},
                )
            return output_units
        return units_a or units_b
    return units_a
