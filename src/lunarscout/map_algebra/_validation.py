from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from ..errors import MapAlgebraGridError, MapAlgebraError
from ..georeference import GeoReference
from ..raster import Raster


def _normalize_scalar(value: Any, *, argument: str = "value") -> Any:
    if isinstance(value, (int, float, np.integer, np.floating)):
        # Preserve explicit NumPy scalar precision. NumPy 2 treats Python
        # scalars as weakly typed, while np.float64/np.int64 are deliberate
        # dtype choices that must participate in shared dtype inference.
        return value
    if isinstance(value, Real):
        return float(value)
    raise MapAlgebraError(
        f"Operand '{argument}' must be a Raster or a real numeric scalar.",
        code="map_algebra_invalid_operand",
        details={"argument": argument, "type": type(value).__name__},
    )


def _is_scalar(value: Any) -> bool:
    return isinstance(value, Real)


def _require_common_grid(operands: list[Raster]) -> None:
    if len(operands) < 2:
        return
    reference = operands[0].georef
    for i, raster in enumerate(operands[1:], start=2):
        from ..alignment import _grid_differences

        differences = _grid_differences(reference, raster.georef, affine_tolerance=0.0)
        if differences:
            raise MapAlgebraGridError(
                f"All raster operands must share the same grid. "
                f"Operand {i} differs from operand 1.",
                code="map_algebra_grid_mismatch",
                details={
                    "differences": differences,
                },
            )


def _infer_output_georef(operands: list[Raster]) -> GeoReference:
    return operands[0].georef.with_nodata(None)


def _as_raster_operand(value: Any, *, argument: str = "value") -> Raster | int | float:
    """Accept a ``Raster`` or a real numeric scalar for eager operations."""
    if isinstance(value, Raster):
        return value
    if _is_scalar(value):
        return _normalize_scalar(value, argument=argument)
    raise MapAlgebraError(
        f"Operand '{argument}' must be a Raster or a real numeric scalar.",
        code="map_algebra_invalid_operand",
        details={"argument": argument, "type": type(value).__name__},
    )


def _as_expression_operand(
    value: Any,
    *,
    argument: str = "value",
    grid_hint: GeoReference | None = None,
) -> Any:
    """Accept a ``RasterExpression``, ``Raster``, or a real numeric scalar
    for expression-building operations.  ``Raster`` objects are wrapped as
    in-memory constant expression nodes."""
    from ._model import RasterExpression as _Expr
    from ._sources import constant as _const

    if isinstance(value, _Expr):
        result: Any = value
    if isinstance(value, Raster):
        result = _const(value)
    elif _is_scalar(value):
        return _normalize_scalar(value, argument=argument)
    elif not isinstance(value, _Expr):
        raise MapAlgebraError(
            f"Operand '{argument}' must be a RasterExpression, Raster, or a real numeric scalar.",
            code="map_algebra_invalid_operand",
            details={"argument": argument, "type": type(value).__name__},
        )

    if grid_hint is not None and result.grid is not None:
        from ..alignment import _grid_differences

        differences = _grid_differences(grid_hint, result.grid, affine_tolerance=0.0)
        if differences:
            raise MapAlgebraGridError(
                f"Operand '{argument}' does not match the required grid.",
                code="map_algebra_grid_mismatch",
                details={"argument": argument, "differences": differences},
            )
    return result
