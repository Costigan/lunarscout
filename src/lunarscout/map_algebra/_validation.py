from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from ..errors import MapAlgebraGridError, MapAlgebraError
from ..georeference import GeoReference
from ..raster import Raster


def _normalize_scalar(value: Any, *, argument: str = "value") -> int | float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value
    if isinstance(value, Real):
        return float(value)
    raise MapAlgebraError(
        f"Operand '{argument}' must be a Raster or a real numeric scalar.",
        code="map_algebra_invalid_operand",
        details={"argument": argument, "type": type(value).__name__},
    )


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


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
