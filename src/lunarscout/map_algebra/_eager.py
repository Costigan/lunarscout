from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..raster import Raster
from ._kernels import _NumericKernel
from ._validation import (
    _infer_output_georef,
    _is_scalar,
    _normalize_scalar,
    _require_common_grid,
)
from ._validity import apply_numeric_domain, intersect_validity


def _dispatch_unary(
    raster_a: Raster,
    kernel: _NumericKernel,
    *,
    operation: str,
    output_name: str | None = None,
) -> Raster:
    result_values = kernel(raster_a.values)
    result_valid = apply_numeric_domain(
        result_values,
        raster_a.valid,
        operation=operation,
        policy="invalid",
    )
    result_units = raster_a.units
    return Raster(
        values=result_values,
        georef=raster_a.georef,
        valid=result_valid,
        units=result_units,
        name=output_name or raster_a.name,
    )


def _dispatch_binary_raster_raster(
    raster_a: Raster,
    raster_b: Raster,
    kernel: _NumericKernel,
    *,
    operation: str,
    output_name: str | None = None,
    output_units: str | None = None,
) -> Raster:
    _require_common_grid([raster_a, raster_b])
    result_values = kernel(raster_a.values, raster_b.values)
    result_valid = intersect_validity(raster_a.valid, raster_b.valid)
    result_valid = apply_numeric_domain(
        result_values,
        result_valid,
        operation=operation,
        policy="invalid",
    )
    return Raster(
        values=result_values,
        georef=_infer_output_georef([raster_a, raster_b]),
        valid=result_valid,
        units=output_units or raster_a.units or raster_b.units,
        name=output_name,
    )


def _dispatch_binary_raster_scalar(
    raster_a: Raster,
    scalar: int | float,
    kernel: Callable[[np.ndarray[Any, Any], int | float], np.ndarray[Any, Any]],
    *,
    operation: str,
    output_name: str | None = None,
    output_units: str | None = None,
) -> Raster:
    result_values = kernel(raster_a.values, scalar)
    result_valid = apply_numeric_domain(
        result_values,
        raster_a.valid,
        operation=operation,
        policy="invalid",
    )
    return Raster(
        values=result_values,
        georef=raster_a.georef,
        valid=result_valid,
        units=output_units or raster_a.units,
        name=output_name or raster_a.name,
    )
