from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..errors import MapAlgebraDTypeError
from ..raster import Raster
from ._dtypes import _is_overflow_safe
from ._kernels import _NumericKernel
from ._validation import (
    _infer_output_georef,
    _require_common_grid,
)
from ._validity import apply_numeric_domain, intersect_validity


def _check_integer_overflow(
    target_dtype: np.dtype[Any],
    float_result: np.ndarray[Any, np.dtype[np.float64]],
    operation: str,
) -> None:
    if not _is_overflow_safe(target_dtype, float_result):
        raise MapAlgebraDTypeError(
            "Integer operation overflow detected. "
            "Use cast() to promote to a wider dtype first, "
            "or use floating-point rasters.",
            code="map_algebra_overflow",
            details={
                "result_dtype": str(target_dtype),
                "operation": operation,
            },
        )


def _dispatch_unary(
    raster_a: Raster,
    kernel: _NumericKernel,
    *,
    operation: str,
    output_name: str | None = None,
    output_units: str | None = None,
    keep_units: bool = True,
) -> Raster:
    if np.issubdtype(raster_a.values.dtype, np.integer):
        float_result = kernel(raster_a.values.astype(np.float64, copy=False))
        target_dtype = np.result_type(raster_a.values.dtype)
        _check_integer_overflow(target_dtype, float_result, operation)
        result_values = float_result.astype(target_dtype, copy=False)
    else:
        result_values = kernel(raster_a.values)

    result_valid = apply_numeric_domain(
        result_values, raster_a.valid, operation=operation, policy="invalid",
    )
    return Raster(
        values=result_values,
        georef=raster_a.georef,
        valid=result_valid,
        units=raster_a.units if keep_units else output_units,
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

    a_int = np.issubdtype(raster_a.values.dtype, np.integer)
    b_int = np.issubdtype(raster_b.values.dtype, np.integer)
    integer_preserving = operation not in ("divide", "power")
    if (a_int or b_int) and integer_preserving:
        fa = raster_a.values.astype(np.float64, copy=False)
        fb = raster_b.values.astype(np.float64, copy=False)
        float_result = kernel(fa, fb)
        target_dtype = np.result_type(raster_a.values.dtype, raster_b.values.dtype)
        _check_integer_overflow(target_dtype, float_result, operation)
        result_values = float_result.astype(target_dtype, copy=False)
    else:
        result_values = kernel(raster_a.values, raster_b.values)

    result_valid = intersect_validity(raster_a.valid, raster_b.valid)
    result_valid = apply_numeric_domain(
        result_values, result_valid, operation=operation, policy="invalid",
    )
    return Raster(
        values=result_values,
        georef=_infer_output_georef([raster_a, raster_b]),
        valid=result_valid,
        units=output_units,
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
    integer_result = np.issubdtype(raster_a.values.dtype, np.integer) and operation not in ("divide", "power")
    if integer_result:
        fa = raster_a.values.astype(np.float64, copy=False)
        fs = float(scalar)
        float_result = kernel(fa, fs)
        target_dtype = np.result_type(raster_a.values.dtype, type(scalar))
        _check_integer_overflow(target_dtype, float_result, operation)
        result_values = float_result.astype(target_dtype, copy=False)
    else:
        result_values = kernel(raster_a.values, scalar)

    result_valid = apply_numeric_domain(
        result_values, raster_a.valid, operation=operation, policy="invalid",
    )
    return Raster(
        values=result_values,
        georef=raster_a.georef,
        valid=result_valid,
        units=output_units or raster_a.units,
        name=output_name or raster_a.name,
    )
