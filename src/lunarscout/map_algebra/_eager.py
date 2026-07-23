from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..errors import MapAlgebraOperationError
from ..raster import Raster
from ._dtypes import (
    OverflowPolicy,
    checked_integer_operation,
    checked_integer_power,
    normalize_overflow,
    result_dtype,
)
from ._kernels import _NumericKernel
from ._validation import _infer_output_georef, _require_common_grid
from ._validity import (
    NumericErrorsPolicy,
    apply_numeric_domain,
    intersect_validity,
    normalize_numeric_errors,
)

_CHECKED_INTEGER_OPERATIONS = frozenset({
    "add", "subtract", "multiply", "floor_divide", "remainder",
    "negative", "absolute", "square",
})
_ZERO_DIVISOR_OPERATIONS = frozenset({"divide", "floor_divide", "remainder"})


def _run_kernel(operation: str, kernel: Callable[..., np.ndarray[Any, Any]], *args: Any) -> np.ndarray[Any, Any]:
    try:
        with np.errstate(all="ignore"):
            return np.asarray(kernel(*args))
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise MapAlgebraOperationError(
            f"Operation '{operation}' failed during numeric execution.",
            code="map_algebra_numeric_execution_failed",
            details={"operation": operation, "error": str(exc)},
        ) from exc


def _domain_errors(
    operation: str,
    shape: tuple[int, ...],
    divisor: np.ndarray[Any, Any] | int | float | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]] | None:
    if operation not in _ZERO_DIVISOR_OPERATIONS or divisor is None:
        return None
    return np.broadcast_to(np.asarray(divisor) == 0, shape)


def _dispatch_unary(
    raster_a: Raster,
    kernel: _NumericKernel,
    *,
    operation: str,
    output_name: str | None = None,
    output_units: str | None = None,
    keep_units: bool = True,
    numeric_errors: NumericErrorsPolicy = "invalid",
    overflow: OverflowPolicy = "raise",
) -> Raster:
    numeric_errors = normalize_numeric_errors(numeric_errors)
    overflow = normalize_overflow(overflow)
    target_dtype = result_dtype(
        (raster_a.dtype,), operation=operation, overflow=overflow,
    )
    if target_dtype.kind in "iu" and operation in _CHECKED_INTEGER_OPERATIONS:
        result_values = checked_integer_operation(
            raster_a.values, None, target_dtype, kernel,
            operation=operation, overflow=overflow, check_mask=raster_a.valid,
        )
    else:
        result_values = _run_kernel(operation, kernel, raster_a.values)
        if result_values.dtype != target_dtype:
            result_values = result_values.astype(target_dtype, copy=False)

    result_valid = apply_numeric_domain(
        result_values, raster_a.valid, operation=operation, policy=numeric_errors,
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
    numeric_errors: NumericErrorsPolicy = "invalid",
    overflow: OverflowPolicy = "raise",
) -> Raster:
    _require_common_grid([raster_a, raster_b])
    numeric_errors = normalize_numeric_errors(numeric_errors)
    overflow = normalize_overflow(overflow)
    target_dtype = result_dtype(
        (raster_a.dtype, raster_b.dtype), operation=operation, overflow=overflow,
    )
    base_valid = intersect_validity(raster_a.valid, raster_b.valid)
    power_domain = None
    if target_dtype.kind in "iu" and operation == "power":
        result_values, power_domain = checked_integer_power(
            raster_a.values,
            raster_b.values,
            target_dtype,
            overflow=overflow,
            check_mask=base_valid,
        )
    elif target_dtype.kind in "iu" and operation in _CHECKED_INTEGER_OPERATIONS:
        result_values = checked_integer_operation(
            raster_a.values, raster_b.values, target_dtype, kernel,
            operation=operation, overflow=overflow, check_mask=base_valid,
        )
    else:
        result_values = _run_kernel(operation, kernel, raster_a.values, raster_b.values)
        if result_values.dtype != target_dtype:
            result_values = result_values.astype(target_dtype, copy=False)

    result_valid = base_valid
    result_valid = apply_numeric_domain(
        result_values,
        result_valid,
        operation=operation,
        policy=numeric_errors,
        domain_errors=(
            power_domain
            if power_domain is not None
            else _domain_errors(operation, result_values.shape, raster_b.values)
        ),
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
    keep_units: bool = True,
    numeric_errors: NumericErrorsPolicy = "invalid",
    overflow: OverflowPolicy = "raise",
    scalar_left: bool = False,
) -> Raster:
    numeric_errors = normalize_numeric_errors(numeric_errors)
    overflow = normalize_overflow(overflow)
    target_dtype = result_dtype(
        (raster_a.dtype,), operation=operation, scalars=(scalar,), overflow=overflow,
        scalar_left=scalar_left,
    )
    power_domain = None
    if target_dtype.kind in "iu" and operation == "power":
        if scalar_left:
            scalar_dtype = (
                scalar.dtype
                if isinstance(scalar, np.generic)
                else np.min_scalar_type(scalar)
            )
            base = np.broadcast_to(
                np.asarray(scalar, dtype=scalar_dtype), raster_a.shape,
            )
            exponent: np.ndarray[Any, Any] | int | np.integer = raster_a.values
        else:
            base = raster_a.values
            exponent = scalar  # type: ignore[assignment]
        result_values, power_domain = checked_integer_power(
            base,
            exponent,
            target_dtype,
            overflow=overflow,
            check_mask=raster_a.valid,
        )
    elif target_dtype.kind in "iu" and operation in _CHECKED_INTEGER_OPERATIONS:
        left = (
            np.broadcast_to(np.asarray(scalar), raster_a.shape)
            if scalar_left else raster_a.values
        )
        right = raster_a.values if scalar_left else scalar
        result_values = checked_integer_operation(
            left, right, target_dtype, kernel,
            operation=operation, overflow=overflow, check_mask=raster_a.valid,
        )
    else:
        args = (scalar, raster_a.values) if scalar_left else (raster_a.values, scalar)
        result_values = _run_kernel(operation, kernel, *args)
        if result_values.dtype != target_dtype:
            result_values = result_values.astype(target_dtype, copy=False)

    result_valid = apply_numeric_domain(
        result_values,
        raster_a.valid,
        operation=operation,
        policy=numeric_errors,
        domain_errors=(
            power_domain
            if power_domain is not None
            else _domain_errors(
                operation, result_values.shape,
                raster_a.values if scalar_left else scalar,
            )
        ),
    )
    return Raster(
        values=result_values,
        georef=raster_a.georef,
        valid=result_valid,
        units=(output_units or raster_a.units) if keep_units else output_units,
        name=output_name or raster_a.name,
    )
