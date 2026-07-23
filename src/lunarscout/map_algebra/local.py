from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, NoReturn

import numpy as np

from ..errors import MapAlgebraDTypeError, MapAlgebraError
from ..raster import Raster, _fill_invalid_exact
from ._dtypes import (
    CastOverflowPolicy,
    OverflowPolicy,
    cast_values,
    normalize_dtype,
    result_dtype,
)
from ._validity import NumericErrorsPolicy
from ._eager import (
    _dispatch_binary_raster_raster,
    _dispatch_binary_raster_scalar,
    _dispatch_unary,
)
from ._kernels import (
    _absolute,
    _add,
    _arccos,
    _arcsin,
    _arctan,
    _arctan2,
    _ceil,
    _clip,
    _cos,
    _divide,
    _equal,
    _exp,
    _floor,
    _floor_divide,
    _greater,
    _greater_equal,
    _hypot,
    _less,
    _less_equal,
    _log,
    _log10,
    _logical_and,
    _logical_not,
    _logical_or,
    _logical_xor,
    _maximum,
    _minimum,
    _multiply,
    _negate,
    _not_equal,
    _power,
    _radians,
    _remainder,
    _round_half_even,
    _sin,
    _sqrt,
    _square,
    _subtract,
    _tan,
    _trunc,
    _degrees,
)
from ._units import (
    multiply_units,
    power_units,
    require_angle_units,
    require_matching_units,
)
from ._validation import (
    _as_raster_operand,
    _is_scalar,
    _normalize_scalar,
    _require_common_grid,
)
from ._validity import (
    where_validity,
)


class _InvalidSentinel:
    _instance: _InvalidSentinel | None = None

    def __new__(cls) -> _InvalidSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ma.invalid"

    def __bool__(self) -> NoReturn:
        raise TypeError("ma.invalid does not support truth testing")


invalid = _InvalidSentinel()


def _infer_output_georef_single(raster: Raster) -> Any:
    from ._validation import _infer_output_georef as _iog
    return _iog([raster])

# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def add(
    a: Raster | int | float,
    b: Raster | int | float,
    *,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(
            a, b, _add, operation="add", output_units=a.units,
            overflow=overflow, numeric_errors=numeric_errors,
        )
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(
            a, _normalize_scalar(b, argument="b"), _add,
            operation="add", overflow=overflow, numeric_errors=numeric_errors,
        )
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"), _add,
            operation="add", overflow=overflow, numeric_errors=numeric_errors,
        )
    raise MapAlgebraError("add() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def subtract(
    a: Raster | int | float,
    b: Raster | int | float,
    *,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(
            a, b, _subtract, operation="subtract", output_units=a.units,
            overflow=overflow, numeric_errors=numeric_errors,
        )
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(
            a, _normalize_scalar(b, argument="b"), _subtract,
            operation="subtract", overflow=overflow,
            numeric_errors=numeric_errors,
        )
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"),
            _subtract,
            operation="subtract", overflow=overflow,
            numeric_errors=numeric_errors, scalar_left=True,
        )
    raise MapAlgebraError("subtract() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def multiply(
    a: Raster | int | float,
    b: Raster | int | float,
    *,
    output_units: str | None = None,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        units = multiply_units(units_a=a.units, units_b=b.units, output_units=output_units)
        return _dispatch_binary_raster_raster(
            a, b, _multiply, operation="multiply", output_units=units,
            overflow=overflow, numeric_errors=numeric_errors,
        )
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(
            a, _normalize_scalar(b, argument="b"), _multiply,
            operation="multiply", overflow=overflow,
            numeric_errors=numeric_errors,
        )
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"), _multiply,
            operation="multiply", overflow=overflow,
            numeric_errors=numeric_errors,
        )
    raise MapAlgebraError("multiply() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def divide(
    a: Raster | int | float,
    b: Raster | int | float,
    *,
    output_units: str | None = None,
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        units = multiply_units(units_a=a.units, units_b=b.units, output_units=output_units)
        return _dispatch_binary_raster_raster(a, b, _divide, operation="divide", output_units=units, numeric_errors=numeric_errors)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _divide, operation="divide", numeric_errors=numeric_errors)
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"),
            _divide,
            operation="divide", numeric_errors=numeric_errors, scalar_left=True,
        )
    raise MapAlgebraError("divide() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def floor_divide(
    a: Raster | int | float,
    b: Raster | int | float,
    *,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        return _dispatch_binary_raster_raster(a, b, _floor_divide, operation="floor_divide", overflow=overflow, numeric_errors=numeric_errors)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _floor_divide, operation="floor_divide", overflow=overflow, numeric_errors=numeric_errors)
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"),
            _floor_divide,
            operation="floor_divide", overflow=overflow, numeric_errors=numeric_errors,
            scalar_left=True,
        )
    raise MapAlgebraError("floor_divide() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def remainder(
    a: Raster | int | float,
    b: Raster | int | float,
    *,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        return _dispatch_binary_raster_raster(a, b, _remainder, operation="remainder", overflow=overflow, numeric_errors=numeric_errors)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _remainder, operation="remainder", overflow=overflow, numeric_errors=numeric_errors)
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"),
            _remainder,
            operation="remainder", overflow=overflow, numeric_errors=numeric_errors,
            scalar_left=True,
        )
    raise MapAlgebraError("remainder() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def power(
    base: Raster | int | float,
    exponent: Raster | int | float,
    *,
    output_units: str | None = None,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    """Raise values to a power with explicit derived-unit declarations.

    A unit-bearing raster base requires a scalar exponent and, unless that
    exponent is one, a non-empty ``output_units`` declaration.
    """
    if isinstance(base, Raster) and isinstance(exponent, Raster):
        _require_common_grid([base, exponent])
        units = power_units(
            base_units=base.units, exponent_units=exponent.units,
            exponent_is_scalar=False, output_units=output_units,
        )
        return _dispatch_binary_raster_raster(
            base, exponent, _power, operation="power",
            output_units=units, overflow=overflow,
            numeric_errors=numeric_errors,
        )
    if isinstance(base, Raster) and _is_scalar(exponent):
        normalized_exponent = _normalize_scalar(exponent, argument="exponent")
        units = power_units(
            base_units=base.units, exponent_units=None,
            exponent_is_scalar=True, exponent_value=normalized_exponent,
            output_units=output_units,
        )
        return _dispatch_binary_raster_scalar(
            base, normalized_exponent, _power,
            operation="power", output_units=units, overflow=overflow,
            numeric_errors=numeric_errors,
        )
    if _is_scalar(base) and isinstance(exponent, Raster):
        units = power_units(
            base_units=None, exponent_units=exponent.units,
            exponent_is_scalar=False, output_units=output_units,
            base_is_raster=False,
        )
        return _dispatch_binary_raster_scalar(
            exponent, _normalize_scalar(base, argument="base"),
            _power,
            operation="power", output_units=units, overflow=overflow,
            numeric_errors=numeric_errors,
            scalar_left=True,
        )
    raise MapAlgebraError("power() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def negative(
    a: Raster,
    *,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    return _dispatch_unary(
        a, _negate, operation="negative", overflow=overflow,
        numeric_errors=numeric_errors,
    )


def positive(a: Raster) -> Raster:
    return a


def absolute(
    a: Raster,
    *,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    return _dispatch_unary(
        a, _absolute, operation="absolute", overflow=overflow,
        numeric_errors=numeric_errors,
    )

# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------


def minimum(
    a: Raster | int | float,
    b: Raster | int | float,
    *,
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(
            a, b, _minimum, operation="minimum", output_units=a.units,
            numeric_errors=numeric_errors,
        )
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(
            a, _normalize_scalar(b, argument="b"), _minimum,
            operation="minimum", numeric_errors=numeric_errors,
        )
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"), _minimum,
            operation="minimum", numeric_errors=numeric_errors,
        )
    raise MapAlgebraError("minimum() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def maximum(
    a: Raster | int | float,
    b: Raster | int | float,
    *,
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(
            a, b, _maximum, operation="maximum", output_units=a.units,
            numeric_errors=numeric_errors,
        )
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(
            a, _normalize_scalar(b, argument="b"), _maximum,
            operation="maximum", numeric_errors=numeric_errors,
        )
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"), _maximum,
            operation="maximum", numeric_errors=numeric_errors,
        )
    raise MapAlgebraError("maximum() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def _require_layer_stack(
    layers: tuple[Raster, ...],
    *,
    operation: str,
) -> tuple[Raster, ...]:
    if not layers:
        raise MapAlgebraError(
            f"{operation}() requires at least one Raster.",
            code="map_algebra_empty_layers",
            details={"operation": operation},
        )
    for index, layer in enumerate(layers):
        if not isinstance(layer, Raster):
            raise MapAlgebraError(
                f"{operation}() layers must be Raster values.",
                code="map_algebra_invalid_layer",
                details={
                    "operation": operation,
                    "layer_index": index,
                    "type": type(layer).__name__,
                },
            )
    return layers


def sum_layers(
    *layers: Raster,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    """Add one or more registered layers with strict validity intersection."""
    checked = _require_layer_stack(layers, operation="sum_layers")
    result = add(
        checked[0], 0, overflow=overflow, numeric_errors=numeric_errors,
    )
    for layer in checked[1:]:
        result = add(
            result, layer, overflow=overflow, numeric_errors=numeric_errors,
        )
    return result


def mean_layers(
    *layers: Raster,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    """Calculate the arithmetic mean with strict validity intersection."""
    checked = _require_layer_stack(layers, operation="mean_layers")
    total = sum_layers(
        *checked, overflow=overflow, numeric_errors=numeric_errors,
    )
    return divide(total, len(checked), numeric_errors=numeric_errors)


def min_layers(
    *layers: Raster,
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    """Calculate the cell-wise minimum with strict validity intersection."""
    checked = _require_layer_stack(layers, operation="min_layers")
    result = minimum(
        checked[0], checked[0], numeric_errors=numeric_errors,
    )
    for layer in checked[1:]:
        result = minimum(result, layer, numeric_errors=numeric_errors)
    return result


def max_layers(
    *layers: Raster,
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    """Calculate the cell-wise maximum with strict validity intersection."""
    checked = _require_layer_stack(layers, operation="max_layers")
    result = maximum(
        checked[0], checked[0], numeric_errors=numeric_errors,
    )
    for layer in checked[1:]:
        result = maximum(result, layer, numeric_errors=numeric_errors)
    return result

# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


_SCALAR_LEFT_SWAP = {
    "less": _greater,
    "less_equal": _greater_equal,
    "greater": _less,
    "greater_equal": _less_equal,
}


def _comparison_helper(
    a: Raster | int | float,
    b: Raster | int | float,
    kernel,
    swapped_kernel,
    operation: str,
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(a, b, kernel, operation=operation, output_units=None)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(
            a, _normalize_scalar(b, argument="b"), kernel, operation=operation,
            keep_units=False,
        )
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"), swapped_kernel,
            operation=operation, keep_units=False,
        )
    raise MapAlgebraError(f"{operation}() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def less(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _less, _greater, "less")


def less_equal(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _less_equal, _greater_equal, "less_equal")


def greater(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _greater, _less, "greater")


def greater_equal(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _greater_equal, _less_equal, "greater_equal")


def equal(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _equal, _equal, "equal")


def not_equal(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _not_equal, _not_equal, "not_equal")


def isclose(
    a: Raster | int | float,
    b: Raster | int | float,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        return _dispatch_binary_raster_raster(
            a, b,
            lambda x, y: np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan),
            operation="isclose", output_units=None,
        )
    if isinstance(a, Raster) and _is_scalar(b):
        s = _normalize_scalar(b, argument="b")
        return _dispatch_binary_raster_scalar(
            a, s,
            lambda arr, v: np.isclose(arr, v, rtol=rtol, atol=atol, equal_nan=equal_nan),
            operation="isclose", keep_units=False,
        )
    if _is_scalar(a) and isinstance(b, Raster):
        s = _normalize_scalar(a, argument="a")
        return _dispatch_binary_raster_scalar(
            b, s,
            lambda arr, v: np.isclose(v, arr, rtol=rtol, atol=atol, equal_nan=equal_nan),
            operation="isclose", keep_units=False,
        )
    raise MapAlgebraError("isclose() requires at least one Raster operand.", code="map_algebra_no_raster_operand")

# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------


def _require_boolean(raster: Raster, *, argument: str) -> None:
    if raster.values.dtype != np.dtype(np.bool_):
        raise MapAlgebraDTypeError(
            f"Operand '{argument}' must have boolean dtype for this operation.",
            code="map_algebra_requires_boolean",
            details={"argument": argument, "dtype": str(raster.values.dtype)},
        )


def logical_not(a: Raster) -> Raster:
    _require_boolean(a, argument="a")
    return _dispatch_unary(a, _logical_not, operation="logical_not")


def logical_and(a: Raster, b: Raster) -> Raster:
    _require_boolean(a, argument="a")
    _require_boolean(b, argument="b")
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(a, b, _logical_and, operation="logical_and")


def logical_or(a: Raster, b: Raster) -> Raster:
    _require_boolean(a, argument="a")
    _require_boolean(b, argument="b")
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(a, b, _logical_or, operation="logical_or")


def logical_xor(a: Raster, b: Raster) -> Raster:
    _require_boolean(a, argument="a")
    _require_boolean(b, argument="b")
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(a, b, _logical_xor, operation="logical_xor")

# ---------------------------------------------------------------------------
# Conditional / validity
# ---------------------------------------------------------------------------


def where(
    condition: Raster,
    x: Raster | int | float | _InvalidSentinel,
    y: Raster | int | float | _InvalidSentinel,
) -> Raster:
    """Select branch values using exact shared dtype and unit rules.

    Raster branches must have matching units. Python integer branches are
    represented exactly; if no supported common integer dtype exists, the
    operation raises rather than selecting an inexact floating dtype.
    """
    _require_boolean(condition, argument="condition")
    x_is_invalid = isinstance(x, _InvalidSentinel)
    y_is_invalid = isinstance(y, _InvalidSentinel)
    cond_values = condition.values
    cond_valid = condition.valid

    raster_operands: list[Raster] = [condition]
    x_raster: Raster | None = None
    y_raster: Raster | None = None
    x_scalar: int | float | None = None
    y_scalar: int | float | None = None

    if not x_is_invalid:
        if isinstance(x, Raster):
            raster_operands.append(x)
            x_raster = x
        else:
            x_scalar = _normalize_scalar(x, argument="x")
    if not y_is_invalid:
        if isinstance(y, Raster):
            raster_operands.append(y)
            y_raster = y
        else:
            y_scalar = _normalize_scalar(y, argument="y")

    _require_common_grid(raster_operands)
    if x_is_invalid and y_is_invalid:
        result_values = np.zeros(condition.shape, dtype=condition.values.dtype)
        return Raster(
            values=result_values,
            georef=_infer_output_georef_single(condition),
            valid=np.zeros(condition.shape, dtype=np.bool_),
        )

    branch_rasters = [
        branch for branch in (x_raster, y_raster) if branch is not None
    ]
    if len(branch_rasters) == 2:
        require_matching_units(
            units_a=branch_rasters[0].units,
            units_b=branch_rasters[1].units,
        )
    output_units = branch_rasters[0].units if branch_rasters else None
    output_dtype = result_dtype(
        tuple(branch.dtype for branch in branch_rasters),
        operation="where",
        scalars=tuple(
            scalar for scalar in (x_scalar, y_scalar) if scalar is not None
        ),
    )
    all_valid = np.ones(condition.shape, dtype=np.bool_)
    all_invalid = np.zeros(condition.shape, dtype=np.bool_)

    def branch_arrays(
        branch: Raster | None,
        scalar: int | float | None,
    ) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, np.dtype[np.bool_]],
    ]:
        if branch is not None:
            return branch.values.astype(output_dtype, copy=False), branch.valid
        if scalar is not None:
            return np.asarray(scalar, dtype=output_dtype), all_valid
        return np.asarray(0, dtype=output_dtype), all_invalid

    x_values, x_valid = branch_arrays(x_raster, x_scalar)
    y_values, y_valid = branch_arrays(y_raster, y_scalar)
    result_values = np.where(cond_values, x_values, y_values)
    result_valid = where_validity(
        cond_values, cond_valid, x_valid, y_valid,
    )
    return Raster(
        values=result_values,
        georef=_infer_output_georef_single(condition),
        valid=result_valid,
        units=output_units,
    )


def coalesce(*operands: Raster | int | float) -> Raster:
    """Select the first valid value without floating-point intermediates.

    Raster operands must have matching units. Python integer fallbacks are
    represented exactly; incompatible signed/unsigned 64-bit domains raise a
    structured dtype error rather than being combined through FP64.
    """
    normalized = tuple(
        _as_raster_operand(op, argument=f"operands[{index}]")
        for index, op in enumerate(operands)
    )
    rasters = [op for op in normalized if isinstance(op, Raster)]
    if not rasters:
        raise MapAlgebraError("coalesce() requires at least one Raster operand.", code="map_algebra_no_raster_operand")
    _require_common_grid(rasters)
    for raster in rasters[1:]:
        require_matching_units(units_a=rasters[0].units, units_b=raster.units)

    scalars = tuple(op for op in normalized if not isinstance(op, Raster))
    target_dtype = result_dtype(
        tuple(raster.dtype for raster in rasters),
        operation="coalesce",
        scalars=scalars,
    )
    result_values = np.zeros(rasters[0].shape, dtype=target_dtype)
    result_valid = np.zeros(rasters[0].shape, dtype=np.bool_)

    for op in normalized:
        still_invalid = ~result_valid
        if not np.any(still_invalid):
            break
        if isinstance(op, Raster):
            selected = still_invalid & op.valid
            result_values[selected] = op.values[selected].astype(
                target_dtype, copy=False,
            )
            result_valid[selected] = True
        else:
            result_values[still_invalid] = np.asarray(op, dtype=target_dtype)
            result_valid[still_invalid] = True
    return Raster(
        values=result_values,
        georef=_infer_output_georef_single(rasters[0]),
        valid=result_valid,
        units=rasters[0].units,
    )


def is_valid(a: Raster) -> Raster:
    return Raster(
        values=a.valid.copy(),
        georef=a.georef,
        valid=np.ones(a.shape, dtype=np.bool_),
        name=f"is_valid({a.name})" if a.name else None,
    )


def is_invalid(a: Raster) -> Raster:
    return Raster(
        values=~a.valid,
        georef=a.georef,
        valid=np.ones(a.shape, dtype=np.bool_),
        name=f"is_invalid({a.name})" if a.name else None,
    )


def set_invalid(raster: Raster, mask: Raster) -> Raster:
    _require_boolean(mask, argument="mask")
    _require_common_grid([raster, mask])
    effective_mask = mask.valid & mask.values
    new_valid = raster.valid & ~effective_mask
    return Raster(
        values=raster.values, georef=raster.georef,
        valid=new_valid, units=raster.units, name=raster.name,
    )


def fill_invalid(raster: Raster, value: int | float) -> Raster:
    """Fill invalid cells using a value exactly representable by the dtype."""
    filled_values, _validated = _fill_invalid_exact(
        raster.values, raster.valid, value,
    )
    return Raster(
        values=filled_values, georef=raster.georef,
        valid=np.ones(raster.shape, dtype=np.bool_),
        units=raster.units, name=raster.name,
    )

# ---------------------------------------------------------------------------
# Clip and cast
# ---------------------------------------------------------------------------


def clip(
    raster: Raster,
    *,
    lower: int | float | None = None,
    upper: int | float | None = None,
) -> Raster:
    if lower is None and upper is None:
        return raster
    result_values = _clip(raster.values, lower, upper)
    result_valid = raster.valid.copy()
    return Raster(
        values=result_values, georef=raster.georef,
        valid=result_valid, units=raster.units, name=raster.name,
    )


def cast(
    raster: Raster,
    dtype: np.dtype[Any] | str,
    *,
    casting: str = "safe",
    overflow: CastOverflowPolicy = "raise",
) -> Raster:
    target_dtype = normalize_dtype(dtype, operation="cast")
    result_values = cast_values(
        raster.values,
        target_dtype,
        casting=casting,  # type: ignore[arg-type]
        overflow=overflow,
        valid=raster.valid,
    )
    return Raster(
        values=result_values, georef=raster.georef,
        valid=raster.valid, units=raster.units, name=raster.name,
    )

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def reclassify_values(
    raster: Raster,
    mapping: Mapping[int | float, int | float],
    *,
    default: int | float | Literal["preserve", "invalidate"] | None = "invalidate",
) -> Raster:
    raster = _require_eager_raster(raster)
    default = "invalidate" if default is None else default
    if isinstance(default, str) and default not in {"preserve", "invalidate"}:
        raise MapAlgebraError(
            "default must be a numeric value, 'preserve', or 'invalidate'.",
            code="map_algebra_invalid_reclassification_default",
            details={"default": default},
        )
    output_values = list(mapping.values())
    if not isinstance(default, str):
        output_values.append(default)
    target_dtype = _classification_dtype(
        output_values,
        fallback=raster.dtype,
        preserve=default == "preserve",
    )
    result = (
        raster.values.astype(target_dtype, copy=True)
        if default == "preserve"
        else np.zeros(raster.shape, dtype=target_dtype)
    )
    matched = np.zeros(raster.shape, dtype=np.bool_)
    for old_val, new_val in mapping.items():
        mask = raster.values == old_val
        result[mask] = new_val
        matched |= mask
    result_valid = raster.valid.copy()
    if default == "invalidate":
        result_valid[~matched] = False
    elif not isinstance(default, str):
        result[~matched] = default
    return Raster(
        values=result,
        georef=raster.georef,
        valid=result_valid,
        units=raster.units,
        name=raster.name,
    )


def reclassify_ranges(
    raster: Raster,
    ranges: Sequence[tuple[int | float, int | float, int | float]],
    *,
    default: int | float | Literal["preserve", "invalidate"] | None = "invalidate",
) -> Raster:
    raster = _require_eager_raster(raster)
    default = "invalidate" if default is None else default
    if isinstance(default, str) and default not in {"preserve", "invalidate"}:
        raise MapAlgebraError(
            "default must be a numeric value, 'preserve', or 'invalidate'.",
            code="map_algebra_invalid_reclassification_default",
            details={"default": default},
        )
    normalized_ranges = tuple(ranges)
    for index, (lower, upper, _) in enumerate(normalized_ranges):
        if not lower < upper:
            raise MapAlgebraError(
                "Every reclassification range must have lower < upper.",
                code="map_algebra_invalid_reclassification_range",
                details={"index": index, "lower": lower, "upper": upper},
            )
    output_values = [new_val for _, _, new_val in normalized_ranges]
    if not isinstance(default, str):
        output_values.append(default)
    target_dtype = _classification_dtype(
        output_values,
        fallback=raster.dtype,
        preserve=default == "preserve",
    )
    result = (
        raster.values.astype(target_dtype, copy=True)
        if default == "preserve"
        else np.zeros(raster.shape, dtype=target_dtype)
    )
    matched = np.zeros(raster.shape, dtype=np.bool_)
    for lower, upper, new_val in normalized_ranges:
        mask = (raster.values >= lower) & (raster.values < upper)
        result[mask] = new_val
        matched |= mask
    result_valid = raster.valid.copy()
    if default == "invalidate":
        result_valid[~matched] = False
    elif not isinstance(default, str):
        result[~matched] = default
    return Raster(
        values=result,
        georef=raster.georef,
        valid=result_valid,
        units=raster.units,
        name=raster.name,
    )


def digitize(
    raster: Raster,
    bins: Sequence[int | float],
    *,
    right: bool = False,
) -> Raster:
    raster = _require_eager_raster(raster)
    bins_arr = np.asarray(tuple(bins))
    if bins_arr.ndim != 1 or bins_arr.dtype.kind not in "iuf" or not np.all(bins_arr[:-1] <= bins_arr[1:]):
        raise MapAlgebraError(
            "bins must be a one-dimensional monotonically increasing numeric sequence.",
            code="map_algebra_invalid_bins",
        )
    result_values = np.digitize(raster.values, bins_arr, right=right)
    result_valid = raster.valid.copy()
    return Raster(
        values=result_values,
        georef=raster.georef,
        valid=result_valid,
        units=None,
        name=raster.name,
    )


def one_hot(
    raster: Raster,
    classes: Sequence[int | float],
) -> tuple[Raster, ...]:
    """Return one Boolean raster per class, in caller-supplied order."""
    raster = _require_eager_raster(raster)
    normalized_classes = tuple(classes)
    if not normalized_classes:
        raise MapAlgebraError(
            "classes must contain at least one value.",
            code="map_algebra_empty_classes",
        )
    return tuple(
        Raster(
            values=raster.values == class_value,
            georef=raster.georef,
            valid=raster.valid.copy(),
            units=None,
            name=f"{raster.name or 'raster'}_{class_value}",
        )
        for class_value in normalized_classes
    )


def _classification_dtype(
    values: Sequence[Any],
    *,
    fallback: np.dtype[Any],
    preserve: bool = False,
) -> np.dtype[Any]:
    """Infer an exact reclassification output through the shared dtype path."""
    if not values:
        return fallback
    return result_dtype(
        (np.dtype(fallback),) if preserve else (),
        operation="reclassify",
        scalars=tuple(values),
    )


def _require_eager_raster(value: Any) -> Raster:
    operand = _as_raster_operand(value, argument="raster")
    if not isinstance(operand, Raster):
        raise MapAlgebraError(
            "This operation requires a Raster operand.",
            code="map_algebra_no_raster_operand",
            details={"type": type(value).__name__},
        )
    return operand

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_minmax(
    raster: Raster,
    *,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
) -> Raster:
    raster = _require_eager_raster(raster)
    output_dtype = result_dtype(
        (raster.dtype,),
        operation="normalize_minmax",
        scalars=(minimum, maximum),
    )
    valid_data = raster.values[raster.valid]
    if valid_data.size == 0:
        return Raster(
            values=np.full(raster.shape, np.nan, dtype=output_dtype),
            georef=raster.georef,
            valid=np.zeros(raster.shape, dtype=np.bool_),
            units=None,
            name=raster.name,
        )
    working_data = valid_data.astype(output_dtype, copy=False)
    vmin = output_dtype.type(working_data.min() if minimum is None else minimum)
    vmax = output_dtype.type(working_data.max() if maximum is None else maximum)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax < vmin:
        raise MapAlgebraError(
            "minimum and maximum must be finite with maximum >= minimum.",
            code="map_algebra_invalid_normalization_statistics",
            details={"minimum": float(vmin), "maximum": float(vmax)},
        )
    if vmax == vmin:
        result_values = np.zeros(raster.shape, dtype=output_dtype)
        result_valid = np.zeros(raster.shape, dtype=np.bool_)
    else:
        values = raster.values.astype(output_dtype, copy=False)
        result_values = (values - vmin) / (vmax - vmin)
        result_valid = raster.valid & np.isfinite(result_values)
    return Raster(
        values=result_values.astype(output_dtype, copy=False),
        georef=raster.georef,
        valid=result_valid,
        units=None,
        name=raster.name,
    )


def standardize(
    raster: Raster,
    *,
    mean: int | float | None = None,
    std: int | float | None = None,
    ddof: float = 0,
) -> Raster:
    raster = _require_eager_raster(raster)
    output_dtype = result_dtype(
        (raster.dtype,),
        operation="standardize",
        scalars=(mean, std),
    )
    valid_data = raster.values[raster.valid]
    if valid_data.size == 0:
        return Raster(
            values=np.full(raster.shape, np.nan, dtype=output_dtype),
            georef=raster.georef,
            valid=np.zeros(raster.shape, dtype=np.bool_),
            units=None,
            name=raster.name,
        )
    if not np.isfinite(ddof) or ddof < 0:
        raise MapAlgebraError(
            "ddof must be finite and non-negative.",
            code="map_algebra_invalid_ddof",
            details={"ddof": ddof},
        )
    working_data = valid_data.astype(output_dtype, copy=False)
    vmean = output_dtype.type(
        working_data.mean(dtype=output_dtype) if mean is None else mean
    )
    vstd = output_dtype.type(
        working_data.std(ddof=ddof, dtype=output_dtype) if std is None else std
    )
    if not np.isfinite(vmean) or not np.isfinite(vstd) or vstd < 0:
        raise MapAlgebraError(
            "mean must be finite and std must be finite and non-negative.",
            code="map_algebra_invalid_normalization_statistics",
            details={"mean": float(vmean), "std": float(vstd)},
        )
    if vstd == 0:
        result_valid = np.zeros(raster.shape, dtype=np.bool_)
        result_values = np.zeros(raster.shape, dtype=output_dtype)
    else:
        values = raster.values.astype(output_dtype, copy=False)
        result_values = (values - vmean) / vstd
        result_valid = raster.valid & np.isfinite(result_values)
    return Raster(
        values=result_values.astype(output_dtype, copy=False),
        georef=raster.georef,
        valid=result_valid,
        units=None,
        name=raster.name,
    )

# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def sqrt(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(a, _sqrt, operation="sqrt", numeric_errors=numeric_errors)


def square(
    a: Raster,
    *,
    overflow: OverflowPolicy = "raise",
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    return _dispatch_unary(a, _square, operation="square", overflow=overflow, numeric_errors=numeric_errors)


def exp(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(a, _exp, operation="exp", numeric_errors=numeric_errors)


def log(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(a, _log, operation="log", numeric_errors=numeric_errors)


def log10(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(a, _log10, operation="log10", numeric_errors=numeric_errors)


def sin(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    angle_unit = require_angle_units(a.units, argument="a")
    if angle_unit == "degrees":
        a_rad = _dispatch_unary(
            a, _radians, operation="to_radians", keep_units=False,
            numeric_errors=numeric_errors,
        )
        result = _dispatch_unary(
            a_rad, _sin, operation="sin", keep_units=False,
            numeric_errors=numeric_errors,
        )
    else:
        result = _dispatch_unary(
            a, _sin, operation="sin", keep_units=False,
            numeric_errors=numeric_errors,
        )
    return result


def cos(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    angle_unit = require_angle_units(a.units, argument="a")
    if angle_unit == "degrees":
        a_rad = _dispatch_unary(
            a, _radians, operation="to_radians", keep_units=False,
            numeric_errors=numeric_errors,
        )
        result = _dispatch_unary(
            a_rad, _cos, operation="cos", keep_units=False,
            numeric_errors=numeric_errors,
        )
    else:
        result = _dispatch_unary(
            a, _cos, operation="cos", keep_units=False,
            numeric_errors=numeric_errors,
        )
    return result


def tan(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    angle_unit = require_angle_units(a.units, argument="a")
    if angle_unit == "degrees":
        a_rad = _dispatch_unary(
            a, _radians, operation="to_radians", keep_units=False,
            numeric_errors=numeric_errors,
        )
        result = _dispatch_unary(
            a_rad, _tan, operation="tan", keep_units=False,
            numeric_errors=numeric_errors,
        )
    else:
        result = _dispatch_unary(
            a, _tan, operation="tan", keep_units=False,
            numeric_errors=numeric_errors,
        )
    return result


def arcsin(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(a, _arcsin, operation="arcsin", output_units="radians", keep_units=False, numeric_errors=numeric_errors)


def arccos(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(a, _arccos, operation="arccos", output_units="radians", keep_units=False, numeric_errors=numeric_errors)


def arctan(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(
        a, _arctan, operation="arctan", output_units="radians",
        keep_units=False, numeric_errors=numeric_errors,
    )


def arctan2(
    a: Raster,
    b: Raster,
    *,
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(
        a, b, _arctan2, operation="arctan2", output_units="radians",
        numeric_errors=numeric_errors,
    )


def degrees(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(
        a, _degrees, operation="degrees", output_units="degrees",
        keep_units=False, numeric_errors=numeric_errors,
    )


def radians(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(
        a, _radians, operation="radians", output_units="radians",
        keep_units=False, numeric_errors=numeric_errors,
    )


def hypot(
    a: Raster,
    b: Raster,
    *,
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(
        a, b, _hypot, operation="hypot", numeric_errors=numeric_errors,
    )


def round_half_even(
    a: Raster,
    ndigits: int = 0,
    *,
    numeric_errors: NumericErrorsPolicy = "invalid",
) -> Raster:
    if ndigits == 0:
        return _dispatch_unary(
            a, _round_half_even, operation="round",
            numeric_errors=numeric_errors,
        )
    factor = 10.0**ndigits
    scaled = _dispatch_binary_raster_scalar(
        a, factor, lambda arr, s: arr * s, operation="round_scale",
        numeric_errors=numeric_errors,
    )
    rounded = _dispatch_unary(
        scaled, _round_half_even, operation="round",
        numeric_errors=numeric_errors,
    )
    return _dispatch_binary_raster_scalar(
        rounded, 1.0 / factor, lambda arr, s: arr * s,
        operation="round_unscale", numeric_errors=numeric_errors,
    )


def floor(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(a, _floor, operation="floor", numeric_errors=numeric_errors)


def ceil(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(a, _ceil, operation="ceil", numeric_errors=numeric_errors)


def trunc(a: Raster, *, numeric_errors: NumericErrorsPolicy = "invalid") -> Raster:
    return _dispatch_unary(a, _trunc, operation="trunc", numeric_errors=numeric_errors)
