from __future__ import annotations

from typing import Any, NoReturn

import numpy as np

from ..errors import MapAlgebraDTypeError, MapAlgebraError
from ..raster import Raster, _validate_nodata_representable
from ._dtypes import cast_values
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
    require_angle_units,
    require_matching_units,
)
from ._validation import _is_scalar, _normalize_scalar, _require_common_grid
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


def add(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(a, b, _add, operation="add", output_units=a.units)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _add, operation="add")
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(b, _normalize_scalar(a, argument="a"), _add, operation="add")
    raise MapAlgebraError("add() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def subtract(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(a, b, _subtract, operation="subtract", output_units=a.units)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _subtract, operation="subtract")
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"),
            lambda arr, s: _subtract(np.full(arr.shape, s, dtype=arr.dtype), arr),
            operation="subtract",
        )
    raise MapAlgebraError("subtract() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def multiply(a: Raster | int | float, b: Raster | int | float, *, output_units: str | None = None) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        units = multiply_units(units_a=a.units, units_b=b.units, output_units=output_units)
        return _dispatch_binary_raster_raster(a, b, _multiply, operation="multiply", output_units=units)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _multiply, operation="multiply")
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(b, _normalize_scalar(a, argument="a"), _multiply, operation="multiply")
    raise MapAlgebraError("multiply() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def divide(a: Raster | int | float, b: Raster | int | float, *, output_units: str | None = None) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        units = multiply_units(units_a=a.units, units_b=b.units, output_units=output_units)
        return _dispatch_binary_raster_raster(a, b, _divide, operation="divide", output_units=units)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _divide, operation="divide")
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"),
            lambda arr, s: np.divide(np.full(arr.shape, s, dtype=arr.dtype), arr),
            operation="divide",
        )
    raise MapAlgebraError("divide() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def floor_divide(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        return _dispatch_binary_raster_raster(a, b, _floor_divide, operation="floor_divide")
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _floor_divide, operation="floor_divide")
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"),
            lambda arr, s: _floor_divide(np.full(arr.shape, s, dtype=arr.dtype), arr),
            operation="floor_divide",
        )
    raise MapAlgebraError("floor_divide() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def remainder(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        return _dispatch_binary_raster_raster(a, b, _remainder, operation="remainder")
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _remainder, operation="remainder")
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"),
            lambda arr, s: _remainder(np.full(arr.shape, s, dtype=arr.dtype), arr),
            operation="remainder",
        )
    raise MapAlgebraError("remainder() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def power(base: Raster | int | float, exponent: Raster | int | float) -> Raster:
    if isinstance(base, Raster) and isinstance(exponent, Raster):
        _require_common_grid([base, exponent])
        return _dispatch_binary_raster_raster(base, exponent, _power, operation="power")
    if isinstance(base, Raster) and _is_scalar(exponent):
        return _dispatch_binary_raster_scalar(base, _normalize_scalar(exponent, argument="exponent"), _power, operation="power")
    if _is_scalar(base) and isinstance(exponent, Raster):
        return _dispatch_binary_raster_scalar(
            exponent, _normalize_scalar(base, argument="base"),
            lambda arr, s: _power(np.full(arr.shape, s, dtype=arr.dtype), arr),
            operation="power",
        )
    raise MapAlgebraError("power() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def negative(a: Raster) -> Raster:
    return _dispatch_unary(a, _negate, operation="negative")


def positive(a: Raster) -> Raster:
    return a


def absolute(a: Raster) -> Raster:
    return _dispatch_unary(a, _absolute, operation="absolute")

# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------


def minimum(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(a, b, _minimum, operation="minimum", output_units=a.units)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _minimum, operation="minimum")
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(b, _normalize_scalar(a, argument="a"), _minimum, operation="minimum")
    raise MapAlgebraError("minimum() requires at least one Raster operand.", code="map_algebra_no_raster_operand")


def maximum(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(a, b, _maximum, operation="maximum", output_units=a.units)
    if isinstance(a, Raster) and _is_scalar(b):
        return _dispatch_binary_raster_scalar(a, _normalize_scalar(b, argument="b"), _maximum, operation="maximum")
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(b, _normalize_scalar(a, argument="a"), _maximum, operation="maximum")
    raise MapAlgebraError("maximum() requires at least one Raster operand.", code="map_algebra_no_raster_operand")

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
        )
    if _is_scalar(a) and isinstance(b, Raster):
        return _dispatch_binary_raster_scalar(
            b, _normalize_scalar(a, argument="a"), swapped_kernel, operation=operation,
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
            operation="isclose",
        )
    if _is_scalar(a) and isinstance(b, Raster):
        s = _normalize_scalar(a, argument="a")
        return _dispatch_binary_raster_scalar(
            b, s,
            lambda arr, v: np.isclose(v, arr, rtol=rtol, atol=atol, equal_nan=equal_nan),
            operation="isclose",
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
    georef = _infer_output_georef_single(condition)

    if x_raster is not None and y_raster is not None:
        result_values = np.where(cond_values, x_raster.values, y_raster.values)
        result_valid = where_validity(cond_values, cond_valid, x_raster.valid, y_raster.valid)
        return Raster(values=result_values, georef=georef, valid=result_valid)

    if x_raster is not None and y_is_invalid:
        result_values = np.where(cond_values, x_raster.values, 0)
        result_valid = cond_valid & cond_values & x_raster.valid
        return Raster(values=result_values, georef=georef, valid=result_valid, units=x_raster.units)

    if x_is_invalid and y_raster is not None:
        result_values = np.where(cond_values, 0, y_raster.values)
        result_valid = cond_valid & ~cond_values & y_raster.valid
        return Raster(values=result_values, georef=georef, valid=result_valid, units=y_raster.units)

    if x_raster is not None and y_scalar is not None:
        result_values = np.where(cond_values, x_raster.values, y_scalar)
        result_valid = where_validity(
            cond_values, cond_valid, x_raster.valid,
            np.ones(condition.shape, dtype=np.bool_),
        )
        return Raster(values=result_values, georef=georef, valid=result_valid, units=x_raster.units)

    if x_scalar is not None and y_raster is not None:
        result_values = np.where(cond_values, x_scalar, y_raster.values)
        result_valid = where_validity(
            cond_values, cond_valid,
            np.ones(condition.shape, dtype=np.bool_),
            y_raster.valid,
        )
        return Raster(values=result_values, georef=georef, valid=result_valid, units=y_raster.units)

    if x_scalar is not None and y_scalar is not None:
        result_values = np.where(cond_values, x_scalar, y_scalar)
        return Raster(values=result_values, georef=georef, valid=cond_valid.copy())

    if x_scalar is not None and y_is_invalid:
        result_values = np.where(cond_values, x_scalar, 0)
        result_valid = cond_valid & cond_values
        return Raster(values=result_values, georef=georef, valid=result_valid)

    if x_is_invalid and y_scalar is not None:
        result_values = np.where(cond_values, 0, y_scalar)
        result_valid = cond_valid & ~cond_values
        return Raster(values=result_values, georef=georef, valid=result_valid)

    if x_is_invalid and y_is_invalid:
        result_values = np.zeros(condition.shape, dtype=condition.values.dtype)
        return Raster(values=result_values, georef=georef, valid=np.zeros(condition.shape, dtype=np.bool_))

    raise MapAlgebraError(
        "where() requires at least one non-invalid Raster or scalar in the x or y branches.",
        code="map_algebra_invalid_where",
    )


def coalesce(*operands: Raster | int | float) -> Raster:
    rasters: list[Raster] = []
    for op in operands:
        if isinstance(op, Raster):
            rasters.append(op)
    if not rasters:
        raise MapAlgebraError("coalesce() requires at least one Raster operand.", code="map_algebra_no_raster_operand")
    _require_common_grid(rasters)

    result_values = rasters[0].values.astype(np.float64, copy=True)
    result_valid = rasters[0].valid.copy()

    for op in operands:
        if op is operands[0]:
            continue
        still_invalid = ~result_valid
        if not np.any(still_invalid):
            break
        if isinstance(op, Raster):
            result_values[still_invalid] = op.values[still_invalid].astype(np.float64)
            result_valid = result_valid | op.valid
        else:
            scalar = float(op)
            result_values[still_invalid] = scalar
            result_valid[still_invalid] = True

    target_dtype = np.result_type(*[r.values.dtype for r in rasters])
    return Raster(
        values=result_values.astype(target_dtype, copy=False),
        georef=_infer_output_georef_single(rasters[0]),
        valid=result_valid,
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
    validated = _validate_nodata_representable(value, raster.values.dtype)
    if validated is None:
        raise MapAlgebraError(
            "fill_invalid() value must not be None.",
            code="map_algebra_invalid_fill",
        )
    filled_values = raster.values.copy()
    filled_values[~raster.valid] = validated
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
) -> Raster:
    target_dtype = np.dtype(dtype)
    result_values = cast_values(raster.values, target_dtype, casting=casting)  # type: ignore[arg-type]
    return Raster(
        values=result_values, georef=raster.georef,
        valid=raster.valid, units=raster.units, name=raster.name,
    )

# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def sqrt(a: Raster) -> Raster:
    return _dispatch_unary(a, _sqrt, operation="sqrt")


def square(a: Raster) -> Raster:
    return _dispatch_unary(a, _square, operation="square")


def exp(a: Raster) -> Raster:
    return _dispatch_unary(a, _exp, operation="exp")


def log(a: Raster) -> Raster:
    return _dispatch_unary(a, _log, operation="log")


def log10(a: Raster) -> Raster:
    return _dispatch_unary(a, _log10, operation="log10")


def sin(a: Raster) -> Raster:
    angle_unit = require_angle_units(a.units, argument="a")
    if angle_unit == "degrees":
        a_rad = _dispatch_unary(a, _radians, operation="to_radians", keep_units=False)
        result = _dispatch_unary(a_rad, _sin, operation="sin", keep_units=False)
    else:
        result = _dispatch_unary(a, _sin, operation="sin", keep_units=False)
    return result


def cos(a: Raster) -> Raster:
    angle_unit = require_angle_units(a.units, argument="a")
    if angle_unit == "degrees":
        a_rad = _dispatch_unary(a, _radians, operation="to_radians", keep_units=False)
        result = _dispatch_unary(a_rad, _cos, operation="cos", keep_units=False)
    else:
        result = _dispatch_unary(a, _cos, operation="cos", keep_units=False)
    return result


def tan(a: Raster) -> Raster:
    angle_unit = require_angle_units(a.units, argument="a")
    if angle_unit == "degrees":
        a_rad = _dispatch_unary(a, _radians, operation="to_radians", keep_units=False)
        result = _dispatch_unary(a_rad, _tan, operation="tan", keep_units=False)
    else:
        result = _dispatch_unary(a, _tan, operation="tan", keep_units=False)
    return result


def arcsin(a: Raster) -> Raster:
    return _dispatch_unary(a, _arcsin, operation="arcsin", output_units="radians", keep_units=False)


def arccos(a: Raster) -> Raster:
    return _dispatch_unary(a, _arccos, operation="arccos", output_units="radians", keep_units=False)


def arctan(a: Raster) -> Raster:
    return _dispatch_unary(a, _arctan, operation="arctan", output_units="radians", keep_units=False)


def arctan2(a: Raster, b: Raster) -> Raster:
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(a, b, _arctan2, operation="arctan2", output_units="radians")


def degrees(a: Raster) -> Raster:
    return _dispatch_unary(a, _degrees, operation="degrees", output_units="degrees", keep_units=False)


def radians(a: Raster) -> Raster:
    return _dispatch_unary(a, _radians, operation="radians", output_units="radians", keep_units=False)


def hypot(a: Raster, b: Raster) -> Raster:
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(a, b, _hypot, operation="hypot")


def round_half_even(a: Raster, ndigits: int = 0) -> Raster:
    if ndigits == 0:
        return _dispatch_unary(a, _round_half_even, operation="round")
    factor = 10.0**ndigits
    scaled = _dispatch_binary_raster_scalar(a, factor, lambda arr, s: arr * s, operation="round_scale")
    rounded = _dispatch_unary(scaled, _round_half_even, operation="round")
    return _dispatch_binary_raster_scalar(rounded, 1.0 / factor, lambda arr, s: arr * s, operation="round_unscale")


def floor(a: Raster) -> Raster:
    return _dispatch_unary(a, _floor, operation="floor")


def ceil(a: Raster) -> Raster:
    return _dispatch_unary(a, _ceil, operation="ceil")


def trunc(a: Raster) -> Raster:
    return _dispatch_unary(a, _trunc, operation="trunc")
