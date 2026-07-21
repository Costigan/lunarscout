from __future__ import annotations

from typing import Any, NoReturn

import numpy as np

from ..errors import MapAlgebraDTypeError, MapAlgebraError, RasterValidationError
from ..raster import Raster
from ._dtypes import cast_values, result_dtype
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
    _divide_from,
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
    _power_from,
    _radians,
    _remainder,
    _round_half_even,
    _sin,
    _sqrt,
    _square,
    _subtract,
    _subtract_from,
    _tan,
    _trunc,
    _degrees,
)
from ._units import require_matching_units
from ._validation import _is_scalar, _normalize_scalar, _require_common_grid
from ._validity import (
    coalesce_validity,
    intersect_validity,
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

# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def add(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(
            a, b, _add, operation="add", output_units=a.units
        )
    if isinstance(a, Raster) and _is_scalar(b):
        scalar = _normalize_scalar(b, argument="b")
        return _dispatch_binary_raster_scalar(a, scalar, _add, operation="add")
    if _is_scalar(a) and isinstance(b, Raster):
        scalar = _normalize_scalar(a, argument="a")
        return _dispatch_binary_raster_scalar(b, scalar, _add, operation="add")
    raise MapAlgebraError(
        "add() requires at least one Raster operand.",
        code="map_algebra_no_raster_operand",
    )


def subtract(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(
            a, b, _subtract, operation="subtract", output_units=a.units
        )
    if isinstance(a, Raster) and _is_scalar(b):
        scalar = _normalize_scalar(b, argument="b")
        return _dispatch_binary_raster_scalar(a, scalar, _subtract, operation="subtract")
    if _is_scalar(a) and isinstance(b, Raster):
        scalar = _normalize_scalar(a, argument="a")
        return _dispatch_binary_raster_scalar(
            b, scalar,
            lambda arr, s: _subtract_from(s, arr),
            operation="subtract",
        )
    raise MapAlgebraError(
        "subtract() requires at least one Raster operand.",
        code="map_algebra_no_raster_operand",
    )


def multiply(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        return _dispatch_binary_raster_raster(
            a, b, _multiply, operation="multiply", output_units=a.units
        )
    if isinstance(a, Raster) and _is_scalar(b):
        scalar = _normalize_scalar(b, argument="b")
        return _dispatch_binary_raster_scalar(a, scalar, _multiply, operation="multiply")
    if _is_scalar(a) and isinstance(b, Raster):
        scalar = _normalize_scalar(a, argument="a")
        return _dispatch_binary_raster_scalar(b, scalar, _multiply, operation="multiply")
    raise MapAlgebraError(
        "multiply() requires at least one Raster operand.",
        code="map_algebra_no_raster_operand",
    )


def divide(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        return _dispatch_binary_raster_raster(
            a, b, _divide, operation="divide"
        )
    if isinstance(a, Raster) and _is_scalar(b):
        scalar = _normalize_scalar(b, argument="b")
        return _dispatch_binary_raster_scalar(a, scalar, _divide, operation="divide")
    if _is_scalar(a) and isinstance(b, Raster):
        scalar = _normalize_scalar(a, argument="a")
        return _dispatch_binary_raster_scalar(
            b, scalar,
            lambda arr, s: _divide_from(s, arr),
            operation="divide",
        )
    raise MapAlgebraError(
        "divide() requires at least one Raster operand.",
        code="map_algebra_no_raster_operand",
    )


def negative(a: Raster) -> Raster:
    return _dispatch_unary(a, _negate, operation="negative")


def absolute(a: Raster) -> Raster:
    return _dispatch_unary(a, _absolute, operation="absolute")


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------


def minimum(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(
            a, b, _minimum, operation="minimum", output_units=a.units
        )
    if isinstance(a, Raster) and _is_scalar(b):
        scalar = _normalize_scalar(b, argument="b")
        return _dispatch_binary_raster_scalar(a, scalar, _minimum, operation="minimum")
    if _is_scalar(a) and isinstance(b, Raster):
        scalar = _normalize_scalar(a, argument="a")
        return _dispatch_binary_raster_scalar(b, scalar, _minimum, operation="minimum")
    raise MapAlgebraError(
        "minimum() requires at least one Raster operand.",
        code="map_algebra_no_raster_operand",
    )


def maximum(a: Raster | int | float, b: Raster | int | float) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(
            a, b, _maximum, operation="maximum", output_units=a.units
        )
    if isinstance(a, Raster) and _is_scalar(b):
        scalar = _normalize_scalar(b, argument="b")
        return _dispatch_binary_raster_scalar(a, scalar, _maximum, operation="maximum")
    if _is_scalar(a) and isinstance(b, Raster):
        scalar = _normalize_scalar(a, argument="a")
        return _dispatch_binary_raster_scalar(b, scalar, _maximum, operation="maximum")
    raise MapAlgebraError(
        "maximum() requires at least one Raster operand.",
        code="map_algebra_no_raster_operand",
    )


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def _comparison_helper(
    a: Raster | int | float,
    b: Raster | int | float,
    kernel,
    operation: str,
) -> Raster:
    if isinstance(a, Raster) and isinstance(b, Raster):
        _require_common_grid([a, b])
        require_matching_units(units_a=a.units, units_b=b.units)
        return _dispatch_binary_raster_raster(
            a, b, kernel, operation=operation, output_units=None
        )
    if isinstance(a, Raster) and _is_scalar(b):
        scalar = _normalize_scalar(b, argument="b")
        return _dispatch_binary_raster_scalar(a, scalar, kernel, operation=operation)
    if _is_scalar(a) and isinstance(b, Raster):
        scalar = _normalize_scalar(a, argument="a")
        return _dispatch_binary_raster_scalar(b, scalar, kernel, operation=operation)
    raise MapAlgebraError(
        f"{operation}() requires at least one Raster operand.",
        code="map_algebra_no_raster_operand",
    )


def less(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _less, "less")


def less_equal(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _less_equal, "less_equal")


def greater(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _greater, "greater")


def greater_equal(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _greater_equal, "greater_equal")


def equal(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _equal, "equal")


def not_equal(a: Raster | int | float, b: Raster | int | float) -> Raster:
    return _comparison_helper(a, b, _not_equal, "not_equal")


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
    return _dispatch_binary_raster_raster(
        a, b, _logical_and, operation="logical_and",
    )


def logical_or(a: Raster, b: Raster) -> Raster:
    _require_boolean(a, argument="a")
    _require_boolean(b, argument="b")
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(
        a, b, _logical_or, operation="logical_or",
    )


def logical_xor(a: Raster, b: Raster) -> Raster:
    _require_boolean(a, argument="a")
    _require_boolean(b, argument="b")
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(
        a, b, _logical_xor, operation="logical_xor",
    )


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
    raster_operands: list[Raster] = [condition]
    x_raster: Raster | None = None
    y_raster: Raster | None = None

    if not x_is_invalid:
        if isinstance(x, Raster):
            raster_operands.append(x)
            x_raster = x
        else:
            x = _normalize_scalar(x, argument="x")
    if not y_is_invalid:
        if isinstance(y, Raster):
            raster_operands.append(y)
            y_raster = y
        else:
            y = _normalize_scalar(y, argument="y")

    if len(raster_operands) == 0:
        raise MapAlgebraError(
            "where() requires at least one Raster operand.",
            code="map_algebra_no_raster_operand",
        )
    _require_common_grid(raster_operands)

    cond_values = condition.values
    cond_valid = condition.valid

    if x_raster is not None and y_raster is not None:
        result_values = np.where(condition.values, x_raster.values, y_raster.values)
        result_valid = where_validity(
            cond_values, cond_valid, x_raster.valid, y_raster.valid,
        )
        return Raster(
            values=result_values,
            georef=_infer_output_georef_single(condition),
            valid=result_valid,
        )

    if x_raster is not None and y_is_invalid:
        result_values = np.where(condition.values, x_raster.values, 0)
        result_valid = cond_valid & condition.values & x_raster.valid
        return Raster(
            values=result_values,
            georef=_infer_output_georef_single(condition),
            valid=result_valid,
        )

    if x_is_invalid and y_raster is not None:
        result_values = np.where(condition.values, 0, y_raster.values)
        result_valid = cond_valid & ~condition.values & y_raster.valid
        return Raster(
            values=result_values,
            georef=_infer_output_georef_single(condition),
            valid=result_valid,
        )

    if x_raster is not None and isinstance(y, (int, float)):
        result_values = np.where(condition.values, x_raster.values, y)
        result_valid = where_validity(
            cond_values, cond_valid, x_raster.valid,
            np.ones(condition.shape, dtype=np.bool_),
        )
        return Raster(
            values=result_values,
            georef=_infer_output_georef_single(condition),
            valid=result_valid,
        )

    if isinstance(x, (int, float)) and y_raster is not None:
        result_values = np.where(condition.values, x, y_raster.values)
        result_valid = where_validity(
            cond_values, cond_valid,
            np.ones(condition.shape, dtype=np.bool_),
            y_raster.valid,
        )
        return Raster(
            values=result_values,
            georef=_infer_output_georef_single(condition),
            valid=result_valid,
        )

    raise MapAlgebraError(
        "where() requires at least one non-invalid Raster in the x or y branches.",
        code="map_algebra_invalid_where",
    )


def _infer_output_georef_single(raster: Raster) -> Any:
    from ._validation import _infer_output_georef as _iog

    return _iog([raster])


def coalesce(*rasters: Raster | int | float) -> Raster:
    raster_list: list[Raster] = []
    scalars: list[int | float] = []
    for i, r in enumerate(rasters):
        if isinstance(r, Raster):
            raster_list.append(r)
        elif _is_scalar(r):
            scalars.append(_normalize_scalar(r, argument=f"arg{i}"))
        else:
            raise MapAlgebraError(
                f"coalesce() operands must be Raster or scalar.",
                code="map_algebra_invalid_operand",
            )

    if not raster_list:
        raise MapAlgebraError(
            "coalesce() requires at least one Raster operand.",
            code="map_algebra_no_raster_operand",
        )
    _require_common_grid(raster_list)

    result_values = raster_list[0].values.copy()
    result_valid = raster_list[0].valid.copy()

    for i, raster in enumerate(raster_list[1:], start=1):
        still_invalid = ~result_valid
        result_values[still_invalid] = raster.values[still_invalid]
        result_valid = result_valid | raster.valid

    for i, scalar in enumerate(scalars):
        still_invalid = ~result_valid
        result_values[still_invalid] = scalar
        result_valid[still_invalid] = True

    return Raster(
        values=result_values,
        georef=_infer_output_georef_single(raster_list[0]),
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
    new_valid = raster.valid & ~mask.values
    return Raster(
        values=raster.values,
        georef=raster.georef,
        valid=new_valid,
        units=raster.units,
        name=raster.name,
    )


def fill_invalid(raster: Raster, value: int | float) -> Raster:
    scalar = _normalize_scalar(value, argument="value")
    filled_values = raster.values.copy()
    filled_values[~raster.valid] = scalar
    return Raster(
        values=filled_values,
        georef=raster.georef,
        valid=np.ones(raster.shape, dtype=np.bool_),
        units=raster.units,
        name=raster.name,
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
        values=result_values,
        georef=raster.georef,
        valid=result_valid,
        units=raster.units,
        name=raster.name,
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
        values=result_values,
        georef=raster.georef,
        valid=raster.valid,
        units=raster.units,
        name=raster.name,
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
    return _dispatch_unary(a, _sin, operation="sin")


def cos(a: Raster) -> Raster:
    return _dispatch_unary(a, _cos, operation="cos")


def tan(a: Raster) -> Raster:
    return _dispatch_unary(a, _tan, operation="tan")


def arcsin(a: Raster) -> Raster:
    return _dispatch_unary(a, _arcsin, operation="arcsin")


def arccos(a: Raster) -> Raster:
    return _dispatch_unary(a, _arccos, operation="arccos")


def arctan(a: Raster) -> Raster:
    return _dispatch_unary(a, _arctan, operation="arctan")


def arctan2(a: Raster, b: Raster) -> Raster:
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(a, b, _arctan2, operation="arctan2")


def degrees(a: Raster) -> Raster:
    return _dispatch_unary(a, _degrees, operation="degrees")


def radians(a: Raster) -> Raster:
    return _dispatch_unary(a, _radians, operation="radians")


def hypot(a: Raster, b: Raster) -> Raster:
    _require_common_grid([a, b])
    return _dispatch_binary_raster_raster(a, b, _hypot, operation="hypot")


def round_half_even(a: Raster) -> Raster:
    return _dispatch_unary(a, _round_half_even, operation="round")


def floor(a: Raster) -> Raster:
    return _dispatch_unary(a, _floor, operation="floor")


def ceil(a: Raster) -> Raster:
    return _dispatch_unary(a, _ceil, operation="ceil")


def trunc(a: Raster) -> Raster:
    return _dispatch_unary(a, _trunc, operation="trunc")
