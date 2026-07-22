from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..errors import MapAlgebraDTypeError

OverflowPolicy = Literal["raise", "wrap", "promote"]
CastingPolicy = Literal["safe", "same_kind", "unsafe"]
CastOverflowPolicy = Literal["raise", "wrap"]

_COMPARISON_OPERATIONS = frozenset({
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "isclose", "logical_and", "logical_or", "logical_xor", "logical_not",
})
_FLOAT_UNARY_OPERATIONS = frozenset({
    "sqrt", "exp", "log", "log10", "sin", "cos", "tan", "arcsin",
    "arccos", "arctan", "degrees", "radians", "floor", "ceil", "trunc",
})
_FLOAT_BINARY_OPERATIONS = frozenset({"arctan2", "hypot"})
_CHECKED_INTEGER_OPERATIONS = frozenset({
    "add", "subtract", "multiply", "floor_divide", "remainder", "negative",
    "absolute", "square", "power",
})
_SELECTION_OPERATIONS = frozenset({"where", "coalesce"})
_SUPPORTED_DTYPES = frozenset(
    np.dtype(dtype)
    for dtype in (
        np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
        np.int64, np.uint64, np.float32, np.float64,
    )
)


def normalize_dtype(value: Any, *, operation: str) -> np.dtype[Any]:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise MapAlgebraDTypeError(
            f"Invalid dtype for operation '{operation}'.",
            code="map_algebra_unsupported_dtype",
            details={"operation": operation, "dtype": repr(value), "error": str(exc)},
        ) from exc
    if dtype not in _SUPPORTED_DTYPES:
        raise MapAlgebraDTypeError(
            f"Dtype {dtype} is not supported for map algebra.",
            code="map_algebra_unsupported_dtype",
            details={"operation": operation, "dtype": str(dtype)},
        )
    return dtype


def normalize_overflow(value: str) -> OverflowPolicy:
    if value not in {"raise", "wrap", "promote"}:
        raise MapAlgebraDTypeError(
            "overflow must be 'raise', 'wrap', or 'promote'.",
            code="map_algebra_invalid_overflow",
            details={"overflow": value},
        )
    return value  # type: ignore[return-value]


def normalize_cast_overflow(value: str) -> CastOverflowPolicy:
    if value not in {"raise", "wrap"}:
        raise MapAlgebraDTypeError(
            "cast overflow must be 'raise' or 'wrap'.",
            code="map_algebra_invalid_cast_overflow",
            details={"overflow": value},
        )
    return value  # type: ignore[return-value]


def result_dtype(
    operand_dtypes: tuple[np.dtype[Any], ...],
    *,
    operation: str,
    scalars: tuple[int | float | bool | None, ...] = (),
    overflow: OverflowPolicy = "raise",
    scalar_left: bool = False,
) -> np.dtype[Any]:
    """Infer one operation dtype without evaluating values.

    This is the shared eager/expression inference entry point. In particular,
    NumPy ``float32`` operations remain ``float32``; the helper never inserts
    FP64 merely to make checking easier.
    """
    overflow = normalize_overflow(overflow)
    scalar_inputs: tuple[Any, ...]
    if operation in _SELECTION_OPERATIONS:
        scalar_inputs = tuple(
            _selection_scalar_input(scalar, operation=operation)
            for scalar in scalars
            if scalar is not None
        )
    else:
        scalar_inputs = tuple(s for s in scalars if s is not None)
    inputs: tuple[Any, ...] = (*operand_dtypes, *scalar_inputs)
    try:
        inferred = np.dtype(np.result_type(*inputs))
    except (TypeError, ValueError) as exc:
        raise MapAlgebraDTypeError(
            f"Cannot determine result dtype for operation '{operation}'.",
            code="map_algebra_dtype_inference_failed",
            details={
                "operation": operation,
                "operand_dtypes": [str(d) for d in operand_dtypes],
                "scalars": [repr(s) for s in scalars],
                "error": str(exc),
            },
        ) from exc

    if operation in _COMPARISON_OPERATIONS:
        return np.dtype(np.bool_)
    if operation in _SELECTION_OPERATIONS:
        selection_input_dtypes = (
            *operand_dtypes,
            *(
                value if isinstance(value, np.dtype) else np.asarray(value).dtype
                for value in scalar_inputs
            ),
        )
        if selection_input_dtypes and all(
            dtype.kind in "biu" for dtype in selection_input_dtypes
        ):
            if inferred.kind not in "biu":
                raise MapAlgebraDTypeError(
                    "No supported integer dtype can represent every selected value.",
                    code="map_algebra_no_exact_promotion",
                    details={
                        "operation": operation,
                        "input_dtypes": [
                            str(dtype) for dtype in selection_input_dtypes
                        ],
                    },
                )
        return normalize_dtype(inferred, operation=operation)
    if operation == "divide":
        if inferred == np.dtype(np.float32):
            return normalize_dtype(inferred, operation=operation)
        if inferred.kind in "biu":
            return np.dtype(np.float64)
        return normalize_dtype(inferred, operation=operation)
    if operation in (_FLOAT_UNARY_OPERATIONS | _FLOAT_BINARY_OPERATIONS) and inferred.kind in "biu":
        return np.dtype(np.float64)
    if overflow == "promote" and inferred.kind in "iu" and operation in _CHECKED_INTEGER_OPERATIONS:
        return _promoted_integer_dtype(
            operand_dtypes, scalars, operation, scalar_left=scalar_left,
        )
    return normalize_dtype(inferred, operation=operation)


def _selection_scalar_input(
    value: int | float | bool,
    *,
    operation: str,
) -> Any:
    """Return a selection input that cannot silently truncate an integer.

    NumPy 2 treats Python scalars as weakly typed.  That is useful for an
    ordinary FP32 scalar such as ``-1.0``, but it lets ``where(uint8, 300)``
    retain uint8 and wrap the selected scalar. Selection is not arithmetic:
    Python integers must be representable by the inferred output dtype, so
    they participate through their smallest exact dtype.
    Python floats retain NumPy's FP32 weak-scalar behavior when their magnitude
    is representable; larger finite values force FP64.
    """
    if isinstance(value, np.generic):
        return normalize_dtype(value.dtype, operation=operation)
    if isinstance(value, bool):
        return np.dtype(np.bool_)
    if isinstance(value, int):
        return normalize_dtype(np.min_scalar_type(value), operation=operation)
    if isinstance(value, float):
        if np.isfinite(value) and abs(value) > float(np.finfo(np.float32).max):
            return np.dtype(np.float64)
        return value
    raise MapAlgebraDTypeError(
        f"Unsupported scalar for operation '{operation}'.",
        code="map_algebra_dtype_inference_failed",
        details={"operation": operation, "scalar_type": type(value).__name__},
    )


def accumulator_dtype(
    source_dtype: np.dtype[Any],
    *,
    operation: str,
) -> np.dtype[Any]:
    if np.issubdtype(source_dtype, np.unsignedinteger):
        return np.dtype(np.uint64)
    if np.issubdtype(source_dtype, np.signedinteger):
        return np.dtype(np.int64)
    if source_dtype == np.dtype(np.float32):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _dtype_bounds(dtype: np.dtype[Any]) -> tuple[int, int]:
    info = np.iinfo(dtype)
    return int(info.min), int(info.max)


def _operand_bounds(
    operand_dtypes: tuple[np.dtype[Any], ...],
    scalars: tuple[int | float | bool | None, ...],
) -> list[tuple[int, int]]:
    bounds = [_dtype_bounds(dtype) for dtype in operand_dtypes]
    for scalar in scalars:
        if scalar is None:
            continue
        if not isinstance(scalar, (int, bool, np.integer)):
            raise MapAlgebraDTypeError(
                "overflow='promote' requires integer operands.",
                code="map_algebra_invalid_overflow_operands",
                details={"scalar": repr(scalar)},
            )
        value = int(scalar)
        bounds.append((value, value))
    return bounds


def _operation_bounds(bounds: list[tuple[int, int]], operation: str) -> tuple[int, int]:
    if operation in {"negative", "absolute", "square"}:
        lower, upper = bounds[0]
        if operation == "negative":
            return -upper, -lower
        if operation == "absolute":
            return 0, max(abs(lower), abs(upper))
        candidates = (lower * lower, lower * upper, upper * upper)
        return min(candidates), max(candidates)
    if len(bounds) != 2:
        raise MapAlgebraDTypeError(
            f"Cannot infer promoted dtype for '{operation}'.",
            code="map_algebra_dtype_inference_failed",
            details={"operation": operation},
        )
    (a0, a1), (b0, b1) = bounds
    if operation == "add":
        return a0 + b0, a1 + b1
    if operation == "subtract":
        return a0 - b1, a1 - b0
    if operation == "multiply":
        products = (a0 * b0, a0 * b1, a1 * b0, a1 * b1)
        return min(products), max(products)
    if operation == "floor_divide":
        # Division cannot increase magnitude except signed minimum / -1.
        if a0 >= 0 and b0 >= 0:
            return 0, a1
        magnitude = max(abs(a0), abs(a1))
        return -magnitude, magnitude
    if operation == "remainder":
        magnitude = max(abs(b0), abs(b1))
        if a0 >= 0 and b0 >= 0:
            return 0, max(0, magnitude - 1)
        return -max(0, magnitude - 1), max(0, magnitude - 1)
    if operation == "power":
        if b0 < 0:
            raise MapAlgebraDTypeError(
                "overflow='promote' cannot infer an integer dtype for negative exponents.",
                code="map_algebra_no_exact_promotion",
                details={"operation": operation, "minimum_exponent": str(b0)},
            )
        cap = 2**64
        magnitude = _bounded_integer_power(max(abs(a0), abs(a1)), b1, cap)
        if b0 == b1:
            if b0 == 0:
                return 1, 1
            if b0 % 2 == 0 or a0 >= 0:
                return 0, magnitude
            return -magnitude, magnitude
        can_be_odd = b0 <= b1 and (b0 % 2 == 1 or b0 < b1)
        return (-magnitude if a0 < 0 and can_be_odd else 0), magnitude
    raise MapAlgebraDTypeError(
        f"overflow='promote' is not supported for '{operation}'.",
        code="map_algebra_unsupported_overflow_promotion",
        details={"operation": operation},
    )


def _bounded_integer_power(base: int, exponent: int, cap: int) -> int:
    """Return ``base**exponent`` capped at ``cap + 1`` without huge integers."""
    result = 1
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            if factor and result > cap // factor:
                return cap + 1
            result *= factor
        remaining >>= 1
        if remaining:
            if factor and factor > cap // factor:
                factor = cap + 1
            else:
                factor *= factor
    return result


def _promoted_integer_dtype(
    operand_dtypes: tuple[np.dtype[Any], ...],
    scalars: tuple[int | float | bool | None, ...],
    operation: str,
    *,
    scalar_left: bool = False,
) -> np.dtype[Any]:
    operand_bounds = _operand_bounds(operand_dtypes, scalars)
    if scalar_left and len(operand_bounds) == 2:
        operand_bounds.reverse()
    lower, upper = _operation_bounds(operand_bounds, operation)
    # A native checked kernel normally needs one dtype capable of representing
    # both operands and results. Absolute value is the one intentional
    # exception: signed minimum can be mapped exactly to an unsigned result by
    # the branch-safe implementation below.
    if operation != "absolute":
        lower = min(lower, *(bound[0] for bound in operand_bounds))
        upper = max(upper, *(bound[1] for bound in operand_bounds))
    candidates = (
        np.dtype(np.uint8), np.dtype(np.int8), np.dtype(np.uint16),
        np.dtype(np.int16), np.dtype(np.uint32), np.dtype(np.int32),
        np.dtype(np.uint64), np.dtype(np.int64),
    )
    for candidate in candidates:
        cmin, cmax = _dtype_bounds(candidate)
        if cmin <= lower and upper <= cmax:
            return candidate
    raise MapAlgebraDTypeError(
        "No supported integer dtype can represent every possible result.",
        code="map_algebra_no_exact_promotion",
        details={"operation": operation, "minimum": str(lower), "maximum": str(upper)},
    )


def _require_values_representable(
    values: np.ndarray[Any, Any],
    dtype: np.dtype[Any],
    check_mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None,
) -> None:
    lower, upper = _dtype_bounds(dtype)
    checked = values if check_mask is None else values[np.broadcast_to(check_mask, values.shape)]
    if checked.size == 0:
        return
    actual_min = int(checked.min())
    actual_max = int(checked.max())
    if actual_min < lower or actual_max > upper:
        raise MapAlgebraDTypeError(
            f"Operand values are not representable by inferred dtype {dtype}.",
            code="map_algebra_overflow",
            details={
                "result_dtype": str(dtype),
                "minimum": str(actual_min),
                "maximum": str(actual_max),
            },
        )


def _overflow_mask(
    a: np.ndarray[Any, Any],
    b: np.ndarray[Any, Any] | None,
    dtype: np.dtype[Any],
    operation: str,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    lower, upper = _dtype_bounds(dtype)
    if operation == "negative":
        return a != 0 if dtype.kind == "u" else a == lower
    if operation == "absolute":
        return np.zeros(a.shape, dtype=np.bool_) if dtype.kind == "u" else a == lower
    if operation == "square":
        b = a
        operation = "multiply"
    assert b is not None
    if operation == "add":
        if dtype.kind == "u":
            return a > upper - b
        return ((b > 0) & (a > upper - b)) | ((b < 0) & (a < lower - b))
    if operation == "subtract":
        if dtype.kind == "u":
            return a < b
        return ((b < 0) & (a > upper + b)) | ((b > 0) & (a < lower + b))
    if operation == "multiply":
        mask = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.bool_)
        aa, bb = np.broadcast_arrays(a, b)
        if dtype.kind == "u":
            nonzero = aa != 0
            mask[nonzero] = bb[nonzero] > upper // aa[nonzero]
            return mask
        pp = (aa > 0) & (bb > 0)
        pn = (aa > 0) & (bb < 0)
        np_ = (aa < 0) & (bb > 0)
        nn = (aa < 0) & (bb < 0)
        mask[pp] = aa[pp] > upper // bb[pp]
        mask[pn] = bb[pn] < lower // aa[pn]
        mask[np_] = aa[np_] < lower // bb[np_]
        mask[nn] = aa[nn] < upper // bb[nn]
        return mask
    if operation == "floor_divide":
        if dtype.kind == "i":
            return (a == lower) & (b == -1)
        return np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.bool_)
    if operation == "remainder":
        return np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.bool_)
    raise MapAlgebraDTypeError(
        f"Checked integer execution is not implemented for '{operation}'.",
        code="map_algebra_unsupported_checked_integer_operation",
        details={"operation": operation},
    )


def checked_integer_operation(
    values_a: np.ndarray[Any, Any],
    values_b: np.ndarray[Any, Any] | int | np.integer | None,
    result_dtype_value: np.dtype[Any],
    op_func: Any,
    *,
    operation: str,
    overflow: OverflowPolicy = "raise",
    check_mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None,
) -> np.ndarray[Any, Any]:
    """Run an integer kernel with exact, native-integer overflow checks."""
    overflow = normalize_overflow(overflow)
    target = np.dtype(result_dtype_value)
    if target.kind not in "iu":
        raise MapAlgebraDTypeError(
            "Checked integer execution requires an integer result dtype.",
            code="map_algebra_invalid_checked_dtype",
            details={"result_dtype": str(target)},
        )
    try:
        signed_absolute_to_unsigned = (
            operation == "absolute"
            and values_a.dtype.kind == "i"
            and target.kind == "u"
        )
        if not signed_absolute_to_unsigned:
            _require_values_representable(values_a, target, check_mask)
        if signed_absolute_to_unsigned:
            result = values_a.astype(target, copy=True)
            negative = values_a < 0
            # ``-(minimum + 1) + 1`` avoids overflowing the signed source
            # dtype and remains exact when converted to the unsigned target.
            result[negative] = (
                (-(values_a[negative] + 1)).astype(target, copy=False) + 1
            )
            return result
        a = values_a.astype(target, casting="unsafe", copy=False)
        b: np.ndarray[Any, Any] | None
        if values_b is None:
            b = None
        elif np.isscalar(values_b):
            scalar = int(values_b)
            lower, upper = _dtype_bounds(target)
            any_checked = check_mask is None or bool(np.any(check_mask))
            if any_checked and (scalar < lower or scalar > upper):
                raise MapAlgebraDTypeError(
                    f"Scalar {scalar} is not representable by inferred dtype {target}.",
                    code="map_algebra_overflow",
                    details={"result_dtype": str(target), "scalar": str(scalar)},
                )
            encoded_scalar = (
                0 if not any_checked and (scalar < lower or scalar > upper)
                else scalar
            )
            b = np.asarray(encoded_scalar, dtype=target)
        else:
            _require_values_representable(values_b, target, check_mask)
            b = values_b.astype(target, casting="unsafe", copy=False)

        overflow_pixels = _overflow_mask(a, b, target, operation)
        if check_mask is not None:
            overflow_pixels &= np.broadcast_to(check_mask, overflow_pixels.shape)
        if overflow != "wrap" and np.any(overflow_pixels):
            raise MapAlgebraDTypeError(
                "Integer operation overflow detected.",
                code="map_algebra_overflow",
                details={
                    "result_dtype": str(target),
                    "operation": operation,
                    "overflow_policy": overflow,
                },
            )
        with np.errstate(all="ignore"):
            return np.asarray(op_func(a) if b is None else op_func(a, b), dtype=target)
    except MapAlgebraDTypeError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise MapAlgebraDTypeError(
            f"Integer operation '{operation}' failed.",
            code="map_algebra_integer_operation_failed",
            details={"operation": operation, "result_dtype": str(target), "error": str(exc)},
        ) from exc


def checked_integer_power(
    values_base: np.ndarray[Any, Any],
    values_exponent: np.ndarray[Any, Any] | int | np.integer,
    result_dtype_value: np.dtype[Any],
    *,
    overflow: OverflowPolicy = "raise",
    check_mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]:
    """Calculate integer powers exactly with bounded repeated squaring.

    The kernel never widens to FP64 or a 64-bit integer dtype solely for
    checking; execution uses the inferred target dtype. Python scalar
    exponents use their smallest exact NumPy integer dtype, while explicit
    NumPy scalar/array dtypes are preserved.
    """
    overflow = normalize_overflow(overflow)
    target = np.dtype(result_dtype_value)
    if target.kind not in "iu":
        raise MapAlgebraDTypeError(
            "Checked integer power requires an integer result dtype.",
            code="map_algebra_invalid_checked_dtype",
            details={"result_dtype": str(target)},
        )

    if isinstance(values_exponent, (int, np.integer)):
        exponent_dtype = (
            values_exponent.dtype
            if isinstance(values_exponent, np.integer)
            else np.min_scalar_type(values_exponent)
        )
        exponent = np.asarray(values_exponent, dtype=exponent_dtype)
    else:
        exponent = np.asarray(values_exponent)
    if exponent.dtype.kind not in "iu":
        raise MapAlgebraDTypeError(
            "Integer power requires an integer exponent.",
            code="map_algebra_invalid_power_exponent",
            details={"exponent_dtype": str(exponent.dtype)},
        )

    shape = np.broadcast_shapes(values_base.shape, exponent.shape)
    base = np.broadcast_to(values_base, shape)
    exponent = np.broadcast_to(exponent, shape)
    valid = (
        np.ones(shape, dtype=np.bool_)
        if check_mask is None
        else np.broadcast_to(check_mask, shape)
    )
    domain_errors = (
        exponent < 0
        if exponent.dtype.kind == "i"
        else np.zeros(shape, dtype=np.bool_)
    )
    active_valid = valid & ~domain_errors
    _require_values_representable(base, target, active_valid)
    encoded_base = base.astype(target, casting="unsafe", copy=False)

    result = np.ones(shape, dtype=target)
    factor = np.array(encoded_base, copy=True)
    remaining = np.where(active_valid, exponent, 0).copy()
    while np.any(remaining != 0):
        odd = active_valid & ((remaining & 1) != 0)
        if np.any(odd):
            overflow_pixels = _overflow_mask(result, factor, target, "multiply") & odd
            if overflow != "wrap" and np.any(overflow_pixels):
                raise MapAlgebraDTypeError(
                    "Integer power overflow detected.",
                    code="map_algebra_overflow",
                    details={
                        "result_dtype": str(target),
                        "operation": "power",
                        "overflow_policy": overflow,
                    },
                )
            with np.errstate(all="ignore"):
                result[odd] = (result[odd] * factor[odd]).astype(target, copy=False)

        remaining = remaining // 2
        needs_factor = active_valid & (remaining != 0)
        if np.any(needs_factor):
            overflow_pixels = (
                _overflow_mask(factor, factor, target, "multiply") & needs_factor
            )
            if overflow != "wrap" and np.any(overflow_pixels):
                raise MapAlgebraDTypeError(
                    "Integer power overflow detected.",
                    code="map_algebra_overflow",
                    details={
                        "result_dtype": str(target),
                        "operation": "power",
                        "overflow_policy": overflow,
                    },
                )
            with np.errstate(all="ignore"):
                factor[needs_factor] = (
                    factor[needs_factor] * factor[needs_factor]
                ).astype(target, copy=False)
    return result, domain_errors


def _integer_cast_out_of_range(
    values: np.ndarray[Any, Any],
    target_dtype: np.dtype[Any],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    if values.dtype.kind == "b":
        source_min, source_max = 0, 1
    else:
        source_min, source_max = _dtype_bounds(values.dtype)
    target_min, target_max = _dtype_bounds(target_dtype)
    bad = np.zeros(values.shape, dtype=np.bool_)
    if target_min > source_min:
        if target_min > source_max:
            bad |= True
        else:
            bad |= values < values.dtype.type(target_min)
    if target_max < source_max:
        if target_max < source_min:
            bad |= True
        else:
            bad |= values > values.dtype.type(target_max)
    return bad


def _floating_cast_out_of_integer_range(
    values: np.ndarray[Any, Any],
    target_dtype: np.dtype[Any],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    target_min, target_max = _dtype_bounds(target_dtype)
    source_type = values.dtype.type
    lower = source_type(target_min)
    upper = source_type(target_max)
    if np.isfinite(lower) and int(lower) < target_min:
        lower = np.nextafter(lower, source_type(np.inf), dtype=values.dtype)
    if np.isfinite(upper) and int(upper) > target_max:
        upper = np.nextafter(upper, source_type(-np.inf), dtype=values.dtype)
    truncated = np.trunc(values)
    return ~np.isfinite(truncated) | (truncated < lower) | (truncated > upper)


def cast_values(
    values: np.ndarray[Any, Any],
    target_dtype: np.dtype[Any],
    *,
    casting: CastingPolicy = "safe",
    overflow: CastOverflowPolicy = "raise",
    valid: np.ndarray[Any, np.dtype[np.bool_]] | None = None,
) -> np.ndarray[Any, Any]:
    overflow = normalize_cast_overflow(overflow)
    if casting not in {"safe", "same_kind", "unsafe"}:
        raise MapAlgebraDTypeError(
            f"Unknown casting policy: {casting}",
            code="map_algebra_invalid_casting",
            details={"casting": casting},
        )
    target_dtype = normalize_dtype(target_dtype, operation="cast")
    checked = values if valid is None else values[valid]
    if overflow == "wrap" and not (
        values.dtype.kind in "iu" and target_dtype.kind in "iu"
    ):
        raise MapAlgebraDTypeError(
            "cast overflow='wrap' is supported only for integer-to-integer casts.",
            code="map_algebra_invalid_cast_overflow",
            details={
                "source_dtype": str(values.dtype),
                "target_dtype": str(target_dtype),
                "overflow": overflow,
            },
        )
    if overflow == "raise" and checked.size:
        if target_dtype.kind == "b":
            bad = (checked < 0) | (checked > 1)
            if values.dtype.kind == "f":
                bad |= ~np.isfinite(checked)
            if np.any(bad):
                raise MapAlgebraDTypeError(
                    "Cast would overflow the Boolean destination range.",
                    code="map_algebra_cast_overflow",
                    details={
                        "source_dtype": str(values.dtype),
                        "target_dtype": str(target_dtype),
                        "affected_values": int(np.count_nonzero(bad)),
                    },
                )
        elif target_dtype.kind in "iu":
            if values.dtype.kind == "f":
                bad = _floating_cast_out_of_integer_range(
                    checked, target_dtype,
                )
            else:
                bad = _integer_cast_out_of_range(checked, target_dtype)
            if np.any(bad):
                raise MapAlgebraDTypeError(
                    "Cast would overflow the requested integer dtype.",
                    code="map_algebra_cast_overflow",
                    details={
                        "source_dtype": str(values.dtype),
                        "target_dtype": str(target_dtype),
                        "affected_values": int(np.count_nonzero(bad)),
                    },
                )
        elif target_dtype.kind == "f":
            with np.errstate(all="ignore"):
                checked_result = checked.astype(target_dtype, casting="unsafe")
            bad = np.isfinite(checked) & ~np.isfinite(checked_result)
            if np.any(bad):
                raise MapAlgebraDTypeError(
                    "Cast would overflow the requested floating dtype.",
                    code="map_algebra_cast_overflow",
                    details={
                        "source_dtype": str(values.dtype),
                        "target_dtype": str(target_dtype),
                        "affected_values": int(np.count_nonzero(bad)),
                    },
                )
    try:
        with np.errstate(all="ignore"):
            return values.astype(
                target_dtype, casting=casting, copy=casting != "unsafe",
            )
    except (TypeError, ValueError, OverflowError) as exc:
        raise MapAlgebraDTypeError(
            f"Cannot cast {values.dtype} to {target_dtype} with casting='{casting}'.",
            code="map_algebra_unsafe_cast",
            details={
                "source_dtype": str(values.dtype),
                "target_dtype": str(target_dtype),
                "casting": casting,
                "error": str(exc),
            },
        ) from exc
