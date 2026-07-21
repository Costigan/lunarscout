from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..errors import MapAlgebraDTypeError

_OverflowPolicy = Literal["raise", "wrap", "promote"]
_CastingPolicy = Literal["safe", "same_kind", "unsafe"]


def result_dtype(
    operand_dtypes: tuple[np.dtype[Any], ...],
    *,
    operation: str,
    scalars: tuple[int | float | bool | None, ...] = (),
) -> np.dtype[Any]:
    try:
        return np.result_type(*operand_dtypes, *scalars)
    except TypeError as exc:
        raise MapAlgebraDTypeError(
            f"Cannot determine result dtype for operation '{operation}'.",
            code="map_algebra_dtype_inference_failed",
            details={
                "operation": operation,
                "operand_dtypes": [str(d) for d in operand_dtypes],
                "error": str(exc),
            },
        ) from exc


def accumulator_dtype(
    source_dtype: np.dtype[Any],
    *,
    operation: str,
) -> np.dtype[Any]:
    if np.issubdtype(source_dtype, np.integer):
        return np.dtype(np.int64)
    if source_dtype == np.dtype(np.float32):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _is_overflow_safe(
    source_dtype: np.dtype[Any],
    float_result: np.ndarray[Any, np.dtype[np.float64]],
) -> bool:
    if np.issubdtype(source_dtype, np.integer):
        limits = np.iinfo(source_dtype)
        return bool(
            np.all(
                (float_result >= float(limits.min))
                & (float_result <= float(limits.max))
                & (float_result == np.floor(float_result))
            )
        )
    if np.issubdtype(source_dtype, np.floating):
        finfo = np.finfo(source_dtype)
        abs_result = np.abs(float_result)
        overflow = abs_result > float(finfo.max)
        underflow = (abs_result > 0) & (abs_result < float(finfo.tiny))
        invalid = ~np.isfinite(float_result)
        if np.any(overflow | underflow | invalid):
            return False
        return bool(np.all(np.isfinite(float_result.astype(source_dtype))))
    return True


def checked_integer_operation(
    values_a: np.ndarray[Any, Any],
    values_b: np.ndarray[Any, Any],
    result_dtype_value: np.dtype[Any],
    op_func,
    *,
    overflow: _OverflowPolicy = "raise",
) -> np.ndarray[Any, Any]:
    if overflow == "wrap":
        return op_func(values_a, values_b).astype(result_dtype_value, copy=False)
    if overflow == "raise":
        float_a = values_a.astype(np.float64, copy=False)
        float_b = values_b.astype(np.float64, copy=False)
        float_result = op_func(float_a, float_b)
        if not _is_overflow_safe(result_dtype_value, float_result):
            raise MapAlgebraDTypeError(
                "Integer operation overflow detected. "
                "Use overflow='wrap' or overflow='promote'.",
                code="map_algebra_overflow",
                details={
                    "result_dtype": str(result_dtype_value),
                    "overflow_policy": overflow,
                },
            )
        return float_result.astype(result_dtype_value, copy=False)
    if overflow == "promote":
        float_result = op_func(
            values_a.astype(np.float64, copy=False),
            values_b.astype(np.float64, copy=False),
        )
        promoted = np.result_type(np.float64, result_dtype_value)
        return float_result.astype(promoted, copy=False)
    raise MapAlgebraDTypeError(
        f"Unknown overflow policy: {overflow}",
        code="map_algebra_invalid_overflow",
        details={"overflow": overflow},
    )


def cast_values(
    values: np.ndarray[Any, Any],
    target_dtype: np.dtype[Any],
    *,
    casting: _CastingPolicy = "safe",
) -> np.ndarray[Any, Any]:
    if casting == "unsafe":
        return values.astype(target_dtype, copy=False)
    if casting == "safe":
        return values.astype(target_dtype, casting="safe", copy=True)
    if casting == "same_kind":
        return values.astype(target_dtype, casting="same_kind", copy=True)
    raise MapAlgebraDTypeError(
        f"Unknown casting policy: {casting}",
        code="map_algebra_invalid_casting",
        details={"casting": casting},
    )
