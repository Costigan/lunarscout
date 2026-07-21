from __future__ import annotations

from typing import Any, Callable

import numpy as np

_NumericKernel = Callable[..., np.ndarray[Any, Any]]


def _add(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a + b


def _subtract(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a - b


def _subtract_from(a: int | float, b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return a - b


def _multiply(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a * b


def _divide(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a / b


def _divide_from(a: int | float, b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return a / b


def _floor_divide(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a // b


def _remainder(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a % b


def _power(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a**b


def _power_from(a: int | float, b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return a**b


def _negate(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return -a


def _absolute(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.abs(a)


def _minimum(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return np.minimum(a, b)


def _maximum(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return np.maximum(a, b)


def _less(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a < b


def _less_equal(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a <= b


def _greater(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a > b


def _greater_equal(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a >= b


def _equal(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a == b


def _not_equal(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any] | int | float) -> np.ndarray[Any, Any]:
    return a != b


def _logical_and(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return a & b


def _logical_or(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return a | b


def _logical_xor(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return a ^ b


def _logical_not(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return ~a


def _clip(
    values: np.ndarray[Any, Any],
    lower: int | float | None,
    upper: int | float | None,
) -> np.ndarray[Any, Any]:
    result = values
    if lower is not None:
        result = np.maximum(result, lower)
    if upper is not None:
        result = np.minimum(result, upper)
    return result


def _sqrt(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.sqrt(a)


def _square(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.square(a)


def _exp(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.exp(a)


def _log(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.log(a)


def _log10(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.log10(a)


def _sin(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.sin(a)


def _cos(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.cos(a)


def _tan(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.tan(a)


def _arcsin(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.arcsin(a)


def _arccos(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.arccos(a)


def _arctan(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.arctan(a)


def _arctan2(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.arctan2(a, b)


def _round_half_even(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.round(a)


def _floor(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.floor(a)


def _ceil(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.ceil(a)


def _trunc(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.trunc(a)


def _degrees(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.degrees(a)


def _radians(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.radians(a)


def _hypot(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.hypot(a, b)
