"""Temporal local operations follow the spatial numeric policy (review finding H4).

Temporal expressions used to evaluate local operations with raw NumPy kernels:
uint8 arithmetic wrapped, scalars were cast to the raster dtype, and x/0,
sqrt(-1) and log(0) produced valid inf/NaN. Each layer is now evaluated by
the spatial numeric core, so a temporal expression must match the same
expression evaluated layer by layer on spatial rasters.
"""
from __future__ import annotations

import numpy as np
import pytest

import lunarscout.map_algebra as ma
from lunarscout.errors import LunarscoutError, MapAlgebraDTypeError
from lunarscout.map_algebra import compute, compute_temporal, temporal_source
from lunarscout.map_algebra._temporal_model import TemporalRaster
from lunarscout.raster import Raster

from tests.map_algebra.test_temporal import _make_georef, _make_times

_GEOREF = _make_georef(width=4, height=2)
_TIMES = _make_times(2)

_SAMPLES = {
    np.dtype(np.int8): [[-128, -1, 0, 1], [2, 50, 100, 127]],
    np.dtype(np.uint8): [[0, 1, 2, 10], [100, 128, 250, 255]],
    np.dtype(np.int64): [[np.iinfo(np.int64).min, -1, 0, 1], [2, 3, 2**40, np.iinfo(np.int64).max]],
    np.dtype(np.uint64): [[0, 1, 2, 3], [2**40, 2**63, 2**64 - 2, 2**64 - 1]],
    np.dtype(np.float32): [[-2.5, -1.0, 0.0, 0.5], [1.0, 2.0, 100.0, 1e30]],
}


def _stack(dtype: np.dtype, *, units: str | None = None) -> TemporalRaster:
    layer = np.array(_SAMPLES[dtype], dtype=dtype)
    values = np.stack([layer, layer[::-1, ::-1].copy()])
    return TemporalRaster(
        values=values, times=_TIMES, georef=_GEOREF,
        valid=np.ones(values.shape, dtype=np.bool_), units=units,
    )


def _layer(stack: TemporalRaster, index: int) -> Raster:
    return Raster(
        values=stack.values[index], georef=_GEOREF,
        valid=stack.valid[index], units=stack.units,
    )


def _outcome(fn):
    try:
        return ("ok", fn())
    except LunarscoutError as error:
        return ("error", error.code)


def _assert_parity(build) -> None:
    """build(operand_factory) must give the same result temporally and per layer."""
    temporal = _outcome(lambda: compute_temporal(build(lambda s: temporal_source(s))))
    layers = [
        _outcome(lambda i=i: compute(build(lambda s: _layer(s, i).expression())))
        for i in range(len(_TIMES))
    ]
    errors = [outcome[1] for outcome in layers if outcome[0] == "error"]
    if temporal[0] == "error" or errors:
        assert temporal == ("error", errors[0] if errors else None), (temporal, layers)
        return
    result = temporal[1]
    expression = build(lambda s: temporal_source(s))
    assert result.dtype == expression.dtype, "build-time dtype differs from computed dtype"
    for index, (_, spatial) in enumerate(layers):
        assert result.dtype == spatial.dtype
        np.testing.assert_array_equal(result.valid[index], spatial.valid)
        np.testing.assert_array_equal(
            result.values[index][spatial.valid], spatial.values[spatial.valid]
        )


_NUMERIC = list(_SAMPLES)
_UNARY = ["negative", "absolute", "square", "sqrt", "exp", "log", "log10",
          "floor", "ceil", "trunc", "degrees", "radians"]
_BINARY = ["add", "subtract", "multiply", "divide", "floor_divide", "remainder",
           "power", "minimum", "maximum", "hypot", "arctan2",
           "less", "less_equal", "greater", "greater_equal", "equal", "not_equal"]


@pytest.mark.parametrize("dtype", _NUMERIC, ids=str)
@pytest.mark.parametrize("operation", _UNARY)
def test_unary_temporal_matches_spatial(operation: str, dtype: np.dtype) -> None:
    stack = _stack(dtype)
    _assert_parity(lambda make: getattr(ma, operation)(make(stack)))


@pytest.mark.parametrize("operation", ["sin", "cos", "tan", "arcsin", "arccos", "arctan"])
def test_trigonometry_temporal_matches_spatial(operation: str) -> None:
    stack = _stack(np.dtype(np.float32), units="radians")
    _assert_parity(lambda make: getattr(ma, operation)(make(stack)))


@pytest.mark.parametrize("dtype", _NUMERIC, ids=str)
@pytest.mark.parametrize("operation", _BINARY)
@pytest.mark.parametrize("form", ["array-array", "array-scalar", "scalar-array"])
def test_binary_temporal_matches_spatial(operation: str, dtype: np.dtype, form: str) -> None:
    left, right = _stack(dtype), _stack(dtype)
    scalar = 3 if dtype.kind in "iu" else 1.5
    if form == "array-array":
        build = lambda make: getattr(ma, operation)(make(left), make(right))  # noqa: E731
    elif form == "array-scalar":
        build = lambda make: getattr(ma, operation)(make(left), scalar)  # noqa: E731
    else:
        build = lambda make: getattr(ma, operation)(scalar, make(left))  # noqa: E731
    _assert_parity(build)


@pytest.mark.parametrize("operation", ["logical_and", "logical_or", "logical_xor"])
def test_boolean_temporal_matches_spatial(operation: str) -> None:
    values = np.array([[[True, False, True, False]] * 2] * 2)
    stack = TemporalRaster(values=values, times=_TIMES, georef=_GEOREF,
                           valid=np.ones(values.shape, dtype=np.bool_))
    other = TemporalRaster(values=~values, times=_TIMES, georef=_GEOREF,
                           valid=np.ones(values.shape, dtype=np.bool_))
    _assert_parity(lambda make: getattr(ma, operation)(make(stack), make(other)))
    _assert_parity(lambda make: ma.logical_not(make(stack)))


def _uint8(values) -> TemporalRaster:
    array = np.array([[values]], dtype=np.uint8).repeat(2, axis=0)
    return TemporalRaster(values=array, times=_TIMES, georef=_make_georef(width=len(values), height=1),
                          valid=np.ones(array.shape, dtype=np.bool_))


@pytest.mark.parametrize(
    "build",
    [
        lambda t: ma.multiply(t, 100),
        lambda t: ma.add(t, 10),
        lambda t: ma.subtract(t, 251),
        lambda t: ma.multiply(t, t),
    ],
    ids=["t*100", "250+10", "t-251", "t*t"],
)
def test_review_uint8_overflows_raise(build) -> None:
    with pytest.raises(MapAlgebraDTypeError) as error:
        compute_temporal(build(temporal_source(_uint8([250]))))
    assert error.value.code == "map_algebra_overflow"


def test_review_uint8_overflow_policies_are_accepted() -> None:
    source = temporal_source(_uint8([250]))
    wrapped = compute_temporal(ma.add(source, 10, overflow="wrap"))
    assert wrapped.dtype == np.dtype(np.uint8) and int(wrapped.values[0, 0, 0]) == 4
    promoted = compute_temporal(ma.add(source, 10, overflow="promote"))
    assert promoted.dtype == np.dtype(np.uint16) and int(promoted.values[0, 0, 0]) == 260


def test_review_float_scalar_is_not_truncated_to_the_raster_dtype() -> None:
    result = compute_temporal(ma.subtract(1.5, temporal_source(_uint8([1]))))
    assert result.dtype.kind == "f"
    assert float(result.values[0, 0, 0]) == 0.5


@pytest.mark.parametrize(
    "build",
    [lambda t: ma.divide(t, 0), lambda t: ma.sqrt(ma.subtract(t, 2.0)), lambda t: ma.log(ma.multiply(t, 0))],
    ids=["t/0", "sqrt(-1)", "log(0)"],
)
def test_review_domain_errors_are_invalid_not_valid_nan(build) -> None:
    result = compute_temporal(build(temporal_source(_uint8([1]))))
    assert not result.valid.any()
    with pytest.raises(LunarscoutError):
        compute_temporal(
            ma.divide(temporal_source(_uint8([1])), 0, numeric_errors="raise")
        )
