from __future__ import annotations

import numpy as np
import pytest

from lunarscout.errors import MapAlgebraDTypeError, MapAlgebraOperationError
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    absolute,
    add,
    compute,
    cast,
    divide,
    log,
    multiply,
    negative,
    raster,
    sqrt,
    subtract,
    write,
)
from tests.map_algebra.conftest import MOON_PROJ4, MOON_WKT


def _raster(values: np.ndarray, *, valid: np.ndarray | None = None):
    height, width = values.shape
    grid = GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=(0.0, 20.0, 0.0, 0.0, 0.0, -20.0),
        width=width,
        height=height,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )
    return raster(values, grid, valid=valid)


@pytest.mark.parametrize(
    ("operation", "values", "operand"),
    [
        (add, np.array([[1.25]], dtype=np.float32), np.float32(2.5)),
        (divide, np.array([[5.0]], dtype=np.float32), np.float32(2.0)),
    ],
)
def test_float32_binary_execution_stays_float32(operation, values, operand):
    result = operation(_raster(values), operand)
    assert result.dtype == np.dtype(np.float32)


def test_explicit_float64_scalar_promotes_float32_input():
    source = _raster(np.array([[1.0]], dtype=np.float32))
    result = add(source, np.float64(0.5))
    assert result.dtype == np.dtype(np.float64)


@pytest.mark.parametrize("operation", [sqrt, log])
def test_float32_unary_execution_stays_float32(operation):
    result = operation(_raster(np.array([[4.0]], dtype=np.float32)))
    assert result.dtype == np.dtype(np.float32)


def test_uint64_above_float_exact_range_remains_exact():
    value = 2**63 + 123
    result = add(_raster(np.array([[value]], dtype=np.uint64)), 1)
    assert result.dtype == np.dtype(np.uint64)
    assert int(result.values[0, 0]) == value + 1


def test_uint64_max_addition_raises_structured_overflow():
    source = _raster(np.array([[np.iinfo(np.uint64).max]], dtype=np.uint64))
    with pytest.raises(MapAlgebraDTypeError) as error:
        add(source, 1)
    assert error.value.code == "map_algebra_overflow"
    assert error.value.details["result_dtype"] == "uint64"


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64],
)
def test_each_integer_width_rejects_addition_past_maximum(dtype):
    source = _raster(np.array([[np.iinfo(dtype).max]], dtype=dtype))
    with pytest.raises(MapAlgebraDTypeError) as error:
        add(source, 1)
    assert error.value.code == "map_algebra_overflow"


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_signed_extreme_additions_that_fit_remain_exact(dtype):
    info = np.iinfo(dtype)
    source = _raster(np.array([[info.min, info.max]], dtype=dtype))
    adjustment = _raster(np.array([[1, -1]], dtype=dtype))
    result = add(source, adjustment)
    np.testing.assert_array_equal(
        result.values, np.array([[int(info.min) + 1, int(info.max) - 1]], dtype=dtype),
    )


@pytest.mark.parametrize(
    ("values", "operation", "operand"),
    [
        (np.array([[np.iinfo(np.int64).max]], dtype=np.int64), add, 1),
        (np.array([[np.iinfo(np.int64).max]], dtype=np.int64), multiply, 2),
        (np.array([[np.iinfo(np.int64).min]], dtype=np.int64), negative, None),
    ],
)
def test_int64_boundaries_raise_without_float_conversion(values, operation, operand):
    with pytest.raises(MapAlgebraDTypeError) as error:
        if operand is None:
            operation(_raster(values))
        else:
            operation(_raster(values), operand)
    assert error.value.code == "map_algebra_overflow"


def test_wrap_policy_is_explicit():
    source = _raster(np.array([[np.iinfo(np.uint8).max]], dtype=np.uint8))
    result = add(source, 1, overflow="wrap")
    assert result.dtype == np.dtype(np.uint8)
    assert int(result.values[0, 0]) == 0


def test_promote_policy_uses_exact_integer_dtype():
    source = _raster(np.array([[np.iinfo(np.uint8).max]], dtype=np.uint8))
    result = add(source, 1, overflow="promote")
    assert result.dtype == np.dtype(np.uint16)
    assert int(result.values[0, 0]) == 256


def test_promoted_absolute_handles_signed_minimum_exactly():
    source = _raster(np.array([[np.iinfo(np.int64).min]], dtype=np.int64))
    result = absolute(source, overflow="promote")
    assert result.dtype == np.dtype(np.uint64)
    assert int(result.values[0, 0]) == 2**63


def test_promote_policy_has_eager_expression_parity():
    source = _raster(np.array([[250]], dtype=np.uint8))
    eager = add(source, 10, overflow="promote")
    expression = add(source.expression(), 10, overflow="promote")
    assert expression.dtype == np.dtype(np.uint16)
    computed = compute(expression)
    assert computed.dtype == eager.dtype
    np.testing.assert_array_equal(computed.values, eager.values)


def test_float32_expression_inference_and_execution_stay_float32():
    source = _raster(np.array([[4.0]], dtype=np.float32)).expression()
    expression = sqrt(source)
    assert expression.dtype == np.dtype(np.float32)
    assert compute(expression).dtype == np.dtype(np.float32)


def test_integer_comparison_returns_boolean_not_integer():
    source = _raster(np.array([[1, 3]], dtype=np.uint64))
    result = source > 2
    assert result.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(result.values, np.array([[False, True]]))


@pytest.mark.parametrize(
    ("policy", "expected_valid"),
    [("invalid", False), ("keep", True)],
)
def test_numeric_error_policy_controls_domain_validity(policy, expected_valid):
    source = _raster(np.array([[-1.0, 4.0]], dtype=np.float32))
    result = sqrt(source, numeric_errors=policy)
    assert bool(result.valid[0, 0]) is expected_valid
    assert result.valid[0, 1]
    assert np.isnan(result.values[0, 0])


def test_numeric_error_raise_is_structured():
    source = _raster(np.array([[0.0, 2.0]], dtype=np.float32))
    with pytest.raises(MapAlgebraOperationError) as error:
        log(source, numeric_errors="raise")
    assert error.value.code == "map_algebra_numeric_error"
    assert error.value.details == {"operation": "log", "affected_pixels": 1}


def test_numeric_error_raise_ignores_already_invalid_pixels():
    source = _raster(
        np.array([[-1.0, 4.0]], dtype=np.float32),
        valid=np.array([[False, True]], dtype=np.bool_),
    )
    result = sqrt(source, numeric_errors="raise")
    np.testing.assert_array_equal(result.valid, np.array([[False, True]]))


def test_integer_overflow_in_invalid_payload_does_not_raise():
    source = _raster(
        np.array([[np.iinfo(np.int64).max, 4]], dtype=np.int64),
        valid=np.array([[False, True]], dtype=np.bool_),
    )
    result = add(source, 1)
    np.testing.assert_array_equal(result.valid, np.array([[False, True]]))
    assert int(result.values[0, 1]) == 5


def test_all_invalid_uint64_does_not_validate_unused_negative_scalar():
    source = _raster(
        np.array([[np.iinfo(np.uint64).max]], dtype=np.uint64),
        valid=np.array([[False]], dtype=np.bool_),
    )
    result = add(source, -1)
    assert not result.valid[0, 0]


def test_integer_divide_by_zero_is_a_domain_error():
    source = _raster(np.array([[4, 5]], dtype=np.int16))
    divisor = _raster(np.array([[2, 0]], dtype=np.int16))
    result = divide(source, divisor)
    np.testing.assert_array_equal(result.valid, np.array([[True, False]]))
    with pytest.raises(MapAlgebraOperationError) as error:
        divide(source, divisor, numeric_errors="raise")
    assert error.value.code == "map_algebra_numeric_error"


def test_scalar_left_subtraction_uses_correct_overflow_direction():
    source = _raster(np.array([[np.iinfo(np.int8).min]], dtype=np.int8))
    with pytest.raises(MapAlgebraDTypeError) as error:
        subtract(0, source)
    assert error.value.code == "map_algebra_overflow"


def test_scalar_left_subtraction_promotes_for_actual_direction():
    source = _raster(np.array([[np.iinfo(np.int8).min]], dtype=np.int8))
    eager = subtract(0, source, overflow="promote")
    expression = subtract(0, source.expression(), overflow="promote")
    assert eager.dtype == np.dtype(np.int16)
    assert expression.dtype == np.dtype(np.int16)
    assert int(compute(expression).values[0, 0]) == 128


def test_scalar_left_division_checks_raster_divisor():
    source = _raster(np.array([[0, 2]], dtype=np.int16))
    result = divide(100, source)
    np.testing.assert_array_equal(result.valid, np.array([[False, True]]))
    assert result.values[0, 1] == 50


def test_numeric_policy_is_part_of_expression_identity_and_execution():
    source = _raster(np.array([[-1.0]], dtype=np.float32)).expression()
    invalid_expr = sqrt(source, numeric_errors="invalid")
    keep_expr = sqrt(source, numeric_errors="keep")
    assert invalid_expr.scientific_identity() != keep_expr.scientific_identity()
    assert not compute(invalid_expr).valid[0, 0]
    assert compute(keep_expr).valid[0, 0]


def test_numeric_policy_has_windowed_validity_parity(tmp_path):
    import rasterio

    source = _raster(np.array([[-1.0, 4.0]], dtype=np.float32)).expression()
    expression = sqrt(source, numeric_errors="invalid")
    eager = compute(expression)
    output = write(
        tmp_path / "sqrt.tif", expression,
        window_width=1, window_height=1,
    )
    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read_masks(1) > 0, eager.valid)
        assert dataset.dtypes[0] == "float32"


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (lambda r: add(r, 1, overflow="saturate"), "map_algebra_invalid_overflow"),
        (lambda r: sqrt(r, numeric_errors="warn"), "map_algebra_invalid_numeric_errors"),
    ],
)
def test_invalid_numeric_policy_is_structured(call, code):
    with pytest.raises((MapAlgebraDTypeError, MapAlgebraOperationError)) as error:
        call(_raster(np.array([[1.0]], dtype=np.float32)))
    assert error.value.code == code


def test_expression_rejects_policy_not_in_public_signature():
    source = _raster(np.array([[1.0]], dtype=np.float32)).expression()
    with pytest.raises(TypeError):
        add(source, 1, numeric_errors="keep")


def test_invalid_cast_dtype_is_structured():
    source = _raster(np.array([[1.0]], dtype=np.float32))
    with pytest.raises(MapAlgebraDTypeError) as error:
        cast(source, "not-a-dtype")
    assert error.value.code == "map_algebra_unsupported_dtype"
