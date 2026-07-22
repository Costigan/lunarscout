from __future__ import annotations

import numpy as np
import pytest

from lunarscout.errors import (
    MapAlgebraDTypeError,
    MapAlgebraOperationError,
    MapAlgebraUnitError,
)
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    absolute,
    add,
    compute,
    cast,
    coalesce,
    divide,
    describe_operation,
    log,
    multiply,
    negative,
    power,
    raster,
    sqrt,
    subtract,
    where,
    write,
)
from tests.map_algebra.conftest import MOON_PROJ4, MOON_WKT


def _raster(
    values: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    units: str | None = None,
):
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
    return raster(values, grid, valid=valid, units=units)


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


def test_coalesce_preserves_uint64_beyond_float_exact_range_across_modes(
    tmp_path,
):
    import rasterio

    value = 2**63 + 123
    missing = _raster(
        np.array([[0, 1]], dtype=np.uint64),
        valid=np.array([[False, True]], dtype=np.bool_),
    )
    fallback = _raster(np.array([[value, value + 1]], dtype=np.uint64))

    eager = coalesce(missing, fallback)
    expression = coalesce(missing.expression(), fallback.expression())
    computed = compute(expression)
    assert eager.dtype == expression.dtype == computed.dtype == np.dtype(np.uint64)
    assert int(eager.values[0, 0]) == value
    np.testing.assert_array_equal(computed.values, eager.values)

    output = write(
        tmp_path / "exact-coalesce.tif",
        expression,
        window_width=1,
        window_height=1,
    )
    with rasterio.open(output) as dataset:
        stored = dataset.read(1)
        assert stored.dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(stored, eager.values)


@pytest.mark.parametrize("operation", [coalesce, where])
def test_selection_scalar_uses_smallest_exact_integer_dtype(operation):
    values = _raster(
        np.array([[1, 2]], dtype=np.uint8),
        valid=np.array([[False, True]], dtype=np.bool_),
    )
    if operation is coalesce:
        eager = operation(values, 300)
        expression = operation(values.expression(), 300)
    else:
        condition = _raster(np.array([[False, True]], dtype=np.bool_))
        eager = operation(condition, values, 300)
        expression = operation(condition.expression(), values.expression(), 300)
    assert eager.dtype == expression.dtype == np.dtype(np.uint16)
    assert int(eager.values[0, 0]) == 300
    np.testing.assert_array_equal(compute(expression).values, eager.values)


@pytest.mark.parametrize("operation", [coalesce, where])
def test_selection_rejects_inexact_int64_uint64_union(operation):
    signed = _raster(np.array([[-1]], dtype=np.int64))
    unsigned = _raster(np.array([[2**63 + 1]], dtype=np.uint64))
    args = (
        (signed, unsigned)
        if operation is coalesce
        else (_raster(np.array([[True]], dtype=np.bool_)), signed, unsigned)
    )
    with pytest.raises(MapAlgebraDTypeError) as eager_error:
        operation(*args)
    assert eager_error.value.code == "map_algebra_no_exact_promotion"

    expression_args = tuple(
        value.expression() if hasattr(value, "expression") else value
        for value in args
    )
    with pytest.raises(MapAlgebraDTypeError) as expression_error:
        operation(*expression_args)
    assert expression_error.value.code == "map_algebra_no_exact_promotion"


@pytest.mark.parametrize("operation", [coalesce, where])
def test_selection_keeps_float32_for_representable_python_float(operation):
    values = _raster(np.array([[1.0, 2.0]], dtype=np.float32))
    if operation is coalesce:
        eager = operation(values, -1.0)
        expression = operation(values.expression(), -1.0)
    else:
        condition = _raster(np.array([[True, False]], dtype=np.bool_))
        eager = operation(condition, values, -1.0)
        expression = operation(condition.expression(), values.expression(), -1.0)
    assert eager.dtype == expression.dtype == np.dtype(np.float32)
    assert compute(expression).dtype == np.dtype(np.float32)


@pytest.mark.parametrize("operation", [coalesce, where])
def test_selection_promotes_float32_for_out_of_range_python_float(operation):
    values = _raster(
        np.array([[1.0]], dtype=np.float32),
        valid=np.array([[False]], dtype=np.bool_),
    )
    scalar = float(np.finfo(np.float32).max) * 2.0
    if operation is coalesce:
        eager = operation(values, scalar)
        expression = operation(values.expression(), scalar)
    else:
        condition = _raster(np.array([[False]], dtype=np.bool_))
        eager = operation(condition, values, scalar)
        expression = operation(condition.expression(), values.expression(), scalar)
    assert eager.dtype == expression.dtype == np.dtype(np.float64)
    assert np.isfinite(eager.values[0, 0])
    assert compute(expression).values[0, 0] == scalar


@pytest.mark.parametrize("operation", [coalesce, where])
def test_selection_units_are_preserved_and_mismatches_are_rejected(operation):
    metres = _raster(np.array([[1.0]], dtype=np.float32)).with_units("metres")
    seconds = _raster(np.array([[2.0]], dtype=np.float32)).with_units("seconds")
    condition = _raster(np.array([[True]], dtype=np.bool_))
    matching_args = (
        (metres, metres)
        if operation is coalesce
        else (condition, metres, metres)
    )
    assert operation(*matching_args).units == "metres"
    matching_expression_args = tuple(
        value.expression() if hasattr(value, "expression") else value
        for value in matching_args
    )
    matching_expression = operation(*matching_expression_args)
    assert matching_expression.units == "metres"
    assert compute(matching_expression).units == "metres"

    mismatching_args = (
        (metres, seconds)
        if operation is coalesce
        else (condition, metres, seconds)
    )
    with pytest.raises(MapAlgebraUnitError) as error:
        operation(*mismatching_args)
    assert error.value.code == "map_algebra_unit_mismatch"
    mismatching_expression_args = tuple(
        value.expression() if hasattr(value, "expression") else value
        for value in mismatching_args
    )
    with pytest.raises(MapAlgebraUnitError) as expression_error:
        operation(*mismatching_expression_args)
    assert expression_error.value.code == "map_algebra_unit_mismatch"


@pytest.mark.parametrize("operation_id", ["local.where", "local.coalesce"])
def test_selection_registry_records_corrected_semantics(operation_id):
    description = describe_operation(operation_id)
    assert description["version"] == 2
    assert description["output_dtype_rule"] == "exact common selection dtype"
    assert "matching units" in description["output_units_rule"]


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
        add(source, 1, output_units="metres")


def test_invalid_cast_dtype_is_structured():
    source = _raster(np.array([[1.0]], dtype=np.float32))
    with pytest.raises(MapAlgebraDTypeError) as error:
        cast(source, "not-a-dtype")
    assert error.value.code == "map_algebra_unsupported_dtype"


def test_integer_power_detects_overflow_without_floating_intermediate():
    source = _raster(np.array([[12]], dtype=np.int8))
    with pytest.raises(MapAlgebraDTypeError) as error:
        power(source, 2)
    assert error.value.code == "map_algebra_overflow"
    assert error.value.details["operation"] == "power"


def test_integer_power_promotes_exactly_with_expression_parity():
    source = _raster(np.array([[-12, 12]], dtype=np.int8))
    eager = power(source, 2, overflow="promote")
    expression = power(source.expression(), 2, overflow="promote")
    assert eager.dtype == np.dtype(np.int16)
    assert expression.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(eager.values, np.array([[144, 144]], dtype=np.int16))
    np.testing.assert_array_equal(compute(expression).values, eager.values)


def test_integer_power_scalar_left_and_raster_exponent_have_expression_parity():
    exponent = _raster(np.array([[0, 3, 7]], dtype=np.uint8))
    eager = power(2, exponent)
    expression = power(2, exponent.expression())
    np.testing.assert_array_equal(
        eager.values, np.array([[1, 8, 128]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(compute(expression).values, eager.values)


def test_uint64_power_one_preserves_value_beyond_float_exact_range():
    value = 2**63 + 123
    result = power(_raster(np.array([[value]], dtype=np.uint64)), 1)
    assert result.dtype == np.dtype(np.uint64)
    assert int(result.values[0, 0]) == value


@pytest.mark.parametrize("as_expression", [False, True])
def test_unit_bearing_power_requires_output_units(as_expression):
    source = _raster(np.array([[2.0]], dtype=np.float32), units="metres")
    operand = source.expression() if as_expression else source
    with pytest.raises(MapAlgebraUnitError) as error:
        power(operand, 2)
    assert error.value.code == "map_algebra_missing_output_units"
    assert error.value.details["base_units"] == "metres"


def test_power_one_preserves_units_without_explicit_output_units():
    source = _raster(np.array([[2.0]], dtype=np.float32), units="metres")
    eager = power(source, 1)
    expression = power(source.expression(), 1)
    assert eager.units == expression.units == "metres"
    assert compute(expression).units == "metres"


def test_unit_bearing_power_explicit_units_have_eager_expression_parity():
    source = _raster(np.array([[2.0, 3.0]], dtype=np.float32), units="metres")
    eager = power(source, 2, output_units="square metres")
    expression = power(
        source.expression(), 2, output_units="square metres",
    )
    assert eager.units == expression.units == "square metres"
    computed = compute(expression)
    assert computed.units == "square metres"
    np.testing.assert_array_equal(computed.values, eager.values)

    other = power(source.expression(), 2, output_units="m2")
    assert expression.scientific_identity() != other.scientific_identity()

    whitespace = power(
        source.expression(), 2, output_units="  square metres  ",
    )
    assert whitespace.units == "square metres"
    assert whitespace.scientific_identity() == expression.scientific_identity()

    unitless = _raster(np.array([[2.0]], dtype=np.float32)).expression()
    assert power(unitless, 2).scientific_identity() == power(
        unitless, 2, output_units=None,
    ).scientific_identity()


@pytest.mark.parametrize("as_expression", [False, True])
def test_unit_bearing_base_rejects_raster_exponent(as_expression):
    base = _raster(np.array([[2.0]], dtype=np.float32), units="metres")
    exponent = _raster(np.array([[2.0]], dtype=np.float32))
    left = base.expression() if as_expression else base
    right = exponent.expression() if as_expression else exponent
    with pytest.raises(MapAlgebraUnitError) as error:
        power(left, right, output_units="square metres")
    assert error.value.code == "map_algebra_unitful_power_requires_scalar_exponent"


@pytest.mark.parametrize("scalar_base", [False, True])
@pytest.mark.parametrize("as_expression", [False, True])
def test_raster_exponent_must_be_dimensionless(scalar_base, as_expression):
    exponent = _raster(
        np.array([[2.0]], dtype=np.float32), units="dimensioned exponent",
    )
    exponent_operand = exponent.expression() if as_expression else exponent
    base = (
        2.0
        if scalar_base
        else _raster(np.array([[2.0]], dtype=np.float32))
    )
    if as_expression and not scalar_base:
        base = base.expression()
    with pytest.raises(MapAlgebraUnitError) as error:
        power(base, exponent_operand)
    assert error.value.code == "map_algebra_dimensioned_exponent"


@pytest.mark.parametrize("as_expression", [False, True])
def test_raster_exponent_rejects_fixed_output_units(as_expression):
    base = _raster(np.array([[2.0]], dtype=np.float32))
    exponent = _raster(np.array([[2.0]], dtype=np.float32))
    left = base.expression() if as_expression else base
    right = exponent.expression() if as_expression else exponent
    with pytest.raises(MapAlgebraUnitError) as error:
        power(left, right, output_units="fixed")
    assert error.value.code == "map_algebra_unexpected_output_units"


@pytest.mark.parametrize("output_units", ["", "   ", 3])
def test_power_output_units_validation_is_structured(output_units):
    source = _raster(np.array([[2.0]], dtype=np.float32), units="metres")
    with pytest.raises(MapAlgebraUnitError) as error:
        power(source, 2, output_units=output_units)
    assert error.value.code == "map_algebra_invalid_output_units"


def test_integer_power_wrap_is_explicit():
    source = _raster(np.array([[16]], dtype=np.uint8))
    result = power(source, 2, overflow="wrap")
    assert result.dtype == np.dtype(np.uint8)
    assert int(result.values[0, 0]) == 0


@pytest.mark.parametrize(
    ("policy", "valid"),
    [("invalid", False), ("keep", True)],
)
def test_negative_integer_exponent_numeric_policy(policy, valid):
    source = _raster(np.array([[2]], dtype=np.int16))
    result = power(source, -1, numeric_errors=policy)
    assert bool(result.valid[0, 0]) is valid
    assert int(result.values[0, 0]) == 1


def test_negative_integer_exponent_raise_is_structured():
    source = _raster(np.array([[2]], dtype=np.int16))
    with pytest.raises(MapAlgebraOperationError) as error:
        power(source, -1, numeric_errors="raise")
    assert error.value.code == "map_algebra_numeric_error"
    assert error.value.details["operation"] == "power"


def test_power_overflow_at_invalid_pixel_is_ignored():
    source = _raster(
        np.array([[100, 2]], dtype=np.int8),
        valid=np.array([[False, True]], dtype=np.bool_),
    )
    result = power(source, 2)
    np.testing.assert_array_equal(result.valid, np.array([[False, True]]))
    assert int(result.values[0, 1]) == 4


def test_cast_overflow_raise_is_value_aware():
    source = _raster(np.array([[127.9, 128.0]], dtype=np.float64))
    valid_first = _raster(
        source.values,
        valid=np.array([[True, False]], dtype=np.bool_),
    )
    result = cast(valid_first, "int8", casting="unsafe")
    assert int(result.values[0, 0]) == 127
    with pytest.raises(MapAlgebraDTypeError) as error:
        cast(source, "int8", casting="unsafe")
    assert error.value.code == "map_algebra_cast_overflow"


def test_cast_integer_wrap_is_explicit_and_expression_identified():
    source = _raster(np.array([[256]], dtype=np.int16))
    eager = cast(source, "uint8", casting="unsafe", overflow="wrap")
    expression = cast(
        source.expression(), "uint8", casting="unsafe", overflow="wrap",
    )
    assert int(eager.values[0, 0]) == 0
    assert int(compute(expression).values[0, 0]) == 0
    assert expression._params_dict["overflow"] == "wrap"


def test_cast_wrap_rejects_noninteger_source():
    source = _raster(np.array([[1.0]], dtype=np.float32))
    with pytest.raises(MapAlgebraDTypeError) as error:
        cast(source, "int8", casting="unsafe", overflow="wrap")
    assert error.value.code == "map_algebra_invalid_cast_overflow"


def test_uint64_to_int64_cast_overflow_is_exact():
    source = _raster(np.array([[2**63 - 1, 2**63]], dtype=np.uint64))
    with pytest.raises(MapAlgebraDTypeError) as error:
        cast(source, "int64", casting="unsafe")
    assert error.value.code == "map_algebra_cast_overflow"
    assert error.value.details["affected_values"] == 1


def test_boolean_cast_overflow_checks_valid_source_range():
    source = _raster(np.array([[-1, 0, 1, 2]], dtype=np.int8))
    with pytest.raises(MapAlgebraDTypeError) as error:
        cast(source, "bool", casting="unsafe")
    assert error.value.code == "map_algebra_cast_overflow"
    assert error.value.details["affected_values"] == 2

    representable = _raster(np.array([[0, 1]], dtype=np.int8))
    result = cast(representable, "bool", casting="unsafe")
    np.testing.assert_array_equal(result.values, np.array([[False, True]]))


def test_float64_to_uint64_uses_exact_representable_boundary():
    too_large = _raster(np.array([[float(2**64)]], dtype=np.float64))
    with pytest.raises(MapAlgebraDTypeError) as error:
        cast(too_large, "uint64", casting="unsafe")
    assert error.value.code == "map_algebra_cast_overflow"

    largest_float_below = np.nextafter(np.float64(2**64), np.float64(-np.inf))
    result = cast(
        _raster(np.array([[largest_float_below]], dtype=np.float64)),
        "uint64",
        casting="unsafe",
    )
    assert int(result.values[0, 0]) == int(largest_float_below)


@pytest.mark.parametrize(
    ("policy", "expected_valid"),
    [("invalid", False), ("keep", True)],
)
def test_float32_arithmetic_overflow_uses_numeric_error_policy(
    policy, expected_valid,
):
    source = _raster(np.array([[np.finfo(np.float32).max]], dtype=np.float32))
    result = multiply(source, np.float32(2), numeric_errors=policy)
    assert result.dtype == np.dtype(np.float32)
    assert np.isinf(result.values[0, 0])
    assert bool(result.valid[0, 0]) is expected_valid


def test_float32_arithmetic_overflow_raise_is_structured():
    source = _raster(np.array([[np.finfo(np.float32).max]], dtype=np.float32))
    with pytest.raises(MapAlgebraOperationError) as error:
        multiply(source, np.float32(2), numeric_errors="raise")
    assert error.value.code == "map_algebra_numeric_error"
    assert error.value.details == {
        "operation": "multiply", "affected_pixels": 1,
    }


def test_power_and_cast_have_windowed_eager_parity(tmp_path):
    import rasterio

    source = _raster(
        np.array([[12, -12, 2], [3, 4, 5]], dtype=np.int8),
        valid=np.array([[True, False, True], [True, True, True]], dtype=np.bool_),
    )
    expression = cast(
        power(source.expression(), 2, overflow="promote"),
        "uint8",
        casting="unsafe",
        overflow="wrap",
    )
    eager = compute(expression)
    output = write(
        tmp_path / "power-cast.tif", expression,
        window_width=2, window_height=1, invalid_value=0,
    )
    with rasterio.open(output) as dataset:
        written = dataset.read(1)
        np.testing.assert_array_equal(written[eager.valid], eager.values[eager.valid])
        assert written[0, 1] == 0
        np.testing.assert_array_equal(dataset.read_masks(1) > 0, eager.valid)
        assert dataset.dtypes[0] == "uint8"


def test_unit_bearing_power_has_windowed_parity(tmp_path):
    import rasterio

    source = _raster(
        np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        units="metres",
    )
    expression = power(
        source.expression(), 2, output_units="square metres",
    )
    eager = compute(expression)
    output = write(
        tmp_path / "unit-power.tif", expression,
        window_width=1, window_height=1,
    )
    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read(1), eager.values)
        np.testing.assert_array_equal(dataset.read_masks(1) > 0, eager.valid)


_SUPPORTED_DTYPES = (
    np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
    np.int64, np.uint64, np.float32, np.float64,
)


@pytest.mark.parametrize("left_dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("right_dtype", _SUPPORTED_DTYPES)
def test_add_dtype_pair_matrix_matches_shared_numpy2_rule(left_dtype, right_dtype):
    left = _raster(np.array([[0]], dtype=left_dtype))
    right = _raster(np.array([[0]], dtype=right_dtype))
    result = add(left, right)
    assert result.dtype == np.dtype(np.result_type(left_dtype, right_dtype))


@pytest.mark.parametrize("source_dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("target_dtype", _SUPPORTED_DTYPES)
def test_unsafe_cast_pair_matrix_for_representable_values(source_dtype, target_dtype):
    source = _raster(np.array([[0, 1]], dtype=source_dtype))
    result = cast(source, target_dtype, casting="unsafe")
    assert result.dtype == np.dtype(target_dtype)
    np.testing.assert_array_equal(
        result.values, np.array([[0, 1]], dtype=target_dtype),
    )
