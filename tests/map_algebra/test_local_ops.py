from __future__ import annotations

import numpy as np
import pytest

from lunarscout.errors import (
    MapAlgebraDTypeError,
    MapAlgebraError,
    MapAlgebraUnitError,
    RasterValidationError,
)
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    absolute,
    add,
    arctan2,
    cast,
    ceil,
    clip,
    coalesce,
    cos,
    divide,
    describe_operation,
    equal,
    fill_invalid,
    floor,
    greater,
    greater_equal,
    hypot,
    invalid,
    is_invalid,
    is_valid,
    less,
    less_equal,
    log,
    log10,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    negative,
    not_equal,
    round,
    set_invalid,
    sin,
    sqrt,
    square,
    subtract,
    trunc,
    where,
)
from lunarscout.map_algebra import raster as ma_raster
from lunarscout.raster import Raster


def _georef_3x4(nodata=None) -> GeoReference:
    return GeoReference(
        projection_wkt=(
            'PROJCS["test",GEOGCS["test",DATUM["test",SPHEROID["test",1.0,0.0]],'
            'PRIMEM["test",0],UNIT["degree",0.0174533]],'
            'PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],'
            'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
            'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'
        ),
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1 +units=m +no_defs",
        affine_transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        width=4,
        height=3,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=nodata,
    )


def _make_raster(values, georef=None, valid=None, units=None, name=None):
    if georef is None:
        georef = GeoReference(
            projection_wkt=(
                'PROJCS["test",GEOGCS["test",DATUM["test",SPHEROID["test",1.0,0.0]],'
                'PRIMEM["test",0],UNIT["degree",0.0174533]],'
                'PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],'
                'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
                'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'
            ),
            projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1 +units=m +no_defs",
            affine_transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            width=values.shape[1],
            height=values.shape[0],
            pixel_size_x=1.0,
            pixel_size_y=-1.0,
            nodata=None,
        )
    if valid is None:
        valid = np.ones(values.shape, dtype=np.bool_)
    return ma_raster(values, georef, valid=valid, units=units, name=name)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_add_raster_raster(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        b = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        r = add(a, b)
        np.testing.assert_array_equal(r.values, np.array([[11.0, 22.0], [33.0, 44.0]], dtype=np.float32))

    def test_add_raster_scalar(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = add(a, 10.0)
        np.testing.assert_array_equal(r.values, np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32))

    def test_add_scalar_raster(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = add(100, a)
        np.testing.assert_array_equal(r.values, np.array([[101.0, 102.0], [103.0, 104.0]], dtype=np.float32))

    def test_subtract_raster_raster(self):
        a = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        b = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = subtract(a, b)
        np.testing.assert_array_equal(r.values, np.array([[9.0, 18.0], [27.0, 36.0]], dtype=np.float32))

    def test_subtract_scalar_from_raster(self):
        a = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        r = subtract(a, 1)
        np.testing.assert_array_equal(r.values, np.array([[9.0, 19.0], [29.0, 39.0]], dtype=np.float32))

    def test_subtract_raster_from_scalar(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = subtract(100, a)
        np.testing.assert_array_equal(r.values, np.array([[99.0, 98.0], [97.0, 96.0]], dtype=np.float32))

    def test_multiply_raster_raster(self):
        a = _make_raster(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        b = _make_raster(np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32))
        r = multiply(a, b)
        np.testing.assert_array_equal(r.values, np.array([[20.0, 30.0], [40.0, 50.0]], dtype=np.float32))

    def test_multiply_scalar(self):
        a = _make_raster(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        r = multiply(a, 3)
        np.testing.assert_array_equal(r.values, np.array([[6.0, 9.0], [12.0, 15.0]], dtype=np.float32))

    def test_divide_raster_by_scalar(self):
        a = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        r = divide(a, 10)
        np.testing.assert_array_equal(r.values, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    def test_divide_scalar_by_raster(self):
        a = _make_raster(np.array([[2.0, 4.0], [5.0, 10.0]], dtype=np.float32))
        r = divide(100, a)
        np.testing.assert_array_equal(r.values, np.array([[50.0, 25.0], [20.0, 10.0]], dtype=np.float32))

    def test_negative(self):
        a = _make_raster(np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32))
        r = negative(a)
        np.testing.assert_array_equal(r.values, np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32))

    def test_absolute(self):
        a = _make_raster(np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32))
        r = absolute(a)
        np.testing.assert_array_equal(r.values, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    def test_minimum(self):
        a = _make_raster(np.array([[5.0, 2.0], [8.0, 1.0]], dtype=np.float32))
        b = _make_raster(np.array([[3.0, 7.0], [6.0, 9.0]], dtype=np.float32))
        r = minimum(a, b)
        np.testing.assert_array_equal(r.values, np.array([[3.0, 2.0], [6.0, 1.0]], dtype=np.float32))

    def test_maximum(self):
        a = _make_raster(np.array([[5.0, 2.0], [8.0, 1.0]], dtype=np.float32))
        b = _make_raster(np.array([[3.0, 7.0], [6.0, 9.0]], dtype=np.float32))
        r = maximum(a, b)
        np.testing.assert_array_equal(r.values, np.array([[5.0, 7.0], [8.0, 9.0]], dtype=np.float32))

    def test_add_validity_intersection(self):
        a = _make_raster(np.zeros((2, 2), dtype=np.float32), valid=np.array([[True, False], [True, True]]))
        b = _make_raster(np.zeros((2, 2), dtype=np.float32), valid=np.array([[False, True], [True, True]]))
        r = add(a, b)
        assert not r.valid[0, 0]
        assert not r.valid[0, 1]
        assert r.valid[1, 0]
        assert r.valid[1, 1]

    def test_add_no_raster_operand_raises(self):
        with pytest.raises(MapAlgebraError, match="requires at least one Raster operand"):
            add(1, 2)

    def test_add_mismatched_grids_raises(self):
        a = _make_raster(np.ones((2, 2), dtype=np.float32))
        b = _make_raster(np.ones((3, 4), dtype=np.float32))
        with pytest.raises(MapAlgebraError):
            add(a, b)

    def test_add_preserves_units(self):
        a = _make_raster(np.ones((2, 2), dtype=np.float32), units="meters")
        b = _make_raster(np.ones((2, 2), dtype=np.float32), units="meters")
        r = add(a, b)
        assert r.units == "meters"

    def test_add_mismatched_units_raises(self):
        a = _make_raster(np.ones((2, 2), dtype=np.float32), units="meters")
        b = _make_raster(np.ones((2, 2), dtype=np.float32), units="degrees")
        with pytest.raises(MapAlgebraUnitError):
            add(a, b)

    def test_multiply_raster_by_scalar_preserves_units(self):
        a = _make_raster(np.ones((2, 2), dtype=np.float32), units="meters")
        r = multiply(a, 2)
        assert r.units == "meters"

    def test_divide_produces_at_least_float32(self):
        a = _make_raster(np.array([[3, 1]], dtype=np.int32))
        r = divide(a, 2)
        assert r.dtype in (np.dtype(np.float32), np.dtype(np.float64))

    def test_add_with_invalid_propagates(self):
        a = _make_raster(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            valid=np.array([[True, False], [True, True]]),
        )
        b = _make_raster(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            valid=np.array([[True, True], [False, True]]),
        )
        r = add(a, b)
        assert not r.valid[0, 1]
        assert not r.valid[1, 0]
        assert r.valid[0, 0]
        assert r.valid[1, 1]


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_less(self):
        a = _make_raster(np.array([[1.0, 5.0], [3.0, 0.0]], dtype=np.float32))
        b = _make_raster(np.array([[2.0, 3.0], [3.0, 1.0]], dtype=np.float32))
        r = less(a, b)
        np.testing.assert_array_equal(r.values, np.array([[True, False], [False, True]]))

    def test_less_scalar(self):
        a = _make_raster(np.array([[1.0, 5.0], [3.0, 0.0]], dtype=np.float32))
        r = less(a, 3)
        np.testing.assert_array_equal(r.values, np.array([[True, False], [False, True]]))

    def test_less_equal(self):
        a = _make_raster(np.array([[1.0, 5.0], [3.0, 0.0]], dtype=np.float32))
        b = _make_raster(np.array([[2.0, 3.0], [3.0, 1.0]], dtype=np.float32))
        r = less_equal(a, b)
        np.testing.assert_array_equal(r.values, np.array([[True, False], [True, True]]))

    def test_greater(self):
        a = _make_raster(np.array([[5.0, 1.0], [3.0, 7.0]], dtype=np.float32))
        b = _make_raster(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        r = greater(a, b)
        np.testing.assert_array_equal(r.values, np.array([[True, False], [False, True]]))

    def test_greater_equal(self):
        a = _make_raster(np.array([[5.0, 1.0], [3.0, 7.0]], dtype=np.float32))
        b = _make_raster(np.array([[2.0, 3.0], [3.0, 5.0]], dtype=np.float32))
        r = greater_equal(a, b)
        np.testing.assert_array_equal(r.values, np.array([[True, False], [True, True]]))

    def test_equal_raster(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        b = _make_raster(np.array([[1.0, 99.0], [3.0, 4.0]], dtype=np.float32))
        r = equal(a, b)
        np.testing.assert_array_equal(r.values, np.array([[True, False], [True, True]]))

    def test_not_equal_raster(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        b = _make_raster(np.array([[1.0, 99.0], [3.0, 4.0]], dtype=np.float32))
        r = not_equal(a, b)
        np.testing.assert_array_equal(r.values, np.array([[False, True], [False, False]]))

    def test_comparison_invalid_pixels_are_invalid(self):
        a = _make_raster(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            valid=np.array([[False, True], [True, True]]),
        )
        b = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = less(a, b)
        assert not r.valid[0, 0]
        assert r.valid[0, 1]

    def test_comparison_returns_bool_dtype(self):
        a = _make_raster(np.ones((2, 2), dtype=np.float32))
        b = _make_raster(np.ones((2, 2), dtype=np.float32))
        r = less(a, b)
        assert r.dtype == np.dtype(np.bool_)

    def test_comparison_mismatched_units_raises(self):
        a = _make_raster(np.ones((2, 2), dtype=np.float32), units="meters")
        b = _make_raster(np.ones((2, 2), dtype=np.float32), units="degrees")
        with pytest.raises(MapAlgebraUnitError):
            less(a, b)


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------


class TestBoolean:
    def test_logical_not(self):
        a = _make_raster(np.array([[True, False], [True, False]]))
        r = logical_not(a)
        np.testing.assert_array_equal(r.values, np.array([[False, True], [False, True]]))

    def test_logical_and(self):
        a = _make_raster(np.array([[True, True], [False, False]]))
        b = _make_raster(np.array([[True, False], [True, False]]))
        r = logical_and(a, b)
        np.testing.assert_array_equal(r.values, np.array([[True, False], [False, False]]))

    def test_logical_or(self):
        a = _make_raster(np.array([[True, True], [False, False]]))
        b = _make_raster(np.array([[True, False], [True, False]]))
        r = logical_or(a, b)
        np.testing.assert_array_equal(r.values, np.array([[True, True], [True, False]]))

    def test_logical_xor(self):
        a = _make_raster(np.array([[True, True], [False, False]]))
        b = _make_raster(np.array([[True, False], [True, False]]))
        r = logical_xor(a, b)
        np.testing.assert_array_equal(r.values, np.array([[False, True], [True, False]]))

    def test_logical_requires_boolean(self):
        a = _make_raster(np.ones((2, 2), dtype=np.float32))
        with pytest.raises(MapAlgebraDTypeError, match="boolean"):
            logical_not(a)

    def test_logical_and_validity_intersection(self):
        a = _make_raster(
            np.array([[True, True], [False, False]]),
            valid=np.array([[True, False], [True, True]]),
        )
        b = _make_raster(
            np.array([[True, False], [True, False]]),
            valid=np.array([[False, True], [True, True]]),
        )
        r = logical_and(a, b)
        assert not r.valid[0, 0]
        assert not r.valid[0, 1]


# ---------------------------------------------------------------------------
# where / coalesce / validity helpers
# ---------------------------------------------------------------------------


class TestWhere:
    def test_where_both_rasters(self):
        cond = _make_raster(np.array([[True, False], [False, True]]))
        x = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        y = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = where(cond, x, y)
        np.testing.assert_array_equal(r.values, np.array([[10.0, 2.0], [3.0, 40.0]], dtype=np.float32))

    def test_where_x_invalid(self):
        cond = _make_raster(np.array([[True, False], [False, True]]))
        y = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = where(cond, invalid, y)
        np.testing.assert_array_equal(r.values, np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32))
        assert not r.valid[0, 0]
        assert not r.valid[1, 1]
        assert r.valid[0, 1]
        assert r.valid[1, 0]

    def test_where_y_invalid(self):
        cond = _make_raster(np.array([[True, False], [False, True]]))
        x = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        r = where(cond, x, invalid)
        assert not r.valid[0, 1]
        assert not r.valid[1, 0]
        assert r.valid[0, 0]
        assert r.valid[1, 1]

    def test_where_raster_scalar(self):
        cond = _make_raster(np.array([[True, False], [True, True]]))
        x = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        r = where(cond, x, -1.0)
        np.testing.assert_array_equal(r.values, np.array([[10.0, -1.0], [30.0, 40.0]], dtype=np.float32))

    def test_where_scalar_raster(self):
        cond = _make_raster(np.array([[True, False], [False, True]]))
        y = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = where(cond, 100.0, y)
        np.testing.assert_array_equal(r.values, np.array([[100.0, 2.0], [3.0, 100.0]], dtype=np.float32))

    def test_where_condition_invalid_propagates(self):
        cond = _make_raster(
            np.array([[True, False], [False, True]]),
            valid=np.array([[False, True], [True, True]]),
        )
        x = _make_raster(np.ones((2, 2), dtype=np.float32))
        y = _make_raster(np.zeros((2, 2), dtype=np.float32))
        r = where(cond, x, y)
        assert not r.valid[0, 0]
        assert r.valid[0, 1]


class TestCoalesce:
    def test_coalesce_first_valid(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        b = _make_raster(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
        r = coalesce(a, b)
        np.testing.assert_array_equal(r.values, a.values)

    def test_coalesce_fallback(self):
        a = _make_raster(
            np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.float32),
            valid=np.array([[True, False], [False, True]]),
        )
        b = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        r = coalesce(a, b)
        assert r.values[0, 0] == 1.0
        assert r.values[0, 1] == 20.0
        assert r.values[1, 0] == 30.0
        assert r.values[1, 1] == 4.0

    def test_coalesce_scalar_fallback(self):
        a = _make_raster(
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            valid=np.array([[True, False], [False, False]]),
        )
        r = coalesce(a, -1.0)
        assert r.values[0, 0] == 1.0
        assert r.values[0, 1] == -1.0
        assert r.values[1, 0] == -1.0
        assert r.values[1, 1] == -1.0
        assert r.all_valid

    def test_coalesce_all_invalid(self):
        a = _make_raster(
            np.zeros((2, 2), dtype=np.float32),
            valid=np.zeros((2, 2), dtype=np.bool_),
        )
        b = _make_raster(
            np.ones((2, 2), dtype=np.float32),
            valid=np.zeros((2, 2), dtype=np.bool_),
        )
        r = coalesce(a, b)
        assert not r.valid.any()


class TestValidityHelpers:
    def test_is_valid(self):
        a = _make_raster(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            valid=np.array([[True, False], [True, True]]),
        )
        r = is_valid(a)
        np.testing.assert_array_equal(r.values, np.array([[True, False], [True, True]]))
        assert r.all_valid

    def test_is_invalid(self):
        a = _make_raster(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            valid=np.array([[True, False], [True, True]]),
        )
        r = is_invalid(a)
        np.testing.assert_array_equal(r.values, np.array([[False, True], [False, False]]))

    def test_set_invalid(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        mask = _make_raster(np.array([[False, True], [False, False]]))
        r = set_invalid(a, mask)
        assert not r.valid[0, 1]
        assert r.valid[0, 0]

    def test_fill_invalid(self):
        a = _make_raster(
            np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.float32),
            valid=np.array([[True, False], [False, True]]),
        )
        r = fill_invalid(a, -9999.0)
        assert r.values[0, 1] == -9999.0
        assert r.values[1, 0] == -9999.0
        assert r.all_valid

    @pytest.mark.parametrize("as_expression", [False, True])
    def test_fill_invalid_rejects_lossy_integer_fill(self, as_expression):
        raster_value = _make_raster(
            np.array([[1, 99], [3, 4]], dtype=np.uint8),
            valid=np.array([[True, False], [True, True]], dtype=np.bool_),
        )
        operand = raster_value.expression() if as_expression else raster_value
        with pytest.raises(RasterValidationError) as error:
            fill_invalid(operand, 300)
        assert error.value.code == "raster_unrepresentable_nodata"

    def test_fill_invalid_preserves_exact_uint64(self):
        fill = 2**63 + 123
        raster_value = _make_raster(
            np.array([[1, 99], [3, 4]], dtype=np.uint64),
            valid=np.array([[True, False], [True, True]], dtype=np.bool_),
        )
        result = fill_invalid(raster_value, fill)
        assert result.dtype == np.dtype(np.uint64)
        assert int(result.values[0, 1]) == fill
        assert result.all_valid

    def test_fill_invalid_registry_records_exact_encoding_contract(self):
        description = describe_operation("local.fill_invalid")
        assert description["version"] == 2
        assert "exactly representable" in description["output_dtype_rule"]
        assert description["validity_rule"] == "all cells valid after exact fill"


# ---------------------------------------------------------------------------
# clip / cast
# ---------------------------------------------------------------------------


class TestClip:
    def test_clip_lower(self):
        a = _make_raster(np.array([[-5.0, 0.0], [5.0, 10.0]], dtype=np.float32))
        r = clip(a, lower=0.0)
        np.testing.assert_array_equal(r.values, np.array([[0.0, 0.0], [5.0, 10.0]], dtype=np.float32))

    def test_clip_upper(self):
        a = _make_raster(np.array([[-5.0, 0.0], [5.0, 20.0]], dtype=np.float32))
        r = clip(a, upper=10.0)
        np.testing.assert_array_equal(r.values, np.array([[-5.0, 0.0], [5.0, 10.0]], dtype=np.float32))

    def test_clip_both(self):
        a = _make_raster(np.array([[-5.0, 0.0], [5.0, 20.0]], dtype=np.float32))
        r = clip(a, lower=0.0, upper=10.0)
        np.testing.assert_array_equal(r.values, np.array([[0.0, 0.0], [5.0, 10.0]], dtype=np.float32))

    def test_clip_preserves_validity(self):
        a = _make_raster(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            valid=np.array([[True, False], [True, True]]),
        )
        r = clip(a, lower=0.0, upper=10.0)
        assert not r.valid[0, 1]


class TestCast:
    def test_cast_to_float64(self):
        a = _make_raster(np.array([[1, 2], [3, 4]], dtype=np.int32))
        r = cast(a, "float64")
        assert r.dtype == np.dtype(np.float64)

    def test_cast_safe_rejects_lossy(self):
        a = _make_raster(np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32))
        with pytest.raises((MapAlgebraDTypeError, TypeError)):
            cast(a, "int32", casting="safe")

    def test_cast_unsafe_allows_lossy(self):
        a = _make_raster(np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32))
        r = cast(a, "int32", casting="unsafe")
        assert r.dtype == np.dtype(np.int32)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


class TestMath:
    def test_sqrt(self):
        a = _make_raster(np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32))
        r = sqrt(a)
        np.testing.assert_allclose(r.values, np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))

    def test_sqrt_negative_invalid(self):
        a = _make_raster(np.array([[4.0, -1.0], [9.0, 16.0]], dtype=np.float32))
        r = sqrt(a)
        assert not r.valid[0, 1]
        assert r.valid[0, 0]

    def test_square(self):
        a = _make_raster(np.array([[2.0, -3.0], [4.0, 0.0]], dtype=np.float32))
        r = square(a)
        np.testing.assert_allclose(r.values, np.array([[4.0, 9.0], [16.0, 0.0]], dtype=np.float32))

    def test_log(self):
        a = _make_raster(np.array([[1.0, np.e], [np.e**2, 1.0]], dtype=np.float32))
        r = log(a)
        np.testing.assert_allclose(r.values, np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32), atol=1e-6)

    def test_log10(self):
        a = _make_raster(np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float32))
        r = log10(a)
        np.testing.assert_allclose(r.values, np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32), atol=1e-6)

    def test_sin_cos(self):
        a = _make_raster(np.array([[0.0, np.pi / 2], [np.pi, 0.0]], dtype=np.float32), units="radians")
        s = sin(a)
        c = cos(a)
        np.testing.assert_allclose(s.values, np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(c.values, np.array([[1.0, 0.0], [-1.0, 1.0]], dtype=np.float32), atol=1e-6)

    def test_hypot(self):
        a = _make_raster(np.array([[3.0, 5.0]], dtype=np.float32))
        b = _make_raster(np.array([[4.0, 12.0]], dtype=np.float32))
        r = hypot(a, b)
        np.testing.assert_allclose(r.values, np.array([[5.0, 13.0]], dtype=np.float32), atol=1e-5)

    def test_arctan2(self):
        y = _make_raster(np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float32))
        x = _make_raster(np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32))
        r = arctan2(y, x)
        np.testing.assert_allclose(r.values[0, 0], np.pi / 4, atol=1e-6)
        np.testing.assert_allclose(r.values[0, 1], -np.pi / 4, atol=1e-6)
        np.testing.assert_allclose(r.values[1, 1], np.pi / 2, atol=1e-6)

    def test_floor(self):
        a = _make_raster(np.array([[1.3, 2.7], [-1.3, -2.7]], dtype=np.float32))
        r = floor(a)
        np.testing.assert_array_equal(r.values, np.array([[1.0, 2.0], [-2.0, -3.0]], dtype=np.float32))

    def test_ceil(self):
        a = _make_raster(np.array([[1.3, 2.7], [-1.3, -2.7]], dtype=np.float32))
        r = ceil(a)
        np.testing.assert_array_equal(r.values, np.array([[2.0, 3.0], [-1.0, -2.0]], dtype=np.float32))

    def test_trunc(self):
        a = _make_raster(np.array([[1.3, 2.7], [-1.3, -2.7]], dtype=np.float32))
        r = trunc(a)
        np.testing.assert_array_equal(r.values, np.array([[1.0, 2.0], [-1.0, -2.0]], dtype=np.float32))

    def test_round(self):
        a = _make_raster(np.array([[1.3, 2.6], [-1.5, -2.5]], dtype=np.float32))
        r = round(a)
        np.testing.assert_array_equal(r.values, np.array([[1.0, 3.0], [-2.0, -2.0]], dtype=np.float32))


# ---------------------------------------------------------------------------
# Operator overloads
# ---------------------------------------------------------------------------


class TestOperatorOverloads:
    def test_add_operator(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        b = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        r = a + b
        np.testing.assert_array_equal(r.values, np.array([[11.0, 22.0], [33.0, 44.0]], dtype=np.float32))

    def test_radd_operator(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = 10 + a
        np.testing.assert_array_equal(r.values, np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32))

    def test_sub_operator(self):
        a = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        r = a - 5
        np.testing.assert_array_equal(r.values, np.array([[5.0, 15.0], [25.0, 35.0]], dtype=np.float32))

    def test_rsub_operator(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = 100 - a
        np.testing.assert_array_equal(r.values, np.array([[99.0, 98.0], [97.0, 96.0]], dtype=np.float32))

    def test_mul_operator(self):
        a = _make_raster(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        r = a * 10
        np.testing.assert_array_equal(r.values, np.array([[20.0, 30.0], [40.0, 50.0]], dtype=np.float32))

    def test_rmul_operator(self):
        a = _make_raster(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        r = 10 * a
        np.testing.assert_array_equal(r.values, np.array([[20.0, 30.0], [40.0, 50.0]], dtype=np.float32))

    def test_truediv_operator(self):
        a = _make_raster(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
        r = a / 10
        np.testing.assert_array_equal(r.values, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    def test_rtruediv_operator(self):
        a = _make_raster(np.array([[2.0, 4.0], [5.0, 10.0]], dtype=np.float32))
        r = 100 / a
        np.testing.assert_array_equal(r.values, np.array([[50.0, 25.0], [20.0, 10.0]], dtype=np.float32))

    def test_floordiv_operator(self):
        a = _make_raster(np.array([[10, 20], [30, 40]], dtype=np.int32))
        r = a // 7
        np.testing.assert_array_equal(r.values, np.array([[1, 2], [4, 5]]))

    def test_mod_operator(self):
        a = _make_raster(np.array([[10, 20], [30, 40]], dtype=np.int32))
        r = a % 7
        np.testing.assert_array_equal(r.values, np.array([[3, 6], [2, 5]]))

    def test_pow_operator(self):
        a = _make_raster(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        r = a**2
        np.testing.assert_allclose(r.values, np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32))

    def test_neg_operator(self):
        a = _make_raster(np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32))
        r = -a
        np.testing.assert_array_equal(r.values, np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32))

    def test_abs_operator(self):
        a = _make_raster(np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32))
        r = abs(a)
        np.testing.assert_array_equal(r.values, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    def test_lt_operator(self):
        a = _make_raster(np.array([[1.0, 5.0], [3.0, 0.0]], dtype=np.float32))
        r = a < 3
        np.testing.assert_array_equal(r.values, np.array([[True, False], [False, True]]))

    def test_le_operator(self):
        a = _make_raster(np.array([[1.0, 3.0], [3.0, 0.0]], dtype=np.float32))
        r = a <= 3
        np.testing.assert_array_equal(r.values, np.array([[True, True], [True, True]]))

    def test_gt_operator(self):
        a = _make_raster(np.array([[5.0, 1.0], [3.0, 7.0]], dtype=np.float32))
        r = a > 3
        np.testing.assert_array_equal(r.values, np.array([[True, False], [False, True]]))

    def test_ge_operator(self):
        a = _make_raster(np.array([[3.0, 1.0], [3.0, 7.0]], dtype=np.float32))
        r = a >= 3
        np.testing.assert_array_equal(r.values, np.array([[True, False], [True, True]]))

    def test_and_operator(self):
        a = _make_raster(np.array([[True, True], [False, False]]))
        b = _make_raster(np.array([[True, False], [True, False]]))
        r = a & b
        np.testing.assert_array_equal(r.values, np.array([[True, False], [False, False]]))

    def test_or_operator(self):
        a = _make_raster(np.array([[True, True], [False, False]]))
        b = _make_raster(np.array([[True, False], [True, False]]))
        r = a | b
        np.testing.assert_array_equal(r.values, np.array([[True, True], [True, False]]))

    def test_xor_operator(self):
        a = _make_raster(np.array([[True, True], [False, False]]))
        b = _make_raster(np.array([[True, False], [True, False]]))
        r = a ^ b
        np.testing.assert_array_equal(r.values, np.array([[False, True], [True, False]]))

    def test_invert_operator(self):
        a = _make_raster(np.array([[True, False], [True, False]]))
        r = ~a
        np.testing.assert_array_equal(r.values, np.array([[False, True], [False, True]]))

    def test_compound_example_expression(self):
        slope = _make_raster(np.array([[5.0, 10.0, 3.0]], dtype=np.float32), units="degrees")
        sun = _make_raster(np.array([[0.8, 0.7, 0.4]], dtype=np.float32), units="fraction")

        candidate = (slope <= 8.0) & (sun >= 0.60)
        np.testing.assert_array_equal(candidate.values, np.array([[True, False, False]]))
        assert candidate.dtype == np.dtype(np.bool_)

    def test_where_expression(self):
        slope = _make_raster(np.array([[12.0, 5.0, 6.0]], dtype=np.float32))
        sun = _make_raster(np.array([[0.4, 0.7, 0.65]], dtype=np.float32))

        candidate = (slope <= 8.0) & (sun >= 0.60)
        score = where(candidate, sun * 0.4 + slope * 0.6, invalid)

        assert not score.valid[0, 0]
        assert score.valid[0, 1]
        assert score.valid[0, 2]
        np.testing.assert_allclose(score.values[0, 1], 0.7 * 0.4 + 5.0 * 0.6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_add_int_raster_keeps_dtype(self):
        a = _make_raster(np.array([[1, 2], [3, 4]], dtype=np.int32))
        b = _make_raster(np.array([[5, 6], [7, 8]], dtype=np.int32))
        r = add(a, b)
        assert r.dtype == np.dtype(np.int32)

    def test_log_negative_produces_nan_invalid(self):
        a = _make_raster(np.array([[1.0, -1.0], [2.0, 0.0]], dtype=np.float32))
        r = log(a)
        assert not r.valid[0, 1]
        assert r.valid[0, 0]
        assert not r.valid[1, 1]

    def test_invalid_sentinel_repr(self):
        assert repr(invalid) == "ma.invalid"

    def test_invalid_sentinel_unhashable(self):
        assert invalid is not None

    def test_pos_operator_identity(self):
        a = _make_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        r = +a
        assert r.values is a.values
        assert r.valid is a.valid
