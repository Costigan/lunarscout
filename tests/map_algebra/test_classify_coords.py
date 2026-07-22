"""Tests for classification, normalization, coordinates, registry, and validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from lunarscout.georeference import GeoReference
from lunarscout.raster import Raster
from lunarscout.map_algebra import (
    digitize,
    normalize_minmax,
    one_hot,
    standardize,
    reclassify_ranges,
    reclassify_values,
    row_indices,
    column_indices,
    projected_x,
    projected_y,
    longitude,
    latitude,
    describe_operation,
    list_operations,
    compute,
)
from lunarscout.errors import MapAlgebraDTypeError, MapAlgebraError
from lunarscout.map_algebra._validation import _as_raster_operand, _as_expression_operand
from lunarscout.map_algebra._model import RasterExpression

from tests.map_algebra.conftest import _georef  # type: ignore[import-untyped]


def _make_georef(width=10, height=8, pixel_x=20.0, pixel_y=-20.0, nodata=None):
    return _georef(width=width, height=height, pixel_x=pixel_x, pixel_y=pixel_y, nodata=nodata)


def _make_raster(values=None, georef=None, dtype=np.float32, units=None):
    if georef is None:
        georef = _make_georef()
    if values is None:
        rng = np.random.default_rng(42)
        values = rng.uniform(0.0, 100.0, size=(georef.height, georef.width)).astype(dtype)
    valid = np.ones((georef.height, georef.width), dtype=np.bool_)
    return Raster(values=values, georef=georef, valid=valid, units=units)


# ---------------------------------------------------------------------------
# reclassify_values
# ---------------------------------------------------------------------------

class TestReclassifyValues:
    def test_basic(self):
        georef = _make_georef(width=3, height=2)
        values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        valid = np.ones((2, 3), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid, units="category")
        result = reclassify_values(raster, {1: 10, 5: 50}, default=0)
        assert result.values[0, 0] == 10
        assert result.values[1, 1] == 50
        assert result.values[0, 1] == 0  # default for unmatched
        assert np.all(result.valid)

    def test_unmatched_invalid_by_default(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[1, 2], [3, 4]], dtype=np.int32)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = reclassify_values(raster, {1: 100})
        assert result.values[0, 0] == 100
        assert not result.valid[0, 1]  # unmatched invalid

    def test_preserves_input_invalid(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[1, 2], [3, 4]], dtype=np.int32)
        valid = np.array([[True, True], [False, True]])
        raster = Raster(values=values, georef=georef, valid=valid)
        result = reclassify_values(raster, {1: 10, 2: 20, 3: 30, 4: 40})
        assert not result.valid[1, 0]

    def test_float_mapping(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = reclassify_values(raster, {1.5: 10.5, 3.0: 30.0}, default=-1.0)
        assert result.values[0, 0] == 10.5
        assert result.values[1, 0] == 30.0
        assert result.values[0, 1] == -1.0

    def test_dtype_inference(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = reclassify_values(raster, {1: 10, 2: 20, 3: 30, 4: 40})
        assert result.dtype == np.dtype(np.uint8)

    @pytest.mark.parametrize("as_expression", [False, True])
    def test_python_uint64_output_is_exact(self, as_expression):
        georef = _make_georef(width=2, height=1)
        raster = Raster(
            values=np.array([[1, 2]], dtype=np.uint8), georef=georef,
            valid=np.ones((1, 2), dtype=np.bool_),
        )
        expected = 2**63 + 17
        operand = raster.expression() if as_expression else raster
        classified = reclassify_values(
            operand, {1: expected}, default=0,
        )
        result = compute(classified) if as_expression else classified
        assert result.dtype == np.dtype(np.uint64)
        assert int(result.values[0, 0]) == expected

    @pytest.mark.parametrize("as_expression", [False, True])
    def test_incompatible_signed_uint64_outputs_raise(self, as_expression):
        raster = _make_raster(
            values=np.array([[1, 2]], dtype=np.uint8),
            georef=_make_georef(width=2, height=1),
        )
        operand = raster.expression() if as_expression else raster
        with pytest.raises(MapAlgebraDTypeError) as error:
            reclassify_values(
                operand, {1: -1, 2: 2**63 + 1}, default="invalidate",
            )
        assert error.value.code == "map_algebra_no_exact_promotion"

    def test_typed_float32_outputs_remain_float32(self):
        raster = _make_raster(
            values=np.array([[1, 2]], dtype=np.uint8),
            georef=_make_georef(width=2, height=1),
        )
        eager = reclassify_values(
            raster, {1: np.float32(0.25)}, default=np.float32(0.5),
        )
        expression = reclassify_values(
            raster.expression(),
            {1: np.float32(0.25)},
            default=np.float32(0.5),
        )
        assert eager.dtype == expression.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(compute(expression).values, eager.values)

    @pytest.mark.parametrize(
        ("class_value", "expected_dtype"),
        [(0.5, np.float32), (0.1, np.float64)],
    )
    def test_python_float_uses_smallest_exact_supported_dtype(
        self, class_value, expected_dtype,
    ):
        raster = _make_raster(
            values=np.array([[1]], dtype=np.uint8),
            georef=_make_georef(width=1, height=1),
        )
        eager = reclassify_values(raster, {1: class_value})
        expression = reclassify_values(raster.expression(), {1: class_value})
        assert eager.dtype == expression.dtype == np.dtype(expected_dtype)

    def test_uint64_output_is_exact(self):
        georef = _make_georef(width=1, height=1)
        raster = Raster(
            values=np.array([[1]], dtype=np.uint64), georef=georef,
            valid=np.ones((1, 1), dtype=np.bool_),
        )
        expected = np.uint64(2**63 + 1)
        result = reclassify_values(raster, {1: expected})
        assert result.values[0, 0] == expected

    def test_preserve_unmatched(self):
        georef = _make_georef(width=2, height=1)
        raster = Raster(
            values=np.array([[1, 2]], dtype=np.int16), georef=georef,
            valid=np.ones((1, 2), dtype=np.bool_),
        )
        result = reclassify_values(raster, {1: 10}, default="preserve")
        np.testing.assert_array_equal(result.values, [[10, 2]])
        assert np.all(result.valid)

    def test_preserve_includes_complete_source_dtype(self):
        georef = _make_georef(width=2, height=1)
        raster = Raster(
            values=np.array([[1, -300]], dtype=np.int16), georef=georef,
            valid=np.ones((1, 2), dtype=np.bool_),
        )
        eager = reclassify_values(raster, {1: 10}, default="preserve")
        expression = reclassify_values(
            raster.expression(), {1: 10}, default="preserve",
        )
        assert eager.dtype == expression.dtype == np.dtype(np.int16)
        np.testing.assert_array_equal(eager.values, [[10, -300]])
        np.testing.assert_array_equal(compute(expression).values, eager.values)


# ---------------------------------------------------------------------------
# reclassify_ranges
# ---------------------------------------------------------------------------

class TestReclassifyRanges:
    def test_ranges(self):
        georef = _make_georef(width=4, height=1)
        values = np.array([[0, 25, 50, 75]], dtype=np.float32)
        valid = np.ones((1, 4), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = reclassify_ranges(raster, [
            (0, 30, 1),
            (30, 70, 2),
            (70, 100, 3),
        ])
        assert result.values[0, 0] == 1
        assert result.values[0, 1] == 1
        assert result.values[0, 2] == 2
        assert result.values[0, 3] == 3

    def test_default_outside_ranges(self):
        georef = _make_georef(width=3, height=1)
        values = np.array([[0, 50, 200]], dtype=np.float32)
        valid = np.ones((1, 3), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = reclassify_ranges(raster, [(10, 100, 1)], default=99)
        assert result.values[0, 0] == 99
        assert result.values[0, 1] == 1
        assert result.values[0, 2] == 99

    def test_ranges_invalid_unmatched_without_default(self):
        georef = _make_georef(width=3, height=1)
        values = np.array([[0, 50, 200]], dtype=np.float32)
        valid = np.ones((1, 3), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = reclassify_ranges(raster, [(10, 100, 1)])
        assert not result.valid[0, 0]
        assert not result.valid[0, 2]
        assert result.valid[0, 1]

    @pytest.mark.parametrize("as_expression", [False, True])
    def test_exact_uint64_output_dtype(self, as_expression):
        raster = _make_raster(
            values=np.array([[0, 5]], dtype=np.uint8),
            georef=_make_georef(width=2, height=1),
        )
        expected = 2**63 + 33
        operand = raster.expression() if as_expression else raster
        classified = reclassify_ranges(
            operand, [(0, 1, expected)], default=0,
        )
        result = compute(classified) if as_expression else classified
        assert result.dtype == np.dtype(np.uint64)
        assert int(result.values[0, 0]) == expected


# ---------------------------------------------------------------------------
# digitize
# ---------------------------------------------------------------------------

class TestDigitize:
    def test_basic(self):
        georef = _make_georef(width=4, height=1)
        values = np.array([[0, 3, 7, 10]], dtype=np.float32)
        valid = np.ones((1, 4), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = digitize(raster, [2, 5, 8])
        assert result.values[0, 0] == 0
        assert result.values[0, 1] == 1
        assert result.values[0, 2] == 2
        assert result.values[0, 3] == 3

    def test_preserves_invalid(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[1, 2], [3, 4]], dtype=np.float32)
        valid = np.array([[True, False], [True, True]])
        raster = Raster(values=values, georef=georef, valid=valid)
        result = digitize(raster, [2, 4])
        assert not result.valid[0, 1]

    def test_no_units(self):
        georef = _make_georef(width=2, height=2)
        values = np.ones((2, 2), dtype=np.float32)
        raster = Raster(values=values, georef=georef, valid=np.ones((2, 2), dtype=np.bool_), units="metres")
        result = digitize(raster, [0.5, 1.5])
        assert result.units is None


class TestOneHot:
    def test_eager(self):
        georef = _make_georef(width=3, height=1)
        raster = Raster(
            values=np.array([[1, 2, 1]], dtype=np.uint8), georef=georef,
            valid=np.array([[True, True, False]]),
        )
        class_one, class_two = one_hot(raster, [1, 2])
        np.testing.assert_array_equal(class_one.values, [[True, False, True]])
        np.testing.assert_array_equal(class_two.values, [[False, True, False]])
        np.testing.assert_array_equal(class_one.valid, raster.valid)

    def test_expression(self):
        raster = _make_raster(
            values=np.array([[1, 2], [2, 1]], dtype=np.uint8),
            georef=_make_georef(width=2, height=2),
        )
        expressions = one_hot(raster.expression(), [1, 2])
        assert all(isinstance(item, RasterExpression) for item in expressions)
        np.testing.assert_array_equal(compute(expressions[0]).values, [[True, False], [False, True]])


# ---------------------------------------------------------------------------
# normalize_minmax
# ---------------------------------------------------------------------------

class TestNormalizeMinmax:
    @pytest.mark.parametrize(
        ("source_dtype", "minimum", "maximum", "expected_dtype"),
        [
            (np.float32, None, None, np.float32),
            (np.int16, 0.0, 20.0, np.float32),
            (np.int32, 0.0, 20.0, np.float64),
            (np.float64, 0.0, 20.0, np.float64),
            (np.int64, 0.0, 20.0, np.float64),
            (np.float32, np.float64(0.0), 20.0, np.float64),
            (np.float32, 0.0, 1e40, np.float64),
        ],
    )
    def test_shared_precision_policy(
        self, source_dtype, minimum, maximum, expected_dtype,
    ):
        raster = _make_raster(
            values=np.array([[0, 5], [10, 15]], dtype=source_dtype),
            georef=_make_georef(width=2, height=2),
        )
        kwargs = {"minimum": minimum, "maximum": maximum}
        eager = normalize_minmax(raster, **kwargs)
        expression = normalize_minmax(raster.expression(), **kwargs)
        assert eager.dtype == expression.dtype == np.dtype(expected_dtype)
        assert compute(expression).dtype == eager.dtype

    def test_basic(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[0, 5], [10, 15]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = normalize_minmax(raster)
        assert result.values[0, 0] == 0.0
        assert result.values[1, 1] == 1.0
        assert result.values[0, 1] == pytest.approx(1.0 / 3.0, abs=1e-10)

    def test_large_adjacent_int32_values_do_not_collapse(self):
        base = 2**30
        raster = _make_raster(
            values=np.array([[base, base + 1]], dtype=np.int32),
            georef=_make_georef(width=2, height=1),
        )
        result = normalize_minmax(raster)
        assert result.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result.values, [[0.0, 1.0]])
        assert result.all_valid

    def test_explicit_range(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[0, 5], [10, 15]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = normalize_minmax(raster, minimum=0.0, maximum=20.0)
        assert result.values[1, 1] == 0.75

    def test_all_invalid(self):
        georef = _make_georef(width=2, height=2)
        values = np.ones((2, 2), dtype=np.float32)
        valid = np.zeros((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = normalize_minmax(raster)
        assert result.dtype == np.dtype(np.float32)
        assert not np.any(result.valid)
        assert np.all(np.isnan(result.values[~result.valid]))

    def test_constant_values(self):
        georef = _make_georef(width=2, height=2)
        values = np.full((2, 2), 5.0, dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = normalize_minmax(raster)
        assert not np.any(result.valid)

    def test_no_units(self):
        georef = _make_georef(width=2, height=2)
        raster = _make_raster(georef=georef, units="metres")
        result = normalize_minmax(raster)
        assert result.units is None

    def test_only_valid_data_used(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[999, 1], [2, 3]], dtype=np.float32)
        valid = np.array([[False, True], [True, True]])
        raster = Raster(values=values, georef=georef, valid=valid)
        result = normalize_minmax(raster)
        assert not result.valid[0, 0]
        assert result.values[1, 1] == 1.0

    def test_all_invalid_is_dimensionless(self):
        raster = _make_raster(units="metres")
        raster = Raster(raster.values, raster.georef, np.zeros(raster.shape, dtype=np.bool_), units="metres")
        assert normalize_minmax(raster).units is None

    def test_expression(self):
        raster = _make_raster()
        expression = normalize_minmax(raster.expression(), minimum=0.0, maximum=100.0)
        assert isinstance(expression, RasterExpression)
        np.testing.assert_allclose(compute(expression).values, raster.values / 100.0)


# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------

class TestStandardize:
    @pytest.mark.parametrize(
        ("source_dtype", "mean", "std", "expected_dtype"),
        [
            (np.float32, None, None, np.float32),
            (np.int16, 0.0, 1.0, np.float32),
            (np.int32, 0.0, 1.0, np.float64),
            (np.float64, 0.0, 1.0, np.float64),
            (np.uint64, 0.0, 1.0, np.float64),
            (np.float32, np.float64(0.0), 1.0, np.float64),
        ],
    )
    def test_shared_precision_policy(
        self, source_dtype, mean, std, expected_dtype,
    ):
        raster = _make_raster(
            values=np.array([[1, 2], [3, 4]], dtype=source_dtype),
            georef=_make_georef(width=2, height=2),
        )
        eager = standardize(raster, mean=mean, std=std)
        expression = standardize(raster.expression(), mean=mean, std=std)
        assert eager.dtype == expression.dtype == np.dtype(expected_dtype)
        assert compute(expression).dtype == eager.dtype

    def test_basic(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[1, 2], [3, 4]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = standardize(raster)
        expected = (values - 2.5) / np.sqrt(1.25)
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_large_adjacent_int32_values_remain_distinct(self):
        base = 2**30
        raster = _make_raster(
            values=np.array([[base, base + 1]], dtype=np.int32),
            georef=_make_georef(width=2, height=1),
        )
        result = standardize(raster)
        assert result.dtype == np.dtype(np.float64)
        assert result.values[0, 0] < result.values[0, 1]
        assert result.all_valid

    def test_explicit_stats(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[1, 2], [3, 4]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = standardize(raster, mean=0.0, std=1.0)
        np.testing.assert_array_equal(result.values, values)

    def test_zero_std(self):
        georef = _make_georef(width=2, height=2)
        values = np.full((2, 2), 5.0, dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = standardize(raster)
        assert not np.any(result.valid)

    def test_all_invalid(self):
        georef = _make_georef(width=2, height=2)
        values = np.ones((2, 2), dtype=np.float32)
        valid = np.zeros((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        result = standardize(raster)
        assert not np.any(result.valid)

    def test_no_units(self):
        georef = _make_georef(width=2, height=2)
        raster = _make_raster(georef=georef, units="metres")
        result = standardize(raster)
        assert result.units is None

    def test_all_invalid_is_dimensionless(self):
        raster = _make_raster(units="metres")
        raster = Raster(raster.values, raster.georef, np.zeros(raster.shape, dtype=np.bool_), units="metres")
        assert standardize(raster).units is None


# ---------------------------------------------------------------------------
# Coordinate rasters
# ---------------------------------------------------------------------------

class TestCoordinateRasters:
    def test_row_indices(self):
        georef = _make_georef(width=4, height=3)
        expression = row_indices(georef)
        assert isinstance(expression, RasterExpression)
        result = compute(expression)
        assert result.shape == (3, 4)
        assert result.units == "pixels"
        assert np.all(result.valid)
        np.testing.assert_array_equal(result.values[:, 0], np.array([0, 1, 2], dtype=np.float64))
        np.testing.assert_array_equal(result.values[0, :], np.zeros(4, dtype=np.float64))

    def test_column_indices(self):
        georef = _make_georef(width=4, height=3)
        result = compute(column_indices(georef))
        assert result.shape == (3, 4)
        assert result.units == "pixels"
        np.testing.assert_array_equal(result.values[0, :], np.array([0, 1, 2, 3], dtype=np.float64))
        np.testing.assert_array_equal(result.values[:, 0], np.zeros(3, dtype=np.float64))

    def test_projected_x(self):
        georef = _make_georef(width=3, height=2, pixel_x=20.0, pixel_y=-20.0)
        result = compute(projected_x(georef, anchor="center"))
        assert result.shape == (2, 3)
        affine = georef.affine_transform
        expected_x0 = affine[0] + 0.5 * affine[1]
        assert result.values[0, 0] == pytest.approx(expected_x0)

    def test_projected_y(self):
        georef = _make_georef(width=3, height=2, pixel_x=20.0, pixel_y=-20.0)
        result = compute(projected_y(georef, anchor="center"))
        assert result.shape == (2, 3)
        affine = georef.affine_transform
        expected_y0 = affine[3] + 0.5 * affine[5]
        assert result.values[0, 0] == pytest.approx(expected_y0)

    def test_anchor_corner(self):
        georef = _make_georef(width=3, height=2, pixel_x=20.0, pixel_y=-20.0)
        result = compute(projected_x(georef, anchor="corner"))
        affine = georef.affine_transform
        assert result.values[0, 0] == affine[0]

    def test_invalid_anchor(self):
        georef = _make_georef()
        with pytest.raises(MapAlgebraError, match="anchor"):
            projected_x(georef, anchor="invalid")

    def test_projected_units_come_from_crs(self):
        from pyproj import CRS

        crs = CRS.from_epsg(2263)
        georef = GeoReference(
            projection_wkt=crs.to_wkt(),
            projection_proj4=crs.to_proj4(),
            affine_transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            width=1, height=1, pixel_size_x=1.0, pixel_size_y=-1.0, nodata=None,
        )
        assert projected_x(georef).units == "US survey foot"

    def test_longitude_latitude(self):
        georef = _make_georef(width=2, height=2)
        result = compute(longitude(georef))
        assert result.shape == (2, 2)
        assert result.units == "degrees"
        assert np.all(result.valid)

        result2 = compute(latitude(georef))
        assert result2.shape == (2, 2)
        assert result2.units == "degrees"
        assert np.all(result2.valid)

    def test_longitude_latitude_values_south_pole(self):
        from lunarscout.georeference import GeoReference
        MOON_WKT_SOUTH = (
            'PROJCS["Moon_South_Pole_Stereographic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Reference_Meridian",0],'
            'UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Polar_Stereographic"],'
            'PARAMETER["latitude_of_origin",-90],'
            'PARAMETER["central_meridian",0],'
            'PARAMETER["scale_factor",1],'
            'PARAMETER["false_easting",0],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1]]'
        )
        georef = GeoReference(
            projection_wkt=MOON_WKT_SOUTH,
            projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs",
            affine_transform=(0.0, 20.0, 0.0, 0.0, 0.0, -20.0),
            width=3, height=3, pixel_size_x=20.0, pixel_size_y=-20.0, nodata=None,
        )
        lon = compute(longitude(georef))
        lat = compute(latitude(georef))
        assert lon.shape == (3, 3)
        assert lat.shape == (3, 3)
        assert lat.units == "degrees"
        assert lon.units == "degrees"
        assert np.all(lat.values <= -89.0)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class TestValidationHelpers:
    def test_as_raster_operand_raster(self):
        raster = _make_raster()
        result = _as_raster_operand(raster, argument="test")
        assert result is raster

    def test_as_raster_operand_scalar(self):
        result = _as_raster_operand(42, argument="test")
        assert result == 42

    def test_as_raster_operand_reject_array(self):
        with pytest.raises(MapAlgebraError, match="Raster or a real numeric"):
            _as_raster_operand(np.array([1, 2, 3]), argument="test")

    def test_as_expression_operand_raster(self):
        raster = _make_raster()
        result = _as_expression_operand(raster, argument="test")
        assert isinstance(result, RasterExpression)

    def test_as_expression_operand_expression(self):
        raster = _make_raster()
        expr = raster.expression()
        result = _as_expression_operand(expr, argument="test")
        assert result is expr

    def test_as_expression_operand_scalar(self):
        result = _as_expression_operand(3.14, argument="test")
        assert result == 3.14

    def test_as_expression_operand_reject_array(self):
        with pytest.raises(MapAlgebraError, match="RasterExpression, Raster, or a real"):
            _as_expression_operand(np.array([1, 2, 3]), argument="test")

    def test_as_expression_operand_grid_hint(self):
        raster = _make_raster()
        shifted = _make_georef()
        shifted = GeoReference(
            projection_wkt=shifted.projection_wkt,
            projection_proj4=shifted.projection_proj4,
            affine_transform=(shifted.affine_transform[0] + 1.0, *shifted.affine_transform[1:]),
            width=shifted.width, height=shifted.height,
            pixel_size_x=shifted.pixel_size_x, pixel_size_y=shifted.pixel_size_y,
            nodata=shifted.nodata,
        )
        with pytest.raises(MapAlgebraError):
            _as_expression_operand(raster, grid_hint=shifted)


# ---------------------------------------------------------------------------
# RasterExpression.describe()
# ---------------------------------------------------------------------------

class TestRasterExpressionDescribe:
    def test_describe(self):
        raster = _make_raster()
        expr = raster.expression()
        desc = expr.describe()
        assert "RasterExpression" in desc
        assert "constant" in desc

    def test_describe_with_units(self):
        raster = _make_raster(units="metres")
        expr = raster.expression()
        desc = expr.describe()
        assert "metres" in desc


# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

class TestOperationRegistry:
    @pytest.mark.parametrize(
        "operation_id",
        ["local.normalize_minmax", "local.standardize"],
    )
    def test_normalization_precision_contract(self, operation_id):
        description = describe_operation(operation_id)
        assert description["version"] == 2
        assert "FP32" in description["output_dtype_rule"]
        assert "typed FP64" in description["output_dtype_rule"]

    @pytest.mark.parametrize(
        "operation_id",
        ["local.reclassify_values", "local.reclassify_ranges"],
    )
    def test_reclassification_exact_dtype_contract(self, operation_id):
        description = describe_operation(operation_id)
        assert description["version"] == 2
        assert "smallest supported dtype" in description["output_dtype_rule"]
        assert "source dtype" in description["output_dtype_rule"]

    def test_describe_builtin(self):
        desc = describe_operation("local.add")
        assert desc["id"] == "local.add"
        assert desc["category"] == "local"
        assert desc["version"] == 2
        assert [item["name"] for item in desc["parameters"]] == [
            "overflow", "numeric_errors",
        ]

    def test_describe_numeric_power_and_cast_policies(self):
        power_desc = describe_operation("local.power")
        assert power_desc["version"] == 2
        assert [item["name"] for item in power_desc["parameters"]] == [
            "overflow", "numeric_errors",
        ]

        cast_desc = describe_operation("local.cast")
        assert cast_desc["version"] == 2
        assert [item["name"] for item in cast_desc["parameters"]] == [
            "casting", "overflow",
        ]

    def test_list_operations(self):
        ops = list_operations()
        assert len(ops) > 20
        ops_local = list_operations(category="local")
        assert all(o["category"] == "local" for o in ops_local)

    def test_unknown_operation(self):
        with pytest.raises(MapAlgebraError, match="Unknown"):
            describe_operation("nonexistent.op")

    def test_registry_is_not_publicly_mutable(self):
        import lunarscout.map_algebra as ma

        assert not hasattr(ma, "register")


# ---------------------------------------------------------------------------
# Hex-float JSON encoding
# ---------------------------------------------------------------------------

class TestHexFloatJson:
    def test_expression_json_uses_typed_scalars(self):
        georef = _make_georef(width=2, height=2)
        values = np.array([[1, 2], [3, 4]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=np.bool_)
        raster = Raster(values=values, georef=georef, valid=valid)
        expr = raster.expression()
        expr2 = expr + 1.5
        js = expr2.to_json()
        assert "hex" in js
        assert '"type":"float"' in js

    def test_canonical_json_is_stable_across_node_ids(self):
        raster = _make_raster()
        first = (raster.expression() + 1.5).to_json()
        second = (raster.expression() + 1.5).to_json()
        assert first == second
        assert first == (raster.expression() + 1.5).to_canonical_json()

    def test_unrepresentable_parameter_is_rejected(self):
        from lunarscout.map_algebra._normalization import normalize_canonical

        with pytest.raises(MapAlgebraError, match="cannot be represented canonically"):
            normalize_canonical(object())

    def test_temporal_bool_is_not_encoded_as_integer(self):
        from lunarscout.map_algebra._temporal_model import _json_operand_tp

        assert _json_operand_tp(True) == {"type": "bool", "value": True}

    def test_temporal_json_uses_typed_scalars(self):
        from lunarscout.map_algebra._temporal_model import TemporalRaster, _temporal_constant
        georef = _make_georef(width=2, height=2)
        times = np.array(["2027-01-01", "2027-01-02"], dtype="datetime64[D]")
        values = np.ones((2, 2, 2), dtype=np.float32)
        valid = np.ones((2, 2, 2), dtype=np.bool_)
        tr = TemporalRaster(values=values, times=times, georef=georef, valid=valid)
        expr = _temporal_constant(tr)
        js = expr.to_json()
        assert '"domain":"temporal"' in js
