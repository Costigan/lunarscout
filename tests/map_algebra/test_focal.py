from __future__ import annotations

import numpy as np
import pytest

from lunarscout.errors import MapAlgebraError
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    Raster,
    closing,
    convolve,
    dilate,
    describe_operation,
    erode,
    focal_count,
    focal_max,
    focal_mean,
    focal_median,
    focal_min,
    focal_range,
    focal_std,
    focal_sum,
    majority,
    opening,
    raster as ma_raster,
)
from tests.map_algebra.conftest import MOON_WKT, MOON_PROJ4


def _georef(h: int, w: int) -> GeoReference:
    return GeoReference(
        projection_wkt=MOON_WKT, projection_proj4=MOON_PROJ4,
        affine_transform=(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
        width=w, height=h, pixel_size_x=20.0, pixel_size_y=-20.0, nodata=None,
    )


def _make(values, units=None):
    return ma_raster(values, _georef(values.shape[0], values.shape[1]), units=units)


class TestFocalSum:
    def test_3x3(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = focal_sum(r, size=3, edge="nearest")
        assert result.values[2, 2] == 9.0

    def test_uint8_sum_wider_dtype(self):
        r = _make(np.full((3, 3), 255, dtype=np.uint8))
        result = focal_sum(r, size=3, edge="nearest")
        assert result.dtype == np.dtype(np.int64)
        assert result.values[1, 1] == 2295

    def test_ignore_invalid_mean(self):
        v = np.ones((3, 3), dtype=np.float32)
        r = ma_raster(v, _georef(3, 3), valid=np.ones((3, 3), dtype=np.bool_))
        r.valid[1, 1] = False
        result = focal_mean(r, size=3, valid_neighbor="ignore_invalid", edge="nearest")
        assert result.values[1, 1] == 1.0

    def test_propagate_center(self):
        v = np.ones((3, 3), dtype=np.float32)
        r = ma_raster(v, _georef(3, 3), valid=np.ones((3, 3), dtype=np.bool_))
        r.valid[1, 1] = False
        result = focal_sum(r, size=3, valid_neighbor="propagate_center", edge="nearest")
        assert not result.valid[1, 1]
        assert result.valid[0, 0]

    def test_invalid_edge(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = focal_sum(r, size=3, edge="invalid")
        assert not result.valid[0, 0]
        assert result.valid[2, 2]

    @pytest.mark.parametrize(
        "operation",
        [
            focal_sum, focal_mean, focal_min, focal_max, focal_range,
            focal_std, focal_median,
        ],
    )
    def test_min_valid_count_controls_ignore_invalid(self, operation):
        values = np.arange(9, dtype=np.float32).reshape(3, 3)
        valid = np.array(
            [[True, True, False], [True, True, False], [True, False, False]],
            dtype=np.bool_,
        )
        source = ma_raster(values, _georef(3, 3), valid=valid)
        accepted = operation(
            source, size=3, edge="nearest", valid_neighbor="ignore_invalid",
            min_valid_count=5,
        )
        rejected = operation(
            source, size=3, edge="nearest", valid_neighbor="ignore_invalid",
            min_valid_count=6,
        )
        assert accepted.valid[1, 1]
        assert not rejected.valid[1, 1]


class TestFocalMean:
    def test_3x3(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = focal_mean(r, size=3, edge="nearest")
        assert abs(result.values[2, 2] - 1.0) < 1e-5

    def test_float_output_dtype(self):
        r = _make(np.ones((5, 5), dtype=np.int32))
        result = focal_mean(r, size=3, edge="nearest")
        assert result.dtype in (np.dtype(np.float32), np.dtype(np.float64))


class TestFocalMinMax:
    def test_min(self):
        v = np.array([[5, 4, 3], [6, 1, 2], [7, 8, 9]], dtype=np.float32)
        result = focal_min(_make(v), size=3, edge="nearest")
        assert result.values[1, 1] == 1.0

    def test_max(self):
        v = np.array([[5, 4, 3], [6, 1, 2], [7, 8, 9]], dtype=np.float32)
        result = focal_max(_make(v), size=3, edge="nearest")
        assert result.values[1, 1] == 9.0

    def test_range(self):
        v = np.array([[5, 4, 3], [6, 1, 2], [7, 8, 9]], dtype=np.float32)
        result = focal_range(_make(v), size=3, edge="nearest")
        assert result.values[1, 1] == 8.0


class TestFocalStd:
    def test_flat(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = focal_std(r, size=3, edge="nearest")
        assert result.values[2, 2] == 0.0

    def test_ddof(self):
        r = _make(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32))
        s0 = focal_std(r, size=3, edge="nearest", ddof=0)
        s1 = focal_std(r, size=3, edge="nearest", ddof=1)
        assert s1.values[1, 1] > s0.values[1, 1]


class TestFocalCount:
    def test_all_valid(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = focal_count(r, size=3, edge="nearest")
        assert result.values[2, 2] == 9

    def test_some_invalid(self):
        v = np.ones((3, 3), dtype=np.float32)
        r = ma_raster(v, _georef(3, 3), valid=np.ones((3, 3), dtype=np.bool_))
        r.valid[0, 0] = False
        result = focal_count(r, size=3, valid_neighbor="ignore_invalid", edge="nearest")
        assert result.values[1, 1] == 8

    def test_size_as_tuple(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = focal_count(r, size=(3, 5), edge="nearest")
        assert result.shape == (5, 5)

    def test_explicit_min_valid_count_preserves_default_zero_count_contract(self):
        source = ma_raster(
            np.ones((3, 3), dtype=np.float32), _georef(3, 3),
            valid=np.zeros((3, 3), dtype=np.bool_),
        )
        default = focal_count(
            source, size=3, edge="nearest", valid_neighbor="ignore_invalid",
        )
        thresholded = focal_count(
            source, size=3, edge="nearest", valid_neighbor="ignore_invalid",
            min_valid_count=1,
        )
        assert default.valid[1, 1]
        assert default.values[1, 1] == 0
        assert not thresholded.valid[1, 1]


class TestFocalMedian:
    def test_median(self):
        v = np.arange(25, dtype=np.float32).reshape(5, 5)
        result = focal_median(_make(v), size=3, edge="nearest")
        assert result.values[2, 2] == 12.0

    def test_ignore_invalid_median(self):
        v = np.ones((3, 3), dtype=np.float32) * 5.0
        r = ma_raster(v, _georef(3, 3), valid=np.ones((3, 3), dtype=np.bool_))
        r.valid[1, 1] = False
        result = focal_median(r, size=3, valid_neighbor="ignore_invalid", edge="nearest")
        assert result.values[1, 1] == 5.0


class TestEdgeModes:
    def test_invalid_edges_false(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        for mode in ["invalid", "constant"]:
            result = focal_sum(r, size=3, edge=mode, cval=0.0)  # type: ignore[arg-type]
            assert not result.valid[0, 0]

    def test_nearest_reflect_wrap_valid(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        for mode in ["nearest", "reflect", "wrap"]:
            result = focal_sum(r, size=3, edge=mode)  # type: ignore[arg-type]
            assert result.valid[0, 0]

    def test_unknown_edge_raises(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        with pytest.raises(MapAlgebraError, match="Unknown edge mode"):
            focal_sum(r, size=3, edge="bogus")  # type: ignore[arg-type]

    def test_unknown_valid_neighbor_raises(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        with pytest.raises(MapAlgebraError, match="Unknown valid_neighbor"):
            focal_sum(r, size=3, valid_neighbor="bogus")  # type: ignore[arg-type]


class TestFootprint:
    def test_rectangular_size(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = focal_sum(r, size=(3, 5), edge="nearest")
        assert result.shape == (5, 5)

    def test_binary_footprint(self):
        fp = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_)
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = focal_sum(r, footprint=fp, edge="nearest")
        assert result.values[2, 2] == 5.0

    def test_even_size_rejected(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        with pytest.raises(MapAlgebraError):
            focal_sum(r, size=2)

    @pytest.mark.parametrize("as_expression", [False, True])
    def test_all_false_footprint_rejected(self, as_expression):
        source = _make(np.ones((3, 3), dtype=np.float32))
        operand = source.expression() if as_expression else source
        with pytest.raises(MapAlgebraError) as error:
            focal_sum(
                operand, footprint=np.zeros((3, 3), dtype=np.bool_),
                valid_neighbor="ignore_invalid", min_valid_count=1,
            )
        assert error.value.code == "map_algebra_invalid_footprint"


class TestConvolution:
    def test_identity(self):
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        r = _make(np.arange(9, dtype=np.float32).reshape(3, 3))
        result = convolve(r, kernel, edge="nearest")
        np.testing.assert_array_equal(result.values[1, 1], 4.0)

    def test_box_blur(self):
        kernel = np.ones((3, 3), dtype=np.float64) / 9.0
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = convolve(r, kernel, edge="nearest")
        assert abs(result.values[2, 2] - 1.0) < 1e-5

    def test_normalize(self):
        kernel = np.ones((3, 3), dtype=np.float64)
        r = _make(np.ones((5, 5), dtype=np.float32))
        result = convolve(r, kernel, normalize=True, edge="nearest")
        assert abs(result.values[2, 2] - 1.0) < 1e-5

    def test_nonfinite_kernel_rejected(self):
        kernel = np.array([[0, 0, 0], [0, np.nan, 0], [0, 0, 0]], dtype=np.float64)
        r = _make(np.ones((3, 3), dtype=np.float32))
        with pytest.raises(MapAlgebraError):
            convolve(r, kernel)

    @pytest.mark.parametrize("as_expression", [False, True])
    def test_complex_kernel_rejected_consistently(self, as_expression):
        kernel = np.ones((3, 3), dtype=np.complex64)
        source = _make(np.ones((3, 3), dtype=np.float32))
        operand = source.expression() if as_expression else source
        with pytest.raises(MapAlgebraError) as error:
            convolve(operand, kernel)
        assert error.value.code == "map_algebra_invalid_kernel"

    def test_one_by_one_kernel(self):
        kernel = np.array([[2.0]], dtype=np.float64)
        r = _make(np.ones((3, 3), dtype=np.float32))
        result = convolve(r, kernel, edge="invalid")
        np.testing.assert_array_equal(result.values, np.full((3, 3), 2.0, dtype=np.float32))

    def test_min_valid_count(self):
        kernel = np.ones((3, 3), dtype=np.float64)
        valid = np.ones((3, 3), dtype=np.bool_)
        valid.flat[:5] = False
        source = ma_raster(
            np.ones((3, 3), dtype=np.float32), _georef(3, 3), valid=valid,
        )
        result = convolve(
            source, kernel, edge="nearest", valid_neighbor="ignore_invalid",
            min_valid_count=5,
        )
        assert not result.valid[1, 1]


class TestMinValidCountValidation:
    @pytest.mark.parametrize("value", [0, 10, 1.5, True])
    def test_invalid_count_is_structured(self, value):
        source = _make(np.ones((3, 3), dtype=np.float32))
        with pytest.raises(MapAlgebraError) as error:
            focal_mean(
                source, size=3, edge="nearest",
                valid_neighbor="ignore_invalid", min_valid_count=value,
            )
        assert error.value.code == "map_algebra_invalid_min_valid_count"

    def test_count_requires_ignore_invalid_policy(self):
        source = _make(np.ones((3, 3), dtype=np.float32))
        with pytest.raises(MapAlgebraError) as error:
            focal_mean(
                source, size=3, edge="nearest",
                valid_neighbor="require_all", min_valid_count=3,
            )
        assert error.value.code == "map_algebra_invalid_min_valid_count"

    def test_cross_footprint_uses_active_cell_count(self):
        source = _make(np.ones((3, 3), dtype=np.float32))
        footprint = np.array(
            [[False, True, False], [True, True, True], [False, True, False]],
            dtype=np.bool_,
        )
        with pytest.raises(MapAlgebraError) as error:
            focal_mean(
                source, footprint=footprint, edge="nearest",
                valid_neighbor="ignore_invalid", min_valid_count=6,
            )
        assert error.value.code == "map_algebra_invalid_min_valid_count"
        assert error.value.details["maximum"] == 5

    def test_registry_metadata_matches_public_parameter(self):
        description = describe_operation("focal.mean")
        assert description["version"] == 2
        assert description["file_backed_available"] is False
        assert [item["name"] for item in description["parameters"]] == [
            "size", "footprint", "edge", "valid_neighbor",
            "min_valid_count", "cval",
        ]


class TestMorphology:
    def test_dilate(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        result = dilate(_make(v), size=3)
        assert np.sum(result.values) == 9

    def test_erode(self):
        v = np.ones((5, 5), dtype=np.bool_)
        result = erode(_make(v), size=3)
        assert not result.values[0, 0]
        assert result.values[2, 2]

    def test_invalid_cell_does_not_dilate(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        r = ma_raster(v, _georef(5, 5), valid=np.ones((5, 5), dtype=np.bool_))
        r.valid[2, 2] = False
        result = dilate(r, size=3)
        assert not result.values[2, 2]
        assert not result.values[1, 2]

    def test_majority_true(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[1:4, 1:4] = True
        result = majority(_make(v), size=3)
        assert result.values[2, 2]

    def test_majority_false(self):
        v = np.zeros((5, 5), dtype=np.bool_)
        v[2, 2] = True
        result = majority(_make(v), size=3)
        assert not result.values[2, 2]

    def test_non_boolean_rejected(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        with pytest.raises(MapAlgebraError):
            dilate(r, size=3)


class TestExpressionDispatch:
    def test_focal_sum_accepts_expression(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        expr = r.expression()
        result_expr = focal_sum(expr, size=3, edge="nearest")
        from lunarscout.map_algebra import RasterExpression
        assert isinstance(result_expr, RasterExpression)
        assert result_expr.operation_id == "focal.sum"

    def test_convolve_accepts_expression(self):
        r = _make(np.ones((5, 5), dtype=np.float32))
        kernel = np.ones((3, 3), dtype=np.float64)
        expr = r.expression()
        result_expr = convolve(expr, kernel, edge="nearest")
        from lunarscout.map_algebra import RasterExpression
        assert isinstance(result_expr, RasterExpression)

    def test_morphology_accepts_expression(self):
        r = _make(np.ones((5, 5), dtype=np.bool_))
        expr = r.expression()
        result_expr = dilate(expr, size=3)
        from lunarscout.map_algebra import RasterExpression
        assert isinstance(result_expr, RasterExpression)

    def test_min_valid_count_changes_identity_and_validates_without_execution(self):
        expression = _make(np.ones((3, 3), dtype=np.float32)).expression()
        three = focal_mean(
            expression, size=3, valid_neighbor="ignore_invalid",
            min_valid_count=3,
        )
        four = focal_mean(
            expression, size=3, valid_neighbor="ignore_invalid",
            min_valid_count=4,
        )
        assert three.scientific_identity() != four.scientific_identity()

        with pytest.raises(MapAlgebraError) as error:
            focal_mean(
                expression, size=3, valid_neighbor="ignore_invalid",
                min_valid_count=10,
            )
        assert error.value.code == "map_algebra_invalid_min_valid_count"
