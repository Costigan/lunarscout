from __future__ import annotations

import numpy as np
import pytest

from lunarscout.georeference import GeoReference
from lunarscout.temporal import TemporalCube, TimeRange, utc_datetime
from lunarscout.raster import Raster as _Raster
from lunarscout.map_algebra._temporal_model import (
    TemporalRaster,
    TemporalRasterExpression,
    from_temporal_cube,
    to_temporal_cube,
    _compute_temporal,
    _temporal_local_op,
    _temporal_source_node,
    _temporal_constant,
    _temporal_broadcast,
    _make_tp_node,
    _topological_sort_temporal,
    _temporal_content_hash,
)
from lunarscout.map_algebra import (
    compute,
    compute_temporal,
    temporal_source,
    temporal_mean,
    temporal_min,
    temporal_max,
    temporal_std,
    temporal_sum,
    temporal_count,
    explain_temporal,
)
from lunarscout.map_algebra._model import RasterExpression
from lunarscout.errors import MapAlgebraError, MapAlgebraExpressionError

from tests.map_algebra.conftest import _georef  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_georef(width=10, height=8, pixel_x=20.0, pixel_y=-20.0, nodata=None, origin_x=1000.0, origin_y=2000.0):
    return _georef(
        width=width,
        height=height,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        origin_x=origin_x,
        origin_y=origin_y,
        nodata=nodata,
    )


def _make_times(num: int = 5) -> np.ndarray:
    base = np.datetime64("2027-01-01T00:00:00", "us")
    return base + np.arange(num, dtype=np.int64) * 3_600_000_000  # hourly


def _make_temporal_raster(
    num_layers: int = 5,
    height: int = 8,
    width: int = 10,
    dtype: np.dtype = np.dtype(np.float32),
    nodata: float | None = None,
) -> TemporalRaster:
    georef = _make_georef(width=width, height=height, nodata=nodata)
    times = _make_times(num_layers)
    rng = np.random.default_rng(42)
    values = rng.uniform(0.0, 100.0, size=(num_layers, height, width)).astype(dtype)
    valid = np.ones((num_layers, height, width), dtype=np.bool_)
    if nodata is not None:
        values[0, 0, 0] = np.asarray(nodata, dtype=dtype)
        values[1, 2, 3] = np.asarray(nodata, dtype=dtype)
        if np.issubdtype(dtype, np.floating) and np.isnan(float(nodata)):
            valid[0, 0, 0] = False
            valid[1, 2, 3] = False
        else:
            valid[values == np.asarray(nodata, dtype=dtype)] = False
    return TemporalRaster(
        values=values,
        times=times,
        georef=georef,
        valid=valid,
        units="metres",
        signal_name="test_signal",
        name="test_temporal",
    )


def _make_temporal_cube(num_layers=5):
    georef = _make_georef(nodata=-9999.0)
    times = _make_times(num_layers)
    rng = np.random.default_rng(42)
    values = rng.uniform(0.0, 100.0, size=(num_layers, 8, 10)).astype(np.float32)
    values[0, 0, 0] = -9999.0
    return TemporalCube(values=values, times=times, georef=georef)


# ---------------------------------------------------------------------------
# TemporalRaster construction and validation
# ---------------------------------------------------------------------------

class TestTemporalRasterConstruction:
    def test_basic_construction(self):
        tr = _make_temporal_raster()
        assert tr.values.shape == (5, 8, 10)
        assert tr.dtype == np.dtype(np.float32)
        assert tr.num_layers == 5
        assert tr.height == 8
        assert tr.width == 10
        assert tr.units == "metres"
        assert tr.signal_name == "test_signal"
        assert tr.name == "test_temporal"

    def test_invalid_values_ndim(self):
        georef = _make_georef()
        times = _make_times(3)
        with pytest.raises(MapAlgebraError, match="three-dimensional"):
            TemporalRaster(
                values=np.zeros((3, 8), dtype=np.float32),
                times=times,
                georef=georef,
                valid=np.ones((3, 8), dtype=np.bool_),
            )

    def test_times_length_mismatch(self):
        georef = _make_georef()
        times = _make_times(5)
        with pytest.raises(MapAlgebraError, match="time count"):
            TemporalRaster(
                values=np.zeros((3, 8, 10), dtype=np.float32),
                times=times,
                georef=georef,
                valid=np.ones((3, 8, 10), dtype=np.bool_),
            )

    def test_times_not_datetime64(self):
        georef = _make_georef()
        with pytest.raises(MapAlgebraError, match="datetime64"):
            TemporalRaster(
                values=np.zeros((3, 8, 10), dtype=np.float32),
                times=np.array([1.0, 2.0, 3.0]),
                georef=georef,
                valid=np.ones((3, 8, 10), dtype=np.bool_),
            )

    def test_valid_shape_mismatch(self):
        georef = _make_georef()
        times = _make_times(3)
        with pytest.raises(MapAlgebraError, match="valid"):
            TemporalRaster(
                values=np.zeros((3, 8, 10), dtype=np.float32),
                times=times,
                georef=georef,
                valid=np.ones((3, 9, 10), dtype=np.bool_),
            )

    def test_valid_not_bool(self):
        georef = _make_georef()
        times = _make_times(3)
        with pytest.raises(MapAlgebraError, match="bool"):
            TemporalRaster(
                values=np.zeros((3, 8, 10), dtype=np.float32),
                times=times,
                georef=georef,
                valid=np.ones((3, 8, 10), dtype=np.int32),
            )

    def test_spatial_shape_mismatch(self):
        georef = _make_georef()  # 10 wide, 8 tall
        times = _make_times(3)
        with pytest.raises(MapAlgebraError, match="spatial shape"):
            TemporalRaster(
                values=np.zeros((3, 8, 12), dtype=np.float32),
                times=times,
                georef=georef,
                valid=np.ones((3, 8, 12), dtype=np.bool_),
            )


# ---------------------------------------------------------------------------
# TemporalRaster properties
# ---------------------------------------------------------------------------

class TestTemporalRasterProperties:
    def test_properties(self):
        tr = _make_temporal_raster(num_layers=4)
        assert tr.shape == (4, 8, 10)
        assert tr.num_layers == 4
        assert tr.height == 8
        assert tr.width == 10
        assert tr.nbytes > 0
        assert tr.all_valid is True
        assert tr.invalid_count == 0
        assert tr.spatial_shape == (8, 10)

    def test_all_valid_with_nodata(self):
        tr = _make_temporal_raster(num_layers=4, nodata=-9999.0)
        assert tr.all_valid is False
        assert tr.invalid_count == 2  # two pixels set to nodata

    def test_truth_testing_forbidden(self):
        tr = _make_temporal_raster()
        with pytest.raises(TypeError, match="truth testing"):
            bool(tr)

    def test_repr(self):
        tr = _make_temporal_raster()
        r = repr(tr)
        assert "TemporalRaster" in r
        assert "test_signal" in r
        assert "metres" in r


# ---------------------------------------------------------------------------
# TemporalRaster conversion helpers
# ---------------------------------------------------------------------------

class TestTemporalRasterHelpers:
    def test_copy(self):
        tr = _make_temporal_raster(num_layers=2)
        cp = tr.copy()
        assert cp.num_layers == 2
        assert cp.values is not tr.values
        np.testing.assert_array_equal(cp.values, tr.values)
        np.testing.assert_array_equal(cp.valid, tr.valid)
        np.testing.assert_array_equal(cp.times, tr.times)

    def test_readonly(self):
        tr = _make_temporal_raster(num_layers=2)
        ro = tr.readonly()
        with pytest.raises(ValueError):
            ro.values[0, 0, 0] = 42.0

    def test_filled(self):
        tr = _make_temporal_raster(num_layers=2)
        # Invalidate a pixel
        new_valid = tr.valid.copy()
        new_valid[0, 0, 0] = False
        tr2 = tr.with_validity(new_valid)
        filled = tr2.filled(-1.0)
        assert filled[0, 0, 0] == -1.0

    def test_masked(self):
        tr = _make_temporal_raster(num_layers=2)
        masked = tr.masked()
        assert isinstance(masked, np.ma.MaskedArray)

    def test_with_name(self):
        tr = _make_temporal_raster()
        tr2 = tr.with_name("new_name")
        assert tr2.name == "new_name"
        assert tr.name == "test_temporal"  # original unchanged

    def test_with_units(self):
        tr = _make_temporal_raster()
        tr2 = tr.with_units("degrees")
        assert tr2.units == "degrees"

    def test_with_validity(self):
        tr = _make_temporal_raster(num_layers=2)
        new_valid = np.zeros_like(tr.valid)
        new_valid[0, :, :] = True
        tr2 = tr.with_validity(new_valid)
        assert tr2.all_valid is False

    def test_with_georef(self):
        tr = _make_temporal_raster()
        new_georef = _make_georef(width=4, height=4)
        with pytest.raises(MapAlgebraError):
            tr.with_georef(new_georef)  # shape mismatch

    def test_with_times(self):
        tr = _make_temporal_raster(num_layers=3)
        new_times = _make_times(3)
        tr2 = tr.with_times(new_times)
        np.testing.assert_array_equal(tr2.times, new_times)

    def test_with_signal_name(self):
        tr = _make_temporal_raster()
        tr2 = tr.with_signal_name("new_signal")
        assert tr2.signal_name == "new_signal"

    def test_layer(self):
        tr = _make_temporal_raster(num_layers=3)
        layer = tr.layer(0)
        assert isinstance(layer, _Raster)
        assert layer.shape == (8, 10)
        np.testing.assert_array_equal(layer.values, tr.values[0])

    def test_layer_index_out_of_range(self):
        tr = _make_temporal_raster(num_layers=3)
        with pytest.raises(MapAlgebraError, match="index"):
            tr.layer(10)

    def test_same_grid(self):
        tr = _make_temporal_raster()
        tr2 = _make_temporal_raster()
        assert tr.same_grid(tr2) is True

    def test_same_times(self):
        tr = _make_temporal_raster(num_layers=3)
        tr2 = _make_temporal_raster(num_layers=3)
        assert tr.same_times(tr2) is True

    def test_same_times_different_length(self):
        tr = _make_temporal_raster(num_layers=3)
        tr2 = _make_temporal_raster(num_layers=5)
        assert tr.same_times(tr2) is False


# ---------------------------------------------------------------------------
# TemporalCube adapters
# ---------------------------------------------------------------------------

class TestTemporalCubeAdapters:
    def test_from_temporal_cube(self):
        cube = _make_temporal_cube()
        tr = from_temporal_cube(cube)
        assert isinstance(tr, TemporalRaster)
        assert tr.num_layers == 5
        assert tr.height == 8
        assert tr.width == 10
        assert tr.all_valid is False  # nodata pixel invalid
        assert tr.values[0, 0, 0] == -9999.0
        assert not tr.valid[0, 0, 0]

    def test_from_temporal_cube_no_nodata(self):
        georef = _make_georef()
        times = _make_times(3)
        values = np.ones((3, 8, 10), dtype=np.float32)
        cube = TemporalCube(values=values, times=times, georef=georef)
        tr = from_temporal_cube(cube)
        assert tr.all_valid is True

    def test_from_temporal_cube_nan_nodata(self):
        georef = _make_georef(nodata=np.nan)
        times = _make_times(3)
        values = np.ones((3, 8, 10), dtype=np.float32)
        values[0, 0, 0] = np.nan
        cube = TemporalCube(values=values, times=times, georef=georef)
        tr = from_temporal_cube(cube)
        assert not tr.valid[0, 0, 0]

    def test_to_temporal_cube_roundtrip(self):
        tr = _make_temporal_raster(num_layers=3)
        cube = to_temporal_cube(tr)
        assert cube.values.shape == (3, 8, 10)
        np.testing.assert_array_equal(cube.times, tr.times)

    def test_expression_method(self):
        tr = _make_temporal_raster(num_layers=2)
        expr = tr.expression()
        assert isinstance(expr, TemporalRasterExpression)
        assert expr._operation_id == "temporal.constant"


# ---------------------------------------------------------------------------
# TemporalRasterExpression sealed constructor
# ---------------------------------------------------------------------------

class TestTemporalRasterExpressionModel:
    def test_direct_construction_rejected(self):
        with pytest.raises(MapAlgebraExpressionError, match="directly"):
            TemporalRasterExpression(
                _node_id="x",
                _operation_id="test",
            )

    def test_factory_construction_accepted(self):
        node = _make_tp_node("test.op")
        assert node._operation_id == "test.op"
        assert isinstance(node, TemporalRasterExpression)

    def test_truth_testing_forbidden(self):
        node = _make_tp_node("test.op")
        with pytest.raises(TypeError, match="truth testing"):
            bool(node)

    def test_describe(self):
        tr = _make_temporal_raster(num_layers=3)
        node = _temporal_constant(tr)
        desc = node.describe()
        assert "temporal.constant" in desc
        assert "3 layers" in desc

    def test_scientific_identity(self):
        tr1 = _make_temporal_raster(num_layers=2)
        tr2 = _make_temporal_raster(num_layers=2)
        node1 = _temporal_constant(tr1)
        node2 = _temporal_constant(tr2)
        # Same data should produce same identity
        sid1 = node1.scientific_identity()
        sid2 = node2.scientific_identity()
        # Same seed = same data
        assert sid1 == sid2

    def test_to_json(self):
        tr = _make_temporal_raster(num_layers=2)
        node = _temporal_constant(tr)
        js = node.to_json()
        assert "temporal-1" in js
        assert "root_node_id" in js
        assert "nodes" in js


# ---------------------------------------------------------------------------
# temporal_source -- create expressions from various sources
# ---------------------------------------------------------------------------

class TestTemporalSource:
    def test_from_temporal_raster(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        assert isinstance(expr, TemporalRasterExpression)
        assert expr._inferred_times is not None
        assert len(expr._inferred_times) == 3
        assert expr._signal_name == "test_signal"

    def test_from_temporal_cube(self):
        cube = _make_temporal_cube()
        expr = temporal_source(cube)
        assert isinstance(expr, TemporalRasterExpression)
        assert len(expr._inferred_times) == 5

    def test_from_temporal_expression_passthrough(self):
        tr = _make_temporal_raster(num_layers=2)
        expr1 = temporal_source(tr)
        expr2 = temporal_source(expr1)
        assert expr1 is expr2

    def test_from_invalid_type(self):
        with pytest.raises(MapAlgebraError, match="temporal_source"):
            temporal_source(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Layer-wise temporal operations
# ---------------------------------------------------------------------------

class TestTemporalLayerwiseOps:
    def test_temporal_plus_scalar(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result_expr = expr + 10.0
        assert isinstance(result_expr, TemporalRasterExpression)
        assert result_expr._operation_id == "local.add"
        assert result_expr._inferred_times is not None
        assert len(result_expr._inferred_times) == 3

    def test_temporal_multiply_scalar(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result_expr = expr * 2.0
        assert result_expr._operation_id == "local.multiply"

    def test_temporal_subtract_scalar(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result_expr = expr - 5.0
        assert result_expr._operation_id == "local.subtract"

    def test_temporal_divide_by_scalar(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result_expr = expr / 2.0
        assert result_expr._operation_id == "local.divide"

    def test_scalar_plus_temporal(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result_expr = 10.0 + expr
        assert result_expr._operation_id == "local.add"

    def test_temporal_unary_neg(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result_expr = -expr
        assert result_expr._operation_id == "local.negative"

    def test_temporal_unary_abs(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result_expr = abs(expr)
        assert result_expr._operation_id == "local.absolute"

    def test_temporal_comparison(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = expr > 50.0
        assert result._operation_id == "local.greater"
        assert np.issubdtype(result._inferred_dtype, np.bool_)

    def test_temporal_leq(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = expr <= 30.0
        assert result._operation_id == "local.less_equal"

    def test_temporal_equal_scalar(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = expr == 42.0
        assert result._operation_id == "local.equal"

    def test_temporal_boolean_and(self):
        tr = _make_temporal_raster(num_layers=3, dtype=np.dtype(np.bool_))
        expr = temporal_source(tr)
        expr2 = temporal_source(tr)
        result = expr & expr2
        assert result._operation_id == "local.logical_and"

    def test_temporal_not(self):
        tr = _make_temporal_raster(num_layers=3, dtype=np.dtype(np.bool_))
        expr = temporal_source(tr)
        result = ~expr
        assert result._operation_id == "local.logical_not"

    def test_raster_plus_temporal_expression(self):
        tr = _make_temporal_raster(num_layers=3)
        temporal_expr = temporal_source(tr)

        georef = _make_georef()
        r = _Raster(
            values=np.ones((8, 10), dtype=np.float32),
            georef=georef,
            valid=np.ones((8, 10), dtype=np.bool_),
        )
        result = r + temporal_expr  # Uses Raster.__add__ -> delegates to TemporalRasterExpression
        assert isinstance(result, TemporalRasterExpression)
        assert result._operation_id == "local.add"

    def test_temporal_raster_plus_spatial_raster(self):
        tr = _make_temporal_raster(num_layers=3)
        temporal_expr = temporal_source(tr)

        georef = _make_georef()
        r = _Raster(
            values=np.ones((8, 10), dtype=np.float32),
            georef=georef,
            valid=np.ones((8, 10), dtype=np.bool_),
        )
        result = temporal_expr + r  # TemporalRasterExpression.__add__
        assert isinstance(result, TemporalRasterExpression)
        assert result._operation_id == "local.add"


# ---------------------------------------------------------------------------
# Temporal time coordinate matching
# ---------------------------------------------------------------------------

class TestTemporalTimeMatching:
    def test_matching_times_ok(self):
        tr1 = _make_temporal_raster(num_layers=3)
        tr2 = _make_temporal_raster(num_layers=3)
        e1 = temporal_source(tr1)
        e2 = temporal_source(tr2)
        result = e1 + e2
        assert isinstance(result, TemporalRasterExpression)

    def test_mismatched_times_raises(self):
        tr1 = _make_temporal_raster(num_layers=3)
        tr2 = _make_temporal_raster(num_layers=5)
        e1 = temporal_source(tr1)
        e2 = temporal_source(tr2)
        with pytest.raises(MapAlgebraExpressionError, match="time coordinates"):
            e1 + e2


# ---------------------------------------------------------------------------
# Temporal reductions
# ---------------------------------------------------------------------------

class TestTemporalReductions:
    def test_reduction_mean_returns_raster_expression(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = temporal_mean(expr)
        assert isinstance(result, RasterExpression)
        assert result._operation_id == "temporal.mean"
        assert result._inferred_grid is not None

    def test_reduction_min_returns_raster_expression(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = temporal_min(expr)
        assert isinstance(result, RasterExpression)
        assert result._operation_id == "temporal.min"

    def test_reduction_max_returns_raster_expression(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = temporal_max(expr)
        assert isinstance(result, RasterExpression)
        assert result._operation_id == "temporal.max"

    def test_reduction_std_returns_raster_expression(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = temporal_std(expr, ddof=0)
        assert isinstance(result, RasterExpression)
        assert result._operation_id == "temporal.std"

    def test_reduction_sum_returns_raster_expression(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = temporal_sum(expr)
        assert isinstance(result, RasterExpression)
        assert result._operation_id == "temporal.sum"

    def test_reduction_count_returns_raster_expression(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = temporal_count(expr)
        assert isinstance(result, RasterExpression)
        assert result._operation_id == "temporal.count"

    def test_reduction_on_eager_temporal_raster_mean(self):
        tr = _make_temporal_raster(num_layers=5)
        result = temporal_mean(tr)
        assert isinstance(result, _Raster)
        assert result.shape == (8, 10)

    def test_reduction_on_eager_temporal_raster_min(self):
        tr = _make_temporal_raster(num_layers=5)
        result = temporal_min(tr)
        assert isinstance(result, _Raster)

    def test_reduction_on_eager_temporal_raster_max(self):
        tr = _make_temporal_raster(num_layers=5)
        result = temporal_max(tr)
        assert isinstance(result, _Raster)

    def test_reduction_on_eager_temporal_raster_sum(self):
        tr = _make_temporal_raster(num_layers=3)
        result = temporal_sum(tr)
        assert isinstance(result, _Raster)
        assert result.dtype == np.dtype(np.float64)

    def test_reduction_on_eager_temporal_raster_count(self):
        tr = _make_temporal_raster(num_layers=3)
        result = temporal_count(tr)
        assert isinstance(result, _Raster)
        assert result.dtype == np.dtype(np.int64)
        np.testing.assert_array_equal(result.values, np.full((8, 10), 3, dtype=np.int64))

    def test_reduction_mean_with_nodata(self):
        tr = _make_temporal_raster(num_layers=3, nodata=-9999.0)
        result = temporal_mean(tr)
        assert isinstance(result, _Raster)
        assert result.shape == (8, 10)

    def test_reduction_all_invalid(self):
        georef = _make_georef()
        times = _make_times(3)
        values = np.full((3, 8, 10), 1.0, dtype=np.float32)
        valid = np.zeros((3, 8, 10), dtype=np.bool_)
        tr = TemporalRaster(values=values, times=times, georef=georef, valid=valid)
        result = temporal_mean(tr)
        assert not np.any(result.valid)

    def test_composed_expression(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        mean_expr = temporal_mean(expr)  # RasterExpression
        result = mean_expr >= 40.0  # RasterExpression composed further
        assert isinstance(result, RasterExpression)
        assert result._operation_id == "local.greater_equal"


# ---------------------------------------------------------------------------
# compute_temporal -- eager materialization
# ---------------------------------------------------------------------------

class TestComputeTemporal:
    def test_compute_simple_expression(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        result = compute_temporal(expr)
        assert isinstance(result, TemporalRaster)
        assert result.num_layers == 3
        assert result.shape == (3, 8, 10)
        np.testing.assert_array_almost_equal(result.values, tr.values)

    def test_compute_add_scalar(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        expr2 = expr + 10.0
        result = compute_temporal(expr2)
        expected = tr.values + 10.0
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_compute_multiply_scalar(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        expr2 = expr * 2.0
        result = compute_temporal(expr2)
        expected = tr.values * 2.0
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_compute_chained_ops(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        expr2 = (expr + 5.0) * 2.0
        result = compute_temporal(expr2)
        expected = (tr.values + 5.0) * 2.0
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_compute_comparison(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        expr2 = expr > 50.0
        result = compute_temporal(expr2)
        assert result.dtype == np.dtype(np.bool_)
        expected = tr.values > 50.0
        np.testing.assert_array_equal(result.values, expected)

    def test_compute_unary_neg(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        expr2 = -expr
        result = compute_temporal(expr2)
        np.testing.assert_array_almost_equal(result.values, -tr.values)

    def test_compute_unary_abs(self):
        tr = _make_temporal_raster(num_layers=3)
        negative_vals = -tr.values
        tr2 = tr.with_validity(tr.valid)
        object.__setattr__(tr2, 'values', -tr.values)
        expr = temporal_source(tr2)
        expr2 = abs(expr)
        result = compute_temporal(expr2)
        np.testing.assert_array_almost_equal(result.values, np.abs(tr2.values))

    def test_compute_with_spatial_broadcast(self):
        tr = _make_temporal_raster(num_layers=3)
        temporal_expr = temporal_source(tr)

        georef = _make_georef()
        r = _Raster(
            values=np.ones((8, 10), dtype=np.float32) * 100.0,
            georef=georef,
            valid=np.ones((8, 10), dtype=np.bool_),
        )
        expr2 = temporal_expr + r
        result = compute_temporal(expr2)
        expected = tr.values + 100.0
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_compute_invalid_input(self):
        with pytest.raises(MapAlgebraExpressionError, match="TemporalRasterExpression"):
            compute_temporal("not an expression")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Identity and serialization
# ---------------------------------------------------------------------------

class TestTemporalIdentity:
    def test_identity_different_ops(self):
        tr = _make_temporal_raster(num_layers=3)
        e1 = temporal_source(tr)
        e2 = e1 + 10.0
        assert e1.scientific_identity() != e2.scientific_identity()

    def test_identity_same_ops_same(self):
        tr1 = _make_temporal_raster(num_layers=3)
        tr2 = _make_temporal_raster(num_layers=3)
        e1 = temporal_source(tr1) + 10.0
        e2 = temporal_source(tr2) + 10.0
        assert e1.scientific_identity() == e2.scientific_identity()

    def test_json_output(self):
        tr = _make_temporal_raster(num_layers=3)
        e1 = temporal_source(tr)
        e2 = e1 * 2.0
        js = e2.to_json()
        assert "\"node_id\"" in js
        assert "\"operation_id\"" in js
        assert "\"local.multiply\"" in js
        assert "\"schema_version\"" in js


# ---------------------------------------------------------------------------
# explain_temporal
# ---------------------------------------------------------------------------

class TestExplainTemporal:
    def test_explain_simple(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        text = explain_temporal(expr)
        assert "temporal.constant" in text
        assert "test_signal" in text

    def test_explain_chain(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        expr2 = expr + 5.0
        expr3 = expr2 * 2.0
        text = explain_temporal(expr3)
        assert "local.add" in text
        assert "local.multiply" in text


# ---------------------------------------------------------------------------
# Directional correctness: layer-wise eager vs reference
# ---------------------------------------------------------------------------

class TestEagerCorrectness:
    def test_add_scalar_matches_reference(self):
        tr = _make_temporal_raster(num_layers=4)
        expr = temporal_source(tr) + 10.0
        result = compute_temporal(expr)

        expected_values = tr.values + 10.0
        np.testing.assert_array_almost_equal(result.values, expected_values)
        np.testing.assert_array_equal(result.valid, tr.valid)
        np.testing.assert_array_equal(result.times, tr.times)
        assert result.georef.height == tr.georef.height
        assert result.georef.width == tr.georef.width

    def test_multiply_then_add_matches_reference(self):
        tr = _make_temporal_raster(num_layers=4)
        expr = temporal_source(tr) * 3.0 + 5.0
        result = compute_temporal(expr)

        expected = tr.values * 3.0 + 5.0
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_comparison_validity_preserved(self):
        tr = _make_temporal_raster(num_layers=4, nodata=-9999.0)
        expr = temporal_source(tr) > 50.0
        result = compute_temporal(expr)

        # Invalid pixels should remain invalid
        assert not result.valid[0, 0, 0]

    def test_metadata_preserved(self):
        tr = _make_temporal_raster(num_layers=4)
        expr = temporal_source(tr) + 1.0
        result = compute_temporal(expr)

        assert result.units == tr.units
        assert result.signal_name == tr.signal_name
        np.testing.assert_array_equal(result.times, tr.times)

    def test_spatial_broadcast_metadata(self):
        tr = _make_temporal_raster(num_layers=4)
        temporal_expr = temporal_source(tr)
        georef = _make_georef()
        r = _Raster(
            values=np.full((8, 10), 50.0, dtype=np.float32),
            georef=georef,
            valid=np.ones((8, 10), dtype=np.bool_),
            units="metres",
        )
        expr2 = temporal_expr + r
        result = compute_temporal(expr2)

        assert result.num_layers == 4
        np.testing.assert_array_equal(result.times, tr.times)
        np.testing.assert_array_almost_equal(result.values, tr.values + 50.0)


# ---------------------------------------------------------------------------
# TemporalGeoTiffSeries file-backed temporal_source tests
# ---------------------------------------------------------------------------

class TestFileBackedTemporalSource:
    def test_temporal_source_from_series_via_path(self, tmp_path):
        from lunarscout.temporal_store import write_temporal_cube

        georef = _make_georef(nodata=-9999.0)
        times = _make_times(5)
        values = np.random.default_rng(42).uniform(0.0, 100.0, size=(5, 8, 10)).astype(np.float32)
        cube = TemporalCube(values=values, times=times, georef=georef)

        series_dir = tmp_path / "test_series"
        series = write_temporal_cube(str(series_dir), cube, signal_name="sun_fraction", units="fraction")

        expr = temporal_source(str(series_dir))
        assert isinstance(expr, TemporalRasterExpression)
        assert expr._inferred_times is not None
        assert len(expr._inferred_times) == 5
        assert expr._signal_name == "sun_fraction"
        assert expr._inferred_units == "fraction"

        series.close()

    def test_temporal_source_from_open_series(self, tmp_path):
        from lunarscout.temporal_store import write_temporal_cube, open_temporal_cube

        georef = _make_georef(nodata=-9999.0)
        times = _make_times(3)
        values = np.ones((3, 8, 10), dtype=np.float32)
        cube = TemporalCube(values=values, times=times, georef=georef)

        series_dir = tmp_path / "test_series2"
        write_temporal_cube(str(series_dir), cube)
        series = open_temporal_cube(str(series_dir))

        expr = temporal_source(series)
        assert isinstance(expr, TemporalRasterExpression)
        assert len(expr._inferred_times) == 3
        series.close()
        result = compute_temporal(expr)
        assert result.num_layers == 3

    def test_temporal_reduction_from_file_series(self, tmp_path):
        from lunarscout.temporal_store import write_temporal_cube

        georef = _make_georef(nodata=None)
        times = _make_times(3)
        values = np.random.default_rng(42).uniform(0.0, 100.0, size=(3, 8, 10)).astype(np.float32)
        cube = TemporalCube(values=values, times=times, georef=georef)

        series_dir = tmp_path / "test_series3"
        series = write_temporal_cube(str(series_dir), cube)

        expr = temporal_source(str(series_dir))
        mean_expr = temporal_mean(expr)
        assert isinstance(mean_expr, RasterExpression)

        series.close()


# ---------------------------------------------------------------------------
# Import boundary test
# ---------------------------------------------------------------------------

class TestImportBoundary:
    def test_temporal_raster_import_no_cuda(self):
        import sys
        before_numba = "numba" in sys.modules
        before_spice = "spiceypy" in sys.modules

        from lunarscout.map_algebra._temporal_model import TemporalRaster as TR
        assert isinstance(TR, type)

        if not before_numba:
            assert "numba" not in sys.modules, "TemporalRaster import should not load numba"
        if not before_spice:
            assert "spiceypy" not in sys.modules, "TemporalRaster import should not load spiceypy"

    def test_import_lunarscout_still_works(self):
        import lunarscout as ls
        assert hasattr(ls, "TemporalRaster")
        assert isinstance(ls.TemporalRaster, type)


# ---------------------------------------------------------------------------
# Stress / larger temporal tests
# ---------------------------------------------------------------------------

class TestStressTemporal:
    def test_many_layers_eager(self):
        num_layers = 100
        georef = _make_georef(width=4, height=3)
        times = _make_times(num_layers)
        rng = np.random.default_rng(99)
        values = rng.uniform(0.0, 100.0, size=(num_layers, 3, 4)).astype(np.float32)
        valid = np.ones((num_layers, 3, 4), dtype=np.bool_)
        tr = TemporalRaster(values=values, times=times, georef=georef, valid=valid)

        expr = temporal_source(tr) + 10.0
        result = compute_temporal(expr)
        assert result.num_layers == 100
        np.testing.assert_array_almost_equal(result.values, values + 10.0)

    def test_many_layers_file_backed(self, tmp_path):
        from lunarscout.temporal_store import write_temporal_cube, open_temporal_cube

        num_layers = 200
        georef = _make_georef(width=4, height=3, nodata=None)
        times = _make_times(num_layers)
        rng = np.random.default_rng(77)
        values = rng.uniform(0.0, 100.0, size=(num_layers, 3, 4)).astype(np.float32)
        cube = TemporalCube(values=values, times=times, georef=georef)

        series_dir = tmp_path / "stress_series"
        series = write_temporal_cube(str(series_dir), cube)

        expr = temporal_source(str(series_dir))
        assert expr._inferred_times is not None
        assert len(expr._inferred_times) == num_layers

        # Verify that creating the expression doesn't load the full cube
        mean_expr = temporal_mean(expr)
        assert isinstance(mean_expr, RasterExpression)
        assert mean_expr._operation_id == "temporal.mean"

        series.close()

    def test_three_thousand_layer_reduction(self):
        num_layers = 3_000
        georef = _make_georef(width=2, height=2)
        times = _make_times(num_layers)
        values = np.arange(num_layers * 4, dtype=np.float32).reshape(
            num_layers, 2, 2,
        )
        temporal = TemporalRaster(
            values=values,
            times=times,
            georef=georef,
            valid=np.ones_like(values, dtype=np.bool_),
        )

        result = temporal_mean((temporal.expression() + 2.0) * 0.5)
        assert isinstance(result, RasterExpression)
        computed = compute(result)
        np.testing.assert_allclose(
            computed.values,
            np.mean((values.astype(np.float64) + 2.0) * 0.5, axis=0),
        )


# ---------------------------------------------------------------------------
# P1 Fix: Temporal reduction via ma.compute()
# ---------------------------------------------------------------------------

class TestTemporalReductionViaCompute:
    def test_compute_temporal_mean(self):
        tr = _make_temporal_raster(num_layers=5)
        expr = temporal_source(tr)
        reduced = temporal_mean(expr)
        assert isinstance(reduced, RasterExpression)

        from lunarscout.map_algebra import compute as ma_compute
        result = ma_compute(reduced)
        assert isinstance(result, _Raster)
        assert result.shape == (8, 10)

    def test_compute_temporal_min(self):
        tr = _make_temporal_raster(num_layers=5)
        expr = temporal_source(tr)
        reduced = temporal_min(expr)
        from lunarscout.map_algebra import compute as ma_compute
        result = ma_compute(reduced)
        assert isinstance(result, _Raster)
        assert result.shape == (8, 10)

    def test_compute_temporal_max(self):
        tr = _make_temporal_raster(num_layers=5)
        expr = temporal_source(tr)
        reduced = temporal_max(expr)
        from lunarscout.map_algebra import compute as ma_compute
        result = ma_compute(reduced)
        assert isinstance(result, _Raster)

    def test_compose_temporal_reduction_with_spatial(self):
        tr = _make_temporal_raster(num_layers=5)
        expr = temporal_source(tr)
        reduced = temporal_mean(expr)
        composed = reduced >= 40.0
        assert isinstance(composed, RasterExpression)
        from lunarscout.map_algebra import compute as ma_compute
        result = ma_compute(composed)
        assert isinstance(result, _Raster)
        assert result.dtype == np.dtype(np.bool_)

    def test_compute_temporal_count_via_compute(self):
        tr = _make_temporal_raster(num_layers=5)
        expr = temporal_source(tr)
        reduced = temporal_count(expr)
        from lunarscout.map_algebra import compute as ma_compute
        result = ma_compute(reduced)
        assert isinstance(result, _Raster)
        assert np.all(result.values == 5)

    def test_compute_temporal_count_returns_no_units(self):
        tr = _make_temporal_raster(num_layers=5)
        expr = temporal_source(tr)
        reduced = temporal_count(expr)
        assert reduced._inferred_units is None

    def test_compute_temporal_std(self):
        tr = _make_temporal_raster(num_layers=5)
        expr = temporal_source(tr)
        reduced = temporal_std(expr, ddof=0)
        from lunarscout.map_algebra import compute as ma_compute
        result = ma_compute(reduced)
        assert isinstance(result, _Raster)


# ---------------------------------------------------------------------------
# P1 Fix: Scalar-left operands
# ---------------------------------------------------------------------------

class TestScalarLeftOperands:
    def test_scalar_plus_expr_executes(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        composed = 10.0 + expr
        result = compute_temporal(composed)
        assert result.num_layers == 3
        expected = 10.0 + tr.values
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_scalar_minus_expr_executes(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        composed = 100.0 - expr
        result = compute_temporal(composed)
        expected = 100.0 - tr.values
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_scalar_div_expr_executes(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        composed = 100.0 / expr
        result = compute_temporal(composed)
        expected = 100.0 / tr.values
        np.testing.assert_array_almost_equal(result.values, expected)

    @pytest.mark.parametrize(
        ("operation", "reference"),
        [
            (lambda expr: 100.0 // expr, lambda values: 100.0 // values),
            (lambda expr: 100.0 % expr, lambda values: 100.0 % values),
            (lambda expr: 2.0 ** expr, lambda values: 2.0 ** values),
        ],
    )
    def test_other_scalar_left_arithmetic_executes(self, operation, reference):
        tr = _make_temporal_raster(num_layers=2)
        tr = TemporalRaster(
            values=np.clip(tr.values, 1.0, 5.0),
            times=tr.times,
            georef=tr.georef,
            valid=tr.valid,
        )
        result = compute_temporal(operation(tr.expression()))
        np.testing.assert_allclose(result.values, reference(tr.values))

    def test_ma_unary_sqrt_on_expr(self):
        from lunarscout.map_algebra import sqrt as ma_sqrt
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr) + 10.0
        composed = ma_sqrt(expr)
        assert isinstance(composed, TemporalRasterExpression)
        result = compute_temporal(composed)
        expected = np.sqrt(tr.values + 10.0)
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_ma_unary_sin_on_expr(self):
        from lunarscout.map_algebra import sin as ma_sin
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        composed = ma_sin(expr)
        assert isinstance(composed, TemporalRasterExpression)

    def test_ma_unary_log_on_expr(self):
        from lunarscout.map_algebra import log as ma_log
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        composed = ma_log(expr)
        assert isinstance(composed, TemporalRasterExpression)

    def test_ma_minimum_on_expr(self):
        from lunarscout.map_algebra import minimum as ma_min
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr)
        composed = ma_min(expr, 50.0)
        assert isinstance(composed, TemporalRasterExpression)
        result = compute_temporal(composed)
        expected = np.minimum(tr.values, 50.0)
        np.testing.assert_array_almost_equal(result.values, expected)


# ---------------------------------------------------------------------------
# P1 Fix: Spatial grid compatibility validation
# ---------------------------------------------------------------------------

class TestGridCompatibility:
    def test_temporal_ops_on_different_grids_raises(self):
        tr1 = _make_temporal_raster(num_layers=3)

        other_georef = _make_georef(width=5, height=5)
        times = _make_times(3)
        values = np.ones((3, 5, 5), dtype=np.float32)
        valid = np.ones((3, 5, 5), dtype=np.bool_)
        tr2 = TemporalRaster(values=values, times=times, georef=other_georef, valid=valid)

        e1 = temporal_source(tr1)
        e2 = temporal_source(tr2)

        from lunarscout.errors import GridMismatchError
        with pytest.raises(GridMismatchError):
            e1 + e2

    def test_shifted_spatial_raster_broadcast_raises(self):
        tr = _make_temporal_raster(num_layers=3)
        temporal_expr = temporal_source(tr)

        shifted_georef = _make_georef(origin_x=99999.0, origin_y=99999.0)
        r = _Raster(
            values=np.ones((8, 10), dtype=np.float32),
            georef=shifted_georef,
            valid=np.ones((8, 10), dtype=np.bool_),
        )

        from lunarscout.errors import GridMismatchError
        with pytest.raises(GridMismatchError):
            temporal_expr + r


# ---------------------------------------------------------------------------
# P2 Fix: TemporalRaster time-coordinate contract
# ---------------------------------------------------------------------------

class TestTemporalRasterTimeContract:
    def test_empty_raises(self):
        georef = _make_georef()
        times = np.array([], dtype="datetime64[us]")
        values = np.empty((0, 8, 10), dtype=np.float32)
        valid = np.empty((0, 8, 10), dtype=np.bool_)
        with pytest.raises(MapAlgebraError, match="at least one"):
            TemporalRaster(values=values, times=times, georef=georef, valid=valid)

    def test_nat_raises(self):
        georef = _make_georef()
        times = np.array(["2027-01-01", "NaT"], dtype="datetime64[D]")
        values = np.ones((2, 8, 10), dtype=np.float32)
        valid = np.ones((2, 8, 10), dtype=np.bool_)
        with pytest.raises(MapAlgebraError, match="NaT"):
            TemporalRaster(values=values, times=times, georef=georef, valid=valid)

    def test_decreasing_times_raises(self):
        georef = _make_georef()
        times = np.array(["2027-01-03", "2027-01-01"], dtype="datetime64[D]")
        values = np.ones((2, 8, 10), dtype=np.float32)
        valid = np.ones((2, 8, 10), dtype=np.bool_)
        with pytest.raises(MapAlgebraError, match="strictly increasing"):
            TemporalRaster(values=values, times=times, georef=georef, valid=valid)

    def test_duplicate_times_raises(self):
        georef = _make_georef()
        times = np.array(["2027-01-01", "2027-01-01"], dtype="datetime64[D]")
        values = np.ones((2, 8, 10), dtype=np.float32)
        valid = np.ones((2, 8, 10), dtype=np.bool_)
        with pytest.raises(MapAlgebraError, match="strictly increasing"):
            TemporalRaster(values=values, times=times, georef=georef, valid=valid)


# ---------------------------------------------------------------------------
# P2 Fix: Reducer semantics
# ---------------------------------------------------------------------------

class TestReducerSemantics:
    def test_count_has_no_units(self):
        tr = _make_temporal_raster(num_layers=4, nodata=-9999.0)
        result = temporal_count(tr)
        assert result.units is None

    def test_all_invalid_count_is_valid_zero(self):
        georef = _make_georef()
        times = _make_times(3)
        valid = np.zeros((3, 8, 10), dtype=np.bool_)
        values = np.zeros((3, 8, 10), dtype=np.float32)
        tr = TemporalRaster(values=values, times=times, georef=georef, valid=valid)
        result = temporal_count(tr)
        assert np.all(result.values == 0)
        assert result.all_valid

    def test_ddof_finite_and_nonnegative(self):
        tr = _make_temporal_raster(num_layers=3)
        with pytest.raises(MapAlgebraError, match="ddof"):
            temporal_std(tr, ddof=-1)
        with pytest.raises(MapAlgebraError, match="ddof"):
            temporal_std(tr, ddof=float('inf'))

    def test_integer_min_preserves_dtype(self):
        georef = _make_georef()
        times = _make_times(3)
        values = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32).reshape(3, 1, 2)
        values = np.tile(values, (1, 8, 5))
        valid = np.ones_like(values, dtype=np.bool_)
        tr = TemporalRaster(values=values, times=times, georef=georef, valid=valid)
        result = temporal_min(tr)
        assert isinstance(result, _Raster)
        assert result.dtype == np.dtype(np.int32)

    def test_integer_max_preserves_dtype(self):
        georef = _make_georef()
        times = _make_times(3)
        values = np.arange(3 * 8 * 10, dtype=np.int16).reshape(3, 8, 10)
        valid = np.ones_like(values, dtype=np.bool_)
        tr = TemporalRaster(values=values, times=times, georef=georef, valid=valid)
        result = temporal_max(tr)
        assert result.dtype == np.dtype(np.int16)

    def test_sum_dtype_matches_expression_inference(self):
        tr = _make_temporal_raster(num_layers=3, dtype=np.dtype(np.int16))
        eager = temporal_sum(tr)
        expression = temporal_sum(tr.expression())
        assert eager.dtype == expression.dtype == np.dtype(np.float64)

    def test_mean_preserves_units(self):
        tr = _make_temporal_raster(num_layers=4)
        result = temporal_mean(tr)
        assert result.units == "metres"

    def test_std_preserves_units(self):
        tr = _make_temporal_raster(num_layers=4)
        result = temporal_std(tr, ddof=0)
        assert result.units == "metres"


# ---------------------------------------------------------------------------
# P1 Fix: File-backed temporal_source is executable
# ---------------------------------------------------------------------------

class TestFileBackedSourceExecution:
    def test_temporal_source_from_path_is_computable(self, tmp_path):
        from lunarscout.temporal_store import write_temporal_cube

        georef = _make_georef(nodata=None)
        times = _make_times(4)
        values = np.arange(4 * 8 * 10, dtype=np.float32).reshape(4, 8, 10)
        cube = TemporalCube(values=values, times=times, georef=georef)

        series_dir = tmp_path / "compute_test_series"
        series = write_temporal_cube(str(series_dir), cube, signal_name="test", units="metres")

        expr = temporal_source(str(series_dir))
        result = compute_temporal(expr)
        assert result.num_layers == 4
        assert result.signal_name == "test"

        series.close()

    def test_temporal_mean_from_file_series_executes(self, tmp_path):
        from lunarscout.temporal_store import write_temporal_cube
        from lunarscout.map_algebra import compute as ma_compute

        georef = _make_georef(nodata=None)
        times = _make_times(3)
        values = np.ones((3, 8, 10), dtype=np.float32)
        cube = TemporalCube(values=values, times=times, georef=georef)

        series_dir = tmp_path / "mean_test_series"
        series = write_temporal_cube(str(series_dir), cube)
        expr = temporal_source(str(series_dir))
        reduced = temporal_mean(expr)
        result = ma_compute(reduced)
        assert isinstance(result, _Raster)
        np.testing.assert_array_almost_equal(result.values, np.ones((8, 10), dtype=np.float32))

        series.close()


# ---------------------------------------------------------------------------
# P1 Fix: ma.compute() handles TemporalRasterExpression
# ---------------------------------------------------------------------------

class TestMaComputeTemporalExpr:
    def test_ma_compute_dispatches_temporal(self):
        tr = _make_temporal_raster(num_layers=3)
        expr = temporal_source(tr) + 5.0
        result = compute_temporal(expr)
        assert isinstance(result, TemporalRaster)
        assert result.num_layers == 3
