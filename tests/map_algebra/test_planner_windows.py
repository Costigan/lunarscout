from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from lunarscout.errors import MapAlgebraExpressionError
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra._planner import plan_expression
from lunarscout.map_algebra._model import RasterExpression
from lunarscout.map_algebra._windows import SourceWindowCache, enumerate_windows
from lunarscout.map_algebra._windowed import execute_windowed
from lunarscout.map_algebra import (
    source,
    compute,
    write,
    raster as ma_raster,
)
from tests.map_algebra.conftest import MOON_WKT, MOON_PROJ4


def _georef(h: int, w: int) -> GeoReference:
    return GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
        width=w, height=h, pixel_size_x=20.0, pixel_size_y=-20.0, nodata=None,
    )


def _write_tiff(tmp_path, name, values, dtype="float32"):
    import rasterio

    h, w = values.shape
    g = _georef(h, w)
    path = tmp_path / name
    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": 1,
        "dtype": dtype,
        "crs": g.projection_wkt,
        "transform": Affine.from_gdal(*g.affine_transform),
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(np.asarray(values, dtype=dtype), 1)
    return path


class TestPlanner:
    def test_registry_file_backed_claims_match_window_executor(self):
        from lunarscout.map_algebra import list_operations
        from lunarscout.map_algebra._windowed import WINDOWED_OPERATION_IDS

        declared = {
            item["id"]
            for item in list_operations(execution_mode="file_backed")
        }
        intrinsic = {"source", "constant"} | {
            operation_id
            for operation_id in declared
            if operation_id.startswith("coordinate.")
        }
        assert declared - intrinsic == set(WINDOWED_OPERATION_IDS)

    def test_plans_simple_source_expression(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        expr = source(p) + 3
        plan = plan_expression(expr)
        assert plan.total_windows == 1
        assert plan.n_sources == 1
        assert plan.n_operations >= 1
        assert plan.window_width == 128
        assert plan.window_height == 128
        assert plan.output_dtype is not None

    def test_plans_multi_source_expression(self, tmp_path):
        p1 = _write_tiff(tmp_path, "a.tif", np.ones((10, 10), dtype=np.float32))
        p2 = _write_tiff(tmp_path, "b.tif", np.ones((10, 10), dtype=np.float32))
        expr = source(p1) + source(p2)
        plan = plan_expression(expr)
        assert plan.n_sources == 2

    def test_rejects_unsupported_focal(self):
        g = _georef(10, 10)
        r = ma_raster(np.ones((10, 10), dtype=np.float32), g)
        from lunarscout.map_algebra import focal_mean

        expr = focal_mean(r.expression(), size=3)
        with pytest.raises(MapAlgebraExpressionError, match="not yet supported"):
            plan_expression(expr)

    def test_reports_window_layout(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((300, 300), dtype=np.float32))
        expr = source(p) + 1
        plan = plan_expression(expr, window_width=100, window_height=100)
        assert plan.n_windows_x == 3
        assert plan.n_windows_y == 3
        assert plan.total_windows == 9

    def test_reports_single_window_for_small_raster(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((5, 5), dtype=np.float32))
        expr = source(p) + 1
        plan = plan_expression(expr)
        assert plan.total_windows == 1
        assert plan.n_windows_x == 1
        assert plan.n_windows_y == 1

    def test_plan_includes_estimated_memory(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((100, 100), dtype=np.float32))
        expr = source(p) + 3
        plan = plan_expression(expr, window_width=64, window_height=64)
        assert plan.estimated_per_window_bytes > 0

    def test_planner_with_coordinate_expression(self):
        g = _georef(50, 50)
        from lunarscout.map_algebra import row_indices

        expr = row_indices(g) * 0.1
        plan = plan_expression(expr)
        assert plan.n_sources >= 1
        assert plan.total_windows >= 1

    def test_rejects_cycle_defensively(self):
        g = _georef(2, 2)
        expression = ma_raster(np.ones((2, 2), dtype=np.float32), g).expression()
        object.__setattr__(expression, "_operands", (expression,))
        with pytest.raises(MapAlgebraExpressionError, match="cycle"):
            plan_expression(expression)

    def test_enforces_node_depth_and_source_limits(self, tmp_path, monkeypatch):
        import lunarscout.map_algebra._planner as planner
        path = _write_tiff(tmp_path, "limits.tif", np.ones((2, 2), dtype=np.float32))

        monkeypatch.setattr(planner, "_MAX_NODES", 1)
        with pytest.raises(MapAlgebraExpressionError) as node_error:
            planner.plan_expression(source(path) + 1)
        assert node_error.value.code == "map_algebra_too_many_nodes"

        monkeypatch.setattr(planner, "_MAX_NODES", 10_000)
        monkeypatch.setattr(planner, "_MAX_DEPTH", 1)
        with pytest.raises(MapAlgebraExpressionError) as depth_error:
            planner.plan_expression(source(path) + 1)
        assert depth_error.value.code == "map_algebra_too_deep"

        monkeypatch.setattr(planner, "_MAX_DEPTH", 500)
        monkeypatch.setattr(planner, "_MAX_SOURCES", 1)
        with pytest.raises(MapAlgebraExpressionError) as source_error:
            planner.plan_expression(source(path) + source(path))
        assert source_error.value.code == "map_algebra_too_many_sources"

    def test_public_plan_propagates_structured_planner_error(self):
        import lunarscout.map_algebra as ma
        expression = ma.focal_mean(
            ma_raster(np.ones((3, 3), dtype=np.float32), _georef(3, 3)).expression(),
            size=3,
        )
        with pytest.raises(MapAlgebraExpressionError) as error:
            ma.plan(expression)
        assert error.value.code == "map_algebra_unsupported_windowed_operation"


class TestWindowEnumeration:
    def test_enumerates_single_window(self):
        windows = list(enumerate_windows(10, 10, 128, 128))
        assert len(windows) == 1
        idx, x0, y0, w, h, n_cols = windows[0]
        assert idx == 0
        assert x0 == 0 and y0 == 0
        assert w == 10 and h == 10

    def test_enumerates_multiple_full_windows(self):
        windows = list(enumerate_windows(256, 256, 128, 128))
        assert len(windows) == 4

    def test_enumerates_partial_edge_windows(self):
        windows = list(enumerate_windows(200, 200, 128, 128))
        assert len(windows) == 4

    def test_non_divisible_dimensions(self):
        windows = list(enumerate_windows(100, 100, 40, 40))
        assert len(windows) == 9

    def test_smaller_than_block(self):
        windows = list(enumerate_windows(5, 3, 128, 128))
        assert len(windows) == 1
        _, x0, y0, w, h, _ = windows[0]
        assert w == 5 and h == 3


class TestSourceWindowCache:
    def test_reads_source_window(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        expr = source(p)
        with SourceWindowCache() as cache:
            vals = cache.read_values(expr, 0, 0, 0, 2, 2)
            np.testing.assert_array_equal(vals, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    def test_caches_repeated_source_windows(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        expr = source(p)
        with SourceWindowCache(max_windows=64) as cache:
            cache.read_values(expr, 0, 0, 0, 5, 5)
            assert cache.window_count == 1
            cache.read_values(expr, 0, 0, 0, 5, 5)
            assert cache.window_count == 1

    def test_cache_bounds_respected(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        expr = source(p)
        with SourceWindowCache(max_windows=3) as cache:
            for i in range(5):
                x0 = (i * 2) % 8
                cache.read_values(expr, i, x0, 0, 2, 2)
            assert cache.window_count <= 3

    def test_closes_datasets_after_use(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        expr = source(p)
        cache = SourceWindowCache()
        cache.read_values(expr, 0, 0, 0, 5, 5)
        assert cache.dataset_count == 1
        cache.close()
        assert cache.dataset_count == 0
        assert cache.is_closed
        assert cache.window_count == 0

    def test_closes_on_failure(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        expr = source(p)
        cache = SourceWindowCache()
        try:
            cache.read_values(expr, 0, 0, 0, 5, 5)
            raise ValueError("injected error")
        except ValueError:
            cache.close()
        assert cache.is_closed

    def test_reads_constant_window(self):
        g = _georef(3, 3)
        r = ma_raster(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), g)
        expr = r.expression()
        with SourceWindowCache() as cache:
            vals = cache.read_values(expr, 0, 1, 0, 2, 2)
            assert vals.shape == (2, 2)
            assert vals[0, 0] == 2
            assert vals[1, 1] == 6

    def test_reads_coordinate_window(self):
        g = _georef(5, 5)
        from lunarscout.map_algebra import row_indices

        expr = row_indices(g)
        with SourceWindowCache() as cache:
            vals = cache.read_values(expr, 0, 0, 2, 5, 3)
            assert vals.shape == (3, 5)
            np.testing.assert_array_equal(vals[:, 0], np.arange(2, 5, dtype=np.int64))


class TestEagerVsWindowedParity:
    def _parity_check(self, tmp_path, values, expr_fn, dtype="float32"):
        p = _write_tiff(tmp_path, "src.tif", values, dtype=dtype)
        src_expr = source(p)
        windowed_expr = expr_fn(src_expr)
        eager_raster = expr_fn(compute(src_expr))
        result_w = write(tmp_path / "out.tif", windowed_expr)
        import rasterio
        with rasterio.open(result_w) as ds:
            w_data = ds.read(1)
        if eager_raster.values.dtype == np.bool_:
            np.testing.assert_array_equal(w_data, eager_raster.values.astype(np.uint8))
        else:
            np.testing.assert_array_equal(w_data, eager_raster.values.astype(w_data.dtype))

    def test_add_parity(self, tmp_path):
        self._parity_check(
            tmp_path,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            lambda x: x + 10.0,
        )

    def test_subtract_parity(self, tmp_path):
        self._parity_check(
            tmp_path,
            np.array([[5.0, 8.0], [3.0, 1.0]], dtype=np.float32),
            lambda x: x - 2.0,
        )

    def test_multiply_parity(self, tmp_path):
        self._parity_check(
            tmp_path,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            lambda x: x * 3.0,
        )

    def test_divide_parity(self, tmp_path):
        self._parity_check(
            tmp_path,
            np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
            lambda x: x / 2.0,
        )

    def test_comparison_parity(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[1.0, 5.0], [3.0, 1.0]], dtype=np.float32))
        s = source(p)
        windowed = s > 3.0
        eager = compute(s) > 3.0
        result_w = write(tmp_path / "out.tif", windowed)
        import rasterio
        with rasterio.open(result_w) as ds:
            w_data = ds.read(1)
        np.testing.assert_array_equal(w_data, eager.values.astype(np.uint8))

    def test_unary_math_parity(self, tmp_path):
        from lunarscout.map_algebra import sqrt as ma_sqrt
        self._parity_check(
            tmp_path,
            np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32),
            lambda x: ma_sqrt(x),
        )

    def test_scalar_left_parity(self, tmp_path):
        self._parity_check(
            tmp_path,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            lambda x: 10.0 - x,
        )


class TestEdgeCases:
    def test_uint8_source(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[0, 64, 128]], dtype=np.uint8), dtype="uint8")
        expr = source(p) + 1
        result = write(tmp_path / "out.tif", expr)
        import rasterio
        with rasterio.open(result) as ds:
            data = ds.read(1)
        assert data.dtype == np.uint8
        np.testing.assert_array_equal(data, np.array([[1, 65, 129]], dtype=np.uint8))

    def test_int32_source(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[-100, 0, 100]], dtype=np.int32), dtype="int32")
        expr = source(p) * 2
        result = write(tmp_path / "out.tif", expr)
        import rasterio
        with rasterio.open(result) as ds:
            data = ds.read(1)
        np.testing.assert_array_equal(data, np.array([[-200, 0, 200]], dtype=np.int32))

    def test_valid_zero_not_invalid(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[0.0, 1.0]], dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p), invalid_value=0.0)
        import rasterio
        with rasterio.open(out) as ds:
            mask = ds.read_masks(1)
        assert mask[0, 1] == 255

    def test_partial_nodata(self, tmp_path):
        import rasterio

        g = _georef(2, 2)
        p = tmp_path / "src.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
            "nodata": -9999.0,
        }
        with rasterio.open(p, "w", **profile) as ds:
            ds.write(np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32), 1)

        expr = source(p) + 1
        out = tmp_path / "out.tif"
        write(out, expr)
        with rasterio.open(out) as ds:
            mask = ds.read_masks(1)
        assert mask[0, 0] == 255
        assert mask[0, 1] == 0
        assert mask[1, 0] == 255
        assert mask[1, 1] == 255

    def test_rotated_grid_source(self, tmp_path):
        import rasterio

        georef = GeoReference(
            projection_wkt=MOON_WKT,
            projection_proj4=MOON_PROJ4,
            affine_transform=(1000.0, 20.0, 1.0, 2000.0, -0.5, -18.0),
            width=2, height=2, pixel_size_x=20.0, pixel_size_y=-18.0, nodata=None,
        )
        p = tmp_path / "src.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": georef.projection_wkt,
            "transform": Affine.from_gdal(*georef.affine_transform),
        }
        with rasterio.open(p, "w", **profile) as ds:
            ds.write(np.ones((2, 2), dtype=np.float32), 1)

        expr = source(p) + 1
        out = tmp_path / "out.tif"
        result = write(out, expr)
        assert result.exists()

    def test_different_window_sizes(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((60, 60), dtype=np.float32))
        expr = source(p)

        for ws in [16, 32, 64, 128]:
            plan = plan_expression(expr, window_width=ws, window_height=ws)
            assert plan.total_windows > 0


class TestMemoryScaling:
    def test_estimated_memory_depends_on_window_not_area(self, tmp_path):
        sizes = [(32, 32), (128, 128), (256, 256)]
        per_window_bytes = []

        for h, w in sizes:
            p = _write_tiff(tmp_path, f"src_{h}x{w}.tif", np.ones((h, w), dtype=np.float32))
            expr = source(p) + 3
            plan = plan_expression(expr, window_width=64, window_height=64)
            per_window_bytes.append(plan.estimated_per_window_bytes)

        ratios = [pb / per_window_bytes[0] for pb in per_window_bytes]
        assert all(0.8 < r < 1.2 for r in ratios), f"Memory ratios should be ~1.0, got {ratios}"


class TestRepeatedSource:
    def test_same_source_reused_in_expression(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        s = source(p)
        expr = s + s
        result = write(tmp_path / "out.tif", expr)
        import rasterio
        with rasterio.open(result) as ds:
            data = ds.read(1)
        np.testing.assert_array_equal(data, np.full((10, 10), 2.0, dtype=np.float32))

    def test_repeated_window_read_cached(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((10, 10), dtype=np.float32))
        expr = source(p)
        with SourceWindowCache(max_windows=10) as cache:
            cache.read_values(expr, 0, 0, 0, 5, 5)
            cache.read_values(expr, 0, 0, 0, 5, 5)
            assert cache.window_count == 1


class TestCoordinateWindows:
    def test_row_indices_window(self):
        g = _georef(20, 20)
        from lunarscout.map_algebra import row_indices

        expr = row_indices(g)
        plan = plan_expression(expr, window_width=10, window_height=10)
        with SourceWindowCache() as cache:
            result = execute_windowed(plan, cache)
            assert result is not None
            assert result.values.shape == (20, 20)
            np.testing.assert_array_equal(result.values[:, 0], np.arange(20, dtype=np.int64))

    def test_column_indices_window(self):
        g = _georef(20, 20)
        from lunarscout.map_algebra import column_indices

        expr = column_indices(g)
        plan = plan_expression(expr, window_width=10, window_height=10)
        with SourceWindowCache() as cache:
            result = execute_windowed(plan, cache)
            assert result is not None
            assert result.values.shape == (20, 20)
            np.testing.assert_array_equal(result.values[0, :], np.arange(20, dtype=np.int64))


class TestCleanup:
    def test_cache_closes_after_error_in_execution(self):
        g = _georef(5, 5)
        r = ma_raster(np.ones((5, 5), dtype=np.float32), g)
        expr = r.expression()
        plan = plan_expression(expr)
        cache = SourceWindowCache()
        try:
            execute_windowed(plan, cache)
        finally:
            cache.close()
        assert cache.is_closed


def _assert_windowed_matches_compute(expression, output_path, **write_kwargs):
    expected = compute(expression)
    write(output_path, expression, **write_kwargs)
    import rasterio
    with rasterio.open(output_path) as dataset:
        actual_values = dataset.read(1)
        actual_valid = dataset.read_masks(1).astype(np.bool_)
    np.testing.assert_array_equal(actual_valid, expected.valid)
    if np.issubdtype(expected.dtype, np.bool_):
        np.testing.assert_array_equal(
            actual_values[actual_valid], expected.values[actual_valid].astype(np.uint8)
        )
    elif np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(
            actual_values[actual_valid], expected.values[actual_valid],
            rtol=1e-6, atol=1e-7,
        )
    else:
        np.testing.assert_array_equal(
            actual_values[actual_valid], expected.values[actual_valid]
        )


class TestWindowedSemanticCoverage:
    @pytest.mark.parametrize(
        "name,operation",
        [
            ("add", lambda x: x + 0.5),
            ("subtract", lambda x: 10.0 - x),
            ("multiply", lambda x: x * 3.0),
            ("divide", lambda x: x / 2.0),
            ("floor_divide", lambda x: x // 2.0),
            ("remainder", lambda x: x % 2.0),
            ("power", lambda x: x ** 2.0),
            ("minimum", lambda x: __import__("lunarscout").map_algebra.minimum(x, 2.0)),
            ("maximum", lambda x: __import__("lunarscout").map_algebra.maximum(x, 2.0)),
            ("less", lambda x: x < 2.0),
            ("less_equal", lambda x: x <= 2.0),
            ("greater", lambda x: x > 2.0),
            ("greater_equal", lambda x: x >= 2.0),
            ("equal", lambda x: x == 2.0),
            ("not_equal", lambda x: x != 2.0),
            ("hypot", lambda x: __import__("lunarscout").map_algebra.hypot(x, x)),
            ("arctan2", lambda x: __import__("lunarscout").map_algebra.arctan2(x, x)),
        ],
    )
    def test_binary_and_comparison_parity(self, tmp_path, name, operation):
        values = np.arange(1, 301 * 5 + 1, dtype=np.float32).reshape(301, 5) / 100.0
        path = _write_tiff(tmp_path, f"{name}.tif", values)
        _assert_windowed_matches_compute(
            operation(source(path)), tmp_path / f"{name}_out.tif",
            window_width=3, window_height=64,
        )

    @pytest.mark.parametrize(
        "name,operation",
        [
            ("negative", lambda ma, x: ma.negative(x)),
            ("absolute", lambda ma, x: ma.absolute(x)),
            ("sqrt", lambda ma, x: ma.sqrt(x)),
            ("square", lambda ma, x: ma.square(x)),
            ("exp", lambda ma, x: ma.exp(x)),
            ("log", lambda ma, x: ma.log(x)),
            ("log10", lambda ma, x: ma.log10(x)),
            ("sin", lambda ma, x: ma.sin(x)),
            ("cos", lambda ma, x: ma.cos(x)),
            ("tan", lambda ma, x: ma.tan(x)),
            ("arcsin", lambda ma, x: ma.arcsin(x)),
            ("arccos", lambda ma, x: ma.arccos(x)),
            ("arctan", lambda ma, x: ma.arctan(x)),
            ("floor", lambda ma, x: ma.floor(x)),
            ("ceil", lambda ma, x: ma.ceil(x)),
            ("trunc", lambda ma, x: ma.trunc(x)),
            ("round", lambda ma, x: ma.round(x)),
            ("degrees", lambda ma, x: ma.degrees(x)),
            ("radians", lambda ma, x: ma.radians(x)),
        ],
    )
    def test_unary_parity(self, tmp_path, name, operation):
        import lunarscout.map_algebra as ma
        values = np.linspace(0.1, 0.9, 301 * 5, dtype=np.float32).reshape(301, 5)
        path = _write_tiff(tmp_path, f"{name}.tif", values)
        _assert_windowed_matches_compute(
            operation(ma, source(path, units="radians")), tmp_path / f"{name}_out.tif",
            window_width=3, window_height=64,
        )

    def test_boolean_and_special_operation_parity(self, tmp_path):
        import lunarscout.map_algebra as ma
        values = np.arange(301 * 5, dtype=np.float32).reshape(301, 5)
        path = _write_tiff(tmp_path, "special.tif", values)
        raster = source(path)
        condition = (raster > 10) & (raster < 1000)
        expression = ma.where(condition, ma.clip(raster, lower=20, upper=900), ma.invalid)
        expression = ma.fill_invalid(expression, -1.0)
        expression = ma.cast(expression, np.float64)
        _assert_windowed_matches_compute(
            expression, tmp_path / "special_out.tif",
            window_width=3, window_height=64,
        )

    def test_boolean_validity_and_coalesce_parity(self, tmp_path):
        import lunarscout.map_algebra as ma
        values = np.arange(301 * 5, dtype=np.float32).reshape(301, 5)
        path = _write_tiff(tmp_path, "validity.tif", values)
        raster = source(path)
        masked = ma.set_invalid(raster, raster < 500)
        expression = ma.coalesce(masked, raster * 2.0)
        expression = ma.where(
            ma.is_valid(expression) & ~ma.is_invalid(expression),
            expression,
            ma.invalid,
        )
        expression = ma.where(
            ((raster > 10) | (raster < 2)) ^ (raster == 5),
            expression,
            raster,
        )
        _assert_windowed_matches_compute(
            expression, tmp_path / "validity_out.tif",
            window_width=3, window_height=64,
        )

    def test_classification_and_supplied_normalization_parity(self, tmp_path):
        import lunarscout.map_algebra as ma
        values = np.arange(301 * 5, dtype=np.float32).reshape(301, 5) % 4
        path = _write_tiff(tmp_path, "classify.tif", values)
        classified = ma.reclassify_values(
            source(path), {0.0: 10.0, 1.0: 20.0, 2.0: 30.0}, default=40.0,
        )
        expression = ma.normalize_minmax(classified, minimum=10.0, maximum=40.0)
        _assert_windowed_matches_compute(
            expression, tmp_path / "classify_out.tif",
            window_width=3, window_height=64,
        )

    @pytest.mark.parametrize("operation", ["ranges", "digitize", "one_hot", "standardize"])
    def test_remaining_classification_and_normalization_parity(self, tmp_path, operation):
        import lunarscout.map_algebra as ma
        values = np.arange(301 * 5, dtype=np.float32).reshape(301, 5) % 5
        path = _write_tiff(tmp_path, f"{operation}.tif", values)
        raster = source(path)
        if operation == "ranges":
            expression = ma.reclassify_ranges(
                raster, ((0.0, 2.0, 10.0), (2.0, 5.0, 20.0)),
            )
        elif operation == "digitize":
            expression = ma.digitize(raster, (1.0, 3.0), right=True)
        elif operation == "one_hot":
            expression = ma.one_hot(raster, (1.0, 3.0))[1]
        else:
            expression = ma.standardize(raster, mean=2.0, std=1.5)
        _assert_windowed_matches_compute(
            expression, tmp_path / f"{operation}_out.tif",
            window_width=3, window_height=64,
        )

    def test_isclose_and_integer_scalar_promotion_parity(self, tmp_path):
        import lunarscout.map_algebra as ma
        values = np.arange(301 * 5, dtype=np.uint8).reshape(301, 5)
        path = _write_tiff(tmp_path, "integer.tif", values, dtype="uint8")
        promoted = source(path) + 0.5
        assert promoted.dtype == compute(promoted).dtype
        expression = ma.isclose(promoted, 10.5, rtol=0.0, atol=0.01)
        _assert_windowed_matches_compute(
            expression, tmp_path / "integer_out.tif",
            window_width=3, window_height=64,
        )

    def test_round_ndigits_parity(self, tmp_path):
        import lunarscout.map_algebra as ma
        values = np.linspace(0.001, 1.999, 301 * 5, dtype=np.float32).reshape(301, 5)
        path = _write_tiff(tmp_path, "round_digits.tif", values)
        _assert_windowed_matches_compute(
            ma.round(source(path), 2), tmp_path / "round_digits_out.tif",
            window_width=3, window_height=64,
        )

    @pytest.mark.parametrize(
        "constructor_name",
        ["row_indices", "column_indices", "projected_x", "projected_y", "longitude", "latitude"],
    )
    def test_coordinate_write_parity(self, tmp_path, constructor_name):
        import lunarscout.map_algebra as ma
        grid = _georef(301, 5)
        expression = getattr(ma, constructor_name)(grid)
        _assert_windowed_matches_compute(
            expression, tmp_path / f"{constructor_name}.tif",
            window_width=3, window_height=64,
        )

    def test_explicit_source_mask_is_read_by_window(self, tmp_path):
        import rasterio
        values = np.arange(301 * 5, dtype=np.float32).reshape(301, 5)
        path = _write_tiff(tmp_path, "masked.tif", values)
        valid = np.ones(values.shape, dtype=np.uint8) * 255
        valid[50:250, 2] = 0
        with rasterio.open(path, "r+") as dataset:
            dataset.write_mask(valid)
        output = tmp_path / "masked_out.tif"
        write(
            output, source(path) + 1.0,
            window_width=3, window_height=64,
        )
        with rasterio.open(output) as dataset:
            np.testing.assert_array_equal(dataset.read_masks(1), valid)

    def test_measured_normalization_rejected_before_staging(self, tmp_path):
        import lunarscout.map_algebra as ma
        path = _write_tiff(tmp_path, "normalize.tif", np.ones((300, 5), dtype=np.float32))
        output = tmp_path / "normalize_out.tif"
        with pytest.raises(
            MapAlgebraExpressionError,
            match="requires explicit minimum and maximum",
        ):
            write(output, ma.normalize_minmax(source(path)))
        assert not output.exists()

    def test_invalid_fill_is_written_per_window(self, tmp_path):
        values = np.arange(301 * 5, dtype=np.float32).reshape(301, 5)
        path = _write_tiff(tmp_path, "invalid.tif", values)
        expression = source(path) / 0.0
        output = tmp_path / "invalid_out.tif"
        write(output, expression, invalid_value=-1234.5, window_width=3, window_height=64)
        import rasterio
        with rasterio.open(output) as dataset:
            np.testing.assert_array_equal(dataset.read(1), np.full(values.shape, -1234.5, dtype=np.float32))
            assert not dataset.read_masks(1).any()
            assert dataset.nodata == pytest.approx(-1234.5)
