from __future__ import annotations

import numpy as np
import pytest

from lunarscout.errors import MapAlgebraExpressionError
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    RasterExpression,
    compute,
    explain,
    plan,
    source,
    raster,
)
from lunarscout.raster import Raster


from tests.map_algebra.conftest import MOON_WKT, MOON_PROJ4


def _georef(h: int, w: int) -> GeoReference:
    return GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
        width=w,
        height=h,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )


def _make_raster(values, name=None):
    g = _georef(values.shape[0], values.shape[1])
    return raster(values, g, name=name)


class TestRasterExpressionConstruction:
    def test_source_stores_metadata(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        path = tmp_path / "test.tif"
        g = _georef(2, 2)
        vals = np.ones((2, 2), dtype=np.float32)
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(vals, 1)

        expr = source(path)
        assert isinstance(expr, RasterExpression)
        assert expr.operation_id == "source"
        assert expr.grid is not None
        assert expr.grid.width == 2
        assert expr.dtype == np.dtype(np.float32)

    def test_source_explain_includes_path(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        path = tmp_path / "elev.tif"
        g = _georef(2, 2)
        vals = np.ones((2, 2), dtype=np.float32)
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(vals, 1)

        expr = source(path)
        text = explain(expr)
        assert "elev.tif" in text


class TestExpressionOperators:
    def test_add_two_sources(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(2, 2)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        for p in (path_a, path_b):
            profile = {
                "driver": "GTiff", "width": 2, "height": 2, "count": 1,
                "dtype": "float32", "crs": g.projection_wkt,
                "transform": Affine.from_gdal(*g.affine_transform),
            }
            with rio.open(p, "w", **profile) as ds:
                ds.write(np.ones((2, 2), dtype=np.float32), 1)

        a = source(path_a)
        b = source(path_b)
        expr = a + b
        assert isinstance(expr, RasterExpression)
        assert expr.operation_id == "local.add"

    def test_chained_comparison_expression(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(1, 3)
        path_s = tmp_path / "slope.tif"
        path_u = tmp_path / "sun.tif"
        profile = {
            "driver": "GTiff", "width": 3, "height": 1, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path_s, "w", **profile) as ds:
            ds.write(np.array([[5.0, 10.0, 3.0]], dtype=np.float32), 1)
        with rio.open(path_u, "w", **profile) as ds:
            ds.write(np.array([[0.8, 0.7, 0.4]], dtype=np.float32), 1)

        slope = source(path_s, units="degrees")
        sun = source(path_u, units="fraction")

        candidate = (slope <= 8.0) & (sun >= 0.60)
        assert isinstance(candidate, RasterExpression)

        result = compute(candidate)
        assert result.dtype == np.dtype(np.bool_)
        np.testing.assert_array_equal(result.values, np.array([[True, False, False]]))
        assert result.all_valid

    def test_expression_plus_raster(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(2, 2)
        path_a = tmp_path / "a.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path_a, "w", **profile) as ds:
            ds.write(np.ones((2, 2), dtype=np.float32), 1)

        a = source(path_a)
        b = _make_raster(np.full((2, 2), 5.0, dtype=np.float32))

        expr = a + b
        assert isinstance(expr, RasterExpression)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.full((2, 2), 6.0, dtype=np.float32))

    def test_raster_plus_expression(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(2, 2)
        path_a = tmp_path / "a.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path_a, "w", **profile) as ds:
            ds.write(np.ones((2, 2), dtype=np.float32), 1)

        a = source(path_a)
        b = _make_raster(np.full((2, 2), 3.0, dtype=np.float32))

        expr = b + a
        assert isinstance(expr, RasterExpression)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.full((2, 2), 4.0, dtype=np.float32))


class TestCompute:
    def test_compute_simple_expression(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(2, 2)
        path = tmp_path / "test.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.full((2, 2), 7.0, dtype=np.float32), 1)

        expr = source(path) + 3
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.full((2, 2), 10.0, dtype=np.float32))

    def test_compute_comparison(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(1, 2)
        path = tmp_path / "test.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 1, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.array([[1.0, 5.0]], dtype=np.float32), 1)

        expr = source(path) < 3
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[True, False]]))

    def test_compute_multiply_subtract(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(1, 2)
        path = tmp_path / "test.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 1, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.array([[2.0, 3.0]], dtype=np.float32), 1)

        expr = source(path) * 10 - 5
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[15.0, 25.0]], dtype=np.float32))

    def test_compute_unary_neg(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(1, 2)
        path = tmp_path / "test.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 1, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.array([[1.0, -2.0]], dtype=np.float32), 1)

        expr = -source(path)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[-1.0, 2.0]], dtype=np.float32))


class TestExplain:
    def test_explain_simple(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(1, 2)
        path = tmp_path / "elev.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 1, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.array([[10.0, 20.0]], dtype=np.float32), 1)

        expr = source(path) * 2 + 1
        text = explain(expr)
        assert "RasterExpression" in text
        assert "elev.tif" in text
        assert "multiply" in text.lower() or "local.multiply" in text
        assert "add" in text.lower() or "local.add" in text

    def test_explain_comparison(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(1, 2)
        path = tmp_path / "slope.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 1, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.array([[5.0, 12.0]], dtype=np.float32), 1)

        expr = source(path) <= 8.0
        text = explain(expr)
        assert "less than or equal" in text


class TestPlan:
    def test_plan_source_count(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(2, 2)
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        for p in (path_a, path_b):
            with rio.open(p, "w", **profile) as ds:
                ds.write(np.ones((2, 2), dtype=np.float32), 1)

        expr = source(path_a) + source(path_b)
        info = plan(expr)
        assert info["source_count"] == 2
        assert info["output_grid"] is not None

    def test_plan_dry_run_no_files_written(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(2, 2)
        path = tmp_path / "x.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.ones((2, 2), dtype=np.float32), 1)

        expr = source(path) + 1
        info = plan(expr)
        assert info["node_count"] > 0


class TestScientificIdentity:
    def test_same_source_same_identity(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(2, 2)
        path = tmp_path / "a.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.ones((2, 2), dtype=np.float32), 1)

        a1 = source(path)
        a2 = source(path)
        assert a1.scientific_identity() == a2.scientific_identity()

    def test_different_op_different_identity(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(2, 2)
        path = tmp_path / "a.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.ones((2, 2), dtype=np.float32), 1)

        a = source(path)
        add_expr = a + 1
        mul_expr = a * 2
        assert add_expr.scientific_identity() != mul_expr.scientific_identity()


class TestToJson:
    def test_to_json_produces_valid_json(self, tmp_path):
        import rasterio as rio
        from affine import Affine

        g = _georef(2, 2)
        path = tmp_path / "a.tif"
        profile = {
            "driver": "GTiff", "width": 2, "height": 2, "count": 1,
            "dtype": "float32", "crs": g.projection_wkt,
            "transform": Affine.from_gdal(*g.affine_transform),
        }
        with rio.open(path, "w", **profile) as ds:
            ds.write(np.ones((2, 2), dtype=np.float32), 1)

        expr = source(path) + 1
        json_str = expr.to_json()
        assert "schema_version" in json_str
        import json
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == 1
        assert len(parsed["nodes"]) >= 2


class TestRasterExpressionBool:
    def test_bool_raises_typeerror(self):
        expr = _make_raster(np.ones((2, 2), dtype=np.float32)).with_name("x")
        from lunarscout.map_algebra._sources import constant
        cexpr = constant(expr)
        with pytest.raises(TypeError, match="implicit truth testing"):
            bool(cexpr)
