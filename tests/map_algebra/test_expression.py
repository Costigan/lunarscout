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
    raster as ma_raster,
)
from lunarscout.raster import Raster
from tests.map_algebra.conftest import MOON_WKT, MOON_PROJ4


def _georef(h: int, w: int) -> GeoReference:
    return GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
        width=w, height=h,
        pixel_size_x=20.0, pixel_size_y=-20.0,
        nodata=None,
    )


def _make_raster(values, name=None):
    g = _georef(values.shape[0], values.shape[1])
    return ma_raster(values, g, name=name)


def _write_tiff(tmp_path, name, values, dtype="float32"):
    import rasterio as rio
    from affine import Affine

    h, w = values.shape
    g = _georef(h, w)
    path = tmp_path / name
    profile = {
        "driver": "GTiff", "width": w, "height": h, "count": 1,
        "dtype": dtype, "crs": g.projection_wkt,
        "transform": Affine.from_gdal(*g.affine_transform),
    }
    with rio.open(path, "w", **profile) as ds:
        ds.write(values.astype(dtype), 1)
    return path


class TestSealedConstructor:
    def test_direct_construction_raises(self):
        with pytest.raises(MapAlgebraExpressionError, match="cannot be constructed directly"):
            RasterExpression(
                _node_id="x", _operation_id="local.add",
                _operands=(), _params=(),
                _inferred_grid=None, _inferred_dtype=None,
                _inferred_units=None, _halo=0,
                _sealed=None,
            )


class TestSource:
    def test_source_metadata_only(self, tmp_path):
        path = _write_tiff(tmp_path, "test.tif", np.ones((2, 2), dtype=np.float32))
        expr = source(path)
        assert isinstance(expr, RasterExpression)
        assert expr.operation_id == "source"
        assert expr.grid is not None
        assert expr.grid.width == 2
        assert expr.dtype == np.dtype(np.float32)

    def test_source_missing_raises_structured(self, tmp_path):
        from lunarscout.errors import GeoTiffOpenError
        with pytest.raises(GeoTiffOpenError, match="does not exist"):
            source(tmp_path / "nonexistent.tif")

    def test_source_sha256_identity(self, tmp_path):
        path = _write_tiff(tmp_path, "test.tif", np.ones((2, 2), dtype=np.float32))
        expr = source(path, identity="sha256")
        params = expr._params_dict
        assert "sha256" in params


class TestExpressionOperators:
    def test_add_two_sources(self, tmp_path):
        path_a = _write_tiff(tmp_path, "a.tif", np.ones((2, 2), dtype=np.float32))
        path_b = _write_tiff(tmp_path, "b.tif", np.ones((2, 2), dtype=np.float32))
        a = source(path_a)
        b = source(path_b)
        expr = a + b
        assert isinstance(expr, RasterExpression)
        assert expr.operation_id == "local.add"

    def test_chained_comparison(self, tmp_path):
        path_s = _write_tiff(tmp_path, "slope.tif", np.array([[5.0, 10.0, 3.0]], dtype=np.float32))
        path_u = _write_tiff(tmp_path, "sun.tif", np.array([[0.8, 0.7, 0.4]], dtype=np.float32))

        slope = source(path_s, units="degrees")
        sun = source(path_u, units="fraction")
        candidate = (slope <= 8.0) & (sun >= 0.60)

        assert isinstance(candidate, RasterExpression)
        result = compute(candidate)
        assert result.dtype == np.dtype(np.bool_)
        np.testing.assert_array_equal(result.values, np.array([[True, False, False]]))
        assert result.units is None

    def test_expression_plus_raster(self, tmp_path):
        path_a = _write_tiff(tmp_path, "a.tif", np.ones((2, 2), dtype=np.float32))
        a = source(path_a)
        b = _make_raster(np.full((2, 2), 5.0, dtype=np.float32))
        expr = a + b
        assert isinstance(expr, RasterExpression)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.full((2, 2), 6.0, dtype=np.float32))

    def test_raster_plus_expression(self, tmp_path):
        path_a = _write_tiff(tmp_path, "a.tif", np.ones((2, 2), dtype=np.float32))
        a = source(path_a)
        b = _make_raster(np.full((2, 2), 3.0, dtype=np.float32))
        expr = b + a
        assert isinstance(expr, RasterExpression)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.full((2, 2), 4.0, dtype=np.float32))

    def test_scalar_left_subtract(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.array([[1.0, 2.0]], dtype=np.float32))
        expr = 10 - source(path)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[9.0, 8.0]], dtype=np.float32))

    def test_scalar_left_divide(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.array([[2.0, 4.0]], dtype=np.float32))
        expr = 100 / source(path)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[50.0, 25.0]], dtype=np.float32))

    def test_source_plus_source_plus_scalar_dag(self, tmp_path):
        path = _write_tiff(tmp_path, "s.tif", np.array([[5.0]], dtype=np.float32))
        s = source(path)
        expr = (s + 1) + (s * 2)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[16.0]]))


class TestCompute:
    def test_simple_arithmetic(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.full((2, 2), 7.0, dtype=np.float32))
        expr = source(path) + 3
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.full((2, 2), 10.0, dtype=np.float32))

    def test_comparison(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.array([[1.0, 5.0]], dtype=np.float32))
        expr = source(path) < 3
        result = compute(expr)
        assert result.dtype == np.dtype(np.bool_)
        np.testing.assert_array_equal(result.values, np.array([[True, False]]))

    def test_multiply_subtract_chain(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.array([[2.0, 3.0]], dtype=np.float32))
        expr = source(path) * 10 - 5
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[15.0, 25.0]], dtype=np.float32))

    def test_unary_neg(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.array([[1.0, -2.0]], dtype=np.float32))
        expr = -source(path)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[-1.0, 2.0]], dtype=np.float32))

    def test_sqrt_expression(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.array([[4.0, 9.0]], dtype=np.float32))
        from lunarscout.map_algebra import sqrt
        expr = sqrt(source(path))
        result = compute(expr)
        np.testing.assert_allclose(result.values, np.array([[2.0, 3.0]], dtype=np.float32))

    def test_logical_ops(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.array([[1.0, 5.0, 0.0]], dtype=np.float32))
        s = source(path)
        expr = (s > 0) & (s < 4)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[True, False, False]]))


class TestExplain:
    def test_includes_path(self, tmp_path):
        path = _write_tiff(tmp_path, "elev.tif", np.ones((2, 2), dtype=np.float32))
        text = explain(source(path))
        assert "elev.tif" in text

    def test_chain(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.ones((1, 2), dtype=np.float32))
        text = explain(source(path) * 2 + 1)
        assert "multiply" in text.lower()
        assert "add" in text.lower()


class TestPlan:
    def test_counts(self, tmp_path):
        path_a = _write_tiff(tmp_path, "a.tif", np.ones((2, 2), dtype=np.float32))
        path_b = _write_tiff(tmp_path, "b.tif", np.ones((2, 2), dtype=np.float32))
        info = plan(source(path_a) + source(path_b))
        assert info["source_count"] == 2
        assert info["output_grid"] is not None

    def test_dry_run_no_writes(self, tmp_path):
        path = _write_tiff(tmp_path, "x.tif", np.ones((2, 2), dtype=np.float32))
        info = plan(source(path) + 1)
        assert info["node_count"] > 0


class TestIdentity:
    def test_different_values_different_identity(self):
        a = _make_raster(np.array([[1.0, 1.0]], dtype=np.float32))
        b = _make_raster(np.array([[9.0, 9.0]], dtype=np.float32))
        from lunarscout.map_algebra._sources import constant
        id_a = constant(a).scientific_identity()
        id_b = constant(b).scientific_identity()
        assert id_a != id_b  # P1#6 fix: distinct values produce distinct hashes

    def test_same_source_same_identity(self, tmp_path):
        path = _write_tiff(tmp_path, "a.tif", np.ones((2, 2), dtype=np.float32))
        assert source(path).scientific_identity() == source(path).scientific_identity()

    def test_json_includes_crs(self, tmp_path):
        path = _write_tiff(tmp_path, "a.tif", np.ones((2, 2), dtype=np.float32))
        j = source(path).to_json()
        assert "crs_wkt" in j

    def test_json_valid(self, tmp_path):
        path = _write_tiff(tmp_path, "a.tif", np.ones((2, 2), dtype=np.float32))
        import json
        parsed = json.loads(source(path).to_json())
        assert parsed["schema_version"] == 2


class TestRasterExpressionBool:
    def test_bool_raises(self):
        r = _make_raster(np.ones((2, 2), dtype=np.float32))
        from lunarscout.map_algebra._sources import constant
        with pytest.raises(TypeError, match="implicit truth testing"):
            bool(constant(r))
