from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from affine import Affine

from lunarscout.errors import (
    MapAlgebraStorageError,
    OutputExistsError,
)
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    RasterExpression,
    compute,
    source,
    write,
    raster as ma_raster,
)
from lunarscout.raster import Raster
from tests.map_algebra.conftest import MOON_WKT, MOON_PROJ4


def _georef(h: int, w: int) -> GeoReference:
    return GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
        width=w, height=h, pixel_size_x=20.0, pixel_size_y=-20.0, nodata=None,
    )


def _write_tiff(tmp_path, name, values, dtype="float32"):
    h, w = values.shape
    g = _georef(h, w)
    path = tmp_path / name
    profile = {
        "driver": "GTiff", "width": w, "height": h, "count": 1,
        "dtype": dtype, "crs": g.projection_wkt,
        "transform": Affine.from_gdal(*g.affine_transform),
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(values.astype(dtype), 1)
    return path


class TestWrite:
    def test_basic_write(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.full((2, 2), 7.0, dtype=np.float32))
        expr = source(p) + 3
        out = tmp_path / "out.tif"
        result = write(out, expr)
        assert result == out
        assert out.exists()
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, np.full((2, 2), 10.0, dtype=np.float32))

    def test_overwrite_protection(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1, overwrite=False)
        with pytest.raises(OutputExistsError):
            write(out, source(p) + 2, overwrite=False)

    def test_overwrite_allowed(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1)
        write(out, source(p) + 5, overwrite=True)
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, np.full((2, 2), 6.0, dtype=np.float32))

    def test_dtype_override(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[1.5, 2.5]], dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p), dtype="int32")
        with rasterio.open(out) as ds:
            data = ds.read(1)
            assert data.dtype == np.int32
            np.testing.assert_array_equal(data, np.array([[1, 2]], dtype=np.int32))

    def test_invalid_value_float(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[0.0, 1.0]], dtype=np.float32))
        src_expr = source(p)
        from lunarscout.map_algebra import where, invalid
        cond = compute(src_expr > 0.5)
        expr = where(cond, compute(src_expr), invalid)
        out = tmp_path / "out.tif"
        write(out, expr.expression(), invalid_value=-9999.0)
        with rasterio.open(out) as ds:
            data = ds.read(1)
            assert data[0, 0] == -9999.0
            assert data[0, 1] == 1.0

    def test_invalid_value_uint8(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[0, 255]], dtype=np.uint8))
        src_expr = source(p)
        from lunarscout.map_algebra import where, invalid
        cond = compute(src_expr > 128)
        expr = where(cond, compute(src_expr), invalid)
        out = tmp_path / "out.tif"
        write(out, expr.expression(), invalid_value=0)
        with rasterio.open(out) as ds:
            data = ds.read(1)
            assert data[0, 0] == 0
            assert data[0, 1] == 255

    def test_manifest_written(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1)
        mf = tmp_path / "out.tif.manifest.json"
        assert mf.exists()
        data = json.loads(mf.read_text())
        assert "scientific_identity" in data
        assert "grid" in data
        assert "dtype" in data

    def test_restart_skips(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        write(out, expr)
        out.unlink()
        result2 = write(out, expr)
        assert result2 == out
        assert out.exists()

    def test_restart_mismatch_rebuilds(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr_a = source(p) + 1
        write(out, expr_a)
        out.unlink()
        expr_b = source(p) + 2
        write(out, expr_b)
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, np.full((2, 2), 3.0, dtype=np.float32))

    def test_start_fresh(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        write(out, expr)
        out.unlink()
        write(out, expr, start_fresh=True)
        assert out.exists()

    def test_compound_expression_write(self, tmp_path):
        ps = _write_tiff(tmp_path, "slope.tif", np.array([[5.0, 10.0, 3.0]], dtype=np.float32))
        pu = _write_tiff(tmp_path, "sun.tif", np.array([[0.8, 0.7, 0.4]], dtype=np.float32))
        slope = source(ps, units="degrees")
        sun = source(pu, units="fraction")
        candidate = (slope <= 8.0) & (sun >= 0.60)
        out = tmp_path / "candidate.tif"
        write(out, candidate, invalid_value=0)
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, np.array([[1, 0, 0]], dtype=np.uint8))

    def test_empty_expression_raises(self, tmp_path):
        r = ma_raster(np.ones((2, 2), dtype=np.float32), _georef(2, 2))
        with pytest.raises(MapAlgebraStorageError, match="Unsupported output dtype"):
            write(tmp_path / "out.tif", r.expression(), dtype="complex64")

    def test_writes_to_nested_directory(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "sub" / "deep" / "out.tif"
        result = write(out, source(p) + 1)
        assert result.exists()


class TestRasterExpressionMethod:
    def test_raster_expression_returns_expression(self, tmp_path):
        r = ma_raster(np.ones((2, 2), dtype=np.float32), _georef(2, 2))
        expr = r.expression()
        assert isinstance(expr, RasterExpression)
        assert expr.operation_id == "constant"

    def test_raster_expression_compute_roundtrip(self, tmp_path):
        r = ma_raster(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), _georef(2, 2))
        expr = r.expression() + 10
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32))

    def test_raster_expression_mixes_with_eager(self, tmp_path):
        r = ma_raster(np.array([[5.0]], dtype=np.float32), _georef(1, 1))
        p = _write_tiff(tmp_path, "src.tif", np.array([[3.0]], dtype=np.float32))
        expr = r.expression() + source(p)
        result = compute(expr)
        np.testing.assert_array_equal(result.values, np.array([[8.0]], dtype=np.float32))
