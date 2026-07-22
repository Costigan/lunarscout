from __future__ import annotations

import json
from pathlib import Path

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


def _read_mask(path):
    with rasterio.open(path) as ds:
        return ds.read_masks(1)


class TestWrite:
    def test_basic_write(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.full((2, 2), 7.0, dtype=np.float32))
        out = tmp_path / "out.tif"
        result = write(out, source(p) + 3)
        assert result == out
        assert out.exists()
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, np.full((2, 2), 10.0, dtype=np.float32))

    def test_gdal_mask_written(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32))
        src_expr = source(p)
        from lunarscout.map_algebra import where, invalid
        cond = compute(src_expr > 1.5)
        expr = where(cond, compute(src_expr), invalid)
        out = tmp_path / "out.tif"
        write(out, expr.expression(), invalid_value=-9999.0)
        mask = _read_mask(out)
        assert mask.shape == (2, 2)
        assert mask[0, 0] == 0
        assert mask[1, 0] == 255
        assert mask[1, 1] == 255

    def test_valid_zero_not_confused_with_nodata(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[0, 255]], dtype=np.uint8))
        src_expr = source(p)
        out = tmp_path / "out.tif"
        write(out, src_expr, invalid_value=0)
        mask = _read_mask(out)
        assert mask[0, 0] == 255
        assert mask[0, 1] == 255
        with rasterio.open(out) as ds:
            data = ds.read(1)
            assert data[0, 0] == 0
            assert data[0, 1] == 255

    def test_overwrite_protection(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1)
        with pytest.raises(OutputExistsError):
            write(out, source(p) + 2)

    def test_overwrite_allowed(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1)
        write(out, source(p) + 5, overwrite=True)
        with rasterio.open(out) as ds:
            data = ds.read(1)
            np.testing.assert_array_equal(data, np.full((2, 2), 6.0, dtype=np.float32))

    def test_failed_manifest_publish_restores_previous_output(self, tmp_path, monkeypatch):
        import lunarscout.map_algebra._writer as writer_module

        source_path = _write_tiff(
            tmp_path, "restore_source.tif", np.ones((2, 2), dtype=np.float32)
        )
        output = tmp_path / "restore.tif"
        write(output, source(source_path) + 1)
        manifest = output.with_suffix(output.suffix + ".manifest.json")
        previous_manifest = manifest.read_bytes()

        real_replace = writer_module.os.replace
        injected = [False]
        staging_manifest = writer_module._staging_manifest_path(output.resolve())

        def fail_manifest_publish(src, dst):
            if Path(src) == staging_manifest and Path(dst) == manifest and not injected[0]:
                injected[0] = True
                raise OSError("injected manifest publication failure")
            return real_replace(src, dst)

        monkeypatch.setattr(writer_module.os, "replace", fail_manifest_publish)
        with pytest.raises(OSError, match="injected"):
            write(output, source(source_path) + 5, overwrite=True)

        with rasterio.open(output) as dataset:
            np.testing.assert_array_equal(
                dataset.read(1), np.full((2, 2), 2.0, dtype=np.float32)
            )
        assert manifest.read_bytes() == previous_manifest

        monkeypatch.setattr(writer_module.os, "replace", real_replace)
        write(output, source(source_path) + 5, overwrite=True)
        with rasterio.open(output) as dataset:
            np.testing.assert_array_equal(
                dataset.read(1), np.full((2, 2), 6.0, dtype=np.float32)
            )

    def test_failed_output_backup_preserves_previous_output(self, tmp_path, monkeypatch):
        import lunarscout.map_algebra._writer as writer_module

        source_path = _write_tiff(
            tmp_path, "backup_source.tif", np.ones((2, 2), dtype=np.float32),
        )
        output = tmp_path / "backup.tif"
        write(output, source(source_path) + 1)
        manifest = output.with_suffix(output.suffix + ".manifest.json")
        previous_manifest = manifest.read_bytes()
        real_replace = writer_module.os.replace

        def fail_output_backup(src, dst):
            if Path(src) == output.resolve():
                raise OSError("injected output backup failure")
            return real_replace(src, dst)

        monkeypatch.setattr(writer_module.os, "replace", fail_output_backup)
        with pytest.raises(OSError, match="backup failure"):
            write(output, source(source_path) + 5, overwrite=True)

        with rasterio.open(output) as dataset:
            np.testing.assert_array_equal(
                dataset.read(1), np.full((2, 2), 2.0, dtype=np.float32),
            )
        assert manifest.read_bytes() == previous_manifest

    def test_dtype_override(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.array([[1.5, 2.5]], dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p), dtype="float64")
        with rasterio.open(out) as ds:
            data = ds.read(1)
            assert data.dtype == np.float64

    def test_manifest_written(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1)
        mf = tmp_path / "out.tif.manifest.json"
        assert mf.exists()
        data = json.loads(mf.read_text())
        assert "scientific_identity" in data
        assert "output_dtype" in data
        assert "invalid_fill" in data

    def test_restart_id_match_output_exists_skips(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        expr = source(p) + 1
        write(out, expr)
        mtime_before = out.stat().st_mtime
        result = write(out, expr)
        assert result == out
        assert out.stat().st_mtime == mtime_before

    def test_restart_id_mismatch_output_absent_rebuilds(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1)
        out.unlink()
        write(out, source(p) + 1)
        assert out.exists()

    def test_start_fresh(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        out = tmp_path / "out.tif"
        write(out, source(p) + 1)
        out.unlink()
        write(out, source(p) + 1, start_fresh=True)
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

    def test_unsupported_dtype_preflight(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        with pytest.raises(MapAlgebraStorageError, match="Unsupported output dtype"):
            write(tmp_path / "out.tif", source(p), dtype="complex64")

    @pytest.mark.parametrize(
        "invalid_value", [-1, 256, 1.5, True, "5", np.inf, -np.inf],
    )
    def test_invalid_fill_preflight_is_exact_and_leaves_no_output_artifacts(
        self, tmp_path, invalid_value,
    ):
        p = _write_tiff(
            tmp_path, "fill-src.tif", np.array([[1, 2]], dtype=np.uint8),
            dtype="uint8",
        )
        out = tmp_path / "invalid-fill.tif"
        with pytest.raises(MapAlgebraStorageError) as error:
            write(out, source(p), invalid_value=invalid_value)
        assert error.value.code == "map_algebra_unrepresentable_invalid_value"
        assert not out.exists()
        assert not any(
            path.name.startswith(".invalid-fill.tif")
            for path in tmp_path.iterdir()
        )

    def test_float32_invalid_fill_must_be_exactly_representable(self, tmp_path):
        p = _write_tiff(
            tmp_path, "float-fill-src.tif",
            np.array([[1.0, 2.0]], dtype=np.float32),
        )
        with pytest.raises(MapAlgebraStorageError) as error:
            write(tmp_path / "float-fill.tif", source(p), invalid_value=0.1)
        assert error.value.code == "map_algebra_unrepresentable_invalid_value"

    def test_explicit_float32_invalid_fill_round_trips(self, tmp_path):
        raster_value = ma_raster(
            np.array([[1.0, 9.0]], dtype=np.float32),
            _georef(1, 2),
            valid=np.array([[True, False]], dtype=np.bool_),
        )
        fill = np.float32(0.1)
        out = write(
            tmp_path / "typed-float-fill.tif",
            raster_value.expression(),
            invalid_value=fill,
        )
        with rasterio.open(out) as dataset:
            values = dataset.read(1)
            assert values[0, 1] == fill
            np.testing.assert_array_equal(
                dataset.read_masks(1) > 0,
                np.array([[True, False]], dtype=np.bool_),
            )

    def test_invalid_fill_preflight_preserves_existing_destination(self, tmp_path):
        p = _write_tiff(
            tmp_path, "preserve-fill-src.tif",
            np.array([[1, 2]], dtype=np.uint8),
            dtype="uint8",
        )
        out = tmp_path / "preserve-invalid-fill.tif"
        write(out, source(p), invalid_value=0)
        before = out.read_bytes()
        with pytest.raises(MapAlgebraStorageError) as error:
            write(out, source(p), overwrite=True, invalid_value=300)
        assert error.value.code == "map_algebra_unrepresentable_invalid_value"
        assert out.read_bytes() == before

    def test_uint64_invalid_fill_beyond_float_exact_range_round_trips(
        self, tmp_path,
    ):
        fill = 2**63 + 123
        raster_value = ma_raster(
            np.array([[7, 99]], dtype=np.uint64),
            _georef(1, 2),
            valid=np.array([[True, False]], dtype=np.bool_),
        )
        out = write(
            tmp_path / "uint64-fill.tif",
            raster_value.expression(),
            invalid_value=fill,
            window_width=1,
            window_height=1,
        )
        with rasterio.open(out) as dataset:
            values = dataset.read(1)
            assert values.dtype == np.dtype(np.uint64)
            assert int(values[0, 0]) == 7
            assert int(values[0, 1]) == fill
            assert dataset.tags(1)["LUNARSCOUT_NODATA_VALUE"] == str(fill)
            np.testing.assert_array_equal(
                dataset.read_masks(1) > 0,
                np.array([[True, False]], dtype=np.bool_),
            )

    def test_lossy_dtype_rejected(self, tmp_path):
        p = _write_tiff(tmp_path, "src.tif", np.ones((2, 2), dtype=np.float32))
        with pytest.raises(MapAlgebraStorageError, match="safely convert"):
            write(tmp_path / "out.tif", source(p), dtype="int32")


class TestRasterExpressionMethod:
    def test_raster_expression_returns_expression(self):
        r = ma_raster(np.ones((2, 2), dtype=np.float32), _georef(2, 2))
        expr = r.expression()
        assert isinstance(expr, RasterExpression)
        assert expr.operation_id == "constant"

    def test_raster_expression_compute_roundtrip(self):
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
