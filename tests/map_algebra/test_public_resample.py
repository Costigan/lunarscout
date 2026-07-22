from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from dataclasses import replace

from lunarscout.georeference import GeoReference
from lunarscout.errors import AlignmentError
from lunarscout.raster import Raster
from lunarscout.map_algebra import (
    align,
    compute,
    raster as ma_raster,
    resample_to,
    source,
    write,
)
from lunarscout.map_algebra._planner import plan_expression
from lunarscout.map_algebra._windowed import execute_windowed
from lunarscout.map_algebra._windows import SourceWindowCache
from tests.map_algebra.conftest import MOON_PROJ4, MOON_WKT


def _grid(
    width, height,
    *,
    affine=None,
    wkt=None,
    proj4=None,
):
    transform = affine or (1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0)
    return GeoReference(
        projection_wkt=wkt or MOON_WKT,
        projection_proj4=proj4 or MOON_PROJ4,
        affine_transform=transform,
        width=width,
        height=height,
        pixel_size_x=transform[1],
        pixel_size_y=transform[5],
        nodata=None,
    )


def _write_source(tmp_path, name, values, grid, *, valid=None):
    import rasterio

    path = tmp_path / name
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": values.dtype,
        "crs": grid.projection_wkt,
        "transform": Affine.from_gdal(*grid.affine_transform),
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1)
        if valid is not None:
            dataset.write_mask(np.asarray(valid, dtype=np.uint8) * 255)
    return path


def _execute_windowed(expression, *, window_width, window_height):
    plan = plan_expression(
        expression, window_width=window_width, window_height=window_height,
    )
    with SourceWindowCache() as cache:
        result = execute_windowed(plan, cache)
    assert result is not None
    return plan, result


def _assert_parity(actual, expected, *, atol=1e-6):
    np.testing.assert_array_equal(actual.valid, expected.valid)
    if expected.dtype.kind == "f":
        np.testing.assert_allclose(
            actual.values[actual.valid],
            expected.values[expected.valid],
            rtol=1e-6,
            atol=atol,
        )
    else:
        np.testing.assert_array_equal(
            actual.values[actual.valid], expected.values[actual.valid],
        )


# ---------------------------------------------------------------------------
# Eager resample / align tests
# ---------------------------------------------------------------------------


class TestPublicResampleEager:
    def _make_raster(self, width=15, height=11):
        grid = _grid(width, height)
        rows, cols = np.indices((height, width), dtype=np.float32)
        values = np.sin(cols / 4.0) + np.cos(rows / 5.0) + cols * 0.03
        raster = ma_raster(values, grid)
        return raster, grid, values

    def _dst_grid(self, width=23, height=17):
        return _grid(width, height, affine=(1013.0, 14.0, 0.0, 1987.0, 0.0, -14.0))

    def test_resample_to_raster_returns_raster(self):
        raster, _, _ = self._make_raster()
        dst = self._dst_grid()
        result = resample_to(raster, dst)
        assert isinstance(result, Raster)
        assert result.values.shape == (dst.height, dst.width)

    def test_resample_to_default_nearest(self):
        raster, _, _ = self._make_raster()
        dst = self._dst_grid()
        result = resample_to(raster, dst)
        assert result.georef.width == dst.width
        assert result.georef.height == dst.height

    def test_resample_to_finer_grid(self):
        raster, _, _ = self._make_raster()
        dst = _grid(30, 24, affine=(1000.0, 10.0, 0.0, 2000.0, 0.0, -10.0))
        result = resample_to(raster, dst, resampling="nearest")
        assert result.values.shape == (24, 30)

    def test_resample_to_coarser_grid(self):
        raster, _, _ = self._make_raster()
        dst = _grid(7, 5, affine=(1000.0, 40.0, 0.0, 2000.0, 0.0, -40.0))
        result = resample_to(raster, dst, resampling="bilinear")
        assert result.values.shape == (5, 7)

    def test_resample_to_shifted_origin(self):
        raster, _, _ = self._make_raster()
        dst = _grid(10, 8, affine=(1050.0, 20.0, 0.0, 1950.0, 0.0, -20.0))
        result = resample_to(raster, dst)
        assert result.values.shape == (8, 10)

    @pytest.mark.parametrize("resampling", ["nearest", "bilinear", "cubic", "lanczos", "average"])
    def test_resampling_algorithm_names(self, resampling):
        raster, _, _ = self._make_raster()
        dst = self._dst_grid()
        result = resample_to(raster, dst, resampling=resampling)
        assert result.valid.any()

    def test_align_returns_raster(self):
        raster, _, _ = self._make_raster()
        dst = self._dst_grid()
        result = align(raster, to=dst)
        assert result.values.shape == (dst.height, dst.width)

    def test_align_rejects_expression(self):
        raster, _, _ = self._make_raster()
        dst = self._dst_grid()
        with pytest.raises(AlignmentError, match="ma.align") as error:
            align(raster.expression(), to=dst)  # type: ignore[arg-type]
        assert error.value.code == "alignment_expression_requires_resample_to"

    def test_align_rejects_non_raster(self):
        dst = self._dst_grid()
        with pytest.raises(TypeError):
            align(np.ones((3, 3)), to=dst)  # type: ignore[arg-type]

    def test_resample_to_rejects_non_raster(self):
        dst = self._dst_grid()
        with pytest.raises(TypeError):
            resample_to(np.ones((3, 3)), dst)  # type: ignore[arg-type]

    def test_resample_to_preserves_validity(self):
        raster, _, _ = self._make_raster()
        dst = _grid(30, 24, affine=(1000.0, 10.0, 0.0, 2000.0, 0.0, -10.0))
        result = resample_to(raster, dst)
        assert result.valid.any()

    def test_resample_to_output_dtype(self):
        raster, _, _ = self._make_raster()
        dst = self._dst_grid()
        result = resample_to(raster, dst, output_dtype=np.float64)
        assert result.dtype == np.float64

    def test_destination_nodata_must_fit_output_dtype(self):
        raster, _, _ = self._make_raster()
        dst = self._dst_grid().with_nodata(-1)
        from lunarscout.errors import GeoTiffMetadataError

        with pytest.raises(GeoTiffMetadataError) as error:
            resample_to(
                raster, dst, output_dtype=np.uint8, allow_unsafe=True,
            )
        assert error.value.code == "geotiff_unrepresentable_nodata"

    def test_resample_to_coverage_threshold(self):
        raster, _, _ = self._make_raster()
        dst = self._dst_grid()
        result = resample_to(raster, dst, validity_coverage_threshold=0.5)
        assert result.valid.any()

    def test_zero_coverage_threshold_does_not_validate_outside_source(self):
        raster, _, _ = self._make_raster(width=4, height=3)
        dst = _grid(
            5, 4,
            affine=(1_000_000.0, 20.0, 0.0, 1_000_000.0, 0.0, -20.0),
        )
        result = resample_to(
            raster, dst, validity_coverage_threshold=0.0,
        )
        assert not result.valid.any()

    def test_align_output_nodata_contract(self):
        raster, _, _ = self._make_raster()
        raster = Raster(
            raster.values,
            raster.georef.with_nodata(-99.0),
            valid=raster.valid,
        )
        dst = self._dst_grid()

        automatic = align(raster, to=dst)
        explicit = align(raster, to=dst, output_nodata=-123.0)
        disabled = align(raster, to=dst, output_nodata=None)

        assert automatic.georef.nodata == -99.0
        assert explicit.georef.nodata == -123.0
        assert disabled.georef.nodata is None

    def test_align_expression_has_structured_error(self):
        raster, _, _ = self._make_raster()
        with pytest.raises(AlignmentError) as error:
            align(raster.expression(), to=self._dst_grid())  # type: ignore[arg-type]
        assert error.value.code == "alignment_expression_requires_resample_to"


# ---------------------------------------------------------------------------
# Expression resample tests
# ---------------------------------------------------------------------------


class TestPublicResampleExpression:
    def _make_source(self, tmp_path, *, width=15, height=11):
        grid = _grid(width, height)
        rows, cols = np.indices((height, width), dtype=np.float32)
        values = np.sin(cols / 4.0) + np.cos(rows / 5.0) + cols * 0.03
        path = _write_source(tmp_path, "src.tif", values, grid)
        return source(path), grid

    def _dst_grid(self, width=23, height=17):
        return _grid(width, height, affine=(1013.0, 14.0, 0.0, 1987.0, 0.0, -14.0))

    def test_resample_to_expression_returns_expression(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        dst = self._dst_grid()
        result = resample_to(expr, dst)
        from lunarscout.map_algebra._model import RasterExpression
        assert isinstance(result, RasterExpression)
        assert result.operation_id == "alignment.resample_to"

    def test_expression_compute_parity(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        dst = self._dst_grid()
        eager = compute(resample_to(expr, dst))
        node = resample_to(expr, dst)
        computed = compute(node)
        _assert_parity(computed, eager)

    @pytest.mark.parametrize("resampling", ["nearest", "bilinear", "cubic", "lanczos", "average"])
    def test_expression_write_parity(self, tmp_path, resampling):
        expr, _ = self._make_source(tmp_path)
        dst = self._dst_grid(13, 11)
        resampled_expr = resample_to(expr, dst, resampling=resampling)
        expected = compute(resampled_expr)
        _, actual = _execute_windowed(resampled_expr, window_width=4, window_height=4)
        _assert_parity(actual, expected, atol=2e-6)

    def test_public_writer_resample_parity(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        resampled_expr = resample_to(
            expr, self._dst_grid(13, 11), resampling="bilinear",
        )
        expected = compute(resampled_expr)
        output = tmp_path / "public_resample.tif"

        write(output, resampled_expr, window_width=4, window_height=3)

        from lunarscout.map_algebra import read
        actual = read(output)
        _assert_parity(actual, expected, atol=2e-6)

    def test_partial_coverage(self, tmp_path):
        source_grid = _grid(8, 6)
        rows, cols = np.indices((6, 8), dtype=np.float32)
        values = cols * 2.0 + rows
        path = _write_source(tmp_path, "partial.tif", values, source_grid)
        dst = _grid(16, 12, affine=(1100.0, 15.0, 0.0, 2050.0, 0.0, -15.0))
        result = compute(resample_to(source(path), dst, resampling="bilinear"))
        assert result.valid.any()

    def test_no_coverage(self, tmp_path):
        source_grid = _grid(4, 3)
        path = _write_source(tmp_path, "no_overlap.tif", np.ones((3, 4), dtype=np.float32), source_grid)
        dst = _grid(5, 4, affine=(1_000_000.0, 20.0, 0.0, 1_000_000.0, 0.0, -20.0))
        result = compute(resample_to(source(path), dst))
        assert not result.valid.any()

    def test_rotated_grids(self, tmp_path):
        source_grid = _grid(15, 11, affine=(1000.0, 20.0, 3.0, 2000.0, 2.0, -20.0))
        dst = _grid(17, 13, affine=(1011.0, 17.0, -1.5, 1991.0, 1.0, -17.0))
        rows, cols = np.indices((11, 15), dtype=np.float32)
        path = _write_source(tmp_path, "rot.tif", cols * 0.75 + rows * 1.25, source_grid)
        result = compute(resample_to(source(path), dst, resampling="bilinear"))
        assert result.valid.any()

    def test_differing_crs(self, tmp_path):
        from pyproj import CRS
        projected = CRS.from_epsg(3857)
        geographic = CRS.from_epsg(4326)
        src_grid = GeoReference(
            projection_wkt=projected.to_wkt(),
            projection_proj4=projected.to_proj4(),
            affine_transform=(-2000.0, 200.0, 0.0, 2000.0, 0.0, -200.0),
            width=12, height=12,
            pixel_size_x=200.0, pixel_size_y=-200.0, nodata=None,
        )
        dst_grid = GeoReference(
            projection_wkt=geographic.to_wkt(),
            projection_proj4=geographic.to_proj4(),
            affine_transform=(-0.018, 0.0015, 0.0, 0.018, 0.0, -0.0015),
            width=14, height=14,
            pixel_size_x=0.0015, pixel_size_y=-0.0015, nodata=None,
        )
        rows, cols = np.indices((12, 12), dtype=np.float32)
        path = _write_source(tmp_path, "crs.tif", cols + rows * 2.0, src_grid)
        result = compute(resample_to(source(path), dst_grid, resampling="bilinear"))
        assert result.valid.any()

    def test_exact_uint64_nearest(self, tmp_path):
        source_grid = _grid(3, 2)
        dst = _grid(6, 4, affine=(1000.0, 10.0, 0.0, 2000.0, 0.0, -10.0))
        values = np.asarray(
            [[2**63 + 11, 2**63 + 12, 2**63 + 13],
             [2**63 + 21, 2**63 + 22, 2**63 + 23]],
            dtype=np.uint64,
        )
        path = _write_source(tmp_path, "u64.tif", values, source_grid)
        result = compute(resample_to(source(path), dst, resampling="nearest"))
        assert int(result.values[0, 0]) == 2**63 + 11

    def test_exact_int64_nearest(self, tmp_path):
        source_grid = _grid(3, 2)
        dst = _grid(6, 4, affine=(1000.0, 10.0, 0.0, 2000.0, 0.0, -10.0))
        values = np.asarray(
            [[-2**62, -2**62 + 1, -2**62 + 2],
             [-2**62 + 10, -2**62 + 11, -2**62 + 12]],
            dtype=np.int64,
        )
        path = _write_source(tmp_path, "i64.tif", values, source_grid)
        result = compute(resample_to(source(path), dst, resampling="nearest"))
        assert int(result.values[0, 0]) == -2**62

    def test_validity_mask_read(self, tmp_path):
        source_grid = _grid(10, 8)
        values = np.arange(80, dtype=np.float32).reshape(8, 10)
        valid = np.ones(values.shape, dtype=np.bool_)
        valid[3:5, 4:6] = False
        path = _write_source(tmp_path, "vmask.tif", values, source_grid, valid=valid)
        dst = _grid(7, 6, affine=(1005.0, 30.0, 0.0, 2005.0, 0.0, -30.0))
        result = compute(resample_to(source(path), dst, resampling="bilinear"))
        assert result.valid.any()
        assert not result.valid.all()

    def test_coverage_threshold_is_stable(self, tmp_path):
        source_grid = _grid(10, 8)
        values = np.arange(80, dtype=np.float32).reshape(8, 10)
        valid = np.ones(values.shape, dtype=np.bool_)
        valid[:, 4:6] = False
        path = _write_source(tmp_path, "cov.tif", values, source_grid, valid=valid)
        dst = _grid(20, 16, affine=(1000.0, 10.0, 0.0, 2000.0, 0.0, -10.0))
        expr = resample_to(
            source(path), dst,
            resampling="bilinear", validity_coverage_threshold=0.75,
        )
        expected = compute(expr)
        _, actual = _execute_windowed(expr, window_width=5, window_height=4)
        _assert_parity(actual, expected)

    def test_expression_identity_changes_with_resampling(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        dst = self._dst_grid()
        r1 = resample_to(expr, dst, resampling="nearest")
        r2 = resample_to(expr, dst, resampling="bilinear")
        assert r1.scientific_identity() != r2.scientific_identity()

    def test_expression_identity_changes_with_coverage(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        dst = self._dst_grid()
        r1 = resample_to(expr, dst)
        r2 = resample_to(expr, dst, validity_coverage_threshold=0.5)
        assert r1.scientific_identity() != r2.scientific_identity()

    def test_no_implicit_resampling_in_binary_ops(self, tmp_path):
        expr1, grid1 = self._make_source(tmp_path, width=10, height=8)
        grid2 = _grid(10, 8, affine=(1050.0, 20.0, 0.0, 2050.0, 0.0, -20.0))
        rows, cols = np.indices((8, 10), dtype=np.float32)
        path2 = _write_source(tmp_path, "src2.tif", cols + rows, grid2)
        expr2 = source(path2)
        assert expr1.grid != expr2.grid
        from lunarscout.errors import GridMismatchError

        with pytest.raises(GridMismatchError):
            _combined = expr1 + expr2

    def test_resample_then_combine(self, tmp_path):
        expr1, grid1 = self._make_source(tmp_path, width=10, height=8)
        grid2 = _grid(10, 8, affine=(1050.0, 20.0, 0.0, 2050.0, 0.0, -20.0))
        rows, cols = np.indices((8, 10), dtype=np.float32)
        path2 = _write_source(tmp_path, "src2.tif", cols + rows, grid2)
        expr2 = source(path2)
        resampled = resample_to(expr2, grid1, resampling="nearest")
        combined = expr1 + resampled
        result = compute(combined)
        assert result.valid.any()


# ---------------------------------------------------------------------------
# Safety rules
# ---------------------------------------------------------------------------


class TestResamplingSafety:
    def _src(self, tmp_path, dtype=np.float32):
        grid = _grid(6, 4)
        values = np.arange(24, dtype=dtype).reshape(4, 6)
        path = _write_source(tmp_path, "src.tif", values, grid)
        return grid, path

    def _dst(self):
        return _grid(12, 8, affine=(1000.0, 10.0, 0.0, 2000.0, 0.0, -10.0))

    def test_nearest_safe_for_categorical(self, tmp_path):
        src_grid, path = self._src(tmp_path, np.int32)
        dst = self._dst()
        result = resample_to(source(path), dst, resampling="nearest")
        assert result.operation_id == "alignment.resample_to"

    def test_mode_safe_for_categorical(self, tmp_path):
        src_grid, path = self._src(tmp_path, np.int32)
        dst = self._dst()
        result = resample_to(source(path), dst, resampling="mode")
        assert result.operation_id == "alignment.resample_to"

    def test_mode_executes_for_boolean_categorical_raster(self):
        grid = _grid(6, 4)
        values = np.asarray([[False, True] * 3] * 4, dtype=np.bool_)
        result = resample_to(
            ma_raster(values, grid), self._dst(), resampling="mode",
        )
        assert isinstance(result, Raster)
        assert result.dtype == np.dtype(np.bool_)
        assert result.valid.any()

    def test_interpolation_unsafe_for_categorical(self, tmp_path):
        src_grid, path = self._src(tmp_path, np.int32)
        dst = self._dst()
        from lunarscout.errors import AlignmentError
        with pytest.raises(AlignmentError, match="not safe for categorical"):
            resample_to(source(path), dst, resampling="bilinear")

    def test_interpolation_unsafe_for_boolean(self, tmp_path):
        grid = _grid(6, 4)
        values = np.array([[0, 1, 0, 1, 0, 1]] * 4, dtype=np.bool_)
        raster = ma_raster(values, grid)
        dst = self._dst()
        from lunarscout.errors import AlignmentError
        with pytest.raises(AlignmentError):
            resample_to(raster, dst, resampling="bilinear")

    def test_mode_unsafe_for_continuous(self, tmp_path):
        src_grid, path = self._src(tmp_path, np.float32)
        dst = self._dst()
        from lunarscout.errors import AlignmentError
        with pytest.raises(AlignmentError, match="mode"):
            resample_to(source(path), dst, resampling="mode")

    def test_allow_unsafe_overrides(self, tmp_path):
        src_grid, path = self._src(tmp_path, np.int32)
        dst = self._dst()
        result = resample_to(source(path), dst, resampling="bilinear", allow_unsafe=True)
        assert result.operation_id == "alignment.resample_to"

    def test_explicit_categorical_false(self, tmp_path):
        src_grid, path = self._src(tmp_path, np.int32)
        dst = self._dst()
        result = resample_to(
            source(path), dst, resampling="bilinear", categorical=False,
            output_dtype=np.float64,
        )
        assert result.operation_id == "alignment.resample_to"

    def test_output_dtype_does_not_change_source_categorical_inference(self, tmp_path):
        _, path = self._src(tmp_path, np.int32)
        with pytest.raises(AlignmentError) as error:
            resample_to(
                source(path), self._dst(), resampling="bilinear",
                output_dtype=np.float64,
            )
        assert error.value.code == "alignment_unsafe_categorical_resampling"

    def test_continuous_integer_interpolation_requires_override(self, tmp_path):
        _, path = self._src(tmp_path, np.int32)
        with pytest.raises(AlignmentError) as error:
            resample_to(
                source(path), self._dst(), resampling="bilinear",
                categorical=False,
            )
        assert error.value.code == "alignment_unsafe_integer_interpolation"

    def test_unsafe_output_dtype_requires_override(self, tmp_path):
        _, path = self._src(tmp_path, np.float64)
        with pytest.raises(AlignmentError) as error:
            resample_to(source(path), self._dst(), output_dtype=np.float32)
        assert error.value.code == "alignment_unsafe_output_dtype"

    def test_boolean_interpolation_override_uses_numeric_output(self):
        grid = _grid(6, 4)
        values = np.asarray([[False, True] * 3] * 4, dtype=np.bool_)
        result = resample_to(
            ma_raster(values, grid), self._dst(), resampling="bilinear",
            categorical=False, output_dtype=np.float32, allow_unsafe=True,
        )
        assert isinstance(result, Raster)
        assert result.dtype == np.dtype(np.float32)
        assert result.valid.any()


# ---------------------------------------------------------------------------
# Registry / list_operations filtering
# ---------------------------------------------------------------------------


class TestRegistryFiltering:
    def test_terrain_operations_listed(self):
        from lunarscout.map_algebra import list_operations
        terrain = [item["id"] for item in list_operations(category="terrain")]
        assert "terrain.slope" in terrain
        assert "terrain.aspect" in terrain
        assert "terrain.hillshade" in terrain

    def test_alignment_operations_listed(self):
        from lunarscout.map_algebra import list_operations
        alignment = [item["id"] for item in list_operations(category="alignment")]
        assert "alignment.resample_to" in alignment

    def test_terrain_file_backed(self):
        from lunarscout.map_algebra import list_operations
        fb = [item["id"] for item in list_operations(execution_mode="file_backed")]
        assert "terrain.slope" in fb
        assert "terrain.aspect" in fb
        assert "terrain.hillshade" in fb
        assert "alignment.resample_to" in fb

    def test_terrain_eager(self):
        from lunarscout.map_algebra import list_operations
        eager = [item["id"] for item in list_operations(execution_mode="eager")]
        assert "terrain.slope" in eager

    def test_describe_terrain(self):
        from lunarscout.map_algebra import describe_operation
        spec = describe_operation("terrain.slope")
        assert spec["id"] == "terrain.slope"
        assert spec["file_backed_available"]
        assert spec["eager_available"]
        assert len(spec["parameters"]) >= 2
        assert spec["output_dtype_rule"] == "float32"

    def test_describe_alignment(self):
        from lunarscout.map_algebra import describe_operation
        spec = describe_operation("alignment.resample_to")
        assert spec["id"] == "alignment.resample_to"
        assert spec["file_backed_available"]
        assert len(spec["parameters"]) >= 2

    def test_no_duplicate_ids(self):
        from lunarscout.map_algebra import list_operations
        all_ops = list_operations()
        ids = [item["id"] for item in all_ops]
        assert len(ids) == len(set(ids))

    def test_valid_semantic_versions(self):
        from lunarscout.map_algebra import list_operations
        for item in list_operations():
            version = item["version"]
            assert isinstance(version, int)
            assert version >= 1

    def test_file_backed_claims_match_windowed(self):
        from lunarscout.map_algebra import list_operations
        from lunarscout.map_algebra._windowed import WINDOWED_OPERATION_IDS
        declared = {
            item["id"]
            for item in list_operations(execution_mode="file_backed")
        }
        intrinsic = {"source", "constant"} | {
            op_id
            for op_id in declared
            if op_id.startswith("coordinate.")
        }
        assert declared - intrinsic == set(WINDOWED_OPERATION_IDS)

    @pytest.mark.parametrize(
        ("operation_id", "function_name"),
        [
            ("terrain.slope", "slope"),
            ("terrain.aspect", "aspect"),
            ("terrain.hillshade", "hillshade"),
            ("alignment.resample_to", "resample_to"),
        ],
    )
    def test_registry_parameters_match_public_signature(
        self, operation_id, function_name,
    ):
        import inspect
        import lunarscout.map_algebra as ma

        from lunarscout.map_algebra import describe_operation

        registry_parameters = {
            item["name"] for item in describe_operation(operation_id)["parameters"]
        }
        signature_parameters = set(inspect.signature(getattr(ma, function_name)).parameters)
        signature_parameters.difference_update({"raster", "grid"})
        assert registry_parameters == signature_parameters
