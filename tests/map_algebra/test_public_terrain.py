from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from lunarscout.georeference import GeoReference
from lunarscout.errors import TerrainOperationError
from lunarscout.raster import Raster
from lunarscout.map_algebra import (
    compute,
    raster as ma_raster,
    slope,
    aspect,
    hillshade,
    source,
    write,
)
from lunarscout.map_algebra._planner import plan_expression
from lunarscout.map_algebra._windowed import execute_windowed
from lunarscout.map_algebra._windows import SourceWindowCache
from tests.map_algebra.conftest import MOON_PROJ4, MOON_WKT


def _grid(width: int, height: int) -> GeoReference:
    transform = (1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0)
    return GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=transform,
        width=width,
        height=height,
        pixel_size_x=transform[1],
        pixel_size_y=transform[5],
        nodata=None,
    )


def _write_source(tmp_path, name: str, values: np.ndarray, grid: GeoReference):
    import rasterio

    path = tmp_path / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=1,
        dtype=values.dtype,
        crs=grid.projection_wkt,
        transform=Affine.from_gdal(*grid.affine_transform),
    ) as dataset:
        dataset.write(values, 1)
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
            actual.values[actual.valid], expected.values[actual.valid]
        )


# ---------------------------------------------------------------------------
# Raster (eager) terrain tests
# ---------------------------------------------------------------------------


class TestPublicTerrainEager:
    def _make_dem(self):
        grid = _grid(9, 7)
        rows, cols = np.indices((grid.height, grid.width), dtype=np.float32)
        values = cols * cols + rows * 3.25 + np.sin(rows / 2.0) * 5.0
        raster = ma_raster(values, grid)
        return raster, grid, values

    def test_slope_eager_returns_raster(self):
        raster, _, _ = self._make_dem()
        result = slope(raster)
        assert isinstance(result, Raster)
        assert result.dtype == np.float32
        assert result.units == "degrees"

    def test_slope_degrees(self):
        raster, _, _ = self._make_dem()
        result = slope(raster, units="degrees")
        assert result.units == "degrees"
        assert result.dtype == np.float32

    def test_slope_percent(self):
        raster, _, _ = self._make_dem()
        result = slope(raster, units="percent")
        assert result.units == "percent"
        assert np.all(result.valid[1:-1, 1:-1])

    def test_slope_scale(self):
        raster, _, _ = self._make_dem()
        s1 = slope(raster, scale=1.0)
        s2 = slope(raster, scale=2.0)
        valid = s1.valid & s2.valid
        assert not np.allclose(s1.values[valid], s2.values[valid])

    def test_aspect_eager_returns_raster(self):
        raster, _, _ = self._make_dem()
        result = aspect(raster)
        assert result.dtype == np.float32
        assert result.units == "degrees"

    def test_aspect_flat_cells_invalid(self):
        grid = _grid(5, 5)
        values = np.ones((5, 5), dtype=np.float32)
        raster = ma_raster(values, grid)
        result = aspect(raster, compute_edges=True)
        assert not result.valid.all()

    def test_hillshade_eager_returns_raster(self):
        raster, _, _ = self._make_dem()
        result = hillshade(raster)
        assert result.dtype == np.uint8
        assert result.units is None

    def test_hillshade_azimuth_altitude(self):
        raster, _, _ = self._make_dem()
        h1 = hillshade(raster, azimuth=315.0, altitude=45.0)
        h2 = hillshade(raster, azimuth=180.0, altitude=30.0)
        valid = h1.valid & h2.valid
        assert not np.array_equal(h1.values[valid], h2.values[valid])

    def test_hillshade_scale_zfactor(self):
        raster, _, _ = self._make_dem()
        h1 = hillshade(raster, scale=1.0, z_factor=1.0)
        h2 = hillshade(raster, scale=1.0, z_factor=2.0)
        valid = h1.valid & h2.valid
        assert not np.array_equal(h1.values[valid], h2.values[valid])

    def test_compute_edges_true(self):
        raster, _, _ = self._make_dem()
        s = slope(raster, compute_edges=True)
        assert s.valid[0, :].any() or s.valid[-1, :].any()

    def test_compute_edges_false_default(self):
        raster, _, _ = self._make_dem()
        s = slope(raster)
        assert not s.valid[0, :].any()
        assert not s.valid[-1, :].any()

    def test_output_nodata_collision_does_not_define_validity(self):
        grid = _grid(7, 5)
        values = np.tile(np.arange(grid.width, dtype=np.float32), (grid.height, 1))
        raster = ma_raster(values, grid)
        result = aspect(raster, output_nodata=270.0, compute_edges=True)
        interior = result.valid[1:-1, 1:-1]
        assert interior.any()
        assert result.georef.nodata == 270.0

    def test_hillshade_valid_zero(self):
        grid = _grid(5, 5)
        values = np.ones((5, 5), dtype=np.float32)
        raster = ma_raster(values, grid)
        result = hillshade(raster, azimuth=0, altitude=0, compute_edges=True)
        assert result.dtype == np.uint8
        valid_interior = result.values[1:-1, 1:-1]
        assert np.any(valid_interior == 0)

    def test_invalid_nodata_rejected(self):
        raster, _, _ = self._make_dem()
        with pytest.raises(TerrainOperationError) as error:
            slope(raster, output_nodata="bad")
        assert error.value.code == "terrain_unrepresentable_output_nodata"

    def test_invalid_units_rejected(self):
        raster, _, _ = self._make_dem()
        with pytest.raises(TerrainOperationError) as error:
            slope(raster, units="radians")
        assert error.value.code == "terrain_invalid_argument"

    def test_invalid_params_rejected(self):
        raster, _, _ = self._make_dem()
        with pytest.raises(TerrainOperationError):
            hillshade(raster, azimuth=-10.0)
        with pytest.raises(TerrainOperationError):
            hillshade(raster, altitude=100.0)
        with pytest.raises(TerrainOperationError) as error:
            slope(raster, scale="bad")  # type: ignore[arg-type]
        assert error.value.code == "terrain_invalid_argument"

    def test_unsupported_operand_type(self):
        grid = _grid(3, 3)
        array = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(TypeError):
            slope(array)  # type: ignore[arg-type]

    def test_eager_dtypes(self):
        raster, _, _ = self._make_dem()
        assert slope(raster).dtype == np.float32
        assert aspect(raster).dtype == np.float32
        assert hillshade(raster).dtype == np.uint8

    def test_shelf_hillshade_identity_changes_for_each_parameter(self):
        raster, _, _ = self._make_dem()
        h1 = hillshade(raster, azimuth=315.0).values
        h2 = hillshade(raster, azimuth=180.0).values
        h3 = hillshade(raster, altitude=30.0).values
        h4 = hillshade(raster, scale=2.0).values
        h5 = hillshade(raster, z_factor=2.0).values
        assert not np.array_equal(h1, h2)
        assert not np.array_equal(h1, h3)
        assert not np.array_equal(h1, h4)
        assert not np.array_equal(h1, h5)

    def test_against_root_terrain_api(self):
        from lunarscout.terrain import slope as root_slope
        from lunarscout.terrain import aspect as root_aspect
        from lunarscout.terrain import hillshade as root_hillshade

        grid = _grid(11, 9)
        rows, cols = np.indices((grid.height, grid.width), dtype=np.float32)
        values = cols * cols + rows * 2.0
        raster = ma_raster(values, grid)

        root_s, _ = root_slope(values, grid, output_nodata=np.nan, units="degrees")
        ma_s = slope(raster)
        np.testing.assert_allclose(root_s[1:-1, 1:-1], ma_s.values[1:-1, 1:-1], rtol=1e-6)

        root_a, _ = root_aspect(values, grid, output_nodata=np.nan)
        ma_a = aspect(raster)
        valid = np.isfinite(root_a)
        np.testing.assert_allclose(root_a[valid], ma_a.values[valid], rtol=1e-6)

        root_h, _ = root_hillshade(values, grid, output_nodata=0)
        ma_h = hillshade(raster)
        np.testing.assert_array_equal(root_h[1:-1, 1:-1], ma_h.values[1:-1, 1:-1])


# ---------------------------------------------------------------------------
# Expression terrain tests
# ---------------------------------------------------------------------------


class TestPublicTerrainExpression:
    def _make_source(self, tmp_path):
        grid = _grid(13, 11)
        rows, cols = np.indices((grid.height, grid.width), dtype=np.float32)
        values = cols * cols + rows * 3.25 + np.sin(rows / 2.0) * 5.0
        path = _write_source(tmp_path, "dem.tif", values, grid)
        return source(path), grid

    def test_slope_expression_returns_expression(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        result = slope(expr)
        from lunarscout.map_algebra._model import RasterExpression
        assert isinstance(result, RasterExpression)
        assert result.operation_id == "terrain.slope"

    def test_aspect_expression_returns_expression(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        result = aspect(expr)
        assert result.operation_id == "terrain.aspect"

    def test_hillshade_expression_returns_expression(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        result = hillshade(expr)
        assert result.operation_id == "terrain.hillshade"

    def test_expression_compute_parity(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        eager_s = compute(slope(expr))
        expr_s = slope(expr)
        computed_s = compute(expr_s)
        _assert_parity(computed_s, eager_s)

    def test_expression_write_parity(self, tmp_path):
        expr, grid = self._make_source(tmp_path)
        terrain_expr = slope(expr)
        expected = compute(terrain_expr)
        _, actual = _execute_windowed(terrain_expr, window_width=5, window_height=4)
        _assert_parity(actual, expected)

    def test_public_writer_terrain_parity(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        terrain_expr = slope(expr, scale=2.0)
        expected = compute(terrain_expr)
        output = tmp_path / "public_slope.tif"

        write(output, terrain_expr, window_width=5, window_height=4)

        from lunarscout.map_algebra import read
        actual = read(output)
        _assert_parity(actual, expected)

    def test_all_operations_expression_parity(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        for op_func, op_name in [(slope, "slope"), (aspect, "aspect"), (hillshade, "hillshade")]:
            eager = compute(op_func(expr))
            terrain_expr = op_func(expr)
            _, actual = _execute_windowed(
                terrain_expr, window_width=5, window_height=4,
            )
            _assert_parity(actual, eager)

    def test_expression_identity_changes_with_parameters(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        s1 = slope(expr, units="degrees")
        s2 = slope(expr, units="percent")
        s3 = slope(expr, scale=2.0)
        h1 = hillshade(expr, azimuth=315.0)
        h2 = hillshade(expr, azimuth=180.0)
        assert s1.operation_id == s2.operation_id == s3.operation_id
        assert s1._params != s2._params
        assert h1._params != h2._params

    def test_invalid_cells_across_window_boundaries(self, tmp_path):
        expr, grid = self._make_source(tmp_path)
        terrain_expr = aspect(expr, compute_edges=False)
        _, actual = _execute_windowed(terrain_expr, window_width=4, window_height=3)
        assert not actual.valid[0, :].any()
        assert not actual.valid[-1, :].any()
        assert not actual.valid[:, 0].any()
        assert not actual.valid[:, -1].any()

    def test_numeric_nodata_collision_remains_valid(self, tmp_path):
        grid = _grid(9, 7)
        values = np.tile(np.arange(grid.width, dtype=np.float32), (grid.height, 1))
        path = _write_source(tmp_path, "collision.tif", values, grid)
        expr = source(path)
        terrain_expr = aspect(expr, output_nodata=270.0)
        _, actual = _execute_windowed(terrain_expr, window_width=3, window_height=3)
        interior_valid = actual.valid[1:-1, 1:-1]
        assert interior_valid.any()
        interior_values = actual.values[1:-1, 1:-1]
        assert np.any(interior_values[interior_valid] == 270.0)

    def test_slope_expression_dtype(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        s = slope(expr)
        assert s.dtype == np.float32
        assert s.units == "degrees"

    def test_hillshade_expression_dtype(self, tmp_path):
        expr, _ = self._make_source(tmp_path)
        h = hillshade(expr)
        assert h.dtype == np.uint8
        assert h.units is None


# ---------------------------------------------------------------------------
# Terrain against the established lunarscout.terrain base
# ---------------------------------------------------------------------------


class TestTerrainAgainstBase:
    def test_slope_matches_base_api(self):
        from lunarscout.terrain import slope as base_slope
        grid = _grid(7, 6)
        rows, cols = np.indices((grid.height, grid.width), dtype=np.float32)
        values = cols * 2.0 + rows * 0.5
        raster = ma_raster(values, grid)
        base, _ = base_slope(values, grid, output_nodata=np.nan, units="degrees")
        ma_result = slope(raster)
        valid = ma_result.valid[1:-1, 1:-1]
        np.testing.assert_allclose(base[1:-1, 1:-1][valid], ma_result.values[1:-1, 1:-1][valid])

    def test_aspect_matches_base_api(self):
        from lunarscout.terrain import aspect as base_aspect
        grid = _grid(7, 6)
        rows, cols = np.indices((grid.height, grid.width), dtype=np.float32)
        values = cols * 2.0 + rows * 0.5
        raster = ma_raster(values, grid)
        base, _ = base_aspect(values, grid, output_nodata=np.nan)
        ma_result = aspect(raster)
        base_valid = np.isfinite(base)
        np.testing.assert_allclose(base[base_valid], ma_result.values[base_valid])

    def test_hillshade_matches_base_api(self):
        from lunarscout.terrain import hillshade as base_hillshade
        grid = _grid(7, 6)
        rows, cols = np.indices((grid.height, grid.width), dtype=np.float32)
        values = cols * 2.0 + rows * 0.5
        raster = ma_raster(values, grid)
        base, _ = base_hillshade(values, grid, output_nodata=0)
        ma_result = hillshade(raster)
        np.testing.assert_array_equal(base[1:-1, 1:-1], ma_result.values[1:-1, 1:-1])


# ---------------------------------------------------------------------------
# Structured error tests
# ---------------------------------------------------------------------------


class TestTerrainErrors:
    def test_expression_rejects_non_elevation_dtype(self, tmp_path):
        grid = _grid(3, 3)
        path = _write_source(tmp_path, "complex.tif", np.ones((3, 3), dtype=np.float32) + 0j, grid)
        with pytest.raises(TerrainOperationError):
            slope(source(path))

    def test_slope_invalid_units(self, tmp_path):
        expr, _ = TestPublicTerrainExpression._make_source(None, tmp_path)
        with pytest.raises(TerrainOperationError):
            slope(expr, units="nonsense")

    def test_hillshade_invalid_azimuth(self, tmp_path):
        expr, _ = TestPublicTerrainExpression._make_source(None, tmp_path)
        with pytest.raises(TerrainOperationError):
            hillshade(expr, azimuth=-1.0)
        with pytest.raises(TerrainOperationError):
            hillshade(expr, azimuth=361.0)

    def test_hillshade_invalid_altitude(self, tmp_path):
        expr, _ = TestPublicTerrainExpression._make_source(None, tmp_path)
        with pytest.raises(TerrainOperationError):
            hillshade(expr, altitude=-1.0)
        with pytest.raises(TerrainOperationError):
            hillshade(expr, altitude=91.0)


# ---------------------------------------------------------------------------
# Canonical identity tests
# ---------------------------------------------------------------------------


class TestTerrainCanonicalIdentity:
    def _make_source(self, tmp_path):
        grid = _grid(7, 6)
        rows, cols = np.indices((grid.height, grid.width), dtype=np.float32)
        values = cols * cols + rows * 3.25
        path = _write_source(tmp_path, "ident.tif", values, grid)
        return source(path)

    def test_slope_identity_changes_with_units(self, tmp_path):
        expr = self._make_source(tmp_path)
        s1 = slope(expr, units="degrees")
        s2 = slope(expr, units="percent")
        assert s1.scientific_identity() != s2.scientific_identity()
        assert s1._params != s2._params

    def test_slope_identity_changes_with_scale(self, tmp_path):
        expr = self._make_source(tmp_path)
        s1 = slope(expr, scale=1.0)
        s2 = slope(expr, scale=2.0)
        assert s1.scientific_identity() != s2.scientific_identity()

    def test_aspect_identity_changes_with_compute_edges(self, tmp_path):
        expr = self._make_source(tmp_path)
        a1 = aspect(expr, compute_edges=False)
        a2 = aspect(expr, compute_edges=True)
        assert a1.scientific_identity() != a2.scientific_identity()

    def test_hillshade_identity_changes_with_azimuth(self, tmp_path):
        expr = self._make_source(tmp_path)
        h1 = hillshade(expr, azimuth=315.0)
        h2 = hillshade(expr, azimuth=180.0)
        assert h1.scientific_identity() != h2.scientific_identity()

    def test_hillshade_identity_changes_with_altitude(self, tmp_path):
        expr = self._make_source(tmp_path)
        h1 = hillshade(expr, altitude=45.0)
        h2 = hillshade(expr, altitude=30.0)
        assert h1.scientific_identity() != h2.scientific_identity()
