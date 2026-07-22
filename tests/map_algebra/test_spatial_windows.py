from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from affine import Affine

from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import compute, source, write
from lunarscout.map_algebra._planner import plan_expression
from lunarscout.map_algebra._spatial import (
    make_resample_expression,
    make_terrain_expression,
    source_window_for_resampling,
)
from lunarscout.map_algebra._windowed import execute_windowed
from lunarscout.map_algebra._windows import SourceWindowCache
from tests.map_algebra.conftest import MOON_PROJ4, MOON_WKT


def _grid(
    width: int,
    height: int,
    *,
    affine: tuple[float, float, float, float, float, float] | None = None,
) -> GeoReference:
    transform = affine or (1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0)
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


def _write_source(
    tmp_path,
    name: str,
    values: np.ndarray,
    grid: GeoReference,
    *,
    valid: np.ndarray | None = None,
):
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
        if valid is not None:
            dataset.write_mask(np.asarray(valid, dtype=np.uint8) * 255)
    return path


def _execute(expression, *, window_width: int, window_height: int):
    plan = plan_expression(
        expression,
        window_width=window_width,
        window_height=window_height,
    )
    with SourceWindowCache() as cache:
        result = execute_windowed(plan, cache)
    assert result is not None
    return plan, result


def _assert_raster_parity(actual, expected, *, atol: float = 1e-6) -> None:
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
            actual.values[actual.valid], expected.values[expected.valid]
        )


@pytest.mark.parametrize(
    ("operation_id", "parameters"),
    [
        ("terrain.slope", {}),
        ("terrain.slope", {"units": "percent", "scale": 2.5}),
        ("terrain.aspect", {}),
        (
            "terrain.hillshade",
            {"azimuth": 123.0, "altitude": 28.0, "scale": 2.0, "z_factor": 1.4},
        ),
    ],
)
def test_terrain_windows_match_whole_array_across_many_seams(
    tmp_path,
    operation_id,
    parameters,
):
    grid = _grid(37, 29)
    rows, columns = np.indices((grid.height, grid.width), dtype=np.float32)
    values = columns * columns + rows * 3.25 + np.sin(rows / 2.0) * 5.0
    valid = np.ones(values.shape, dtype=np.bool_)
    valid[6, 8] = False
    valid[15:17, 22] = False
    path = _write_source(tmp_path, "terrain.tif", values, grid, valid=valid)
    expression = make_terrain_expression(operation_id, source(path), **parameters)

    expected = compute(expression)
    plan, actual = _execute(expression, window_width=7, window_height=5)

    assert plan.maximum_halo == 1
    _assert_raster_parity(actual, expected)


@pytest.mark.parametrize("operation_id", ["terrain.slope", "terrain.aspect", "terrain.hillshade"])
def test_terrain_compute_edges_matches_at_dataset_boundary(tmp_path, operation_id):
    grid = _grid(11, 9)
    values = np.add.outer(
        np.arange(grid.height, dtype=np.float32) * 2.0,
        np.arange(grid.width, dtype=np.float32),
    )
    path = _write_source(tmp_path, "edges.tif", values, grid)
    expression = make_terrain_expression(
        operation_id, source(path), compute_edges=True,
    )

    expected = compute(expression)
    _, actual = _execute(expression, window_width=4, window_height=3)

    assert actual.valid[[0, -1], :].all()
    assert actual.valid[:, [0, -1]].all()
    _assert_raster_parity(actual, expected)


def test_nested_terrain_accumulates_halo_and_crops_once_per_node(tmp_path):
    grid = _grid(25, 19)
    rows, columns = np.indices((grid.height, grid.width), dtype=np.float32)
    path = _write_source(tmp_path, "nested.tif", columns**2 + rows**2, grid)
    first = make_terrain_expression("terrain.slope", source(path))
    expression = make_terrain_expression("terrain.slope", first)

    expected = compute(expression)
    plan, actual = _execute(expression, window_width=6, window_height=5)

    assert plan.maximum_halo == 2
    _assert_raster_parity(actual, expected)


def test_local_expression_after_terrain_keeps_halo_semantics(tmp_path):
    grid = _grid(23, 17)
    rows, columns = np.indices((grid.height, grid.width), dtype=np.float32)
    path = _write_source(tmp_path, "local.tif", columns**2 + rows * 0.25, grid)
    terrain = make_terrain_expression("terrain.slope", source(path))
    expression = (terrain + 2.0) * 3.0

    expected = compute(expression)
    _, actual = _execute(expression, window_width=5, window_height=4)

    _assert_raster_parity(actual, expected)


def test_terrain_output_nodata_does_not_define_canonical_validity(tmp_path):
    grid = _grid(13, 11)
    values = np.tile(np.arange(grid.width, dtype=np.float32), (grid.height, 1))
    path = _write_source(tmp_path, "nodata_collision.tif", values, grid)
    expression = make_terrain_expression(
        "terrain.aspect", source(path), output_nodata=270.0,
    )

    expected = compute(expression)
    _, actual = _execute(expression, window_width=4, window_height=3)

    assert np.all(actual.values[1:-1, 1:-1] == 270.0)
    assert actual.valid[1:-1, 1:-1].all()
    assert not actual.valid[0, :].any()
    assert actual.georef.nodata == 270.0
    _assert_raster_parity(actual, expected)


@pytest.mark.parametrize("resampling", ["nearest", "bilinear", "cubic", "lanczos", "average"])
def test_resample_windows_match_whole_array_for_shift_and_scale(tmp_path, resampling):
    source_grid = _grid(31, 27)
    destination_grid = _grid(
        43,
        35,
        affine=(1013.0, 14.0, 0.0, 1987.0, 0.0, -14.0),
    )
    rows, columns = np.indices((source_grid.height, source_grid.width), dtype=np.float32)
    values = np.sin(columns / 4.0) + np.cos(rows / 5.0) + columns * 0.03
    valid = np.ones(values.shape, dtype=np.bool_)
    valid[9:14, 12:16] = False
    path = _write_source(tmp_path, f"{resampling}.tif", values, source_grid, valid=valid)
    expression = make_resample_expression(
        source(path), destination_grid, resampling=resampling,
    )

    expected = compute(expression)
    plan, actual = _execute(expression, window_width=8, window_height=6)

    assert plan.maximum_halo == 0
    _assert_raster_parity(actual, expected, atol=2e-6)


def test_resample_rotated_grids_has_no_internal_seams(tmp_path):
    source_grid = _grid(
        29,
        25,
        affine=(1000.0, 20.0, 3.0, 2000.0, 2.0, -20.0),
    )
    destination_grid = _grid(
        33,
        28,
        affine=(1011.0, 17.0, -1.5, 1991.0, 1.0, -17.0),
    )
    rows, columns = np.indices((source_grid.height, source_grid.width), dtype=np.float32)
    values = columns * 0.75 + rows * 1.25
    path = _write_source(tmp_path, "rotated.tif", values, source_grid)
    expression = make_resample_expression(
        source(path), destination_grid, resampling="bilinear",
    )

    expected = compute(expression)
    _, actual = _execute(expression, window_width=7, window_height=5)

    _assert_raster_parity(actual, expected)


def test_resample_across_crs_matches_whole_array(tmp_path):
    from pyproj import CRS

    projected = CRS.from_epsg(3857)
    geographic = CRS.from_epsg(4326)
    source_grid = GeoReference(
        projection_wkt=projected.to_wkt(),
        projection_proj4=projected.to_proj4(),
        affine_transform=(-2000.0, 200.0, 0.0, 2000.0, 0.0, -200.0),
        width=20,
        height=20,
        pixel_size_x=200.0,
        pixel_size_y=-200.0,
        nodata=None,
    )
    destination_grid = GeoReference(
        projection_wkt=geographic.to_wkt(),
        projection_proj4=geographic.to_proj4(),
        affine_transform=(-0.018, 0.0015, 0.0, 0.018, 0.0, -0.0015),
        width=24,
        height=24,
        pixel_size_x=0.0015,
        pixel_size_y=-0.0015,
        nodata=None,
    )
    rows, columns = np.indices((20, 20), dtype=np.float32)
    path = _write_source(tmp_path, "cross_crs.tif", columns + rows * 2.0, source_grid)
    expression = make_resample_expression(
        source(path), destination_grid, resampling="bilinear",
    )

    expected = compute(expression)
    _, actual = _execute(expression, window_width=5, window_height=7)

    _assert_raster_parity(actual, expected, atol=2e-6)


def test_resample_of_terrain_expression_remains_bounded_and_matches_compute(tmp_path):
    source_grid = _grid(35, 31)
    destination_grid = _grid(
        48,
        41,
        affine=(1005.0, 14.0, 0.0, 1995.0, 0.0, -14.0),
    )
    rows, columns = np.indices((source_grid.height, source_grid.width), dtype=np.float32)
    path = _write_source(tmp_path, "terrain_resample.tif", columns**2 + rows, source_grid)
    terrain = make_terrain_expression("terrain.slope", source(path))
    expression = make_resample_expression(
        terrain, destination_grid, resampling="bilinear",
    )

    expected = compute(expression)
    plan, actual = _execute(expression, window_width=9, window_height=7)

    assert plan.maximum_halo == 1
    _assert_raster_parity(actual, expected)


def test_resample_validity_coverage_threshold_is_window_stable(tmp_path):
    source_grid = _grid(12, 10)
    destination_grid = _grid(
        24,
        20,
        affine=(1000.0, 10.0, 0.0, 2000.0, 0.0, -10.0),
    )
    values = np.arange(120, dtype=np.float32).reshape(10, 12)
    valid = np.ones(values.shape, dtype=np.bool_)
    valid[:, 5:7] = False
    path = _write_source(tmp_path, "coverage.tif", values, source_grid, valid=valid)
    expression = make_resample_expression(
        source(path),
        destination_grid,
        resampling="bilinear",
        validity_coverage_threshold=0.75,
    )

    expected = compute(expression)
    _, actual = _execute(expression, window_width=5, window_height=4)

    _assert_raster_parity(actual, expected)


def test_resample_nonoverlap_returns_canonical_invalid(tmp_path):
    source_grid = _grid(4, 4)
    destination_grid = _grid(
        5,
        3,
        affine=(1_000_000.0, 20.0, 0.0, 1_000_000.0, 0.0, -20.0),
    )
    path = _write_source(
        tmp_path, "nonoverlap.tif", np.ones((4, 4), dtype=np.float32), source_grid,
    )
    expression = make_resample_expression(source(path), destination_grid)

    expected = compute(expression)
    _, actual = _execute(expression, window_width=2, window_height=2)

    assert not actual.valid.any()
    _assert_raster_parity(actual, expected)


def test_resample_nearest_preserves_exact_uint64_payload(tmp_path):
    source_grid = _grid(3, 2)
    destination_grid = _grid(
        6,
        4,
        affine=(1000.0, 10.0, 0.0, 2000.0, 0.0, -10.0),
    )
    values = np.asarray(
        [[2**63 + 11, 2**63 + 12, 2**63 + 13], [2**63 + 21, 2**63 + 22, 2**63 + 23]],
        dtype=np.uint64,
    )
    path = _write_source(tmp_path, "uint64.tif", values, source_grid)
    expression = make_resample_expression(
        source(path), destination_grid, resampling="nearest",
    )

    expected = compute(expression)
    _, actual = _execute(expression, window_width=2, window_height=3)

    _assert_raster_parity(actual, expected)
    assert int(actual.values[0, 0]) == 2**63 + 11


def test_source_window_mapping_is_clipped_and_smaller_than_large_source():
    source_grid = _grid(10_000, 8_000)
    destination_grid = replace(source_grid, width=500, height=400)

    request = source_window_for_resampling(
        source_grid,
        destination_grid,
        x0=120,
        y0=90,
        width=64,
        height=48,
        resampling="bilinear",
    )

    assert request is not None
    x0, y0, width, height = request
    assert x0 >= 0 and y0 >= 0
    assert width < source_grid.width // 10
    assert height < source_grid.height // 10


def test_windowed_resample_reads_only_mapped_source_footprints(tmp_path, monkeypatch):
    source_grid = _grid(400, 300)
    destination_grid = _grid(
        60,
        50,
        affine=(2200.0, 20.0, 0.0, 1200.0, 0.0, -20.0),
    )
    values = np.arange(400 * 300, dtype=np.float32).reshape(300, 400)
    path = _write_source(tmp_path, "bounded_resample.tif", values, source_grid)
    expression = make_resample_expression(
        source(path), destination_grid, resampling="bilinear",
    )
    requests: list[tuple[int, int]] = []
    original = SourceWindowCache._read_file_window

    def recording_read(self, node, x0, y0, width, height):
        requests.append((width, height))
        return original(self, node, x0, y0, width, height)

    monkeypatch.setattr(SourceWindowCache, "_read_file_window", recording_read)
    _, actual = _execute(expression, window_width=12, window_height=10)

    assert actual.valid.any()
    assert requests
    assert max(width for width, _ in requests) <= 15
    assert max(height for _, height in requests) <= 13


def test_window_cache_distinguishes_halo_requests_for_same_output_window(tmp_path):
    grid = _grid(9, 8)
    values = np.arange(72, dtype=np.float32).reshape(8, 9)
    expression = source(_write_source(tmp_path, "cache.tif", values, grid))

    with SourceWindowCache(max_windows=4) as cache:
        first = cache.read_values(expression, 0, 1, 1, 3, 3)
        second = cache.read_values(expression, 0, 0, 0, 5, 5)

        assert cache.window_count == 2
        assert first.shape == (3, 3)
        assert second.shape == (5, 5)


def test_windowed_terrain_write_publishes_expected_mask(tmp_path):
    grid = _grid(19, 15)
    rows, columns = np.indices((grid.height, grid.width), dtype=np.float32)
    path = _write_source(tmp_path, "write_terrain.tif", columns**2 + rows, grid)
    expression = make_terrain_expression("terrain.aspect", source(path))
    expected = compute(expression)
    output = tmp_path / "aspect.tif"

    write(output, expression, window_width=6, window_height=4, invalid_value=-9999.0)

    import rasterio
    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read_masks(1).astype(np.bool_), expected.valid)
        values = dataset.read(1)
        np.testing.assert_allclose(values[expected.valid], expected.values[expected.valid])
