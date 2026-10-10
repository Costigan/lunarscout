from __future__ import annotations

import numpy as np
import pytest

from lunarscout import GeoReference, TerrainOperationError, aspect, hillshade, slope


@pytest.fixture
def terrain_georef(lunar_projection) -> GeoReference:
    return GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        width=7,
        height=7,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=-9999.0,
    )


@pytest.fixture
def east_rising_plane() -> np.ndarray:
    return np.tile(np.arange(7, dtype=np.float32), (7, 1))


def test_slope_defaults_to_degrees_and_gdal_edge_behavior(
    east_rising_plane,
    terrain_georef,
) -> None:
    result, result_georef = slope(
        east_rising_plane,
        terrain_georef,
        output_nodata=-1234.0,
    )

    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(result[1:-1, 1:-1], 45.0)
    assert np.all(result[0, :] == -1234.0)
    assert np.all(result[-1, :] == -1234.0)
    assert np.all(result[:, 0] == -1234.0)
    assert np.all(result[:, -1] == -1234.0)
    assert result_georef is not terrain_georef
    assert result_georef.affine_transform == terrain_georef.affine_transform
    assert result_georef.nodata == -1234.0


def test_slope_supports_percent_units(east_rising_plane, terrain_georef) -> None:
    result, _ = slope(
        east_rising_plane,
        terrain_georef,
        output_nodata=-9999.0,
        units="percent",
    )

    np.testing.assert_allclose(result[1:-1, 1:-1], 100.0)


def test_slope_compute_edges_fills_edges(east_rising_plane, terrain_georef) -> None:
    result, _ = slope(
        east_rising_plane,
        terrain_georef,
        output_nodata=-9999.0,
        compute_edges=True,
    )

    assert np.all(result != -9999.0)
    np.testing.assert_allclose(result[1:-1, :], 45.0)
    np.testing.assert_allclose(result[[0, -1], 1:-1], 45.0)
    # gdaldem -compute_edges replicates the column in the four corner
    # windows, which halves the east gradient there: atan(0.5).
    np.testing.assert_allclose(result[[0, 0, -1, -1], [0, -1, 0, -1]], 26.565052, atol=1e-5)


def test_slope_propagates_input_nodata_through_required_neighborhood(
    east_rising_plane,
    terrain_georef,
) -> None:
    source = east_rising_plane.copy()
    source[3, 3] = -9999.0

    result, _ = slope(source, terrain_georef, output_nodata=-1234.0)

    assert np.all(result[2:5, 2:5] == -1234.0)
    assert result[1, 1] == pytest.approx(45.0)


def test_aspect_uses_gdal_azimuth_convention(east_rising_plane, terrain_georef) -> None:
    result, result_georef = aspect(
        east_rising_plane,
        terrain_georef,
        output_nodata=-9999.0,
    )

    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(result[1:-1, 1:-1], 270.0)
    assert result_georef.nodata == -9999.0


def test_flat_aspect_is_output_nodata(terrain_georef) -> None:
    result, _ = aspect(
        np.ones((7, 7), dtype=np.float32),
        terrain_georef,
        output_nodata=-1234.0,
    )

    assert np.all(result == -1234.0)


def test_hillshade_preserves_gdal_uint8_output_and_custom_nodata(
    east_rising_plane,
    terrain_georef,
) -> None:
    result, result_georef = hillshade(
        east_rising_plane,
        terrain_georef,
        output_nodata=255,
    )

    assert result.dtype == np.dtype(np.uint8)
    assert np.all(result[1:-1, 1:-1] == 218)
    assert np.all(result[0, :] == 255)
    assert result_georef.nodata == 255


def test_hillshade_rejects_nodata_outside_uint8_range(
    east_rising_plane,
    terrain_georef,
) -> None:
    with pytest.raises(TerrainOperationError) as captured:
        hillshade(
            east_rising_plane,
            terrain_georef,
            output_nodata=-1,
        )

    assert captured.value.code == "terrain_unrepresentable_output_nodata"


def test_terrain_operations_reject_shape_mismatch(terrain_georef) -> None:
    with pytest.raises(TerrainOperationError) as captured:
        slope(
            np.ones((6, 7), dtype=np.float32),
            terrain_georef,
            output_nodata=-9999.0,
        )

    assert captured.value.code == "terrain_shape_mismatch"


def test_terrain_operations_reject_complex_values(terrain_georef) -> None:
    with pytest.raises(TerrainOperationError) as captured:
        aspect(
            np.ones((7, 7), dtype=np.complex64),
            terrain_georef,
            output_nodata=-9999.0,
        )

    assert captured.value.code == "terrain_unsupported_datatype"


def test_terrain_operations_translate_gdal_unsupported_dtype(terrain_georef) -> None:
    with pytest.raises(TerrainOperationError) as captured:
        slope(
            np.ones((7, 7), dtype=np.float16),
            terrain_georef,
            output_nodata=-9999.0,
        )

    assert captured.value.code == "terrain_unsupported_source"


def test_terrain_operations_reject_unrepresentable_source_nodata(terrain_georef) -> None:
    with pytest.raises(TerrainOperationError) as captured:
        slope(
            np.ones((7, 7), dtype=np.uint8),
            terrain_georef,
            output_nodata=-9999.0,
        )

    assert captured.value.code == "terrain_unsupported_source"


def test_slope_rejects_unknown_units(east_rising_plane, terrain_georef) -> None:
    with pytest.raises(TerrainOperationError) as captured:
        slope(
            east_rising_plane,
            terrain_georef,
            output_nodata=-9999.0,
            units="radians",  # type: ignore[arg-type]
        )

    assert captured.value.code == "terrain_invalid_argument"


def test_with_nodata_returns_new_immutable_georeference(terrain_georef) -> None:
    result = terrain_georef.with_nodata(0)

    assert result is not terrain_georef
    assert result.nodata == 0
    assert terrain_georef.nodata == -9999.0
    assert result.affine_transform == terrain_georef.affine_transform


# Review finding H2: slope, aspect and hillshade claimed GDAL Horn
# compatibility but used np.gradient (two-point central differences).

import shutil
import subprocess

from pyproj import CRS as _CRS


def _relief(height: int = 24, width: int = 30) -> np.ndarray:
    return np.random.default_rng(0).normal(0.0, 5.0, (height, width)).astype(np.float32)


def _grid(lunar_projection, values, *, xres=2.0, yres=-2.0, rotation=0.0) -> GeoReference:
    return GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=(1000.0, xres, rotation, 2000.0, 0.0, yres),
        width=values.shape[1], height=values.shape[0],
        pixel_size_x=xres, pixel_size_y=yres, nodata=-9999.0,
    )


def _horn_reference(z: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Independent interior Horn gradient for a north-up grid (east, south)."""
    z = z.astype(np.float64)
    east = ((z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])) / (8 * res)
    south = ((z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])) / (8 * res)
    return east, south


def test_interior_slope_and_aspect_use_the_horn_kernel(lunar_projection) -> None:
    values = _relief()
    grid = _grid(lunar_projection, values)
    east, south = _horn_reference(values, 2.0)
    slope_values, _ = slope(values, grid, output_nodata=-9999.0)
    np.testing.assert_allclose(
        slope_values[1:-1, 1:-1], np.degrees(np.arctan(np.hypot(east, south))), atol=1e-4
    )
    aspect_values, _ = aspect(values, grid, output_nodata=-9999.0)
    expected = (90.0 - np.degrees(np.arctan2(south, -east))) % 360.0
    difference = (aspect_values[1:-1, 1:-1] - expected + 180.0) % 360.0 - 180.0
    assert np.abs(difference).max() < 1e-3


@pytest.mark.skipif(shutil.which("gdaldem") is None, reason="gdaldem is not installed")
@pytest.mark.parametrize("compute_edges", [False, True])
def test_terrain_matches_gdaldem_on_random_relief(tmp_path, lunar_projection, compute_edges) -> None:
    import rasterio
    from rasterio.transform import Affine

    values = _relief()
    grid = _grid(lunar_projection, values)
    source = tmp_path / "dem.tif"
    with rasterio.open(
        source, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
        count=1, dtype="float32", crs=lunar_projection[0],
        transform=Affine(2.0, 0.0, 1000.0, 0.0, -2.0, 2000.0),
    ) as dataset:
        dataset.write(values, 1)

    def gdaldem(kind: str, *extra: str) -> np.ndarray:
        output = tmp_path / f"{kind}-{'-'.join(extra) or 'default'}.tif"
        arguments = ["gdaldem", kind, str(source), str(output), "-q", *extra]
        if compute_edges:
            arguments.append("-compute_edges")
        subprocess.run(arguments, check=True)
        with rasterio.open(output) as dataset:
            return dataset.read(1)

    region = (slice(None), slice(None)) if compute_edges else (slice(1, -1), slice(1, -1))
    ours, _ = slope(values, grid, output_nodata=-9999.0, compute_edges=compute_edges)
    np.testing.assert_allclose(ours[region], gdaldem("slope")[region], atol=1e-4)
    ours, _ = slope(values, grid, output_nodata=-9999.0, units="percent", compute_edges=compute_edges)
    np.testing.assert_allclose(ours[region], gdaldem("slope", "-p")[region], rtol=1e-5, atol=1e-3)
    ours, _ = aspect(values, grid, output_nodata=-9999.0, compute_edges=compute_edges)
    difference = (ours[region] - gdaldem("aspect")[region] + 180.0) % 360.0 - 180.0
    assert np.abs(difference).max() < 1e-3
    ours, _ = hillshade(values, grid, output_nodata=0, compute_edges=compute_edges)
    np.testing.assert_array_equal(ours[region], gdaldem("hillshade")[region])


def test_aspect_is_physical_on_south_up_grids(lunar_projection) -> None:
    values = _relief()
    north_up, _ = aspect(values, _grid(lunar_projection, values), output_nodata=-9999.0)
    # The same terrain stored south-up: rows reversed, positive row step.
    south_up, _ = aspect(
        values[::-1].copy(), _grid(lunar_projection, values, yres=2.0), output_nodata=-9999.0
    )
    difference = (south_up[::-1][1:-1, 1:-1] - north_up[1:-1, 1:-1] + 180.0) % 360.0 - 180.0
    assert np.abs(difference).max() < 1e-3


def test_rotated_grids_are_rejected(lunar_projection) -> None:
    values = _relief()
    with pytest.raises(TerrainOperationError) as error:
        slope(values, _grid(lunar_projection, values, rotation=0.5), output_nodata=-9999.0)
    assert error.value.code == "terrain_rotated_grid_unsupported"


def test_geographic_grids_require_scale() -> None:
    values = _relief()
    geographic = _CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
    grid = GeoReference(
        projection_wkt=geographic.to_wkt(), projection_proj4=geographic.to_proj4(),
        affine_transform=(0.0, 0.01, 0.0, -80.0, 0.0, -0.01),
        width=values.shape[1], height=values.shape[0],
        pixel_size_x=0.01, pixel_size_y=-0.01, nodata=-9999.0,
    )
    with pytest.raises(TerrainOperationError) as error:
        slope(values, grid, output_nodata=-9999.0)
    assert error.value.code == "terrain_geographic_grid_requires_scale"
    slope(values, grid, output_nodata=-9999.0, scale=30323.0)  # metres per degree
