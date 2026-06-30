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
    np.testing.assert_allclose(result[[0, 0, -1, -1], [0, -1, 0, -1]], 45.0)


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
