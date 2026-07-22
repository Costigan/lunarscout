from __future__ import annotations

import numpy as np
import pytest

import lunarscout as ls
import lunarscout.map_algebra as ma
from lunarscout.errors import MapAlgebraError, RegionOperationError
from lunarscout.georeference import GeoReference
from tests.map_algebra.conftest import MOON_PROJ4, MOON_WKT


def _raster(
    values: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    rotated: bool = False,
):
    height, width = values.shape
    transform = (
        (100.0, 20.0, 3.0, 200.0, -2.0, -20.0)
        if rotated
        else (100.0, 20.0, 0.0, 200.0, 0.0, -20.0)
    )
    grid = GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=transform,
        width=width,
        height=height,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=0,
    )
    return ma.raster(values, grid, valid=valid, units="candidate")


def test_label_regions_supports_explicit_connectivity():
    source = _raster(np.array([[True, False], [False, True]], dtype=np.bool_))
    labels8 = ma.label_regions(source)
    labels4 = ma.label_regions(source, connectivity=4)

    np.testing.assert_array_equal(
        labels8.values, np.array([[1, 0], [0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        labels4.values, np.array([[1, 0], [0, 2]], dtype=np.int32),
    )
    assert labels8.dtype == labels4.dtype == np.dtype(np.int32)
    assert labels8.units is None


def test_region_sizes_and_filter_use_selected_connectivity():
    source = _raster(np.array([[True, False], [False, True]], dtype=np.bool_))
    sizes8 = ma.region_sizes(source, connectivity=8)
    sizes4 = ma.region_sizes(source, connectivity=4)
    np.testing.assert_array_equal(
        sizes8.values, np.array([[2, 0], [0, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        sizes4.values, np.array([[1, 0], [0, 1]], dtype=np.int32),
    )

    kept8 = ma.filter_regions_by_size(source, threshold=2, connectivity=8)
    kept4 = ma.filter_regions_by_size(source, threshold=2, connectivity=4)
    np.testing.assert_array_equal(kept8.values, source.values)
    assert not np.any(kept4.values)


def test_find_borders_connectivity_changes_internal_border_definition():
    values = np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=np.bool_,
    )
    source = _raster(values)
    borders4 = ma.find_borders(source, connectivity=4)
    borders8 = ma.find_borders(source, connectivity=8)
    assert not borders4.values[1, 1]
    assert borders8.values[1, 1]


def test_region_adapters_preserve_grid_and_canonical_validity():
    values = np.array(
        [[True, True, False], [False, True, True], [True, False, True]],
        dtype=np.bool_,
    )
    valid = np.array(
        [[True, False, True], [True, True, True], [False, True, True]],
        dtype=np.bool_,
    )
    source = _raster(values, valid=valid, rotated=True)
    original_values = source.values.copy()
    original_valid = source.valid.copy()

    outputs = (
        ma.label_regions(source),
        ma.region_sizes(source),
        ma.filter_regions_by_size(source, threshold=1),
        ma.find_borders(source),
    )
    for output in outputs:
        assert output.georef == source.georef
        np.testing.assert_array_equal(output.valid, valid)
        assert output.units is None
    np.testing.assert_array_equal(source.values, original_values)
    np.testing.assert_array_equal(source.valid, original_valid)


def test_invalid_true_payload_does_not_connect_regions():
    source = _raster(
        np.array([[True, True, True]], dtype=np.bool_),
        valid=np.array([[True, False, True]], dtype=np.bool_),
    )
    labels = ma.label_regions(source, connectivity=4)
    assert labels.values[0, 0] == 1
    assert labels.values[0, 2] == 2
    assert not labels.valid[0, 1]


def test_all_invalid_input_remains_all_invalid():
    source = _raster(
        np.ones((2, 2), dtype=np.bool_),
        valid=np.zeros((2, 2), dtype=np.bool_),
    )
    for operation in (
        ma.label_regions, ma.region_sizes, ma.find_borders,
    ):
        result = operation(source)
        assert not np.any(result.valid)


def test_cleanup_matches_existing_array_api():
    values = np.zeros((7, 7), dtype=np.bool_)
    values[2:5, 2:5] = True
    values[3, 5] = True
    source = _raster(values)
    adapted = ma.filter_regions_by_size(
        source, threshold=9, cleanup="opening", iterations=1,
        connectivity=8,
    )
    expected, _ = ls.filter_regions_by_size(
        values, threshold=9, cleanup="opening", iterations=1,
        connectivity=8,
    )
    np.testing.assert_array_equal(adapted.values, expected)


@pytest.mark.parametrize("connectivity", [0, 6, True, 4.0, "8", [4]])
def test_connectivity_validation_is_structured(connectivity):
    source = _raster(np.ones((2, 2), dtype=np.bool_))
    with pytest.raises(RegionOperationError) as error:
        ma.label_regions(source, connectivity=connectivity)
    assert error.value.code == "region_invalid_argument"
    assert error.value.details["argument"] == "connectivity"


def test_numpy_integer_connectivity_is_accepted():
    source = _raster(np.eye(2, dtype=np.bool_))
    result = ma.label_regions(source, connectivity=np.int64(4))
    assert result.values[1, 1] == 2


def test_map_algebra_region_adapters_require_eager_boolean_rasters():
    numeric = _raster(np.ones((2, 2), dtype=np.uint8))
    with pytest.raises(MapAlgebraError) as dtype_error:
        ma.label_regions(numeric)
    assert dtype_error.value.code == "map_algebra_requires_boolean"

    boolean = _raster(np.ones((2, 2), dtype=np.bool_))
    with pytest.raises(MapAlgebraError) as expression_error:
        ma.label_regions(boolean.expression())
    assert expression_error.value.code == "map_algebra_invalid_region_operand"


@pytest.mark.parametrize(
    ("operation_id", "parameters", "dtype"),
    [
        (
            "region.label_regions",
            ["cleanup", "iterations", "connectivity"],
            "int32",
        ),
        (
            "region.region_sizes",
            ["cleanup", "iterations", "connectivity"],
            "int32",
        ),
        (
            "region.filter_regions_by_size",
            ["threshold", "comparator", "cleanup", "iterations", "connectivity"],
            "bool",
        ),
        ("region.find_borders", ["connectivity"], "bool"),
    ],
)
def test_region_registry_metadata(operation_id, parameters, dtype):
    description = ma.describe_operation(operation_id)
    assert description["category"] == "region"
    assert description["file_backed_available"] is False
    assert description["output_dtype_rule"] == dtype
    assert [item["name"] for item in description["parameters"]] == parameters


def test_existing_array_api_keeps_eight_neighbor_default_and_accepts_four():
    mask = np.array([[True, False], [False, True]], dtype=np.bool_)
    default, _ = ls.label_regions(mask)
    explicit8, _ = ls.label_regions(mask, connectivity=8)
    explicit4, _ = ls.label_regions(mask, connectivity=4)
    np.testing.assert_array_equal(default, explicit8)
    assert explicit4[1, 1] == 2
