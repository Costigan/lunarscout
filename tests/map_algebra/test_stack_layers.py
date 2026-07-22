from __future__ import annotations

import numpy as np
import pytest

from lunarscout.errors import MapAlgebraDTypeError, MapAlgebraError
from lunarscout.georeference import GeoReference
from lunarscout.map_algebra import (
    compute,
    describe_operation,
    max_layers,
    mean_layers,
    min_layers,
    raster,
    sum_layers,
    write,
)
from tests.map_algebra.conftest import MOON_PROJ4, MOON_WKT


def _raster(
    values: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    units: str | None = None,
    x_origin: float = 0.0,
):
    height, width = values.shape
    grid = GeoReference(
        projection_wkt=MOON_WKT,
        projection_proj4=MOON_PROJ4,
        affine_transform=(x_origin, 20.0, 0.0, 0.0, 0.0, -20.0),
        width=width,
        height=height,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )
    return raster(values, grid, valid=valid, units=units)


@pytest.mark.parametrize(
    "operation", [sum_layers, mean_layers, min_layers, max_layers],
)
def test_layer_stack_requires_at_least_one_layer(operation):
    with pytest.raises(MapAlgebraError) as error:
        operation()
    assert error.value.code == "map_algebra_empty_layers"


@pytest.mark.parametrize("as_expression", [False, True])
def test_layer_stack_rejects_scalar_layers(as_expression):
    source = _raster(np.ones((1, 1), dtype=np.float32))
    first = source.expression() if as_expression else source
    with pytest.raises(MapAlgebraError) as error:
        sum_layers(first, 2)
    assert error.value.code == "map_algebra_invalid_layer"
    assert error.value.details["layer_index"] == 1


def test_layer_stack_values_validity_units_and_dtypes():
    first = _raster(
        np.array([[1, 5], [3, 8]], dtype=np.float32),
        valid=np.array([[True, True], [False, True]], dtype=np.bool_),
        units="metres",
    )
    second = _raster(
        np.array([[2, 4], [6, 7]], dtype=np.float32),
        valid=np.array([[True, False], [True, True]], dtype=np.bool_),
        units="metres",
    )
    third = _raster(
        np.array([[3, 3], [9, 6]], dtype=np.float32),
        units="metres",
    )

    expected_valid = np.array([[True, False], [False, True]], dtype=np.bool_)
    total = sum_layers(first, second, third)
    mean = mean_layers(first, second, third)
    minimum = min_layers(first, second, third)
    maximum = max_layers(first, second, third)

    assert total.dtype == np.dtype(np.float32)
    assert mean.dtype == np.dtype(np.float32)
    assert total.units == mean.units == minimum.units == maximum.units == "metres"
    for result in (total, mean, minimum, maximum):
        np.testing.assert_array_equal(result.valid, expected_valid)
    assert total.values[0, 0] == 6
    assert mean.values[0, 0] == 2
    assert minimum.values[1, 1] == 6
    assert maximum.values[1, 1] == 8


def test_single_integer_layer_mean_uses_true_division():
    source = _raster(np.array([[1, 2]], dtype=np.int16))
    total = sum_layers(source)
    mean = mean_layers(source)
    assert total.dtype == np.dtype(np.int16)
    assert mean.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(mean.values, np.array([[1.0, 2.0]]))


def test_stack_overflow_policy_reuses_exact_local_arithmetic():
    source = _raster(np.array([[200]], dtype=np.uint8))
    with pytest.raises(MapAlgebraDTypeError) as error:
        sum_layers(source, source)
    assert error.value.code == "map_algebra_overflow"

    promoted = sum_layers(source, source, overflow="promote")
    assert promoted.dtype == np.dtype(np.uint16)
    assert int(promoted.values[0, 0]) == 400


def test_stack_rejects_grid_and_unit_mismatches():
    source = _raster(np.ones((2, 2), dtype=np.float32), units="metres")
    shifted = _raster(
        np.ones((2, 2), dtype=np.float32), units="metres", x_origin=20.0,
    )
    with pytest.raises(MapAlgebraError) as grid_error:
        sum_layers(source, shifted)
    assert grid_error.value.code == "map_algebra_grid_mismatch"

    unknown = _raster(np.ones((2, 2), dtype=np.float32), units=None)
    with pytest.raises(MapAlgebraError) as unit_error:
        max_layers(source, unknown)
    assert unit_error.value.code == "map_algebra_unknown_units"


def test_stack_expression_eager_and_windowed_parity(tmp_path):
    import rasterio

    first = _raster(np.arange(12, dtype=np.float32).reshape(3, 4), units="m")
    second = _raster(np.full((3, 4), 2.0, dtype=np.float32), units="m")
    expression = mean_layers(first.expression(), second)
    eager = mean_layers(first, second)
    computed = compute(expression)
    np.testing.assert_array_equal(computed.values, eager.values)
    np.testing.assert_array_equal(computed.valid, eager.valid)

    output = write(
        tmp_path / "mean-layers.tif", expression,
        window_width=2, window_height=2,
    )
    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read(1), eager.values)
        np.testing.assert_array_equal(dataset.read_masks(1) > 0, eager.valid)
        assert dataset.dtypes[0] == "float32"


def test_stack_policies_participate_in_composed_expression_identity():
    source = _raster(np.array([[1.0]], dtype=np.float32)).expression()
    invalid = sum_layers(source, source, numeric_errors="invalid")
    keep = sum_layers(source, source, numeric_errors="keep")
    assert invalid.scientific_identity() != keep.scientific_identity()

    integers = _raster(np.array([[1]], dtype=np.uint8)).expression()
    wrapped = sum_layers(integers, integers, overflow="wrap")
    promoted = sum_layers(integers, integers, overflow="promote")
    assert wrapped.scientific_identity() != promoted.scientific_identity()


@pytest.mark.parametrize(
    ("operation_id", "parameters"),
    [
        ("local.sum_layers", ["overflow", "numeric_errors"]),
        ("local.mean_layers", ["overflow", "numeric_errors"]),
        ("local.min_layers", ["numeric_errors"]),
        ("local.max_layers", ["numeric_errors"]),
    ],
)
def test_stack_registry_metadata(operation_id, parameters):
    description = describe_operation(operation_id)
    assert description["arity"] is None
    # The named helper expands into ordinary file-backed local nodes; no
    # standalone variadic stack node is advertised to the window executor.
    assert description["file_backed_available"] is False
    assert [item["name"] for item in description["parameters"]] == parameters
