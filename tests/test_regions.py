from __future__ import annotations

import numpy as np
import pytest

from lunarscout import (
    GeoReference,
    RegionOperationError,
    filter_regions_by_size,
    find_borders,
    label_regions,
    region_sizes,
)


def _georef(lunar_projection, shape, *, nodata=None) -> GeoReference:
    return GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        width=int(shape[1]),
        height=int(shape[0]),
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=nodata,
    )


def test_label_regions_uses_eight_neighbor_connectivity() -> None:
    mask = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)

    labels, result_georef = label_regions(mask)

    assert labels.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(labels, np.asarray([[1, 0], [0, 1]], dtype=np.int32))
    assert result_georef is None


def test_numeric_masks_treat_nonzero_as_true() -> None:
    mask = np.asarray([[0, 2, 0], [0, -3, 0]], dtype=np.int16)

    labels, _ = label_regions(mask)

    np.testing.assert_array_equal(
        labels,
        np.asarray([[0, 1, 0], [0, 1, 0]], dtype=np.int32),
    )


def test_auto_nodata_uses_georeference_and_restores_int32_nodata(lunar_projection) -> None:
    mask = np.asarray([[1, -9999, 1], [1, 0, 1]], dtype=np.int16)
    georef = _georef(lunar_projection, mask.shape, nodata=-9999)

    labels, result_georef = label_regions(mask, georef)

    assert labels.dtype == np.dtype(np.int32)
    assert labels[0, 1] == -9999
    assert labels[0, 0] == 1
    assert labels[0, 2] == 2  # invalid/false pixels separate the two regions
    assert result_georef is not georef
    assert result_georef is not None
    assert result_georef.nodata == -9999


def test_explicit_none_disables_georeference_nodata(lunar_projection) -> None:
    mask = np.asarray([[0, -9999, 0]], dtype=np.int16)
    georef = _georef(lunar_projection, mask.shape, nodata=-9999)

    labels, result_georef = label_regions(mask, georef, nodata=None)

    np.testing.assert_array_equal(labels, np.asarray([[0, 1, 0]], dtype=np.int32))
    assert result_georef is not None
    assert result_georef.nodata is None


def test_explicit_nodata_override_without_georeference() -> None:
    mask = np.asarray([[1, 255, 1]], dtype=np.uint8)

    labels, result_georef = label_regions(mask, nodata=255)

    np.testing.assert_array_equal(labels, np.asarray([[1, 255, 2]], dtype=np.int32))
    assert result_georef is None


def test_region_sizes_use_int32_and_zero_background() -> None:
    mask = np.asarray(
        [[1, 1, 0, 1], [1, 0, 0, 1]],
        dtype=np.uint8,
    )

    sizes, _ = region_sizes(mask)

    assert sizes.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        sizes,
        np.asarray([[3, 3, 0, 2], [3, 0, 0, 2]], dtype=np.int32),
    )


def test_filter_supports_greater_equal_and_less_equal_comparators() -> None:
    mask = np.asarray(
        [[1, 1, 0, 1, 0], [1, 0, 0, 0, 0]],
        dtype=np.uint8,
    )

    large, _ = filter_regions_by_size(mask, threshold=2, comparator=">=")
    small, _ = filter_regions_by_size(mask, threshold=1, comparator="<=")

    np.testing.assert_array_equal(
        large,
        np.asarray([[True, True, False, False, False], [True, False, False, False, False]]),
    )
    np.testing.assert_array_equal(
        small,
        np.asarray([[False, False, False, True, False], [False, False, False, False, False]]),
    )


def test_cleanup_seed_selection_preserves_original_region_shape() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    mask[3, 5] = True  # one-pixel spur removed by opening

    filtered, _ = filter_regions_by_size(
        mask,
        threshold=9,
        comparator=">=",
        cleanup="opening",
        iterations=1,
    )

    np.testing.assert_array_equal(filtered, mask)


def test_boolean_results_are_plain_arrays_without_nodata() -> None:
    mask = np.ones((3, 3), dtype=bool)

    filtered, filtered_georef = filter_regions_by_size(mask, threshold=1)
    borders, borders_georef = find_borders(mask)

    assert type(filtered) is np.ndarray
    assert filtered.dtype == np.dtype(np.bool_)
    assert filtered_georef is None
    assert type(borders) is np.ndarray
    assert borders_georef is None
    np.testing.assert_array_equal(
        borders,
        np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    )


def test_boolean_results_are_masked_when_nodata_processing_is_active(lunar_projection) -> None:
    mask = np.asarray([[1, -9999, 1], [1, 0, 1]], dtype=np.int16)
    georef = _georef(lunar_projection, mask.shape, nodata=-9999)

    filtered, filtered_georef = filter_regions_by_size(mask, georef, threshold=1)
    borders, borders_georef = find_borders(mask, georef)

    assert isinstance(filtered, np.ma.MaskedArray)
    assert isinstance(borders, np.ma.MaskedArray)
    assert filtered.mask[0, 1]
    assert borders.mask[0, 1]
    assert filtered_georef is not None
    assert filtered_georef.nodata == -9999
    assert borders_georef is not None
    assert borders_georef.nodata == -9999


def test_masked_input_is_preserved_without_georeference() -> None:
    mask = np.ma.array(
        [[True, True], [False, True]],
        mask=[[False, True], [False, False]],
    )

    filtered, result_georef = filter_regions_by_size(mask, threshold=1)

    assert isinstance(filtered, np.ma.MaskedArray)
    assert filtered.mask[0, 1]
    assert result_georef is None


def test_label_regions_rejects_unrepresentable_int32_nodata(lunar_projection) -> None:
    mask = np.asarray([[1.0, np.nan]], dtype=np.float32)
    georef = _georef(lunar_projection, mask.shape, nodata=np.nan)

    with pytest.raises(RegionOperationError) as captured:
        label_regions(mask, georef)

    assert captured.value.code == "region_unrepresentable_nodata"


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"cleanup": "closing"}, "region_invalid_argument"),
        ({"iterations": -1}, "region_invalid_argument"),
    ],
)
def test_region_operations_validate_cleanup(kwargs, code) -> None:
    with pytest.raises(RegionOperationError) as captured:
        label_regions(np.ones((2, 2), dtype=bool), **kwargs)

    assert captured.value.code == code


def test_filter_validates_threshold_and_comparator() -> None:
    mask = np.ones((2, 2), dtype=bool)

    with pytest.raises(RegionOperationError):
        filter_regions_by_size(mask, threshold=-1)
    with pytest.raises(RegionOperationError):
        filter_regions_by_size(mask, threshold=1, comparator="==")  # type: ignore[arg-type]


def test_region_operations_validate_georeference_shape(lunar_projection) -> None:
    georef = _georef(lunar_projection, (3, 3))

    with pytest.raises(RegionOperationError) as captured:
        region_sizes(np.ones((2, 3), dtype=bool), georef)

    assert captured.value.code == "region_shape_mismatch"


@pytest.mark.parametrize("dtype", [np.complex64, object])
def test_region_operations_reject_unsupported_mask_dtype(dtype) -> None:
    with pytest.raises(RegionOperationError) as captured:
        label_regions(np.ones((2, 2), dtype=dtype))

    assert captured.value.code == "region_unsupported_datatype"
