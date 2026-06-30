from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import lunarscout as ls


def _georef(
    lunar_projection: tuple[str, str],
    *,
    width: int = 2,
    height: int = 2,
    affine: tuple[float, float, float, float, float, float] = (
        0.0,
        2.0,
        0.0,
        4.0,
        0.0,
        -2.0,
    ),
    nodata: int | float | None = None,
) -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine,
        width=width,
        height=height,
        pixel_size_x=affine[1],
        pixel_size_y=affine[5],
        nodata=nodata,
    )


def test_same_grid_ignores_nodata(lunar_projection):
    left = _georef(lunar_projection, nodata=-1)
    right = replace(left, nodata=255)

    assert ls.same_grid(left, right)
    ls.require_same_grid(left, right)


def test_same_grid_compares_crs_semantically(lunar_projection):
    from pyproj import CRS

    alternate_wkt = CRS.from_user_input("ESRI:103878").to_wkt(pretty=True)
    left = _georef(lunar_projection)
    right = replace(left, projection_wkt=alternate_wkt)

    assert left.projection_wkt != right.projection_wkt
    assert ls.same_grid(left, right)


def test_same_grid_is_exact_by_default_and_can_use_explicit_tolerance(lunar_projection):
    left = _georef(lunar_projection)
    affine = list(left.affine_transform)
    affine[0] += 1e-9
    right = replace(left, affine_transform=tuple(affine))

    assert not ls.same_grid(left, right)
    assert ls.same_grid(left, right, affine_tolerance=1e-8)


def test_require_same_grid_reports_differences(lunar_projection):
    left = _georef(lunar_projection)
    right = _georef(lunar_projection, width=3)

    with pytest.raises(ls.GridMismatchError) as raised:
        ls.require_same_grid(left, right)

    assert raised.value.code == "grid_mismatch"
    assert raised.value.details["differences"]["width"] == [2, 3]


def test_same_grid_rejects_invalid_tolerance(lunar_projection):
    georef = _georef(lunar_projection)

    with pytest.raises(ls.AlignmentError) as raised:
        ls.same_grid(georef, georef, affine_tolerance=-1)

    assert raised.value.code == "alignment_invalid_tolerance"


def test_align_nearest_to_finer_grid_preserves_dtype(lunar_projection):
    source_georef = _georef(lunar_projection, nodata=-1)
    destination_georef = _georef(
        lunar_projection,
        width=4,
        height=4,
        affine=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
        nodata=999,
    )
    source = np.asarray([[1, 2], [3, 4]], dtype=np.int16)

    aligned, aligned_georef = ls.align(source, source_georef, to=destination_georef)

    assert aligned.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(
        aligned,
        np.asarray(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
            dtype=np.int16,
        ),
    )
    assert aligned_georef.nodata == -1
    assert ls.same_grid(aligned_georef, destination_georef)


def test_align_accepts_explicit_output_dtype_and_nodata(lunar_projection):
    georef = _georef(lunar_projection, nodata=-1)
    source = np.asarray([[1, 2], [3, 4]], dtype=np.int16)

    aligned, aligned_georef = ls.align(
        source,
        georef,
        to=georef,
        output_dtype=np.float32,
        output_nodata=np.nan,
    )

    assert aligned.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(aligned, source.astype(np.float32))
    assert np.isnan(aligned_georef.nodata)


def test_align_none_disables_destination_nodata(lunar_projection):
    georef = _georef(lunar_projection, nodata=-1)

    _aligned, aligned_georef = ls.align(
        np.ones((2, 2), dtype=np.int16),
        georef,
        to=georef,
        output_nodata=None,
    )

    assert aligned_georef.nodata is None


@pytest.mark.parametrize(
    ("dtype", "nodata"),
    [(np.int64, -(2**63)), (np.uint64, 2**64 - 1)],
)
def test_align_preserves_exact_64_bit_nodata(lunar_projection, dtype, nodata):
    source_georef = _georef(lunar_projection, width=1, height=1, nodata=nodata)
    destination_georef = _georef(lunar_projection, width=2, height=2)

    aligned, aligned_georef = ls.align(
        np.asarray([[7]], dtype=dtype), source_georef, to=destination_georef
    )

    assert aligned.dtype == np.dtype(dtype)
    assert aligned[0, 1] == nodata
    assert aligned_georef.nodata == nodata


def test_align_uses_source_nodata_for_uncovered_pixels(lunar_projection):
    source_georef = _georef(lunar_projection, width=1, height=1, nodata=-99)
    destination_georef = _georef(lunar_projection, width=2, height=2, nodata=None)

    aligned, aligned_georef = ls.align(
        np.asarray([[7]], dtype=np.int16), source_georef, to=destination_georef
    )

    np.testing.assert_array_equal(aligned, np.asarray([[7, -99], [-99, -99]], dtype=np.int16))
    assert aligned_georef.nodata == -99


def test_align_rejects_shape_mismatch(lunar_projection):
    georef = _georef(lunar_projection)

    with pytest.raises(ls.AlignmentError) as raised:
        ls.align(np.ones((3, 2), dtype=np.float32), georef, to=georef)

    assert raised.value.code == "alignment_source_shape_mismatch"


def test_align_rejects_unknown_resampling(lunar_projection):
    georef = _georef(lunar_projection)

    with pytest.raises(ls.AlignmentError) as raised:
        ls.align(
            np.ones((2, 2), dtype=np.float32),
            georef,
            to=georef,
            resampling="invented",  # type: ignore[arg-type]
        )

    assert raised.value.code == "alignment_invalid_resampling"
    assert "nearest" in raised.value.details["available"]


def test_align_rejects_nodata_not_representable_by_output_dtype(lunar_projection):
    georef = _georef(lunar_projection, nodata=-1)

    with pytest.raises(ls.GeoTiffMetadataError) as raised:
        ls.align(
            np.ones((2, 2), dtype=np.int16),
            georef,
            to=georef,
            output_dtype=np.uint8,
        )

    assert raised.value.code == "geotiff_unrepresentable_nodata"


def test_available_resampling_algorithms_match_gdal_capabilities():
    algorithms = ls.available_resampling_algorithms()

    assert algorithms[:5] == ("nearest", "bilinear", "cubic", "cubicspline", "lanczos")
    assert "average" in algorithms
    assert "rms" in algorithms


@pytest.mark.parametrize("resampling", ls.available_resampling_algorithms())
def test_every_reported_resampling_algorithm_is_usable(lunar_projection, resampling):
    georef = _georef(lunar_projection)
    source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    aligned, _aligned_georef = ls.align(
        source,
        georef,
        to=georef,
        resampling=resampling,
    )

    assert aligned.shape == source.shape
    assert aligned.dtype == source.dtype
