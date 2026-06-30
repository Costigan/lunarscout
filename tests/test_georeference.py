from __future__ import annotations

import numpy as np
import pytest

from lunarscout import GeoReference, GeoReferenceError


@pytest.fixture
def georef(lunar_projection) -> GeoReference:
    return GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=(1000.0, 20.0, 2.0, 2000.0, -1.0, -20.0),
        width=10,
        height=5,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=-9999.0,
    )


def test_pixel_coordinates_default_to_centers(georef: GeoReference) -> None:
    easting, northing = georef.pixel_to_projected(0, 0)
    corner_easting, corner_northing = georef.pixel_to_projected(0, 0, anchor="corner")

    assert easting == pytest.approx(1011.0)
    assert northing == pytest.approx(1989.5)
    assert corner_easting == pytest.approx(1000.0)
    assert corner_northing == pytest.approx(2000.0)


def test_projected_pixel_round_trip_does_not_round(georef: GeoReference) -> None:
    easting, northing = georef.pixel_to_projected(2.25, 3.75)
    column, row = georef.projected_to_pixel(easting, northing)

    assert column == pytest.approx(2.25)
    assert row == pytest.approx(3.75)


def test_vectorized_affine_conversion_broadcasts_inputs(georef: GeoReference) -> None:
    columns = np.asarray([[0.0], [1.0]])
    rows = np.asarray([0.0, 1.0, 2.0])

    eastings, northings = georef.pixel_to_projected(columns, rows)
    round_trip_columns, round_trip_rows = georef.projected_to_pixel(eastings, northings)

    assert isinstance(eastings, np.ndarray)
    assert eastings.shape == (2, 3)
    np.testing.assert_allclose(
        round_trip_columns,
        np.broadcast_to(columns, (2, 3)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        round_trip_rows,
        np.broadcast_to(rows, (2, 3)),
        atol=1e-12,
    )


def test_lunar_lonlat_round_trip_supports_arrays(georef: GeoReference) -> None:
    columns = np.asarray([0.0, 3.0, 9.0])
    rows = np.asarray([0.0, 2.0, 4.0])

    longitudes, latitudes = georef.pixel_to_lonlat(columns, rows)
    result_columns, result_rows = georef.lonlat_to_pixel(longitudes, latitudes)

    np.testing.assert_allclose(result_columns, columns, atol=1e-8)
    np.testing.assert_allclose(result_rows, rows, atol=1e-8)
    assert np.all(np.asarray(latitudes) < 0.0)


def test_projected_bounds_use_all_rotated_corners(georef: GeoReference) -> None:
    corners = [
        georef.pixel_to_projected(0, 0, anchor="corner"),
        georef.pixel_to_projected(georef.width, 0, anchor="corner"),
        georef.pixel_to_projected(0, georef.height, anchor="corner"),
        georef.pixel_to_projected(georef.width, georef.height, anchor="corner"),
    ]

    assert georef.projected_bounds() == pytest.approx(
        (
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        )
    )


def test_contains_pixel_supports_scalars_and_arrays(georef: GeoReference) -> None:
    assert georef.contains_pixel(0, 0) is True
    assert georef.contains_pixel(10, 0) is False
    result = georef.contains_pixel(np.asarray([0, 9, 10]), np.asarray([0, 4, 4]))
    np.testing.assert_array_equal(result, np.asarray([True, True, False]))


def test_georeference_rejects_noninvertible_affine(lunar_projection) -> None:
    with pytest.raises(GeoReferenceError) as captured:
        GeoReference(
            projection_wkt=lunar_projection[0],
            projection_proj4=lunar_projection[1],
            affine_transform=(0.0, 1.0, 2.0, 0.0, 2.0, 4.0),
            width=1,
            height=1,
            pixel_size_x=1.0,
            pixel_size_y=4.0,
            nodata=None,
        )

    assert captured.value.code == "georeference_noninvertible_affine"


def test_georeference_rejects_inconsistent_signed_pixel_size(lunar_projection) -> None:
    with pytest.raises(GeoReferenceError) as captured:
        GeoReference(
            projection_wkt=lunar_projection[0],
            projection_proj4=lunar_projection[1],
            affine_transform=(0.0, 20.0, 0.0, 0.0, 0.0, -20.0),
            width=1,
            height=1,
            pixel_size_x=20.0,
            pixel_size_y=20.0,
            nodata=None,
        )

    assert captured.value.code == "georeference_inconsistent_pixel_size"
