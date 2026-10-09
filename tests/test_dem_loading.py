from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import Affine

from lunarscout.geotiff import _resolve_dem_path, read_dem_raster
from lunarscout.products import _load_dem, _normalize_dem_elevations


RADIUS_M = 1_737_400.0


def _write_dem(
    tmp_path: Path,
    name: str,
    values: np.ndarray,
    *,
    scale: float | None = None,
    offset: float | None = None,
    nodata: float | None = None,
) -> Path:
    path = tmp_path / name
    crs = CRS.from_user_input("ESRI:103878")
    profile = {
        "driver": "GTiff",
        "width": int(values.shape[1]),
        "height": int(values.shape[0]),
        "count": 1,
        "dtype": values.dtype,
        "crs": crs.to_wkt(),
        "transform": Affine.from_gdal(1000.0, 20.0, 0.0, 2000.0, 0.0, -20.0),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1)
        if scale is not None:
            dataset.scales = (scale,)
        if offset is not None:
            dataset.offsets = (offset,)
    return path


def test_normalize_applies_declared_scale_and_undoes_radius_offset() -> None:
    raw = np.array([-14594, 0, 14054], dtype=np.int16)

    result = _normalize_dem_elevations(
        raw, scale=0.5, offset=RADIUS_M, radius_m=RADIUS_M
    )

    np.testing.assert_allclose(result, raw.astype(np.float32) * 0.5)


def test_normalize_leaves_plain_sphere_relative_metres_unchanged() -> None:
    values = np.array([-4380.5, 0.0, 2713.0], dtype=np.float32)

    result = _normalize_dem_elevations(
        values, scale=1.0, offset=0.0, radius_m=RADIUS_M
    )

    np.testing.assert_allclose(result, values)


def test_normalize_neutralises_radius_offset_when_offset_is_reference_radius() -> None:
    elevation_from_sphere = np.array([-1000.0, 0.0, 2000.0], dtype=np.float64)

    result = _normalize_dem_elevations(
        elevation_from_sphere, scale=1.0, offset=RADIUS_M, radius_m=RADIUS_M
    )

    np.testing.assert_allclose(result, elevation_from_sphere)


def test_normalize_detects_radius_from_centre_by_magnitude() -> None:
    radius_from_centre = np.array([1730103.0, 1744427.0], dtype=np.float64)

    result = _normalize_dem_elevations(
        radius_from_centre, scale=None, offset=None, radius_m=RADIUS_M
    )

    np.testing.assert_allclose(result, radius_from_centre - RADIUS_M)


def test_normalize_applies_ordinary_offset() -> None:
    values = np.array([10.0, 20.0], dtype=np.float32)

    result = _normalize_dem_elevations(
        values, scale=1.0, offset=100.0, radius_m=RADIUS_M
    )

    np.testing.assert_allclose(result, [110.0, 120.0])


def test_normalize_masks_declared_nodata_in_raw_units() -> None:
    raw = np.array([-9999, 100, 200], dtype=np.int16)

    result = _normalize_dem_elevations(
        raw, scale=0.5, offset=None, radius_m=RADIUS_M, nodata=-9999
    )

    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1:], [50.0, 100.0])


def test_normalize_does_not_modify_the_caller_array() -> None:
    values = np.array([1730103.0, 1744427.0, np.nan], dtype=np.float32)
    original = values.copy()

    _normalize_dem_elevations(values, scale=None, offset=None, radius_m=RADIUS_M)

    np.testing.assert_array_equal(values, original)


def test_normalize_detects_radius_from_centre_despite_nodata_zero() -> None:
    radius_from_centre = np.array([0.0, 1730103.0, 1744427.0], dtype=np.float64)

    result = _normalize_dem_elevations(
        radius_from_centre, scale=None, offset=None, radius_m=RADIUS_M, nodata=0.0
    )

    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1:], radius_from_centre[1:] - RADIUS_M)


def test_normalize_all_nodata_returns_all_nan() -> None:
    result = _normalize_dem_elevations(
        np.full(4, -9999.0), scale=None, offset=None, radius_m=RADIUS_M,
        nodata=-9999.0,
    )

    assert result.dtype == np.float32
    assert np.isnan(result).all()


def test_load_dem_masks_the_declared_nodata_value(tmp_path: Path) -> None:
    values = np.array([[10.0, -9999.0], [30.0, 40.0]], dtype=np.float32)
    path = _write_dem(tmp_path, "nodata.tif", values, nodata=-9999.0)

    dem, _ = _load_dem(path)

    assert np.isnan(dem.elevation_m[0, 1])
    np.testing.assert_allclose(dem.elevation_m[[0, 1, 1], [0, 0, 1]], [10.0, 30.0, 40.0])


def test_read_dem_raster_returns_declared_scale_and_offset(tmp_path: Path) -> None:
    path = _write_dem(
        tmp_path,
        "scaled.tif",
        np.arange(12, dtype=np.int16).reshape(3, 4),
        scale=0.5,
        offset=RADIUS_M,
    )

    values, georef, scale, offset = read_dem_raster(path)

    assert values.dtype == np.dtype(np.int16)
    assert georef is not None
    assert scale == 0.5
    assert offset == RADIUS_M


def test_resolve_dem_path_discovers_pds_label(tmp_path: Path) -> None:
    data = tmp_path / "ldem.img"
    label = tmp_path / "ldem.lbl"
    label.write_text("PDS3")

    assert _resolve_dem_path(data) == label
    assert _resolve_dem_path(label) == label

    bare = tmp_path / "no-label.img"
    assert _resolve_dem_path(bare) == bare

    geotiff = tmp_path / "dem.tif"
    assert _resolve_dem_path(geotiff) == geotiff


def test_load_dem_normalizes_scaled_int16_dem(tmp_path: Path) -> None:
    raw = np.array([[-14594, 0], [14054, 100]], dtype=np.int16)
    path = _write_dem(
        tmp_path, "scaled-dem.tif", raw, scale=0.5, offset=RADIUS_M
    )

    dem, georef = _load_dem(path)

    assert georef is not None
    np.testing.assert_allclose(dem.elevation_m, raw.astype(np.float32) * 0.5)
