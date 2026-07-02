from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio

import lunarscout.geotiff as geotiff_module
from lunarscout import (
    GeoReference,
    GeoTiffBandError,
    GeoTiffDataTypeError,
    GeoTiffMetadataError,
    GeoTiffOpenError,
    GeoTiffWriteError,
    OutputExistsError,
    read_geotiff,
    write_geotiff,
)


SUPPORTED_DTYPES = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float32,
    np.float64,
]


def test_creation_options_include_tiling_compression_predictor_and_bigtiff_policy() -> None:
    assert geotiff_module._creation_options(np.dtype(np.int16)) == [
        "TILED=YES",
        "BLOCKXSIZE=128",
        "BLOCKYSIZE=128",
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "BIGTIFF=IF_SAFER",
    ]
    assert "PREDICTOR=3" in geotiff_module._creation_options(np.dtype(np.float32))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_read_preserves_native_noncomplex_dtype(dtype, make_geotiff) -> None:
    values = np.arange(12, dtype=dtype).reshape(3, 4)
    path = make_geotiff(f"input-{np.dtype(dtype).name}.tif", [values])

    result, georef = read_geotiff(path)

    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, values)
    assert georef is not None


def test_read_selects_requested_one_based_band(make_geotiff) -> None:
    first = np.ones((3, 4), dtype=np.int16)
    second = np.full((3, 4), 7, dtype=np.int16)
    path = make_geotiff("multiband.tif", [first, second])

    result, _georef = read_geotiff(path, band=2)

    np.testing.assert_array_equal(result, second)


def test_read_rejects_band_zero_distinctly(make_geotiff) -> None:
    path = make_geotiff("band-zero.tif", [np.ones((2, 2), dtype=np.uint8)])

    with pytest.raises(GeoTiffBandError) as captured:
        read_geotiff(path, band=0)

    assert captured.value.code == "geotiff_invalid_band"


def test_read_rejects_non_numeric_band_distinctly(make_geotiff) -> None:
    path = make_geotiff("band-text.tif", [np.ones((2, 2), dtype=np.uint8)])

    with pytest.raises(GeoTiffBandError) as captured:
        read_geotiff(path, band="first")  # type: ignore[arg-type]

    assert captured.value.code == "geotiff_invalid_band"


def test_read_rejects_out_of_range_band_distinctly(make_geotiff) -> None:
    path = make_geotiff("band-range.tif", [np.ones((2, 2), dtype=np.uint8)])

    with pytest.raises(GeoTiffBandError) as captured:
        read_geotiff(path, band=2)

    assert captured.value.code == "geotiff_band_out_of_range"


@pytest.mark.parametrize(
    ("dtype", "nodata"),
    [
        (np.int16, -32768),
        (np.uint16, 65535),
        (np.int64, -9_007_199_254_740_991),
        (np.uint64, 9_007_199_254_740_993),
        (np.float32, -9999.0),
        (np.float64, np.nan),
        (np.float32, None),
    ],
)
def test_read_preserves_actual_nodata_value(dtype, nodata, make_geotiff) -> None:
    path = make_geotiff(
        f"nodata-{np.dtype(dtype).name}.tif",
        [np.ones((2, 3), dtype=dtype)],
        nodata=nodata,
    )

    _result, georef = read_geotiff(path)

    assert georef is not None
    if isinstance(nodata, float) and np.isnan(nodata):
        assert np.isnan(georef.nodata)
    else:
        assert georef.nodata == nodata


@pytest.mark.parametrize(
    ("projection", "transform"),
    [(False, False), (False, True), (True, False)],
)
def test_read_returns_none_for_incomplete_georeferencing(
    projection,
    transform,
    make_geotiff,
) -> None:
    path = make_geotiff(
        f"incomplete-{projection}-{transform}.tif",
        [np.ones((2, 3), dtype=np.uint8)],
        projection=projection,
        transform=transform,
    )

    result, georef = read_geotiff(path)

    assert result.shape == (2, 3)
    assert georef is None


def test_read_populates_complete_georeference(make_geotiff, lunar_projection, affine_transform) -> None:
    path = make_geotiff(
        "complete.tif",
        [np.ones((2, 3), dtype=np.float32)],
        nodata=-9999.0,
    )

    _result, georef = read_geotiff(path)

    assert georef is not None
    assert georef.width == 3
    assert georef.height == 2
    assert georef.affine_transform == affine_transform
    assert georef.pixel_size_x == 20.0
    assert georef.pixel_size_y == -20.0
    assert georef.nodata == -9999.0
    assert georef.projection_wkt
    assert "+proj=stere" in georef.projection_proj4


def test_read_suppresses_pyproj_proj4_information_loss_warning(make_geotiff) -> None:
    path = make_geotiff(
        "complete-warning-free.tif",
        [np.ones((2, 3), dtype=np.float32)],
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _result, georef = read_geotiff(path)

    assert georef is not None
    messages = [str(item.message) for item in captured]
    assert not any(
        "You will likely lose important projection information" in message
        for message in messages
    )


def test_read_rejects_complex_geotiff(make_geotiff) -> None:
    path = make_geotiff(
        "complex.tif",
        [np.ones((2, 2), dtype=np.complex64)],
    )

    with pytest.raises(GeoTiffDataTypeError):
        read_geotiff(path)


def test_read_rejects_missing_file() -> None:
    with pytest.raises(GeoTiffOpenError) as captured:
        read_geotiff("does-not-exist.tif")

    assert captured.value.code == "geotiff_file_not_found"


def test_read_rejects_existing_non_geotiff_with_stable_error(tmp_path) -> None:
    path = tmp_path / "not-a-geotiff.tif"
    path.write_text("not raster data", encoding="utf-8")

    with pytest.raises(GeoTiffOpenError) as captured:
        read_geotiff(path)

    assert captured.value.code == "geotiff_unreadable_or_unsupported"


def test_write_round_trip_preserves_dtype_georeferencing_nodata_and_creation_defaults(
    tmp_path,
    lunar_projection,
    affine_transform,
) -> None:
    path = tmp_path / "output.tif"
    values = np.arange(130 * 130, dtype=np.float32).reshape(130, 130)
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=130,
        height=130,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=-9999.0,
    )

    returned = write_geotiff(path, values, georef)
    result, result_georef = read_geotiff(path)

    assert returned == path.resolve()
    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result, values)
    assert result_georef is not None
    assert result_georef.affine_transform == affine_transform
    assert result_georef.nodata == -9999.0
    with rasterio.open(path) as dataset:
        assert dataset.block_shapes[0] == (128, 128)
        image_structure = dataset.tags(ns="IMAGE_STRUCTURE")
        assert image_structure["COMPRESSION"] == "DEFLATE"
        assert image_structure["PREDICTOR"] == "3"
        assert image_structure["INTERLEAVE"] == "BAND"


def test_write_integer_output_uses_integer_predictor(
    tmp_path,
    lunar_projection,
    affine_transform,
) -> None:
    path = tmp_path / "integer-compression.tif"
    values = np.ones((130, 130), dtype=np.int16)
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=130,
        height=130,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=-32768,
    )

    write_geotiff(path, values, georef)

    with rasterio.open(path) as dataset:
        image_structure = dataset.tags(ns="IMAGE_STRUCTURE")
        assert image_structure["COMPRESSION"] == "DEFLATE"
        assert image_structure["PREDICTOR"] == "2"


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_write_read_round_trip_supported_dtype(dtype, tmp_path, lunar_projection, affine_transform) -> None:
    path = tmp_path / f"round-trip-{np.dtype(dtype).name}.tif"
    values = np.arange(12, dtype=dtype).reshape(3, 4)
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=4,
        height=3,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )

    write_geotiff(path, values, georef)
    result, _result_georef = read_geotiff(path)

    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result, values)


def test_write_rejects_shape_mismatch(tmp_path, lunar_projection, affine_transform) -> None:
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=4,
        height=3,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )

    with pytest.raises(GeoTiffWriteError) as captured:
        write_geotiff(tmp_path / "wrong-shape.tif", np.ones((2, 4)), georef)

    assert captured.value.code == "geotiff_shape_mismatch"


def test_write_rejects_unrepresentable_nodata(tmp_path, lunar_projection, affine_transform) -> None:
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=2,
        height=2,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=-1,
    )

    with pytest.raises(GeoTiffMetadataError) as captured:
        write_geotiff(tmp_path / "bad-nodata.tif", np.ones((2, 2), dtype=np.uint8), georef)

    assert captured.value.code == "geotiff_unrepresentable_nodata"


def test_write_preserves_uint64_nodata_above_float_exact_range(
    tmp_path,
    lunar_projection,
    affine_transform,
) -> None:
    nodata = 9_007_199_254_740_993
    path = tmp_path / "uint64-nodata.tif"
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=2,
        height=2,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=nodata,
    )

    write_geotiff(path, np.ones((2, 2), dtype=np.uint64), georef)
    _result, result_georef = read_geotiff(path)

    assert result_georef is not None
    assert result_georef.nodata == nodata


def test_write_rejects_boolean_array_without_implicit_conversion(
    tmp_path,
    lunar_projection,
    affine_transform,
) -> None:
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=2,
        height=2,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )

    with pytest.raises(GeoTiffDataTypeError):
        write_geotiff(tmp_path / "bool.tif", np.ones((2, 2), dtype=np.bool_), georef)


def test_write_overwrite_false_preserves_existing_file(
    tmp_path,
    lunar_projection,
    affine_transform,
) -> None:
    path = tmp_path / "existing.tif"
    original = b"existing-content"
    path.write_bytes(original)
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=2,
        height=2,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )

    with pytest.raises(OutputExistsError):
        write_geotiff(path, np.ones((2, 2), dtype=np.uint8), georef)

    assert path.read_bytes() == original


def test_write_overwrite_true_replaces_existing_file(
    tmp_path,
    lunar_projection,
    affine_transform,
) -> None:
    path = tmp_path / "replace.tif"
    path.write_bytes(b"existing-content")
    values = np.full((2, 2), 8, dtype=np.uint8)
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=2,
        height=2,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )

    write_geotiff(path, values, georef, overwrite=True)

    result, _ = read_geotiff(path)
    np.testing.assert_array_equal(result, values)


def test_write_replace_failure_preserves_destination_and_cleans_temporary_file(
    monkeypatch,
    tmp_path,
    lunar_projection,
    affine_transform,
) -> None:
    path = tmp_path / "replace-failure.tif"
    original = b"existing-content"
    path.write_bytes(original)
    georef = GeoReference(
        projection_wkt=lunar_projection[0],
        projection_proj4=lunar_projection[1],
        affine_transform=affine_transform,
        width=2,
        height=2,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )

    def fail_replace(_source, _destination):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(geotiff_module.os, "replace", fail_replace)

    with pytest.raises(GeoTiffWriteError):
        write_geotiff(path, np.ones((2, 2), dtype=np.uint8), georef, overwrite=True)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp.tif")) == []


def test_import_does_not_load_pythonnet_or_moonlib() -> None:
    package_root = Path(__file__).parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(package_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import lunarscout; "
                "assert 'pythonnet' not in sys.modules; "
                "assert 'moonlib' not in sys.modules"
            ),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
