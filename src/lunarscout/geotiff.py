from __future__ import annotations

import os
import tempfile
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import rasterio
from pyproj import CRS
from rasterio.transform import Affine

from .errors import (
    GeoTiffBandError,
    GeoTiffDataTypeError,
    GeoTiffMetadataError,
    GeoTiffOpenError,
    GeoTiffWriteError,
    OutputExistsError,
)
from .georeference import GeoReference

_NODATA_TAG = "LUNARSCOUT_NODATA_VALUE"
_SUPPORTED_GEOTIFF_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.int8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.uint32),
    np.dtype(np.int32),
    np.dtype(np.uint64),
    np.dtype(np.int64),
    np.dtype(np.float32),
    np.dtype(np.float64),
}


def _projection_proj4(projection_wkt: str) -> str:
    try:
        proj4 = str(CRS.from_wkt(projection_wkt).to_proj4() or "").strip()
    except Exception as exc:
        raise GeoTiffMetadataError(
            "Unable to convert the GeoTIFF projection from WKT to PROJ.4.",
            details={"error": str(exc)},
        ) from exc
    if not proj4:
        raise GeoTiffMetadataError("The GeoTIFF projection has no PROJ.4 representation.")
    return proj4


def read_geotiff(
    filename: str | Path,
    band: int = 1,
) -> tuple[NDArray[Any], GeoReference | None]:
    """Read one GeoTIFF band as its native NumPy dtype and georeferencing."""

    path = Path(filename).expanduser()
    try:
        band_number = int(band)
    except (TypeError, ValueError, OverflowError):
        band_number = 0
    if band_number != band or band_number < 1:
        raise GeoTiffBandError(
            "GeoTIFF band numbers are one-based and must be positive integers.",
            code="geotiff_invalid_band",
            details={"band": band},
        )
    if not path.exists() or not path.is_file():
        raise GeoTiffOpenError(
            f"GeoTIFF file does not exist: {path}",
            code="geotiff_file_not_found",
            details={"path": str(path)},
        )

    try:
        try:
            dataset = rasterio.open(path)
        except Exception as exc:
            raise GeoTiffOpenError(
                f"File is not a readable GeoTIFF: {path}",
                code="geotiff_unreadable_or_unsupported",
                details={"path": str(path), "error": str(exc)},
            ) from exc
        with dataset:
            if dataset.driver != "GTiff":
                raise GeoTiffOpenError(
                    f"File is not a readable GeoTIFF: {path}",
                    code="geotiff_unreadable_or_unsupported",
                    details={"path": str(path), "driver": dataset.driver},
                )
            if band_number > int(dataset.count):
                raise GeoTiffBandError(
                    f"GeoTIFF band {band_number} is out of range.",
                    code="geotiff_band_out_of_range",
                    details={"band": band_number, "band_count": int(dataset.count)},
                )
            dtype = np.dtype(dataset.dtypes[band_number - 1])
            if np.issubdtype(dtype, np.complexfloating):
                raise GeoTiffDataTypeError(
                    "Complex GeoTIFF datatypes are not supported in v0.1.",
                    details={"band": band_number, "dtype": str(dtype)},
                )
            values = np.asarray(dataset.read(band_number))
            if values.ndim != 2:
                raise GeoTiffOpenError(
                    "A single GeoTIFF band must produce a two-dimensional array.",
                    code="geotiff_invalid_array_shape",
                    details={"shape": list(values.shape)},
                )

            projection_wkt = dataset.crs.to_wkt() if dataset.crs is not None else ""
            if not projection_wkt or dataset.transform == Affine.identity():
                return values, None
            affine_tuple = tuple(float(value) for value in dataset.transform.to_gdal())
            if len(affine_tuple) != 6:
                raise GeoTiffMetadataError(
                    "Rasterio returned an invalid affine transform.",
                    details={"coefficient_count": len(affine_tuple)},
                )
            nodata = dataset.nodatavals[band_number - 1]
            nodata_tag = dataset.tags(band_number).get(_NODATA_TAG)
            if nodata_tag is not None and np.issubdtype(dtype, np.integer):
                nodata = int(nodata_tag)
            georef = GeoReference(
                projection_wkt=projection_wkt,
                projection_proj4=_projection_proj4(projection_wkt),
                affine_transform=affine_tuple,  # type: ignore[arg-type]
                width=int(dataset.width),
                height=int(dataset.height),
                pixel_size_x=float(affine_tuple[1]),
                pixel_size_y=float(affine_tuple[5]),
                nodata=nodata,
            )
            return values, georef
    except (GeoTiffOpenError, GeoTiffBandError, GeoTiffDataTypeError, GeoTiffMetadataError):
        raise
    except Exception as exc:
        raise GeoTiffOpenError(
            f"Unable to read GeoTIFF: {path}",
            code="geotiff_read_failed",
            details={"path": str(path), "band": band_number, "error": str(exc)},
        ) from exc


def _validate_geotiff_dtype(array: NDArray[Any]) -> None:
    if np.issubdtype(array.dtype, np.bool_):
        raise GeoTiffDataTypeError(
            "Boolean arrays must be converted to an explicit integer GeoTIFF datatype.",
            details={"dtype": str(array.dtype)},
        )
    if np.issubdtype(array.dtype, np.complexfloating):
        raise GeoTiffDataTypeError(
            "Complex NumPy datatypes are not supported in v0.1.",
            details={"dtype": str(array.dtype)},
        )
    if np.dtype(array.dtype) not in _SUPPORTED_GEOTIFF_DTYPES:
        raise GeoTiffDataTypeError(
            "The NumPy datatype cannot be represented by GeoTIFF.",
            details={"dtype": str(array.dtype)},
        )


def _validate_nodata(dtype: np.dtype[Any], nodata: int | float | None) -> None:
    if nodata is None:
        return
    if np.issubdtype(dtype, np.bool_):
        raise GeoTiffDataTypeError(
            "Boolean arrays are not a supported GeoTIFF output datatype.",
            details={"dtype": str(dtype)},
        )
    if np.issubdtype(dtype, np.integer):
        if isinstance(nodata, Integral):
            integer_nodata = int(nodata)
        elif isinstance(nodata, Real) and np.isfinite(nodata) and float(nodata).is_integer():
            integer_nodata = int(nodata)
        else:
            raise GeoTiffMetadataError(
                "Integer output nodata must be a finite integer value.",
                code="geotiff_unrepresentable_nodata",
                details={"dtype": str(dtype), "nodata": nodata},
            )
        limits = np.iinfo(dtype)
        if integer_nodata < int(limits.min) or integer_nodata > int(limits.max):
            raise GeoTiffMetadataError(
                "Output nodata cannot be represented by the array dtype.",
                code="geotiff_unrepresentable_nodata",
                details={"dtype": str(dtype), "nodata": nodata},
            )
        return
    if np.issubdtype(dtype, np.floating):
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(nodata, dtype=dtype).item()
        if np.isfinite(nodata) and not np.isfinite(converted):
            raise GeoTiffMetadataError(
                "Output nodata cannot be represented by the array dtype.",
                code="geotiff_unrepresentable_nodata",
                details={"dtype": str(dtype), "nodata": nodata},
            )
        return
    raise GeoTiffDataTypeError(
        "The NumPy datatype is not supported for GeoTIFF output.",
        details={"dtype": str(dtype)},
    )


def _creation_options(dtype: np.dtype[Any]) -> list[str]:
    predictor = "3" if np.issubdtype(dtype, np.floating) else "2"
    return [
        "TILED=YES",
        "BLOCKXSIZE=128",
        "BLOCKYSIZE=128",
        "COMPRESS=DEFLATE",
        f"PREDICTOR={predictor}",
        "BIGTIFF=IF_SAFER",
    ]


def write_geotiff(
    filename: str | Path,
    array: NDArray[Any],
    georef: GeoReference,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one NumPy array band as a tiled GeoTIFF."""

    path = Path(filename).expanduser().resolve()
    values = np.asarray(array)
    if values.ndim != 2:
        raise GeoTiffWriteError(
            "GeoTIFF output must be a two-dimensional NumPy array.",
            code="geotiff_invalid_array_shape",
            details={"shape": list(values.shape)},
        )
    expected_shape = (int(georef.height), int(georef.width))
    if values.shape != expected_shape:
        raise GeoTiffWriteError(
            "Array shape does not match GeoReference dimensions.",
            code="geotiff_shape_mismatch",
            details={"shape": list(values.shape), "expected_shape": list(expected_shape)},
        )
    _validate_geotiff_dtype(values)
    _validate_nodata(values.dtype, georef.nodata)
    if path.exists() and not overwrite:
        raise OutputExistsError(
            f"GeoTIFF output already exists: {path}",
            details={"path": str(path)},
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp.tif",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        profile = {
            "driver": "GTiff",
            "width": int(georef.width),
            "height": int(georef.height),
            "count": 1,
            "dtype": values.dtype,
            "crs": georef.projection_wkt,
            "transform": Affine.from_gdal(*georef.affine_transform),
            "tiled": True,
            "blockxsize": 128,
            "blockysize": 128,
            "compress": "deflate",
            "predictor": 3 if np.issubdtype(values.dtype, np.floating) else 2,
            "BIGTIFF": "IF_SAFER",
        }
        if georef.nodata is not None:
            profile["nodata"] = georef.nodata
        with rasterio.open(temporary_path, "w", **profile) as dataset:
            dataset.write(values, 1)
            if georef.nodata is not None and np.issubdtype(values.dtype, np.integer):
                dataset.update_tags(1, **{_NODATA_TAG: str(int(georef.nodata))})

        if path.exists() and not overwrite:
            raise OutputExistsError(
                f"GeoTIFF output was created concurrently: {path}",
                details={"path": str(path)},
            )
        os.replace(temporary_path, path)
        temporary_path = None
        return path
    except (GeoTiffWriteError, GeoTiffDataTypeError, GeoTiffMetadataError, OutputExistsError):
        raise
    except Exception as exc:
        raise GeoTiffWriteError(
            f"Unable to write GeoTIFF: {path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
