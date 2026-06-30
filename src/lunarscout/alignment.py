from __future__ import annotations

from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS
from rasterio.enums import Resampling as RasterioResampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from .errors import AlignmentError, GridMismatchError
from .georeference import GeoReference
from .geotiff import (
    _validate_nodata,
)


Resampling: TypeAlias = Literal[
    "nearest",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "median",
    "q1",
    "q3",
    "sum",
    "rms",
]
Nodata: TypeAlias = int | float | None | Literal["auto"]


_RESAMPLING_ALGORITHMS = (
    "nearest",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "median",
    "q1",
    "q3",
    "sum",
    "rms",
)

_RASTERIO_RESAMPLING_NAMES = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "cubic": "cubic",
    "cubicspline": "cubic_spline",
    "lanczos": "lanczos",
    "average": "average",
    "mode": "mode",
    "max": "max",
    "min": "min",
    "median": "med",
    "q1": "q1",
    "q3": "q3",
    "sum": "sum",
    "rms": "rms",
}

_GDAL_RESAMPLING_CONSTANTS = (
    ("nearest", "GRA_NearestNeighbour"),
    ("bilinear", "GRA_Bilinear"),
    ("cubic", "GRA_Cubic"),
    ("cubicspline", "GRA_CubicSpline"),
    ("lanczos", "GRA_Lanczos"),
    ("average", "GRA_Average"),
    ("mode", "GRA_Mode"),
    ("max", "GRA_Max"),
    ("min", "GRA_Min"),
    ("median", "GRA_Med"),
    ("q1", "GRA_Q1"),
    ("q3", "GRA_Q3"),
    ("sum", "GRA_Sum"),
    ("rms", "GRA_RMS"),
)


def available_resampling_algorithms() -> tuple[str, ...]:
    """Return the alignment algorithms supported by Rasterio."""

    return tuple(
        name
        for name in _RESAMPLING_ALGORITHMS
        if hasattr(RasterioResampling, _RASTERIO_RESAMPLING_NAMES[name])
    )


def _same_crs(left: GeoReference, right: GeoReference) -> bool:
    try:
        return CRS.from_wkt(left.projection_wkt) == CRS.from_wkt(right.projection_wkt)
    except Exception as exc:
        raise AlignmentError(
            "Unable to compare raster coordinate reference systems.",
            code="alignment_crs_comparison_failed",
            details={"error": str(exc)},
        ) from exc


def _grid_differences(
    left: GeoReference,
    right: GeoReference,
    *,
    affine_tolerance: float,
) -> dict[str, Any]:
    if not np.isfinite(affine_tolerance) or affine_tolerance < 0:
        raise AlignmentError(
            "Affine tolerance must be a finite, non-negative number.",
            code="alignment_invalid_tolerance",
            details={"affine_tolerance": affine_tolerance},
        )
    differences: dict[str, Any] = {}
    if left.width != right.width:
        differences["width"] = [left.width, right.width]
    if left.height != right.height:
        differences["height"] = [left.height, right.height]
    if not _same_crs(left, right):
        differences["crs"] = [left.projection_wkt, right.projection_wkt]
    left_affine = np.asarray(left.affine_transform, dtype=np.float64)
    right_affine = np.asarray(right.affine_transform, dtype=np.float64)
    affine_equal = np.array_equal(left_affine, right_affine)
    if affine_tolerance > 0:
        affine_equal = bool(
            np.allclose(left_affine, right_affine, rtol=0.0, atol=affine_tolerance)
        )
    if not affine_equal:
        differences["affine_transform"] = [
            list(left.affine_transform),
            list(right.affine_transform),
        ]
    return differences


def same_grid(
    left: GeoReference,
    right: GeoReference,
    *,
    affine_tolerance: float = 0.0,
) -> bool:
    """Return whether two references describe the same pixel grid.

    Nodata is deliberately ignored. Dimensions and the affine transform are
    exact by default; CRS comparison is semantic through GDAL/OSR.
    """

    return not _grid_differences(left, right, affine_tolerance=affine_tolerance)


def require_same_grid(
    left: GeoReference,
    right: GeoReference,
    *,
    affine_tolerance: float = 0.0,
) -> None:
    """Raise ``GridMismatchError`` unless two references use the same grid."""

    differences = _grid_differences(left, right, affine_tolerance=affine_tolerance)
    if differences:
        raise GridMismatchError(
            "Raster grids do not match; call align() explicitly.",
            details={"differences": differences},
        )


def _resampling_algorithm(name: str) -> RasterioResampling:
    normalized = str(name).strip().lower()
    rasterio_name = _RASTERIO_RESAMPLING_NAMES.get(normalized)
    if rasterio_name and hasattr(RasterioResampling, rasterio_name):
        return getattr(RasterioResampling, rasterio_name)
    raise AlignmentError(
        f"Unsupported resampling algorithm: {name}",
        code="alignment_invalid_resampling",
        details={
            "resampling": name,
            "available": list(available_resampling_algorithms()),
        },
    )


def align(
    source: NDArray[Any],
    source_georef: GeoReference,
    *,
    to: GeoReference,
    resampling: Resampling = "nearest",
    output_nodata: Nodata = "auto",
    output_dtype: np.dtype[Any] | type[Any] | str | None = None,
) -> tuple[NDArray[Any], GeoReference]:
    """Explicitly reproject/resample an array onto an exact destination grid."""

    values = np.asarray(source)
    expected_shape = (source_georef.height, source_georef.width)
    if values.ndim != 2:
        raise AlignmentError(
            "Alignment input must be a two-dimensional NumPy array.",
            code="alignment_invalid_array_shape",
            details={"shape": list(values.shape)},
        )
    if values.shape != expected_shape:
        raise AlignmentError(
            "Alignment input shape does not match its GeoReference dimensions.",
            code="alignment_source_shape_mismatch",
            details={"shape": list(values.shape), "expected_shape": list(expected_shape)},
        )

    try:
        dtype = values.dtype if output_dtype is None else np.dtype(output_dtype)
    except (TypeError, ValueError) as exc:
        raise AlignmentError(
            "Invalid alignment output dtype.",
            code="alignment_invalid_output_dtype",
            details={"output_dtype": str(output_dtype)},
        ) from exc
    typed_probe = np.empty(0, dtype=dtype)
    from .geotiff import _validate_geotiff_dtype

    _validate_geotiff_dtype(values)
    _validate_geotiff_dtype(typed_probe)

    if output_nodata == "auto":
        destination_nodata = source_georef.nodata
    elif output_nodata is None or isinstance(output_nodata, (int, float, np.integer, np.floating)):
        destination_nodata = output_nodata
    else:
        raise AlignmentError(
            "output_nodata must be 'auto', None, or a numeric value.",
            code="alignment_invalid_output_nodata",
            details={"output_nodata": output_nodata},
        )
    if isinstance(destination_nodata, (np.integer, np.floating)):
        destination_nodata = destination_nodata.item()
    _validate_nodata(dtype, destination_nodata)
    resampling_algorithm = _resampling_algorithm(resampling)
    try:
        fill_value = destination_nodata if destination_nodata is not None else 0
        aligned = np.full((to.height, to.width), fill_value, dtype=dtype)
        reproject(
            source=values,
            destination=aligned,
            src_transform=Affine.from_gdal(*source_georef.affine_transform),
            src_crs=source_georef.projection_wkt,
            src_nodata=source_georef.nodata,
            dst_transform=Affine.from_gdal(*to.affine_transform),
            dst_crs=to.projection_wkt,
            dst_nodata=destination_nodata,
            resampling=resampling_algorithm,
        )
        if destination_nodata is not None:
            rows, cols = np.indices((to.height, to.width), dtype=np.float64)
            eastings, northings = to.pixel_to_projected(cols, rows, anchor="center")
            assert isinstance(eastings, np.ndarray)
            assert isinstance(northings, np.ndarray)
            source_cols, source_rows = source_georef.projected_to_pixel(
                eastings,
                northings,
                anchor="center",
            )
            assert isinstance(source_cols, np.ndarray)
            assert isinstance(source_rows, np.ndarray)
            outside = (
                (source_cols < -0.5)
                | (source_cols > float(source_georef.width) - 0.5)
                | (source_rows < -0.5)
                | (source_rows > float(source_georef.height) - 0.5)
            )
            aligned[outside] = destination_nodata
        if aligned.dtype != dtype:
            aligned = aligned.astype(dtype, copy=False)
        return aligned, to.with_nodata(destination_nodata)
    except AlignmentError:
        raise
    except Exception as exc:
        raise AlignmentError(
            "GDAL could not align the raster to the destination grid.",
            code="alignment_failed",
            details={"error": str(exc), "resampling": resampling},
        ) from exc
