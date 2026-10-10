from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .errors import GeoTiffError, GeoTiffMetadataError, TerrainOperationError
from .georeference import GeoReference
from .geotiff import _validate_geotiff_dtype, _validate_nodata


def _validate_source(array: NDArray[Any], georef: GeoReference) -> NDArray[Any]:
    values = np.asarray(array)
    if values.ndim != 2:
        raise TerrainOperationError(
            "Terrain operations require a two-dimensional NumPy array.",
            code="terrain_invalid_array_shape",
            details={"shape": list(values.shape)},
        )
    expected_shape = (int(georef.height), int(georef.width))
    if values.shape != expected_shape:
        raise TerrainOperationError(
            "Array shape does not match GeoReference dimensions.",
            code="terrain_shape_mismatch",
            details={"shape": list(values.shape), "expected_shape": list(expected_shape)},
        )
    if np.issubdtype(values.dtype, np.bool_) or np.issubdtype(
        values.dtype,
        np.complexfloating,
    ):
        raise TerrainOperationError(
            "Terrain source values must use a real numeric NumPy dtype.",
            code="terrain_unsupported_datatype",
            details={"dtype": str(values.dtype)},
        )
    return values


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise TerrainOperationError(
            f"{name} must be finite and greater than zero.",
            code="terrain_invalid_argument",
            details={"argument": name, "value": value},
        )
    return result


def _replace_nodata(
    values: NDArray[Any],
    *,
    native_nodata: int | float | None,
    output_nodata: int | float,
) -> NDArray[Any]:
    try:
        _validate_nodata(values.dtype, output_nodata)
    except GeoTiffMetadataError as exc:
        raise TerrainOperationError(
            "Output nodata cannot be represented by the terrain output dtype.",
            code="terrain_unrepresentable_output_nodata",
            details={
                "dtype": str(values.dtype),
                "output_nodata": output_nodata,
                "cause": exc.code,
            },
        ) from exc
    if native_nodata is None:
        return values
    if isinstance(native_nodata, float) and np.isnan(native_nodata):
        invalid = np.isnan(values)
    else:
        invalid = values == native_nodata
    if not np.any(invalid):
        return values
    output = np.array(values, copy=True)
    output[invalid] = output_nodata
    return output


def _prepared_elevation(
    array: NDArray[Any],
    georef: GeoReference,
    *,
    output_nodata: int | float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = _validate_source(array, georef)
    try:
        _validate_geotiff_dtype(values)
        _validate_nodata(values.dtype, georef.nodata)
    except GeoTiffError as exc:
        raise TerrainOperationError(
            "Terrain source dtype or nodata is not representable as a GeoTIFF.",
            code="terrain_unsupported_source",
            details={
                "dtype": str(values.dtype),
                "nodata": georef.nodata,
                "cause": exc.code,
            },
        ) from exc
    _validate_nodata(np.dtype(np.float32), output_nodata)
    elevation = np.asarray(values, dtype=np.float64)
    invalid = ~np.isfinite(elevation)
    if georef.nodata is not None:
        if isinstance(georef.nodata, float) and np.isnan(georef.nodata):
            invalid |= np.isnan(elevation)
        else:
            invalid |= elevation == georef.nodata
    elevation = np.array(elevation, copy=True)
    elevation[invalid] = np.nan
    neighborhood_invalid = np.array(invalid, copy=True)
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            src_rows = slice(max(0, -row_offset), invalid.shape[0] - max(0, row_offset))
            src_cols = slice(max(0, -col_offset), invalid.shape[1] - max(0, col_offset))
            dst_rows = slice(max(0, row_offset), invalid.shape[0] - max(0, -row_offset))
            dst_cols = slice(max(0, col_offset), invalid.shape[1] - max(0, -col_offset))
            neighborhood_invalid[dst_rows, dst_cols] |= invalid[src_rows, src_cols]
    return elevation, neighborhood_invalid


def _require_supported_grid(georef: GeoReference, *, scale: float) -> None:
    transform = georef.affine_transform
    if float(transform[2]) != 0.0 or float(transform[4]) != 0.0:
        raise TerrainOperationError(
            "Terrain operations require a north-up or south-up grid without rotation.",
            code="terrain_rotated_grid_unsupported",
            details={"affine_transform": list(transform)},
        )
    if scale == 1.0:
        from pyproj import CRS

        try:
            geographic = CRS.from_wkt(georef.projection_wkt).is_geographic
        except Exception:  # noqa: BLE001 - an unreadable CRS is not rejected here
            geographic = False
        if geographic:
            raise TerrainOperationError(
                "Terrain operations on a geographic (degree) grid need scale, the "
                "ratio of horizontal units to elevation units (as gdaldem -s).",
                code="terrain_geographic_grid_requires_scale",
            )


def _horn_window_sums(
    padded: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Horn's weighted differences over every 3x3 window of a padded grid.

    Returns (right - left, bottom - top) column and row differences in
    elevation units, not yet divided by 8 x pixel size.
    """
    a, b, c = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    d, f = padded[1:-1, :-2], padded[1:-1, 2:]
    g, h, i = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
    return (c + 2.0 * f + i) - (a + 2.0 * d + g), (g + 2.0 * h + i) - (a + 2.0 * b + c)


def _gradient(elevation: NDArray[np.float64], georef: GeoReference) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Horn (gdaldem) gradient: dz/d(east) and dz/d(south) for every pixel.

    Missing neighbours at the raster edge follow gdaldem -compute_edges:
    each is linearly extrapolated (2 x edge - inner) along its row or column,
    except in the four corner windows, where rows are extrapolated and
    columns are replicated. Edge values are discarded unless compute_edges
    is requested. Signed pixel sizes make the result physical on north-up
    and south-up grids; on north-up grids it equals gdaldem's.
    """
    xres = float(georef.pixel_size_x)
    yres = float(georef.pixel_size_y)
    if xres == 0.0 or yres == 0.0 or not np.isfinite(xres) or not np.isfinite(yres):
        raise TerrainOperationError(
            "Terrain operations require non-zero pixel sizes.",
            code="terrain_invalid_georeference",
        )
    height, width = elevation.shape
    padded = np.empty((height + 2, width + 2), dtype=np.float64)
    padded[1:-1, 1:-1] = elevation
    if height >= 2:
        padded[0, 1:-1] = 2.0 * elevation[0] - elevation[1]
        padded[-1, 1:-1] = 2.0 * elevation[-1] - elevation[-2]
    else:
        padded[0, 1:-1] = padded[-1, 1:-1] = elevation[0]
    if width >= 2:
        padded[:, 0] = 2.0 * padded[:, 1] - padded[:, 2]
        padded[:, -1] = 2.0 * padded[:, -2] - padded[:, -3]
    else:
        padded[:, 0] = padded[:, -1] = padded[:, 1]
    column_difference, row_difference = _horn_window_sums(padded)
    if height >= 2 and width >= 2:
        for row, inner in ((0, 1), (height - 1, height - 2)):
            for column in (0, width - 1):
                columns = [max(column - 1, 0), column, min(column + 1, width - 1)]
                edge = elevation[row, columns]
                beyond = 2.0 * edge - elevation[inner, columns]
                window = np.vstack(
                    (beyond, edge, elevation[inner, columns]) if row == 0
                    else (elevation[inner, columns], edge, beyond)
                )
                corner_columns, corner_rows = _horn_window_sums(np.pad(window, 1))
                column_difference[row, column] = corner_columns[1, 1]
                row_difference[row, column] = corner_rows[1, 1]
    # Pixel columns run east when xres > 0; rows run south when yres < 0.
    dz_east = column_difference / (8.0 * xres)
    dz_south = row_difference / (8.0 * -yres)
    return dz_east, dz_south


def _apply_invalid_edges(
    output: NDArray[Any],
    invalid: NDArray[np.bool_],
    *,
    output_nodata: int | float,
    compute_edges: bool,
) -> NDArray[Any]:
    mask = np.array(invalid, copy=True)
    if not compute_edges:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    result = np.array(output, copy=True)
    result[mask] = output_nodata
    return result


def slope(
    array: NDArray[Any],
    georef: GeoReference,
    *,
    output_nodata: int | float,
    units: Literal["degrees", "percent"] = "degrees",
    compute_edges: bool = False,
    scale: float = 1.0,
) -> tuple[NDArray[np.float32], GeoReference]:
    """Calculate GDAL-compatible Horn slope from an elevation array."""

    if units not in {"degrees", "percent"}:
        raise TerrainOperationError(
            "Slope units must be 'degrees' or 'percent'.",
            code="terrain_invalid_argument",
            details={"argument": "units", "value": units},
        )
    elevation, invalid = _prepared_elevation(array, georef, output_nodata=output_nodata)
    scale_value = _positive_finite(scale, name="scale")
    _require_supported_grid(georef, scale=scale_value)
    dzdx, dzdy = _gradient(elevation / scale_value, georef)
    rise_run = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    if units == "degrees":
        output = np.degrees(np.arctan(rise_run)).astype(np.float32)
    else:
        output = (rise_run * 100.0).astype(np.float32)
    return (
        _apply_invalid_edges(
            output,
            invalid,
            output_nodata=output_nodata,
            compute_edges=bool(compute_edges),
        ).astype(np.float32),
        georef.with_nodata(output_nodata),
    )


def aspect(
    array: NDArray[Any],
    georef: GeoReference,
    *,
    output_nodata: int | float,
    compute_edges: bool = False,
) -> tuple[NDArray[np.float32], GeoReference]:
    """Calculate GDAL-compatible Horn aspect in azimuth degrees."""

    elevation, invalid = _prepared_elevation(array, georef, output_nodata=output_nodata)
    _require_supported_grid(georef, scale=2.0)  # aspect is independent of scale
    dzdx, dzdy = _gradient(elevation, georef)
    aspect_degrees = (90.0 - np.degrees(np.arctan2(dzdy, -dzdx))) % 360.0
    flat = np.isclose(dzdx, 0.0) & np.isclose(dzdy, 0.0)
    aspect_degrees[flat] = output_nodata
    invalid = invalid | flat
    return (
        _apply_invalid_edges(
            aspect_degrees.astype(np.float32),
            invalid,
            output_nodata=output_nodata,
            compute_edges=bool(compute_edges),
        ).astype(np.float32),
        georef.with_nodata(output_nodata),
    )


def hillshade(
    array: NDArray[Any],
    georef: GeoReference,
    *,
    output_nodata: int | float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    compute_edges: bool = False,
    scale: float = 1.0,
    z_factor: float = 1.0,
) -> tuple[NDArray[np.uint8], GeoReference]:
    """Calculate GDAL-compatible single-direction hillshade."""

    azimuth_value = float(azimuth)
    altitude_value = float(altitude)
    if not np.isfinite(azimuth_value) or not 0.0 <= azimuth_value <= 360.0:
        raise TerrainOperationError(
            "Hillshade azimuth must be finite and between 0 and 360 degrees.",
            code="terrain_invalid_argument",
            details={"argument": "azimuth", "value": azimuth},
        )
    if not np.isfinite(altitude_value) or not 0.0 <= altitude_value <= 90.0:
        raise TerrainOperationError(
            "Hillshade altitude must be finite and between 0 and 90 degrees.",
            code="terrain_invalid_argument",
            details={"argument": "altitude", "value": altitude},
        )
    try:
        _validate_nodata(np.dtype(np.uint8), output_nodata)
    except GeoTiffMetadataError as exc:
        raise TerrainOperationError(
            "Output nodata cannot be represented by the terrain output dtype.",
            code="terrain_unrepresentable_output_nodata",
            details={"dtype": "uint8", "output_nodata": output_nodata},
        ) from exc
    elevation, invalid = _prepared_elevation(array, georef, output_nodata=output_nodata)
    scale_value = _positive_finite(scale, name="scale")
    _require_supported_grid(georef, scale=scale_value)
    z_scale = _positive_finite(z_factor, name="z_factor") / scale_value
    dzdx, dzdy = _gradient(elevation * z_scale, georef)
    slope_rad = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    aspect_rad = np.arctan2(dzdy, -dzdx)
    azimuth_rad = np.radians(360.0 - azimuth_value + 90.0)
    altitude_rad = np.radians(altitude_value)
    shaded = (
        np.sin(altitude_rad) * np.cos(slope_rad)
        + np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    # gdaldem: 1 + 254 x cos(incidence), and 1 where the cell faces away.
    output = np.rint(np.where(shaded <= 0.0, 1.0, 1.0 + 254.0 * shaded)).astype(np.uint8)
    return (
        _apply_invalid_edges(
            output,
            invalid,
            output_nodata=output_nodata,
            compute_edges=bool(compute_edges),
        ).astype(np.uint8),
        georef.with_nodata(output_nodata),
    )
