from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..alignment import _resampling_algorithm
from ..errors import (
    AlignmentError,
    GeoTiffError,
    MapAlgebraExpressionError,
    TerrainOperationError,
)
from ..georeference import GeoReference
from ..raster import Raster
from ._model import RasterExpression, _make_expr_node


_TERRAIN_OPERATION_IDS = frozenset({
    "terrain.slope",
    "terrain.aspect",
    "terrain.hillshade",
})

_RESAMPLING_SUPPORT = {
    "nearest": 1,
    "bilinear": 1,
    "cubic": 2,
    "cubicspline": 2,
    "lanczos": 3,
    "average": 1,
    "mode": 1,
    "max": 1,
    "min": 1,
    "median": 1,
    "q1": 1,
    "q3": 1,
    "sum": 1,
    "rms": 1,
}


def make_terrain_expression(
    operation_id: str,
    raster: RasterExpression,
    **parameters: Any,
) -> RasterExpression:
    """Construct a validated terrain expression for public dispatch wrappers."""
    if operation_id not in _TERRAIN_OPERATION_IDS:
        raise MapAlgebraExpressionError(
            f"Unknown terrain expression operation: {operation_id}",
            code="map_algebra_unknown_operation",
            details={"operation_id": operation_id},
        )
    if raster.grid is None or raster.dtype is None:
        raise MapAlgebraExpressionError(
            "Terrain expressions require an inferred grid and dtype.",
            code="map_algebra_missing_inference",
            details={"operation_id": operation_id},
        )
    if raster.dtype.kind not in "iuf":
        raise TerrainOperationError(
            "Terrain source values must use a real numeric NumPy dtype.",
            code="terrain_unsupported_datatype",
            details={"dtype": str(raster.dtype)},
        )
    from ..geotiff import _validate_geotiff_dtype

    try:
        _validate_geotiff_dtype(np.empty(0, dtype=raster.dtype))
    except GeoTiffError as exc:
        raise TerrainOperationError(
            "Terrain source dtype cannot be represented by the scientific terrain kernel.",
            code="terrain_unsupported_source",
            details={"dtype": str(raster.dtype), "cause": exc.code},
        ) from exc

    params = _normalize_terrain_parameters(operation_id, parameters)
    if operation_id == "terrain.slope":
        units = str(params["units"])
        dtype = np.dtype(np.float32)
    elif operation_id == "terrain.aspect":
        units = "degrees"
        dtype = np.dtype(np.float32)
    else:
        units = None
        dtype = np.dtype(np.uint8)
    output_grid = raster.grid.with_nodata(params["output_nodata"])
    return _make_expr_node(
        operation_id,
        (raster,),
        grid=output_grid,
        dtype=dtype,
        units=units,
        halo=1,
        params=params,
    )


def make_resample_expression(
    raster: RasterExpression,
    grid: GeoReference,
    *,
    resampling: str = "nearest",
    output_dtype: Any = None,
    validity_coverage_threshold: float | None = None,
) -> RasterExpression:
    """Construct a validated, explicit cross-grid resampling expression."""
    if not isinstance(grid, GeoReference):
        raise AlignmentError(
            "resample_to() requires a GeoReference destination grid.",
            code="alignment_invalid_destination_grid",
            details={"type": type(grid).__name__},
        )
    normalized_resampling = str(resampling).strip().lower()
    _resampling_algorithm(normalized_resampling)
    try:
        dtype = raster.dtype if output_dtype is None else np.dtype(output_dtype)
    except (TypeError, ValueError) as exc:
        raise AlignmentError(
            "Invalid resampling output dtype.",
            code="alignment_invalid_output_dtype",
            details={"output_dtype": str(output_dtype)},
        ) from exc
    if dtype is None:
        raise MapAlgebraExpressionError(
            "Resampling expressions require an inferred source dtype.",
            code="map_algebra_missing_inference",
            details={"operation_id": "alignment.resample_to"},
        )
    from ..geotiff import _validate_nodata

    _validate_nodata(dtype, grid.nodata)
    threshold: float | None = None
    if validity_coverage_threshold is not None:
        try:
            threshold = float(validity_coverage_threshold)
        except (TypeError, ValueError) as exc:
            raise AlignmentError(
                "Validity coverage threshold must be a real number between zero and one.",
                code="alignment_invalid_validity_threshold",
                details={"validity_coverage_threshold": validity_coverage_threshold},
            ) from exc
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise AlignmentError(
                "Validity coverage threshold must be between zero and one.",
                code="alignment_invalid_validity_threshold",
                details={"validity_coverage_threshold": validity_coverage_threshold},
            )
    return _make_expr_node(
        "alignment.resample_to",
        (raster,),
        grid=grid,
        dtype=dtype,
        units=raster.units,
        params={
            "resampling": normalized_resampling,
            "output_dtype": dtype,
            "validity_coverage_threshold": threshold,
        },
    )


def _normalize_terrain_parameters(
    operation_id: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    unexpected = set(parameters)
    normalized: dict[str, Any] = {}
    compute_edges = parameters.get("compute_edges", False)
    unexpected.discard("compute_edges")
    normalized["compute_edges"] = bool(compute_edges)

    if operation_id == "terrain.hillshade":
        output_nodata = parameters.get("output_nodata", 0)
        output_dtype = np.dtype(np.uint8)
    else:
        output_nodata = parameters.get("output_nodata", np.nan)
        output_dtype = np.dtype(np.float32)
    unexpected.discard("output_nodata")
    from ..geotiff import _validate_nodata

    try:
        _validate_nodata(output_dtype, output_nodata)
    except (GeoTiffError, TypeError, ValueError) as exc:
        raise TerrainOperationError(
            "Output nodata cannot be represented by the terrain output dtype.",
            code="terrain_unrepresentable_output_nodata",
            details={"dtype": str(output_dtype), "output_nodata": output_nodata},
        ) from exc
    normalized["output_nodata"] = output_nodata

    if operation_id == "terrain.slope":
        units = parameters.get("units", "degrees")
        unexpected.discard("units")
        if units not in {"degrees", "percent"}:
            raise TerrainOperationError(
                "Slope units must be 'degrees' or 'percent'.",
                code="terrain_invalid_argument",
                details={"argument": "units", "value": units},
            )
        normalized["units"] = units
        normalized["scale"] = _positive_finite_parameter(parameters, "scale", 1.0)
        unexpected.discard("scale")
    elif operation_id == "terrain.hillshade":
        normalized["scale"] = _positive_finite_parameter(parameters, "scale", 1.0)
        normalized["z_factor"] = _positive_finite_parameter(parameters, "z_factor", 1.0)
        unexpected.difference_update({"scale", "z_factor"})
        for name, default, lower, upper in (
            ("azimuth", 315.0, 0.0, 360.0),
            ("altitude", 45.0, 0.0, 90.0),
        ):
            raw_value = parameters.get(name, default)
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise TerrainOperationError(
                    f"Hillshade {name} must be a finite real number.",
                    code="terrain_invalid_argument",
                    details={"argument": name, "value": raw_value},
                ) from exc
            unexpected.discard(name)
            if not np.isfinite(value) or not lower <= value <= upper:
                raise TerrainOperationError(
                    f"Hillshade {name} must be finite and between {lower:g} and {upper:g} degrees.",
                    code="terrain_invalid_argument",
                    details={"argument": name, "value": parameters.get(name, default)},
                )
            normalized[name] = value
    if unexpected:
        name = sorted(unexpected)[0]
        raise TypeError(f"Unexpected terrain parameter: {name}")
    return normalized


def _positive_finite_parameter(
    parameters: dict[str, Any],
    name: str,
    default: float,
) -> float:
    raw_value = parameters.get(name, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise TerrainOperationError(
            f"{name} must be finite and greater than zero.",
            code="terrain_invalid_argument",
            details={"argument": name, "value": raw_value},
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise TerrainOperationError(
            f"{name} must be finite and greater than zero.",
            code="terrain_invalid_argument",
            details={"argument": name, "value": parameters.get(name, default)},
        )
    return value


def evaluate_terrain(node: RasterExpression, source: Raster) -> Raster:
    """Evaluate one terrain node through the existing scientific kernels."""
    from ..terrain import aspect, hillshade, slope

    params = node._params_dict
    values = np.asarray(source.values, dtype=np.float64).copy()
    values[~source.valid] = np.nan
    source_grid = replace(source.georef, nodata=np.nan)
    compute_edges = bool(params.get("compute_edges", False))
    output_nodata = params["output_nodata"]

    if node._operation_id == "terrain.slope":
        output, _ = slope(
            values,
            source_grid,
            output_nodata=np.nan,
            units=params["units"],
            compute_edges=compute_edges,
            scale=params["scale"],
        )
        valid = np.isfinite(output)
    elif node._operation_id == "terrain.aspect":
        output, _ = aspect(
            values,
            source_grid,
            output_nodata=np.nan,
            compute_edges=compute_edges,
        )
        valid = np.isfinite(output)
    elif node._operation_id == "terrain.hillshade":
        with np.errstate(invalid="ignore"):
            output, _ = hillshade(
                values,
                source_grid,
                output_nodata=0,
                azimuth=params["azimuth"],
                altitude=params["altitude"],
                compute_edges=compute_edges,
                scale=params["scale"],
                z_factor=params["z_factor"],
            )
        valid = _terrain_validity(source.valid, source.values, compute_edges=compute_edges)
    else:
        raise MapAlgebraExpressionError(
            f"Unsupported terrain operation: {node._operation_id}",
            code="map_algebra_expression_eval_failed",
            details={"operation_id": node._operation_id},
        )
    output = np.asarray(output).copy()
    output[~valid] = output_nodata
    return Raster(
        values=output,
        georef=source.georef.with_nodata(output_nodata),
        valid=valid,
        units=node.units,
        name=source.name,
        validity_provenance="terrain-neighborhood",
    )


def _terrain_validity(
    source_valid: NDArray[np.bool_],
    source_values: NDArray[Any],
    *,
    compute_edges: bool,
) -> NDArray[np.bool_]:
    invalid = ~source_valid | ~np.isfinite(np.asarray(source_values, dtype=np.float64))
    padded = np.pad(invalid, 1, mode="constant", constant_values=False)
    neighborhood_invalid = np.zeros_like(invalid)
    for row_offset in range(3):
        for column_offset in range(3):
            neighborhood_invalid |= padded[
                row_offset : row_offset + invalid.shape[0],
                column_offset : column_offset + invalid.shape[1],
            ]
    if not compute_edges:
        neighborhood_invalid[0, :] = True
        neighborhood_invalid[-1, :] = True
        neighborhood_invalid[:, 0] = True
        neighborhood_invalid[:, -1] = True
    return ~neighborhood_invalid


def source_window_for_resampling(
    source_grid: GeoReference,
    destination_grid: GeoReference,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    resampling: str,
) -> tuple[int, int, int, int] | None:
    """Return a conservative source-pixel window for one destination window."""
    from rasterio.warp import transform_bounds

    destination_window = _window_grid(destination_grid, x0, y0, width, height)
    corners_x, corners_y = _grid_outer_corners(destination_window)
    left = float(np.min(corners_x))
    right = float(np.max(corners_x))
    bottom = float(np.min(corners_y))
    top = float(np.max(corners_y))
    try:
        src_left, src_bottom, src_right, src_top = transform_bounds(
            destination_grid.projection_wkt,
            source_grid.projection_wkt,
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
    except Exception as exc:
        raise AlignmentError(
            "Unable to map a destination window into the source grid.",
            code="alignment_window_transform_failed",
            details={"x": x0, "y": y0, "width": width, "height": height},
        ) from exc

    envelope_x = np.asarray([src_left, src_right, src_right, src_left])
    envelope_y = np.asarray([src_top, src_top, src_bottom, src_bottom])
    if not np.all(np.isfinite(envelope_x)) or not np.all(np.isfinite(envelope_y)):
        raise AlignmentError(
            "Destination window transforms outside the finite source CRS domain.",
            code="alignment_window_transform_failed",
            details={"x": x0, "y": y0, "width": width, "height": height},
        )
    columns, rows = source_grid.projected_to_pixel(envelope_x, envelope_y, anchor="corner")
    columns = np.asarray(columns, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    support = _RESAMPLING_SUPPORT[resampling]
    request_x0 = int(np.floor(np.min(columns))) - support
    request_y0 = int(np.floor(np.min(rows))) - support
    request_x1 = int(np.ceil(np.max(columns))) + support
    request_y1 = int(np.ceil(np.max(rows))) + support
    clipped_x0 = max(0, request_x0)
    clipped_y0 = max(0, request_y0)
    clipped_x1 = min(source_grid.width, request_x1)
    clipped_y1 = min(source_grid.height, request_y1)
    if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
        return None
    return (
        clipped_x0,
        clipped_y0,
        clipped_x1 - clipped_x0,
        clipped_y1 - clipped_y0,
    )


def evaluate_resample(
    node: RasterExpression,
    source: Raster | None,
    destination_grid: GeoReference,
) -> Raster:
    """Resample a bounded source raster into one exact destination grid."""
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    params = node._params_dict
    dtype = node.dtype or (source.dtype if source is not None else np.dtype(np.float64))
    output = np.zeros((destination_grid.height, destination_grid.width), dtype=dtype)
    valid = np.zeros(output.shape, dtype=np.bool_)
    if source is None:
        return Raster(output, destination_grid, valid=valid, units=node.units)

    algorithm = _resampling_algorithm(params["resampling"])
    try:
        if params["resampling"] == "nearest":
            output, nearest_valid = _resample_nearest_exact(
                source, destination_grid, dtype=dtype,
            )
        else:
            source_values = source.values
            if source_values.dtype.kind == "b":
                # GDAL has no Boolean raster datatype. An explicitly
                # overridden Boolean interpolation is evaluated numerically.
                source_values = source_values.astype(np.uint8)
            source_data = np.ma.array(source_values, mask=~source.valid)
            destination_nodata: int | float = np.nan if dtype.kind == "f" else 0
            warp_output = (
                np.zeros(output.shape, dtype=np.uint8)
                if dtype.kind == "b"
                else output
            )
            reproject(
                source=source_data,
                destination=warp_output,
                src_transform=Affine.from_gdal(*source.georef.affine_transform),
                src_crs=source.georef.projection_wkt,
                src_nodata=None,
                dst_transform=Affine.from_gdal(*destination_grid.affine_transform),
                dst_crs=destination_grid.projection_wkt,
                dst_nodata=destination_nodata,
                resampling=algorithm,
            )
            if dtype.kind == "b":
                output = warp_output.astype(np.bool_)
            nearest_valid = None
        threshold = params.get("validity_coverage_threshold")
        if threshold is None:
            if nearest_valid is not None:
                valid = nearest_valid
            else:
                validity_values = np.zeros(output.shape, dtype=np.uint8)
                reproject(
                    source=source.valid.astype(np.uint8),
                    destination=validity_values,
                    src_transform=Affine.from_gdal(*source.georef.affine_transform),
                    src_crs=source.georef.projection_wkt,
                    src_nodata=0,
                    dst_transform=Affine.from_gdal(*destination_grid.affine_transform),
                    dst_crs=destination_grid.projection_wkt,
                    dst_nodata=0,
                    resampling=_resampling_algorithm("nearest"),
                )
                valid = validity_values != 0
        else:
            coverage = np.zeros(output.shape, dtype=np.float32)
            reproject(
                source=source.valid.astype(np.float32),
                destination=coverage,
                src_transform=Affine.from_gdal(*source.georef.affine_transform),
                src_crs=source.georef.projection_wkt,
                src_nodata=None,
                dst_transform=Affine.from_gdal(*destination_grid.affine_transform),
                dst_crs=destination_grid.projection_wkt,
                dst_nodata=0.0,
                resampling=_resampling_algorithm("average"),
            )
            valid = (coverage > 0.0) & (coverage >= float(threshold))
    except Exception as exc:
        raise AlignmentError(
            "GDAL could not resample the requested raster window.",
            code="alignment_failed",
            details={"error": str(exc), "resampling": params["resampling"]},
        ) from exc
    return Raster(
        values=output,
        georef=destination_grid,
        valid=valid,
        units=node.units,
        name=source.name,
        validity_provenance="resampled-validity",
    )


def _resample_nearest_exact(
    source: Raster,
    destination_grid: GeoReference,
    *,
    dtype: np.dtype[Any],
) -> tuple[NDArray[Any], NDArray[np.bool_]]:
    """Nearest-neighbor resampling without GDAL's lossy uint64 conversion."""
    from pyproj import CRS, Transformer

    rows, columns = np.indices(
        (destination_grid.height, destination_grid.width), dtype=np.float64,
    )
    x_values, y_values = destination_grid.pixel_to_projected(
        columns, rows, anchor="center",
    )
    destination_crs = CRS.from_wkt(destination_grid.projection_wkt)
    source_crs = CRS.from_wkt(source.georef.projection_wkt)
    if destination_crs != source_crs:
        transformer = Transformer.from_crs(destination_crs, source_crs, always_xy=True)
        x_values, y_values = transformer.transform(x_values, y_values)
    source_columns, source_rows = source.georef.projected_to_pixel(
        x_values, y_values, anchor="corner",
    )
    source_columns_float = np.asarray(source_columns, dtype=np.float64)
    source_rows_float = np.asarray(source_rows, dtype=np.float64)
    finite = np.isfinite(source_columns_float) & np.isfinite(source_rows_float)
    source_columns = np.zeros(source_columns_float.shape, dtype=np.int64)
    source_rows = np.zeros(source_rows_float.shape, dtype=np.int64)
    source_columns[finite] = np.floor(source_columns_float[finite]).astype(np.int64)
    source_rows[finite] = np.floor(source_rows_float[finite]).astype(np.int64)
    inside = (
        finite
        & (source_columns >= 0)
        & (source_columns < source.georef.width)
        & (source_rows >= 0)
        & (source_rows < source.georef.height)
    )
    output = np.zeros(inside.shape, dtype=dtype)
    valid = np.zeros(inside.shape, dtype=np.bool_)
    output[inside] = source.values[source_rows[inside], source_columns[inside]].astype(
        dtype, copy=False,
    )
    valid[inside] = source.valid[source_rows[inside], source_columns[inside]]
    return output, valid


def _grid_outer_corners(grid: GeoReference) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    columns = np.asarray([0.0, grid.width, grid.width, 0.0])
    rows = np.asarray([0.0, 0.0, grid.height, grid.height])
    x_values, y_values = grid.pixel_to_projected(columns, rows, anchor="corner")
    return np.asarray(x_values, dtype=np.float64), np.asarray(y_values, dtype=np.float64)


def _window_grid(
    grid: GeoReference,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> GeoReference:
    origin_x, pixel_x, rotation_x, origin_y, rotation_y, pixel_y = grid.affine_transform
    return replace(
        grid,
        affine_transform=(
            origin_x + x0 * pixel_x + y0 * rotation_x,
            pixel_x,
            rotation_x,
            origin_y + x0 * rotation_y + y0 * pixel_y,
            rotation_y,
            pixel_y,
        ),
        width=width,
        height=height,
        nodata=None,
    )
