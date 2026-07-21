from __future__ import annotations

from typing import Literal

import numpy as np

from ..errors import MapAlgebraError
from ..georeference import GeoReference
from ..raster import Raster
from ._model import RasterExpression, _make_expr_node


Anchor = Literal["center", "corner"]


def row_indices(georef: GeoReference) -> RasterExpression:
    return _coordinate_expression(
        "coordinate.row_indices", georef, dtype=np.dtype(np.int64), units="pixels"
    )


def column_indices(georef: GeoReference) -> RasterExpression:
    return _coordinate_expression(
        "coordinate.column_indices", georef, dtype=np.dtype(np.int64), units="pixels"
    )


def projected_x(
    georef: GeoReference,
    *,
    anchor: Anchor = "center",
) -> RasterExpression:
    _anchor_offset(anchor)
    return _coordinate_expression(
        "coordinate.projected_x",
        georef,
        dtype=np.dtype(np.float64),
        units=_crs_axis_unit(georef, axis=0),
        params={"anchor": anchor},
    )


def projected_y(
    georef: GeoReference,
    *,
    anchor: Anchor = "center",
) -> RasterExpression:
    _anchor_offset(anchor)
    return _coordinate_expression(
        "coordinate.projected_y",
        georef,
        dtype=np.dtype(np.float64),
        units=_crs_axis_unit(georef, axis=1),
        params={"anchor": anchor},
    )


def longitude(
    georef: GeoReference,
    *,
    anchor: Anchor = "center",
) -> RasterExpression:
    _anchor_offset(anchor)
    return _coordinate_expression(
        "coordinate.longitude",
        georef,
        dtype=np.dtype(np.float64),
        units="degrees",
        params={"anchor": anchor},
    )


def latitude(
    georef: GeoReference,
    *,
    anchor: Anchor = "center",
) -> RasterExpression:
    _anchor_offset(anchor)
    return _coordinate_expression(
        "coordinate.latitude",
        georef,
        dtype=np.dtype(np.float64),
        units="degrees",
        params={"anchor": anchor},
    )


def _coordinate_expression(
    operation_id: str,
    georef: GeoReference,
    *,
    dtype: np.dtype,
    units: str | None,
    params: dict[str, object] | None = None,
) -> RasterExpression:
    if not isinstance(georef, GeoReference):
        raise MapAlgebraError(
            "Coordinate constructors require a GeoReference.",
            code="map_algebra_invalid_grid",
            details={"type": type(georef).__name__},
        )
    return _make_expr_node(
        operation_id,
        (),
        grid=georef.with_nodata(None),
        dtype=dtype,
        units=units,
        params=params,
    )


def _anchor_offset(anchor: str) -> float:
    if anchor == "center":
        return 0.5
    if anchor == "corner":
        return 0.0
    raise MapAlgebraError(
        f"Invalid anchor value: {anchor!r}. Use 'center' or 'corner'.",
        code="map_algebra_invalid_anchor",
        details={"anchor": anchor},
    )


def _crs_axis_unit(georef: GeoReference, *, axis: int) -> str | None:
    try:
        from pyproj import CRS

        axis_info = CRS.from_wkt(georef.projection_wkt).axis_info
    except Exception as exc:
        raise MapAlgebraError(
            "The grid CRS cannot be interpreted for coordinate generation.",
            code="map_algebra_invalid_crs",
            details={"error": str(exc)},
        ) from exc
    if axis >= len(axis_info):
        return None
    unit_name = str(axis_info[axis].unit_name or "").strip()
    return unit_name or None


def _compute_coordinate(expression: RasterExpression) -> Raster:
    georef = expression.grid
    if georef is None:
        raise MapAlgebraError(
            "Coordinate expression has no inferred grid.",
            code="map_algebra_invalid_grid",
        )
    operation_id = expression.operation_id
    anchor = str(expression._params_dict.get("anchor", "center"))
    offset = _anchor_offset(anchor)
    shape = (georef.height, georef.width)

    if operation_id == "coordinate.row_indices":
        values = np.broadcast_to(
            np.arange(georef.height, dtype=np.int64).reshape(-1, 1), shape
        ).copy()
    elif operation_id == "coordinate.column_indices":
        values = np.broadcast_to(
            np.arange(georef.width, dtype=np.int64).reshape(1, -1), shape
        ).copy()
    else:
        rows, cols = np.indices(shape, dtype=np.float64)
        affine = georef.affine_transform
        x_values = affine[0] + (cols + offset) * affine[1] + (rows + offset) * affine[2]
        y_values = affine[3] + (cols + offset) * affine[4] + (rows + offset) * affine[5]
        if operation_id == "coordinate.projected_x":
            values = x_values
        elif operation_id == "coordinate.projected_y":
            values = y_values
        elif operation_id in {"coordinate.longitude", "coordinate.latitude"}:
            values = _transform_to_geodetic(
                georef,
                x_values,
                y_values,
                longitude=operation_id == "coordinate.longitude",
            )
        else:
            raise MapAlgebraError(
                f"Unknown coordinate operation: {operation_id}",
                code="map_algebra_unknown_operation",
                details={"operation_id": operation_id},
            )

    valid = np.isfinite(values) if values.dtype.kind == "f" else np.ones(shape, dtype=np.bool_)
    return Raster(
        values=values,
        georef=georef,
        valid=valid,
        units=expression.units,
        name=operation_id.removeprefix("coordinate."),
    )


def _transform_to_geodetic(
    georef: GeoReference,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    longitude: bool,
) -> np.ndarray:
    try:
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise MapAlgebraError(
            "pyproj is required for longitude and latitude generation.",
            code="map_algebra_pyproj_required",
        ) from exc

    try:
        source_crs = CRS.from_wkt(georef.projection_wkt)
        geodetic_crs = source_crs.geodetic_crs
        if geodetic_crs is None:
            raise ValueError("CRS has no geodetic CRS")
        transformer = Transformer.from_crs(source_crs, geodetic_crs, always_xy=True)
        lon_values, lat_values = transformer.transform(x_values, y_values)
    except Exception as exc:
        raise MapAlgebraError(
            "Coordinates cannot be transformed to the grid's geodetic CRS.",
            code="map_algebra_unknown_geodetic_crs",
            details={"error": str(exc)},
        ) from exc
    selected = lon_values if longitude else lat_values
    return np.asarray(selected, dtype=np.float64).reshape(x_values.shape)
