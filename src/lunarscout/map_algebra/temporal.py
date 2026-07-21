from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..errors import MapAlgebraExpressionError, MapAlgebraError
from ..georeference import GeoReference
from ..raster import Raster
from ..temporal import TemporalCube
from ..temporal_store import TemporalGeoTiffSeries
from ._model import RasterExpression, _make_expr_node
from ._temporal_model import (
    TemporalRaster,
    TemporalRasterExpression,
    _compute_temporal,
    _iter_temporal_layers,
    _temporal_constant,
    _temporal_local_op,
    _temporal_source_node,
    _reduction_output_dtype,
    _reduction_output_units,
    from_temporal_cube,
)
from . import _validity as _val


# ---------------------------------------------------------------------------
# temporal_source
# ---------------------------------------------------------------------------

def temporal_source(
    source: TemporalGeoTiffSeries | TemporalRaster | TemporalCube | str | Path,
    *,
    units: str | None = None,
) -> TemporalRasterExpression:
    if isinstance(source, TemporalRasterExpression):
        return source

    if isinstance(source, TemporalRaster):
        return _temporal_constant(source)

    if isinstance(source, TemporalCube):
        tr = from_temporal_cube(source, units=units)
        return _temporal_constant(tr)

    if isinstance(source, (str, Path)):
        from ..temporal_store import open_temporal_cube
        series = open_temporal_cube(Path(source))
        try:
            units_val = units if units is not None else series.units
            return _temporal_source_node(
                str(Path(source).resolve()),
                georef=series.georef,
                dtype=series.dtype,
                times=series.times,
                signal_name=series.signal_name,
                units=units_val,
            )
        finally:
            series.close()

    if isinstance(source, TemporalGeoTiffSeries):
        units_val = units if units is not None else source.units
        return _temporal_source_node(
            str(source.root),
            georef=source.georef,
            dtype=source.dtype,
            times=source.times,
            signal_name=source.signal_name,
            units=units_val,
        )

    raise MapAlgebraError(
        f"temporal_source() requires a TemporalGeoTiffSeries, TemporalRaster, TemporalCube, "
        f"or series path, got {type(source).__name__}.",
        code="map_algebra_invalid_temporal_source",
        details={"type": type(source).__name__},
    )


# ---------------------------------------------------------------------------
# Temporal reductions -- produce RasterExpression (spatial result)
# ---------------------------------------------------------------------------

def temporal_mean(expr: TemporalRasterExpression | TemporalRaster) -> RasterExpression | Raster:
    if isinstance(expr, TemporalRaster):
        return _eager_temporal_reduce(expr, "mean")
    return _temporal_reduction_spatial("temporal.mean", expr)


def temporal_min(expr: TemporalRasterExpression | TemporalRaster) -> RasterExpression | Raster:
    if isinstance(expr, TemporalRaster):
        return _eager_temporal_reduce(expr, "min")
    return _temporal_reduction_spatial("temporal.min", expr)


def temporal_max(expr: TemporalRasterExpression | TemporalRaster) -> RasterExpression | Raster:
    if isinstance(expr, TemporalRaster):
        return _eager_temporal_reduce(expr, "max")
    return _temporal_reduction_spatial("temporal.max", expr)


def temporal_std(expr: TemporalRasterExpression | TemporalRaster, *, ddof: float = 0) -> RasterExpression | Raster:
    if not (np.isfinite(ddof) and ddof >= 0):
        raise MapAlgebraError(
            "ddof must be a finite nonnegative number.",
            code="temporal_invalid_ddof",
            details={"ddof": ddof},
        )
    if isinstance(expr, TemporalRaster):
        return _eager_temporal_reduce(expr, "std", ddof=ddof)
    return _temporal_reduction_spatial("temporal.std", expr, ddof=ddof)


def temporal_sum(expr: TemporalRasterExpression | TemporalRaster) -> RasterExpression | Raster:
    if isinstance(expr, TemporalRaster):
        return _eager_temporal_reduce(expr, "sum")
    return _temporal_reduction_spatial("temporal.sum", expr)


def temporal_count(expr: TemporalRasterExpression | TemporalRaster) -> RasterExpression | Raster:
    if isinstance(expr, TemporalRaster):
        return _eager_temporal_reduce(expr, "count")
    return _temporal_reduction_spatial("temporal.count", expr)


# ---------------------------------------------------------------------------
# In-memory temporal reduction helpers
# ---------------------------------------------------------------------------

def _eager_temporal_reduce(tr: TemporalRaster, operation: str, **opts: Any) -> Raster:
    valid_3d = tr.valid
    values = tr.values

    if operation == "mean":
        if not np.any(valid_3d):
            result = np.full(tr.spatial_shape, np.nan, dtype=np.float64)
        else:
            masked = np.ma.array(values.astype(np.float64), mask=~valid_3d)
            result = np.asarray(masked.mean(axis=0))
    elif operation == "min":
        cond = valid_3d
        if np.issubdtype(values.dtype, np.bool_):
            neutral = True
        elif np.issubdtype(values.dtype, np.integer):
            neutral = np.iinfo(values.dtype).max
        else:
            neutral = np.inf
        result = np.min(np.where(cond, values, neutral), axis=0).astype(
            values.dtype, copy=False,
        )
    elif operation == "max":
        cond = valid_3d
        if np.issubdtype(values.dtype, np.bool_):
            neutral = False
        elif np.issubdtype(values.dtype, np.integer):
            neutral = np.iinfo(values.dtype).min
        else:
            neutral = -np.inf
        result = np.max(np.where(cond, values, neutral), axis=0).astype(
            values.dtype, copy=False,
        )
    elif operation == "std":
        ddof = opts.get("ddof", 0)
        if not np.any(valid_3d):
            result = np.full(tr.spatial_shape, np.nan, dtype=np.float64)
        else:
            masked = np.ma.array(values.astype(np.float64), mask=~valid_3d)
            result = np.asarray(masked.std(axis=0, ddof=ddof))
    elif operation == "sum":
        cond = valid_3d
        result = np.sum(
            np.where(cond, values, 0), axis=0, dtype=np.float64,
        )
    elif operation == "count":
        result = np.sum(valid_3d, axis=0, dtype=np.int64)
    else:
        raise MapAlgebraError(
            f"Unknown temporal reduction operation: {operation}",
            code="temporal_unknown_reduction",
            details={"operation": operation},
        )

    valid_count = np.sum(valid_3d, axis=0, dtype=np.int64)
    if operation == "count":
        result_valid = np.ones(tr.spatial_shape, dtype=np.bool_)
    elif operation == "std":
        result_valid = valid_count > float(opts.get("ddof", 0))
    else:
        result_valid = valid_count > 0
    output_georef = tr.georef.with_nodata(None)

    output_units = tr.units
    if operation == "count":
        output_units = None

    return Raster(
        values=result,
        georef=output_georef,
        valid=result_valid,
        units=output_units,
        name=tr.name,
    )


def _reduce_temporal_expression(
    expression: TemporalRasterExpression,
    operation: str,
    *,
    ddof: float = 0,
) -> Raster:
    """Reduce a temporal graph while retaining only one computed layer."""
    grid = expression.grid
    if grid is None:
        raise MapAlgebraExpressionError(
            "Cannot reduce a temporal expression with no grid.",
            code="temporal_no_grid",
        )

    shape = (grid.height, grid.width)
    count = np.zeros(shape, dtype=np.int64)
    aggregate: np.ndarray | None = None
    mean = np.zeros(shape, dtype=np.float64)
    m2 = np.zeros(shape, dtype=np.float64)

    for layer in _iter_temporal_layers(expression):
        valid = layer.valid
        values = layer.values
        if operation == "count":
            count[valid] += 1
            continue
        if operation == "sum":
            if aggregate is None:
                aggregate = np.zeros(shape, dtype=np.float64)
            aggregate[valid] += values[valid].astype(np.float64, copy=False)
        elif operation in {"mean", "std"}:
            previous = count[valid].astype(np.float64, copy=False)
            samples = values[valid].astype(np.float64, copy=False)
            delta = samples - mean[valid]
            new_count = previous + 1.0
            mean[valid] += delta / new_count
            if operation == "std":
                m2[valid] += delta * (samples - mean[valid])
        elif operation in {"min", "max"}:
            if aggregate is None:
                aggregate = np.zeros(shape, dtype=values.dtype)
            first = valid & (count == 0)
            subsequent = valid & (count > 0)
            aggregate[first] = values[first]
            reducer = np.minimum if operation == "min" else np.maximum
            aggregate[subsequent] = reducer(
                aggregate[subsequent], values[subsequent],
            )
        else:  # pragma: no cover - private dispatch controls operation
            raise AssertionError(operation)
        count[valid] += 1

    if operation == "count":
        output = count
        output_valid = np.ones(shape, dtype=np.bool_)
        units = None
    elif operation == "mean":
        output = mean
        output_valid = count > 0
        units = expression.units
    elif operation == "std":
        output = np.zeros(shape, dtype=np.float64)
        output_valid = count > ddof
        np.divide(m2, count - ddof, out=output, where=output_valid)
        np.sqrt(output, out=output)
        units = expression.units
    else:
        assert aggregate is not None
        output = aggregate
        output_valid = count > 0
        units = expression.units

    return Raster(
        values=output,
        georef=grid.with_nodata(None),
        valid=output_valid,
        units=units,
        name=expression._signal_name,
    )


# ---------------------------------------------------------------------------
# compute_temporal
# ---------------------------------------------------------------------------

def compute_temporal(expression: TemporalRasterExpression) -> TemporalRaster:
    if not isinstance(expression, TemporalRasterExpression):
        raise MapAlgebraExpressionError(
            "compute_temporal() requires a TemporalRasterExpression.",
            code="map_algebra_invalid_operand",
            details={"type": type(expression).__name__},
        )
    return _compute_temporal(expression)


# ---------------------------------------------------------------------------
# explain_temporal
# ---------------------------------------------------------------------------

def explain_temporal(expression: TemporalRasterExpression) -> str:
    from ._temporal_model import _topological_sort_temporal
    nodes = _topological_sort_temporal(expression)

    lines = ["Temporal expression:"]
    for node in nodes:
        op_desc = f"  {node._operation_id}"
        if node._inferred_times is not None:
            op_desc += f" (times={len(node._inferred_times)})"
        else:
            op_desc += " (reduced)"
        if node._inferred_dtype is not None:
            op_desc += f" dtype={node._inferred_dtype.name}"
        if node._inferred_units is not None:
            op_desc += f" units={node._inferred_units!r}"
        if node._signal_name is not None:
            op_desc += f" signal={node._signal_name!r}"
        lines.append(op_desc)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal: temporal reduction that returns a RasterExpression node
# ---------------------------------------------------------------------------

def _temporal_reduction_spatial(op_id: str, expr: TemporalRasterExpression, **params: Any) -> RasterExpression:
    return _make_expr_node(
        op_id, (expr,),
        grid=expr._inferred_grid,
        dtype=_reduction_output_dtype(op_id, expr._inferred_dtype),
        units=_reduction_output_units(op_id, expr._inferred_units),
        params=params,
    )
