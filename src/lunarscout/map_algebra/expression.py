from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..errors import MapAlgebraExpressionError
from ..raster import Raster
from ._model import RasterExpression, _next_id, _make_expr_node
from ._sources import constant, source
from ._validation import _is_scalar, _normalize_scalar

# ---------------------------------------------------------------------------
# Compute — delegates to Phase B eager dispatch
# ---------------------------------------------------------------------------


def _load_source_raster(expr: RasterExpression) -> Raster:
    from . import read as _ma_read
    params = expr._params_dict
    path = Path(params["path"])
    band = int(params["band"])
    return _ma_read(path, band=band, units=expr._inferred_units)


def _load_constant_raster(expr: RasterExpression) -> Raster:
    return expr._operands[0]


_BINARY_OP_IDS = {
    "local.add", "local.subtract", "local.multiply", "local.divide",
    "local.floor_divide", "local.remainder", "local.power",
    "local.minimum", "local.maximum",
    "local.less", "local.less_equal", "local.greater", "local.greater_equal",
    "local.equal", "local.not_equal",
    "local.logical_and", "local.logical_or", "local.logical_xor",
    "local.hypot", "local.arctan2",
}

_UNARY_OP_IDS = {
    "local.negative", "local.absolute", "local.sqrt", "local.square",
    "local.exp", "local.log", "local.log10",
    "local.sin", "local.cos", "local.tan",
    "local.arcsin", "local.arccos", "local.arctan",
    "local.logical_not", "local.floor", "local.ceil", "local.trunc",
    "local.round", "local.degrees", "local.radians",
}

_SPECIAL_OPS = {
    "local.where", "local.coalesce", "local.clip", "local.cast",
    "local.set_invalid", "local.fill_invalid",
    "local.is_valid", "local.is_invalid",
    "local.reclassify_values", "local.reclassify_ranges", "local.digitize",
    "local.one_hot", "local.normalize_minmax", "local.standardize",
    "local.isclose",
}

_COORDINATE_OPS = {
    "coordinate.row_indices", "coordinate.column_indices",
    "coordinate.projected_x", "coordinate.projected_y",
    "coordinate.longitude", "coordinate.latitude",
}

_TEMPORAL_REDUCTION_OP_IDS = {
    "temporal.mean", "temporal.min", "temporal.max",
    "temporal.std", "temporal.sum", "temporal.count",
}


def compute(expression: RasterExpression) -> Raster:
    nodes = expression._all_nodes()
    cache: dict[str, Raster] = {}

    for node in nodes:
        op_id = node._operation_id

        if op_id == "source":
            cache[node._node_id] = _load_source_raster(node)
            continue
        if op_id == "constant":
            cache[node._node_id] = _load_constant_raster(node)
            continue
        if op_id in _COORDINATE_OPS:
            from .coordinates import _compute_coordinate

            cache[node._node_id] = _compute_coordinate(node)
            continue

        operands: list[Any] = []
        for op in node._operands:
            if isinstance(op, RasterExpression):
                operands.append(cache[op._node_id])
            else:
                operands.append(op)

        if op_id in _BINARY_OP_IDS:
            result = _eval_binary(node, operands)
        elif op_id in _UNARY_OP_IDS:
            result = _eval_unary(node, operands)
        elif op_id in _SPECIAL_OPS:
            result = _eval_special(node, operands)
        elif op_id in _TEMPORAL_REDUCTION_OP_IDS:
            result = _eval_temporal_reduction(node, operands)
        elif op_id in {"terrain.slope", "terrain.aspect", "terrain.hillshade"}:
            from ._spatial import evaluate_terrain

            result = evaluate_terrain(node, operands[0])
        elif op_id == "alignment.resample_to":
            from ._spatial import evaluate_resample

            if node.grid is None:
                raise MapAlgebraExpressionError(
                    "Resampling expression has no destination grid.",
                    code="map_algebra_missing_output_grid",
                )
            result = evaluate_resample(node, operands[0], node.grid)
        else:
            raise MapAlgebraExpressionError(
                f"Unknown operation in compute: {op_id}",
                code="map_algebra_expression_eval_failed",
            )
        cache[node._node_id] = result

    return cache[expression._node_id]


def _eval_binary(node: RasterExpression, operands: list[Any]) -> Raster:
    from lunarscout.map_algebra import local as _ma

    a, b = operands[0], operands[1]
    numeric_errors = node._params_dict.get("numeric_errors", "invalid")
    overflow = node._params_dict.get("overflow", "raise")
    output_units = node._params_dict.get("output_units")
    if node._operation_id == "local.add":
        return _ma.add(
            a, b, overflow=overflow, numeric_errors=numeric_errors,
        )
    elif node._operation_id == "local.subtract":
        return _ma.subtract(
            a, b, overflow=overflow, numeric_errors=numeric_errors,
        )
    elif node._operation_id == "local.multiply":
        return _ma.multiply(
            a, b, output_units=output_units, overflow=overflow,
            numeric_errors=numeric_errors,
        )
    elif node._operation_id == "local.divide":
        return _ma.divide(a, b, output_units=output_units, numeric_errors=numeric_errors)
    elif node._operation_id == "local.minimum":
        return _ma.minimum(a, b, numeric_errors=numeric_errors)
    elif node._operation_id == "local.maximum":
        return _ma.maximum(a, b, numeric_errors=numeric_errors)
    elif node._operation_id == "local.less":
        return _ma.less(a, b)
    elif node._operation_id == "local.less_equal":
        return _ma.less_equal(a, b)
    elif node._operation_id == "local.greater":
        return _ma.greater(a, b)
    elif node._operation_id == "local.greater_equal":
        return _ma.greater_equal(a, b)
    elif node._operation_id == "local.equal":
        return _ma.equal(a, b)
    elif node._operation_id == "local.not_equal":
        return _ma.not_equal(a, b)
    elif node._operation_id == "local.logical_and":
        return _ma.logical_and(a, b)
    elif node._operation_id == "local.logical_or":
        return _ma.logical_or(a, b)
    elif node._operation_id == "local.logical_xor":
        return _ma.logical_xor(a, b)
    elif node._operation_id == "local.floor_divide":
        return _ma.floor_divide(a, b, overflow=overflow, numeric_errors=numeric_errors)
    elif node._operation_id == "local.remainder":
        return _ma.remainder(a, b, overflow=overflow, numeric_errors=numeric_errors)
    elif node._operation_id == "local.power":
        return _ma.power(
            a, b, output_units=output_units, overflow=overflow,
            numeric_errors=numeric_errors,
        )
    elif node._operation_id == "local.hypot":
        return _ma.hypot(a, b, numeric_errors=numeric_errors)
    elif node._operation_id == "local.arctan2":
        return _ma.arctan2(a, b, numeric_errors=numeric_errors)
    raise MapAlgebraExpressionError(
        f"Unsupported binary op: {node._operation_id}",
        code="map_algebra_expression_eval_failed",
    )


def _eval_unary(node: RasterExpression, operands: list[Any]) -> Raster:
    from lunarscout.map_algebra import local as _ma

    a = operands[0]
    op_id = node._operation_id
    numeric_errors = node._params_dict.get("numeric_errors", "invalid")
    overflow = node._params_dict.get("overflow", "raise")
    if op_id == "local.negative":
        return _ma.negative(
            a, overflow=overflow, numeric_errors=numeric_errors,
        )
    elif op_id == "local.absolute":
        return _ma.absolute(
            a, overflow=overflow, numeric_errors=numeric_errors,
        )
    elif op_id == "local.sqrt":
        return _ma.sqrt(a, numeric_errors=numeric_errors)
    elif op_id == "local.square":
        return _ma.square(a, overflow=overflow, numeric_errors=numeric_errors)
    elif op_id == "local.exp":
        return _ma.exp(a, numeric_errors=numeric_errors)
    elif op_id == "local.log":
        return _ma.log(a, numeric_errors=numeric_errors)
    elif op_id == "local.log10":
        return _ma.log10(a, numeric_errors=numeric_errors)
    elif op_id == "local.sin":
        return _ma.sin(a, numeric_errors=numeric_errors)
    elif op_id == "local.cos":
        return _ma.cos(a, numeric_errors=numeric_errors)
    elif op_id == "local.tan":
        return _ma.tan(a, numeric_errors=numeric_errors)
    elif op_id == "local.arcsin":
        return _ma.arcsin(a, numeric_errors=numeric_errors)
    elif op_id == "local.arccos":
        return _ma.arccos(a, numeric_errors=numeric_errors)
    elif op_id == "local.arctan":
        return _ma.arctan(a, numeric_errors=numeric_errors)
    elif op_id == "local.logical_not":
        return _ma.logical_not(a)
    elif op_id == "local.floor":
        return _ma.floor(a, numeric_errors=numeric_errors)
    elif op_id == "local.ceil":
        return _ma.ceil(a, numeric_errors=numeric_errors)
    elif op_id == "local.trunc":
        return _ma.trunc(a, numeric_errors=numeric_errors)
    elif op_id == "local.round":
        return _ma.round_half_even(
            a,
            ndigits=node._params_dict.get("ndigits", 0),
            numeric_errors=numeric_errors,
        )
    elif op_id == "local.degrees":
        return _ma.degrees(a, numeric_errors=numeric_errors)
    elif op_id == "local.radians":
        return _ma.radians(a, numeric_errors=numeric_errors)
    raise MapAlgebraExpressionError(
        f"Unsupported unary op: {op_id}",
        code="map_algebra_expression_eval_failed",
    )


def _eval_special(node: RasterExpression, operands: list[Any]) -> Raster:
    from lunarscout.map_algebra import local as _ma

    op_id = node._operation_id
    if op_id == "local.where":
        branches = [
            _ma.invalid if isinstance(value, str) and value == "invalid" else value
            for value in operands[1:]
        ]
        return _ma.where(operands[0], branches[0], branches[1])
    elif op_id == "local.isclose":
        return _ma.isclose(
            operands[0], operands[1],
            rtol=node._params_dict["rtol"],
            atol=node._params_dict["atol"],
            equal_nan=node._params_dict["equal_nan"],
        )
    elif op_id == "local.coalesce":
        return _ma.coalesce(*operands)
    elif op_id == "local.clip":
        return _ma.clip(
            operands[0],
            lower=node._params_dict.get("lower"),
            upper=node._params_dict.get("upper"),
        )
    elif op_id == "local.cast":
        return _ma.cast(
            operands[0], operands[1],
            casting=node._params_dict.get("casting", "safe"),
            overflow=node._params_dict.get("overflow", "raise"),
        )
    elif op_id == "local.set_invalid":
        return _ma.set_invalid(operands[0], operands[1])
    elif op_id == "local.fill_invalid":
        return _ma.fill_invalid(operands[0], operands[1])
    elif op_id == "local.is_valid":
        return _ma.is_valid(operands[0])
    elif op_id == "local.is_invalid":
        return _ma.is_invalid(operands[0])
    elif op_id == "local.reclassify_values":
        return _ma.reclassify_values(
            operands[0], dict(node._params_dict["mapping"]),
            default=node._params_dict["default"],
        )
    elif op_id == "local.reclassify_ranges":
        return _ma.reclassify_ranges(
            operands[0], node._params_dict["ranges"],
            default=node._params_dict["default"],
        )
    elif op_id == "local.digitize":
        return _ma.digitize(
            operands[0], node._params_dict["bins"], right=node._params_dict["right"]
        )
    elif op_id == "local.one_hot":
        return _ma.one_hot(operands[0], (node._params_dict["class_value"],))[0]
    elif op_id == "local.normalize_minmax":
        return _ma.normalize_minmax(
            operands[0], minimum=node._params_dict["minimum"],
            maximum=node._params_dict["maximum"],
        )
    elif op_id == "local.standardize":
        return _ma.standardize(
            operands[0], mean=node._params_dict["mean"], std=node._params_dict["std"],
            ddof=node._params_dict["ddof"],
        )
    raise MapAlgebraExpressionError(
        f"Unsupported special op: {op_id}",
        code="map_algebra_expression_eval_failed",
    )


def _eval_temporal_reduction(node: RasterExpression, operands: list[Any]) -> Raster:
    op_id = node._operation_id
    temporal_expr = node._operands[0]

    from ._temporal_model import TemporalRasterExpression
    from . import temporal as _temporal_ma

    if not isinstance(temporal_expr, TemporalRasterExpression):
        raise MapAlgebraExpressionError(
            "Temporal reduction operand must be a TemporalRasterExpression.",
            code="map_algebra_expression_eval_failed",
            details={"type": type(temporal_expr).__name__},
        )

    if op_id == "temporal.mean":
        return _temporal_ma._reduce_temporal_expression(temporal_expr, "mean")
    elif op_id == "temporal.min":
        return _temporal_ma._reduce_temporal_expression(temporal_expr, "min")
    elif op_id == "temporal.max":
        return _temporal_ma._reduce_temporal_expression(temporal_expr, "max")
    elif op_id == "temporal.std":
        ddof = node._params_dict.get("ddof", 0)
        return _temporal_ma._reduce_temporal_expression(
            temporal_expr, "std", ddof=ddof,
        )
    elif op_id == "temporal.sum":
        return _temporal_ma._reduce_temporal_expression(temporal_expr, "sum")
    elif op_id == "temporal.count":
        return _temporal_ma._reduce_temporal_expression(temporal_expr, "count")
    raise MapAlgebraExpressionError(
        f"Unsupported temporal reduction: {op_id}",
        code="map_algebra_expression_eval_failed",
    )


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

_OP_DESCRIPTIONS: dict[str, str] = {
    "source": "read",
    "constant": "in-memory constant",
    "local.add": "add", "local.subtract": "subtract",
    "local.multiply": "multiply", "local.divide": "divide",
    "local.floor_divide": "floor divide", "local.remainder": "remainder",
    "local.power": "power",
    "local.minimum": "minimum", "local.maximum": "maximum",
    "local.less": "less than", "local.less_equal": "less than or equal to",
    "local.greater": "greater than", "local.greater_equal": "greater than or equal to",
    "local.equal": "equal to", "local.not_equal": "not equal to",
    "local.logical_and": "and", "local.logical_or": "or",
    "local.logical_xor": "xor", "local.logical_not": "not",
    "local.negative": "negate", "local.absolute": "absolute value",
    "local.sqrt": "square root", "local.square": "square",
    "local.exp": "exponential", "local.log": "natural logarithm",
    "local.log10": "base-10 logarithm",
    "local.sin": "sine", "local.cos": "cosine", "local.tan": "tangent",
    "local.arcsin": "arcsine", "local.arccos": "arccosine", "local.arctan": "arctangent",
    "local.arctan2": "arctan2", "local.hypot": "hypotenuse",
    "local.floor": "floor", "local.ceil": "ceil", "local.trunc": "truncate",
    "local.round": "round",
    "local.where": "where", "local.coalesce": "coalesce",
    "local.is_valid": "is valid", "local.is_invalid": "is invalid",
    "local.clip": "clip", "local.cast": "cast",
}


def explain(expression: RasterExpression) -> str:
    nodes = expression._all_nodes()
    lines: list[str] = [f"RasterExpression with {len(nodes)} node(s):"]
    for node in nodes:
        desc = _OP_DESCRIPTIONS.get(node._operation_id, node._operation_id)
        line = f"  [{node._node_id}] {desc}"
        if node._operation_id == "source":
            line += f" from {node._params_dict.get('path', '?')}"
        elif node._operation_id == "constant":
            line += f" ({node._inferred_dtype})"
        else:
            op_strs = []
            for op in node._operands:
                if isinstance(op, RasterExpression):
                    op_strs.append(f"[{op._node_id}]")
                else:
                    op_strs.append(repr(op))
            line += f"({', '.join(op_strs)})"
        if node._inferred_dtype is not None:
            line += f" -> {node._inferred_dtype.name}"
        if node._inferred_grid is not None:
            line += f" @ {node._inferred_grid.width}x{node._inferred_grid.height}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan (dry-run validation)
# ---------------------------------------------------------------------------


def plan(expression: RasterExpression, *, output: str | None = None) -> dict[str, Any]:
    nodes = expression._all_nodes()
    sources = [n for n in nodes if n._operation_id == "source"]
    constants = [n for n in nodes if n._operation_id == "constant"]
    ops = [n for n in nodes if n._operation_id not in ("source", "constant")]

    source_descs = []
    for s in sources:
        source_descs.append({
            "path": s._params_dict.get("path", ""),
            "band": s._params_dict.get("band", 1),
            "grid": f"{s._inferred_grid.width}x{s._inferred_grid.height}" if s._inferred_grid else None,
            "dtype": str(s._inferred_dtype) if s._inferred_dtype else None,
        })

    grid_str = None
    if expression._inferred_grid is not None:
        grid_str = f"{expression._inferred_grid.width}x{expression._inferred_grid.height}"

    from ._planner import plan_expression as _plan_expression

    ep = _plan_expression(expression)
    planned_dtype = expression._inferred_dtype
    if planned_dtype == np.dtype(np.bool_):
        planned_dtype = np.dtype(np.uint8)
    default_fill: int | float = (
        float(np.nan)
        if planned_dtype is not None and np.issubdtype(planned_dtype, np.floating)
        else 0
    )
    planner_info: dict[str, Any] = {
        "window_width": ep.window_width,
        "window_height": ep.window_height,
        "n_windows_x": ep.n_windows_x,
        "n_windows_y": ep.n_windows_y,
        "total_windows": ep.total_windows,
        "passes": ep.n_passes,
        "n_sources": ep.n_sources,
        "n_operations": ep.n_operations,
        "halos": ep.halos,
        "estimated_peak_bytes": ep.estimated_per_window_bytes,
        "journal_available": ep.journal_available,
        "supports_progress": ep.supports_progress,
        "supports_cancellation": ep.supports_cancellation,
        "resumable_stages": ep.resumable_stages,
    }
    if planned_dtype is not None:
        planner_info["default_write_journal_identity"] = ep.journal_identity(
            expression, planned_dtype, default_fill,
        )
        planner_info["default_write_journal_identity_inputs"] = (
            ep.journal_identity_inputs(expression, planned_dtype, default_fill)
        )

    return {
        "node_count": len(nodes),
        "source_count": len(sources),
        "constant_count": len(constants),
        "operation_count": len(ops),
        "output_grid": grid_str,
        "output_dtype": str(expression._inferred_dtype) if expression._inferred_dtype else None,
        "output_units": expression._inferred_units,
        "sources": source_descs,
        "output_path": output,
        "planner": planner_info,
    }
