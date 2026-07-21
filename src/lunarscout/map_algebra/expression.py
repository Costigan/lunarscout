from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..errors import MapAlgebraError, MapAlgebraExpressionError
from ..georeference import GeoReference
from ..raster import Raster
from ._kernels import (
    _absolute,
    _add,
    _arccos,
    _arcsin,
    _arctan,
    _cos,
    _divide,
    _equal,
    _exp,
    _floor,
    _greater,
    _greater_equal,
    _hypot,
    _less,
    _less_equal,
    _log,
    _log10,
    _logical_and,
    _logical_not,
    _logical_or,
    _logical_xor,
    _maximum,
    _minimum,
    _multiply,
    _negate,
    _not_equal,
    _sin,
    _sqrt,
    _square,
    _subtract,
    _tan,
)
from ._model import RasterExpression, _new_node_id
from ._sources import constant, source
from ._validation import _is_scalar, _normalize_scalar, _require_common_grid

# ---------------------------------------------------------------------------
# Operation builder helpers
# ---------------------------------------------------------------------------


def _op_expr(
    operation_id: str,
    operands: tuple[Any, ...],
    *,
    output_grid: GeoReference | None = None,
    output_dtype: np.dtype[Any] | None = None,
    output_units: str | None = None,
    halo: int = 0,
    params: dict[str, Any] | None = None,
) -> RasterExpression:
    return RasterExpression(
        _node_id=_new_node_id(),
        _operation_id=operation_id,
        _operands=operands,
        _params=params or {},
        _inferred_grid=output_grid,
        _inferred_dtype=output_dtype,
        _inferred_units=output_units,
        _halo=halo,
    )


def _to_expr_or_scalar(
    value: Any,
) -> RasterExpression | int | float:
    if isinstance(value, RasterExpression):
        return value
    if isinstance(value, Raster):
        return constant(value)
    if _is_scalar(value):
        return _normalize_scalar(value)
    raise MapAlgebraError(
        f"Unsupported expression operand type: {type(value).__name__}",
        code="map_algebra_invalid_operand",
    )


def _infer_grid(operands: list[Any]) -> GeoReference | None:
    for op in operands:
        if isinstance(op, RasterExpression) and op.grid is not None:
            return op.grid
    return None


def _infer_dtype(operands: list[Any]) -> np.dtype[Any] | None:
    dtypes = []
    for op in operands:
        if isinstance(op, RasterExpression) and op.dtype is not None:
            dtypes.append(op.dtype)
    if not dtypes:
        return None
    return np.result_type(*dtypes)


def _infer_units_from_first_raster(operands: list[Any]) -> str | None:
    for op in operands:
        if isinstance(op, RasterExpression) and op.units is not None:
            return op.units
    return None

# ---------------------------------------------------------------------------
# Compute (eager materialization)
# ---------------------------------------------------------------------------


def _load_source(expr: RasterExpression) -> Raster:
    from . import read as _ma_read

    params = expr._params
    path = Path(params["path"])
    band = int(params["band"])
    units = expr._inferred_units
    return _ma_read(path, band=band, units=units)


def _load_constant(expr: RasterExpression) -> Raster:
    return expr._operands[0]


_UNARY_KERNELS: dict[str, Callable] = {
    "local.negative": _negate,
    "local.absolute": _absolute,
    "local.sqrt": _sqrt,
    "local.square": _square,
    "local.exp": _exp,
    "local.log": _log,
    "local.log10": _log10,
    "local.sin": _sin,
    "local.cos": _cos,
    "local.tan": _tan,
    "local.arcsin": _arcsin,
    "local.arccos": _arccos,
    "local.arctan": _arctan,
    "local.logical_not": _logical_not,
    "local.floor": _floor,
}

_BINARY_KERNELS: dict[str, Callable] = {
    "local.add": _add,
    "local.subtract": _subtract,
    "local.multiply": _multiply,
    "local.divide": _divide,
    "local.minimum": _minimum,
    "local.maximum": _maximum,
    "local.less": _less,
    "local.less_equal": _less_equal,
    "local.greater": _greater,
    "local.greater_equal": _greater_equal,
    "local.equal": _equal,
    "local.not_equal": _not_equal,
    "local.logical_and": _logical_and,
    "local.logical_or": _logical_or,
    "local.logical_xor": _logical_xor,
    "local.hypot": _hypot,
}


def compute(expression: RasterExpression) -> Raster:
    nodes = expression._walk_nodes()
    nodes.reverse()
    cache: dict[str, Raster] = {}

    for node in nodes:
        if node._operation_id == "source":
            cache[node._node_id] = _load_source(node)
        elif node._operation_id == "constant":
            cache[node._node_id] = _load_constant(node)
        else:
            operands = []
            for op in node._operands:
                if isinstance(op, RasterExpression):
                    operands.append(cache[op._node_id])
                else:
                    operands.append(op)

            raster_ops = [o for o in operands if isinstance(o, Raster)]
            if not raster_ops:
                raise MapAlgebraExpressionError(
                    f"No Raster operands for operation '{node._operation_id}'.",
                    code="map_algebra_expression_eval_failed",
                )

            kernel = _UNARY_KERNELS.get(node._operation_id) or _BINARY_KERNELS.get(node._operation_id)
            if kernel is None:
                raise MapAlgebraExpressionError(
                    f"Unknown operation in compute: {node._operation_id}",
                    code="map_algebra_expression_eval_failed",
                )

            from ._eager import (
                _dispatch_binary_raster_raster,
                _dispatch_binary_raster_scalar,
                _dispatch_unary,
            )

            if node._operation_id in _UNARY_KERNELS:
                result = _dispatch_unary(raster_ops[0], kernel, operation=node._operation_id)
            elif len(operands) == 2:
                a, b = operands[0], operands[1]
                if isinstance(a, Raster) and isinstance(b, Raster):
                    result = _dispatch_binary_raster_raster(a, b, kernel, operation=node._operation_id)
                elif isinstance(a, Raster):
                    result = _dispatch_binary_raster_scalar(a, b, kernel, operation=node._operation_id)
                else:
                    result = _dispatch_binary_raster_scalar(b, a, kernel, operation=node._operation_id)
            else:
                raise MapAlgebraExpressionError(
                    f"Unsupported arity for compute: {node._operation_id}",
                    code="map_algebra_expression_eval_failed",
                )
            cache[node._node_id] = result

    return cache[expression._node_id]


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


_OP_DESCRIPTIONS: dict[str, str] = {
    "source": "read",
    "constant": "in-memory constant",
    "local.add": "add",
    "local.subtract": "subtract",
    "local.multiply": "multiply",
    "local.divide": "divide",
    "local.minimum": "minimum",
    "local.maximum": "maximum",
    "local.less": "less than",
    "local.less_equal": "less than or equal to",
    "local.greater": "greater than",
    "local.greater_equal": "greater than or equal to",
    "local.equal": "equal to",
    "local.not_equal": "not equal to",
    "local.logical_and": "and",
    "local.logical_or": "or",
    "local.logical_xor": "xor",
    "local.logical_not": "not",
    "local.negative": "negate",
    "local.absolute": "absolute value",
    "local.sqrt": "square root",
    "local.square": "square",
    "local.exp": "exponential",
    "local.log": "natural logarithm",
    "local.log10": "base-10 logarithm",
    "local.sin": "sine",
    "local.cos": "cosine",
    "local.tan": "tangent",
    "local.arcsin": "arcsine",
    "local.arccos": "arccosine",
    "local.arctan": "arctangent",
}


def explain(expression: RasterExpression) -> str:
    nodes = expression._walk_nodes()
    lines: list[str] = []
    lines.append("RasterExpression with %d node(s):" % len(nodes))
    for node in nodes:
        desc = _OP_DESCRIPTIONS.get(node._operation_id, node._operation_id)
        line = f"  [{node._node_id}] {desc}"
        if node._operation_id == "source":
            params = node._params
            line += f" from {params.get('path', '?')}"
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
    nodes = expression._walk_nodes()
    sources = [n for n in nodes if n._operation_id == "source"]
    constants = [n for n in nodes if n._operation_id == "constant"]
    ops = [n for n in nodes if n._operation_id not in ("source", "constant")]

    source_descs = []
    for s in sources:
        source_descs.append({
            "path": s._params.get("path", ""),
            "band": s._params.get("band", 1),
            "grid": f"{s._inferred_grid.width}x{s._inferred_grid.height}" if s._inferred_grid else None,
            "dtype": str(s._inferred_dtype) if s._inferred_dtype else None,
        })

    return {
        "node_count": len(nodes),
        "source_count": len(sources),
        "constant_count": len(constants),
        "operation_count": len(ops),
        "output_grid": str(expression._inferred_grid.width) + "x" + str(expression._inferred_grid.height) if expression._inferred_grid else None,
        "output_dtype": str(expression._inferred_dtype) if expression._inferred_dtype else None,
        "output_units": expression._inferred_units,
        "sources": source_descs,
        "output_path": output,
    }


# ---------------------------------------------------------------------------
# RasterExpression operator overloads
# ---------------------------------------------------------------------------


def _binary_op(
    self_expr: RasterExpression,
    other: Any,
    op_id: str,
    swap: bool = False,
) -> RasterExpression:
    if isinstance(self_expr, Raster):
        self_expr = constant(self_expr)
    other_val = _to_expr_or_scalar(other)

    if swap:
        operands = (other_val, self_expr)
    else:
        operands = (self_expr, other_val)

    raster_operands = [op for op in operands if isinstance(op, RasterExpression)]
    grid = _infer_grid(raster_operands)
    dtype = _infer_dtype(raster_operands)
    units = _infer_units_from_first_raster(raster_operands)

    return _op_expr(op_id, operands, output_grid=grid, output_dtype=dtype, output_units=units)


def _unary_op(self_expr: RasterExpression, op_id: str) -> RasterExpression:
    if isinstance(self_expr, Raster):
        self_expr = constant(self_expr)
    return _op_expr(
        op_id, (self_expr,),
        output_grid=self_expr.grid, output_dtype=self_expr.dtype,
        output_units=self_expr.units,
    )
