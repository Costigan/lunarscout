from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import MapAlgebraExpressionError
from ..georeference import GeoReference
from ..raster import Raster


_SEALED_TOKEN = object()


def _new_node_id(graph_label: str) -> str:
    return graph_label


@dataclass(frozen=True, slots=True, eq=False)
class RasterExpression:
    """Immutable description of a calculation that has not yet run.

    Direct construction is prohibited; obtain instances from ``ma.source()``,
    ``Raster.expression()``, or map-algebra operators.
    """

    _node_id: str
    _operation_id: str
    _operands: tuple[Any, ...]
    _params: tuple[tuple[str, Any], ...]
    _inferred_grid: GeoReference | None
    _inferred_dtype: np.dtype[Any] | None
    _inferred_units: str | None
    _halo: int

    _sealed: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._sealed is not _SEALED_TOKEN:
            raise MapAlgebraExpressionError(
                "RasterExpression cannot be constructed directly. "
                "Use ma.source(), Raster.expression(), or map-algebra operators.",
                code="map_algebra_sealed_constructor",
            )

    __hash__ = None

    def __bool__(self) -> None:
        raise TypeError(
            "RasterExpression does not support implicit truth testing. "
            "Use ma.compute() to materialize."
        )

    @property
    def grid(self) -> GeoReference | None:
        return self._inferred_grid

    @property
    def dtype(self) -> np.dtype[Any] | None:
        return self._inferred_dtype

    @property
    def units(self) -> str | None:
        return self._inferred_units

    @property
    def halo(self) -> int:
        return self._halo

    @property
    def operation_id(self) -> str:
        return self._operation_id

    # -----------------------------------------------------------------
    # Topological sort
    # -----------------------------------------------------------------

    @staticmethod
    def _topological_sort(root: RasterExpression) -> list[RasterExpression]:
        edges: dict[str, list[str]] = {}
        indegree: dict[str, int] = {}
        node_map: dict[str, RasterExpression] = {}

        queue: deque[RasterExpression] = deque([root])
        while queue:
            node = queue.popleft()
            nid = node._node_id
            if nid in node_map:
                continue
            node_map[nid] = node
            edges.setdefault(nid, [])
            indegree.setdefault(nid, 0)
            for op in node._operands:
                if isinstance(op, RasterExpression):
                    child_nid = op._node_id
                    edges.setdefault(child_nid, [])
                    indegree.setdefault(child_nid, 0)
                    edges[child_nid].append(nid)
                    indegree[nid] = indegree.get(nid, 0) + 1
                    queue.append(op)

        result: list[RasterExpression] = []
        ready = deque(nid for nid, deg in indegree.items() if deg == 0)
        while ready:
            nid = ready.popleft()
            result.append(node_map[nid])
            for parent_nid in edges.get(nid, []):
                indegree[parent_nid] -= 1
                if indegree[parent_nid] == 0:
                    ready.append(parent_nid)

        if len(result) != len(node_map):
            raise MapAlgebraExpressionError(
                "Expression graph contains a cycle.",
                code="map_algebra_expression_cycle",
            )
        return result

    def _all_nodes(self) -> list[RasterExpression]:
        return self._topological_sort(self)

    # -----------------------------------------------------------------
    # Identity and serialization
    # -----------------------------------------------------------------

    def _source_descriptor(self) -> dict[str, Any] | None:
        if self._operation_id == "source":
            return dict(self._params)
        if self._operation_id == "constant":
            r = self._operands[0]
            return {
                "source": "constant",
                "shape": list(r.shape),
                "dtype": str(r.dtype),
            }
        return None

    def _content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self._operation_id.encode())
        for op in self._operands:
            if isinstance(op, RasterExpression):
                h.update(op._content_hash().encode())
            elif isinstance(op, Raster):
                h.update(op.values.tobytes())
                h.update(op.valid.tobytes())
                parts = [
                    str(op.shape), str(op.dtype), str(op.units),
                    str(op.georef.width), str(op.georef.height),
                    str(op.georef.projection_wkt),
                    ",".join(str(v) for v in op.georef.affine_transform),
                ]
                h.update("|".join(parts).encode())
            else:
                h.update(repr(op).encode())
        for k in sorted(dict(self._params).keys()):
            h.update(f"{k}={self._params_dict[k]!r}".encode())
        return h.hexdigest()

    @property
    def _params_dict(self) -> dict[str, Any]:
        return dict(self._params)

    def scientific_identity(self) -> str:
        return "sha256:" + self._content_hash()

    def to_json(self) -> str:
        nodes = self._all_nodes()
        node_list: list[dict[str, Any]] = []
        sources_list: list[dict[str, Any]] = []

        for node in nodes:
            entry: dict[str, Any] = {
                "node_id": node._node_id,
                "operation_id": node._operation_id,
                "params": {k: _json_safe(v) for k, v in dict(node._params).items()},
                "operands": [
                    op._node_id if isinstance(op, RasterExpression) else _json_safe(op)
                    for op in node._operands
                ],
                "grid": _grid_json(node._inferred_grid),
                "dtype": str(node._inferred_dtype) if node._inferred_dtype is not None else None,
                "units": node._inferred_units,
                "halo": node._halo,
            }
            node_list.append(entry)
            desc = node._source_descriptor()
            if desc is not None:
                sources_list.append({"node_id": node._node_id, **desc})

        return json.dumps(
            {"schema_version": 2, "root_node_id": self._node_id,
             "nodes": node_list, "sources": sources_list},
            sort_keys=True, indent=2,
        )

    # -----------------------------------------------------------------
    # Operator overloads — all return new expression nodes
    # -----------------------------------------------------------------

    def __add__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.add")

    def __radd__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.add", swap=True)

    def __sub__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.subtract")

    def __rsub__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.subtract", swap=True)

    def __mul__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.multiply")

    def __rmul__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.multiply")

    def __truediv__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.divide")

    def __rtruediv__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.divide", swap=True)

    def __floordiv__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.floor_divide")

    def __mod__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.remainder")

    def __lt__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.less")

    def __le__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.less_equal")

    def __gt__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.greater")

    def __ge__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.greater_equal")

    def __eq__(self, other: object) -> RasterExpression:  # type: ignore[override]
        return _new_expr_op(self, other, "local.equal")

    def __ne__(self, other: object) -> RasterExpression:  # type: ignore[override]
        return _new_expr_op(self, other, "local.not_equal")

    def __and__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.logical_and")

    def __or__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.logical_or")

    def __xor__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.logical_xor")

    def __neg__(self) -> RasterExpression:
        return _new_expr_unary(self, "local.negative")

    def __abs__(self) -> RasterExpression:
        return _new_expr_unary(self, "local.absolute")

    def __invert__(self) -> RasterExpression:
        return _new_expr_unary(self, "local.logical_not")


# ---------------------------------------------------------------------------
# Internal factory helpers
# ---------------------------------------------------------------------------

_counter: int = 0


def _next_id() -> str:
    global _counter
    _counter += 1
    return f"n{_counter}"


def _make_expr_node(
    operation_id: str,
    operands: tuple[Any, ...],
    *,
    grid: GeoReference | None = None,
    dtype: np.dtype[Any] | None = None,
    units: str | None = None,
    halo: int = 0,
    params: dict[str, Any] | None = None,
) -> RasterExpression:
    return RasterExpression(
        _node_id=_next_id(),
        _operation_id=operation_id,
        _operands=operands,
        _params=tuple(sorted((params or {}).items())),
        _inferred_grid=grid,
        _inferred_dtype=dtype,
        _inferred_units=units,
        _halo=halo,
        _sealed=_SEALED_TOKEN,
    )


def _to_expr_or_scalar(value: Any) -> RasterExpression | int | float:
    if isinstance(value, RasterExpression):
        return value
    if isinstance(value, Raster):
        from ._sources import constant
        return constant(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if isinstance(value, (float, np.floating)) else int(value)
    raise MapAlgebraExpressionError(
        f"Unsupported expression operand type: {type(value).__name__}",
        code="map_algebra_invalid_operand",
    )


def _infer_common_grid(operands: list[Any]) -> GeoReference | None:
    grids = []
    for op in operands:
        if isinstance(op, RasterExpression) and op.grid is not None:
            grids.append(op.grid)
    if len(grids) < 2:
        return grids[0] if grids else None
    from ..alignment import require_same_grid
    require_same_grid(grids[0], grids[1])
    for g in grids[2:]:
        require_same_grid(grids[0], g)
    return grids[0]


def _infer_comparison_dtype() -> np.dtype[Any]:
    return np.dtype(np.bool_)


def _infer_expr_units(
    operands: list[Any],
    *,
    operation_id: str,
) -> str | None:
    units = []
    for op in operands:
        if isinstance(op, RasterExpression) and op.units is not None:
            units.append(op.units)
    if operation_id in (
        "local.less", "local.less_equal", "local.greater", "local.greater_equal",
        "local.equal", "local.not_equal", "local.logical_and", "local.logical_or",
        "local.logical_xor", "local.logical_not",
    ):
        return None
    if operation_id in ("local.add", "local.subtract", "local.minimum", "local.maximum"):
        if len(units) == 0:
            return None
        if len(units) == 2 and units[0] != units[1]:
            raise MapAlgebraExpressionError(
                f"Unit mismatch in '{operation_id}': {units[0]} vs {units[1]}",
                code="map_algebra_unit_mismatch",
            )
        return units[0] if units else None
    if operation_id in ("local.multiply", "local.divide"):
        return None
    return units[0] if units else None


def _new_expr_op(
    self_expr: RasterExpression | Raster,
    other: Any,
    op_id: str,
    swap: bool = False,
) -> RasterExpression:
    left = _to_expr_or_scalar(self_expr)
    right = _to_expr_or_scalar(other)
    if swap:
        left, right = right, left

    raster_ops = [op for op in (left, right) if isinstance(op, RasterExpression)]
    op_list = [left, right]

    grid = _infer_common_grid(raster_ops)

    is_comparison = op_id in (
        "local.less", "local.less_equal", "local.greater", "local.greater_equal",
        "local.equal", "local.not_equal",
    )
    is_boolean = op_id in ("local.logical_and", "local.logical_or", "local.logical_xor")

    if is_comparison or is_boolean:
        dtype = _infer_comparison_dtype()
    else:
        dtypes = [op.dtype for op in raster_ops if op.dtype is not None]
        dtype = np.result_type(*dtypes) if dtypes else None

    units = _infer_expr_units(raster_ops, operation_id=op_id)
    return _make_expr_node(op_id, tuple(op_list), grid=grid, dtype=dtype, units=units)


def _new_expr_unary(
    self_expr: RasterExpression | Raster,
    op_id: str,
) -> RasterExpression:
    expr = _to_expr_or_scalar(self_expr)
    if isinstance(expr, RasterExpression):
        is_logic = op_id == "local.logical_not"
        return _make_expr_node(
            op_id, (expr,),
            grid=expr.grid,
            dtype=_infer_comparison_dtype() if is_logic else expr.dtype,
            units=None,
        )
    raise MapAlgebraExpressionError(
        f"Unary operation requires a Raster or RasterExpression operand.",
        code="map_algebra_invalid_operand",
    )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_safe(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, Path):
        return str(v)
    return repr(v)


def _grid_json(georef: GeoReference | None) -> dict[str, Any] | None:
    if georef is None:
        return None
    return {
        "width": georef.width,
        "height": georef.height,
        "affine": [float(x) for x in georef.affine_transform],
        "pixel_size_x": georef.pixel_size_x,
        "pixel_size_y": georef.pixel_size_y,
        "crs_wkt": georef.projection_wkt,
        "nodata": repr(georef.nodata),
    }
