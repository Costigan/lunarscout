from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..georeference import GeoReference
from ..raster import Raster


def _new_node_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True, slots=True, eq=False)
class RasterExpression:
    """Immutable description of a calculation that has not yet run.

    ``eq=False`` is deliberate; expression identity uses node-id comparison
    and the named helpers ``scientific_identity()`` / ``to_json()``.
    """

    _node_id: str
    _operation_id: str
    _operands: tuple[Any, ...]
    _params: dict[str, Any]
    _inferred_grid: GeoReference | None
    _inferred_dtype: np.dtype[Any] | None
    _inferred_units: str | None
    _halo: int

    __hash__ = None

    def __bool__(self) -> None:
        raise TypeError(
            "RasterExpression does not support implicit truth testing. "
            "Use ma.compute(expr) to materialize or ma.explain(expr) / "
            "ma.plan(expr) to inspect."
        )

    def __post_init__(self) -> None:
        if not self._node_id:
            raise ValueError("Expression node ID is required.")

    # -----------------------------------------------------------------
    # Source identity
    # -----------------------------------------------------------------

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

    def _source_descriptor(self) -> dict[str, Any] | None:
        if self._operation_id == "source":
            return dict(self._params)
        if self._operation_id == "constant":
            return {"source": "constant", "name": self._params.get("name", "")}
        return None

    def _walk_nodes(self) -> list[RasterExpression]:
        nodes: list[RasterExpression] = []
        visited: set[str] = set()

        def visit(node: RasterExpression) -> None:
            if node._node_id in visited:
                return
            visited.add(node._node_id)
            nodes.append(node)
            for op in node._operands:
                if isinstance(op, RasterExpression):
                    visit(op)

        visit(self)
        return nodes

    # -----------------------------------------------------------------
    # Operator overloads
    # -----------------------------------------------------------------

    def __add__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.add")

    def __radd__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.add")

    def __sub__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.subtract")

    def __rsub__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.subtract", swap=True)

    def __mul__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.multiply")

    def __rmul__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.multiply")

    def __truediv__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.divide")

    def __rtruediv__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.divide", swap=True)

    def __lt__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.less")

    def __le__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.less_equal")

    def __gt__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.greater")

    def __ge__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.greater_equal")

    def __eq__(self, other: object) -> RasterExpression:  # type: ignore[override]
        from .expression import _binary_op
        return _binary_op(self, other, "local.equal")

    def __ne__(self, other: object) -> RasterExpression:  # type: ignore[override]
        from .expression import _binary_op
        return _binary_op(self, other, "local.not_equal")

    def __and__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.logical_and")

    def __or__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.logical_or")

    def __xor__(self, other: object) -> RasterExpression:
        from .expression import _binary_op
        return _binary_op(self, other, "local.logical_xor")

    def __neg__(self) -> RasterExpression:
        from .expression import _unary_op
        return _unary_op(self, "local.negative")

    def __abs__(self) -> RasterExpression:
        from .expression import _unary_op
        return _unary_op(self, "local.absolute")

    def __invert__(self) -> RasterExpression:
        from .expression import _unary_op
        return _unary_op(self, "local.logical_not")

    # -----------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------

    def scientific_identity(self) -> str:
        parts = [self._operation_id]
        for op in self._operands:
            if isinstance(op, RasterExpression):
                parts.append(op.scientific_identity())
            elif isinstance(op, Raster):
                parts.append(f"raster({op.shape},{op.dtype.name})")
            else:
                parts.append(repr(op))
        for k in sorted(self._params.keys()):
            parts.append(f"{k}={self._params[k]!r}")
        if self._inferred_grid is not None:
            parts.append(f"grid({self._inferred_grid.width}x{self._inferred_grid.height})")
        return "|".join(parts)

    def to_json(self) -> str:
        nodes = self._walk_nodes()
        result: dict[str, Any] = {
            "schema_version": 1,
            "root_node_id": self._node_id,
            "nodes": [],
            "sources": [],
        }
        for node in nodes:
            entry: dict[str, Any] = {
                "node_id": node._node_id,
                "operation_id": node._operation_id,
                "params": {k: _json_safe(v) for k, v in node._params.items()},
                "operands": [
                    op._node_id if isinstance(op, RasterExpression) else _json_safe(op)
                    for op in node._operands
                ],
                "inferred_grid": _grid_json(node._inferred_grid),
                "inferred_dtype": str(node._inferred_dtype) if node._inferred_dtype is not None else None,
                "inferred_units": node._inferred_units,
                "halo": node._halo,
            }
            result["nodes"].append(entry)
            desc = node._source_descriptor()
            if desc is not None:
                result["sources"].append({"node_id": node._node_id, **desc})
        return json.dumps(result, sort_keys=True, indent=2)


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
        "pixel_size_x": georef.pixel_size_x,
        "pixel_size_y": georef.pixel_size_y,
    }
