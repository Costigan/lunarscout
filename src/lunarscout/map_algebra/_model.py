from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import MapAlgebraExpressionError
from ..georeference import GeoReference
from ..raster import Raster
from ._normalization import (
    CANONICAL_SCHEMA_VERSION,
    NORMALIZATION_VERSION,
    normalize_canonical,
    normalize_crs_wkt,
)
from ._registry import get_operation_spec


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

    def describe(self) -> str:
        nodes = self._all_nodes()
        parts = [f"{self._operation_id}; {len(nodes)} node(s)"]
        if self._inferred_grid is not None:
            parts.append(f"@{self._inferred_grid.width}x{self._inferred_grid.height}")
        if self._inferred_dtype is not None:
            parts.append(str(self._inferred_dtype))
        if self._inferred_units is not None:
            parts.append(self._inferred_units)
        return f"RasterExpression({' '.join(parts)})"

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
        payload = self._scientific_payload()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _scientific_payload(self) -> dict[str, Any]:
        spec = get_operation_spec(self._operation_id)
        return {
            "operation_id": self._operation_id,
            "semantic_version": spec.version,
            "parameters": {
                key: normalize_canonical(value) for key, value in self._params
            },
            "operands": [
                operand._scientific_payload()
                if isinstance(operand, RasterExpression)
                else _canonical_operand(operand)
                for operand in self._operands
            ],
            "grid": _grid_json(self._inferred_grid),
            "dtype": str(self._inferred_dtype) if self._inferred_dtype is not None else None,
            "units": self._inferred_units,
            "halo": self._halo,
        }

    @property
    def _params_dict(self) -> dict[str, Any]:
        return dict(self._params)

    def scientific_identity(self) -> str:
        return "sha256:" + self._content_hash()

    def to_json(self) -> str:
        nodes = self._all_nodes()
        canonical_ids = {node._node_id: f"n{index}" for index, node in enumerate(nodes)}
        node_list: list[dict[str, Any]] = []
        sources_list: list[dict[str, Any]] = []

        for node in nodes:
            entry: dict[str, Any] = {
                "node_id": canonical_ids[node._node_id],
                "operation_id": node._operation_id,
                "semantic_version": get_operation_spec(node._operation_id).version,
                "normalized_parameters": {
                    k: normalize_canonical(v) for k, v in node._params
                },
                "operands": [
                    {"type": "node", "node_id": canonical_ids[op._node_id]}
                    if isinstance(op, RasterExpression) else _canonical_operand(op)
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
                sources_list.append({
                    "node_id": canonical_ids[node._node_id],
                    "descriptor": normalize_canonical(desc),
                })

        return json.dumps(
            {
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "root_node_id": canonical_ids[self._node_id],
                "nodes": node_list,
                "sources": sources_list,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_canonical_json(self) -> str:
        """Return the versioned canonical expression representation."""
        return self.to_json()

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

    def __rfloordiv__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.floor_divide", swap=True)

    def __mod__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.remainder")

    def __rmod__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.remainder", swap=True)

    def __pow__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.power")

    def __rpow__(self, other: object) -> RasterExpression:
        return _new_expr_op(self, other, "local.power", swap=True)

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
    get_operation_spec(operation_id)
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
    from ._validation import _as_expression_operand

    return _as_expression_operand(value)


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
    output_units: str | None = None,
    operands_in_order: list[Any] | None = None,
) -> str | None:
    units: list[str | None] = []
    for op in operands:
        if isinstance(op, RasterExpression):
            units.append(op.units)
    equality_required = operation_id in (
        "local.add", "local.subtract", "local.minimum", "local.maximum",
        "local.less", "local.less_equal", "local.greater", "local.greater_equal",
        "local.equal", "local.not_equal", "local.isclose",
    )
    if equality_required and len(units) > 1 and any(unit != units[0] for unit in units[1:]):
        raise MapAlgebraExpressionError(
            f"Unit mismatch in '{operation_id}': {units}",
            code="map_algebra_unit_mismatch",
            details={"operation_id": operation_id, "units": units},
        )
    if operation_id in (
        "local.less", "local.less_equal", "local.greater", "local.greater_equal",
        "local.equal", "local.not_equal", "local.logical_and", "local.logical_or",
        "local.logical_xor", "local.logical_not",
    ):
        return None
    if operation_id in ("local.add", "local.subtract", "local.minimum", "local.maximum"):
        if len(units) == 0:
            return None
        return units[0] if units else None
    if operation_id in ("local.multiply", "local.divide"):
        if output_units is not None:
            return output_units
        if len(units) > 1 and all(unit is not None for unit in units):
            raise MapAlgebraExpressionError(
                f"'{operation_id}' with two unit-bearing rasters requires explicit output units.",
                code="map_algebra_missing_output_units",
                details={"operation_id": operation_id, "units": units},
            )
        return None
    if operation_id == "local.power":
        from ._units import power_units

        ordered = operands_in_order or operands
        base = ordered[0]
        exponent = ordered[1]
        base_is_raster = isinstance(base, RasterExpression)
        exponent_is_scalar = not isinstance(exponent, RasterExpression)
        return power_units(
            base_units=base.units if base_is_raster else None,
            exponent_units=(
                exponent.units
                if isinstance(exponent, RasterExpression)
                else None
            ),
            exponent_is_scalar=exponent_is_scalar,
            exponent_value=exponent if exponent_is_scalar else None,
            output_units=output_units,
            base_is_raster=base_is_raster,
        )
    if operation_id == "local.arctan2":
        return "radians"
    if operation_id == "local.hypot":
        return None
    return units[0] if units else None


def _new_expr_op(
    self_expr: RasterExpression | Raster,
    other: Any,
    op_id: str,
    swap: bool = False,
    params: dict[str, Any] | None = None,
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

    if is_boolean:
        for index, operand in enumerate(raster_ops):
            if operand.dtype != np.dtype(np.bool_):
                raise MapAlgebraExpressionError(
                    f"Boolean operation '{op_id}' requires Boolean raster operands.",
                    code="map_algebra_requires_boolean",
                    details={"argument_index": index, "dtype": str(operand.dtype)},
                )

    if is_comparison or is_boolean:
        dtype = _infer_comparison_dtype()
    else:
        from ._dtypes import result_dtype

        operand_dtypes = tuple(
            op.dtype for op in op_list
            if isinstance(op, RasterExpression) and op.dtype is not None
        )
        scalars = tuple(
            op for op in op_list if not isinstance(op, RasterExpression)
        )
        dtype = result_dtype(
            operand_dtypes,
            operation=op_id.removeprefix("local."),
            scalars=scalars,
            overflow=(params or {}).get("overflow", "raise"),
            scalar_left=(
                len(op_list) == 2
                and not isinstance(op_list[0], RasterExpression)
                and isinstance(op_list[1], RasterExpression)
            ),
        ) if operand_dtypes else None

    units = _infer_expr_units(
        raster_ops, operation_id=op_id,
        output_units=(params or {}).get("output_units"),
        operands_in_order=op_list,
    )
    return _make_expr_node(
        op_id, tuple(op_list), grid=grid, dtype=dtype, units=units, params=params,
    )


def _new_expr_unary(
    self_expr: RasterExpression | Raster,
    op_id: str,
    *,
    params: dict[str, Any] | None = None,
) -> RasterExpression:
    expr = _to_expr_or_scalar(self_expr)
    if isinstance(expr, RasterExpression):
        is_logic = op_id == "local.logical_not"
        if is_logic and expr.dtype != np.dtype(np.bool_):
            raise MapAlgebraExpressionError(
                "logical_not requires a Boolean raster operand.",
                code="map_algebra_requires_boolean",
                details={"dtype": str(expr.dtype)},
            )
        if op_id in {"local.sin", "local.cos", "local.tan"}:
            from ._units import require_angle_units
            require_angle_units(expr.units, argument="a")
        output_units = (
            "radians" if op_id in {"local.arcsin", "local.arccos", "local.arctan", "local.radians"}
            else "degrees" if op_id == "local.degrees"
            else None if op_id in {"local.sin", "local.cos", "local.tan"}
            else expr.units
        )
        from ._dtypes import result_dtype

        dtype = (
            _infer_comparison_dtype()
            if is_logic
            else result_dtype(
                (expr.dtype,),
                operation=op_id.removeprefix("local."),
                overflow=(params or {}).get("overflow", "raise"),
            ) if expr.dtype is not None else None
        )
        return _make_expr_node(
            op_id, (expr,),
            grid=expr.grid,
            dtype=dtype,
            units=output_units,
            params=params,
        )
    raise MapAlgebraExpressionError(
        f"Unary operation requires a Raster or RasterExpression operand.",
        code="map_algebra_invalid_operand",
    )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _canonical_operand(value: Any) -> Any:
    from ._temporal_model import TemporalRasterExpression

    if isinstance(value, TemporalRasterExpression):
        return {
            "type": "temporal_expression",
            "scientific_identity": value.scientific_identity(),
        }
    if isinstance(value, Raster):
        digest = hashlib.sha256()
        digest.update(value.values.tobytes(order="C"))
        digest.update(value.valid.tobytes(order="C"))
        return {
            "type": "raster",
            "content_sha256": digest.hexdigest(),
            "shape": [str(size) for size in value.shape],
            "dtype": value.dtype.str,
            "grid": _grid_json(value.georef),
            "units": value.units,
        }
    return normalize_canonical(value)


def _grid_json(georef: GeoReference | None) -> dict[str, Any] | None:
    if georef is None:
        return None
    return {
        "width": normalize_canonical(georef.width),
        "height": normalize_canonical(georef.height),
        "affine": normalize_canonical(tuple(georef.affine_transform)),
        "pixel_size_x": normalize_canonical(georef.pixel_size_x),
        "pixel_size_y": normalize_canonical(georef.pixel_size_y),
        "crs_wkt": normalize_crs_wkt(georef.projection_wkt),
        "nodata": normalize_canonical(georef.nodata),
    }
