from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import (
    MapAlgebraError,
    MapAlgebraExpressionError,
    MapAlgebraGridError,
)
from ..georeference import GeoReference
from ..temporal import TemporalCube
from ..raster import Raster as _Raster, _validate_raster_dtype
from ._model import RasterExpression
from ._normalization import (
    CANONICAL_SCHEMA_VERSION,
    NORMALIZATION_VERSION,
    normalize_canonical,
    normalize_crs_wkt,
)
from ._registry import get_operation_spec
from ._sources import constant as _spatial_constant
from ._validation import _is_scalar


# ---------------------------------------------------------------------------
# TemporalRaster -- eager, in-memory temporal raster
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, eq=False)
class TemporalRaster:
    values: np.ndarray
    times: np.ndarray
    georef: GeoReference
    valid: np.ndarray
    units: str | None = None
    signal_name: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise MapAlgebraError(
                "TemporalRaster values must be a three-dimensional array (time, y, x).",
                code="temporal_raster_invalid_shape",
                details={"ndim": int(self.values.ndim)},
            )
        _validate_raster_dtype(self.values.dtype)
        if self.times.ndim != 1:
            raise MapAlgebraError(
                "TemporalRaster times must be a one-dimensional array.",
                code="temporal_raster_invalid_times",
                details={"ndim": int(self.times.ndim)},
            )
        if not np.issubdtype(self.times.dtype, np.datetime64):
            raise MapAlgebraError(
                "TemporalRaster times must have datetime64 dtype.",
                code="temporal_raster_invalid_times_dtype",
                details={"dtype": str(self.times.dtype)},
            )
        if len(self.times) == 0 or self.values.shape[0] == 0:
            raise MapAlgebraError(
                "TemporalRaster must contain at least one time sample.",
                code="temporal_raster_empty",
            )
        if len(self.times) != self.values.shape[0]:
            raise MapAlgebraError(
                "TemporalRaster time count must match the time dimension.",
                code="temporal_raster_time_shape_mismatch",
                details={"times": int(len(self.times)), "values_time": int(self.values.shape[0])},
            )
        if self.valid.shape != self.values.shape:
            raise MapAlgebraError(
                "TemporalRaster valid array shape must match values shape.",
                code="temporal_raster_valid_shape_mismatch",
                details={"valid_shape": list(self.valid.shape), "values_shape": list(self.values.shape)},
            )
        if self.valid.dtype != np.dtype(np.bool_):
            raise MapAlgebraError(
                "TemporalRaster valid must have bool dtype.",
                code="temporal_raster_valid_dtype",
                details={"dtype": str(self.valid.dtype)},
            )
        expected_spatial = (int(self.georef.height), int(self.georef.width))
        if self.values.shape[1:] != expected_spatial:
            raise MapAlgebraError(
                "TemporalRaster spatial shape does not match GeoReference dimensions.",
                code="temporal_raster_spatial_mismatch",
                details={
                    "shape": list(self.values.shape[1:]),
                    "expected": list(expected_spatial),
                },
            )
        if np.any(np.isnat(self.times)):
            raise MapAlgebraError(
                "TemporalRaster times must not contain NaT.",
                code="temporal_raster_nat",
            )
        if len(self.times) > 1:
            diffs = np.diff(self.times.astype(np.int64))
            if np.any(diffs <= 0):
                raise MapAlgebraError(
                    "TemporalRaster times must be strictly increasing.",
                    code="temporal_raster_times_not_increasing",
                )

    def __bool__(self) -> None:
        raise TypeError(
            "TemporalRaster does not support implicit truth testing. "
            "Use .all_valid, .same_grid(), .same_times(), "
            "or materialize with ma.compute()."
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_layers, self.height, self.width)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    @property
    def height(self) -> int:
        return int(self.georef.height)

    @property
    def width(self) -> int:
        return int(self.georef.width)

    @property
    def num_layers(self) -> int:
        return int(self.values.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes + self.valid.nbytes + self.times.nbytes)

    @property
    def all_valid(self) -> bool:
        return bool(np.all(self.valid))

    @property
    def invalid_count(self) -> int:
        return int(np.sum(~self.valid))

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def copy(self) -> TemporalRaster:
        return TemporalRaster(
            values=self.values.copy(),
            times=self.times.copy(),
            georef=self.georef,
            valid=self.valid.copy(),
            units=self.units,
            signal_name=self.signal_name,
            name=self.name,
        )

    def readonly(self) -> TemporalRaster:
        values = self.values.copy()
        values.flags.writeable = False
        valid = self.valid.copy()
        valid.flags.writeable = False
        return TemporalRaster(
            values=values,
            times=self.times.copy(),
            georef=self.georef,
            valid=valid,
            units=self.units,
            signal_name=self.signal_name,
            name=self.name,
        )

    def filled(self, value: int | float = 0) -> np.ndarray:
        result = self.values.copy()
        result[~self.valid] = value
        return result

    def masked(self) -> np.ma.MaskedArray:
        return np.ma.array(self.values, mask=~self.valid)

    def with_name(self, name: str | None) -> TemporalRaster:
        return TemporalRaster(
            values=self.values, times=self.times, georef=self.georef,
            valid=self.valid, units=self.units,
            signal_name=self.signal_name, name=name,
        )

    def with_units(self, units: str | None) -> TemporalRaster:
        return TemporalRaster(
            values=self.values, times=self.times, georef=self.georef,
            valid=self.valid, units=units,
            signal_name=self.signal_name, name=self.name,
        )

    def with_validity(self, valid: np.ndarray) -> TemporalRaster:
        return TemporalRaster(
            values=self.values, times=self.times, georef=self.georef,
            valid=valid, units=self.units,
            signal_name=self.signal_name, name=self.name,
        )

    def with_georef(self, georef: GeoReference) -> TemporalRaster:
        return TemporalRaster(
            values=self.values, times=self.times, georef=georef,
            valid=self.valid, units=self.units,
            signal_name=self.signal_name, name=self.name,
        )

    def with_times(self, times: np.ndarray) -> TemporalRaster:
        return TemporalRaster(
            values=self.values, times=times, georef=self.georef,
            valid=self.valid, units=self.units,
            signal_name=self.signal_name, name=self.name,
        )

    def with_signal_name(self, signal_name: str | None) -> TemporalRaster:
        return TemporalRaster(
            values=self.values, times=self.times, georef=self.georef,
            valid=self.valid, units=self.units,
            signal_name=signal_name, name=self.name,
        )

    def same_grid(self, other: TemporalRaster | _Raster) -> bool:
        from ..alignment import same_grid as _same_grid
        if isinstance(other, TemporalRaster):
            return _same_grid(self.georef, other.georef)
        if isinstance(other, _Raster):
            return _same_grid(self.georef, other.georef)
        return False

    def same_times(self, other: TemporalRaster) -> bool:
        if len(self.times) != len(other.times):
            return False
        return bool(np.array_equal(self.times, other.times))

    def layer(self, index: int) -> _Raster:
        if index < 0 or index >= self.num_layers:
            raise MapAlgebraError(
                "Layer index out of range.",
                code="temporal_raster_layer_index",
                details={"index": index, "num_layers": self.num_layers},
            )
        return _Raster(
            values=self.values[index].copy(),
            georef=self.georef,
            valid=self.valid[index].copy(),
            units=self.units,
            name=f"{self.name or 'temporal'}_{index}",
        )

    def expression(self) -> TemporalRasterExpression:
        return _temporal_constant(self)

    __hash__ = None

    def __repr__(self) -> str:
        total = self.num_layers * self.height * self.width
        parts = [
            f"TemporalRaster(shape=({self.num_layers}, {self.height}, {self.width})",
            f"dtype={self.dtype.name}",
        ]
        if self.all_valid:
            parts.append("valid=all")
        else:
            parts.append(f"valid={total - self.invalid_count}/{total}")
        if self.signal_name is not None:
            parts.append(f"signal={self.signal_name!r}")
        if self.units is not None:
            parts.append(f"units={self.units!r}")
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def from_temporal_cube(cube: TemporalCube, units: str | None = None, name: str | None = None) -> TemporalRaster:
    values = np.asarray(cube.values)
    raw_times = cube.times
    if hasattr(raw_times, 'values'):
        times = np.asarray(raw_times.values)  # type: ignore[union-attr]
    else:
        times = np.asarray(raw_times)
    georef = cube.georef
    nodata = georef.nodata
    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            valid = ~np.isnan(values)
        else:
            valid = values != np.asarray(nodata, dtype=values.dtype)
    else:
        valid = np.ones(values.shape, dtype=np.bool_)
    return TemporalRaster(
        values=values,
        times=times,
        georef=georef,
        valid=valid,
        units=units,
        name=name,
    )


def to_temporal_cube(tr: TemporalRaster) -> TemporalCube:
    values = tr.values.copy()
    nodata = tr.georef.nodata
    if nodata is not None:
        values[~tr.valid] = nodata
    return TemporalCube(values=values, times=tr.times, georef=tr.georef)


# ---------------------------------------------------------------------------
# TemporalRasterExpression
# ---------------------------------------------------------------------------

class __TP_SEALED:
    pass

_TP_SEAL = __TP_SEALED()

_tp_counter: int = 0


def _next_tp_id() -> str:
    global _tp_counter
    _tp_counter += 1
    return f"tp{_tp_counter:08d}"


@dataclass(frozen=True, slots=True)
class TemporalRasterExpression:
    _node_id: str = field(compare=False, hash=False)
    _operation_id: str
    _operands: tuple[Any, ...] = field(default_factory=tuple)
    _params: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    _inferred_grid: GeoReference | None = None
    _inferred_dtype: np.dtype[Any] | None = None
    _inferred_units: str | None = None
    _inferred_times: np.ndarray | None = None
    _halo: int = 0
    _signal_name: str | None = None
    _sealed: object = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self._sealed is not _TP_SEAL:
            raise MapAlgebraExpressionError(
                "TemporalRasterExpression cannot be constructed directly. "
                "Use ma.temporal_source(), ma.temporal_mean(), or map-algebra operators.",
                code="map_algebra_sealed_constructor",
            )

    __hash__ = None

    def __bool__(self) -> None:
        raise TypeError(
            "TemporalRasterExpression does not support implicit truth testing. "
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
    def times(self) -> np.ndarray | None:
        return self._inferred_times

    @property
    def operation_id(self) -> str:
        return self._operation_id

    @property
    def _params_dict(self) -> dict[str, Any]:
        return dict(self._params)

    def describe(self) -> str:
        parts = [f"operation: {self._operation_id}"]
        if self._inferred_grid is not None:
            parts.append(f"grid: ({self._inferred_grid.height}x{self._inferred_grid.width})")
        if self._inferred_dtype is not None:
            parts.append(f"dtype: {self._inferred_dtype.name}")
        if self._inferred_units is not None:
            parts.append(f"units: {self._inferred_units}")
        if self._inferred_times is not None:
            parts.append(f"times: {len(self._inferred_times)} layers")
        else:
            parts.append("times: (reduced -- spatial result)")
        if self._signal_name is not None:
            parts.append(f"signal: {self._signal_name}")
        return f"TemporalRasterExpression({', '.join(parts)})"

    def scientific_identity(self) -> str:
        encoded = self.to_canonical_json().encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def to_json(self) -> str:
        return _temporal_to_json(self)

    def to_canonical_json(self) -> str:
        """Return the versioned canonical temporal expression representation."""
        return self.to_json()

    # -----------------------------------------------------------------
    # Operator overloads
    # -----------------------------------------------------------------

    def __add__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.add", self, other)

    def __radd__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.add", other, self)

    def __sub__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.subtract", self, other)

    def __rsub__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.subtract", other, self)

    def __mul__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.multiply", self, other)

    def __rmul__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.multiply", other, self)

    def __truediv__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.divide", self, other)

    def __rtruediv__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.divide", other, self)

    def __floordiv__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.floor_divide", self, other)

    def __rfloordiv__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.floor_divide", other, self)

    def __mod__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.remainder", self, other)

    def __rmod__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.remainder", other, self)

    def __pow__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.power", self, other)

    def __rpow__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.power", other, self)

    def __lt__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.less", self, other)

    def __le__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.less_equal", self, other)

    def __gt__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.greater", self, other)

    def __ge__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.greater_equal", self, other)

    def __eq__(self, other: object) -> TemporalRasterExpression:  # type: ignore[override]
        return _temporal_local_op("local.equal", self, other)

    def __ne__(self, other: object) -> TemporalRasterExpression:  # type: ignore[override]
        return _temporal_local_op("local.not_equal", self, other)

    def __and__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.logical_and", self, other)

    def __rand__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.logical_and", other, self)

    def __or__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.logical_or", self, other)

    def __ror__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.logical_or", other, self)

    def __xor__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.logical_xor", self, other)

    def __rxor__(self, other: object) -> TemporalRasterExpression:
        return _temporal_local_op("local.logical_xor", other, self)

    def __neg__(self) -> TemporalRasterExpression:
        return _temporal_local_op("local.negative", self)

    def __abs__(self) -> TemporalRasterExpression:
        return _temporal_local_op("local.absolute", self)

    def __invert__(self) -> TemporalRasterExpression:
        return _temporal_local_op("local.logical_not", self)


# ---------------------------------------------------------------------------
# Node factory helpers
# ---------------------------------------------------------------------------

def _make_tp_node(
    operation_id: str,
    *operands: Any,
    grid: GeoReference | None = None,
    dtype: np.dtype[Any] | None = None,
    units: str | None = None,
    times: np.ndarray | None = None,
    halo: int = 0,
    signal_name: str | None = None,
    params: dict[str, Any] | None = None,
) -> TemporalRasterExpression:
    get_operation_spec(operation_id)
    sorted_params = tuple(sorted((params or {}).items()))
    return TemporalRasterExpression(
        _node_id=_next_tp_id(),
        _operation_id=operation_id,
        _operands=tuple(operands),
        _params=sorted_params,
        _inferred_grid=grid,
        _inferred_dtype=dtype,
        _inferred_units=units,
        _inferred_times=times,
        _halo=halo,
        _signal_name=signal_name,
        _sealed=_TP_SEAL,
    )


def _temporal_constant(tr: TemporalRaster) -> TemporalRasterExpression:
    return _make_tp_node(
        "temporal.constant",
        tr,
        grid=tr.georef,
        dtype=tr.dtype,
        units=tr.units,
        times=tr.times,
        signal_name=tr.signal_name,
        params={"signal_name": tr.signal_name, "name": tr.name},
    )


def _temporal_broadcast(spatial: _Raster | RasterExpression, *, times_hint: np.ndarray | None = None) -> TemporalRasterExpression:
    if isinstance(spatial, _Raster):
        expr = _spatial_constant(spatial)
    else:
        expr = spatial
    grid = getattr(expr, '_inferred_grid', None)
    dtype = getattr(expr, '_inferred_dtype', None)
    units = getattr(expr, '_inferred_units', None)
    return _make_tp_node(
        "temporal.broadcast",
        expr,
        grid=grid,
        dtype=dtype,
        units=units,
        times=times_hint,
        params={},
    )


def _temporal_source_node(
    series_path: str,
    georef: GeoReference,
    dtype: np.dtype[Any],
    times: np.ndarray,
    signal_name: str | None = None,
    units: str | None = None,
) -> TemporalRasterExpression:
    params: dict[str, Any] = {
        "path": series_path,
        "signal_name": signal_name,
        "dtype": str(dtype),
    }
    return _make_tp_node(
        "temporal.source",
        grid=georef,
        dtype=dtype,
        units=units,
        times=times,
        signal_name=signal_name,
        params=params,
    )


def _temporal_local_op(op_id: str, *operands: Any) -> TemporalRasterExpression:
    tp_operands: list[Any] = []
    temporal_ops: list[TemporalRasterExpression] = []
    spatial_ops: list[Any] = []

    for op in operands:
        if isinstance(op, TemporalRasterExpression):
            tp_operands.append(op)
            temporal_ops.append(op)
        elif isinstance(op, TemporalRaster):
            const = _temporal_constant(op)
            tp_operands.append(const)
            temporal_ops.append(const)
        elif isinstance(op, (_Raster, RasterExpression)):
            tp_operands.append(op)
            spatial_ops.append(op)
        else:
            tp_operands.append(op)

    # Infer and validate time coordinates from temporal operands
    times: np.ndarray | None = None
    primary_times: np.ndarray | None = None
    for t_op in temporal_ops:
        if t_op._inferred_times is not None:
            if primary_times is None:
                primary_times = t_op._inferred_times
            elif not np.array_equal(primary_times, t_op._inferred_times):
                raise MapAlgebraExpressionError(
                    "Temporal operands must have matching time coordinates.",
                    code="temporal_times_mismatch",
                    details={
                        "a_count": int(len(primary_times)),
                        "b_count": int(len(t_op._inferred_times)),
                    },
                )
    times = primary_times

    # Broadcast spatial operands using inferred times
    for i, op in enumerate(tp_operands):
        if isinstance(op, (_Raster, RasterExpression)):
            tp_operands[i] = _temporal_broadcast(op, times_hint=times)

    # Infer and validate spatial grid compatibility
    all_grids: list[GeoReference] = []
    for op in tp_operands:
        if isinstance(op, TemporalRasterExpression) and op._inferred_grid is not None:
            all_grids.append(op._inferred_grid)
    if len(all_grids) >= 2:
        from ..alignment import require_same_grid
        require_same_grid(all_grids[0], all_grids[1])
        for g in all_grids[2:]:
            require_same_grid(all_grids[0], g)
    grid = all_grids[0] if all_grids else None

    # Infer dtype
    is_comparison = op_id in (
        "local.less", "local.less_equal", "local.greater", "local.greater_equal",
        "local.equal", "local.not_equal",
    )
    is_boolean = op_id in ("local.logical_and", "local.logical_or", "local.logical_xor", "local.logical_not")
    if is_comparison or is_boolean:
        dtype = np.dtype(np.bool_)
    else:
        dtypes: list[np.dtype[Any]] = []
        for op in tp_operands:
            if isinstance(op, TemporalRasterExpression) and op._inferred_dtype is not None:
                dtypes.append(np.dtype(op._inferred_dtype))
            elif _is_scalar(op):
                dtypes.append(np.result_type(type(op)))
        dtype = np.result_type(*dtypes) if dtypes else None

    # Infer and validate units
    if is_comparison or is_boolean:
        units = None
    else:
        t_units = [
            op._inferred_units
            for op in tp_operands
            if isinstance(op, TemporalRasterExpression) and op._inferred_units is not None
        ]
        if op_id in ("local.add", "local.subtract"):
            if len(t_units) >= 2 and t_units[0] != t_units[1]:
                raise MapAlgebraExpressionError(
                    f"Unit mismatch in temporal '{op_id}': {t_units[0]} vs {t_units[1]}",
                    code="map_algebra_unit_mismatch",
                    details={"units": t_units},
                )
            units = t_units[0] if t_units else None
        elif op_id in ("local.multiply", "local.divide"):
            units = None
        else:
            units = t_units[0] if t_units else None

    signal_name = None
    for op in tp_operands:
        if isinstance(op, TemporalRasterExpression) and op._signal_name is not None:
            signal_name = op._signal_name
            break

    return _make_tp_node(op_id, *tp_operands, grid=grid, dtype=dtype, units=units, times=times, signal_name=signal_name)


def _temporal_reduction_node(op_id: str, expr: TemporalRasterExpression, **params: Any) -> RasterExpression:
    from ._model import _make_expr_node
    return _make_expr_node(
        op_id, (expr,),
        grid=expr._inferred_grid,
        dtype=_reduction_output_dtype(op_id, expr._inferred_dtype),
        units=_reduction_output_units(op_id, expr._inferred_units),
        params=params,
    )


def _reduction_output_dtype(op_id: str, source_dtype: np.dtype[Any] | None) -> np.dtype[Any] | None:
    if source_dtype is None:
        return None
    from ._dtypes import accumulator_dtype

    if op_id in {
        "temporal.mean", "temporal.min", "temporal.max",
        "temporal.std", "temporal.sum", "temporal.count",
    }:
        return accumulator_dtype(np.dtype(source_dtype), operation=op_id)
    return np.dtype(source_dtype)


def _reduction_output_units(op_id: str, source_units: str | None) -> str | None:
    if op_id == "temporal.count":
        return None
    return source_units


# ---------------------------------------------------------------------------
# Topological sort, identity, serialization
# ---------------------------------------------------------------------------

def _topological_sort_temporal(root: TemporalRasterExpression) -> list[TemporalRasterExpression]:
    from collections import deque

    edges: dict[str, list[str]] = {}
    indegree: dict[str, int] = {}
    node_map: dict[str, TemporalRasterExpression] = {}

    queue: deque[TemporalRasterExpression] = deque([root])
    while queue:
        node = queue.popleft()
        nid = node._node_id
        if nid in node_map:
            continue
        node_map[nid] = node
        edges.setdefault(nid, [])
        indegree.setdefault(nid, 0)
        for op in node._operands:
            if isinstance(op, (TemporalRasterExpression, RasterExpression)):
                child_nid: str = op._node_id
                edges.setdefault(child_nid, [])
                indegree.setdefault(child_nid, 0)
                edges[child_nid].append(nid)
                indegree[nid] = indegree.get(nid, 0) + 1
                if isinstance(op, TemporalRasterExpression):
                    queue.append(op)

    result: list[TemporalRasterExpression] = []
    ready = deque(nid for nid, deg in indegree.items() if deg == 0)
    while ready:
        nid = ready.popleft()
        if nid in node_map:
            result.append(node_map[nid])
        for parent_nid in edges.get(nid, []):
            indegree[parent_nid] -= 1
            if indegree[parent_nid] == 0:
                ready.append(parent_nid)

    if len(result) != len(node_map):
        raise MapAlgebraExpressionError(
            "Temporal expression graph contains a cycle.",
            code="map_algebra_expression_cycle",
        )
    return result


def _temporal_to_json(expr: TemporalRasterExpression) -> str:
    nodes = _topological_sort_temporal(expr)
    canonical_ids = {node._node_id: f"n{index}" for index, node in enumerate(nodes)}
    node_list: list[dict[str, Any]] = []

    for node in nodes:
        entry: dict[str, Any] = {
            "node_id": canonical_ids[node._node_id],
            "operation_id": node._operation_id,
            "semantic_version": get_operation_spec(node._operation_id).version,
            "normalized_parameters": {
                k: normalize_canonical(v) for k, v in node._params_dict.items()
            },
            "operands": [
                _json_operand_tp(op, canonical_ids=canonical_ids) for op in node._operands
            ],
            "grid": _grid_json_tp(node._inferred_grid),
            "dtype": str(node._inferred_dtype) if node._inferred_dtype is not None else None,
            "units": node._inferred_units,
            "times_count": len(node._inferred_times) if node._inferred_times is not None else None,
            "times": _times_json_tp(node._inferred_times),
            "times_reduced": node._inferred_times is None,
            "signal_name": node._signal_name,
        }
        node_list.append(entry)

    return json.dumps(
        {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "domain": "temporal",
            "root_node_id": canonical_ids[expr._node_id],
            "nodes": node_list,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_operand_tp(op: Any, *, canonical_ids: dict[str, str] | None = None) -> Any:
    if isinstance(op, TemporalRasterExpression):
        node_id = canonical_ids.get(op._node_id, op._node_id) if canonical_ids else op._node_id
        return {"type": "temporal_expression", "node_id": node_id}
    if isinstance(op, RasterExpression):
        return {
            "type": "raster_expression",
            "scientific_identity": op.scientific_identity(),
        }
    if isinstance(op, _Raster):
        import hashlib

        digest = hashlib.sha256()
        digest.update(op.values.tobytes(order="C"))
        digest.update(op.valid.tobytes(order="C"))
        return {
            "type": "raster",
            "shape": [str(size) for size in op.shape],
            "dtype": op.dtype.str,
            "content_sha256": digest.hexdigest(),
        }
    if isinstance(op, TemporalRaster):
        import hashlib

        digest = hashlib.sha256()
        digest.update(op.values.tobytes(order="C"))
        digest.update(op.valid.tobytes(order="C"))
        digest.update(op.times.tobytes(order="C"))
        return {
            "type": "temporal_raster",
            "shape": [str(size) for size in op.shape],
            "dtype": op.dtype.str,
            "content_sha256": digest.hexdigest(),
        }
    return normalize_canonical(op)


def _grid_json_tp(georef: GeoReference | None) -> dict[str, Any] | None:
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


def _times_json_tp(times: np.ndarray | None) -> dict[str, Any] | None:
    if times is None:
        return None
    digest = hashlib.sha256(np.ascontiguousarray(times).tobytes()).hexdigest()
    return {
        "dtype": times.dtype.str,
        "count": normalize_canonical(len(times)),
        "sha256": digest,
    }


# ---------------------------------------------------------------------------
# Eager computation of TemporalRasterExpression
# ---------------------------------------------------------------------------

def _compute_temporal(expression: TemporalRasterExpression) -> TemporalRaster:
    from .expression import compute as _compute_spatial
    from ._kernels import (
        _add, _subtract, _multiply, _divide, _floor_divide, _remainder,
        _power, _negate, _absolute, _sqrt, _square, _exp, _log, _log10,
        _sin, _cos, _tan, _arcsin, _arccos, _arctan, _degrees, _radians,
        _less, _less_equal, _greater, _greater_equal, _equal, _not_equal,
        _logical_and, _logical_or, _logical_xor, _logical_not,
        _maximum, _minimum, _clip, _hypot, _arctan2,
        _round_half_even, _floor, _ceil, _trunc,
    )

    spatial_cache: dict[str, _Raster] = {}

    def _resolve_spatial(se: RasterExpression) -> _Raster:
        key = se._node_id
        if key not in spatial_cache:
            spatial_cache[key] = _compute_spatial(se)
        return spatial_cache[key]

    _UNARY_KERNELS: dict[str, Any] = {
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
        "local.degrees": _degrees,
        "local.radians": _radians,
        "local.logical_not": _logical_not,
        "local.floor": _floor,
        "local.ceil": _ceil,
        "local.trunc": _trunc,
        "local.round": _round_half_even,
    }

    _BINARY_KERNELS: dict[str, Any] = {
        "local.add": _add,
        "local.subtract": _subtract,
        "local.multiply": _multiply,
        "local.divide": _divide,
        "local.floor_divide": _floor_divide,
        "local.remainder": _remainder,
        "local.power": _power,
        "local.less": _less,
        "local.less_equal": _less_equal,
        "local.greater": _greater,
        "local.greater_equal": _greater_equal,
        "local.equal": _equal,
        "local.not_equal": _not_equal,
        "local.logical_and": _logical_and,
        "local.logical_or": _logical_or,
        "local.logical_xor": _logical_xor,
        "local.minimum": _minimum,
        "local.maximum": _maximum,
        "local.hypot": _hypot,
        "local.arctan2": _arctan2,
    }

    sorted_nodes = _topological_sort_temporal(expression)
    node_results: dict[str, np.ndarray] = {}
    node_valid: dict[str, np.ndarray] = {}

    times = expression._inferred_times
    if times is None:
        raise MapAlgebraExpressionError(
            "Cannot compute temporal expression with no time coordinates.",
            code="temporal_no_times",
        )

    grid = expression._inferred_grid
    if grid is None:
        raise MapAlgebraExpressionError(
            "Cannot compute temporal expression with no grid.",
            code="temporal_no_grid",
        )

    nt = len(times)

    for node in sorted_nodes:
        op_id = node._operation_id

        if op_id == "temporal.constant":
            for operand in node._operands:
                if isinstance(operand, TemporalRaster):
                    tr = operand
                    node_results[node._node_id] = tr.values
                    node_valid[node._node_id] = tr.valid
                    break
            else:
                raise MapAlgebraExpressionError(
                    "Temporal constant node missing raster data.",
                    code="temporal_internal_error",
                )
            continue

        if op_id == "temporal.source":
            path_str = node._params_dict.get("path", "")
            if path_str:
                from ..temporal_store import open_temporal_cube
                series_obj = open_temporal_cube(path_str)
                try:
                    all_layers = np.empty((nt, grid.height, grid.width), dtype=node._inferred_dtype)
                    all_valid = np.empty((nt, grid.height, grid.width), dtype=np.bool_)
                    for t_ix in range(nt):
                        layer_arr, layer_georef = series_obj.read_layer(t_ix)
                        all_layers[t_ix] = layer_arr
                        nodata_v = layer_georef.nodata
                        if nodata_v is not None:
                            if isinstance(nodata_v, float) and np.isnan(nodata_v):
                                all_valid[t_ix] = ~np.isnan(layer_arr)
                            else:
                                all_valid[t_ix] = layer_arr != np.asarray(nodata_v, dtype=layer_arr.dtype)
                        else:
                            all_valid[t_ix] = np.ones((grid.height, grid.width), dtype=np.bool_)
                    node_results[node._node_id] = all_layers
                    node_valid[node._node_id] = all_valid
                finally:
                    series_obj.close()
                continue
            raise MapAlgebraExpressionError(
                "Temporal source node has no path.",
                code="temporal_internal_error",
                details={"params": list(node._params_dict.keys())},
            )

        if op_id == "temporal.broadcast":
            spatial_expr = node._operands[0]
            spatial_raster = _resolve_spatial(spatial_expr)
            vals = np.tile(spatial_raster.values, (nt, 1, 1))
            valid = np.tile(spatial_raster.valid, (nt, 1, 1))
            node_results[node._node_id] = vals
            node_valid[node._node_id] = valid
            continue

        if op_id in _UNARY_KERNELS:
            kernel = _UNARY_KERNELS[op_id]
            operand = node._operands[0]
            val = _get_op_result(operand, node_results)
            vld = _get_op_valid(operand, node_valid)
            result = kernel(val)
            node_results[node._node_id] = result
            node_valid[node._node_id] = vld.copy()
            continue

        if op_id in _BINARY_KERNELS:
            kernel = _BINARY_KERNELS[op_id]
            a = node._operands[0]
            b = node._operands[1]

            if _is_scalar(a):
                a_val = float(a) if isinstance(a, (float, np.floating)) else int(a)
                b_val = _get_op_result(b, node_results)
                b_vld = _get_op_valid(b, node_valid)
                result = kernel(np.full(b_val.shape, a_val, dtype=b_val.dtype), b_val)
                node_results[node._node_id] = result
                node_valid[node._node_id] = b_vld.copy()
            elif _is_scalar(b):
                b_val = float(b) if isinstance(b, (float, np.floating)) else int(b)
                a_val = _get_op_result(a, node_results)
                a_vld = _get_op_valid(a, node_valid)
                result = kernel(a_val, b_val)
                node_results[node._node_id] = result
                node_valid[node._node_id] = a_vld.copy()
            else:
                a_val = _get_op_result(a, node_results)
                a_vld = _get_op_valid(a, node_valid)
                b_val = _get_op_result(b, node_results)
                b_vld = _get_op_valid(b, node_valid)
                result = kernel(a_val, b_val)
                node_results[node._node_id] = result
                node_valid[node._node_id] = a_vld & b_vld
            continue

        if op_id == "local.clip":
            val = _get_op_result(node._operands[0], node_results)
            vld = _get_op_valid(node._operands[0], node_valid)
            lower = node._params_dict.get("lower")
            upper = node._params_dict.get("upper")
            result = _clip(val, lower, upper)
            node_results[node._node_id] = result
            node_valid[node._node_id] = vld.copy()
            continue

        raise MapAlgebraExpressionError(
            f"Unsupported temporal operation: {op_id}",
            code="temporal_unsupported_operation",
            details={"operation": op_id},
        )

    root_val = node_results[expression._node_id]
    root_valid = node_valid[expression._node_id]
    return TemporalRaster(
        values=root_val,
        times=times,
        georef=grid,
        valid=root_valid,
        units=expression._inferred_units,
        signal_name=expression._signal_name,
    )


def _iter_temporal_layers(expression: TemporalRasterExpression):
    """Evaluate a temporal graph one time layer at a time.

    This is the bounded-time execution path used by temporal reductions.  It
    deliberately keeps source descriptors, rather than live series handles,
    in the expression graph and owns any series handles opened for execution.
    """
    from ..temporal_store import open_temporal_cube

    source_series: dict[str, Any] = {}
    for node in _topological_sort_temporal(expression):
        if node._operation_id != "temporal.source":
            continue
        path = str(node._params_dict.get("path", ""))
        if not path:
            raise MapAlgebraExpressionError(
                "Temporal source node has no path.",
                code="temporal_internal_error",
            )
        if path not in source_series:
            source_series[path] = open_temporal_cube(path)

    times = expression._inferred_times
    if times is None:
        raise MapAlgebraExpressionError(
            "Cannot evaluate temporal expression with no time coordinates.",
            code="temporal_no_times",
        )

    try:
        for time_index, time_value in enumerate(times):
            cache: dict[str, TemporalRasterExpression] = {}

            def clone(node: TemporalRasterExpression) -> TemporalRasterExpression:
                cached = cache.get(node._node_id)
                if cached is not None:
                    return cached

                if node._operation_id == "temporal.source":
                    path = str(node._params_dict["path"])
                    values, georef = source_series[path].read_layer(time_index)
                    nodata = georef.nodata
                    if nodata is None:
                        valid = np.ones(values.shape, dtype=np.bool_)
                    elif isinstance(nodata, float) and np.isnan(nodata):
                        valid = ~np.isnan(values)
                    else:
                        valid = values != np.asarray(nodata, dtype=values.dtype)
                    layer = TemporalRaster(
                        values=np.asarray(values)[None, :, :],
                        times=np.asarray([time_value]),
                        georef=georef,
                        valid=np.asarray(valid)[None, :, :],
                        units=node._inferred_units,
                        signal_name=node._signal_name,
                    )
                    result = _temporal_constant(layer)
                elif node._operation_id == "temporal.constant":
                    temporal = next(
                        op for op in node._operands if isinstance(op, TemporalRaster)
                    )
                    layer = TemporalRaster(
                        values=temporal.values[time_index : time_index + 1],
                        times=temporal.times[time_index : time_index + 1],
                        georef=temporal.georef,
                        valid=temporal.valid[time_index : time_index + 1],
                        units=temporal.units,
                        signal_name=temporal.signal_name,
                        name=temporal.name,
                    )
                    result = _temporal_constant(layer)
                elif node._operation_id == "temporal.broadcast":
                    result = _temporal_broadcast(
                        node._operands[0], times_hint=np.asarray([time_value]),
                    )
                else:
                    operands = tuple(
                        clone(op) if isinstance(op, TemporalRasterExpression) else op
                        for op in node._operands
                    )
                    result = _temporal_local_op(node._operation_id, *operands)
                cache[node._node_id] = result
                return result

            yield _compute_temporal(clone(expression)).layer(0)
    finally:
        for series in source_series.values():
            series.close()


def _get_op_result(op: Any, node_results: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(op, TemporalRasterExpression):
        return node_results[op._node_id]
    if isinstance(op, RasterExpression):
        raise MapAlgebraExpressionError(
            "Bare RasterExpression in temporal graph; should be wrapped in broadcast.",
            code="temporal_internal_error",
        )
    raise MapAlgebraExpressionError(
        f"Unexpected operand type in temporal graph: {type(op).__name__}",
        code="temporal_internal_error",
    )


def _get_op_valid(op: Any, node_valid: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(op, TemporalRasterExpression):
        return node_valid[op._node_id]
    raise MapAlgebraExpressionError(
        f"Unexpected operand type in temporal graph: {type(op).__name__}",
        code="temporal_internal_error",
    )
