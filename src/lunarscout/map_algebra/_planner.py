from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import MapAlgebraExpressionError
from ..georeference import GeoReference
from ._model import RasterExpression
from ._registry import get_operation_spec

_MAX_NODES = 10_000
_MAX_DEPTH = 500
_MAX_SOURCES = 1_000
_DEFAULT_WINDOW = 128


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    topo_order: tuple[RasterExpression, ...]
    sources: tuple[RasterExpression, ...]
    operations: tuple[RasterExpression, ...]
    grid: GeoReference | None
    output_dtype: np.dtype[Any] | None
    output_units: str | None
    window_width: int
    window_height: int
    n_windows_x: int
    n_windows_y: int
    total_windows: int
    estimated_per_window_bytes: int
    maximum_halo: int

    @property
    def n_passes(self) -> int:
        return 1

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def n_operations(self) -> int:
        return len(self.operations)

    @property
    def halos(self) -> int:
        return self.maximum_halo


def plan_expression(
    expression: RasterExpression,
    *,
    window_width: int = _DEFAULT_WINDOW,
    window_height: int = _DEFAULT_WINDOW,
) -> ExecutionPlan:
    if (
        not isinstance(window_width, int)
        or isinstance(window_width, bool)
        or not isinstance(window_height, int)
        or isinstance(window_height, bool)
        or window_width < 1
        or window_height < 1
    ):
        raise MapAlgebraExpressionError(
            "Window dimensions must be positive.",
            code="map_algebra_invalid_window",
            details={"window_width": window_width, "window_height": window_height},
        )

    topo = _topological_validate(expression)
    _enforce_limits(topo)
    _reject_unsupported(topo)

    sources: list[RasterExpression] = []
    operations: list[RasterExpression] = []
    for node in topo:
        if node._operation_id in ("source", "constant") or node._operation_id.startswith("coordinate."):
            sources.append(node)
        else:
            operations.append(node)

    grid = expression._inferred_grid
    if grid is None:
        raise MapAlgebraExpressionError(
            "Expression has no inferred output grid.",
            code="map_algebra_missing_output_grid",
        )

    nw = max(1, (grid.width + window_width - 1) // window_width)
    nh = max(1, (grid.height + window_height - 1) // window_height)

    maximum_halo = _maximum_source_halo(topo)
    per_window = _estimate_per_window(
        sources, operations, window_width, window_height, maximum_halo,
    )

    return ExecutionPlan(
        topo_order=tuple(topo),
        sources=tuple(sources),
        operations=tuple(operations),
        grid=grid,
        output_dtype=expression._inferred_dtype,
        output_units=expression._inferred_units,
        window_width=window_width,
        window_height=window_height,
        n_windows_x=nw,
        n_windows_y=nh,
        total_windows=nw * nh,
        estimated_per_window_bytes=per_window,
        maximum_halo=maximum_halo,
    )


def _topological_validate(root: RasterExpression) -> list[RasterExpression]:
    return root._all_nodes()


def _enforce_limits(topo: list[RasterExpression]) -> None:
    if len(topo) > _MAX_NODES:
        raise MapAlgebraExpressionError(
            f"Expression graph has {len(topo)} nodes, exceeding the limit of {_MAX_NODES}.",
            code="map_algebra_too_many_nodes",
            details={"node_count": len(topo), "max": _MAX_NODES},
        )

    depth = _max_depth(topo)
    if depth > _MAX_DEPTH:
        raise MapAlgebraExpressionError(
            f"Expression graph depth {depth} exceeds the limit of {_MAX_DEPTH}.",
            code="map_algebra_too_deep",
            details={"depth": depth, "max": _MAX_DEPTH},
        )

    source_count = sum(1 for n in topo if n._operation_id == "source")
    if source_count > _MAX_SOURCES:
        raise MapAlgebraExpressionError(
            f"Expression graph has {source_count} source nodes, exceeding the limit of {_MAX_SOURCES}.",
            code="map_algebra_too_many_sources",
            details={"source_count": source_count, "max": _MAX_SOURCES},
        )


def _max_depth(topo: list[RasterExpression]) -> int:
    depths: dict[str, int] = {}
    for node in topo:
        child_depths = [
            depths[operand._node_id]
            for operand in node._operands
            if isinstance(operand, RasterExpression)
        ]
        depths[node._node_id] = max(child_depths, default=0) + 1
    return max(depths.values(), default=0)


_UNSUPPORTED_CATEGORIES = frozenset({"focal", "global", "zonal", "distance", "temporal"})


def _reject_unsupported(topo: list[RasterExpression]) -> None:
    for node in topo:
        spec = get_operation_spec(node._operation_id)
        if (
            spec.category in _UNSUPPORTED_CATEGORIES
            or (node._operation_id != "constant" and not spec.file_backed_available)
        ):
            raise MapAlgebraExpressionError(
                f"Operation '{node._operation_id}' is not yet supported "
                f"in windowed (file-backed) execution mode.",
                code="map_algebra_unsupported_windowed_operation",
                details={
                    "operation_id": node._operation_id,
                    "category": spec.category,
                },
            )
        if node._halo and node._operation_id not in {
            "terrain.slope", "terrain.aspect", "terrain.hillshade",
        }:
            raise MapAlgebraExpressionError(
                f"Operation '{node._operation_id}' has no reviewed halo-aware window kernel.",
                code="map_algebra_unsupported_windowed_operation",
                details={
                    "operation_id": node._operation_id,
                    "category": spec.category,
                    "halo": node._halo,
                },
            )
        if node._operation_id == "local.normalize_minmax":
            params = node._params_dict
            if params.get("minimum") is None or params.get("maximum") is None:
                raise MapAlgebraExpressionError(
                    "Windowed normalize_minmax requires explicit minimum and maximum values.",
                    code="map_algebra_windowed_statistics_required",
                    details={"operation_id": node._operation_id},
                )
            minimum = float(params["minimum"])
            maximum = float(params["maximum"])
            if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum < minimum:
                raise MapAlgebraExpressionError(
                    "Windowed normalization statistics must be finite with maximum >= minimum.",
                    code="map_algebra_invalid_normalization_statistics",
                    details={"minimum": minimum, "maximum": maximum},
                )
        if node._operation_id == "local.standardize":
            params = node._params_dict
            if params.get("mean") is None or params.get("std") is None:
                raise MapAlgebraExpressionError(
                    "Windowed standardize requires explicit mean and standard deviation values.",
                    code="map_algebra_windowed_statistics_required",
                    details={"operation_id": node._operation_id},
                )
            mean = float(params["mean"])
            std = float(params["std"])
            ddof = float(params.get("ddof", 0))
            if (
                not np.isfinite(mean)
                or not np.isfinite(std)
                or std < 0
                or not np.isfinite(ddof)
                or ddof < 0
            ):
                raise MapAlgebraExpressionError(
                    "Windowed standardization statistics and ddof are invalid.",
                    code="map_algebra_invalid_normalization_statistics",
                    details={"mean": mean, "std": std, "ddof": ddof},
                )


def _estimate_per_window(
    sources: list[RasterExpression],
    operations: list[RasterExpression],
    window_width: int,
    window_height: int,
    maximum_halo: int,
) -> int:
    cells = (window_width + 2 * maximum_halo) * (window_height + 2 * maximum_halo)
    source_bytes = sum(
        cells * ((source._inferred_dtype or np.dtype(np.float64)).itemsize + 1)
        for source in sources
    )
    operation_bytes = sum(
        cells * ((operation._inferred_dtype or np.dtype(np.float64)).itemsize + 1)
        for operation in operations
    )
    # Sources exist once in the bounded source cache and once as Raster views in
    # the per-window graph. Three operation-sized buffers conservatively cover
    # the result plus ordinary NumPy kernel temporaries.
    return source_bytes * 2 + operation_bytes * 3 + cells * 2


def _maximum_source_halo(topo: list[RasterExpression]) -> int:
    """Return the largest cumulative same-grid halo requested from a source."""
    required: dict[str, int] = {node._node_id: 0 for node in topo}
    for node in reversed(topo):
        node_requirement = required[node._node_id]
        child_requirement = node_requirement + max(0, int(node._halo))
        for operand in node._operands:
            if not isinstance(operand, RasterExpression):
                continue
            if node._operation_id == "alignment.resample_to":
                # Resampling maps into another pixel coordinate system. Its
                # interpolation support is accounted for by the source-window
                # mapper rather than expressed as an output-grid halo.
                propagated = 0
            else:
                propagated = child_requirement
            required[operand._node_id] = max(required[operand._node_id], propagated)
    return max(required.values(), default=0)
