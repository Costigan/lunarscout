from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

from ..errors import MapAlgebraExpressionError
from ..georeference import GeoReference
from ..raster import Raster
from ._model import RasterExpression
from ._planner import ExecutionPlan
from ._windows import SourceWindowCache, enumerate_windows


WriteBlock = Callable[
    [int, int, int, int, int, np.ndarray[Any, Any], np.ndarray[Any, Any]],
    None,
]


def execute_windowed(
    plan: ExecutionPlan,
    cache: SourceWindowCache,
    *,
    write_block: WriteBlock | None = None,
) -> Raster | None:
    """Execute a local/coordinate plan one bounded output window at a time."""
    if plan.grid is None:
        raise MapAlgebraExpressionError(
            "Plan has no output grid.",
            code="map_algebra_missing_output_grid",
        )

    grid = plan.grid
    collecting = write_block is None
    full_values: np.ndarray[Any, Any] | None = None
    full_valid: np.ndarray[Any, Any] | None = None
    if collecting:
        full_dtype = plan.output_dtype or np.dtype(np.float64)
        full_values = np.zeros((grid.height, grid.width), dtype=full_dtype)
        full_valid = np.zeros((grid.height, grid.width), dtype=np.bool_)

    try:
        for idx, x0, y0, width, height, _ in enumerate_windows(
            grid.width,
            grid.height,
            plan.window_width,
            plan.window_height,
        ):
            values, valid = _execute_window(
                plan, cache, idx, x0, y0, width, height,
            )
            if write_block is not None:
                write_block(idx, x0, y0, width, height, values, valid)
            else:
                assert full_values is not None and full_valid is not None
                full_values[y0 : y0 + height, x0 : x0 + width] = values
                full_valid[y0 : y0 + height, x0 : x0 + width] = valid
            cache.discard_window(idx)
    except Exception:
        cache.close()
        raise

    if not collecting:
        return None
    assert full_values is not None and full_valid is not None
    return Raster(
        values=full_values,
        georef=grid,
        valid=full_valid,
        units=plan.output_units,
    )


def _window_grid(
    grid: GeoReference,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> GeoReference:
    origin_x, pixel_x, rotation_x, origin_y, rotation_y, pixel_y = grid.affine_transform
    affine = (
        origin_x + x0 * pixel_x + y0 * rotation_x,
        pixel_x,
        rotation_x,
        origin_y + x0 * rotation_y + y0 * pixel_y,
        rotation_y,
        pixel_y,
    )
    return replace(
        grid,
        affine_transform=affine,
        width=width,
        height=height,
        nodata=None,
    )


def _execute_window(
    plan: ExecutionPlan,
    cache: SourceWindowCache,
    window_idx: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    from .expression import _eval_binary, _eval_special, _eval_unary

    assert plan.grid is not None
    window_grid = _window_grid(plan.grid, x0, y0, width, height)
    results: dict[str, Raster] = {}

    for node in plan.topo_order:
        operation_id = node._operation_id
        if operation_id in {"source", "constant"} or operation_id.startswith("coordinate."):
            results[node._node_id] = Raster(
                values=cache.read_values(node, window_idx, x0, y0, width, height),
                georef=window_grid,
                valid=cache.read_valid(node, window_idx, x0, y0, width, height),
                units=node._inferred_units,
            )
            continue

        operands = [
            results[operand._node_id]
            if isinstance(operand, RasterExpression)
            else operand
            for operand in node._operands
        ]
        if operation_id in _BINARY_OPERATION_IDS:
            result = _eval_binary(node, operands)
        elif operation_id in _UNARY_OPERATION_IDS:
            result = _eval_unary(node, operands)
        elif operation_id in _SPECIAL_OPERATION_IDS:
            result = _eval_special(node, operands)
        else:
            raise MapAlgebraExpressionError(
                f"Operation '{operation_id}' has no windowed kernel.",
                code="map_algebra_unsupported_windowed_operation",
                details={"operation_id": operation_id},
            )
        results[node._node_id] = result

    root = results[plan.topo_order[-1]._node_id]
    return root.values, root.valid


_BINARY_OPERATION_IDS = frozenset({
    "local.add", "local.subtract", "local.multiply", "local.divide",
    "local.floor_divide", "local.remainder", "local.power",
    "local.minimum", "local.maximum", "local.less", "local.less_equal",
    "local.greater", "local.greater_equal", "local.equal", "local.not_equal",
    "local.logical_and", "local.logical_or", "local.logical_xor",
    "local.hypot", "local.arctan2",
})

_UNARY_OPERATION_IDS = frozenset({
    "local.negative", "local.absolute", "local.sqrt", "local.square",
    "local.exp", "local.log", "local.log10", "local.sin", "local.cos",
    "local.tan", "local.arcsin", "local.arccos", "local.arctan",
    "local.logical_not", "local.floor", "local.ceil", "local.trunc",
    "local.round", "local.degrees", "local.radians",
})

_SPECIAL_OPERATION_IDS = frozenset({
    "local.where", "local.coalesce", "local.clip", "local.cast",
    "local.set_invalid", "local.fill_invalid", "local.is_valid",
    "local.is_invalid", "local.reclassify_values", "local.reclassify_ranges",
    "local.digitize", "local.one_hot", "local.normalize_minmax",
    "local.standardize", "local.isclose",
})

WINDOWED_OPERATION_IDS = (
    _BINARY_OPERATION_IDS | _UNARY_OPERATION_IDS | _SPECIAL_OPERATION_IDS
)
