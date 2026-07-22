from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

from ..errors import (
    MapAlgebraExpressionError,
    OperationCancelledError,
)
from ..georeference import GeoReference
from ..raster import Raster
from ._model import RasterExpression
from ._planner import ExecutionPlan
from ._windows import SourceWindowCache, enumerate_windows


WriteBlock = Callable[
    [int, int, int, int, int, np.ndarray[Any, Any], np.ndarray[Any, Any]],
    None,
]

CancellationCheck = Callable[[], bool]
ProgressCallback = Callable[[int, int, int], None]


def execute_windowed(
    plan: ExecutionPlan,
    cache: SourceWindowCache,
    *,
    write_block: WriteBlock | None = None,
    completed_windows: set[int] | None = None,
    cancellation_requested: CancellationCheck | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Raster | None:
    if plan.grid is None:
        raise MapAlgebraExpressionError(
            "Plan has no output grid.",
            code="map_algebra_missing_output_grid",
        )

    grid = plan.grid
    completed = completed_windows if completed_windows is not None else set()
    total = plan.total_windows
    collecting = write_block is None
    if collecting and completed:
        raise MapAlgebraExpressionError(
            "Completed windows can only be skipped when a write callback is supplied.",
            code="map_algebra_invalid_completed_windows",
            details={"completed_windows": len(completed)},
        )
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
            if idx in completed:
                cache.discard_window(idx)
                continue

            if cancellation_requested is not None and cancellation_requested():
                raise OperationCancelledError(
                    f"Windowed execution cancelled before window "
                    f"{len(completed) + 1} "
                    f"of {total}.",
                    code="map_algebra_cancelled",
                    details={
                        "completed_windows": len(completed),
                        "total_windows": total,
                        "window_index": idx,
                    },
                )

            values, valid = _execute_window(
                plan, cache, idx, x0, y0, width, height,
            )
            if write_block is not None:
                write_block(idx, x0, y0, width, height, values, valid)
            else:
                assert full_values is not None and full_valid is not None
                full_values[y0 : y0 + height, x0 : x0 + width] = values
                full_valid[y0 : y0 + height, x0 : x0 + width] = valid
            completed.add(idx)
            cache.discard_window(idx)

            if progress_callback is not None:
                progress_callback(len(completed), total, idx)
    except OperationCancelledError:
        cache.close()
        raise
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
    assert plan.grid is not None
    memo: dict[tuple[str, int, int, int, int], Raster] = {}
    root = _execute_node_window(
        plan.topo_order[-1], cache, window_idx, x0, y0, width, height, memo,
    )
    return root.values, root.valid


def _execute_node_window(
    node: RasterExpression,
    cache: SourceWindowCache,
    window_idx: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
    memo: dict[tuple[str, int, int, int, int], Raster],
) -> Raster:
    from ._spatial import (
        evaluate_resample,
        evaluate_terrain,
        source_window_for_resampling,
    )
    from .expression import _eval_binary, _eval_special, _eval_unary

    key = (node._node_id, x0, y0, width, height)
    cached = memo.get(key)
    if cached is not None:
        return cached
    grid = node.grid
    if grid is None:
        raise MapAlgebraExpressionError(
            f"Operation '{node.operation_id}' has no inferred grid.",
            code="map_algebra_missing_output_grid",
            details={"operation_id": node.operation_id},
        )
    window_grid = _window_grid(grid, x0, y0, width, height)
    operation_id = node.operation_id

    if operation_id in {"source", "constant"} or operation_id.startswith("coordinate."):
        result = Raster(
            values=cache.read_values(node, window_idx, x0, y0, width, height),
            georef=window_grid,
            valid=cache.read_valid(node, window_idx, x0, y0, width, height),
            units=node.units,
        )
    elif operation_id in _TERRAIN_OPERATION_IDS:
        operand = _single_expression_operand(node)
        expanded_x0 = max(0, x0 - node.halo)
        expanded_y0 = max(0, y0 - node.halo)
        expanded_x1 = min(grid.width, x0 + width + node.halo)
        expanded_y1 = min(grid.height, y0 + height + node.halo)
        expanded = _execute_node_window(
            operand,
            cache,
            window_idx,
            expanded_x0,
            expanded_y0,
            expanded_x1 - expanded_x0,
            expanded_y1 - expanded_y0,
            memo,
        )
        terrain = evaluate_terrain(node, expanded)
        crop_x0 = x0 - expanded_x0
        crop_y0 = y0 - expanded_y0
        result = Raster(
            values=terrain.values[
                crop_y0 : crop_y0 + height,
                crop_x0 : crop_x0 + width,
            ],
            georef=window_grid,
            valid=terrain.valid[
                crop_y0 : crop_y0 + height,
                crop_x0 : crop_x0 + width,
            ],
            units=node.units,
            validity_provenance=terrain.validity_provenance,
        )
    elif operation_id == "alignment.resample_to":
        operand = _single_expression_operand(node)
        if operand.grid is None:
            raise MapAlgebraExpressionError(
                "Resampling source has no inferred grid.",
                code="map_algebra_missing_output_grid",
            )
        source_request = source_window_for_resampling(
            operand.grid,
            grid,
            x0=x0,
            y0=y0,
            width=width,
            height=height,
            resampling=str(node._params_dict["resampling"]),
        )
        if source_request is None:
            source = None
        else:
            source = _execute_node_window(
                operand, cache, window_idx, *source_request, memo,
            )
        result = evaluate_resample(node, source, window_grid)
    else:
        operands = [
            _execute_node_window(
                operand, cache, window_idx, x0, y0, width, height, memo,
            )
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
    memo[key] = result
    return result


def _single_expression_operand(node: RasterExpression) -> RasterExpression:
    if len(node._operands) != 1 or not isinstance(node._operands[0], RasterExpression):
        raise MapAlgebraExpressionError(
            f"Operation '{node.operation_id}' requires one raster expression operand.",
            code="map_algebra_expression_eval_failed",
            details={"operation_id": node.operation_id},
        )
    return node._operands[0]


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
    | frozenset({
        "terrain.slope", "terrain.aspect", "terrain.hillshade",
        "alignment.resample_to",
    })
)

_TERRAIN_OPERATION_IDS = frozenset({
    "terrain.slope", "terrain.aspect", "terrain.hillshade",
})
