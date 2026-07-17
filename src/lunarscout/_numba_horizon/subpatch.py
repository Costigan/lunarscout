"""CPU oracle for production subpatch selection and interpolation."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .contract import PyramidArrays
from .kernel_math import (
    clamp_subpatch_center,
    interpolate_segments,
    interpolation_selection,
)


def interpolate_pixel_segment(
    segment_values: npt.ArrayLike,
    primary_pyramid: PyramidArrays,
    active_pyramid: PyramidArrays,
    *,
    tile_column: int,
    tile_row: int,
    tile_width: int,
    subpatch_size: int,
    pixel_column: int,
    pixel_row: int,
    azimuth: int,
    pass_index: int,
) -> npt.NDArray[np.float32]:
    """Reproduce C# halo selection, center clamping, shifting, and lerp."""
    values = np.asarray(segment_values, dtype=np.float32)
    indices = interpolation_selection(
        pixel_column, pixel_row, tile_width, subpatch_size
    )
    corner_indices, tx, ty = indices[:4], indices[4], indices[5]
    count = tile_width // subpatch_size + 2
    left = corner_indices[0] % count
    top = corner_indices[0] // count
    requested_left = tile_column + (left - 1) * subpatch_size + subpatch_size // 2
    requested_right = requested_left + subpatch_size
    requested_top = tile_row + (top - 1) * subpatch_size + subpatch_size // 2
    requested_bottom = requested_top + subpatch_size
    center_left = clamp_subpatch_center(
        requested_left, primary_pyramid.level0.shape[1], subpatch_size
    ) - tile_column
    center_right = clamp_subpatch_center(
        requested_right, primary_pyramid.level0.shape[1], subpatch_size
    ) - tile_column
    center_top = clamp_subpatch_center(
        requested_top, primary_pyramid.level0.shape[0], subpatch_size
    ) - tile_row
    center_bottom = clamp_subpatch_center(
        requested_bottom, primary_pyramid.level0.shape[0], subpatch_size
    ) - tile_row
    primary_map = primary_pyramid.map_parameters
    active_map = active_pyramid.map_parameters
    primary_resolution = math.hypot(float(primary_map[6]), float(primary_map[9]))
    active_resolution = math.hypot(float(active_map[6]), float(active_map[9]))
    scale_ratio = np.float32(primary_resolution / active_resolution)
    shifts = np.array(
        (
            (pixel_column - center_left, pixel_row - center_top),
            (pixel_column - center_right, pixel_row - center_top),
            (pixel_column - center_left, pixel_row - center_bottom),
            (pixel_column - center_right, pixel_row - center_bottom),
        ),
        dtype=np.float32,
    )
    return interpolate_segments(
        values[azimuth, list(corner_indices), pass_index],
        shifts,
        scale_ratio,
        tx,
        ty,
    )
