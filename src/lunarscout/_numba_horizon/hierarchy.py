"""CPU diagnostic oracle for current production hierarchical traversal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .contract import PyramidArrays
from .kernel_math import (
    evaluate_planar_chord,
    evaluate_quartic,
    evaluate_tangent,
    is_valid_elevation,
    sample_bilinear,
)


BEAM_WIDTH_RAD = np.float32(2.0 * np.pi / 1440.0)


@dataclass(frozen=True, slots=True)
class HierarchyTrace:
    maximum_slope: np.float32
    values: npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TraversalCounters:
    """Trace-derived equivalents of the C# traversal profile counters."""

    iterations: int
    level0_samples: int
    culled_blocks: int
    out_of_bounds: int
    nodata_skips: int


def traversal_counters(trace: npt.ArrayLike) -> TraversalCounters:
    """Summarize a retained hierarchy trace using the C# action codes."""
    values = np.asarray(trace, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("hierarchy trace must have shape (steps, 12)")
    actions = values[:, 11]
    return TraversalCounters(
        iterations=int(np.count_nonzero(actions != 0.0)),
        level0_samples=int(np.count_nonzero(actions == 4.0)),
        culled_blocks=int(np.count_nonzero(actions == 1.0)),
        out_of_bounds=int(np.count_nonzero(actions == 3.0)),
        nodata_skips=int(np.count_nonzero(actions == 2.0)),
    )


def _start_level(true_distance_m: np.float32, map_resolution_m: np.float32, levels: int):
    footprint = true_distance_m * BEAM_WIDTH_RAD
    level = levels - 1
    while level > 0:
        side = np.float32(1 << (level * 2)) * map_resolution_m
        if side <= footprint:
            break
        level -= 1
    return level


def _cell(pyramid: PyramidArrays, level: int, x: int, y: int) -> np.float32:
    _, offset, width, _ = pyramid.levels[level]
    if level == 0:
        return pyramid.level0[y, x]
    return pyramid.mips[offset + y * width + x]


def _bilinear_bound(
    pyramid: PyramidArrays, level: int, x: int, y: int
) -> np.float32:
    """Bound bilinear samples whose lower-left corner is in this cell.

    Bilinear sampling reads the current, right, bottom, and bottom-right
    elevations.  At mip levels the same four-cell maximum is conservative:
    each value already bounds every elevation in its source block.
    """
    _, _, width, height = pyramid.levels[level]
    maximum = np.float32(-32000.0)
    for neighbor_y in range(y, min(y + 2, int(height))):
        for neighbor_x in range(x, min(x + 2, int(width))):
            value = _cell(pyramid, level, neighbor_x, neighbor_y)
            if np.isfinite(value) and value > -20000.0:
                maximum = max(maximum, value)
    return maximum


def traverse_hierarchy(
    segment: npt.ArrayLike,
    pyramid: PyramidArrays,
    *, observer_z_m: float,
    radius_m: float,
    map_resolution_m: float,
    pass_index: int,
) -> HierarchyTrace:
    """Run the current C# hierarchy algorithm and retain every branch record."""
    segment = np.asarray(segment, dtype=np.float32)
    map_resolution = np.float32(map_resolution_m)
    observer_z = np.float32(observer_z_m)
    radius = np.float32(radius_m)
    minimum_step = np.float32(0.5) * map_resolution / np.float32(1000.0)
    primary_far_step = np.float32(0.8) * map_resolution / np.float32(1000.0)
    current = np.float32(-1e30)
    s_start = segment[12]
    s = max(s_start, np.float32(0.001))
    trace = []
    level_count = len(pyramid.levels)
    width, height = pyramid.level0.shape[1], pyramid.level0.shape[0]
    while s <= segment[13]:
        delta = s - s_start
        pixel_x = evaluate_quartic(segment[0], segment[4:8], delta)
        pixel_y = evaluate_quartic(segment[1], segment[8:12], delta)
        if (
            not np.isfinite(pixel_x) or not np.isfinite(pixel_y)
            or pixel_x < 0 or pixel_y < 0
            or pixel_x >= width - 1 or pixel_y >= height - 1
        ):
            break
        planar_x = (pixel_x - segment[0]) * map_resolution
        planar_y = (pixel_y - segment[1]) * map_resolution
        planar_m = np.sqrt(planar_x * planar_x + planar_y * planar_y).astype(np.float32)
        if s < np.float32(0.5):
            true_m = s * np.float32(1000.0)
        else:
            true_m = segment[14] * np.float32(1000.0) + evaluate_planar_chord(
                segment, planar_m
            )
        level = _start_level(true_m, map_resolution, level_count)
        ray_out = False
        while level >= 0:
            _, _, level_width, level_height = pyramid.levels[level]
            shift = level * 2
            scale = 1 << shift
            cell_x = int(pixel_x) >> shift
            cell_y = int(pixel_y) >> shift
            row = np.array(
                (
                    s, true_m, level, cell_x, cell_y, pixel_x, pixel_y,
                    np.nan, np.nan, np.nan, 0.0, -1.0,
                ),
                dtype=np.float32,
            )
            if not (0 <= cell_x < level_width and 0 <= cell_y < level_height):
                advance = max(
                    np.float32(0.001),
                    np.float32(scale) * map_resolution / np.float32(1000.0),
                )
                row[10], row[11] = advance, 3.0
                trace.append(row)
                s += advance
                ray_out = True
                break
            maximum_height = _bilinear_bound(
                pyramid, level, cell_x, cell_y
            )
            row[7] = maximum_height
            minimum_x = np.float32(cell_x * scale)
            minimum_y = np.float32(cell_y * scale)
            maximum_x = minimum_x + np.float32(scale)
            maximum_y = minimum_y + np.float32(scale)
            dx, dy = evaluate_tangent(segment[4:8], segment[8:12], delta)
            inverse_x = np.float32(1.0) / dx if abs(dx) > 1e-8 else np.float32(1e30)
            inverse_y = np.float32(1.0) / dy if abs(dy) > 1e-8 else np.float32(1e30)
            t1 = (minimum_x - pixel_x) * inverse_x
            t2 = (maximum_x - pixel_x) * inverse_x
            t3 = (minimum_y - pixel_y) * inverse_y
            t4 = (maximum_y - pixel_y) * inverse_y
            exit_distance = min(max(t1, t2), max(t3, t4))
            fallback = np.float32(scale) * map_resolution * np.float32(0.0005)
            distance_to_exit = exit_distance if exit_distance > 0 else fallback
            if maximum_height < -20000.0:
                advance = (
                    distance_to_exit + np.float32(0.0001)
                    if distance_to_exit > 0 else fallback
                )
                row[10], row[11] = advance, 2.0
                trace.append(row)
                s += advance
                break
            block_m = np.float32(scale) * map_resolution
            true_near = max(true_m - block_m, np.float32(1.0))
            observer_radius = radius + observer_z
            if s < np.float32(0.5):
                possible = (maximum_height - observer_z) / true_near
            else:
                squared = true_near * true_near
                local_z = (
                    (maximum_height - observer_z)
                    * (np.float32(2.0) * radius + maximum_height + observer_z)
                    - squared
                ) / (np.float32(2.0) * observer_radius)
                local_x_squared = squared - local_z * local_z
                local_x = (
                    np.sqrt(local_x_squared).astype(np.float32)
                    if local_x_squared > 0 else np.float32(1e-6)
                )
                possible = local_z / local_x
            if possible <= current:
                advance = (
                    distance_to_exit + np.float32(0.0001)
                    if distance_to_exit > 0 else fallback
                )
                row[10], row[11] = advance, 1.0
                trace.append(row)
                s += advance
                break
            if level == 0:
                height_value = sample_bilinear(
                    pyramid.level0, float(pixel_x), float(pixel_y)
                )
                slope = np.float32(-1e30)
                if is_valid_elevation(float(height_value)):
                    if s < np.float32(0.5):
                        slope = (height_value - observer_z) / true_m
                    else:
                        squared = true_m * true_m
                        point_radius = radius + height_value
                        local_z = (
                            (point_radius - observer_radius)
                            * (point_radius + observer_radius) - squared
                        ) / (np.float32(2.0) * observer_radius)
                        local_x_squared = squared - local_z * local_z
                        local_x = (
                            np.sqrt(local_x_squared).astype(np.float32)
                            if local_x_squared > 0 else np.float32(1e-6)
                        )
                        slope = local_z / local_x
                    current = max(current, slope)
                row[8], row[9] = height_value, slope
                magnitude = np.sqrt(dx * dx + dy * dy).astype(np.float32)
                pixel_step = (
                    np.float32(1.0) / magnitude
                    if magnitude > 1e-6 else np.float32(0.0005)
                )
                margin = current - slope
                margin_step = (
                    margin * true_m * np.float32(1.732 / 1000.0)
                    if margin > 0 else np.float32(0.0)
                )
                angular_step = true_m * np.float32(0.00151 / 1000.0)
                advance = max(pixel_step, min(margin_step, angular_step))
                if s < np.float32(0.5):
                    advance *= np.float32(0.25)
                floor = (
                    primary_far_step
                    if pass_index == 0 and true_m >= 100.0 else minimum_step
                )
                advance = max(advance, floor)
                boundary_advance = (
                    distance_to_exit + np.float32(0.0001)
                    if distance_to_exit > 0 else fallback
                )
                advance = min(advance, boundary_advance)
                row[10], row[11] = advance, 4.0
                trace.append(row)
                s += advance
                break
            row[11] = 0.0
            trace.append(row)
            level -= 1
        if ray_out:
            break
    values = np.ascontiguousarray(trace, dtype=np.float32)
    if not len(values):
        values = np.empty((0, 12), dtype=np.float32)
    maximum = current if current > -1e29 else np.float32(-np.inf)
    return HierarchyTrace(maximum, values)
