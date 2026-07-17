"""Lazy, diagnostic Numba CUDA mechanics for Phase 4A.

This module deliberately contains no module-level Numba import. Constructing a
``CudaSession`` is the explicit boundary that imports Numba CUDA and selects a
device.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class CudaBackendError(RuntimeError):
    """Raised when the explicitly requested prototype CUDA backend is unavailable."""


_KERNELS = None


def _build_kernels(cuda):
    from numba import float32, int32

    @cuda.jit(device=True)
    def valid_elevation(value):
        return not math.isnan(value) and not math.isinf(value) and value > -20000.0

    @cuda.jit(device=True)
    def hierarchy_bilinear_bound(
        level0, mips, offset, width, height, level, cell_x, cell_y
    ):
        maximum = float32(-32000.0)
        for neighbor_y in range(cell_y, min(cell_y + 2, height)):
            for neighbor_x in range(cell_x, min(cell_x + 2, width)):
                if level == 0:
                    value = level0[neighbor_y, neighbor_x]
                else:
                    value = mips[offset + neighbor_y * width + neighbor_x]
                if valid_elevation(value) and value > maximum:
                    maximum = value
        return maximum

    @cuda.jit(device=True)
    def evaluate(x0, a1, a2, a3, a4, distance):
        distance = float32(distance)
        distance2 = float32(distance * distance)
        distance3 = float32(distance2 * distance)
        distance4 = float32(distance2 * distance2)
        result = float32(x0 + float32(a1 * distance))
        result = float32(result + float32(a2 * distance2))
        result = float32(result + float32(a3 * distance3))
        return float32(result + float32(a4 * distance4))

    @cuda.jit(device=True)
    def tangent(a1, a2, a3, a4, distance):
        distance = float32(distance)
        distance2 = float32(distance * distance)
        distance3 = float32(distance2 * distance)
        result = float32(a1 + float32(float32(2.0) * a2 * distance))
        result = float32(
            result + float32(float32(3.0) * a3 * distance2)
        )
        return float32(
            result + float32(float32(4.0) * a4 * distance3)
        )

    @cuda.jit(device=True)
    def planar_chord(segment, distance):
        distance = float32(distance)
        distance2 = float32(distance * distance)
        distance3 = float32(distance2 * distance)
        result = float32(segment[15] * distance)
        result = float32(result + float32(segment[16] * distance2))
        return float32(result + float32(segment[17] * distance3))

    @cuda.jit(device=True)
    def clamp_center(requested, dem_size, subpatch_size):
        half = subpatch_size // 2
        maximum = dem_size - half
        if maximum < half:
            maximum = half
        if requested < half:
            return half
        if requested > maximum:
            return maximum
        return requested

    @cuda.jit(device=True)
    def bilinear(elevation, column, row):
        width = elevation.shape[1]
        height = elevation.shape[0]
        column = float32(min(max(column, float32(0.0)), float32(width) - float32(1.0001)))
        row = float32(min(max(row, float32(0.0)), float32(height) - float32(1.0001)))
        x0 = int(math.floor(column))
        y0 = int(math.floor(row))
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)
        q00 = elevation[y0, x0]
        q10 = elevation[y0, x1]
        q01 = elevation[y1, x0]
        q11 = elevation[y1, x1]
        if not (
            valid_elevation(q00) and valid_elevation(q10)
            and valid_elevation(q01) and valid_elevation(q11)
        ):
            return float32(-32000.0)
        tx = float32(column - x0)
        ty = float32(row - y0)
        top = float32(q00 + tx * float32(q10 - q00))
        bottom = float32(q01 + tx * float32(q11 - q01))
        return float32(top + ty * (bottom - top))

    @cuda.jit
    def mapping_kernel(output, pixel_count, azimuth_count):
        pixel, azimuth = cuda.grid(2)
        if pixel >= pixel_count or azimuth >= azimuth_count:
            return
        output[pixel, azimuth] = pixel * azimuth_count + azimuth

    @cuda.jit
    def helper_kernel(
        segments, shifts, scales, weights, distances, planar_distances,
        elevation, sample_coordinates, requested_centers, dem_sizes,
        tile_widths, subpatch_sizes, pixel_coordinates, output,
    ):
        index = cuda.grid(1)
        if index >= output.shape[0]:
            return
        interpolated = cuda.local.array(18, dtype=float32)
        for field in range(18):
            values = cuda.local.array(4, dtype=float32)
            for corner in range(4):
                value = segments[index, corner, field]
                if field == 0 or field == 2:
                    value += shifts[index, corner, 0] * scales[index]
                elif field == 1 or field == 3:
                    value += shifts[index, corner, 1] * scales[index]
                values[corner] = value
            top = values[0] + (values[1] - values[0]) * weights[index, 0]
            bottom = values[2] + (values[3] - values[2]) * weights[index, 0]
            interpolated[field] = top + (bottom - top) * weights[index, 1]
        distance = distances[index]
        output[index, 0] = evaluate(
            interpolated[2], interpolated[4], interpolated[5],
            interpolated[6], interpolated[7], distance,
        )
        output[index, 1] = evaluate(
            interpolated[3], interpolated[8], interpolated[9],
            interpolated[10], interpolated[11], distance,
        )
        output[index, 2] = tangent(
            interpolated[4], interpolated[5], interpolated[6],
            interpolated[7], distance,
        )
        output[index, 3] = tangent(
            interpolated[8], interpolated[9], interpolated[10],
            interpolated[11], distance,
        )
        output[index, 4] = planar_chord(interpolated, planar_distances[index])
        sampled = bilinear(
            elevation, sample_coordinates[index, 0], sample_coordinates[index, 1]
        )
        output[index, 5] = sampled
        output[index, 6] = 1.0 if valid_elevation(sampled) else 0.0
        output[index, 7] = clamp_center(
            requested_centers[index], dem_sizes[index], subpatch_sizes[index]
        )

        count = tile_widths[index] // subpatch_sizes[index] + 2
        gx = (
            (pixel_coordinates[index, 0] - subpatch_sizes[index] / 2.0)
            / subpatch_sizes[index] + 1.0
        )
        gy = (
            (pixel_coordinates[index, 1] - subpatch_sizes[index] / 2.0)
            / subpatch_sizes[index] + 1.0
        )
        left = int(gx)
        top = int(gy)
        tx = gx - left
        ty = gy - top
        if left < 0:
            left = 0
            tx = 0.0
        if top < 0:
            top = 0
            ty = 0.0
        if left > count - 2:
            left = count - 2
            tx = 1.0
        if top > count - 2:
            top = count - 2
            ty = 1.0
        output[index, 8] = top * count + left
        output[index, 9] = top * count + left + 1
        output[index, 10] = (top + 1) * count + left
        output[index, 11] = (top + 1) * count + left + 1
        output[index, 12] = tx
        output[index, 13] = ty

    @cuda.jit
    def fixed_step_kernel(
        segments, elevation, observer_z, radius, map_resolution, step_km,
        maximum_slopes, traces, trace_counts,
    ):
        index = cuda.grid(1)
        if index >= segments.shape[0]:
            return
        segment = segments[index]
        s_start = segment[12]
        s = max(s_start, 0.001) + step_km
        current = -1e30
        count = 0
        while s <= segment[13] and count < traces.shape[1]:
            delta = s - s_start
            pixel_x = evaluate(
                segment[0], segment[4], segment[5], segment[6], segment[7], delta
            )
            pixel_y = evaluate(
                segment[1], segment[8], segment[9], segment[10], segment[11], delta
            )
            if (
                math.isnan(pixel_x) or math.isnan(pixel_y)
                or pixel_x < 0.0 or pixel_y < 0.0
                or pixel_x >= elevation.shape[1] - 1.0
                or pixel_y >= elevation.shape[0] - 1.0
            ):
                break
            planar_x = (pixel_x - segment[0]) * map_resolution
            planar_y = (pixel_y - segment[1]) * map_resolution
            planar_m = math.sqrt(planar_x * planar_x + planar_y * planar_y)
            if s < 0.5:
                true_m = s * 1000.0
            else:
                true_m = segment[14] * 1000.0 + planar_chord(segment, planar_m)
            height = bilinear(elevation, pixel_x, pixel_y)
            slope = -1e30
            if valid_elevation(height):
                if s < 0.5:
                    slope = (
                        (height - observer_z[index]) / true_m
                        if true_m > 1e-6 else -1e30
                    )
                else:
                    observer_radius = radius[index] + observer_z[index]
                    distance_squared = true_m * true_m
                    local_z = (
                        (height - observer_z[index])
                        * (2.0 * radius[index] + height + observer_z[index])
                        - distance_squared
                    ) / (2.0 * observer_radius)
                    local_x_squared = distance_squared - local_z * local_z
                    local_x = math.sqrt(local_x_squared) if local_x_squared > 0 else 1e-6
                    slope = local_z / local_x if local_x != 0 else -1e30
                if slope > current:
                    current = slope
            traces[index, count, 0] = s
            traces[index, count, 1] = true_m
            traces[index, count, 2] = pixel_x
            traces[index, count, 3] = pixel_y
            traces[index, count, 4] = height
            traces[index, count, 5] = slope
            traces[index, count, 6] = current
            count += 1
            s += step_km
        trace_counts[index] = count
        maximum_slopes[index] = current if current > -1e29 else -np.inf

    @cuda.jit
    def adaptive_kernel(
        segments, elevation, observer_z, radius, map_resolution, pass_index,
        maximum_slopes, traces, trace_counts,
    ):
        index = cuda.grid(1)
        if index >= segments.shape[0]:
            return
        segment = segments[index]
        s_start = segment[12]
        s = float32(max(s_start, float32(0.001)))
        current = float32(-1e30)
        minimum_step = float32(
            float32(float32(0.5) * map_resolution) * float32(0.001)
        )
        primary_far_step = float32(
            float32(float32(0.8) * map_resolution) * float32(0.001)
        )
        count = 0
        while s <= segment[13] and count < traces.shape[1]:
            delta = s - s_start
            pixel_x = evaluate(
                segment[0], segment[4], segment[5], segment[6], segment[7], delta
            )
            pixel_y = evaluate(
                segment[1], segment[8], segment[9], segment[10], segment[11], delta
            )
            if (
                math.isnan(pixel_x) or math.isnan(pixel_y)
                or pixel_x < 0.0 or pixel_y < 0.0
                or pixel_x >= elevation.shape[1] - 1.0
                or pixel_y >= elevation.shape[0] - 1.0
            ):
                break
            planar_x = (pixel_x - segment[0]) * map_resolution
            planar_y = (pixel_y - segment[1]) * map_resolution
            planar_m = math.sqrt(planar_x * planar_x + planar_y * planar_y)
            if s < 0.5:
                true_m = s * 1000.0
            else:
                true_m = segment[14] * 1000.0 + planar_chord(segment, planar_m)
            height = bilinear(elevation, pixel_x, pixel_y)
            slope = -1e30
            if valid_elevation(height):
                if s < 0.5:
                    slope = (
                        (height - observer_z[index]) / true_m
                        if true_m > 1e-6 else -1e30
                    )
                else:
                    observer_radius = radius[index] + observer_z[index]
                    distance_squared = true_m * true_m
                    local_z = (
                        (height - observer_z[index])
                        * (2.0 * radius[index] + height + observer_z[index])
                        - distance_squared
                    ) / (2.0 * observer_radius)
                    local_x_squared = distance_squared - local_z * local_z
                    local_x = math.sqrt(local_x_squared) if local_x_squared > 0 else 1e-6
                    slope = local_z / local_x if local_x != 0 else -1e30
                if slope > current:
                    current = slope
            dx = tangent(segment[4], segment[5], segment[6], segment[7], delta)
            dy = tangent(segment[8], segment[9], segment[10], segment[11], delta)
            magnitude = math.sqrt(dx * dx + dy * dy)
            pixel_step = 1.0 / magnitude if magnitude > 1e-6 else 0.001
            margin = current - slope
            margin_step = margin * true_m * 1.732 / 1000.0 if margin > 0 else 0.0
            angular_step = true_m * 0.00151 / 1000.0
            advance = max(pixel_step, min(margin_step, angular_step))
            if s < 0.5:
                advance *= 0.25
            floor = (
                primary_far_step
                if pass_index == 0 and true_m >= 100.0 else minimum_step
            )
            advance = max(advance, floor)
            traces[index, count, 0] = s
            traces[index, count, 1] = true_m
            traces[index, count, 2] = pixel_x
            traces[index, count, 3] = pixel_y
            traces[index, count, 4] = height
            traces[index, count, 5] = slope
            traces[index, count, 6] = current
            traces[index, count, 7] = advance
            count += 1
            s += advance
        trace_counts[index] = count
        maximum_slopes[index] = current if current > -1e29 else -np.inf

    @cuda.jit
    def hierarchy_kernel(
        segments, level0, mips, levels, observer_z, radius, map_resolution,
        pass_index, maximum_slopes, traces, trace_counts,
    ):
        index = cuda.grid(1)
        if index >= segments.shape[0]:
            return
        segment = segments[index]
        s_start = segment[12]
        s = float32(max(s_start, float32(0.001)))
        current = float32(-1e30)
        minimum_step = float32(
            float32(float32(0.5) * map_resolution) * float32(0.001)
        )
        primary_far_step = float32(
            float32(float32(0.8) * map_resolution) * float32(0.001)
        )
        count = 0
        ray_out = False
        while s <= segment[13] and count < traces.shape[1]:
            delta = s - s_start
            pixel_x = evaluate(
                segment[0], segment[4], segment[5], segment[6], segment[7], delta
            )
            pixel_y = evaluate(
                segment[1], segment[8], segment[9], segment[10], segment[11], delta
            )
            if (
                math.isnan(pixel_x) or math.isnan(pixel_y)
                or pixel_x < 0.0 or pixel_y < 0.0
                or pixel_x >= level0.shape[1] - 1.0
                or pixel_y >= level0.shape[0] - 1.0
            ):
                break
            planar_x = (pixel_x - segment[0]) * map_resolution
            planar_y = (pixel_y - segment[1]) * map_resolution
            planar_m = math.sqrt(planar_x * planar_x + planar_y * planar_y)
            if s < 0.5:
                true_m = float32(s * float32(1000.0))
            else:
                true_m = float32(
                    segment[14] * float32(1000.0)
                    + planar_chord(segment, planar_m)
                )
            footprint = float32(
                true_m * float32(2.0 * math.pi / 1440.0)
            )
            level = levels.shape[0] - 1
            while level > 0:
                side = (1 << (level * 2)) * map_resolution
                if side <= footprint:
                    break
                level -= 1
            while level >= 0 and count < traces.shape[1]:
                offset = levels[level, 1]
                level_width = levels[level, 2]
                level_height = levels[level, 3]
                shift = level * 2
                scale = 1 << shift
                cell_x = int(pixel_x) >> shift
                cell_y = int(pixel_y) >> shift
                traces[index, count, 0] = s
                traces[index, count, 1] = true_m
                traces[index, count, 2] = level
                traces[index, count, 3] = cell_x
                traces[index, count, 4] = cell_y
                traces[index, count, 5] = pixel_x
                traces[index, count, 6] = pixel_y
                traces[index, count, 7] = np.nan
                traces[index, count, 8] = np.nan
                traces[index, count, 9] = np.nan
                traces[index, count, 10] = 0.0
                traces[index, count, 11] = -1.0
                if not (
                    0 <= cell_x < level_width and 0 <= cell_y < level_height
                ):
                    step_km = float32(
                        float32(float32(scale) * map_resolution)
                        * float32(0.001)
                    )
                    advance = float32(max(float32(0.001), step_km))
                    traces[index, count, 10] = advance
                    traces[index, count, 11] = 3.0
                    count += 1
                    s = float32(s + advance)
                    ray_out = True
                    break
                maximum_height = hierarchy_bilinear_bound(
                    level0,
                    mips,
                    offset,
                    level_width,
                    level_height,
                    level,
                    cell_x,
                    cell_y,
                )
                traces[index, count, 7] = maximum_height
                minimum_x = float32(cell_x * scale)
                minimum_y = float32(cell_y * scale)
                maximum_x = float32(minimum_x + float32(scale))
                maximum_y = float32(minimum_y + float32(scale))
                dx = tangent(segment[4], segment[5], segment[6], segment[7], delta)
                dy = tangent(segment[8], segment[9], segment[10], segment[11], delta)
                inverse_x = (
                    float32(float32(1.0) / dx)
                    if abs(dx) > float32(1e-8) else float32(1e30)
                )
                inverse_y = (
                    float32(float32(1.0) / dy)
                    if abs(dy) > float32(1e-8) else float32(1e30)
                )
                t1 = float32(float32(minimum_x - pixel_x) * inverse_x)
                t2 = float32(float32(maximum_x - pixel_x) * inverse_x)
                t3 = float32(float32(minimum_y - pixel_y) * inverse_y)
                t4 = float32(float32(maximum_y - pixel_y) * inverse_y)
                exit_distance = float32(min(max(t1, t2), max(t3, t4)))
                fallback = float32(
                    float32(
                        float32(float32(scale) * map_resolution) * float32(0.5)
                    ) * float32(0.001)
                )
                distance_to_exit = float32(
                    exit_distance if exit_distance > 0 else fallback
                )
                if maximum_height < -20000.0:
                    advance = float32(
                        distance_to_exit + float32(0.0001)
                        if distance_to_exit > 0 else fallback
                    )
                    traces[index, count, 10] = advance
                    traces[index, count, 11] = 2.0
                    count += 1
                    s = float32(s + advance)
                    break
                block_m = float32(float32(scale) * map_resolution)
                true_near = float32(max(
                    float32(true_m - block_m), float32(1.0)
                ))
                observer_radius = float32(radius[index] + observer_z[index])
                if s < 0.5:
                    possible = float32(
                        float32(maximum_height - observer_z[index]) / true_near
                    )
                else:
                    squared = float32(true_near * true_near)
                    height_delta = float32(maximum_height - observer_z[index])
                    radius_sum = float32(
                        float32(float32(2.0) * radius[index])
                        + maximum_height
                    )
                    radius_sum = float32(radius_sum + observer_z[index])
                    numerator = float32(
                        float32(height_delta * radius_sum) - squared
                    )
                    denominator = float32(float32(2.0) * observer_radius)
                    local_z = float32(numerator / denominator)
                    local_x_squared = float32(
                        squared - float32(local_z * local_z)
                    )
                    local_x = (
                        float32(math.sqrt(local_x_squared))
                        if local_x_squared > 0 else float32(1e-6)
                    )
                    possible = float32(local_z / local_x)
                if possible <= current:
                    advance = float32(
                        distance_to_exit + float32(0.0001)
                        if distance_to_exit > 0 else fallback
                    )
                    traces[index, count, 10] = advance
                    traces[index, count, 11] = 1.0
                    count += 1
                    s = float32(s + advance)
                    break
                if level == 0:
                    height = bilinear(level0, pixel_x, pixel_y)
                    slope = -1e30
                    if valid_elevation(height):
                        if s < 0.5:
                            slope = float32(
                                float32(height - observer_z[index]) / true_m
                            )
                        else:
                            squared = float32(true_m * true_m)
                            point_radius = float32(radius[index] + height)
                            radius_delta = float32(point_radius - observer_radius)
                            radius_sum = float32(point_radius + observer_radius)
                            numerator = float32(
                                float32(radius_delta * radius_sum) - squared
                            )
                            denominator = float32(
                                float32(2.0) * observer_radius
                            )
                            local_z = float32(numerator / denominator)
                            local_x_squared = float32(
                                squared - float32(local_z * local_z)
                            )
                            local_x = (
                                float32(math.sqrt(local_x_squared))
                                if local_x_squared > 0 else float32(1e-6)
                            )
                            slope = float32(local_z / local_x)
                        if slope > current:
                            current = slope
                    traces[index, count, 8] = height
                    traces[index, count, 9] = slope
                    magnitude = math.sqrt(dx * dx + dy * dy)
                    pixel_step = 1.0 / magnitude if magnitude > 1e-6 else 0.0005
                    margin = current - slope
                    margin_step = (
                        margin * true_m * 1.732 / 1000.0 if margin > 0 else 0.0
                    )
                    angular_step = true_m * 0.00151 / 1000.0
                    advance = max(pixel_step, min(margin_step, angular_step))
                    if s < 0.5:
                        advance *= 0.25
                    floor = (
                        primary_far_step
                        if pass_index == 0 and true_m >= 100.0 else minimum_step
                    )
                    advance = float32(max(advance, floor))
                    boundary_advance = float32(
                        distance_to_exit + float32(0.0001)
                        if distance_to_exit > 0 else fallback
                    )
                    advance = float32(min(advance, boundary_advance))
                    traces[index, count, 10] = advance
                    traces[index, count, 11] = 4.0
                    count += 1
                    s = float32(s + advance)
                    break
                traces[index, count, 11] = 0.0
                count += 1
                level -= 1
            if ray_out:
                break
        trace_counts[index] = count
        maximum_slopes[index] = current if current > -1e29 else -np.inf

    @cuda.jit(device=True)
    def hierarchy_maximum(
        segment, level0, mips, levels, observer_z, radius, map_resolution,
        pass_index, initial,
    ):
        s_start = segment[12]
        s = float32(max(s_start, float32(0.001)))
        current = float32(-1e30) if math.isinf(initial) else float32(initial)
        minimum_step = float32(
            float32(float32(0.5) * map_resolution) * float32(0.001)
        )
        primary_far_step = float32(
            float32(float32(0.8) * map_resolution) * float32(0.001)
        )
        ray_out = False
        while s <= segment[13]:
            delta = s - s_start
            pixel_x = evaluate(
                segment[0], segment[4], segment[5], segment[6], segment[7], delta
            )
            pixel_y = evaluate(
                segment[1], segment[8], segment[9], segment[10], segment[11], delta
            )
            if (
                math.isnan(pixel_x) or math.isnan(pixel_y)
                or pixel_x < 0.0 or pixel_y < 0.0
                or pixel_x >= level0.shape[1] - 1.0
                or pixel_y >= level0.shape[0] - 1.0
            ):
                break
            planar_x = (pixel_x - segment[0]) * map_resolution
            planar_y = (pixel_y - segment[1]) * map_resolution
            planar_m = math.sqrt(planar_x * planar_x + planar_y * planar_y)
            if s < 0.5:
                true_m = float32(s * float32(1000.0))
            else:
                true_m = float32(
                    segment[14] * float32(1000.0)
                    + planar_chord(segment, planar_m)
                )
            footprint = float32(
                true_m * float32(2.0 * math.pi / 1440.0)
            )
            level = levels.shape[0] - 1
            while level > 0:
                side = (1 << (level * 2)) * map_resolution
                if side <= footprint:
                    break
                level -= 1
            while level >= 0:
                offset = levels[level, 1]
                level_width = levels[level, 2]
                level_height = levels[level, 3]
                shift = level * 2
                scale = 1 << shift
                cell_x = int(pixel_x) >> shift
                cell_y = int(pixel_y) >> shift
                if not (0 <= cell_x < level_width and 0 <= cell_y < level_height):
                    step_km = float32(
                        float32(float32(scale) * map_resolution)
                        * float32(0.001)
                    )
                    advance = float32(max(float32(0.001), step_km))
                    s = float32(s + advance)
                    ray_out = True
                    break
                maximum_height = hierarchy_bilinear_bound(
                    level0,
                    mips,
                    offset,
                    level_width,
                    level_height,
                    level,
                    cell_x,
                    cell_y,
                )
                minimum_x = float32(cell_x * scale)
                minimum_y = float32(cell_y * scale)
                maximum_x = float32(minimum_x + float32(scale))
                maximum_y = float32(minimum_y + float32(scale))
                dx = tangent(segment[4], segment[5], segment[6], segment[7], delta)
                dy = tangent(segment[8], segment[9], segment[10], segment[11], delta)
                inverse_x = (
                    float32(float32(1.0) / dx)
                    if abs(dx) > float32(1e-8) else float32(1e30)
                )
                inverse_y = (
                    float32(float32(1.0) / dy)
                    if abs(dy) > float32(1e-8) else float32(1e30)
                )
                t1 = float32(float32(minimum_x - pixel_x) * inverse_x)
                t2 = float32(float32(maximum_x - pixel_x) * inverse_x)
                t3 = float32(float32(minimum_y - pixel_y) * inverse_y)
                t4 = float32(float32(maximum_y - pixel_y) * inverse_y)
                exit_distance = float32(min(max(t1, t2), max(t3, t4)))
                fallback = float32(
                    float32(
                        float32(float32(scale) * map_resolution) * float32(0.5)
                    ) * float32(0.001)
                )
                distance_to_exit = float32(
                    exit_distance if exit_distance > 0 else fallback
                )
                if maximum_height < -20000.0:
                    advance = float32(
                        distance_to_exit + float32(0.0001)
                        if distance_to_exit > 0 else fallback
                    )
                    s = float32(s + advance)
                    break
                block_m = float32(float32(scale) * map_resolution)
                true_near = float32(max(
                    float32(true_m - block_m), float32(1.0)
                ))
                observer_radius = float32(radius + observer_z)
                if s < 0.5:
                    possible = float32(
                        float32(maximum_height - observer_z) / true_near
                    )
                else:
                    squared = float32(true_near * true_near)
                    height_delta = float32(maximum_height - observer_z)
                    radius_sum = float32(
                        float32(float32(2.0) * radius) + maximum_height
                    )
                    radius_sum = float32(radius_sum + observer_z)
                    numerator = float32(
                        float32(height_delta * radius_sum) - squared
                    )
                    denominator = float32(float32(2.0) * observer_radius)
                    local_z = float32(numerator / denominator)
                    local_x_squared = float32(
                        squared - float32(local_z * local_z)
                    )
                    local_x = (
                        float32(math.sqrt(local_x_squared))
                        if local_x_squared > 0 else float32(1e-6)
                    )
                    possible = float32(local_z / local_x)
                if possible <= current:
                    advance = float32(
                        distance_to_exit + float32(0.0001)
                        if distance_to_exit > 0 else fallback
                    )
                    s = float32(s + advance)
                    break
                if level == 0:
                    height = bilinear(level0, pixel_x, pixel_y)
                    slope = -1e30
                    if valid_elevation(height):
                        if s < 0.5:
                            slope = float32(
                                float32(height - observer_z) / true_m
                            )
                        else:
                            squared = float32(true_m * true_m)
                            point_radius = float32(radius + height)
                            radius_delta = float32(point_radius - observer_radius)
                            radius_sum = float32(point_radius + observer_radius)
                            numerator = float32(
                                float32(radius_delta * radius_sum) - squared
                            )
                            denominator = float32(
                                float32(2.0) * observer_radius
                            )
                            local_z = float32(numerator / denominator)
                            local_x_squared = float32(
                                squared - float32(local_z * local_z)
                            )
                            local_x = (
                                float32(math.sqrt(local_x_squared))
                                if local_x_squared > 0 else float32(1e-6)
                            )
                            slope = float32(local_z / local_x)
                        if slope > current:
                            current = slope
                    magnitude = math.sqrt(dx * dx + dy * dy)
                    pixel_step = 1.0 / magnitude if magnitude > 1e-6 else 0.0005
                    margin = current - slope
                    margin_step = (
                        margin * true_m * 1.732 / 1000.0 if margin > 0 else 0.0
                    )
                    angular_step = true_m * 0.00151 / 1000.0
                    advance = max(pixel_step, min(margin_step, angular_step))
                    if s < 0.5:
                        advance *= 0.25
                    floor = (
                        primary_far_step
                        if pass_index == 0 and true_m >= 100.0 else minimum_step
                    )
                    advance = float32(max(advance, floor))
                    boundary_advance = float32(
                        distance_to_exit + float32(0.0001)
                        if distance_to_exit > 0 else fallback
                    )
                    advance = float32(min(advance, boundary_advance))
                    s = float32(s + advance)
                    break
                level -= 1
            if ray_out:
                break
        return current if current > -1e29 else -np.inf

    @cuda.jit(device=True)
    def interpolate_subpatch_segment(
        segments, primary_level0, primary_map, active_map, tile_column,
        tile_row, tile_width, subpatch_size, pass_index, pixel, azimuth,
        segment,
    ):
        row = pixel // tile_width
        column = pixel % tile_width
        count = tile_width // subpatch_size + 2
        gx = float32(
            float32(column - subpatch_size // 2) / float32(subpatch_size)
            + float32(1.0)
        )
        gy = float32(
            float32(row - subpatch_size // 2) / float32(subpatch_size)
            + float32(1.0)
        )
        left = int(gx)
        top = int(gy)
        tx = float32(gx - left)
        ty = float32(gy - top)
        if left < 0:
            left = 0
            tx = 0.0
        if top < 0:
            top = 0
            ty = 0.0
        if left > count - 2:
            left = count - 2
            tx = 1.0
        if top > count - 2:
            top = count - 2
            ty = 1.0
        indices = cuda.local.array(4, dtype=int32)
        indices[0] = top * count + left
        indices[1] = top * count + left + 1
        indices[2] = (top + 1) * count + left
        indices[3] = (top + 1) * count + left + 1
        primary_resolution = float32(math.sqrt(float32(
            float32(primary_map[6] * primary_map[6])
            + float32(primary_map[9] * primary_map[9])
        )))
        active_resolution = float32(math.sqrt(float32(
            float32(active_map[6] * active_map[6])
            + float32(active_map[9] * active_map[9])
        )))
        scale_ratio = float32(primary_resolution / active_resolution)
        requested_left = tile_column + (left - 1) * subpatch_size + subpatch_size // 2
        requested_right = requested_left + subpatch_size
        requested_top = tile_row + (top - 1) * subpatch_size + subpatch_size // 2
        requested_bottom = requested_top + subpatch_size
        center_left = clamp_center(
            requested_left, primary_level0.shape[1], subpatch_size
        ) - tile_column
        center_right = clamp_center(
            requested_right, primary_level0.shape[1], subpatch_size
        ) - tile_column
        center_top = clamp_center(
            requested_top, primary_level0.shape[0], subpatch_size
        ) - tile_row
        center_bottom = clamp_center(
            requested_bottom, primary_level0.shape[0], subpatch_size
        ) - tile_row
        shifts_x = cuda.local.array(4, dtype=float32)
        shifts_y = cuda.local.array(4, dtype=float32)
        shifts_x[0] = float32(float32(column - center_left) * scale_ratio)
        shifts_x[1] = float32(float32(column - center_right) * scale_ratio)
        shifts_x[2] = shifts_x[0]
        shifts_x[3] = shifts_x[1]
        shifts_y[0] = float32(float32(row - center_top) * scale_ratio)
        shifts_y[1] = shifts_y[0]
        shifts_y[2] = float32(float32(row - center_bottom) * scale_ratio)
        shifts_y[3] = shifts_y[2]
        for field in range(18):
            values = cuda.local.array(4, dtype=float32)
            for corner in range(4):
                value = segments[azimuth, indices[corner], pass_index, field]
                if field == 0 or field == 2:
                    value = float32(value + shifts_x[corner])
                elif field == 1 or field == 3:
                    value = float32(value + shifts_y[corner])
                values[corner] = value
            upper_difference = float32(values[1] - values[0])
            lower_difference = float32(values[3] - values[2])
            upper = float32(cuda.fma(upper_difference, tx, values[0]))
            lower = float32(cuda.fma(lower_difference, tx, values[2]))
            vertical_difference = float32(lower - upper)
            segment[field] = float32(cuda.fma(vertical_difference, ty, upper))

    @cuda.jit
    def subpatch_interpolation_kernel(
        segments, primary_level0, primary_map, active_map, tile_column,
        tile_row, tile_width, subpatch_size, pass_index, pixels, azimuths,
        output,
    ):
        index = cuda.grid(1)
        if index >= output.shape[0]:
            return
        segment = cuda.local.array(18, dtype=float32)
        interpolate_subpatch_segment(
            segments, primary_level0, primary_map, active_map, tile_column,
            tile_row, tile_width, subpatch_size, pass_index, pixels[index],
            azimuths[index], segment,
        )
        for field in range(18):
            output[index, field] = segment[field]

    @cuda.jit
    def subpatch_hierarchy_kernel(
        segments, primary_level0, primary_map, active_level0, active_mips,
        active_levels, active_map, active_projection, tile_column, tile_row,
        tile_width, tile_height, subpatch_size, pass_index, observer_elevation,
        output,
    ):
        pixel, azimuth = cuda.grid(2)
        if pixel >= tile_width * tile_height or azimuth >= segments.shape[0]:
            return
        row = pixel // tile_width
        column = pixel % tile_width
        active_column_resolution = float32(math.sqrt(float32(
            float32(active_map[6] * active_map[6])
            + float32(active_map[9] * active_map[9])
        )))
        active_row_resolution = float32(math.sqrt(float32(
            float32(active_map[7] * active_map[7])
            + float32(active_map[10] * active_map[10])
        )))
        segment = cuda.local.array(18, dtype=float32)
        interpolate_subpatch_segment(
            segments, primary_level0, primary_map, active_map, tile_column,
            tile_row, tile_width, subpatch_size, pass_index, pixel, azimuth,
            segment,
        )
        observer_terrain = bilinear(
            primary_level0, tile_column + column, tile_row + row
        )
        observer_z = float32(observer_terrain + observer_elevation)
        map_resolution = float32(
            float32(active_column_resolution + active_row_resolution)
            * float32(0.5)
        )
        output[pixel, azimuth] = hierarchy_maximum(
            segment, active_level0, active_mips, active_levels, observer_z,
            active_projection[0], map_resolution, pass_index,
            output[pixel, azimuth],
        )

    return (
        mapping_kernel, helper_kernel, fixed_step_kernel, adaptive_kernel,
        hierarchy_kernel, subpatch_interpolation_kernel, subpatch_hierarchy_kernel,
    )


@dataclass(frozen=True, slots=True)
class CudaDeviceInfo:
    name: str
    compute_capability: tuple[int, int]


class CudaSession:
    """An explicitly initialized diagnostic CUDA session."""

    def __init__(self, device_id: int = 0) -> None:
        global _KERNELS
        try:
            from numba import cuda
        except ImportError as error:
            raise CudaBackendError("Numba CUDA is not installed") from error
        if not cuda.is_available():
            raise CudaBackendError("Numba CUDA cannot see a usable device")
        try:
            cuda.select_device(device_id)
            device = cuda.get_current_device()
        except Exception as error:
            raise CudaBackendError(f"cannot select CUDA device {device_id}") from error
        self._cuda = cuda
        name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
        self.info = CudaDeviceInfo(name, tuple(device.compute_capability))
        if _KERNELS is None:
            _KERNELS = _build_kernels(cuda)
        (
            self._mapping_kernel, self._helper_kernel, self._fixed_step_kernel,
            self._adaptive_kernel, self._hierarchy_kernel,
            self._subpatch_interpolation_kernel,
            self._subpatch_hierarchy_kernel,
        ) = _KERNELS

    def index_mapping(self, pixel_count: int, azimuth_count: int) -> np.ndarray:
        if pixel_count <= 0 or azimuth_count <= 0:
            raise ValueError("mapping dimensions must be positive")
        host = np.full((pixel_count, azimuth_count), -np.inf, dtype=np.float32)
        device = self._cuda.to_device(host)
        threads = (16, 16)
        blocks = (
            (pixel_count + threads[0] - 1) // threads[0],
            (azimuth_count + threads[1] - 1) // threads[1],
        )
        self._mapping_kernel[blocks, threads](device, pixel_count, azimuth_count)
        self._cuda.synchronize()
        device.copy_to_host(host)
        return host

    def helper_diagnostics(self, **arrays) -> np.ndarray:
        count = arrays["segments"].shape[0]
        output = self._cuda.device_array((count, 14), dtype=np.float32)
        arguments = [self._cuda.to_device(np.ascontiguousarray(arrays[name])) for name in (
            "segments", "shifts", "scales", "weights", "distances",
            "planar_distances", "elevation", "sample_coordinates",
            "requested_centers", "dem_sizes", "tile_widths", "subpatch_sizes",
            "pixel_coordinates",
        )]
        threads = 128
        blocks = (count + threads - 1) // threads
        self._helper_kernel[blocks, threads](*arguments, output)
        self._cuda.synchronize()
        return output.copy_to_host()

    def fixed_step_level0(
        self, segments, elevation, observer_z, radius, map_resolution,
        *, step_km: float = 0.0012, trace_capacity: int = 8192,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        segments = np.ascontiguousarray(segments, dtype=np.float32)
        elevation = np.ascontiguousarray(elevation, dtype=np.float32)
        observer_z = np.ascontiguousarray(observer_z, dtype=np.float32)
        radius = np.ascontiguousarray(radius, dtype=np.float32)
        count = len(segments)
        device_segments = self._cuda.to_device(segments)
        device_elevation = self._cuda.to_device(elevation)
        device_observer_z = self._cuda.to_device(observer_z)
        device_radius = self._cuda.to_device(radius)
        maximum = self._cuda.device_array(count, dtype=np.float32)
        traces = self._cuda.device_array((count, trace_capacity, 7), dtype=np.float32)
        trace_counts = self._cuda.device_array(count, dtype=np.int32)
        threads = 128
        blocks = (count + threads - 1) // threads
        self._fixed_step_kernel[blocks, threads](
            device_segments, device_elevation, device_observer_z, device_radius,
            np.float32(map_resolution), np.float32(step_km), maximum, traces,
            trace_counts,
        )
        self._cuda.synchronize()
        return (
            maximum.copy_to_host(), traces.copy_to_host(), trace_counts.copy_to_host()
        )

    def adaptive_level0(
        self, segments, elevation, observer_z, radius, map_resolution,
        *, pass_index: int = 0, trace_capacity: int = 8192,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        segments = np.ascontiguousarray(segments, dtype=np.float32)
        elevation = np.ascontiguousarray(elevation, dtype=np.float32)
        observer_z = np.ascontiguousarray(observer_z, dtype=np.float32)
        radius = np.ascontiguousarray(radius, dtype=np.float32)
        count = len(segments)
        device_segments = self._cuda.to_device(segments)
        device_elevation = self._cuda.to_device(elevation)
        device_observer_z = self._cuda.to_device(observer_z)
        device_radius = self._cuda.to_device(radius)
        maximum = self._cuda.device_array(count, dtype=np.float32)
        traces = self._cuda.device_array((count, trace_capacity, 8), dtype=np.float32)
        trace_counts = self._cuda.device_array(count, dtype=np.int32)
        threads = 128
        blocks = (count + threads - 1) // threads
        self._adaptive_kernel[blocks, threads](
            device_segments, device_elevation, device_observer_z, device_radius,
            np.float32(map_resolution), np.int32(pass_index), maximum, traces,
            trace_counts,
        )
        self._cuda.synchronize()
        return maximum.copy_to_host(), traces.copy_to_host(), trace_counts.copy_to_host()

    def hierarchical(
        self, segments, pyramid, observer_z, radius, map_resolution,
        *, pass_index: int, trace_capacity: int = 16384,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        segments = np.ascontiguousarray(segments, dtype=np.float32)
        observer_z = np.ascontiguousarray(observer_z, dtype=np.float32)
        radius = np.ascontiguousarray(radius, dtype=np.float32)
        count = len(segments)
        maximum = self._cuda.device_array(count, dtype=np.float32)
        traces = self._cuda.device_array((count, trace_capacity, 12), dtype=np.float32)
        trace_counts = self._cuda.device_array(count, dtype=np.int32)
        threads = 128
        blocks = (count + threads - 1) // threads
        self._hierarchy_kernel[blocks, threads](
            self._cuda.to_device(segments), self._cuda.to_device(pyramid.level0),
            self._cuda.to_device(pyramid.mips), self._cuda.to_device(pyramid.levels),
            self._cuda.to_device(observer_z), self._cuda.to_device(radius),
            np.float32(map_resolution), np.int32(pass_index), maximum, traces,
            trace_counts,
        )
        self._cuda.synchronize()
        return maximum.copy_to_host(), traces.copy_to_host(), trace_counts.copy_to_host()

    def subpatch_hierarchical_pass(
        self,
        segment_values,
        primary_pyramid,
        active_pyramid,
        *,
        tile_column: int,
        tile_row: int,
        tile_width: int,
        tile_height: int,
        subpatch_size: int,
        pass_index: int,
        observer_elevation_m: float = 0.0,
        slopes=None,
    ) -> np.ndarray:
        """Run one production-shaped DEM pass and retain slope units."""
        segments = np.ascontiguousarray(segment_values, dtype=np.float32)
        if segments.ndim != 4 or segments.shape[-1] != 18:
            raise ValueError("segment_values must have shape (azimuth, center, DEM, 18)")
        if not 0 <= pass_index < segments.shape[2]:
            raise ValueError("pass_index is outside the segment DEM axis")
        expected_centers = (tile_width // subpatch_size + 2) ** 2
        if segments.shape[1] != expected_centers:
            raise ValueError("segment center count does not match tile/subpatch layout")
        pixel_count = tile_width * tile_height
        output_shape = (pixel_count, segments.shape[0])
        if slopes is None:
            host_output = np.full(output_shape, -np.inf, dtype=np.float32)
        else:
            host_output = np.ascontiguousarray(slopes, dtype=np.float32)
            if host_output.shape != output_shape:
                raise ValueError(f"slopes must have shape {output_shape}")
        device_output = self._cuda.to_device(host_output)
        threads = (8, 32)
        blocks = (
            (pixel_count + threads[0] - 1) // threads[0],
            (segments.shape[0] + threads[1] - 1) // threads[1],
        )
        self._subpatch_hierarchy_kernel[blocks, threads](
            self._cuda.to_device(segments),
            self._cuda.to_device(primary_pyramid.level0),
            self._cuda.to_device(primary_pyramid.map_parameters),
            self._cuda.to_device(active_pyramid.level0),
            self._cuda.to_device(active_pyramid.mips),
            self._cuda.to_device(active_pyramid.levels),
            self._cuda.to_device(active_pyramid.map_parameters),
            self._cuda.to_device(active_pyramid.projection_parameters),
            np.int32(tile_column), np.int32(tile_row), np.int32(tile_width),
            np.int32(tile_height), np.int32(subpatch_size), np.int32(pass_index),
            np.float32(observer_elevation_m), device_output,
        )
        self._cuda.synchronize()
        return device_output.copy_to_host()

    def subpatch_interpolation(
        self, segment_values, primary_pyramid, active_pyramid, *, tile_column,
        tile_row, tile_width, subpatch_size, pass_index, pixels, azimuths,
    ) -> np.ndarray:
        """Return selected device-interpolated segments for diagnostics."""
        segments = np.ascontiguousarray(segment_values, dtype=np.float32)
        pixels = np.ascontiguousarray(pixels, dtype=np.int32)
        azimuths = np.ascontiguousarray(azimuths, dtype=np.int32)
        if pixels.shape != azimuths.shape or pixels.ndim != 1:
            raise ValueError("pixels and azimuths must be equal-length vectors")
        output = self._cuda.device_array((len(pixels), 18), dtype=np.float32)
        threads = 128
        blocks = (len(pixels) + threads - 1) // threads
        self._subpatch_interpolation_kernel[blocks, threads](
            self._cuda.to_device(segments),
            self._cuda.to_device(primary_pyramid.level0),
            self._cuda.to_device(primary_pyramid.map_parameters),
            self._cuda.to_device(active_pyramid.map_parameters),
            np.int32(tile_column), np.int32(tile_row), np.int32(tile_width),
            np.int32(subpatch_size), np.int32(pass_index),
            self._cuda.to_device(pixels), self._cuda.to_device(azimuths), output,
        )
        self._cuda.synchronize()
        return output.copy_to_host()
