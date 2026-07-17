"""Independent CPU oracle for diagnostic fixed-step level-0 traversal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .kernel_math import (
    evaluate_planar_chord,
    evaluate_quartic,
    evaluate_tangent,
    is_valid_elevation,
    sample_bilinear,
)


FIXED_STEP_KM = np.float32(0.0012)


@dataclass(frozen=True, slots=True)
class FixedStepTrace:
    maximum_slope: np.float32
    values: npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AdaptiveTrace:
    maximum_slope: np.float32
    values: npt.NDArray[np.float32]


def traverse_level0_fixed_step(
    segment: npt.ArrayLike,
    elevation_m: npt.NDArray[np.float32],
    *, observer_z_m: float,
    radius_m: float,
    map_resolution_m: float,
    step_km: float = float(FIXED_STEP_KM),
) -> FixedStepTrace:
    """March one fitted ray with the C# diagnostic fixed-step arithmetic."""
    segment = np.asarray(segment, dtype=np.float32)
    step = np.float32(step_km)
    s_start = segment[12]
    s = max(s_start, np.float32(0.001)) + step
    current = np.float32(-1e30)
    trace = []
    while s <= segment[13]:
        delta = s - s_start
        pixel_x = evaluate_quartic(segment[0], segment[4:8], delta)
        pixel_y = evaluate_quartic(segment[1], segment[8:12], delta)
        if (
            not np.isfinite(pixel_x) or not np.isfinite(pixel_y)
            or pixel_x < 0 or pixel_y < 0
            or pixel_x >= elevation_m.shape[1] - 1
            or pixel_y >= elevation_m.shape[0] - 1
        ):
            break
        planar_x = (pixel_x - segment[0]) * np.float32(map_resolution_m)
        planar_y = (pixel_y - segment[1]) * np.float32(map_resolution_m)
        planar_m = np.sqrt(planar_x * planar_x + planar_y * planar_y).astype(np.float32)
        if s < np.float32(0.5):
            true_m = s * np.float32(1000.0)
        else:
            true_m = (
                segment[14] * np.float32(1000.0)
                + evaluate_planar_chord(segment, planar_m)
            )
        height = sample_bilinear(elevation_m, float(pixel_x), float(pixel_y))
        slope = np.float32(-1e30)
        if is_valid_elevation(float(height)):
            if s < np.float32(0.5):
                slope = (
                    (height - np.float32(observer_z_m)) / true_m
                    if true_m > np.float32(1e-6) else np.float32(-1e30)
                )
            else:
                observer_radius = np.float32(radius_m + observer_z_m)
                distance_squared = true_m * true_m
                local_z = (
                    (height - np.float32(observer_z_m))
                    * (np.float32(2.0 * radius_m) + height + np.float32(observer_z_m))
                    - distance_squared
                ) / (np.float32(2.0) * observer_radius)
                local_x_squared = distance_squared - local_z * local_z
                local_x = (
                    np.sqrt(local_x_squared).astype(np.float32)
                    if local_x_squared > 0 else np.float32(1e-6)
                )
                slope = local_z / local_x if local_x != 0 else np.float32(-1e30)
            current = max(current, slope)
        trace.append((s, true_m, pixel_x, pixel_y, height, slope, current))
        s += step
    values = np.ascontiguousarray(trace, dtype=np.float32)
    if not len(values):
        values = np.empty((0, 7), dtype=np.float32)
    maximum = current if current > np.float32(-1e29) else np.float32(-np.inf)
    return FixedStepTrace(maximum, values)


def traverse_level0_adaptive(
    segment: npt.ArrayLike,
    elevation_m: npt.NDArray[np.float32],
    *, observer_z_m: float,
    radius_m: float,
    map_resolution_m: float,
    pass_index: int = 0,
) -> AdaptiveTrace:
    """March one fitted ray with current production adaptive level-0 rules."""
    segment = np.asarray(segment, dtype=np.float32)
    s_start = segment[12]
    s = max(s_start, np.float32(0.001))
    current = np.float32(-1e30)
    minimum_step = np.float32(0.5 * map_resolution_m / 1000.0)
    primary_far_step = np.float32(0.8 * map_resolution_m / 1000.0)
    trace = []
    while s <= segment[13]:
        delta = s - s_start
        pixel_x = evaluate_quartic(segment[0], segment[4:8], delta)
        pixel_y = evaluate_quartic(segment[1], segment[8:12], delta)
        if (
            not np.isfinite(pixel_x) or not np.isfinite(pixel_y)
            or pixel_x < 0 or pixel_y < 0
            or pixel_x >= elevation_m.shape[1] - 1
            or pixel_y >= elevation_m.shape[0] - 1
        ):
            break
        planar_x = (pixel_x - segment[0]) * np.float32(map_resolution_m)
        planar_y = (pixel_y - segment[1]) * np.float32(map_resolution_m)
        planar_m = np.sqrt(planar_x * planar_x + planar_y * planar_y).astype(np.float32)
        if s < np.float32(0.5):
            true_m = s * np.float32(1000.0)
        else:
            true_m = segment[14] * np.float32(1000.0) + evaluate_planar_chord(
                segment, planar_m
            )
        height = sample_bilinear(elevation_m, float(pixel_x), float(pixel_y))
        slope = np.float32(-1e30)
        if is_valid_elevation(float(height)):
            if s < np.float32(0.5):
                slope = (
                    (height - np.float32(observer_z_m)) / true_m
                    if true_m > np.float32(1e-6) else np.float32(-1e30)
                )
            else:
                observer_radius = np.float32(radius_m + observer_z_m)
                distance_squared = true_m * true_m
                local_z = (
                    (height - np.float32(observer_z_m))
                    * (np.float32(2.0 * radius_m) + height + np.float32(observer_z_m))
                    - distance_squared
                ) / (np.float32(2.0) * observer_radius)
                local_x_squared = distance_squared - local_z * local_z
                local_x = (
                    np.sqrt(local_x_squared).astype(np.float32)
                    if local_x_squared > 0 else np.float32(1e-6)
                )
                slope = local_z / local_x if local_x != 0 else np.float32(-1e30)
            current = max(current, slope)
        dx, dy = evaluate_tangent(segment[4:8], segment[8:12], delta)
        magnitude = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        pixel_step = np.float32(1.0) / magnitude if magnitude > 1e-6 else np.float32(0.001)
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
            if pass_index == 0 and true_m >= np.float32(100.0)
            else minimum_step
        )
        advance = max(advance, floor)
        trace.append((s, true_m, pixel_x, pixel_y, height, slope, current, advance))
        s += advance
    values = np.ascontiguousarray(trace, dtype=np.float32)
    if not len(values):
        values = np.empty((0, 8), dtype=np.float32)
    maximum = current if current > np.float32(-1e29) else np.float32(-np.inf)
    return AdaptiveTrace(maximum, values)
