"""CPU oracle helpers for the diagnostic Python horizon kernel stages."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


INVALID_ELEVATION_M = np.float32(-32000.0)


def is_valid_elevation(value: float) -> bool:
    return bool(np.isfinite(value) and value > -20000.0)


def evaluate_quartic(x0: float, coefficients: npt.ArrayLike, distance: float) -> np.float32:
    a1, a2, a3, a4 = np.asarray(coefficients, dtype=np.float32)
    s = np.float32(distance)
    s2 = s * s
    return np.float32(x0) + a1 * s + a2 * s2 + a3 * s2 * s + a4 * s2 * s2


def evaluate_tangent(
    x_coefficients: npt.ArrayLike,
    y_coefficients: npt.ArrayLike,
    distance: float,
) -> tuple[np.float32, np.float32]:
    a1, a2, a3, a4 = np.asarray(x_coefficients, dtype=np.float32)
    b1, b2, b3, b4 = np.asarray(y_coefficients, dtype=np.float32)
    s = np.float32(distance)
    s2, s3 = s * s, s * s * s
    return (
        a1 + np.float32(2) * a2 * s + np.float32(3) * a3 * s2
        + np.float32(4) * a4 * s3,
        b1 + np.float32(2) * b2 * s + np.float32(3) * b3 * s2
        + np.float32(4) * b4 * s3,
    )


def evaluate_planar_chord(segment: npt.ArrayLike, planar_m: float) -> np.float32:
    values = np.asarray(segment, dtype=np.float32)
    distance = np.float32(planar_m)
    return (
        values[15] * distance
        + values[16] * distance * distance
        + values[17] * distance * distance * distance
    )


def sample_bilinear(elevation: npt.NDArray[np.float32], column: float, row: float) -> np.float32:
    column = float(np.clip(column, 0.0, elevation.shape[1] - 1.0001))
    row = float(np.clip(row, 0.0, elevation.shape[0] - 1.0001))
    x0, y0 = int(np.floor(column)), int(np.floor(row))
    x1, y1 = min(x0 + 1, elevation.shape[1] - 1), min(y0 + 1, elevation.shape[0] - 1)
    values = (
        elevation[y0, x0], elevation[y0, x1],
        elevation[y1, x0], elevation[y1, x1],
    )
    if not all(is_valid_elevation(float(value)) for value in values):
        return INVALID_ELEVATION_M
    tx, ty = np.float32(column - x0), np.float32(row - y0)
    top = values[0] + tx * (values[1] - values[0])
    bottom = values[2] + tx * (values[3] - values[2])
    return np.float32(top + ty * (bottom - top))


def clamp_subpatch_center(requested: int, dem_size: int, subpatch_size: int) -> int:
    half = subpatch_size // 2
    return min(max(requested, half), max(half, dem_size - half))


def interpolation_selection(
    pixel_column: int, pixel_row: int, tile_width: int, subpatch_size: int
) -> tuple[int, int, int, int, np.float32, np.float32]:
    count = tile_width // subpatch_size + 2
    gx = np.float32(
        (np.float32(pixel_column) - np.float32(subpatch_size) / np.float32(2))
        / np.float32(subpatch_size) + np.float32(1)
    )
    gy = np.float32(
        (np.float32(pixel_row) - np.float32(subpatch_size) / np.float32(2))
        / np.float32(subpatch_size) + np.float32(1)
    )
    left, top = int(gx), int(gy)
    tx, ty = np.float32(gx - left), np.float32(gy - top)
    if left < 0:
        left, tx = 0, np.float32(0)
    if top < 0:
        top, ty = 0, np.float32(0)
    if left > count - 2:
        left, tx = count - 2, np.float32(1)
    if top > count - 2:
        top, ty = count - 2, np.float32(1)
    right, bottom = left + 1, top + 1
    return (
        top * count + left,
        top * count + right,
        bottom * count + left,
        bottom * count + right,
        tx,
        ty,
    )


def shift_segment(
    segment: npt.ArrayLike, delta_column: float, delta_row: float, scale_ratio: float
) -> npt.NDArray[np.float32]:
    result = np.array(segment, dtype=np.float32, copy=True)
    dx = np.float32(delta_column) * np.float32(scale_ratio)
    dy = np.float32(delta_row) * np.float32(scale_ratio)
    result[0] += dx
    result[1] += dy
    result[2] += dx
    result[3] += dy
    return result


def interpolate_segments(
    segments: npt.ArrayLike, shifts: npt.ArrayLike, scale_ratio: float,
    tx: float, ty: float,
) -> npt.NDArray[np.float32]:
    values = np.asarray(segments, dtype=np.float32)
    deltas = np.asarray(shifts, dtype=np.float32)
    shifted = np.stack(
        [shift_segment(values[index], *deltas[index], scale_ratio) for index in range(4)]
    )
    tx, ty = np.float32(tx), np.float32(ty)
    top = shifted[0] + (shifted[1] - shifted[0]) * tx
    bottom = shifted[2] + (shifted[3] - shifted[2]) * tx
    return np.ascontiguousarray(top + (bottom - top) * ty, dtype=np.float32)
