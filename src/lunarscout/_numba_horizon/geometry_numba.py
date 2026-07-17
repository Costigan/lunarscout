"""Optional Numba CPU kernels for measured host-side segment fitting.

This module is intentionally not imported by :mod:`lunarscout` or by the
private data-contract package. Import it only when the Phase 3 prototype
dependency is installed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange


def _cached_njit(**options):
    """Use Numba's disk cache when its cache location is writable."""

    def decorate(function):
        try:
            return njit(cache=True, **options)(function)
        except RuntimeError as error:
            if "no locator available" not in str(error):
                raise
            return njit(**options)(function)

    return decorate


@_cached_njit()
def _solve(matrix: npt.NDArray[np.float64], rhs: npt.NDArray[np.float64]):
    size = rhs.shape[0]
    augmented = np.empty((size, size + 1), dtype=np.float64)
    for row in range(size):
        for column in range(size):
            augmented[row, column] = matrix[row, column]
        augmented[row, size] = rhs[row]
    for index in range(size):
        pivot = index
        maximum = abs(augmented[index, index])
        for row in range(index + 1, size):
            value = abs(augmented[row, index])
            if value > maximum:
                maximum = value
                pivot = row
        if maximum < 1e-12:
            return np.zeros(size, dtype=np.float64), False
        if pivot != index:
            for column in range(index, size + 1):
                value = augmented[index, column]
                augmented[index, column] = augmented[pivot, column]
                augmented[pivot, column] = value
        divisor = augmented[index, index]
        for column in range(index, size + 1):
            augmented[index, column] /= divisor
        for row in range(size):
            if row == index:
                continue
            factor = augmented[row, index]
            if abs(factor) < 1e-12:
                continue
            for column in range(index, size + 1):
                augmented[row, column] -= factor * augmented[index, column]
    return augmented[:, size].copy(), True


@_cached_njit()
def _quartic(samples, count, value_column, value0, anchor_km, span_km):
    normal = np.zeros((4, 4), dtype=np.float64)
    rhs = np.zeros(4, dtype=np.float64)
    for sample_index in range(count):
        parameter = (samples[sample_index, 0] / 1000.0 - anchor_km) / span_km
        powers = np.empty(4, dtype=np.float64)
        powers[0] = parameter
        for power in range(1, 4):
            powers[power] = powers[power - 1] * parameter
        value = samples[sample_index, value_column] - value0
        for row in range(4):
            rhs[row] += powers[row] * value
            for column in range(4):
                normal[row, column] += powers[row] * powers[column]
    solution, valid = _solve(normal, rhs)
    if not valid:
        return np.zeros(4, dtype=np.float64)
    inverse = 1.0 / span_km
    scale = inverse
    for power in range(4):
        solution[power] *= scale
        scale *= inverse
    return solution


@_cached_njit()
def _chord_distance(observer, latitude, longitude, radius):
    cos_latitude = np.cos(latitude)
    x = radius * cos_latitude * np.cos(longitude) - observer[0]
    y = radius * cos_latitude * np.sin(longitude) - observer[1]
    z = radius * np.sin(latitude) - observer[2]
    return np.sqrt(x * x + y * y + z * z)


@_cached_njit()
def _fit_chord(samples, count, map_resolution, observer, radius):
    identity = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if count < 2 or map_resolution <= 0.0:
        return identity
    planar = np.zeros(16, dtype=np.float64)
    chord = np.zeros(16, dtype=np.float64)
    chord0 = _chord_distance(observer, samples[0, 3], samples[0, 4], radius)
    maximum_planar = 0.0
    used = 1
    for index in range(1, min(count, 16)):
        if samples[index, 1] < 0.0 or samples[index, 2] < 0.0:
            continue
        dx = (samples[index, 1] - samples[0, 1]) * map_resolution
        dy = (samples[index, 2] - samples[0, 2]) * map_resolution
        distance = np.sqrt(dx * dx + dy * dy)
        planar[used] = distance
        chord[used] = (
            _chord_distance(observer, samples[index, 3], samples[index, 4], radius)
            - chord0
        )
        maximum_planar = max(maximum_planar, distance)
        used += 1
    if used < 2 or maximum_planar < 500.0:
        return identity
    normal = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for index in range(used):
        x = planar[index]
        powers = np.array((x, x * x, x * x * x), dtype=np.float64)
        for row in range(3):
            rhs[row] += powers[row] * chord[index]
            for column in range(3):
                normal[row, column] += powers[row] * powers[column]
    solution, valid = _solve(normal, rhs)
    if not valid or solution[0] < 0.5 or solution[0] > 2.0:
        return identity
    return solution


@_cached_njit()
def _fit_one(samples, count, map_resolution, observer, radius, fallback_start):
    result = np.zeros(18, dtype=np.float32)
    if count < 3:
        x = samples[0, 1] if count else 0.0
        y = samples[0, 2] if count else 0.0
        start = samples[0, 0] / 1000.0 if count else fallback_start / 1000.0
        result[0:4] = np.array((x, y, x, y), dtype=np.float32)
        result[12:16] = np.array((start, start, start, 1.0), dtype=np.float32)
        return result
    x0 = samples[0, 1]
    y0 = samples[0, 2]
    anchor_km = samples[0, 0] / 1000.0
    end_km = samples[count - 1, 0] / 1000.0
    span_km = max(0.001, end_km - anchor_km)
    x_coefficients = _quartic(samples, count, 1, x0, anchor_km, span_km)
    y_coefficients = _quartic(samples, count, 2, y0, anchor_km, span_km)
    chord = _fit_chord(samples, count, map_resolution, observer, radius)
    result[0] = x0
    result[1] = y0
    result[2] = x0
    result[3] = y0
    result[4:8] = x_coefficients
    result[8:12] = y_coefficients
    result[12] = anchor_km
    result[13] = end_km
    result[14] = anchor_km
    result[15:18] = chord
    return result


@_cached_njit()
def _sample_chord(observer, direction, distance, elevation, transform, projection):
    x_vector = observer[0] + direction[0] * distance
    y_vector = observer[1] + direction[1] * distance
    z_vector = observer[2] + direction[2] * distance
    longitude = np.arctan2(y_vector, x_vector)
    if longitude < 0.0:
        longitude += 2.0 * np.pi
    latitude = np.arctan2(z_vector, np.sqrt(x_vector * x_vector + y_vector * y_vector))

    sin_latitude = np.sin(latitude)
    cos_latitude = np.cos(latitude)
    sin_origin = np.sin(projection[1])
    cos_origin = np.cos(projection[1])
    delta = longitude - projection[2]
    denominator = (
        1.0 + sin_origin * sin_latitude
        + cos_origin * cos_latitude * np.cos(delta)
    )
    if abs(denominator) < 1e-10:
        denominator = 1e-10
    scale = 2.0 * projection[3] * projection[0] / denominator
    projected_x = scale * cos_latitude * np.sin(delta) + projection[4]
    projected_y = (
        scale
        * (cos_origin * sin_latitude - sin_origin * cos_latitude * np.cos(delta))
        + projection[5]
    )
    determinant = transform[1] * transform[5] - transform[2] * transform[4]
    dx = projected_x - transform[0]
    dy = projected_y - transform[3]
    column = (transform[5] * dx - transform[2] * dy) / determinant
    row = (-transform[4] * dx + transform[1] * dy) / determinant
    inside = 0.0 <= column < elevation.shape[1] and 0.0 <= row < elevation.shape[0]
    terrain = 0.0
    if inside:
        x1 = int(column)
        y1 = int(row)
        x2 = min(x1 + 1, elevation.shape[1] - 1)
        y2 = min(y1 + 1, elevation.shape[0] - 1)
        terrain = (
            elevation[y1, x1] * (x2 - column) * (y2 - row)
            + elevation[y1, x2] * (column - x1) * (y2 - row)
            + elevation[y2, x1] * (x2 - column) * (row - y1)
            + elevation[y2, x2] * (column - x1) * (row - y1)
        )
    sample = np.array(
        (distance, column, row, latitude, longitude, row, column, terrain),
        dtype=np.float64,
    )
    return inside, sample


@_cached_njit()
def _build_samples(observer, direction, start, maximum, elevation, transform, projection):
    samples = np.zeros((16, 8), dtype=np.float64)
    inside, first = _sample_chord(
        observer, direction, start, elevation, transform, projection
    )
    if not inside:
        return samples, 0
    samples[0] = first
    count = 1
    final_inside = first
    inside, end = _sample_chord(
        observer, direction, maximum, elevation, transform, projection
    )
    if inside:
        final_inside = end
    else:
        low = start
        high = maximum
        best = first
        for _ in range(24):
            middle = 0.5 * (low + high)
            inside, sample = _sample_chord(
                observer, direction, middle, elevation, transform, projection
            )
            if inside:
                low = middle
                best = sample
            else:
                high = middle
        final_inside = best

    span = max(0.0, final_inside[0] - start)
    if span > 1e-3:
        for index in range(1, 11):
            target = start + span * index / 10.0
            if target > final_inside[0] - 0.5:
                break
            if target - samples[count - 1, 0] < 0.5:
                continue
            inside, sample = _sample_chord(
                observer, direction, target, elevation, transform, projection
            )
            if inside:
                samples[count] = sample
                count += 1
        if count < 16 and final_inside[0] - samples[count - 1, 0] > 0.5:
            samples[count] = final_inside
            count += 1

    required_end = samples[0, 0] + 100.0
    map_resolution = 0.5 * (
        np.sqrt(transform[1] ** 2 + transform[4] ** 2)
        + np.sqrt(transform[2] ** 2 + transform[5] ** 2)
    )
    if count < 4 or samples[count - 1, 0] < required_end:
        step = max(10.0, map_resolution * 2.0)
        current = samples[count - 1, 0]
        extend_limit = max(final_inside[0], required_end)
        while (
            (count < 4 or samples[count - 1, 0] < required_end)
            and current < extend_limit
            and count < 16
        ):
            current = min(extend_limit, current + step)
            inside, sample = _sample_chord(
                observer, direction, current, elevation, transform, projection
            )
            if not inside or current - samples[count - 1, 0] < 0.5:
                break
            samples[count] = sample
            count += 1
    return samples, count


@_cached_njit()
def _generate_one(
    elevation, transform, projection, observer, direction, start, maximum,
    map_resolution, correction_radius,
):
    samples, count = _build_samples(
        observer, direction, start, maximum, elevation, transform, projection
    )
    return _fit_one(
        samples, count, map_resolution, observer, correction_radius, start
    )


@_cached_njit()
def fit_segments_serial(samples, counts, map_resolutions, observers, radii, starts):
    """Fit an independent padded batch in deterministic serial order."""
    result = np.empty((samples.shape[0], 18), dtype=np.float32)
    for index in range(samples.shape[0]):
        result[index] = _fit_one(
            samples[index], counts[index], map_resolutions[index], observers[index],
            radii[index], starts[index],
        )
    return result


@_cached_njit(parallel=True)
def fit_segments_parallel(samples, counts, map_resolutions, observers, radii, starts):
    """Fit an independent padded batch with one Numba iteration per segment."""
    result = np.empty((samples.shape[0], 18), dtype=np.float32)
    for index in prange(samples.shape[0]):
        result[index] = _fit_one(
            samples[index], counts[index], map_resolutions[index], observers[index],
            radii[index], starts[index],
        )
    return result


@_cached_njit()
def generate_segments_serial(
    elevation, transform, projection, observers, directions, starts, maximums,
    map_resolution, correction_radii,
):
    """Sample one DEM and fit an independent segment batch serially."""
    result = np.empty((observers.shape[0], 18), dtype=np.float32)
    for index in range(observers.shape[0]):
        result[index] = _generate_one(
            elevation, transform, projection, observers[index], directions[index],
            starts[index], maximums[index], map_resolution, correction_radii[index],
        )
    return result


@_cached_njit(parallel=True)
def generate_segments_parallel(
    elevation, transform, projection, observers, directions, starts, maximums,
    map_resolution, correction_radii,
):
    """Sample one DEM and fit an independent segment batch in parallel."""
    result = np.empty((observers.shape[0], 18), dtype=np.float32)
    for index in prange(observers.shape[0]):
        result[index] = _generate_one(
            elevation, transform, projection, observers[index], directions[index],
            starts[index], maximums[index], map_resolution, correction_radii[index],
        )
    return result
