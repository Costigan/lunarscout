"""Host-side geometry for the experimental Python/Numba horizon backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt

from .contract import DEVICE_FLOAT_DTYPE, SEGMENT_FIELDS, ContractValidationError


MIN_RAY_SAMPLE_COUNT = 4
MIN_RAY_SAMPLE_SPAN_METERS = 100.0
MAX_RAY_SAMPLE_CAPACITY = 16
MIN_PLANAR_SPAN_FOR_CHORD_FIT_METERS = 500.0
RAY_SAMPLE_FIELDS = (
    "distance_m", "pixel_x", "pixel_y", "latitude_rad", "longitude_rad",
    "row", "column", "terrain_height_m",
)


@dataclass(frozen=True, slots=True)
class ProjectionParameters:
    radius_m: float
    latitude_origin_rad: float
    longitude_origin_rad: float
    scale: float
    false_easting_m: float
    false_northing_m: float

    @classmethod
    def from_array(cls, values: npt.ArrayLike) -> ProjectionParameters:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ContractValidationError("projection parameters must be six finite values")
        return cls(*(float(value) for value in array))


@dataclass(frozen=True, slots=True)
class DemGrid:
    elevation_m: npt.NDArray[np.float32]
    geo_transform: npt.NDArray[np.float64]
    projection: ProjectionParameters

    def __post_init__(self) -> None:
        if (
            not isinstance(self.elevation_m, np.ndarray)
            or self.elevation_m.dtype != DEVICE_FLOAT_DTYPE
            or self.elevation_m.ndim != 2
            or not self.elevation_m.flags.c_contiguous
        ):
            raise ContractValidationError("DEM elevation must be C-contiguous float32[y, x]")
        if (
            not isinstance(self.geo_transform, np.ndarray)
            or self.geo_transform.dtype != np.dtype("<f8")
            or self.geo_transform.shape != (6,)
            or not self.geo_transform.flags.c_contiguous
            or not np.all(np.isfinite(self.geo_transform))
        ):
            raise ContractValidationError("DEM geotransform must be six C-contiguous float64 values")
        determinant = (
            self.geo_transform[1] * self.geo_transform[5]
            - self.geo_transform[2] * self.geo_transform[4]
        )
        if abs(determinant) < 1e-12:
            raise ContractValidationError("DEM geotransform is singular")

    @property
    def height(self) -> int:
        return self.elevation_m.shape[0]

    @property
    def width(self) -> int:
        return self.elevation_m.shape[1]

    @property
    def map_resolution_m(self) -> float:
        transform = self.geo_transform
        column = np.hypot(transform[1], transform[4])
        row = np.hypot(transform[2], transform[5])
        return float((column + row) * 0.5)

    def pixel_to_crs(self, column: float, row: float) -> tuple[float, float]:
        transform = self.geo_transform
        return (
            float(transform[0] + transform[1] * column + transform[2] * row),
            float(transform[3] + transform[4] * column + transform[5] * row),
        )

    def crs_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        transform = self.geo_transform
        determinant = transform[1] * transform[5] - transform[2] * transform[4]
        dx, dy = x - transform[0], y - transform[3]
        column = (transform[5] * dx - transform[2] * dy) / determinant
        row = (-transform[4] * dx + transform[1] * dy) / determinant
        return float(column), float(row)

    def lon_lat_to_pixel(self, longitude_rad: float, latitude_rad: float) -> tuple[float, float]:
        x, y = project_stereographic(latitude_rad, longitude_rad, self.projection)
        return self.crs_to_pixel(x, y)

    def elevation(self, column: float, row: float) -> float:
        x1, y1 = int(column), int(row)
        x2, y2 = min(x1 + 1, self.width - 1), min(y1 + 1, self.height - 1)
        q11 = float(self.elevation_m[y1, x1])
        q12 = float(self.elevation_m[y2, x1])
        q21 = float(self.elevation_m[y1, x2])
        q22 = float(self.elevation_m[y2, x2])
        return (
            q11 * (x2 - column) * (y2 - row)
            + q21 * (column - x1) * (y2 - row)
            + q12 * (x2 - column) * (row - y1)
            + q22 * (column - x1) * (row - y1)
        )


@dataclass(frozen=True, slots=True)
class DemSegmentContext:
    dem: DemGrid
    map_resolution_m: float
    ray_limit_m: float


@dataclass(frozen=True, slots=True)
class GridConvergenceInput:
    gamma_center_rad: float
    d_gamma_dx_rad_per_pixel: float
    d_gamma_dy_rad_per_pixel: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(tuple(self))):
            raise ContractValidationError("grid convergence input must be finite")

    def __iter__(self):
        yield self.gamma_center_rad
        yield self.d_gamma_dx_rad_per_pixel
        yield self.d_gamma_dy_rad_per_pixel


@dataclass(frozen=True, slots=True)
class SubpatchCenter:
    index: int
    grid_row: int
    grid_column: int
    requested_center_column: int
    requested_center_row: int
    segment_center_column: int
    segment_center_row: int


def inverse_stereographic(
    x: float, y: float, parameters: ProjectionParameters
) -> tuple[float, float]:
    xp = x - parameters.false_easting_m
    yp = y - parameters.false_northing_m
    rho = np.hypot(xp, yp)
    if rho < 1e-9:
        return parameters.latitude_origin_rad, parameters.longitude_origin_rad
    c = 2.0 * np.arctan2(rho, 2.0 * parameters.scale * parameters.radius_m)
    sin_c, cos_c = np.sin(c), np.cos(c)
    sin_origin, cos_origin = (
        np.sin(parameters.latitude_origin_rad),
        np.cos(parameters.latitude_origin_rad),
    )
    latitude = np.arcsin(
        cos_c * sin_origin + yp * sin_c * cos_origin / rho
    )
    longitude = parameters.longitude_origin_rad + np.arctan2(
        xp * sin_c,
        rho * cos_origin * cos_c - yp * sin_origin * sin_c,
    )
    return float(latitude), float(longitude)


def project_stereographic(
    latitude_rad: float, longitude_rad: float, parameters: ProjectionParameters
) -> tuple[float, float]:
    sin_lat, cos_lat = np.sin(latitude_rad), np.cos(latitude_rad)
    sin_origin, cos_origin = (
        np.sin(parameters.latitude_origin_rad),
        np.cos(parameters.latitude_origin_rad),
    )
    delta = longitude_rad - parameters.longitude_origin_rad
    denominator = 1.0 + sin_origin * sin_lat + cos_origin * cos_lat * np.cos(delta)
    if abs(denominator) < 1e-10:
        denominator = 1e-10
    scale = 2.0 * parameters.scale * parameters.radius_m / denominator
    x = scale * cos_lat * np.sin(delta) + parameters.false_easting_m
    y = scale * (cos_origin * sin_lat - sin_origin * cos_lat * np.cos(delta))
    y += parameters.false_northing_m
    return float(x), float(y)


def lat_lon_to_vector(
    latitude_rad: float, longitude_rad: float, radius_m: float
) -> npt.NDArray[np.float64]:
    cos_lat = np.cos(latitude_rad)
    return np.array(
        (
            radius_m * cos_lat * np.cos(longitude_rad),
            radius_m * cos_lat * np.sin(longitude_rad),
            radius_m * np.sin(latitude_rad),
        ),
        dtype=np.float64,
    )


def vector_to_lat_lon(vector: npt.ArrayLike) -> tuple[float, float]:
    x, y, z = np.asarray(vector, dtype=np.float64)
    longitude = np.arctan2(y, x)
    if longitude < 0.0:
        longitude += 2.0 * np.pi
    latitude = np.arctan2(z, np.hypot(x, y))
    return float(latitude), float(longitude)


def enu_to_moon_matrix(latitude_rad: float, longitude_rad: float) -> npt.NDArray[np.float64]:
    cos_lat, sin_lat = np.cos(latitude_rad), np.sin(latitude_rad)
    cos_lon, sin_lon = np.cos(longitude_rad), np.sin(longitude_rad)
    east = (-sin_lon, cos_lon, 0.0)
    north = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
    up = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
    return np.ascontiguousarray((east, north, up), dtype=np.float64)


def azimuth_direction(
    enu_to_moon: npt.NDArray[np.float64], azimuth_rad: float
) -> npt.NDArray[np.float64]:
    angle = np.pi / 2.0 - azimuth_rad
    local = np.array((np.cos(angle), np.sin(angle), 0.0), dtype=np.float64)
    direction = local @ enu_to_moon
    return np.ascontiguousarray(direction / np.linalg.norm(direction))


def try_sample_chord(
    observer_vector_m: npt.NDArray[np.float64],
    direction_moon_centered: npt.NDArray[np.float64],
    distance_m: float,
    dem: DemGrid,
) -> tuple[bool, npt.NDArray[np.float64]]:
    sample_vector = observer_vector_m + direction_moon_centered * distance_m
    latitude, longitude = vector_to_lat_lon(sample_vector)
    column, row = dem.lon_lat_to_pixel(longitude, latitude)
    inside = 0.0 <= column < dem.width and 0.0 <= row < dem.height
    terrain = dem.elevation(column, row) if inside else 0.0
    sample = np.array(
        (distance_m, column, row, latitude, longitude, row, column, terrain),
        dtype=np.float64,
    )
    return inside, sample


def build_ray_samples(
    observer_vector_m: npt.NDArray[np.float64],
    direction_moon_centered: npt.NDArray[np.float64],
    start_distance_m: float,
    maximum_distance_m: float,
    dem: DemGrid,
) -> npt.NDArray[np.float64]:
    inside, start = try_sample_chord(
        observer_vector_m, direction_moon_centered, start_distance_m, dem
    )
    if not inside:
        return np.empty((0, len(RAY_SAMPLE_FIELDS)), dtype=np.float64)
    samples = [start]
    final_inside = start
    inside, end = try_sample_chord(
        observer_vector_m, direction_moon_centered, maximum_distance_m, dem
    )
    if inside:
        final_inside = end
    else:
        low, high = start_distance_m, maximum_distance_m
        best = start
        for _ in range(24):
            middle = 0.5 * (low + high)
            inside, sample = try_sample_chord(
                observer_vector_m, direction_moon_centered, middle, dem
            )
            if inside:
                low, best = middle, sample
            else:
                high = middle
        final_inside = best

    span = max(0.0, final_inside[0] - start_distance_m)
    if span > 1e-3:
        for index in range(1, 11):
            target = start_distance_m + span * index / 10.0
            if target > final_inside[0] - 0.5:
                break
            if target - samples[-1][0] < 0.5:
                continue
            inside, sample = try_sample_chord(
                observer_vector_m, direction_moon_centered, target, dem
            )
            if inside:
                samples.append(sample)
        if len(samples) < MAX_RAY_SAMPLE_CAPACITY and final_inside[0] - samples[-1][0] > 0.5:
            samples.append(final_inside)

    required_end = samples[0][0] + MIN_RAY_SAMPLE_SPAN_METERS
    if len(samples) < MIN_RAY_SAMPLE_COUNT or samples[-1][0] < required_end:
        step = max(10.0, dem.map_resolution_m * 2.0)
        current = samples[-1][0]
        extend_limit = max(final_inside[0], required_end)
        while (
            (len(samples) < MIN_RAY_SAMPLE_COUNT or samples[-1][0] < required_end)
            and current < extend_limit
            and len(samples) < MAX_RAY_SAMPLE_CAPACITY
        ):
            current = min(extend_limit, current + step)
            inside, sample = try_sample_chord(
                observer_vector_m, direction_moon_centered, current, dem
            )
            if not inside or current - samples[-1][0] < 0.5:
                break
            samples.append(sample)
    return np.ascontiguousarray(samples, dtype=np.float64)


def _solve_linear_system(matrix: npt.NDArray[np.float64], rhs: npt.NDArray[np.float64]):
    size = len(rhs)
    augmented = np.empty((size, size + 1), dtype=np.float64)
    augmented[:, :size] = matrix
    augmented[:, size] = rhs
    for index in range(size):
        pivot = index + int(np.argmax(np.abs(augmented[index:, index])))
        maximum = abs(augmented[pivot, index])
        if maximum < 1e-12:
            return None
        if pivot != index:
            augmented[[index, pivot], index:] = augmented[[pivot, index], index:]
        augmented[index, index:] /= augmented[index, index]
        for row in range(size):
            if row == index:
                continue
            factor = augmented[row, index]
            if abs(factor) < 1e-12:
                continue
            augmented[row, index:] -= factor * augmented[index, index:]
    return augmented[:, size].copy()


def fit_quartic_no_intercept(
    parameter: npt.ArrayLike, values: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    s = np.asarray(parameter, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    powers = np.column_stack(tuple(s**power for power in range(1, 5)))
    solution = _solve_linear_system(powers.T @ powers, powers.T @ v)
    return np.zeros(4, dtype=np.float64) if solution is None else solution


def fit_cubic_no_intercept(
    parameter: npt.ArrayLike, values: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    x = np.asarray(parameter, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    powers = np.column_stack((x, x * x, x * x * x))
    solution = _solve_linear_system(powers.T @ powers, powers.T @ y)
    return np.array((1.0, 0.0, 0.0)) if solution is None else solution


def chord_distance_on_sphere(
    observer_vector_m: npt.NDArray[np.float64],
    latitude_rad: float,
    longitude_rad: float,
    sphere_radius_m: float,
) -> float:
    surface = lat_lon_to_vector(latitude_rad, longitude_rad, sphere_radius_m)
    return float(np.linalg.norm(surface - observer_vector_m))


def fit_planar_to_chord(
    samples: npt.NDArray[np.float64],
    map_resolution_m: float,
    observer_vector_m: npt.NDArray[np.float64],
    sphere_radius_m: float,
) -> npt.NDArray[np.float64]:
    result = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if len(samples) < 2 or map_resolution_m <= 0.0:
        return result
    x0, y0 = samples[0, 1], samples[0, 2]
    chord0 = chord_distance_on_sphere(
        observer_vector_m, samples[0, 3], samples[0, 4], sphere_radius_m
    )
    planar, chord = [0.0], [0.0]
    maximum_planar = 0.0
    for sample in samples[1:MAX_RAY_SAMPLE_CAPACITY]:
        if sample[1] < 0.0 or sample[2] < 0.0:
            continue
        distance = np.hypot(
            (sample[1] - x0) * map_resolution_m,
            (sample[2] - y0) * map_resolution_m,
        )
        planar.append(float(distance))
        chord.append(
            chord_distance_on_sphere(
                observer_vector_m, sample[3], sample[4], sphere_radius_m
            ) - chord0
        )
        maximum_planar = max(maximum_planar, float(distance))
    if len(planar) < 2 or maximum_planar < 1e-4:
        return result
    if maximum_planar < MIN_PLANAR_SPAN_FOR_CHORD_FIT_METERS:
        return result
    result = fit_cubic_no_intercept(planar, chord)
    if result[0] < 0.5 or result[0] > 2.0:
        return np.array((1.0, 0.0, 0.0), dtype=np.float64)
    return result


def fit_ray_segment(
    samples: npt.NDArray[np.float64],
    map_resolution_m: float,
    observer_vector_m: npt.NDArray[np.float64],
    correction_sphere_radius_m: float,
    fallback_start_m: float,
) -> npt.NDArray[np.float32]:
    if len(samples) < 3:
        x = samples[0, 1] if len(samples) else 0.0
        y = samples[0, 2] if len(samples) else 0.0
        start = samples[0, 0] / 1000.0 if len(samples) else fallback_start_m / 1000.0
        return np.array(
            (x, y, x, y, 0, 0, 0, 0, 0, 0, 0, 0, start, start, start, 1, 0, 0),
            dtype=np.float32,
        )
    x0, y0 = samples[0, 1], samples[0, 2]
    anchor_km = samples[0, 0] / 1000.0
    end_km = samples[-1, 0] / 1000.0
    span_km = max(0.001, end_km - anchor_km)
    normalized = (samples[:, 0] / 1000.0 - anchor_km) / span_km
    x_coefficients = fit_quartic_no_intercept(normalized, samples[:, 1] - x0)
    y_coefficients = fit_quartic_no_intercept(normalized, samples[:, 2] - y0)
    inverse = 1.0 / span_km
    scales = np.array((inverse, inverse**2, inverse**3, inverse**4))
    x_coefficients *= scales
    y_coefficients *= scales
    chord = fit_planar_to_chord(
        samples, map_resolution_m, observer_vector_m, correction_sphere_radius_m
    )
    return np.array(
        (x0, y0, x0, y0, *x_coefficients, *y_coefficients,
         anchor_km, end_km, anchor_km, *chord),
        dtype=np.float32,
    )


def build_dem_segment_contexts(
    dems: Iterable[DemGrid], maximum_distance_m: float
) -> tuple[DemSegmentContext, ...]:
    contexts = []
    for dem in dems:
        resolution = dem.map_resolution_m
        size_m = min(dem.width * resolution, dem.height * resolution)
        contexts.append(
            DemSegmentContext(dem, resolution, min(maximum_distance_m, size_m * 1.2))
        )
    return tuple(contexts)


def clamp_subpatch_center(requested: int, dem_size: int, subpatch_size: int) -> int:
    half = subpatch_size // 2
    return min(max(requested, half), max(half, dem_size - half))


def build_subpatch_centers(
    *, tile_column: int, tile_row: int, tile_width: int,
    subpatch_size: int, primary_width: int, primary_height: int,
) -> tuple[SubpatchCenter, ...]:
    count = tile_width // subpatch_size + 2
    centers = []
    for index in range(count * count):
        grid_row, grid_column = divmod(index, count)
        requested_column = tile_column + (grid_column - 1) * subpatch_size + subpatch_size // 2
        requested_row = tile_row + (grid_row - 1) * subpatch_size + subpatch_size // 2
        centers.append(
            SubpatchCenter(
                index, grid_row, grid_column, requested_column, requested_row,
                clamp_subpatch_center(requested_column, primary_width, subpatch_size),
                clamp_subpatch_center(requested_row, primary_height, subpatch_size),
            )
        )
    return tuple(centers)


class SubpatchSegmentCache:
    """Deterministic per-center segment cache; no compiled dictionary is assumed."""

    def __init__(
        self, dems: Iterable[DemGrid], *, azimuth_count: int,
        maximum_distance_m: float, observer_elevation_m: float,
    ) -> None:
        self.dems = tuple(dems)
        if not self.dems:
            raise ContractValidationError("at least one DEM is required")
        self.azimuth_count = azimuth_count
        self.maximum_distance_m = maximum_distance_m
        self.observer_elevation_m = observer_elevation_m
        self.contexts = build_dem_segment_contexts(self.dems, maximum_distance_m)
        self._segments: dict[tuple[int, int], npt.NDArray[np.float32]] = {}

    def get(self, center_column: int, center_row: int) -> npt.NDArray[np.float32]:
        key = center_column, center_row
        if key not in self._segments:
            self._segments[key] = self._compute(center_column, center_row)
        return self._segments[key]

    def _compute(self, center_column: int, center_row: int) -> npt.NDArray[np.float32]:
        primary = self.dems[0]
        x, y = primary.pixel_to_crs(center_column, center_row)
        latitude, longitude = inverse_stereographic(x, y, primary.projection)
        terrain_column = min(max(float(center_column), 0.0), primary.width - 1.001)
        terrain_row = min(max(float(center_row), 0.0), primary.height - 1.001)
        center_terrain = primary.elevation(terrain_column, terrain_row)
        observer = lat_lon_to_vector(
            latitude, longitude,
            primary.projection.radius_m + center_terrain + self.observer_elevation_m,
        )
        rotation = enu_to_moon_matrix(latitude, longitude)
        result = np.zeros(
            (self.azimuth_count, len(self.dems), len(SEGMENT_FIELDS)), dtype=np.float32
        )
        for azimuth_index in range(self.azimuth_count):
            direction = azimuth_direction(
                rotation, azimuth_index * 2.0 * np.pi / self.azimuth_count
            )
            start_distance = 1.0
            for dem_index, context in enumerate(self.contexts):
                samples = build_ray_samples(
                    observer, direction, start_distance, context.ray_limit_m, context.dem
                )
                result[azimuth_index, dem_index] = fit_ray_segment(
                    samples, context.map_resolution_m, observer,
                    primary.projection.radius_m + center_terrain, start_distance,
                )
                last_distance = samples[-1, 0] if len(samples) else start_distance
                start_distance = min(context.ray_limit_m, float(last_distance))
                if start_distance >= self.maximum_distance_m:
                    break
        return result


def build_subpatch_segments(
    dems: Iterable[DemGrid], *, tile_column: int, tile_row: int,
    tile_width: int, azimuth_count: int, maximum_distance_m: float,
    observer_elevation_m: float, subpatch_size: int,
    grid_convergence: GridConvergenceInput,
) -> tuple[npt.NDArray[np.float32], tuple[SubpatchCenter, ...], GridConvergenceInput]:
    dem_tuple = tuple(dems)
    centers = build_subpatch_centers(
        tile_column=tile_column, tile_row=tile_row, tile_width=tile_width,
        subpatch_size=subpatch_size, primary_width=dem_tuple[0].width,
        primary_height=dem_tuple[0].height,
    )
    cache = SubpatchSegmentCache(
        dem_tuple, azimuth_count=azimuth_count,
        maximum_distance_m=maximum_distance_m,
        observer_elevation_m=observer_elevation_m,
    )
    result = np.empty(
        (azimuth_count, len(centers), len(dem_tuple), len(SEGMENT_FIELDS)),
        dtype=np.float32,
    )
    for center in centers:
        result[:, center.index] = cache.get(
            center.segment_center_column, center.segment_center_row
        )
    return result, centers, grid_convergence


def build_subpatch_segments_numba(
    dems: Iterable[DemGrid], *, tile_column: int, tile_row: int,
    tile_width: int, azimuth_count: int, maximum_distance_m: float,
    observer_elevation_m: float, subpatch_size: int,
    grid_convergence: GridConvergenceInput, parallel: bool = True,
) -> tuple[npt.NDArray[np.float32], tuple[SubpatchCenter, ...], GridConvergenceInput]:
    """Build the complete dense segment tensor with optional Numba CPU kernels.

    Importing ordinary Lunarscout or :mod:`geometry` does not import Numba. The
    prototype dependency is loaded only when this explicitly experimental
    function is called.
    """
    from .geometry_numba import generate_segments_parallel, generate_segments_serial

    dem_tuple = tuple(dems)
    if not dem_tuple:
        raise ContractValidationError("at least one DEM is required")
    centers = build_subpatch_centers(
        tile_column=tile_column, tile_row=tile_row, tile_width=tile_width,
        subpatch_size=subpatch_size, primary_width=dem_tuple[0].width,
        primary_height=dem_tuple[0].height,
    )
    unique_keys = tuple(dict.fromkeys(
        (center.segment_center_column, center.segment_center_row) for center in centers
    ))
    key_to_index = {key: index for index, key in enumerate(unique_keys)}
    primary = dem_tuple[0]
    observers = np.empty((len(unique_keys), 3), dtype=np.float64)
    correction_radii = np.empty(len(unique_keys), dtype=np.float64)
    directions = np.empty((len(unique_keys), azimuth_count, 3), dtype=np.float64)
    for center_index, (center_column, center_row) in enumerate(unique_keys):
        x, y = primary.pixel_to_crs(center_column, center_row)
        latitude, longitude = inverse_stereographic(x, y, primary.projection)
        terrain_column = min(max(float(center_column), 0.0), primary.width - 1.001)
        terrain_row = min(max(float(center_row), 0.0), primary.height - 1.001)
        terrain = primary.elevation(terrain_column, terrain_row)
        correction_radii[center_index] = primary.projection.radius_m + terrain
        observers[center_index] = lat_lon_to_vector(
            latitude, longitude,
            correction_radii[center_index] + observer_elevation_m,
        )
        rotation = enu_to_moon_matrix(latitude, longitude)
        for azimuth_index in range(azimuth_count):
            directions[center_index, azimuth_index] = azimuth_direction(
                rotation, azimuth_index * 2.0 * np.pi / azimuth_count
            )

    job_count = len(unique_keys) * azimuth_count
    job_observers = np.repeat(observers, azimuth_count, axis=0)
    job_directions = directions.reshape(job_count, 3)
    job_radii = np.repeat(correction_radii, azimuth_count)
    starts = np.ones(job_count, dtype=np.float64)
    unique_result = np.zeros(
        (azimuth_count, len(unique_keys), len(dem_tuple), len(SEGMENT_FIELDS)),
        dtype=np.float32,
    )
    contexts = build_dem_segment_contexts(dem_tuple, maximum_distance_m)
    generator = generate_segments_parallel if parallel else generate_segments_serial
    for dem_index, context in enumerate(contexts):
        active = np.flatnonzero(starts < maximum_distance_m)
        if not len(active):
            break
        projection = np.array(
            (
                context.dem.projection.radius_m,
                context.dem.projection.latitude_origin_rad,
                context.dem.projection.longitude_origin_rad,
                context.dem.projection.scale,
                context.dem.projection.false_easting_m,
                context.dem.projection.false_northing_m,
            ),
            dtype=np.float64,
        )
        fitted = generator(
            context.dem.elevation_m,
            context.dem.geo_transform,
            projection,
            job_observers[active],
            job_directions[active],
            starts[active],
            np.full(len(active), context.ray_limit_m, dtype=np.float64),
            context.map_resolution_m,
            job_radii[active],
        )
        active_centers = active // azimuth_count
        active_azimuths = active % azimuth_count
        unique_result[active_azimuths, active_centers, dem_index] = fitted
        last_distance = fitted[:, 13].astype(np.float64) * 1000.0
        starts[active] = np.minimum(context.ray_limit_m, last_distance)

    result = np.empty(
        (azimuth_count, len(centers), len(dem_tuple), len(SEGMENT_FIELDS)),
        dtype=np.float32,
    )
    for center in centers:
        result[:, center.index] = unique_result[
            :, key_to_index[(center.segment_center_column, center.segment_center_row)]
        ]
    return result, centers, grid_convergence
