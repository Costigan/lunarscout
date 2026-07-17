"""Correctness-first private PSR reduction and patch calculation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .geometry import DemGrid, inverse_stereographic


SUN_ANGULAR_SIZE_DEG = np.float32(0.545)
PSR_VALUE = np.uint8(255)
NON_PSR_VALUE = np.uint8(0)


def _validate_vectors(vectors_m: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.ascontiguousarray(vectors_m, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1:] != (3,)
        or values.shape[0] == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Moon-ME vectors must be finite values shaped (time, 3)")
    return values


def _pixel_frame(
    dem: DemGrid, row: int, column: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    crs_x, crs_y = dem.pixel_to_crs(column, row)
    latitude, longitude = inverse_stereographic(crs_x, crs_y, dem.projection)
    cos_lat, sin_lat = np.cos(latitude), np.sin(latitude)
    cos_lon, sin_lon = np.cos(longitude), np.sin(longitude)
    up = np.asarray(
        (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), dtype=np.float64
    )
    east = np.asarray((-sin_lon, cos_lon, 0.0), dtype=np.float64)
    north = np.asarray(
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), dtype=np.float64
    )
    radius = dem.projection.radius_m + float(dem.elevation_m[row, column])
    observer = radius * up
    rotation = np.ascontiguousarray((east, north, up), dtype=np.float64).T
    translation = -(observer @ rotation)
    return rotation, translation


def _azimuth_elevation_deg(
    vectors_m: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    enu = vectors_m @ rotation + translation
    azimuth = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360.0
    elevation = np.degrees(np.arctan2(enu[:, 2], np.hypot(enu[:, 0], enu[:, 1])))
    return azimuth, elevation


def reduce_sun_vectors_for_psr(
    dem: DemGrid, sun_vectors_m: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Reproduce the C# five-viewpoint, 1,440-bin reduction heuristic."""
    vectors = _validate_vectors(sun_vectors_m)
    points = (
        (0, 0),
        (0, dem.width - 1),
        (dem.height - 1, 0),
        (dem.height - 1, dem.width - 1),
        (dem.height // 2, dem.width // 2),
    )
    selected = np.full((len(points), AZIMUTH_COUNT), -1, dtype=np.int64)
    maxima = np.full((len(points), AZIMUTH_COUNT), -np.inf, dtype=np.float64)
    for point_index, (row, column) in enumerate(points):
        rotation, translation = _pixel_frame(dem, row, column)
        azimuth, elevation = _azimuth_elevation_deg(vectors, rotation, translation)
        bins = (azimuth * (AZIMUTH_COUNT / 360.0)).astype(np.int64) % AZIMUTH_COUNT
        for vector_index, azimuth_bin in enumerate(bins):
            if elevation[vector_index] > maxima[point_index, azimuth_bin]:
                maxima[point_index, azimuth_bin] = elevation[vector_index]
                selected[point_index, azimuth_bin] = vector_index
    ordered_indices: list[int] = []
    seen: set[int] = set()
    for index in selected.ravel():
        value = int(index)
        if value >= 0 and value not in seen:
            seen.add(value)
            ordered_indices.append(value)
    indices = np.asarray(ordered_indices, dtype=np.int64)
    return np.ascontiguousarray(vectors[indices]), indices


def compute_psr_patch_reference(
    dem: DemGrid,
    horizons_deg: npt.ArrayLike,
    sun_vectors_m: npt.ArrayLike,
    *,
    tile_y: int,
    tile_x: int,
    valid_height: int = PATCH_SIZE,
    valid_width: int = PATCH_SIZE,
    sun_angular_size_deg: float = float(SUN_ANGULAR_SIZE_DEG),
) -> npt.NDArray[np.uint8]:
    """Calculate one PSR tile with the accepted C# upper-limb semantics."""
    if not 1 <= valid_width <= PATCH_SIZE or not 1 <= valid_height <= PATCH_SIZE:
        raise ValueError("valid patch dimensions must be between 1 and 128")
    if tile_x < 0 or tile_y < 0 or tile_x + valid_width > dem.width or tile_y + valid_height > dem.height:
        raise ValueError("PSR patch falls outside the DEM")
    horizons = np.asarray(horizons_deg, dtype=np.float32)
    if horizons.shape == (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT):
        horizons = horizons[:valid_height, :valid_width]
    if horizons.shape != (valid_height, valid_width, AZIMUTH_COUNT):
        raise ValueError(
            "horizons must have shape (valid_height, valid_width, 1440) or (128, 128, 1440)"
        )
    if not np.all(np.isfinite(horizons)):
        raise ValueError("horizons must contain finite elevations")
    vectors = _validate_vectors(sun_vectors_m)
    output = np.full((valid_height, valid_width), PSR_VALUE, dtype=np.uint8)
    limb_threshold = -float(sun_angular_size_deg) / 2.0
    for local_y in range(valid_height):
        for local_x in range(valid_width):
            rotation, translation = _pixel_frame(
                dem, tile_y + local_y, tile_x + local_x
            )
            azimuth, elevation = _azimuth_elevation_deg(
                vectors, rotation, translation
            )
            positions = azimuth * (AZIMUTH_COUNT / 360.0)
            lower = np.floor(positions).astype(np.int64) % AZIMUTH_COUNT
            upper = (lower + 1) % AZIMUTH_COUNT
            fractions = positions - np.floor(positions)
            pixel_horizon = horizons[local_y, local_x]
            interpolated = pixel_horizon[lower] + fractions * (
                pixel_horizon[upper] - pixel_horizon[lower]
            )
            if np.any(elevation - interpolated > limb_threshold):
                output[local_y, local_x] = NON_PSR_VALUE
    return output
