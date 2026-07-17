"""Independent direct-vector ray calculation used to audit fitted CUDA rays."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .geometry import (
    DemGrid,
    enu_to_moon_matrix,
    lat_lon_to_vector,
    project_stereographic,
    vector_to_lat_lon,
)


def direct_reference_trace(
    dem: DemGrid,
    observer_vector_m: npt.ArrayLike,
    direction_moon_centered: npt.ArrayLike,
    distances_m: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate the C# ReferenceRayEmulator base-ray geometry directly.

    Returned fields are distance_m, pixel_x, pixel_y, elevation_m, and exact
    spherical slope. No fitted ray segment or production near-field shortcut is
    used.
    """
    observer = np.asarray(observer_vector_m, dtype=np.float64)
    direction = np.asarray(direction_moon_centered, dtype=np.float64)
    observer_latitude, observer_longitude = vector_to_lat_lon(observer)
    moon_to_observer = enu_to_moon_matrix(observer_latitude, observer_longitude)
    rows = []
    for distance in np.asarray(distances_m, dtype=np.float64):
        walker = observer + direction * distance
        latitude, longitude = vector_to_lat_lon(walker)
        x, y = project_stereographic(latitude, longitude, dem.projection)
        column, row = dem.crs_to_pixel(x, y)
        if not (0.0 <= column < dem.width and 0.0 <= row < dem.height):
            break
        elevation = dem.elevation(column, row)
        surface = lat_lon_to_vector(
            latitude, longitude, dem.projection.radius_m + elevation
        )
        local = moon_to_observer @ (surface - observer)
        horizontal = np.hypot(local[0], local[1])
        slope = local[2] / horizontal
        rows.append((distance, column, row, elevation, slope))
    return np.ascontiguousarray(rows, dtype=np.float64)
