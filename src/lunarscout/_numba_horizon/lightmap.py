"""Private reference calculation for byte-valued time-series lightmaps."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .geometry import DemGrid
from .psr import _azimuth_elevation_deg, _pixel_frame, _validate_vectors


SUN_HALF_ANGLE_DEG = np.float32(0.27)
_BUCKET_HALF_WIDTH_DEG = np.float32(0.125)
_SLICE_COUNT = 16
_SLICE_STEP_DEG = np.float32(0.135)
_HALF_CIRCLE = np.asarray(
    [
        np.sqrt(64.0 - (7.5 - index) ** 2) / 8.0 * float(SUN_HALF_ANGLE_DEG)
        for index in range(_SLICE_COUNT)
    ],
    dtype=np.float32,
)
_photon_sum = np.float32(0.0)
for _column_height in _HALF_CIRCLE:
    _photon_sum = np.float32(_photon_sum + _column_height)
_MAX_PHOTONS = np.float32(np.float32(2.0) * _photon_sum)


def _sun_fraction_reference(
    horizon_deg: npt.NDArray[np.float32], azimuth_deg: float, elevation_deg: float
) -> np.float32:
    """Reproduce C# ``BuilderSunFraction`` including float32 accumulation."""
    left_position = np.float32(
        (np.float32(azimuth_deg) - SUN_HALF_ANGLE_DEG - _BUCKET_HALF_WIDTH_DEG)
        * np.float32(4.0)
    )
    # C# float-to-int conversion truncates toward zero, including below zero.
    left = int(left_position)
    fraction = np.float32(left_position - np.float32(left))
    if left < 0:
        left += AZIMUTH_COUNT
    elif left >= AZIMUTH_COUNT:
        left -= AZIMUTH_COUNT
    right = left + 1
    if right >= AZIMUTH_COUNT:
        right = 0
    left_elevation = horizon_deg[left]
    right_elevation = horizon_deg[right]
    delta = np.float32(right_elevation - left_elevation)
    photons = np.float32(0.0)
    for sun_column in _HALF_CIRCLE:
        horizon_elevation = np.float32(fraction * delta + left_elevation)
        sun_top = np.float32(np.float32(elevation_deg) + sun_column)
        if horizon_elevation < sun_top:
            angle_delta = np.float32(sun_top - horizon_elevation)
            column_height = np.float32(sun_column + sun_column)
            if angle_delta > column_height:
                angle_delta = column_height
            photons = np.float32(photons + angle_delta)
        fraction = np.float32(fraction + _SLICE_STEP_DEG)
        if fraction >= np.float32(1.0):
            left = right
            right += 1
            if right >= AZIMUTH_COUNT:
                right = 0
            left_elevation = right_elevation
            right_elevation = horizon_deg[right]
            delta = np.float32(right_elevation - left_elevation)
            fraction = np.float32(fraction - np.float32(1.0))
    return np.float32(photons / _MAX_PHOTONS)


def iter_lightmap_patch_reference(
    dem: DemGrid,
    horizons_deg: npt.ArrayLike,
    sun_vectors_m: npt.ArrayLike,
    *,
    tile_y: int,
    tile_x: int,
    valid_height: int = PATCH_SIZE,
    valid_width: int = PATCH_SIZE,
) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield one 128-by-128 byte tile per vector without a time cube."""
    if not 1 <= valid_width <= PATCH_SIZE or not 1 <= valid_height <= PATCH_SIZE:
        raise ValueError("valid patch dimensions must be between 1 and 128")
    if (
        tile_x < 0
        or tile_y < 0
        or tile_x + valid_width > dem.width
        or tile_y + valid_height > dem.height
    ):
        raise ValueError("lightmap patch falls outside the DEM")
    horizons = np.asarray(horizons_deg, dtype=np.float32)
    if horizons.shape == (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT):
        horizons = horizons[:valid_height, :valid_width]
    if horizons.shape != (valid_height, valid_width, AZIMUTH_COUNT):
        raise ValueError(
            "horizons must have shape (valid_height, valid_width, 1440) "
            "or (128, 128, 1440)"
        )
    if not np.all(np.isfinite(horizons)):
        raise ValueError("horizons must contain finite elevations")
    vectors = _validate_vectors(sun_vectors_m)
    frames = [
        [
            _pixel_frame(dem, tile_y + local_y, tile_x + local_x)
            for local_x in range(valid_width)
        ]
        for local_y in range(valid_height)
    ]
    for vector in vectors:
        output = np.empty((valid_height, valid_width), dtype=np.uint8)
        vector_row = vector[None, :]
        for local_y in range(valid_height):
            for local_x in range(valid_width):
                rotation, translation = frames[local_y][local_x]
                azimuth, elevation = _azimuth_elevation_deg(
                    vector_row, rotation, translation
                )
                fraction = _sun_fraction_reference(
                    horizons[local_y, local_x], azimuth[0], elevation[0]
                )
                output[local_y, local_x] = np.uint8(
                    np.float32(255.0) * fraction
                )
        yield output


def compute_lightmap_patch_reference(
    dem: DemGrid,
    horizons_deg: npt.ArrayLike,
    sun_vectors_m: npt.ArrayLike,
    **patch: int,
) -> npt.NDArray[np.uint8]:
    """Materialize the reference iterator for small correctness tests only."""
    return np.stack(
        tuple(iter_lightmap_patch_reference(dem, horizons_deg, sun_vectors_m, **patch))
    )
