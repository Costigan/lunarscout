"""Compiled, bounded CPU backend for private byte-valued lightmaps."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .geometry import DemGrid
from .lightmap import _HALF_CIRCLE, _MAX_PHOTONS
from .psr import _validate_vectors


_CPU_FUNCTION = None
_CPU_FUNCTION_LOCK = threading.Lock()


def _build_cpu_function():
    from numba import njit, prange

    def calculate(
        patch_dem,
        geotransform,
        projection,
        vectors,
        horizons,
        tile_x,
        tile_y,
        valid_width,
        valid_height,
    ):
        output = np.zeros(
            (vectors.shape[0], PATCH_SIZE, PATCH_SIZE), dtype=np.uint8
        )
        for index in prange(PATCH_SIZE * PATCH_SIZE):
            sample = index % PATCH_SIZE
            line = index // PATCH_SIZE
            if sample >= valid_width or line >= valid_height:
                continue
            absolute_sample = tile_x + sample
            absolute_line = tile_y + line
            crs_x = (
                geotransform[0]
                + geotransform[1] * absolute_sample
                + geotransform[2] * absolute_line
            )
            crs_y = (
                geotransform[3]
                + geotransform[4] * absolute_sample
                + geotransform[5] * absolute_line
            )
            xp = crs_x - projection[4]
            yp = crs_y - projection[5]
            rho = math.sqrt(xp * xp + yp * yp)
            if rho <= 1e-12:
                longitude = projection[2]
                latitude = projection[1]
            else:
                c = 2.0 * math.atan2(rho, 2.0 * projection[3] * projection[0])
                sin_c = math.sin(c)
                cos_c = math.cos(c)
                cos_latitude_origin = math.cos(projection[1])
                sin_latitude_origin = math.sin(projection[1])
                latitude = math.asin(
                    cos_c * sin_latitude_origin
                    + yp * sin_c * cos_latitude_origin / rho
                )
                longitude = projection[2] + math.atan2(
                    xp * sin_c,
                    rho * cos_latitude_origin * cos_c
                    - yp * sin_latitude_origin * sin_c,
                )
            radius = projection[0] + float(patch_dem[line, sample])
            cos_latitude = math.cos(latitude)
            sin_latitude = math.sin(latitude)
            cos_longitude = math.cos(longitude)
            sin_longitude = math.sin(longitude)
            moon_x = radius * cos_latitude * cos_longitude
            moon_y = radius * cos_latitude * sin_longitude
            moon_z = radius * sin_latitude
            up_x = cos_latitude * cos_longitude
            up_y = cos_latitude * sin_longitude
            up_z = sin_latitude
            east_x = -sin_longitude
            east_y = cos_longitude
            north_x = -sin_latitude * cos_longitude
            north_y = -sin_latitude * sin_longitude
            north_z = cos_latitude
            tx = np.float32(-(moon_x * east_x + moon_y * east_y))
            ty = np.float32(
                -(moon_x * north_x + moon_y * north_y + moon_z * north_z)
            )
            tz = np.float32(-(moon_x * up_x + moon_y * up_y + moon_z * up_z))
            r00, r01, r02 = np.float32(east_x), np.float32(north_x), np.float32(up_x)
            r10, r11, r12 = np.float32(east_y), np.float32(north_y), np.float32(up_y)
            r20, r21, r22 = np.float32(0.0), np.float32(north_z), np.float32(up_z)

            for time_index in range(vectors.shape[0]):
                vx, vy, vz = vectors[time_index]
                enu_x = np.float32(vx * r00 + vy * r10 + vz * r20 + tx)
                enu_y = np.float32(vx * r01 + vy * r11 + vz * r21 + ty)
                enu_z = np.float32(vx * r02 + vy * r12 + vz * r22 + tz)
                horizontal = np.float32(
                    math.sqrt(np.float32(enu_x * enu_x + enu_y * enu_y))
                )
                elevation = np.float32(
                    math.atan2(enu_z, horizontal) * np.float32(57.29577951308232)
                )
                azimuth = np.float32(math.atan2(enu_x, enu_y))
                if azimuth < np.float32(0.0):
                    azimuth = np.float32(azimuth + np.float32(math.pi * 2.0))
                azimuth_deg = np.float32(
                    azimuth * np.float32(57.29577951308232)
                )
                left_position = np.float32(
                    (azimuth_deg - np.float32(0.27) - np.float32(0.125))
                    * np.float32(4.0)
                )
                left = int(left_position)
                fraction = np.float32(left_position - np.float32(left))
                if left < 0:
                    left += AZIMUTH_COUNT
                elif left >= AZIMUTH_COUNT:
                    left -= AZIMUTH_COUNT
                right = left + 1
                if right >= AZIMUTH_COUNT:
                    right = 0
                left_elevation = horizons[line, sample, left]
                right_elevation = horizons[line, sample, right]
                delta = np.float32(right_elevation - left_elevation)
                photons = np.float32(0.0)
                for slice_index in range(16):
                    horizon = np.float32(fraction * delta + left_elevation)
                    column = _HALF_CIRCLE[slice_index]
                    sun_top = np.float32(elevation + column)
                    if horizon < sun_top:
                        angle_delta = np.float32(sun_top - horizon)
                        column_height = np.float32(column + column)
                        if angle_delta > column_height:
                            angle_delta = column_height
                        photons = np.float32(photons + angle_delta)
                    fraction = np.float32(fraction + np.float32(0.135))
                    if fraction >= np.float32(1.0):
                        left = right
                        right += 1
                        if right >= AZIMUTH_COUNT:
                            right = 0
                        left_elevation = right_elevation
                        right_elevation = horizons[line, sample, right]
                        delta = np.float32(right_elevation - left_elevation)
                        fraction = np.float32(fraction - np.float32(1.0))
                visible = np.float32(photons / np.float32(_MAX_PHOTONS))
                output[time_index, line, sample] = int(
                    np.float32(255.0) * visible
                )
        return output

    try:
        return njit(parallel=True, cache=True)(calculate)
    except RuntimeError as error:
        if "no locator available" not in str(error):
            raise
        return njit(parallel=True)(calculate)


class LightmapCpuSession:
    """Numba-parallel CPU calculation with bounded time-batch output."""

    def __init__(self, *, time_batch_size: int = 32) -> None:
        global _CPU_FUNCTION
        if time_batch_size < 1:
            raise ValueError("time_batch_size must be positive")
        with _CPU_FUNCTION_LOCK:
            if _CPU_FUNCTION is None:
                _CPU_FUNCTION = _build_cpu_function()
        self._calculate = _CPU_FUNCTION
        self.time_batch_size = int(time_batch_size)

    def iter_patch_tiles(self, dem: DemGrid, horizons_deg: npt.ArrayLike, sun_vectors_m: npt.ArrayLike, *, tile_y: int, tile_x: int, valid_height: int = PATCH_SIZE, valid_width: int = PATCH_SIZE) -> Iterator[npt.NDArray[np.uint8]]:
        vectors = np.ascontiguousarray(_validate_vectors(sun_vectors_m), dtype=np.float32)
        horizons = np.ascontiguousarray(horizons_deg, dtype=np.float32)
        if horizons.shape != (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT):
            raise ValueError("horizons must have shape (128, 128, 1440)")
        patch_dem = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        patch_dem[:valid_height, :valid_width] = dem.elevation_m[tile_y:tile_y + valid_height, tile_x:tile_x + valid_width]
        projection = np.asarray((dem.projection.radius_m, dem.projection.latitude_origin_rad, dem.projection.longitude_origin_rad, dem.projection.scale, dem.projection.false_easting_m, dem.projection.false_northing_m), dtype=np.float64)
        for start in range(0, vectors.shape[0], self.time_batch_size):
            batch = self._calculate(patch_dem, dem.geo_transform, projection, vectors[start:start + self.time_batch_size], horizons, tile_x, tile_y, valid_width, valid_height)
            for tile in batch:
                yield np.ascontiguousarray(tile[:valid_height, :valid_width])
