"""Bounded Numba CUDA backend for private byte-valued lightmaps."""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from .cuda_backend import CudaBackendError
from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .geometry import DemGrid
from .lightmap import _HALF_CIRCLE, _MAX_PHOTONS
from .psr import _validate_vectors


_LIGHTMAP_KERNEL = None
_LIGHTMAP_KERNEL_LOCK = threading.Lock()


def _build_lightmap_kernel(cuda):
    from numba import float32

    def cuda_jit_cached(function):
        try:
            return cuda.jit(cache=True)(function)
        except RuntimeError as error:
            if "no locator available" not in str(error):
                raise
            return cuda.jit(function)

    @cuda_jit_cached
    def lightmap_kernel(
        patch_dem,
        geotransform,
        projection,
        sun_vectors,
        horizons,
        vector_start,
        vector_count,
        tile_x,
        tile_y,
        valid_width,
        valid_height,
        output,
    ):
        index = cuda.grid(1)
        if index >= PATCH_SIZE * PATCH_SIZE:
            return
        sample = index % PATCH_SIZE
        line = index // PATCH_SIZE
        if sample >= valid_width or line >= valid_height:
            for batch_index in range(vector_count):
                output[batch_index, index] = 0
            return

        absolute_sample = tile_x + sample
        absolute_line = tile_y + line
        crs_x = geotransform[0] + geotransform[1] * absolute_sample + geotransform[2] * absolute_line
        crs_y = geotransform[3] + geotransform[4] * absolute_sample + geotransform[5] * absolute_line
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
            latitude = math.asin(cos_c * sin_latitude_origin + yp * sin_c * cos_latitude_origin / rho)
            longitude = projection[2] + math.atan2(
                xp * sin_c,
                rho * cos_latitude_origin * cos_c - yp * sin_latitude_origin * sin_c,
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
        tx = float32(-(moon_x * east_x + moon_y * east_y))
        ty = float32(-(moon_x * north_x + moon_y * north_y + moon_z * north_z))
        tz = float32(-(moon_x * up_x + moon_y * up_y + moon_z * up_z))
        r00, r01, r02 = float32(east_x), float32(north_x), float32(up_x)
        r10, r11, r12 = float32(east_y), float32(north_y), float32(up_y)
        r20, r21, r22 = float32(0.0), float32(north_z), float32(up_z)
        columns = cuda.const.array_like(_HALF_CIRCLE)

        for batch_index in range(vector_count):
            vector_index = vector_start + batch_index
            vx = sun_vectors[vector_index, 0]
            vy = sun_vectors[vector_index, 1]
            vz = sun_vectors[vector_index, 2]
            enu_x = float32(vx * r00 + vy * r10 + vz * r20 + tx)
            enu_y = float32(vx * r01 + vy * r11 + vz * r21 + ty)
            enu_z = float32(vx * r02 + vy * r12 + vz * r22 + tz)
            horizontal = float32(math.sqrt(float32(enu_x * enu_x + enu_y * enu_y)))
            elevation = float32(math.atan2(enu_z, horizontal) * float32(57.29577951308232))
            azimuth = float32(math.atan2(enu_x, enu_y))
            if azimuth < float32(0.0):
                azimuth = float32(azimuth + float32(math.pi * 2.0))
            azimuth_deg = float32(azimuth * float32(57.29577951308232))
            left_position = float32((azimuth_deg - float32(0.27) - float32(0.125)) * float32(4.0))
            left = int(left_position)
            fraction = float32(left_position - float32(left))
            if left < 0:
                left += AZIMUTH_COUNT
            elif left >= AZIMUTH_COUNT:
                left -= AZIMUTH_COUNT
            right = left + 1
            if right >= AZIMUTH_COUNT:
                right = 0
            left_elevation = horizons[line, sample, left]
            right_elevation = horizons[line, sample, right]
            delta = float32(right_elevation - left_elevation)
            photons = float32(0.0)
            for slice_index in range(16):
                horizon = float32(fraction * delta + left_elevation)
                column = columns[slice_index]
                sun_top = float32(elevation + column)
                if horizon < sun_top:
                    angle_delta = float32(sun_top - horizon)
                    column_height = float32(column + column)
                    if angle_delta > column_height:
                        angle_delta = column_height
                    photons = float32(photons + angle_delta)
                fraction = float32(fraction + float32(0.135))
                if fraction >= float32(1.0):
                    left = right
                    right += 1
                    if right >= AZIMUTH_COUNT:
                        right = 0
                    left_elevation = right_elevation
                    right_elevation = horizons[line, sample, right]
                    delta = float32(right_elevation - left_elevation)
                    fraction = float32(fraction - float32(1.0))
            visible = float32(photons / float32(_MAX_PHOTONS))
            output[batch_index, index] = int(float32(255.0) * visible)

    return lightmap_kernel


class LightmapCudaSession:
    """Reusable CUDA buffers with output bounded by ``time_batch_size``."""

    def __init__(self, *, device_id: int = 0, time_batch_size: int = 32) -> None:
        global _LIGHTMAP_KERNEL
        if time_batch_size < 1:
            raise ValueError("time_batch_size must be positive")
        try:
            from numba import cuda
        except ImportError as error:
            raise CudaBackendError("Numba CUDA is not installed") from error
        if not cuda.is_available():
            raise CudaBackendError("Numba CUDA cannot see a usable device")
        try:
            cuda.select_device(device_id)
        except Exception as error:
            raise CudaBackendError(f"cannot select CUDA device {device_id}") from error
        with _LIGHTMAP_KERNEL_LOCK:
            if _LIGHTMAP_KERNEL is None:
                _LIGHTMAP_KERNEL = _build_lightmap_kernel(cuda)
        self._cuda = cuda
        self._kernel = _LIGHTMAP_KERNEL
        self.time_batch_size = int(time_batch_size)
        self._lock = threading.Lock()
        self._dem = cuda.device_array((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        self._geotransform = cuda.device_array(6, dtype=np.float64)
        self._projection = cuda.device_array(6, dtype=np.float64)
        self._horizons = cuda.device_array((PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT), dtype=np.float32)
        self._output = cuda.device_array((self.time_batch_size, PATCH_SIZE * PATCH_SIZE), dtype=np.uint8)
        self._vectors = None
        self._vector_key = None

    def iter_patch_tiles(self, dem: DemGrid, horizons_deg: npt.ArrayLike, sun_vectors_m: npt.ArrayLike, *, tile_y: int, tile_x: int, valid_height: int = PATCH_SIZE, valid_width: int = PATCH_SIZE) -> Iterator[npt.NDArray[np.uint8]]:
        vectors64 = _validate_vectors(sun_vectors_m)
        vectors = np.ascontiguousarray(vectors64, dtype=np.float32)
        horizons = np.ascontiguousarray(horizons_deg, dtype=np.float32)
        if horizons.shape != (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT):
            raise ValueError("horizons must have shape (128, 128, 1440)")
        if not 1 <= valid_width <= PATCH_SIZE or not 1 <= valid_height <= PATCH_SIZE:
            raise ValueError("valid patch dimensions must be between 1 and 128")
        patch_dem = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        patch_dem[:valid_height, :valid_width] = dem.elevation_m[tile_y:tile_y + valid_height, tile_x:tile_x + valid_width]
        projection = np.asarray((dem.projection.radius_m, dem.projection.latitude_origin_rad, dem.projection.longitude_origin_rad, dem.projection.scale, dem.projection.false_easting_m, dem.projection.false_northing_m), dtype=np.float64)
        vector_key = hashlib.sha256(vectors.tobytes()).digest()
        with self._lock:
            if self._vector_key != vector_key:
                self._vectors = self._cuda.to_device(vectors)
                self._vector_key = vector_key
            self._dem.copy_to_device(patch_dem)
            self._geotransform.copy_to_device(dem.geo_transform)
            self._projection.copy_to_device(projection)
            self._horizons.copy_to_device(horizons)
            threads = 128
            blocks = (PATCH_SIZE * PATCH_SIZE + threads - 1) // threads
            for start in range(0, vectors.shape[0], self.time_batch_size):
                count = min(self.time_batch_size, vectors.shape[0] - start)
                self._kernel[blocks, threads](self._dem, self._geotransform, self._projection, self._vectors, self._horizons, start, count, tile_x, tile_y, valid_width, valid_height, self._output)
                self._cuda.synchronize()
                batch = self._output[:count].copy_to_host().reshape(count, PATCH_SIZE, PATCH_SIZE)
                for tile in batch:
                    yield np.ascontiguousarray(tile[:valid_height, :valid_width])
