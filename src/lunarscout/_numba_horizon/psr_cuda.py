"""Lazy Numba CUDA backend for the private Phase 6B PSR product."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import threading
import time

import numpy as np
import numpy.typing as npt

from .cuda_backend import CudaBackendError
from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .geometry import DemGrid


_PSR_KERNEL = None
_PSR_KERNEL_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class PsrCudaPatchTimings:
    """Detailed CUDA boundary timings for one PSR patch.

    ``kernel_execution_seconds`` is device-event time. The synchronization
    boundary is wall time and includes waiting for that kernel, so those two
    values are deliberately non-additive.
    """

    tile_y: int
    tile_x: int
    host_preparation_seconds: float
    h2d_dem_seconds: float
    h2d_metadata_seconds: float
    h2d_horizon_seconds: float
    h2d_vectors_seconds: float
    kernel_launch_seconds: float
    kernel_execution_seconds: float
    synchronization_boundary_seconds: float
    d2h_result_seconds: float
    total_seconds: float


def _build_psr_kernel(cuda):
    from numba import float32

    def cuda_jit_cached(function):
        try:
            return cuda.jit(cache=True)(function)
        except RuntimeError as error:
            if "no locator available" not in str(error):
                raise
            return cuda.jit(function)

    @cuda_jit_cached
    def psr_kernel(
        patch_dem,
        geotransform,
        projection,
        sun_vectors,
        horizons,
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
            output[index] = 0
            return

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
        translation_x = -(moon_x * east_x + moon_y * east_y)
        translation_y = -(
            moon_x * north_x + moon_y * north_y + moon_z * north_z
        )
        translation_z = -(moon_x * up_x + moon_y * up_y + moon_z * up_z)

        r00 = float32(east_x)
        r01 = float32(north_x)
        r02 = float32(up_x)
        r10 = float32(east_y)
        r11 = float32(north_y)
        r12 = float32(up_y)
        r20 = float32(0.0)
        r21 = float32(north_z)
        r22 = float32(up_z)
        tx = float32(translation_x)
        ty = float32(translation_y)
        tz = float32(translation_z)

        is_psr = 255
        for vector_index in range(sun_vectors.shape[0]):
            vector_x = sun_vectors[vector_index, 0]
            vector_y = sun_vectors[vector_index, 1]
            vector_z = sun_vectors[vector_index, 2]
            enu_x = float32(
                vector_x * r00 + vector_y * r10 + vector_z * r20 + tx
            )
            enu_y = float32(
                vector_x * r01 + vector_y * r11 + vector_z * r21 + ty
            )
            enu_z = float32(
                vector_x * r02 + vector_y * r12 + vector_z * r22 + tz
            )
            horizontal = float32(math.sqrt(float32(enu_x * enu_x + enu_y * enu_y)))
            elevation_deg = float32(
                math.atan2(enu_z, horizontal) * float32(57.29577951308232)
            )
            azimuth = float32(math.atan2(enu_x, enu_y))
            if azimuth < float32(0.0):
                azimuth = float32(azimuth + float32(math.pi * 2.0))
            azimuth_deg = float32(azimuth * float32(57.29577951308232))
            position = float32(azimuth_deg * float32(4.0))
            if position >= float32(AZIMUTH_COUNT):
                position = float32(0.0)
            lower = int(position)
            fraction = float32(position - float32(lower))
            upper = lower + 1
            if upper >= AZIMUTH_COUNT:
                upper = 0
            horizon_1 = horizons[line, sample, lower]
            horizon_2 = horizons[line, sample, upper]
            horizon = float32(
                horizon_1 + fraction * float32(horizon_2 - horizon_1)
            )
            if float32(elevation_deg - horizon) > float32(-0.545 / 2.0):
                is_psr = 0
        output[index] = is_psr

    return psr_kernel


class PsrCudaSession:
    """Reusable fixed-shape CUDA buffers and the production-shaped PSR kernel."""

    def __init__(
        self,
        device_id: int = 0,
        *,
        timing_callback: Callable[[PsrCudaPatchTimings], None] | None = None,
    ) -> None:
        global _PSR_KERNEL
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
        with _PSR_KERNEL_LOCK:
            if _PSR_KERNEL is None:
                _PSR_KERNEL = _build_psr_kernel(cuda)
        self._cuda = cuda
        self._kernel = _PSR_KERNEL
        self._lock = threading.Lock()
        self._device_dem = cuda.device_array((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        self._device_geotransform = cuda.device_array(6, dtype=np.float64)
        self._device_projection = cuda.device_array(6, dtype=np.float64)
        self._device_horizons = cuda.device_array(
            (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT), dtype=np.float32
        )
        self._device_output = cuda.device_array(PATCH_SIZE * PATCH_SIZE, dtype=np.uint8)
        self._device_vectors = None
        self._vector_capacity = 0
        self._resident_vectors: npt.NDArray[np.float32] | None = None
        self._resident_metadata_key: tuple[float, ...] | None = None
        self._timing_callback = timing_callback

    def allocate_pinned_horizon_buffer(self) -> npt.NDArray[np.float32]:
        """Allocate one fixed-shape host buffer suitable for direct decode/H2D."""
        return self._cuda.pinned_array(
            (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT), dtype=np.float32
        )

    def compute_patch(
        self,
        dem: DemGrid,
        horizons_deg: npt.ArrayLike,
        sun_vectors_m: npt.ArrayLike,
        *,
        tile_y: int,
        tile_x: int,
        valid_height: int = PATCH_SIZE,
        valid_width: int = PATCH_SIZE,
    ) -> npt.NDArray[np.uint8]:
        total_started = time.perf_counter()
        host_started = total_started
        if not 1 <= valid_width <= PATCH_SIZE or not 1 <= valid_height <= PATCH_SIZE:
            raise ValueError("valid patch dimensions must be between 1 and 128")
        if tile_x < 0 or tile_y < 0 or tile_x + valid_width > dem.width or tile_y + valid_height > dem.height:
            raise ValueError("PSR patch falls outside the DEM")
        horizons = np.ascontiguousarray(horizons_deg, dtype=np.float32)
        if horizons.shape != (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT):
            raise ValueError("horizons must have shape (128, 128, 1440)")
        if not np.all(np.isfinite(horizons)):
            raise ValueError("horizons must contain finite elevations")
        vectors = np.ascontiguousarray(sun_vectors_m, dtype=np.float32)
        if (
            vectors.ndim != 2
            or vectors.shape[1:] != (3,)
            or vectors.shape[0] == 0
            or not np.all(np.isfinite(vectors))
        ):
            raise ValueError("Sun vectors must be finite values shaped (time, 3)")
        patch_dem = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        patch_dem[:valid_height, :valid_width] = dem.elevation_m[
            tile_y : tile_y + valid_height, tile_x : tile_x + valid_width
        ]
        projection = np.asarray(
            (
                dem.projection.radius_m,
                dem.projection.latitude_origin_rad,
                dem.projection.longitude_origin_rad,
                dem.projection.scale,
                dem.projection.false_easting_m,
                dem.projection.false_northing_m,
            ),
            dtype=np.float64,
        )
        host_preparation_seconds = time.perf_counter() - host_started
        with self._lock:
            if vectors.shape[0] > self._vector_capacity:
                self._device_vectors = self._cuda.device_array(
                    vectors.shape, dtype=np.float32
                )
                self._vector_capacity = vectors.shape[0]
                self._resident_vectors = None
            device_vectors = self._device_vectors[: vectors.shape[0]]
            transfer_started = time.perf_counter()
            self._device_dem.copy_to_device(patch_dem)
            h2d_dem_seconds = time.perf_counter() - transfer_started
            metadata_key = tuple(float(value) for value in dem.geo_transform) + tuple(
                float(value) for value in projection
            )
            if metadata_key != self._resident_metadata_key:
                transfer_started = time.perf_counter()
                self._device_geotransform.copy_to_device(dem.geo_transform)
                self._device_projection.copy_to_device(projection)
                h2d_metadata_seconds = time.perf_counter() - transfer_started
                self._resident_metadata_key = metadata_key
            else:
                h2d_metadata_seconds = 0.0
            transfer_started = time.perf_counter()
            self._device_horizons.copy_to_device(horizons)
            h2d_horizon_seconds = time.perf_counter() - transfer_started
            if (
                self._resident_vectors is None
                or self._resident_vectors.shape != vectors.shape
                or not np.array_equal(self._resident_vectors, vectors)
            ):
                transfer_started = time.perf_counter()
                device_vectors.copy_to_device(vectors)
                h2d_vectors_seconds = time.perf_counter() - transfer_started
                self._resident_vectors = vectors.copy()
            else:
                h2d_vectors_seconds = 0.0
            threads = 128
            blocks = (PATCH_SIZE * PATCH_SIZE + threads - 1) // threads
            start_event = None
            end_event = None
            if self._timing_callback is not None:
                start_event = self._cuda.event(timing=True)
                end_event = self._cuda.event(timing=True)
                start_event.record()
            launch_started = time.perf_counter()
            self._kernel[blocks, threads](
                self._device_dem,
                self._device_geotransform,
                self._device_projection,
                device_vectors,
                self._device_horizons,
                tile_x,
                tile_y,
                valid_width,
                valid_height,
                self._device_output,
            )
            kernel_launch_seconds = time.perf_counter() - launch_started
            if end_event is not None:
                end_event.record()
            sync_started = time.perf_counter()
            if end_event is None:
                self._cuda.synchronize()
                kernel_execution_seconds = 0.0
            else:
                end_event.synchronize()
                kernel_execution_seconds = (
                    start_event.elapsed_time(end_event) / 1000.0
                )
            synchronization_boundary_seconds = time.perf_counter() - sync_started
            copy_started = time.perf_counter()
            output = self._device_output.copy_to_host().reshape(PATCH_SIZE, PATCH_SIZE)
            d2h_result_seconds = time.perf_counter() - copy_started
        result = np.ascontiguousarray(output[:valid_height, :valid_width])
        if self._timing_callback is not None:
            self._timing_callback(
                PsrCudaPatchTimings(
                    tile_y=tile_y,
                    tile_x=tile_x,
                    host_preparation_seconds=host_preparation_seconds,
                    h2d_dem_seconds=h2d_dem_seconds,
                    h2d_metadata_seconds=h2d_metadata_seconds,
                    h2d_horizon_seconds=h2d_horizon_seconds,
                    h2d_vectors_seconds=h2d_vectors_seconds,
                    kernel_launch_seconds=kernel_launch_seconds,
                    kernel_execution_seconds=kernel_execution_seconds,
                    synchronization_boundary_seconds=synchronization_boundary_seconds,
                    d2h_result_seconds=d2h_result_seconds,
                    total_seconds=time.perf_counter() - total_started,
                )
            )
        return result
