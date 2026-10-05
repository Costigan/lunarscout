"""CUDA safe-haven geometry pairing and device-side monthly reduction."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from datetime import datetime

import numpy as np
import numpy.typing as npt

from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .lightmap_cuda import LightmapCudaSession


_REDUCER = None
_FINALIZER = None
_REDUCER_LOCK = threading.Lock()


def _build_reducer(cuda):
    from numba import float32

    @cuda.jit
    def reducer(
        fractions, margins, month_map, start, count, final, valid_width, valid_height,
        sunlight_threshold, earth_threshold, touched, best, had_outage, was_above,
        run_length, min_month, max_month, error_flag,
    ):
        pixel = cuda.grid(1)
        if pixel >= PATCH_SIZE * PATCH_SIZE:
            return
        line = pixel // PATCH_SIZE
        sample = pixel % PATCH_SIZE
        valid = sample < valid_width and line < valid_height
        run = run_length[pixel]
        lo = min_month[pixel]
        hi = max_month[pixel]
        for offset in range(count):
            fraction = fractions[offset, pixel]
            margin = margins[offset, pixel]
            if valid and (not math.isfinite(fraction) or not math.isfinite(margin)):
                cuda.atomic.max(error_flag, 0, 1)
                continue
            if not valid:
                continue
            month = month_map[start + offset]
            earth_below = margin < earth_threshold
            sun_low = fraction < sunlight_threshold
            if month >= 0:
                if earth_below:
                    had_outage[month, pixel] = 1
                else:
                    was_above[month, pixel] = 1
            if sun_low:
                run += 1
            elif run > 0:
                for band in range(lo, hi + 1):
                    if touched[band, pixel]:
                        if run > best[band, pixel]:
                            best[band, pixel] = run
                        touched[band, pixel] = 0
                run = 0
                lo = 2147483647
                hi = -1
            if sun_low and earth_below and month >= 0:
                touched[month, pixel] = 1
                if month < lo:
                    lo = month
                if month > hi:
                    hi = month
        if final and run > 0:
            for band in range(lo, hi + 1):
                if touched[band, pixel]:
                    if run > best[band, pixel]:
                        best[band, pixel] = run
                    touched[band, pixel] = 0
        run_length[pixel] = run
        min_month[pixel] = lo
        max_month[pixel] = hi

    return reducer


def _build_finalizer(cuda):
    from numba import float32

    @cuda.jit
    def finalizer(best, had_outage, was_above, output, month_count, step):
        pixel = cuda.grid(1)
        if pixel >= PATCH_SIZE * PATCH_SIZE:
            return
        for band in range(month_count):
            if had_outage[band, pixel] == 0 or was_above[band, pixel] == 0:
                output[band, pixel] = math.nan
            else:
                output[band, pixel] = float32(best[band, pixel]) * step

    return finalizer


class SafeHavenCudaCancelled(RuntimeError):
    """Cancellation observed at a CUDA batch boundary."""


class SafeHavenCudaSession(LightmapCudaSession):
    """One coherent geometry/reduction session for a safe-haven patch."""

    def __init__(self, *, device_id: int = 0, time_batch_size: int = 32) -> None:
        super().__init__(device_id=device_id, time_batch_size=time_batch_size)
        global _REDUCER, _FINALIZER
        with _REDUCER_LOCK:
            if _REDUCER is None:
                _REDUCER = _build_reducer(self._cuda)
            if _FINALIZER is None:
                _FINALIZER = _build_finalizer(self._cuda)
        self._reducer = _REDUCER
        self._finalizer = _FINALIZER
        n = PATCH_SIZE * PATCH_SIZE
        self._month_map = None
        self._month_map_key = None
        self._sun_vectors = None
        self._sun_vectors_key = None
        self._earth_vectors = None
        self._earth_vectors_key = None
        self._state_months = 0
        self._error_flag = self._cuda.device_array(1, dtype=np.int32)
        self._duration_output = None
        self._duration_months = 0
        self._run_length = self._cuda.device_array(n, dtype=np.int32)
        self._min_month = self._cuda.device_array(n, dtype=np.int32)
        self._max_month = self._cuda.device_array(n, dtype=np.int32)

    def reduce_patch(
        self, dem, horizons_deg, sun_vectors_m, earth_vectors_m, *,
        tile_y: int, tile_x: int, valid_height: int, valid_width: int,
        month_bands: tuple[tuple[datetime, datetime], ...],
        month_index_of: npt.NDArray[np.int32], time_count: int,
        sunlight_threshold: float, earth_threshold_deg: float,
        time_step_hours: float, cancellation_requested: Callable[[], bool] | None = None,
    ) -> tuple[npt.NDArray[np.float32], ...]:
        if time_count < 1 or len(month_bands) < 1:
            raise ValueError("time_count and month bands must be positive")
        if not 1 <= valid_width <= PATCH_SIZE or not 1 <= valid_height <= PATCH_SIZE:
            raise ValueError("valid patch dimensions must be between 1 and 128")
        if month_index_of.shape != (time_count,):
            raise ValueError("month_index_of must have shape (time_count,)")
        if not 0.0 <= sunlight_threshold <= 1.0:
            raise ValueError("sunlight_threshold must be between zero and one")
        if not np.isfinite(earth_threshold_deg):
            raise ValueError("earth_threshold_deg must be finite")
        if not np.isfinite(time_step_hours) or time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive and finite")
        if np.any(month_index_of < -1) or np.any(month_index_of >= len(month_bands)):
            raise ValueError("month_index_of contains an invalid band index")
        horizons_array = np.asarray(horizons_deg)
        if horizons_array.shape != (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT):
            raise ValueError("horizons must have shape (128, 128, 1440)")
        sun = np.ascontiguousarray(np.asarray(sun_vectors_m, dtype=np.float32))
        earth = np.ascontiguousarray(np.asarray(earth_vectors_m, dtype=np.float32))
        if sun.shape != (time_count, 3) or earth.shape != (time_count, 3):
            raise ValueError("vectors must have shape (time_count, 3)")
        if not np.all(np.isfinite(sun)) or not np.all(np.isfinite(earth)):
            raise ValueError("vectors must be finite")
        month_count = len(month_bands)
        n = PATCH_SIZE * PATCH_SIZE
        with self._lock:
            if self._state_months != month_count:
                shape = (month_count, n)
                self._touched = self._cuda.device_array(shape, dtype=np.uint8)
                self._best = self._cuda.device_array(shape, dtype=np.int32)
                self._had_outage = self._cuda.device_array(shape, dtype=np.uint8)
                self._was_above = self._cuda.device_array(shape, dtype=np.uint8)
                self._state_months = month_count
            zero = np.zeros((month_count, n), dtype=np.uint8)
            self._touched.copy_to_device(zero)
            self._had_outage.copy_to_device(zero)
            self._was_above.copy_to_device(zero)
            self._best.copy_to_device(np.zeros((month_count, n), dtype=np.int32))
            if self._duration_output is None or self._duration_months != month_count:
                self._duration_output = self._cuda.device_array((month_count, n), dtype=np.float32)
                self._duration_months = month_count
            self._run_length.copy_to_device(np.zeros(n, dtype=np.int32))
            self._min_month.copy_to_device(np.full(n, 2147483647, dtype=np.int32))
            self._max_month.copy_to_device(np.full(n, -1, dtype=np.int32))
            self._error_flag.copy_to_device(np.zeros(1, dtype=np.int32))
            month_map = np.ascontiguousarray(month_index_of, dtype=np.int32)
            month_key = month_map.tobytes()
            if self._month_map_key != month_key:
                self._month_map = self._cuda.to_device(month_map)
                self._month_map_key = month_key
            sun_key = sun.tobytes()
            if self._sun_vectors_key != sun_key:
                self._sun_vectors = self._cuda.to_device(sun)
                self._sun_vectors_key = sun_key
            earth_key = earth.tobytes()
            if self._earth_vectors_key != earth_key:
                self._earth_vectors = self._cuda.to_device(earth)
                self._earth_vectors_key = earth_key
            self._dem.copy_to_device(np.pad(
                np.asarray(dem.elevation_m[tile_y:tile_y + valid_height, tile_x:tile_x + valid_width], dtype=np.float32),
                ((0, PATCH_SIZE - valid_height), (0, PATCH_SIZE - valid_width)),
            ))
            self._geotransform.copy_to_device(dem.geo_transform)
            self._projection.copy_to_device(np.asarray((dem.projection.radius_m, dem.projection.latitude_origin_rad, dem.projection.longitude_origin_rad, dem.projection.scale, dem.projection.false_easting_m, dem.projection.false_northing_m), dtype=np.float64))
            self._horizons.copy_to_device(np.ascontiguousarray(horizons_array, dtype=np.float32))
            threads = 128
            blocks = (n + threads - 1) // threads
            for start in range(0, time_count, self.time_batch_size):
                if cancellation_requested and cancellation_requested():
                    raise SafeHavenCudaCancelled("safe-haven generation was cancelled")
                count = min(self.time_batch_size, time_count - start)
                for vectors, margin in ((self._sun_vectors, False), (self._earth_vectors, True)):
                    self._kernel[blocks, threads](self._dem, self._geotransform, self._projection, vectors, self._horizons, start, count, tile_x, tile_y, valid_width, valid_height, self._output, self._fraction_output, self._margin_output, margin)
                self._reducer[blocks, threads](self._fraction_output, self._margin_output, self._month_map, start, count, start + count == time_count, valid_width, valid_height, np.float32(sunlight_threshold), np.float32(earth_threshold_deg), self._touched, self._best, self._had_outage, self._was_above, self._run_length, self._min_month, self._max_month, self._error_flag)
                self._cuda.synchronize()
                if cancellation_requested and cancellation_requested():
                    raise SafeHavenCudaCancelled("safe-haven generation was cancelled")
            if int(self._error_flag.copy_to_host()[0]):
                raise ValueError("non-finite CUDA lighting signal")
            self._finalizer[blocks, threads](self._best, self._had_outage, self._was_above, self._duration_output, month_count, np.float32(time_step_hours))
            self._cuda.synchronize()
            durations = self._duration_output.copy_to_host()
        results = []
        for band in range(month_count):
            out = durations[band]
            results.append(np.ascontiguousarray(out.reshape(PATCH_SIZE, PATCH_SIZE)[:valid_height, :valid_width]))
        return tuple(results)
