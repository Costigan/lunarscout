"""CUDA safe-haven geometry pairing and the device-side safe-haven state machine."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from datetime import datetime

import numpy as np
import numpy.typing as npt

from .cuda_backend import cuda_session_setup
from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .lightmap_cuda import LightmapCudaSession
from .safe_haven import (
    MAX_DEGENERATE_WINDOW,
    OPERATING_FUZZ,
    RING_ROWS,
    STATE_ROWS,
    UNSET,
    merge_gap_limit,
    new_state,
)


_REDUCER = None
_FINALIZER = None
_REDUCER_LOCK = threading.Lock()


def _cuda_jit_cached(cuda, function):
    """Compile with Numba's on-disk cache when the source has a locator."""
    try:
        return cuda.jit(cache=True)(function)
    except RuntimeError as error:
        if "no locator available" not in str(error):
            raise
        return cuda.jit(function)


def _build_reducer(cuda):
    """The per-pixel safe-haven machine as a kernel over one batch of time steps.

    The machine itself is ``safe_haven.make_machine``: the same source that runs
    on the CPU.  State lives in global memory between batches.
    """
    from .safe_haven import make_machine

    machine = make_machine(cuda.jit(device=True))
    advance_delayed = machine.advance_delayed
    flush_delayed = machine.flush_delayed
    finish_pixel = machine.finish_pixel

    def reducer(
        fractions, margins, elevations, month_map, start, count, final, time_count,
        window, valid_width, valid_height, sunlight_threshold, earth_threshold,
        operating_threshold, gap_limit, S, R, best, have, joint_total, joint_best,
        error_flag, pixel_valid,
    ):
        pixel = cuda.grid(1)
        if pixel >= PATCH_SIZE * PATCH_SIZE:
            return
        line = pixel // PATCH_SIZE
        sample = pixel % PATCH_SIZE
        if sample >= valid_width or line >= valid_height:
            return
        if pixel_valid[pixel] == 0:      # on an invalid DEM node: its signals are NaN
            return
        for offset in range(count):
            fraction = fractions[offset, pixel]
            margin = margins[offset, pixel]
            elevation = elevations[offset, pixel]
            if (
                not math.isfinite(fraction) or not math.isfinite(margin)
                or not math.isfinite(elevation)
            ):
                cuda.atomic.max(error_flag, 0, 1)
                continue
            advance_delayed(
                S, R, pixel, best, have, joint_total, joint_best, fraction, margin,
                elevation, start + offset, month_map, window, sunlight_threshold,
                earth_threshold, operating_threshold, gap_limit,
            )
        if final:
            flush_delayed(
                S, R, pixel, best, have, joint_total, joint_best, time_count,
                month_map, window, sunlight_threshold, earth_threshold,
                operating_threshold, gap_limit,
            )
            finish_pixel(S, pixel, best)

    return _cuda_jit_cached(cuda, reducer)


def _build_finalizer(cuda):
    from numba import float32

    from .safe_haven import UNSET

    def finalizer(
        best, have, joint_total, joint_best, hibernation, total, longest,
        month_count, step, error_flag,
    ):
        pixel = cuda.grid(1)
        if pixel >= PATCH_SIZE * PATCH_SIZE:
            return
        for band in range(month_count):
            if have[band, pixel] == 0:
                hibernation[band, pixel] = math.nan
            elif best[band, pixel] == UNSET:
                hibernation[band, pixel] = math.nan
                cuda.atomic.max(error_flag, 0, 2)    # a hibernation was never scored
            else:
                hibernation[band, pixel] = float32(best[band, pixel]) * step
            total[band, pixel] = float32(joint_total[band, pixel]) * step
            longest[band, pixel] = float32(joint_best[band, pixel]) * step

    return _cuda_jit_cached(cuda, finalizer)


class SafeHavenCudaCancelled(RuntimeError):
    """Cancellation observed at a CUDA batch boundary."""


class SafeHavenCudaSession(LightmapCudaSession):
    """One coherent geometry/reduction session for a safe-haven patch."""

    def __init__(self, *, device_id: int = 0, time_batch_size: int = 32) -> None:
        super().__init__(device_id=device_id, time_batch_size=time_batch_size)
        global _REDUCER, _FINALIZER
        with cuda_session_setup("Safe-haven CUDA session"):
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
            self._state = self._cuda.device_array((STATE_ROWS, n), dtype=np.int32)
            self._ring = self._cuda.device_array((RING_ROWS, n), dtype=np.float32)
            self._elevation_output = self._cuda.device_array(
                (self.time_batch_size, n), dtype=np.float32
            )
            self._pixel_valid = self._cuda.device_array(n, dtype=np.uint8)
            self._host_state = new_state(n)
            self._outputs = None
            self._output_months = 0

    def reduce_patch(
        self, dem, horizons_deg, sun_vectors_m, earth_vectors_m, *,
        tile_y: int, tile_x: int, valid_height: int, valid_width: int,
        month_bands: tuple[tuple[datetime, datetime], ...],
        month_index_of: npt.NDArray[np.int32], time_count: int,
        sunlight_threshold: float, earth_threshold_deg: float,
        operating_sunlight_fraction: float, merge_gap_hours: float,
        time_step_hours: float, cancellation_requested: Callable[[], bool] | None = None,
        invalid_pixels: npt.ArrayLike | None = None,
        degenerate_window_samples: int = 0,
    ) -> tuple[tuple[npt.NDArray[np.float32], ...], ...]:
        """Return ``(hibernation, joint_total, joint_longest)`` tiles per band.

        ``degenerate_window_samples`` is the half-window of the degenerate-outage
        rule (``docs/safe-haven-state-machine.md``, section 2.1); 0 turns it off.
        """
        if (
            int(degenerate_window_samples) != degenerate_window_samples
            or not 0 <= degenerate_window_samples <= MAX_DEGENERATE_WINDOW
        ):
            raise ValueError(
                f"degenerate_window_samples must be an integer from 0 to {MAX_DEGENERATE_WINDOW}"
            )
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
        if not 0.0 <= operating_sunlight_fraction <= 1.0:
            raise ValueError("operating_sunlight_fraction must be between zero and one")
        if not np.isfinite(merge_gap_hours) or merge_gap_hours < 0.0:
            raise ValueError("merge_gap_hours must be finite and nonnegative")
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
        pixel_valid = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        if invalid_pixels is not None:
            invalid = np.asarray(invalid_pixels, dtype=np.bool_)
            if invalid.shape != (valid_height, valid_width):
                raise ValueError("invalid_pixels must have the patch's valid shape")
            pixel_valid[:valid_height, :valid_width][invalid] = 0
        gap_limit = merge_gap_limit(merge_gap_hours, time_step_hours)
        operating_threshold = np.float32(operating_sunlight_fraction - OPERATING_FUZZ)
        with self._lock:
            if self._state_months != month_count:
                shape = (month_count, n)
                self._best = self._cuda.device_array(shape, dtype=np.int32)
                self._have = self._cuda.device_array(shape, dtype=np.uint8)
                self._joint_total = self._cuda.device_array(shape, dtype=np.int32)
                self._joint_best = self._cuda.device_array(shape, dtype=np.int32)
                self._state_months = month_count
                self._host_best_init = np.full(shape, UNSET, dtype=np.int32)
                self._host_zero_bytes = np.zeros(shape, dtype=np.uint8)
                self._host_zero_ints = np.zeros(shape, dtype=np.int32)
            self._best.copy_to_device(self._host_best_init)
            self._have.copy_to_device(self._host_zero_bytes)
            self._joint_total.copy_to_device(self._host_zero_ints)
            self._joint_best.copy_to_device(self._host_zero_ints)
            self._state.copy_to_device(self._host_state)
            if self._outputs is None or self._output_months != month_count:
                self._outputs = tuple(
                    self._cuda.device_array((month_count, n), dtype=np.float32)
                    for _ in range(3)
                )
                self._output_months = month_count
            self._error_flag.copy_to_device(np.zeros(1, dtype=np.int32))
            self._pixel_valid.copy_to_device(np.ascontiguousarray(pixel_valid.reshape(-1)))
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
            # The lightmap iterators share these inputs; force a re-upload.
            self._input_owner = None
            threads = 128
            blocks = (n + threads - 1) // threads
            for start in range(0, time_count, self.time_batch_size):
                if cancellation_requested and cancellation_requested():
                    raise SafeHavenCudaCancelled("safe-haven generation was cancelled")
                count = min(self.time_batch_size, time_count - start)
                # The Earth call writes the elevation above a flat horizon into
                # the array given as its fraction output.
                for vectors, fraction_output, margin in (
                    (self._sun_vectors, self._fraction_output, False),
                    (self._earth_vectors, self._elevation_output, True),
                ):
                    self._kernel[blocks, threads](self._dem, self._geotransform, self._projection, vectors, self._horizons, start, count, tile_x, tile_y, valid_width, valid_height, self._output, fraction_output, self._margin_output, margin)
                self._reducer[blocks, threads](
                    self._fraction_output, self._margin_output, self._elevation_output,
                    self._month_map, start, count, start + count == time_count,
                    time_count, np.int32(degenerate_window_samples), valid_width,
                    valid_height, np.float32(sunlight_threshold),
                    np.float32(earth_threshold_deg), operating_threshold,
                    np.int32(gap_limit), self._state, self._ring, self._best,
                    self._have, self._joint_total, self._joint_best, self._error_flag,
                    self._pixel_valid,
                )
                self._cuda.synchronize()
                if cancellation_requested and cancellation_requested():
                    raise SafeHavenCudaCancelled("safe-haven generation was cancelled")
            if int(self._error_flag.copy_to_host()[0]) == 1:
                raise ValueError("non-finite CUDA lighting signal")
            self._finalizer[blocks, threads](
                self._best, self._have, self._joint_total, self._joint_best,
                *self._outputs, month_count, np.float32(time_step_hours),
                self._error_flag,
            )
            self._cuda.synchronize()
            if int(self._error_flag.copy_to_host()[0]) == 2:
                raise ValueError("internal error: a hibernation was never scored")
            hosts = [output.copy_to_host() for output in self._outputs]
        results = []
        for host in hosts:
            results.append(tuple(
                np.ascontiguousarray(
                    host[band].reshape(PATCH_SIZE, PATCH_SIZE)[:valid_height, :valid_width]
                )
                for band in range(month_count)
            ))
        return tuple(results)
