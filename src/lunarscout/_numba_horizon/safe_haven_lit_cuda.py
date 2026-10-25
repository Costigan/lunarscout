"""CUDA lighting of stage-3 units: the lightmap, thresholded and bit-packed on the device (private).

A unit is 24 hours of flood steps of one 128 x 128 patch, stored as packed lit bits
(``safe_haven_litstore``: ``(steps_per_chunk, 128, 2)`` 64-bit words, bit ``k`` of word ``w`` in a
row being column ``64 * w + k``). ``LitCudaSession`` computes them with the production CUDA
lightmap kernel (the one stage 1 runs) followed by a small packing kernel, so only the packed bits,
33 times fewer than the float32 fractions, cross the PCIe bus. The result matches
``safe_haven_litstore.light_patch_chunks_fused`` (the CPU route), which is its test, except at pixel-steps
whose Sun fraction lies within a few float32 steps of the lit threshold: there the GPU and CPU arithmetic
can differ by up to about three ulps and the bit can fall either way (3 bits in 106 million pixel-steps
in one traced cell window; 12 of 4,181 table rows differ, each by one pixel).

Stage 0 uses the same session for the thresholded-lighting store (``safe_haven_lightstore``):
``light_patch_planes`` runs the production kernel on the Sun vectors (two packings, at the shadow and
operating fractions) and again on the Earth vectors with the margin output on (one packing, at the
outage elevation), so three bit-planes per step come back and nothing else crosses the bus. The general
lighting paths are not touched.

The packing kernel has a fixed trip count (64 bits per word), checks every index against its array
bounds and launches exactly one thread per output word, so it cannot loop without end or write out
of range.
"""

from __future__ import annotations

import threading

import numpy as np
import numpy.typing as npt

from .cuda_backend import cuda_session_setup
from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .lightmap_cuda import LightmapCudaSession
from .psr import _validate_vectors
from .safe_haven_lightstore import PLANE_COUNT, LightStoreSettings, plane_thresholds

WORDS_PER_ROW = PATCH_SIZE // 64
THREADS = 128

_PACK_KERNEL = None
_PACK_KERNEL_LOCK = threading.Lock()


def _cuda_jit_cached(cuda, function):
    """Compile with Numba's on-disk cache when the source has a locator."""
    try:
        return cuda.jit(cache=True)(function)
    except RuntimeError as error:
        if "no locator available" not in str(error):
            raise
        return cuda.jit(function)


def _build_pack_kernel(cuda):
    """One thread per output word: set bit ``k`` where the fraction reaches the threshold."""

    def pack(fractions, step_count, valid_width, valid_height, threshold, words):
        index = cuda.grid(1)
        if index >= step_count * PATCH_SIZE * WORDS_PER_ROW:
            return
        word = index % WORDS_PER_ROW
        row = (index // WORDS_PER_ROW) % PATCH_SIZE
        step = index // (WORDS_PER_ROW * PATCH_SIZE)
        bits = np.uint64(0)
        if row < valid_height:
            base = row * PATCH_SIZE + word * 64
            for k in range(64):                                   # fixed trip count
                column = word * 64 + k
                if column < valid_width:
                    if np.float64(fractions[step, base + k]) >= threshold:
                        bits |= np.uint64(1) << np.uint64(k)
        words[step, row, word] = bits

    return _cuda_jit_cached(cuda, pack)


class LitCudaSession(LightmapCudaSession):
    """Lights a patch's units on the device: the lightmap kernel, then the packing kernel."""

    def __init__(self, *, device_id: int = 0, steps_per_chunk: int = 45, chunks_per_batch: int = 4) -> None:
        if int(steps_per_chunk) != steps_per_chunk or steps_per_chunk < 1:
            raise ValueError("steps_per_chunk must be a positive integer")
        if int(chunks_per_batch) != chunks_per_batch or chunks_per_batch < 1:
            raise ValueError("chunks_per_batch must be a positive integer")
        super().__init__(device_id=device_id, time_batch_size=int(steps_per_chunk) * int(chunks_per_batch))
        global _PACK_KERNEL
        with cuda_session_setup("Stage-3 CUDA lighting session"):
            with _PACK_KERNEL_LOCK:
                if _PACK_KERNEL is None:
                    _PACK_KERNEL = _build_pack_kernel(self._cuda)
            self._pack = _PACK_KERNEL
            self.steps_per_chunk = int(steps_per_chunk)
            self.chunks_per_batch = int(chunks_per_batch)
            self._bits = self._cuda.device_array(
                (self.time_batch_size, PATCH_SIZE, WORDS_PER_ROW), dtype=np.uint64)
            self._planes = self._cuda.device_array(
                (PLANE_COUNT, self.time_batch_size, PATCH_SIZE, WORDS_PER_ROW), dtype=np.uint64)

    def light_patch(
        self,
        dem,
        horizons_deg: npt.ArrayLike,
        chunk_vectors: list[npt.ArrayLike],
        *,
        tile_y: int,
        tile_x: int,
        valid_height: int,
        valid_width: int,
        threshold: float,
    ) -> list[npt.NDArray[np.uint64]]:
        """Packed lit bits, ``(steps_per_chunk, 128, 2)``, for each chunk's Sun vectors.

        The patch's inputs go to the device once, and the units come back batch by batch. A pixel
        outside the valid ``valid_height`` x ``valid_width`` area is never lit.
        """
        spc = self.steps_per_chunk
        if not chunk_vectors:
            return []
        if not 1 <= valid_width <= PATCH_SIZE or not 1 <= valid_height <= PATCH_SIZE:
            raise ValueError("valid patch dimensions must be between 1 and 128")
        arrays = [_validate_vectors(vectors) for vectors in chunk_vectors]
        if any(array.shape[0] != spc for array in arrays):
            raise ValueError(f"every chunk needs {spc} Sun vectors")
        vectors = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
        horizons = np.ascontiguousarray(horizons_deg, dtype=np.float32)
        if horizons.shape != (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT):
            raise ValueError("horizons must have shape (128, 128, 1440)")
        patch_dem = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        patch_dem[:valid_height, :valid_width] = dem.elevation_m[
            tile_y:tile_y + valid_height, tile_x:tile_x + valid_width]
        projection = np.asarray((
            dem.projection.radius_m, dem.projection.latitude_origin_rad, dem.projection.longitude_origin_rad,
            dem.projection.scale, dem.projection.false_easting_m, dem.projection.false_northing_m,
        ), dtype=np.float64)
        total = vectors.shape[0]
        blocks = (PATCH_SIZE * PATCH_SIZE + THREADS - 1) // THREADS
        units: list[npt.NDArray[np.uint64]] = []
        with self._lock:
            self._dem.copy_to_device(patch_dem)
            self._geotransform.copy_to_device(dem.geo_transform)
            self._projection.copy_to_device(projection)
            self._horizons.copy_to_device(horizons)
            self._vectors = self._cuda.to_device(vectors)
            self._vector_key = None                       # the base class's cached vectors are not these
            self._input_owner = None
            for start in range(0, total, self.time_batch_size):
                count = min(self.time_batch_size, total - start)
                self._kernel[blocks, THREADS](
                    self._dem, self._geotransform, self._projection, self._vectors, self._horizons,
                    start, count, tile_x, tile_y, valid_width, valid_height,
                    self._output, self._fraction_output, self._margin_output, False)
                words = count * PATCH_SIZE * WORDS_PER_ROW
                self._pack[(words + THREADS - 1) // THREADS, THREADS](
                    self._fraction_output, count, valid_width, valid_height, np.float64(threshold), self._bits)
                self._cuda.synchronize()
                host = self._bits[:count].copy_to_host()
                for first in range(0, count, spc):
                    units.append(np.ascontiguousarray(host[first:first + spc]))
        return units

    def light_patch_planes(
        self,
        dem,
        horizons_deg: npt.ArrayLike,
        sun_vectors: npt.ArrayLike,
        earth_vectors: npt.ArrayLike,
        settings: LightStoreSettings,
        *,
        tile_y: int,
        tile_x: int,
        valid_height: int,
        valid_width: int,
    ) -> npt.NDArray[np.uint64]:
        """The store's three packed planes for one patch, ``(3, steps, 128, 2)``, in batches of
        ``time_batch_size`` steps. The patch's inputs go to the device once. A pixel outside the valid
        ``valid_height`` x ``valid_width`` area has every bit clear."""
        if not 1 <= valid_width <= PATCH_SIZE or not 1 <= valid_height <= PATCH_SIZE:
            raise ValueError("valid patch dimensions must be between 1 and 128")
        sun = np.ascontiguousarray(_validate_vectors(sun_vectors), dtype=np.float32)
        earth = np.ascontiguousarray(_validate_vectors(earth_vectors), dtype=np.float32)
        if sun.shape != earth.shape:
            raise ValueError("the Sun and Earth vectors must have the same shape")
        horizons = np.ascontiguousarray(horizons_deg, dtype=np.float32)
        if horizons.shape != (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT):
            raise ValueError("horizons must have shape (128, 128, 1440)")
        patch_dem = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        patch_dem[:valid_height, :valid_width] = dem.elevation_m[
            tile_y:tile_y + valid_height, tile_x:tile_x + valid_width]
        projection = np.asarray((
            dem.projection.radius_m, dem.projection.latitude_origin_rad, dem.projection.longitude_origin_rad,
            dem.projection.scale, dem.projection.false_easting_m, dem.projection.false_northing_m,
        ), dtype=np.float64)
        shadow, operating, outage = plane_thresholds(settings)
        total = sun.shape[0]
        blocks = (PATCH_SIZE * PATCH_SIZE + THREADS - 1) // THREADS
        planes = np.zeros((PLANE_COUNT, total, PATCH_SIZE, WORDS_PER_ROW), dtype=np.uint64)
        with self._lock:
            self._dem.copy_to_device(patch_dem)
            self._geotransform.copy_to_device(dem.geo_transform)
            self._projection.copy_to_device(projection)
            self._horizons.copy_to_device(horizons)
            device_sun = self._cuda.to_device(sun)
            device_earth = self._cuda.to_device(earth)
            self._vectors = None
            self._vector_key = None                       # the base class's cached vectors are not these
            self._input_owner = None
            for start in range(0, total, self.time_batch_size):
                count = min(self.time_batch_size, total - start)
                pack_blocks = (count * PATCH_SIZE * WORDS_PER_ROW + THREADS - 1) // THREADS
                self._kernel[blocks, THREADS](
                    self._dem, self._geotransform, self._projection, device_sun, self._horizons,
                    start, count, tile_x, tile_y, valid_width, valid_height,
                    self._output, self._fraction_output, self._margin_output, False)
                for plane, threshold in ((0, shadow), (1, operating)):
                    self._pack[pack_blocks, THREADS](
                        self._fraction_output, count, valid_width, valid_height, np.float64(threshold),
                        self._planes[plane])
                self._kernel[blocks, THREADS](
                    self._dem, self._geotransform, self._projection, device_earth, self._horizons,
                    start, count, tile_x, tile_y, valid_width, valid_height,
                    self._output, self._fraction_output, self._margin_output, True)
                self._pack[pack_blocks, THREADS](
                    self._margin_output, count, valid_width, valid_height, np.float64(outage), self._planes[2])
                self._cuda.synchronize()
                for plane in range(PLANE_COUNT):
                    planes[plane, start:start + count] = self._planes[plane][:count].copy_to_host()
        return planes
