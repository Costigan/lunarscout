"""Stage 3 fed from the stage-0 light store (private).

Stage 3 needs lighting in two places: pass 1 follows single pixels through time (the hibernation of
record, the Earth gate, the flood boundaries), and the floods need the lit bits of whole patches over
24-hour chunks. Without a store both are lit from the horizons. With a store neither touches a horizon:

* :class:`StoreSignals` gives a pixel's series from its stored bits, in the form pass 1 already takes (a
  Sun fraction of 0, 0.5 or 1 and an Earth margin of -90 or +90 degrees, which carry exactly what the
  planes know), so the same rules run on it. Earth's flat-horizon elevation, which the degenerate-outage
  rule needs, is the series stage 1 used for the pixel's patch (computed at the valid pixel nearest the
  patch centre, ``safe_haven_stage1_store.flat_earth_elevation``), so that pass 1 reproduces stage 1's
  hibernation exactly.
* :func:`fill_lit_store_from_light_store` fills ``LitStore`` units, ``(patch, chunk)`` of packed lit bits,
  from the store's operating plane. The unit layout is the store's own, so a unit is a slice of a
  decoded chunk.

The flood grid is the store's grid: both count steps of ``step_minutes`` from 1970-01-01, so flood step
``k`` is store step ``k``. Steps outside the stored range have no data and count as dark (no sunlight, no
Earth); the flood windows of seeds within a day or so of the ends of the series can reach them.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .file_format import PATCH_SIZE
from .safe_haven_litstore import FillCounts, LitStore
from .safe_haven_lightstore import LightStore
from .safe_haven_stage0 import LightStoreDirectory
from .safe_haven_lightstore import patch_geometry
from .safe_haven_stage1_store import (
    CLEAR_MARGIN_DEG,
    PARTIAL_FRACTION,
    check_thresholds,
    first_store_step,
    flat_earth_elevation,
)

MICROSECONDS_PER_MINUTE = 60_000_000
_SUN_FULL, _SUN_PARTIAL, _EARTH = 1, 0, 2        # plane numbers: operating, shadow threshold, Earth
_RECENT_READS = 2


def check_run(directory: LightStoreDirectory, *, flood_step_minutes: int, times: npt.ArrayLike,
              stage1_step_hours: float, sunlight_threshold: float, operating_sunlight_fraction: float,
              earth_threshold_deg: float) -> None:
    """Refuse a store that cannot answer this run: the flood grid and the stage-1 samples must be the
    store's step and lie on its grid inside its range, and the thresholds must be the store's."""
    if int(flood_step_minutes) != directory.settings.step_minutes:
        raise InputError(
            "the flood step is not the light store's step", code="safe_haven_store_step_mismatch",
            details={"flood_step_minutes": flood_step_minutes, "store_step_minutes": directory.settings.step_minutes})
    check_thresholds(directory, sunlight_threshold=sunlight_threshold,
                     operating_sunlight_fraction=operating_sunlight_fraction,
                     earth_threshold_deg=earth_threshold_deg)
    stamps = np.asarray(times).astype("datetime64[us]")
    first_store_step(directory, stamps, stage1_step_hours)


class StoreSignals:
    """A pixel's Sun fraction and Earth margin from its stored bits (``PixelSignals``' interface).

    ``series(row, column, vectors_m, margin=..., flat_elevation=..., times_us=...)`` needs ``times_us``,
    the UTC times in microseconds since the epoch, which must lie on the store's grid; ``vectors_m`` is
    used only for ``flat_elevation``. A pixel whose patch has no file (it had no horizon tile) gives
    ``None``. Steps outside the stored range are dark. Open files are kept for the ``patch_cache_size``
    most recently used patches.
    """

    def __init__(self, directory: LightStoreDirectory, dem, *, patch_cache_size: int = 4) -> None:
        from .lightmap_cpu import LightmapCpuSession

        self._directory = directory
        self._dem = dem
        self._session = LightmapCpuSession(time_batch_size=1)
        self._step_us = directory.settings.step_minutes * MICROSECONDS_PER_MINUTE
        self._files: OrderedDict[tuple[int, int], LightStore | None] = OrderedDict()
        self._file_cache_size = int(patch_cache_size)
        self._recent: OrderedDict[tuple, npt.NDArray[np.uint64]] = OrderedDict()

    def close(self) -> None:
        for opened in self._files.values():
            if opened is not None:
                opened.close()
        self._files.clear()
        self._recent.clear()

    def _file(self, patch_row: int, patch_column: int) -> LightStore | None:
        key = (patch_row, patch_column)
        if key in self._files:
            self._files.move_to_end(key)
            return self._files[key]
        opened = self._directory.open_patch(patch_row, patch_column)
        self._files[key] = opened
        while len(self._files) > self._file_cache_size:
            evicted_key, evicted = self._files.popitem(last=False)
            if evicted is not None:
                evicted.close()
            for recent in [k for k in self._recent if k[0] == evicted_key]:
                del self._recent[recent]
        return opened

    def _planes(self, key: tuple[int, int], opened: LightStore, first: int, stop: int) -> npt.NDArray[np.uint64]:
        cached = self._recent.get((key, first, stop))
        if cached is not None:
            return cached
        planes = opened.read_steps(first, stop)
        self._recent[(key, first, stop)] = planes
        while len(self._recent) > _RECENT_READS:
            self._recent.popitem(last=False)
        return planes

    def series(self, row: int, column: int, vectors_m: npt.ArrayLike | None, *, margin: bool,
               flat_elevation: bool = False, times_us: npt.ArrayLike | None = None):
        """Float32 values per time, or ``None`` when the pixel has no file."""
        row, column = int(row), int(column)
        if row < 0 or column < 0 or row >= self._dem.height or column >= self._dem.width:
            return None
        if flat_elevation:
            if not margin or vectors_m is None:
                raise ValueError("flat_elevation needs margin and the Earth vectors")
            patch_row, patch_column = row // PATCH_SIZE, column // PATCH_SIZE
            if self._file(patch_row, patch_column) is None:
                return None
            tile_y, tile_x, valid_height, valid_width = patch_geometry(self._dem, patch_row, patch_column)
            return flat_earth_elevation(self._dem, vectors_m, valid_height, valid_width, tile_y, tile_x,
                                        self._session)
        if times_us is None:
            raise ValueError("StoreSignals needs times_us")
        key = (row // PATCH_SIZE, column // PATCH_SIZE)
        opened = self._file(*key)
        if opened is None:
            return None
        times = np.asarray(times_us, dtype=np.int64)
        if np.any(times % self._step_us):
            raise InputError("a time is not on the light store's step grid", code="safe_haven_store_time_misaligned")
        steps = times // self._step_us
        dark = np.float32(-CLEAR_MARGIN_DEG) if margin else np.float32(0.0)
        out = np.full(steps.shape, dark, dtype=np.float32)
        first = max(int(steps.min()), opened.first_step)
        stop = min(int(steps.max()) + 1, opened.stop_step)
        if first >= stop:
            return out
        planes = self._planes(key, opened, first, stop)
        inside = (steps >= first) & (steps < stop)
        offsets = steps[inside] - first
        line, word, shift = row % PATCH_SIZE, (column % PATCH_SIZE) >> 6, np.uint64(column % 64)
        one = np.uint64(1)
        if margin:
            visible = (planes[_EARTH, offsets, line, word] >> shift) & one
            out[inside] = np.where(visible == 1, np.float32(CLEAR_MARGIN_DEG), np.float32(-CLEAR_MARGIN_DEG))
        else:
            full = (planes[_SUN_FULL, offsets, line, word] >> shift) & one
            partial = (planes[_SUN_PARTIAL, offsets, line, word] >> shift) & one
            out[inside] = np.where(full == 1, np.float32(1.0),
                                   np.where(partial == 1, np.float32(PARTIAL_FRACTION), np.float32(0.0)))
        return out


def unit_bits(opened: LightStore, chunk: int, steps_per_chunk: int) -> npt.NDArray[np.uint64]:
    """The operating plane of one 24-hour chunk as a lit-store unit ``(steps_per_chunk, 128, 2)``;
    steps outside the stored range are zero (dark)."""
    first, stop = chunk * steps_per_chunk, (chunk + 1) * steps_per_chunk
    bits = np.zeros((steps_per_chunk, PATCH_SIZE, 2), dtype=np.uint64)
    low, high = max(first, opened.first_step), min(stop, opened.stop_step)
    if low < high:
        bits[low - first: high - first] = opened.read_steps(low, high)[_SUN_FULL]
    return bits


def fill_lit_store_from_light_store(
    store: LitStore, *, directory: LightStoreDirectory, patches, chunks,
) -> FillCounts:
    """Put every missing ``(patch, chunk)`` of ``patches`` x ``chunks`` into the lit store from the
    light store. A patch without a file is stored as all dark; one outside the DEM is never stored."""
    wanted = sorted(set(int(c) for c in chunks))
    counts = FillCounts()
    spc = store.grid.steps_per_chunk
    for patch_row, patch_column in sorted(set((int(r), int(c)) for r, c in patches)):
        if store.outside(patch_row, patch_column):
            continue
        todo = [c for c in wanted if not store.has(patch_row, patch_column, c)]
        counts.reused += len(wanted) - len(todo)
        if not todo:
            continue
        opened = directory.open_patch(patch_row, patch_column)
        counts.patches_read += 1
        if opened is None:
            counts.patches_without_horizons += 1
            for chunk in todo:
                store.put(patch_row, patch_column, chunk, np.zeros(store.chunk_shape, dtype=np.uint64))
        else:
            with opened:
                for chunk in todo:
                    store.put(patch_row, patch_column, chunk, unit_bits(opened, chunk, spc))
        counts.computed += len(todo)
    return counts
