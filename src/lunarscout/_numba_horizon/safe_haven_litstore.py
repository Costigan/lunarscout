"""Stage-2 pass 2: the per-month lit-bit store, CPU reference route (private).

``docs/safe-haven-windows-spec.md`` section 8.  The unit of storage is one
``(patch, chunk)``: a patch is 128 x 128 pixels and a chunk is 24 hours of flood
steps on the global flood grid (``FloodGrid.steps_per_chunk``).  Each unit is stored
as packed bits, ``(steps_per_chunk, 128, 2)`` words, bit ``k`` of word ``w`` in a row
being column ``64 * w + k`` of the patch (the layout of ``safe_haven_flood_packed``).

The lighting here is the CPU lightmap path followed by the stage-1 lit threshold and
bit-packing.  It is the reference for the GPU route, which will compute the same bits
with the CUDA lightmap kernel and a second thresholding kernel.  A pixel outside the
valid part of a patch, a patch with no horizon tile, and a patch outside the DEM are
never lit.  The store holds one month: build a new one per month.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt

from .file_format import PATCH_SIZE, HorizonTileStore, read_patch_horizons
from .safe_haven_flood import AFTER, BEFORE, FloodResult, lit_threshold_test
from .safe_haven_flood_packed import WORD_BITS, flood_side_packed, pack_bits
from .safe_haven_windows import FloodGrid

WORDS_PER_PATCH_ROW = PATCH_SIZE // WORD_BITS


class LitStoreError(RuntimeError):
    """A chunk the flood needs is missing from the store."""


@dataclass
class FillCounts:
    """What ``fill_lit_store`` did; every field adds across calls."""

    computed: int = 0            # (patch, chunk) units lit
    reused: int = 0              # units already in the store
    patches_without_horizons: int = 0
    patches_read: int = 0


class LitStore:
    """Packed lit bits for one month, keyed by ``(patch_row, patch_column, chunk)``."""

    def __init__(self, grid: FloodGrid, dem_shape: tuple[int, int]) -> None:
        self.grid = grid
        self.dem_shape = (int(dem_shape[0]), int(dem_shape[1]))
        self._chunks: dict[tuple[int, int, int], npt.NDArray[np.uint64]] = {}

    @property
    def chunk_shape(self) -> tuple[int, int, int]:
        return (self.grid.steps_per_chunk, PATCH_SIZE, WORDS_PER_PATCH_ROW)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def nbytes(self) -> int:
        return sum(chunk.nbytes for chunk in self._chunks.values())

    def clear(self) -> None:
        self._chunks.clear()

    def has(self, patch_row: int, patch_column: int, chunk: int) -> bool:
        return self.outside(patch_row, patch_column) or (patch_row, patch_column, chunk) in self._chunks

    def outside(self, patch_row: int, patch_column: int) -> bool:
        """Whether the patch lies wholly outside the DEM (never lit, never stored)."""
        return (
            patch_row < 0 or patch_column < 0
            or patch_row * PATCH_SIZE >= self.dem_shape[0]
            or patch_column * PATCH_SIZE >= self.dem_shape[1]
        )

    def put(self, patch_row: int, patch_column: int, chunk: int, bits: npt.NDArray[np.uint64]) -> None:
        bits = np.ascontiguousarray(bits, dtype=np.uint64)
        if bits.shape != self.chunk_shape:
            raise ValueError(f"a chunk must have shape {self.chunk_shape}")
        self._chunks[(int(patch_row), int(patch_column), int(chunk))] = bits

    def chunk_range(self, first_step: int, last_step: int) -> range:
        spc = self.grid.steps_per_chunk
        return range(first_step // spc, last_step // spc + 1)

    def assemble(
        self, patch_row0: int, patch_column0: int, patch_rows: int, patch_columns: int,
        first_step: int, last_step: int,
    ) -> npt.NDArray[np.uint64]:
        """Packed lit bits ``(steps, rows, words)`` over a patch-aligned window and
        flood steps ``first_step .. last_step`` inclusive."""
        if last_step < first_step:
            raise ValueError("last_step must not be before first_step")
        spc = self.grid.steps_per_chunk
        out = np.zeros(
            (last_step - first_step + 1, patch_rows * PATCH_SIZE, patch_columns * WORDS_PER_PATCH_ROW),
            dtype=np.uint64,
        )
        missing = []
        for patch_row in range(patch_row0, patch_row0 + patch_rows):
            for patch_column in range(patch_column0, patch_column0 + patch_columns):
                if self.outside(patch_row, patch_column):
                    continue
                for chunk in self.chunk_range(first_step, last_step):
                    data = self._chunks.get((patch_row, patch_column, chunk))
                    if data is None:
                        missing.append((patch_row, patch_column, chunk))
                        continue
                    low, high = max(first_step, chunk * spc), min(last_step, (chunk + 1) * spc - 1)
                    out[
                        low - first_step: high - first_step + 1,
                        (patch_row - patch_row0) * PATCH_SIZE: (patch_row - patch_row0 + 1) * PATCH_SIZE,
                        (patch_column - patch_column0) * WORDS_PER_PATCH_ROW:
                        (patch_column - patch_column0 + 1) * WORDS_PER_PATCH_ROW,
                    ] = data[low - chunk * spc: high - chunk * spc + 1]
        if missing:
            raise LitStoreError(
                f"{len(missing)} (patch row, patch column, chunk) units are missing, "
                f"first {missing[0]}"
            )
        return out

    def pixel_lit(self, row: int, column: int, step: int) -> bool:
        """Whether one DEM pixel is lit at a flood step (``False`` outside the DEM)."""
        patch_row, patch_column = row // PATCH_SIZE, column // PATCH_SIZE
        if row < 0 or column < 0 or self.outside(patch_row, patch_column):
            return False
        spc = self.grid.steps_per_chunk
        data = self._chunks.get((patch_row, patch_column, step // spc))
        if data is None:
            raise LitStoreError(f"unit {(patch_row, patch_column, step // spc)} is missing")
        word = int(data[step % spc, row % PATCH_SIZE, (column % PATCH_SIZE) // WORD_BITS])
        return bool((word >> (column % WORD_BITS)) & 1)

    def side_lit(
        self, window: tuple[int, int, int, int], boundary_step: int, direction: int, steps: int,
    ) -> npt.NDArray[np.uint64]:
        """The lit masks of one flood side, ordered along the flood: element ``j - 1`` is
        the mask at ``boundary_step + direction * j``, as ``flood_side_packed`` takes them.

        ``window`` is ``(patch_row0, patch_column0, patch_rows, patch_columns)``.
        """
        if steps < 1:
            return np.zeros((0, window[2] * PATCH_SIZE, window[3] * WORDS_PER_PATCH_ROW), dtype=np.uint64)
        if direction == AFTER:
            return self.assemble(*window, boundary_step + 1, boundary_step + steps)
        if direction == BEFORE:
            return self.assemble(*window, boundary_step - steps, boundary_step - 1)[::-1]
        raise ValueError("direction must be BEFORE (-1) or AFTER (+1)")


def light_patch_chunks(
    session,
    dem,
    horizons: npt.NDArray[np.float32],
    patch_row: int,
    patch_column: int,
    grid: FloodGrid,
    chunks: Iterable[int],
    sun_vectors_at: Callable[[npt.NDArray[np.datetime64]], npt.ArrayLike],
    operating_sunlight_fraction: float = 1.0,
) -> dict[int, npt.NDArray[np.uint64]]:
    """Packed lit bits of one patch for the given chunks (the CPU reference lighting).

    Sun visible fraction from the CPU lightmap, thresholded with the stage-1 operating
    rule, with pixels outside the DEM left unlit, then bit-packed.
    """
    tile_y, tile_x = patch_row * PATCH_SIZE, patch_column * PATCH_SIZE
    valid_height = min(PATCH_SIZE, dem.elevation_m.shape[0] - tile_y)
    valid_width = min(PATCH_SIZE, dem.elevation_m.shape[1] - tile_x)
    spc = grid.steps_per_chunk
    result: dict[int, npt.NDArray[np.uint64]] = {}
    for chunk in sorted(set(int(c) for c in chunks)):
        steps = np.arange(chunk * spc, (chunk + 1) * spc, dtype=np.int64)
        vectors = np.asarray(sun_vectors_at(grid.times(steps)), dtype=np.float64)
        lit = np.zeros((spc, PATCH_SIZE, PATCH_SIZE), dtype=bool)
        done = 0
        for fractions in session.iter_patch_fraction_batches(
            dem, horizons, vectors, tile_y=tile_y, tile_x=tile_x,
            valid_height=valid_height, valid_width=valid_width,
        ):
            lit[done: done + fractions.shape[0], :valid_height, :valid_width] = lit_threshold_test(
                fractions, operating_sunlight_fraction
            )
            done += fractions.shape[0]
        result[chunk] = pack_bits(lit)
    return result


def fill_lit_store(
    store: LitStore,
    *,
    dem,
    horizon_store: HorizonTileStore,
    patches: Iterable[tuple[int, int]],
    chunks: Iterable[int],
    sun_vectors_at: Callable[[npt.NDArray[np.datetime64]], npt.ArrayLike],
    operating_sunlight_fraction: float = 1.0,
    observer_height_m: float = 0.0,
    session=None,
) -> FillCounts:
    """Light every missing ``(patch, chunk)`` of ``patches`` x ``chunks`` into the store.

    Each patch's horizons are read once, however many chunks it needs, and units already
    stored are reused.  A patch without a horizon tile is stored as all unlit.
    """
    if session is None:
        from .lightmap_cpu import LightmapCpuSession

        session = LightmapCpuSession()
    wanted = sorted(set(int(c) for c in chunks))
    counts = FillCounts()
    for patch_row, patch_column in sorted(set((int(r), int(c)) for r, c in patches)):
        if store.outside(patch_row, patch_column):
            continue
        todo = [c for c in wanted if not store.has(patch_row, patch_column, c)]
        counts.reused += len(wanted) - len(todo)
        if not todo:
            continue
        horizons = read_patch_horizons(
            horizon_store, patch_row * PATCH_SIZE, patch_column * PATCH_SIZE, observer_height_m
        )
        counts.patches_read += 1
        if horizons is None:
            counts.patches_without_horizons += 1
            for chunk in todo:
                store.put(patch_row, patch_column, chunk, np.zeros(store.chunk_shape, dtype=np.uint64))
            counts.computed += len(todo)
            continue
        for chunk, bits in light_patch_chunks(
            session, dem, horizons, patch_row, patch_column, store.grid, todo,
            sun_vectors_at, operating_sunlight_fraction,
        ).items():
            store.put(patch_row, patch_column, chunk, bits)
        counts.computed += len(todo)
    return counts


def flood_seed_from_store(
    store: LitStore,
    passable_packed: npt.NDArray[np.uint64],
    window: tuple[int, int, int, int],
    seed: tuple[int, int],
    boundary_before_step: int,
    boundary_after_step: int,
    *,
    steps: int,
    connectivity: int = 8,
    compiled: bool = True,
) -> FloodResult | None:
    """Both sides of a seed's flood from the stored lit bits, or ``None`` on a boundary mismatch.

    ``seed`` is in window pixels (row, column); ``window`` is ``(patch_row0, patch_column0,
    patch_rows, patch_columns)`` and ``passable_packed`` the window's passable mask.
    The seed must be lit at its boundary in the stored bits (spec section 8, "Boundary
    agreement"); if a rounding difference at the fuzz limit makes it unlit, the boundary
    moves one step away from the hibernation when the stored chunks cover that, and
    otherwise the seed is a mismatch.
    """
    row = window[0] * PATCH_SIZE + seed[0]
    column = window[1] * PATCH_SIZE + seed[1]
    counts = []
    for boundary, direction in ((boundary_before_step, BEFORE), (boundary_after_step, AFTER)):
        for candidate in (boundary, boundary + direction):
            try:
                if not store.pixel_lit(row, column, candidate):
                    continue
                lit = store.side_lit(window, candidate, direction, steps)
            except LitStoreError:
                continue
            counts.append(flood_side_packed(
                lit, passable_packed, seed, steps=steps, connectivity=connectivity, compiled=compiled,
            )[0])
            break
        else:
            return None
    return FloodResult(*counts)
