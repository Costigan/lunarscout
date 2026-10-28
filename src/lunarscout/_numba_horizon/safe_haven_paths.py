"""Rover paths along the stage-4 predecessor edges (private).

Stage 3 keeps frontiers, not paths. To draw an edge ``A -> B`` as the route a rover could take, this module
floods forward from seed ``A`` again, exactly as stage 3 did, and keeps the frontier after every step; then it
traces back from ``B``'s pixel at ``B``'s before boundary, one step at a time, to ``A``'s pixel at its after boundary.

The trace of one step works on the pixels around the current pixel ``p`` (a window of ``moves`` pixels each way): it
recomputes the sub-move frontiers ``F_1 .. F_moves`` of that step from the stored frontier of the step before, in
the pixels that are passable and lit in the step, and walks back from ``p``, preferring to stay where it is and
otherwise taking the first neighbour that is in the frontier before the move. The result is one feasible,
time-respecting route (at most ``moves_per_step`` one-pixel moves per step, every pixel entered lit and passable
in its step, staying only on lit pixels), not a shortest one. The path is the pixel after every one-pixel move,
consecutive pixels being neighbours, from ``A``'s pixel to ``B``'s.

The stored frontiers and lit masks are boxes around the seed (``moves * step`` pixels each way), about 100 MB each
for a 28-day flood. Seeds are processed in parallel, one flood per predecessor.
"""

from __future__ import annotations

import multiprocessing
import signal
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .file_format import PATCH_SIZE
from .safe_haven_flood_packed import WORD_BITS
from .safe_haven_floods import (
    FloodContext, FloodSettings, WORDS_PER_PATCH_ROW, advance_kernel, load_cached_passable,
)
from .safe_haven_stage0 import LightStoreDirectory

_ORDER_Y = (0, -1, 1, 0, 0, -1, -1, 1, 1)
_ORDER_X = (0, 0, 0, -1, 1, -1, 1, -1, 1)


# ---------------------------------------------------------------------------
# The trace kernel
# ---------------------------------------------------------------------------


def build_trace(jit):
    """The trace kernel with ``jit`` applied (identity for plain Python)."""

    @jit
    def bit(flat, meta, j, row, col):
        """Pixel ``(row, col)`` of snapshot ``j`` (0 outside its box)."""
        r = row - meta[j, 3]
        w = (col >> 6) - meta[j, 4]
        if col < 0 or r < 0 or r >= meta[j, 1] or w < 0 or w >= meta[j, 2]:
            return 0
        return np.int64((flat[meta[j, 0] + r * meta[j, 2] + w] >> np.uint64(col & 63)) & np.uint64(1))

    @jit
    def trace(front, fmeta, mask, mmeta, row, col, steps, eight, moves, out_row, out_col):
        """Trace back from pixel ``(row, col)`` in the frontier of step ``steps``; fills ``out_row`` and ``out_col``
        with the pixel after every one-pixel move, last first, and returns their number (-1: no route)."""
        side = 2 * moves + 1
        frontier = np.zeros((moves + 1, side, side), dtype=np.uint8)
        lit = np.zeros((side, side), dtype=np.uint8)
        order_y = np.array(_ORDER_Y)
        order_x = np.array(_ORDER_X)
        n = 1
        out_row[0] = row
        out_col[0] = col
        for j in range(steps, 0, -1):
            for dy in range(side):
                for dx in range(side):
                    frontier[0, dy, dx] = bit(front, fmeta, j - 1, row + dy - moves, col + dx - moves)
                    lit[dy, dx] = bit(mask, mmeta, j, row + dy - moves, col + dx - moves)
            for m in range(1, moves + 1):
                for dy in range(side):
                    for dx in range(side):
                        grown = 0
                        for ey in range(-1, 2):
                            for ex in range(-1, 2):
                                if not eight and ey != 0 and ex != 0:
                                    continue
                                y = dy + ey
                                x = dx + ex
                                if y >= 0 and y < side and x >= 0 and x < side and frontier[m - 1, y, x] != 0:
                                    grown = 1
                        frontier[m, dy, dx] = grown & lit[dy, dx]
            y = moves
            x = moves
            if frontier[moves, y, x] == 0:
                return -1
            for m in range(moves, 0, -1):
                if frontier[m - 1, y, x] != 0:
                    continue                                   # stay: no new pixel
                found = False
                for k in range(1, 9):
                    ey = order_y[k]
                    ex = order_x[k]
                    if not eight and ey != 0 and ex != 0:
                        continue
                    ny = y + ey
                    nx = x + ex
                    if ny >= 0 and ny < side and nx >= 0 and nx < side and frontier[m - 1, ny, nx] != 0:
                        y = ny
                        x = nx
                        found = True
                        break
                if not found:
                    return -1
                out_row[n] = row + y - moves
                out_col[n] = col + x - moves
                n += 1
            row = row + y - moves
            col = col + x - moves
        return n

    return trace


trace_python = build_trace(lambda function: function)
_COMPILED: dict = {}


def trace_kernel(compiled: bool = True):
    if not compiled:
        return trace_python
    if "trace" not in _COMPILED:
        from numba import njit

        _COMPILED["trace"] = build_trace(njit(cache=False, nogil=True))
    return _COMPILED["trace"]


# ---------------------------------------------------------------------------
# One source seed
# ---------------------------------------------------------------------------


def _geometry(steps: int, moves: int, local_row: int, local_column: int, rows: int, words: int):
    """Snapshot boxes of steps ``0..steps``: rows ``[offset, rows, words, row0, word0]`` and the total size."""
    meta = np.zeros((steps + 1, 5), dtype=np.int64)
    total = 0
    for j in range(steps + 1):
        reach = moves * j
        row0, row1 = max(0, local_row - reach), min(rows - 1, local_row + reach)
        word0, word1 = max(0, local_column - reach) >> 6, min(words * 64 - 1, local_column + reach) >> 6
        meta[j] = (total, row1 - row0 + 1, word1 - word0 + 1, row0, word0)
        total += (row1 - row0 + 1) * (word1 - word0 + 1)
    return meta, total


def reconstruct_source(context: FloodContext, source: dict, targets: list, *, compiled: bool = True) -> list:
    """Paths from ``source`` to each of ``targets`` ``[(target_id, row, column, flood_step)]``: a list of
    ``(target_id, rows, columns)`` int32 arrays in raster coordinates, from the source's pixel to the target's."""
    context.ensure_process_state()
    settings, cache, buffers = context.settings, context.cache, context.buffers
    height, width = context.dem_shape
    moves, spd = settings.moves_per_step, context.steps_per_day
    eight = settings.connectivity == 8
    steps = max(int(t[3]) for t in targets)
    if steps < 1 or steps > context.max_steps:
        raise InputError("a target is not due within the flood", code="paths_step", details={"steps": steps})
    patches = buffers["patches"]
    seed_row, seed_column, after = source["row"], source["column"], source["after_step"]
    patch_row, patch_column = seed_row // PATCH_SIZE, seed_column // PATCH_SIZE
    pr0, pc0 = max(0, patch_row - patches), max(0, patch_column - patches)
    row0, col0 = pr0 * PATCH_SIZE, pc0 * PATCH_SIZE
    row1 = min(height, (patch_row + patches + 1) * PATCH_SIZE)
    col1 = min(width, (patch_column + patches + 1) * PATCH_SIZE)
    rows, words = row1 - row0, -(-(col1 - col0) // WORD_BITS)
    word0 = col0 // WORD_BITS
    passable = context.passable[row0:row1, word0:word0 + words]
    cur, nxt, lit = buffers["cur"][:rows, :words], buffers["nxt"][:rows, :words], buffers["lit"]
    cur[:] = 0
    nxt[:] = 0
    local_row, local_column = seed_row - row0, seed_column - col0
    cur[local_row, local_column >> 6] = np.uint64(1) << np.uint64(local_column & 63)

    meta, total = _geometry(steps, moves, local_row, local_column, rows, words)
    front = np.empty(total, dtype=np.uint64)
    mask = np.zeros(total, dtype=np.uint64)

    def keep(buffer, j, array, other=None) -> None:
        offset, n_rows, n_words, first_row, first_word = (int(v) for v in meta[j])
        region = array[first_row:first_row + n_rows, first_word:first_word + n_words]
        if other is not None:
            region = region & other[first_row:first_row + n_rows, first_word:first_word + n_words]
        buffer[offset:offset + n_rows * n_words] = region.ravel()

    keep(front, 0, cur)
    kernel = advance_kernel(compiled)
    dummy = np.zeros((1, 1), dtype=np.uint64)
    touched = np.zeros(3, dtype=np.int64)                  # >= 0: no PSR is looked for
    zero3 = np.zeros(3, dtype=np.int64)
    none = np.zeros(0, dtype=np.int64)
    done_total, moves_done = 0, 0
    while done_total < steps:
        first_abs = after + done_total + 1
        day = first_abs // spd
        nsteps = min(steps - done_total, (day + 1) * spd - first_abs)
        box = moves_done + nsteps * moves
        box_rows = (max(0, seed_row - box) // PATCH_SIZE, min(height - 1, seed_row + box) // PATCH_SIZE)
        box_cols = (max(0, seed_column - box) // PATCH_SIZE, min(width - 1, seed_column + box) // PATCH_SIZE)
        a = first_abs - day * spd
        for pr in range(box_rows[0], box_rows[1] + 1):
            for pc in range(box_cols[0], box_cols[1] + 1):
                unit = cache.get(pr, pc, day)
                target = lit[:nsteps, (pr - pr0) * PATCH_SIZE:(pr - pr0 + 1) * PATCH_SIZE,
                             (pc - pc0) * WORDS_PER_PATCH_ROW:(pc - pc0 + 1) * WORDS_PER_PATCH_ROW]
                if unit is None:
                    target[:] = 0
                else:
                    target[:] = unit[a:a + nsteps]
        for k in range(nsteps):
            done, flips, _count, died = kernel(
                cur, nxt, lit[k:k + 1, :rows, :words], 1, moves, moves_done, local_row, local_column, passable,
                dummy, dummy, dummy, eight, done_total, -1, touched, zero3, zero3, none, none, none, none)
            if died:
                raise InputError("a flood died before the step of one of its targets", code="paths_flood_died",
                                 details={"seed_id": source["seed_id"], "step": done_total + 1})
            if flips & 1:
                cur, nxt = nxt, cur
            done_total += 1
            moves_done += moves
            keep(front, done_total, cur)
            keep(mask, done_total, lit[k, :rows, :words], passable)

    trace = trace_kernel(compiled)
    paths = []
    for target_id, target_row, target_column, step in targets:
        out_row = np.empty(moves * int(step) + 1, dtype=np.int64)
        out_column = np.empty(moves * int(step) + 1, dtype=np.int64)
        n = trace(front, meta, mask, meta, int(target_row) - row0, int(target_column) - col0, int(step), eight, moves,
                  out_row, out_column)
        if n < 0:
            raise InputError("no route to a target the flood reached", code="paths_trace_failed",
                             details={"seed_id": source["seed_id"], "target_id": int(target_id), "step": int(step)})
        paths.append((int(target_id), (out_row[:n][::-1] + row0).astype(np.int32),
                      (out_column[:n][::-1] + col0).astype(np.int32)))
    return paths


# ---------------------------------------------------------------------------
# Every predecessor edge
# ---------------------------------------------------------------------------


@dataclass
class PathSet:
    """The paths of the predecessor edges, keyed by the edge's head seed: ``path(head)`` is its pixels, from the
    predecessor's to the head's."""

    heads: npt.NDArray[np.int64]          # head seed ids, ascending
    offsets: npt.NDArray[np.int64]        # len(heads) + 1 offsets into rows and columns
    rows: npt.NDArray[np.int32]
    columns: npt.NDArray[np.int32]

    def index(self, head_ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        ids = np.asarray(head_ids, dtype=np.int64)
        position = np.searchsorted(self.heads, ids)
        clipped = np.minimum(position, max(len(self.heads) - 1, 0))
        if len(ids) and (not len(self.heads) or (self.heads[clipped] != ids).any()):
            raise InputError("a head seed has no path", code="paths_missing")
        return position

    def path(self, head_id: int):
        i = int(self.index([head_id])[0])
        return self.rows[self.offsets[i]:self.offsets[i + 1]], self.columns[self.offsets[i]:self.offsets[i + 1]]


_RUN: dict = {}


def _init() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _task(task):
    source, targets = task
    return reconstruct_source(_RUN["context"], source, targets, compiled=_RUN["compiled"])


def reconstruct_paths(
    rows: list[dict], *, directory: LightStoreDirectory, passable_path, dem_shape: tuple[int, int],
    settings: FloodSettings = FloodSettings(), step_minutes: int | None = None, workers: int = 1,
    compiled: bool = True, progress: Callable[[int, int], None] | None = None,
) -> PathSet:
    """The path of every predecessor edge of the stage-4 table ``rows`` (``read_chain_table``): a rover leaves the
    predecessor's pixel at its ``after_step`` and is on the seed's pixel at the seed's ``before_step``. The passable
    mask must be the one stage 3 cached at ``passable_path`` for this store and slope limit."""
    step_minutes = directory.settings.step_minutes if step_minutes is None else step_minutes
    passable = load_cached_passable(passable_path, directory, settings, dem_shape)
    if passable is None:
        raise InputError(f"{passable_path} is not the passable mask of this store and slope limit; run stage 3 first",
                         code="paths_passable_missing", details={"path": str(passable_path)})
    by_id = {r["seed_id"]: r for r in rows}
    groups: dict[int, list] = {}
    for row in rows:
        if row["predecessor_id"] > 0:
            source = by_id[row["predecessor_id"]]
            if row["before_step"] is None:
                raise InputError("a predecessor edge leads to a seed with no before boundary", code="paths_step",
                                 details={"seed_id": row["seed_id"]})
            groups.setdefault(source["seed_id"], []).append(
                (row["seed_id"], row["row"], row["column"], row["before_step"] - source["after_step"]))
    context = FloodContext(
        settings=settings, dem_shape=dem_shape, step_minutes=step_minutes, passable=passable, psr_masks=(), psr_ids=None,
        directory=directory, seeds=[], by_cycle={}, later=np.zeros((0, 4), dtype=np.int64),
        unit_budget=settings.unit_cache_mib * 2**20)
    tasks = [(by_id[s], t) for s, t in sorted(groups.items())]
    found: dict[int, tuple] = {}
    pool = None
    try:
        if workers > 1 and tasks:
            _RUN["context"], _RUN["compiled"] = context, compiled
            pool = multiprocessing.get_context("fork").Pool(workers, initializer=_init)
            results = pool.imap_unordered(_task, tasks, chunksize=1)
        else:
            results = (reconstruct_source(context, source, targets, compiled=compiled) for source, targets in tasks)
        for number, paths in enumerate(results, start=1):
            for head, path_rows, path_columns in paths:
                found[head] = (path_rows, path_columns)
            if progress is not None:
                progress(number, len(tasks))
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        _RUN.clear()
    heads = np.array(sorted(found), dtype=np.int64)
    lengths = np.array([len(found[int(h)][0]) for h in heads], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    flat_rows = np.concatenate([found[int(h)][0] for h in heads]) if len(heads) else np.zeros(0, dtype=np.int32)
    flat_columns = np.concatenate([found[int(h)][1] for h in heads]) if len(heads) else np.zeros(0, dtype=np.int32)
    return PathSet(heads, offsets, flat_rows, flat_columns)
