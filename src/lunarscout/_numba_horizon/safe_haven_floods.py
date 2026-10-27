"""Stage 3 of the rover safe-haven chain: a forward flood from every plan seed (private).

For each seed of the stage-2 table the rover starts at the seed pixel on the seed's ``after_step`` (the first
lit step after its outage) and moves forward in time. In every step the rover makes ``moves_per_step``
one-pixel moves, each into a passable pixel that is lit in that step (or stays where it is, if that pixel is
lit); the frontier is every pixel it can be on. The flood ends when the frontier is empty or after
``max_days``. It reads lit bits from the stage-0 light store on demand, 24 hours of one patch at a time, so a
flood that dies in a day reads a day.

Each seed gets:

* ``lit_endurance_hours``: the hours from the after boundary through the last step in which the rover can still
  be on lit terrain, at most ``max_days``. A value at the cap is a lower bound.
* ``reach48_pixels``: the frontier's size 48 hours after the boundary (0 if it died earlier).
* the first touch of a small, a medium and a large permanently shadowed region (PSR): the hours after the
  boundary at which a passable PSR pixel of that class is next to the frontier, and the id of that region.
* the edges to seeds of the next Earth cycle: a seed is reached when the frontier holds its pixel at its
  ``before_step`` (blank before steps are never reached).

The passable mask depends on the terrain only: a valid height, a slope of at most ``max_slope_deg`` (Horn, as
``lunarscout.slope``) and a patch that has a light-store file. It is cached next to the output.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing
import os
import signal
import time as clock
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .file_format import PATCH_SIZE
from .safe_haven_flood_packed import WORD_BITS, pack_bits
from .safe_haven_plan import SEED_COLUMNS, read_seed_table
from .safe_haven_stage0 import LightStoreDirectory, patch_file
from .safe_haven_stage1_store import _INVALID_NODE_M, store_identity

logger = logging.getLogger(__name__)

WORDS_PER_PATCH_ROW = PATCH_SIZE // WORD_BITS
_OPERATING_PLANE = 1
FLOOD_COLUMNS = (
    "seed_id", "seed_passable", "lit_endurance_hours", "reach48_pixels",
    "touch_small_hours", "touch_small_psr", "touch_medium_hours", "touch_medium_psr",
    "touch_large_hours", "touch_large_psr",
)
EDGE_COLUMNS = ("seed_id", "target_id", "arrival_hours")
PSR_CLASSES = ("small", "medium", "large")
EDGE_RULES = ("next_cycle", "any_later")


@dataclass(frozen=True)
class FloodSettings:
    """The flood's parameters."""

    max_days: float = 28.0
    moves_per_step: int = 4
    connectivity: int = 8
    max_slope_deg: float = 20.0
    #: PSR classes by area in square metres: small is [0, small_max), medium [small_max, medium_max], large above.
    psr_small_max_m2: float = 40.0
    psr_medium_max_m2: float = 1_000_000.0
    reach_hours: float = 48.0
    #: Which seeds a flood may arrive at: ``"next_cycle"`` (the seeds whose Earth cycle follows the seed's) or
    #: ``"any_later"`` (every seed whose before boundary is later than the flood's start).
    edge_rule: str = "next_cycle"
    unit_cache_mib: int = 1024

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_days) and self.max_days > 0):
            raise ValueError("max_days must be positive")
        if int(self.moves_per_step) != self.moves_per_step or self.moves_per_step < 1:
            raise ValueError("moves_per_step must be a positive integer")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if not 0 < self.max_slope_deg < 90:
            raise ValueError("max_slope_deg must be in (0, 90)")
        if not 0 <= self.psr_small_max_m2 <= self.psr_medium_max_m2:
            raise ValueError("the PSR class limits must be ordered")
        if self.edge_rule not in EDGE_RULES:
            raise ValueError(f"edge_rule must be one of {EDGE_RULES}")


# ---------------------------------------------------------------------------
# Terrain: the passable mask and the PSR class masks, packed over the whole raster
# ---------------------------------------------------------------------------

_STRIP_ROWS = 512


def passable_mask(dem, georef, directory: LightStoreDirectory, max_slope_deg: float) -> npt.NDArray[np.uint64]:
    """Packed passable pixels ``(height, ceil(width / 64))``: valid heights, slope at most ``max_slope_deg``
    (Horn, ``lunarscout.slope``; a pixel next to an invalid height or the raster's edge has none) and a patch
    with a light-store file."""
    from ..terrain import slope

    height, width = dem.height, dem.width
    words = -(-width // WORD_BITS)
    out = np.zeros((height, words), dtype=np.uint64)
    present = np.array([[patch_file(directory.directory, r, c).exists()
                         for c in range(-(-width // PATCH_SIZE))] for r in range(-(-height // PATCH_SIZE))])
    for row0 in range(0, height, _STRIP_ROWS):
        row1 = min(height, row0 + _STRIP_ROWS)
        top, bottom = max(0, row0 - 1), min(height, row1 + 1)
        strip = np.asarray(dem.elevation_m[top:bottom], dtype=np.float64)
        strip = np.where(strip > _INVALID_NODE_M, strip, np.nan)
        degrees, _ = slope(strip, dataclasses.replace(georef, height=strip.shape[0]), output_nodata=-1.0)
        degrees = degrees[row0 - top: row0 - top + (row1 - row0)]
        with np.errstate(invalid="ignore"):
            ok = (degrees >= 0.0) & (degrees <= max_slope_deg)
        rows = np.arange(row0, row1)[:, None] // PATCH_SIZE
        columns = np.arange(width)[None, :] // PATCH_SIZE
        ok &= present[rows, columns]
        out[row0:row1] = pack_bits(ok)
    return out


def psr_class_masks(ids_path: str | Path, table_path: str | Path, settings: FloodSettings):
    """``(ids, (small, medium, large))``: the PSR id raster (int32) and the packed mask of each class."""
    from ..geotiff import read_geotiff
    from .safe_haven_psr import read_region_table

    ids, _georef = read_geotiff(ids_path)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    regions = read_region_table(table_path)
    if int(ids.max(initial=0)) > len(regions):
        raise InputError("the PSR id raster holds ids the region table does not", code="floods_psr_mismatch")
    category = np.zeros(len(regions) + 1, dtype=np.int8)
    category[1:] = np.where(regions.area_m2 < settings.psr_small_max_m2, 1,
                            np.where(regions.area_m2 <= settings.psr_medium_max_m2, 2, 3))
    words = -(-ids.shape[1] // WORD_BITS)
    masks = [np.zeros((ids.shape[0], words), dtype=np.uint64) for _ in PSR_CLASSES]
    for row0 in range(0, ids.shape[0], _STRIP_ROWS):
        classes = category[ids[row0:row0 + _STRIP_ROWS]]
        for number, mask in enumerate(masks, start=1):
            mask[row0:row0 + _STRIP_ROWS] = pack_bits(classes == number)
    return ids, tuple(masks)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


def build_advance(jit):
    """The flood kernel with ``jit`` applied to it and its helpers (identity for plain Python)."""

    @jit
    def popcount(value):
        one, two, four = np.uint64(1), np.uint64(2), np.uint64(4)
        value = value - ((value >> one) & np.uint64(0x5555555555555555))
        value = (value & np.uint64(0x3333333333333333)) + ((value >> two) & np.uint64(0x3333333333333333))
        value = (value + (value >> four)) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return (value * np.uint64(0x0101010101010101)) >> np.uint64(56)

    @jit
    def spread(cur, row, word, words):
        one, top = np.uint64(1), np.uint64(63)
        x = cur[row, word]
        below = cur[row, word - 1] if word > 0 else np.uint64(0)
        above = cur[row, word + 1] if word + 1 < words else np.uint64(0)
        return x | (x << one) | (below >> top) | (x >> one) | (above << top)

    @jit
    def lowest_bit(value):
        low = value & (~value + np.uint64(1))
        return np.int64(popcount(low - np.uint64(1)))

    @jit
    def advance(cur, nxt, lit, nsteps, moves, moves_done, seed_row, seed_col, passable, psr_small, psr_medium,
                psr_large, eight, step0, count_step, touch_step, touch_row, touch_col,
                t_row, t_col, t_step, t_hit):
        """Advance the frontier through ``nsteps`` steps; returns ``(done, flips, count, died)``.

        ``lit[k]`` is the lit mask of flood step ``step0 + k + 1`` and ``moves_done`` the moves already made
        (they set the box around the seed that the frontier can fill). ``done`` is the steps completed with a
        non-empty frontier; ``died`` says the frontier emptied in the next one. ``flips`` is odd when the
        live frontier is in ``nxt``. ``count`` is the frontier's size after step ``count_step`` (-1 if not
        reached). A PSR class touched for the first time records its flood step and pixel in ``touch_*``;
        a target ``t`` whose flood step ``t_step[t]`` is reached with its pixel in the frontier sets ``t_hit[t]``.
        """
        rows, words = passable.shape
        zero = np.uint64(0)
        flips = 0
        count = -1
        for k in range(nsteps):
            j = step0 + k + 1
            for mv in range(moves):
                move = moves_done + k * moves + mv + 1
                row_first = max(0, seed_row - move)
                row_last = min(rows - 1, seed_row + move)
                word_first = max(0, seed_col - move) >> 6
                word_last = min(words * 64 - 1, seed_col + move) >> 6
                any_bits = zero
                for row in range(row_first, row_last + 1):
                    for word in range(word_first, word_last + 1):
                        grown = spread(cur, row, word, words)
                        if eight:
                            if row > 0:
                                grown |= spread(cur, row - 1, word, words)
                            if row + 1 < rows:
                                grown |= spread(cur, row + 1, word, words)
                        else:
                            if row > 0:
                                grown |= cur[row - 1, word]
                            if row + 1 < rows:
                                grown |= cur[row + 1, word]
                        base = grown & passable[row, word]
                        if touch_step[0] < 0:
                            hit = base & psr_small[row, word]
                            if hit != zero:
                                touch_step[0], touch_row[0] = j, row
                                touch_col[0] = word * 64 + lowest_bit(hit)
                        if touch_step[1] < 0:
                            hit = base & psr_medium[row, word]
                            if hit != zero:
                                touch_step[1], touch_row[1] = j, row
                                touch_col[1] = word * 64 + lowest_bit(hit)
                        if touch_step[2] < 0:
                            hit = base & psr_large[row, word]
                            if hit != zero:
                                touch_step[2], touch_row[2] = j, row
                                touch_col[2] = word * 64 + lowest_bit(hit)
                        value = base & lit[k, row, word]
                        nxt[row, word] = value
                        any_bits |= value
                cur, nxt = nxt, cur
                flips += 1
                if any_bits == zero:
                    return k, flips, count, 1
            if j == count_step:
                count = 0
                row_first = max(0, seed_row - (moves_done + (k + 1) * moves))
                row_last = min(rows - 1, seed_row + moves_done + (k + 1) * moves)
                for row in range(row_first, row_last + 1):
                    for word in range(words):
                        count += np.int64(popcount(cur[row, word]))
            for t in range(len(t_step)):
                if t_step[t] == j:
                    if (cur[t_row[t], t_col[t] >> 6] >> np.uint64(t_col[t] & 63)) & np.uint64(1):
                        t_hit[t] = 1
        return nsteps, flips, count, 0

    return advance


advance_python = build_advance(lambda function: function)
_COMPILED: dict = {}


def advance_kernel(compiled: bool = True):
    if not compiled:
        return advance_python
    if "advance" not in _COMPILED:
        from numba import njit

        _COMPILED["advance"] = build_advance(njit(cache=False, nogil=True))
    return _COMPILED["advance"]


# ---------------------------------------------------------------------------
# Lit units from the store
# ---------------------------------------------------------------------------


class UnitCache:
    """Operating-plane units ``(steps_per_day, 128, 2)`` of ``(patch row, patch column, day)``, read from the
    light store a stored chunk at a time and kept within a byte budget (least recently used first out).
    ``None`` is a dark unit (no file, or outside the stored range)."""

    def __init__(self, directory: LightStoreDirectory, dem_shape: tuple[int, int], steps_per_day: int,
                 budget_bytes: int) -> None:
        self.directory = directory
        self.dem_shape = dem_shape
        self.spd = steps_per_day
        self.budget = int(budget_bytes)
        self._units: OrderedDict[tuple[int, int, int], npt.NDArray[np.uint64] | None] = OrderedDict()
        self._absent: set[tuple[int, int]] = set()
        self._bytes = 0
        self.hits = self.misses = 0

    def outside(self, patch_row: int, patch_column: int) -> bool:
        return (patch_row < 0 or patch_column < 0 or patch_row * PATCH_SIZE >= self.dem_shape[0]
                or patch_column * PATCH_SIZE >= self.dem_shape[1])

    def _put(self, key, bits) -> None:
        self._units[key] = bits
        self._bytes += 0 if bits is None else bits.nbytes
        while self._bytes > self.budget and len(self._units) > 1:
            _key, old = self._units.popitem(last=False)
            self._bytes -= 0 if old is None else old.nbytes

    def get(self, patch_row: int, patch_column: int, day: int) -> npt.NDArray[np.uint64] | None:
        key = (patch_row, patch_column, day)
        if key in self._units:
            self._units.move_to_end(key)
            self.hits += 1
            return self._units[key]
        self.misses += 1
        spd, first, stop = self.spd, self.directory.first_step, self.directory.stop_step
        if (patch_row, patch_column) in self._absent or self.outside(patch_row, patch_column):
            self._put(key, None)
            return None
        low, high = day * spd, (day + 1) * spd
        if high <= first or low >= stop:
            self._put(key, None)
            return None
        opened = self.directory.open_patch(patch_row, patch_column)
        if opened is None:
            self._absent.add((patch_row, patch_column))
            self._put(key, None)
            return None
        result = None
        with opened:
            per_chunk = self.directory.settings.steps_per_chunk
            aligned = (first % spd == 0) and (per_chunk % spd == 0)
            if aligned:                    # decode the whole stored chunk and keep every day in it
                block = (low - first) // per_chunk if low >= first else 0
                lo, hi = first + block * per_chunk, min(first + (block + 1) * per_chunk, stop)
                plane = opened.read_steps(lo, hi)[_OPERATING_PLANE]
                for d in range(lo // spd, -(-hi // spd)):
                    bits = np.zeros((spd, PATCH_SIZE, WORDS_PER_PATCH_ROW), dtype=np.uint64)
                    a, b = max(lo, d * spd), min(hi, (d + 1) * spd)
                    bits[a - d * spd: b - d * spd] = plane[a - lo: b - lo]
                    self._put((patch_row, patch_column, d), bits)
                    if d == day:
                        result = bits
            else:
                bits = np.zeros((spd, PATCH_SIZE, WORDS_PER_PATCH_ROW), dtype=np.uint64)
                a, b = max(low, first), min(high, stop)
                bits[a - low: b - low] = opened.read_steps(a, b)[_OPERATING_PLANE]
                self._put(key, bits)
                result = bits
        return result


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


@dataclass
class FloodContext:
    """What every flood reads; built once, before the workers fork."""

    settings: FloodSettings
    dem_shape: tuple[int, int]
    step_minutes: int
    passable: npt.NDArray[np.uint64]
    psr_masks: tuple
    psr_ids: npt.NDArray[np.int32]
    directory: LightStoreDirectory
    seeds: list[dict]
    by_cycle: dict
    later: dict
    unit_budget: int
    # per process
    cache: UnitCache | None = None
    buffers: dict | None = None

    @property
    def steps_per_day(self) -> int:
        return 1440 // self.step_minutes

    @property
    def max_steps(self) -> int:
        return int(math.floor(self.settings.max_days * 1440 / self.step_minutes + 1e-9))

    @property
    def reach_pixels(self) -> int:
        return self.settings.moves_per_step * self.max_steps

    def ensure_process_state(self) -> None:
        if self.cache is None:
            self.cache = UnitCache(self.directory, self.dem_shape, self.steps_per_day, self.unit_budget)
            patches = -(-(self.reach_pixels + PATCH_SIZE - 1) // PATCH_SIZE)
            side = (2 * patches + 1) * PATCH_SIZE
            self.buffers = {
                "patches": patches,
                "lit": np.zeros((self.steps_per_day, side, side // WORD_BITS), dtype=np.uint64),
                "cur": np.zeros((side, side // WORD_BITS), dtype=np.uint64),
                "nxt": np.zeros((side, side // WORD_BITS), dtype=np.uint64),
            }


def build_targets(seeds: list[dict], settings: FloodSettings):
    """Index the seeds that a flood can arrive at: ``(by_cycle, later)`` as arrays ``(id, row, column, before)``."""
    usable = [s for s in seeds if s["before_step"] is not None]
    table = np.array([[s["seed_id"], s["row"], s["column"], s["before_step"], s["earth_cycle"]] for s in usable],
                     dtype=np.int64).reshape(-1, 5)
    by_cycle = {int(c): table[table[:, 4] == c][:, :4] for c in np.unique(table[:, 4]) if c >= 0}
    return by_cycle, table[:, :4]


def flood_seed(context: FloodContext, seed: dict) -> tuple[dict, list[tuple[int, int]], dict]:
    """Flood one seed: ``(row, edges, counts)``. ``row`` follows ``FLOOD_COLUMNS``; ``edges`` are
    ``(target_id, arrival_flood_step)``."""
    context.ensure_process_state()
    settings, cache, buffers = context.settings, context.cache, context.buffers
    height, width = context.dem_shape
    step_hours = context.step_minutes / 60.0
    moves, reach, spd = settings.moves_per_step, context.reach_pixels, context.steps_per_day
    patches = buffers["patches"]
    seed_row, seed_column, after = seed["row"], seed["column"], seed["after_step"]
    patch_row, patch_column = seed_row // PATCH_SIZE, seed_column // PATCH_SIZE
    pr0, pc0 = max(0, patch_row - patches), max(0, patch_column - patches)
    row0, col0 = pr0 * PATCH_SIZE, pc0 * PATCH_SIZE
    row1 = min(height, (patch_row + patches + 1) * PATCH_SIZE)
    col1 = min(width, (patch_column + patches + 1) * PATCH_SIZE)
    rows, words = row1 - row0, -(-(col1 - col0) // WORD_BITS)
    word0 = col0 // WORD_BITS
    passable = context.passable[row0:row1, word0:word0 + words]
    psr = [mask[row0:row1, word0:word0 + words] for mask in context.psr_masks]
    cur, nxt, lit = buffers["cur"][:rows, :words], buffers["nxt"][:rows, :words], buffers["lit"]
    cur[:] = 0
    nxt[:] = 0
    local_row, local_column = seed_row - row0, seed_column - col0
    cur[local_row, local_column >> 6] = np.uint64(1) << np.uint64(local_column & 63)
    seed_ok = bool(context.passable[seed_row, seed_column >> 6] >> np.uint64(seed_column & 63) & np.uint64(1))

    # the seeds this flood can arrive at: inside the reach and due within the flood
    if settings.edge_rule == "next_cycle" and seed["earth_cycle"] >= 0:
        pool = context.by_cycle.get(seed["earth_cycle"] + 1)
    elif settings.edge_rule == "any_later":
        pool = context.later
    else:
        pool = None
    if pool is None or not len(pool):
        pool = np.zeros((0, 4), dtype=np.int64)
    due = pool[:, 3] - after
    near = ((due >= 1) & (due <= context.max_steps) & (pool[:, 0] != seed["seed_id"])
            & (np.abs(pool[:, 1] - seed_row) <= reach) & (np.abs(pool[:, 2] - seed_column) <= reach))
    targets = pool[near]
    t_row = (targets[:, 1] - row0).astype(np.int64)
    t_col = (targets[:, 2] - col0).astype(np.int64)
    t_step = (targets[:, 3] - after).astype(np.int64)
    t_hit = np.zeros(len(targets), dtype=np.int64)
    touch_step = np.full(3, -1, dtype=np.int64)
    touch_row = np.zeros(3, dtype=np.int64)
    touch_col = np.zeros(3, dtype=np.int64)

    kernel = advance_kernel()
    eight = settings.connectivity == 8
    count_step = int(round(settings.reach_hours * 60 / context.step_minutes))
    done_total, moves_done, count, died = 0, 0, 0, False
    total = context.max_steps - 1                     # the seed's own step is step 0 of max_steps
    while done_total < total:
        first_abs = after + done_total + 1             # absolute step of the next flood step
        day = first_abs // spd
        nsteps = min(total - done_total, (day + 1) * spd - first_abs)
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
        done, flips, counted, dead = kernel(
            cur, nxt, lit[:nsteps, :rows, :words], nsteps, moves, moves_done, local_row, local_column, passable,
            psr[0], psr[1], psr[2], eight, done_total, count_step, touch_step, touch_row, touch_col,
            t_row, t_col, t_step, t_hit)
        if flips & 1:
            cur, nxt = nxt, cur
        if counted >= 0:
            count = counted
        done_total += done
        moves_done += done * moves
        if dead:
            died = True
            break
    row = {"seed_id": seed["seed_id"], "seed_passable": int(seed_ok),
           "lit_endurance_hours": (done_total + 1) * step_hours, "reach48_pixels": count}
    for number, name in enumerate(PSR_CLASSES):
        if touch_step[number] >= 0:
            psr_id = int(context.psr_ids[row0 + touch_row[number], col0 + touch_col[number]])
            row[f"touch_{name}_hours"], row[f"touch_{name}_psr"] = int(touch_step[number]) * step_hours, psr_id
        else:
            row[f"touch_{name}_hours"] = row[f"touch_{name}_psr"] = None
    edges = [(int(targets[i, 0]), int(t_step[i])) for i in np.flatnonzero(t_hit)]
    return row, edges, {"died": died}


def flood_group(context: FloodContext, seed_ids: list[int]):
    """Flood the seeds of one cell-month: ``(first id, rows, edges)``."""
    rows, edges = [], []
    step_hours = context.step_minutes / 60.0
    for seed_id in seed_ids:
        row, found, _counts = flood_seed(context, context.seeds[seed_id - 1])
        rows.append(row)
        edges.extend((seed_id, target, step * step_hours) for target, step in found)
    return seed_ids[0], rows, edges


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

_RUN: dict = {}


def _flood_init() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _flood_task(seed_ids: list[int]):
    return flood_group(_RUN["context"], seed_ids)


def _format(value) -> str:
    if value is None:
        return ""
    return f"{value:g}" if isinstance(value, float) else str(value)


def _write_csv(path: Path, columns, rows) -> None:
    staging = path.with_name(path.name + ".partial")
    try:
        with staging.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            for row in rows:
                writer.writerow([_format(row[name]) for name in columns] if isinstance(row, dict)
                                else [_format(v) for v in row])
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def read_flood_table(path: str | Path) -> list[dict]:
    """The per-seed table written by :func:`run_floods`; numbers are ints or floats, blanks are ``None``."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FLOOD_COLUMNS:
            raise InputError(f"{path} is not a flood table", code="floods_table_unreadable")
        rows = []
        for record in reader:
            row = {}
            for name, value in record.items():
                row[name] = None if value == "" else (float(value) if "hours" in name else int(value))
            rows.append(row)
    return rows


def read_edge_table(path: str | Path) -> list[tuple[int, int, float]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader, ())) != EDGE_COLUMNS:
            raise InputError(f"{path} is not an edge table", code="floods_table_unreadable")
        return [(int(a), int(b), float(c)) for a, b, c in reader]


@dataclass
class FloodReport:
    seeds: int
    seeds_resumed: int
    seeds_not_on_passable_terrain: int
    seeds_at_cap: int
    seeds_with_edges: int
    edges: int
    touches: dict
    settings: FloodSettings
    passable_pixels: int
    seconds: float = 0.0

    def as_dict(self) -> dict:
        record = asdict(self)
        record["settings"] = asdict(self.settings)
        return record


def _file_identity(path) -> list:
    stat = Path(path).stat()
    return [str(path), int(stat.st_size), int(stat.st_mtime_ns)]


def run_floods(
    output_path: str | Path,
    edges_path: str | Path,
    *,
    seeds_path: str | Path,
    dem,
    georef,
    directory: LightStoreDirectory,
    psr_ids_path: str | Path,
    psr_table_path: str | Path,
    passable_path: str | Path,
    settings: FloodSettings = FloodSettings(),
    seed_ids: list[int] | None = None,
    workers: int = 1,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> FloodReport:
    """Flood every seed of the stage-2 table and write the per-seed table and the edge table.

    The run checkpoints to ``output_path + ".checkpoint.jsonl"`` after every cell-month and resumes from it with
    the same inputs and settings; the checkpoint is removed when the tables are complete. ``seed_ids`` limits the
    run to those seeds (a trial). The tables do not depend on ``workers``.
    """
    started = clock.monotonic()
    output_path, edges_path, passable_path = Path(output_path), Path(edges_path), Path(passable_path)
    report_path = output_path.with_name(output_path.name + ".report.json")
    checkpoint = output_path.with_name(output_path.name + ".checkpoint.jsonl")
    for path in (output_path, edges_path, report_path):
        if path.exists() and not overwrite:
            raise InputError(f"{path} exists; pass overwrite=True to replace it", code="floods_output_exists",
                             details={"path": str(path)})
    seeds = read_seed_table(seeds_path)
    if [s["seed_id"] for s in seeds] != list(range(1, len(seeds) + 1)):
        raise InputError("the seed ids are not 1..N in order", code="floods_seeds_unreadable")
    step_minutes = directory.settings.step_minutes
    if 1440 % step_minutes:
        raise InputError("the light store's step does not divide a day", code="floods_step")

    passable = _load_passable(passable_path, dem, georef, directory, settings)
    ids, masks = psr_class_masks(psr_ids_path, psr_table_path, settings)
    if ids.shape != (dem.height, dem.width):
        raise InputError("the PSR id raster and the DEM differ in size", code="floods_grid_mismatch")
    by_cycle, later = build_targets(seeds, settings)
    context = FloodContext(
        settings=settings, dem_shape=(dem.height, dem.width), step_minutes=step_minutes, passable=passable,
        psr_masks=masks, psr_ids=ids, directory=directory, seeds=seeds, by_cycle=by_cycle, later=later,
        unit_budget=settings.unit_cache_mib * 2**20)

    wanted = set(range(1, len(seeds) + 1)) if seed_ids is None else {int(i) for i in seed_ids}
    groups: dict[tuple, list[int]] = {}
    for seed in seeds:
        if seed["seed_id"] in wanted:
            groups.setdefault((seed["month"], seed["cell_row"], seed["cell_col"]), []).append(seed["seed_id"])
    tasks = list(groups.values())
    fingerprint = {
        "seeds": _file_identity(seeds_path), "store": store_identity(directory), "settings": asdict(settings),
        "psr": [_file_identity(psr_ids_path), _file_identity(psr_table_path)],
        "seed_ids": [len(wanted), hashlib.sha256(str(sorted(wanted)).encode()).hexdigest()],
    }
    done_rows, done_edges, done_keys = _read_checkpoint(checkpoint, fingerprint)
    if not done_keys:
        checkpoint.write_text(json.dumps({"type": "header", "fingerprint": fingerprint}) + "\n", encoding="utf-8")
    pending = [t for t in tasks if t[0] not in done_keys]
    resumed = sum(len(t) for t in tasks if t[0] in done_keys)

    pool = None
    try:
        with checkpoint.open("a", encoding="utf-8") as log:
            if workers > 1 and pending:
                _RUN["context"] = context
                pool = multiprocessing.get_context("fork").Pool(workers, initializer=_flood_init)
                results = pool.imap_unordered(_flood_task, pending, chunksize=1)
            else:
                results = (flood_group(context, t) for t in pending)
            for number, (key, rows, edges) in enumerate(results, start=1):
                done_rows.extend(rows)
                done_edges.extend(edges)
                log.write(json.dumps({"type": "group", "key": key, "rows": rows, "edges": edges}) + "\n")
                log.flush()
                if progress is not None:
                    progress(number, len(pending))
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        _RUN.clear()

    done_rows.sort(key=lambda r: r["seed_id"])
    done_edges.sort(key=lambda e: (e[0], e[1]))
    _write_csv(output_path, FLOOD_COLUMNS, done_rows)
    _write_csv(edges_path, EDGE_COLUMNS, done_edges)
    cap_hours = context.max_steps * context.step_minutes / 60.0
    report = FloodReport(
        seeds=len(done_rows), seeds_resumed=resumed,
        seeds_not_on_passable_terrain=sum(not r["seed_passable"] for r in done_rows),
        seeds_at_cap=sum(r["lit_endurance_hours"] >= cap_hours for r in done_rows),
        seeds_with_edges=len({e[0] for e in done_edges}), edges=len(done_edges),
        touches={name: sum(r[f"touch_{name}_hours"] is not None for r in done_rows) for name in PSR_CLASSES},
        settings=settings, passable_pixels=int(sum(int(np.unpackbits(passable[i:i + 4096].view(np.uint8)).sum())
                                                   for i in range(0, passable.shape[0], 4096))),
        seconds=clock.monotonic() - started)
    report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    return report


def _read_checkpoint(path: Path, fingerprint: dict):
    rows, edges, keys = [], [], set()
    if not path.exists():
        return rows, edges, keys
    with path.open(encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    try:
        header = json.loads(lines[0])
        same = header.get("type") == "header" and header["fingerprint"] == fingerprint
    except (IndexError, ValueError, KeyError):
        same = False
    if not same:
        raise InputError(f"{path} was made with other inputs or settings; delete it to start again",
                         code="floods_checkpoint_mismatch", details={"path": str(path)})
    for line in lines[1:]:
        try:
            record = json.loads(line)
        except ValueError:
            break                                    # a torn last line: that group is flooded again
        rows.extend(record["rows"])
        edges.extend(tuple(e) for e in record["edges"])
        keys.add(record["key"])
    return rows, edges, keys


def _load_passable(path: Path, dem, georef, directory: LightStoreDirectory, settings: FloodSettings):
    identity = hashlib.sha256(json.dumps(
        {"store": store_identity(directory), "slope": settings.max_slope_deg, "shape": [dem.height, dem.width]},
        sort_keys=True).encode()).hexdigest()
    sidecar = path.with_name(path.name + ".json")
    if path.exists() and sidecar.exists():
        try:
            if json.loads(sidecar.read_text())["identity"] == identity:
                return np.load(path, mmap_mode="r")
        except (ValueError, KeyError, OSError):
            pass
    mask = passable_mask(dem, georef, directory, settings.max_slope_deg)
    staging = path.with_name(path.name + ".partial.npy")
    np.save(staging, mask)
    os.replace(staging, path)
    sidecar.write_text(json.dumps({"identity": identity, "max_slope_deg": settings.max_slope_deg}) + "\n")
    return mask
