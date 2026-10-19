"""Stage-2 pass 1: seeds, boundaries and lighting ranges for 5 km safe-haven cells.

The rules are specified in ``docs/safe-haven-windows-spec.md`` (sections 3 to 6
and 8, 9).  Pass 1 reads the stage-1 rasters, picks up to ``max_seeds_per_cell_month``
seeds in every cell and month that has a haven, recomputes each seed's hibernation
with the stage-1 rules, finds its two flood boundaries, and reports how much
lighting the floods will need.  It runs no flood and no patch lighting.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .file_format import AZIMUTH_COUNT, PATCH_SIZE, HorizonTileStore, read_patch_horizons
from .safe_haven import OPERATING_FUZZ, _month_indices_map, build_month_bands, merge_gap_limit

MINUTES_PER_DAY = 1440
MICROSECONDS_PER_MINUTE = 60_000_000
MICROSECONDS_PER_HOUR = 3_600_000_000

VectorSource = Callable[[npt.NDArray[np.datetime64]], npt.ArrayLike]


class PlanConsistencyError(RuntimeError):
    """Pass 1 recomputed a stage-1 value and got a different one."""


# ---------------------------------------------------------------------------
# Parameters and the flood time grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowParameters:
    """The stage-2 parameters of the specification, section 2."""

    survival_limit_hours: float = 72.0
    operational_hours: float = 48.0
    rover_speed_m_s: float = 0.01
    connectivity: int = 8
    max_seeds_per_cell_month: int = 5
    seed_separation_m: float = 250.0
    seed_timing_tolerance_hours: float = 6.0
    haven_neighbourhood_m: float = 100.0
    min_nearby_haven_fraction: float = 0.05
    boundary_search_hours: float = 24.0
    operating_sunlight_fraction: float = 1.0
    cell_patches: int = 2
    #: Pixels in a near-duplicate probe budget: how many pixels within the
    #: separation of an accepted seed are examined for a different timing.
    timing_probes_per_seed: int = 8

    def __post_init__(self) -> None:
        positive = (
            "survival_limit_hours", "operational_hours", "rover_speed_m_s",
            "seed_separation_m", "haven_neighbourhood_m", "boundary_search_hours",
            "operating_sunlight_fraction",
        )
        for name in positive:
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite")
        for name in ("seed_timing_tolerance_hours",):
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if not 0.0 <= self.min_nearby_haven_fraction <= 1.0:
            raise ValueError("min_nearby_haven_fraction must be in [0, 1]")
        for name in ("max_seeds_per_cell_month", "cell_patches", "timing_probes_per_seed"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def flood_step_minutes(pixel_size_m: float, rover_speed_m_s: float) -> int:
    """The flood step: the largest whole-minute divisor of 24 h not above the
    time the rover needs to cross one pixel (specification, section 6)."""
    implied = pixel_size_m / rover_speed_m_s / 60.0
    candidates = [
        minutes for minutes in range(1, MINUTES_PER_DAY + 1)
        if MINUTES_PER_DAY % minutes == 0 and minutes <= implied + 1e-9
    ]
    if not candidates:
        raise ValueError("the rover crosses a pixel in under a minute; no flood step fits")
    return max(candidates)


@dataclass(frozen=True)
class FloodGrid:
    """Times ``k * step_minutes`` after 1970-01-01T00:00:00Z.

    The step divides 24 hours, so 24 h chunks are whole numbers of steps and lie
    on the same grid in every run.
    """

    step_minutes: int

    def __post_init__(self) -> None:
        if self.step_minutes < 1 or MINUTES_PER_DAY % self.step_minutes:
            raise ValueError("the flood step must divide 24 hours")

    @property
    def step_us(self) -> int:
        return self.step_minutes * MICROSECONDS_PER_MINUTE

    @property
    def steps_per_chunk(self) -> int:
        return MINUTES_PER_DAY // self.step_minutes

    def floor_index(self, time_us: int) -> int:
        return int(time_us) // self.step_us

    def ceil_index(self, time_us: int) -> int:
        return -((-int(time_us)) // self.step_us)

    def times(self, indices: npt.ArrayLike) -> npt.NDArray[np.datetime64]:
        return (np.asarray(indices, dtype=np.int64) * self.step_us).astype("datetime64[us]")

    def step_count(self, hours: float) -> int:
        return int(math.ceil(hours * 60.0 / self.step_minutes - 1e-9))


# ---------------------------------------------------------------------------
# Raster neighbourhoods
# ---------------------------------------------------------------------------


def disk_offsets(radius_m: float, pixel_size_m: float) -> list[tuple[int, int]]:
    """Pixel offsets whose centres are within ``radius_m`` of the centre (inclusive)."""
    reach = int(math.floor(radius_m / pixel_size_m + 1e-9))
    limit = (radius_m / pixel_size_m) ** 2 + 1e-9
    return [
        (dr, dc)
        for dr in range(-reach, reach + 1)
        for dc in range(-reach, reach + 1)
        if dr * dr + dc * dc <= limit
    ]


def nearby_haven_counts(
    haven: npt.NDArray[np.bool_], radius_m: float, pixel_size_m: float
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """For every pixel, haven pixels and pixels inside the DEM within the disk.

    Pixels beyond the array edge are outside the DEM and are not counted.
    """
    offsets = disk_offsets(radius_m, pixel_size_m)
    reach = max(max(abs(dr), abs(dc)) for dr, dc in offsets)
    height, width = haven.shape
    padded = np.zeros((height + 2 * reach, width + 2 * reach), dtype=np.int32)
    inside = np.zeros_like(padded)
    padded[reach:reach + height, reach:reach + width] = haven
    inside[reach:reach + height, reach:reach + width] = 1
    haven_count = np.zeros((height, width), dtype=np.int32)
    pixel_count = np.zeros((height, width), dtype=np.int32)
    for dr, dc in offsets:
        window = (slice(reach + dr, reach + dr + height), slice(reach + dc, reach + dc + width))
        haven_count += padded[window]
        pixel_count += inside[window]
    return haven_count, pixel_count


def box_sums(values: npt.NDArray[np.floating], half_width: int) -> npt.NDArray[np.float64]:
    """Sum of ``values`` over the square of side ``2 * half_width + 1``, clipped to
    the array, for every pixel.  NaN counts as 0.  Linear time."""
    height, width = values.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(np.nan_to_num(values, nan=0.0).astype(np.float64), axis=0), axis=1)
    rows = np.arange(height)
    columns = np.arange(width)
    top = np.clip(rows - half_width, 0, height)
    bottom = np.clip(rows + half_width + 1, 0, height)
    left = np.clip(columns - half_width, 0, width)
    right = np.clip(columns + half_width + 1, 0, width)
    return (
        integral[np.ix_(bottom, right)] - integral[np.ix_(top, right)]
        - integral[np.ix_(bottom, left)] + integral[np.ix_(top, left)]
    )


# ---------------------------------------------------------------------------
# One pixel's hibernation of record (stage-1 rules, specification section 3)
# ---------------------------------------------------------------------------


def shadow_run_spans(is_shadow: npt.NDArray[np.bool_], gap_limit: int) -> npt.NDArray[np.int64]:
    """Merged shadow runs as ``(first, last)`` rows.  A sunlit gap of ``gap_limit``
    samples or more splits runs; a shorter gap, or none, joins them."""
    index = np.flatnonzero(is_shadow)
    if index.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    split = np.flatnonzero(np.diff(index) - 1 >= gap_limit)
    firsts = np.concatenate(([index[0]], index[split + 1]))
    lasts = np.concatenate((index[split], [index[-1]]))
    return np.stack((firsts, lasts), axis=1).astype(np.int64)


def outage_spans(is_outage: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    """Hibernations as ``(first, last)`` rows: maximal stretches of outage samples."""
    flags = np.concatenate(([False], np.asarray(is_outage, dtype=bool), [False]))
    change = np.flatnonzero(flags[1:] != flags[:-1])
    return np.stack((change[0::2], change[1::2] - 1), axis=1).astype(np.int64)


@dataclass(frozen=True)
class HibernationRecord:
    """The hibernation that sets a month's value, in global sample indices."""

    score_samples: int
    outage_first: int
    outage_last: int
    run_first: int      # the longest touching run; -1 when the score is 0
    run_last: int


@dataclass(frozen=True)
class RecordSearch:
    record: HibernationRecord | None
    open_start: bool    # a touching run may begin before the examined span
    open_end: bool      # a credited hibernation or touching run may continue past it


def hibernation_of_record(
    is_shadow: npt.NDArray[np.bool_],
    is_outage: npt.NDArray[np.bool_],
    sample_month: npt.NDArray[np.integer],
    month: int,
    *,
    gap_limit: int,
    offset: int,
) -> RecordSearch:
    """The hibernation starting in ``month`` with the lowest score (the earlier on a tie).

    The arrays cover global samples ``offset .. offset + len - 1``.  A hibernation
    already under way at the first examined sample has an unknown start and is not
    credited, as in stage 1.  ``open_start`` and ``open_end`` say whether the span
    is too short to be sure of the score.
    """
    count = len(is_shadow)
    gap_limit = max(1, int(gap_limit))
    runs = shadow_run_spans(is_shadow, gap_limit)
    best: HibernationRecord | None = None
    best_open_start = open_end = False
    for first, last in outage_spans(is_outage):
        if first == 0 or sample_month[first] != month:
            continue
        touching = runs[(runs[:, 0] <= last) & (runs[:, 1] >= first)]
        open_here = False
        if touching.size:
            lengths = touching[:, 1] - touching[:, 0] + 1
            longest = int(np.argmax(lengths))
            run_first, run_last = (int(value) for value in touching[longest])
            score = int(lengths[longest])
            for touch_first, touch_last in touching:
                if offset > 0 and touch_first < gap_limit:
                    open_here = True
                if count - 1 - touch_last < gap_limit:
                    open_end = True
        else:
            run_first = run_last = -1
            score = 0
        if last == count - 1:
            open_end = True
        candidate = HibernationRecord(
            score, offset + int(first), offset + int(last),
            -1 if run_first < 0 else offset + run_first,
            -1 if run_last < 0 else offset + run_last,
        )
        if best is None or score < best.score_samples:
            best, best_open_start = candidate, open_here
    # A run that reaches back past the span only makes a hibernation score higher
    # than computed, and the span is long enough that such a score already exceeds
    # any survivable one, so only the record itself needs a known start.
    open_start = best_open_start
    return RecordSearch(best, open_start, open_end)


# ---------------------------------------------------------------------------
# Pixel signals from the horizons
# ---------------------------------------------------------------------------


class PixelSignals:
    """Sun fraction and Earth margin of single pixels, with the patch kernel's arithmetic."""

    def __init__(self, dem, horizon_store: HorizonTileStore, observer_height_m: float = 0.0,
                 *, patch_cache_size: int = 4) -> None:
        from .lightmap_cpu import LightmapCpuSession

        self._dem = dem
        self._store = horizon_store
        self._observer = float(observer_height_m)
        self._session = LightmapCpuSession(time_batch_size=1)
        self._cache: OrderedDict[tuple[int, int], npt.NDArray[np.float32] | None] = OrderedDict()
        self._cache_size = int(patch_cache_size)

    def horizon_column(self, row: int, column: int) -> npt.NDArray[np.float32] | None:
        key = (row // PATCH_SIZE * PATCH_SIZE, column // PATCH_SIZE * PATCH_SIZE)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = read_patch_horizons(self._store, key[0], key[1], self._observer)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        patch = self._cache[key]
        return None if patch is None else patch[row - key[0], column - key[1]]

    def series(self, row: int, column: int, vectors_m: npt.ArrayLike, *, margin: bool):
        """Float32 values per vector, or ``None`` when the pixel has no horizon."""
        horizon = self.horizon_column(row, column)
        if horizon is None:
            return None
        return self._session.pixel_series(
            self._dem, horizon, vectors_m, row=row, column=column, margin=margin
        )


# ---------------------------------------------------------------------------
# Time context shared by every pixel
# ---------------------------------------------------------------------------


@dataclass
class TimeContext:
    """The stage-1 sampling and the vectors on it."""

    times_us: npt.NDArray[np.int64]
    step_hours: float
    sample_month: npt.NDArray[np.int32]
    month_first: npt.NDArray[np.int64]
    month_last: npt.NDArray[np.int64]
    month_count: int
    sun_vectors: npt.NDArray[np.float64]
    earth_vectors: npt.NDArray[np.float64]
    sun_vectors_at: VectorSource
    earth_vectors_at: VectorSource

    @classmethod
    def build(cls, times: npt.ArrayLike, sun_vectors_at: VectorSource,
              earth_vectors_at: VectorSource) -> "TimeContext":
        stamps = np.asarray(times).astype("datetime64[us]")
        if stamps.ndim != 1 or stamps.size < 2:
            raise ValueError("times must be a one-dimensional array of at least two UTC times")
        times_us = stamps.astype(np.int64)
        steps = np.diff(times_us)
        if np.any(steps != steps[0]) or steps[0] <= 0:
            raise ValueError("times must be uniformly spaced and increasing")
        moments = [value.replace(tzinfo=timezone.utc) for value in stamps.astype(object)]
        bands = build_month_bands(moments)
        sample_month = _month_indices_map(moments, bands).astype(np.int32)
        month_first = np.array([int(np.flatnonzero(sample_month == m)[0]) for m in range(len(bands))])
        month_last = np.array([int(np.flatnonzero(sample_month == m)[-1]) for m in range(len(bands))])
        return cls(
            times_us=times_us,
            step_hours=float(steps[0]) / MICROSECONDS_PER_HOUR,
            sample_month=sample_month,
            month_first=month_first,
            month_last=month_last,
            month_count=len(bands),
            sun_vectors=np.asarray(sun_vectors_at(stamps), dtype=np.float64),
            earth_vectors=np.asarray(earth_vectors_at(stamps), dtype=np.float64),
            sun_vectors_at=sun_vectors_at,
            earth_vectors_at=earth_vectors_at,
        )


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelRecord:
    """A candidate's hibernation of record with its timing, before any boundary search."""

    row: int
    column: int
    score_hours: float
    outage_first: int
    outage_last: int
    run_first: int
    run_last: int
    outage_start_us: int          # T_s: the first outage sample
    outage_end_us: int            # T_e: the first Earth-available sample after it
    earth_visible_before_hours: float
    earth_visible_after_hours: float


@dataclass(frozen=True)
class Seed:
    """A seed with its flood boundaries (flood-grid step indices)."""

    row: int
    column: int
    hibernation_hours: float
    outage_start_us: int
    outage_end_us: int
    boundary_before_step: int
    boundary_after_step: int
    nearby_haven_count: int
    nearby_pixel_count: int
    nearby_haven_fraction: float
    earth_visible_before_hours: float
    earth_visible_after_hours: float


@dataclass
class PlanCounts:
    """Counters for the pass-1 report; every field adds across cell-months."""

    cell_months_examined: int = 0
    cell_months_with_haven: int = 0
    cell_months_kept: int = 0
    haven_pixels: int = 0
    dropped_by_nearby_fraction: int = 0
    timelines_evaluated: int = 0
    seeds_selected: int = 0
    seeds_valid: int = 0
    invalid_no_horizon: int = 0
    invalid_no_record: int = 0
    invalid_outage_unclosed: int = 0
    invalid_no_lit_boundary: int = 0
    tasks_unshared: int = 0
    tasks_shared: int = 0

    def add(self, other: "PlanCounts") -> None:
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))


class CandidateEvaluator:
    """Recompute a pixel's hibernation of record on the stage-1 grid (section 5, step 5)."""

    def __init__(self, signals: PixelSignals, time: TimeContext, parameters: WindowParameters,
                 *, earth_elevation_threshold_deg: float = 2.0,
                 sunlight_fraction_threshold: float = 0.2, merge_gap_hours: float = 24.0) -> None:
        self.signals = signals
        self.time = time
        self.parameters = parameters
        self.earth_threshold = float(earth_elevation_threshold_deg)
        self.sun_threshold = float(sunlight_fraction_threshold)
        self.gap_limit = max(1, merge_gap_limit(merge_gap_hours, time.step_hours))
        step = time.step_hours
        self.lookback = max(
            int(math.ceil(parameters.survival_limit_hours / step)) + self.gap_limit + 2,
            int(math.ceil(parameters.operational_hours / step)) + 2,
        )
        self.forward = int(math.ceil(15.0 * 24.0 / step))
        self.after_samples = int(math.ceil(parameters.operational_hours / step)) + 1

    def evaluate(self, row: int, column: int, month: int) -> PixelRecord | str:
        """The pixel's record, or the reason it has none: ``no_horizon``, ``no_record``,
        ``outage_unclosed``."""
        time = self.time
        count = len(time.times_us)
        first = int(time.month_first[month])
        last = int(time.month_last[month])
        start = max(0, first - self.lookback)
        stop = min(count, last + 1 + self.forward)
        while True:
            sun = self.signals.series(row, column, time.sun_vectors[start:stop], margin=False)
            if sun is None:
                return "no_horizon"
            earth = self.signals.series(row, column, time.earth_vectors[start:stop], margin=True)
            if not (np.isfinite(sun).all() and np.isfinite(earth).all()):
                return "no_horizon"
            is_shadow = sun.astype(np.float64) < self.sun_threshold
            is_outage = earth.astype(np.float64) < self.earth_threshold
            found = hibernation_of_record(
                is_shadow, is_outage, time.sample_month[start:stop], month,
                gap_limit=self.gap_limit, offset=start,
            )
            if found.record is None:
                if found.open_end and stop < count:
                    stop = min(count, stop + self.forward)
                    continue
                return "no_record"
            if found.open_start:
                raise PlanConsistencyError(
                    f"pixel ({row}, {column}) month {month}: a shadow run reaches back "
                    "past the examined span"
                )
            record = found.record
            needed = record.outage_last + 1 + self.after_samples
            if stop < count and (found.open_end or stop <= needed):
                stop = min(count, max(stop + self.forward, needed + 1))
                continue
            break
        if record.outage_last + 1 >= count:
            return "outage_unclosed"
        available = ~is_outage
        before = _contiguous_true(available[: record.outage_first - start][::-1])
        after = _contiguous_true(available[record.outage_last + 1 - start:])
        cap = self.parameters.operational_hours
        return PixelRecord(
            row=row, column=column,
            score_hours=float(np.float32(record.score_samples) * np.float32(time.step_hours)),
            outage_first=record.outage_first, outage_last=record.outage_last,
            run_first=record.run_first, run_last=record.run_last,
            outage_start_us=int(time.times_us[record.outage_first]),
            outage_end_us=int(time.times_us[record.outage_last + 1]),
            earth_visible_before_hours=min(cap, before * time.step_hours),
            earth_visible_after_hours=min(cap, after * time.step_hours),
        )


def _contiguous_true(flags: npt.NDArray[np.bool_]) -> int:
    if flags.size == 0:
        return 0
    stop = np.flatnonzero(~flags)
    return int(stop[0]) if stop.size else int(flags.size)


def find_boundaries(
    signals: PixelSignals,
    time: TimeContext,
    grid: FloodGrid,
    record: PixelRecord,
    parameters: WindowParameters,
) -> tuple[int, int] | None:
    """The before and after flood boundaries as flood-grid step indices (section 6),
    or ``None`` when the seed is not lit within ``boundary_search_hours``."""
    cap = int(math.floor(parameters.boundary_search_hours * 60.0 / grid.step_minutes + 1e-9))
    nominal_before = grid.floor_index(record.outage_start_us)
    nominal_after = grid.ceil_index(record.outage_end_us)
    before_steps = np.arange(nominal_before - cap, nominal_before + 1, dtype=np.int64)
    after_steps = np.arange(nominal_after, nominal_after + cap + 1, dtype=np.int64)
    steps = np.concatenate((before_steps, after_steps))
    fractions = signals.series(
        record.row, record.column, time.sun_vectors_at(grid.times(steps)), margin=False
    )
    if fractions is None:
        return None
    lit = fractions.astype(np.float64) >= parameters.operating_sunlight_fraction - OPERATING_FUZZ
    before_lit = np.flatnonzero(lit[: before_steps.size])
    after_lit = np.flatnonzero(lit[before_steps.size:])
    if before_lit.size == 0 or after_lit.size == 0:
        return None
    return int(before_steps[before_lit[-1]]), int(after_steps[after_lit[0]])


def select_seeds(
    *,
    hibernation_hours: npt.NDArray[np.float64],
    joint_total_hours: npt.NDArray[np.float64],
    window_origin: tuple[int, int],
    core: tuple[int, int, int, int],
    month: int,
    evaluator: CandidateEvaluator,
    grid: FloodGrid,
    pixel_size_m: float,
    expected: Callable[[float], float],
    counts: PlanCounts,
) -> list[Seed]:
    """Seeds of one cell and month (section 5), with their flood boundaries.

    ``hibernation_hours`` and ``joint_total_hours`` are the stage-1 values over a
    window around the cell (NaN is NODATA); ``core`` is the cell, as
    ``(row0, row1, col0, col1)`` inside the window.  Pixels outside the window's
    cell never become seeds.
    """
    parameters = evaluator.parameters
    time = evaluator.time
    row0, row1, col0, col1 = core
    haven = np.zeros(hibernation_hours.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        haven = hibernation_hours <= parameters.survival_limit_hours
    inside_cell = np.zeros_like(haven)
    inside_cell[row0:row1, col0:col1] = True
    counts.haven_pixels += int((haven & inside_cell).sum())
    if not (haven & inside_cell).any():
        return []
    counts.cell_months_with_haven += 1

    haven_count, pixel_count = nearby_haven_counts(
        haven, parameters.haven_neighbourhood_m, pixel_size_m
    )
    fraction = haven_count / np.maximum(pixel_count, 1)
    eligible = haven & inside_cell & (fraction >= parameters.min_nearby_haven_fraction - 1e-12)
    counts.dropped_by_nearby_fraction += int((haven & inside_cell).sum() - eligible.sum())
    if not eligible.any():
        return []

    reach = grid.step_count(parameters.operational_hours)
    proxy = box_sums(joint_total_hours, reach)
    rows, columns = np.nonzero(eligible)
    order = np.lexsort((columns, rows, -proxy[rows, columns]))

    separation2 = (parameters.seed_separation_m / pixel_size_m) ** 2
    tolerance_us = parameters.seed_timing_tolerance_hours * MICROSECONDS_PER_HOUR
    accepted: list[PixelRecord] = []
    local: list[tuple[int, int]] = []
    probes: list[int] = []
    for position in order:
        if len(accepted) == parameters.max_seeds_per_cell_month:
            break
        row, column = int(rows[position]), int(columns[position])
        near_indices: list[int] = []
        if accepted:
            distances = np.array([(row - r) ** 2 + (column - c) ** 2 for r, c in local])
            near_indices = [int(i) for i in np.flatnonzero(distances < separation2 - 1e-9)]
            if near_indices:
                nearest = int(min(near_indices, key=lambda i: (distances[i], i)))
                if probes[nearest] >= parameters.timing_probes_per_seed:
                    continue
                probes[nearest] += 1
        global_row, global_column = window_origin[0] + row, window_origin[1] + column
        counts.timelines_evaluated += 1
        result = evaluator.evaluate(global_row, global_column, month)
        if isinstance(result, str):
            counts.invalid_no_horizon += result == "no_horizon"
            counts.invalid_no_record += result == "no_record"
            counts.invalid_outage_unclosed += result == "outage_unclosed"
            continue
        stage_one = float(hibernation_hours[row, column])
        if not math.isclose(expected(result.score_hours), stage_one, rel_tol=0.0, abs_tol=1e-6):
            raise PlanConsistencyError(
                f"pixel ({global_row}, {global_column}) month {month}: stage 1 holds "
                f"{stage_one} h but the recomputed hibernation scores {result.score_hours} h"
            )
        if near_indices and any(
            abs(result.outage_start_us - accepted[i].outage_start_us) <= tolerance_us
            and abs(result.outage_end_us - accepted[i].outage_end_us) <= tolerance_us
            for i in near_indices
        ):
            continue
        accepted.append(result)
        local.append((row, column))
        probes.append(0)

    counts.seeds_selected += len(accepted)
    seeds: list[Seed] = []
    for record, (row, column) in zip(accepted, local):
        boundaries = find_boundaries(evaluator.signals, time, grid, record, parameters)
        if boundaries is None:
            counts.invalid_no_lit_boundary += 1
            continue
        seeds.append(Seed(
            row=record.row, column=record.column,
            hibernation_hours=record.score_hours,
            outage_start_us=record.outage_start_us, outage_end_us=record.outage_end_us,
            boundary_before_step=boundaries[0], boundary_after_step=boundaries[1],
            nearby_haven_count=int(haven_count[row, column]),
            nearby_pixel_count=int(pixel_count[row, column]),
            nearby_haven_fraction=float(fraction[row, column]),
            earth_visible_before_hours=record.earth_visible_before_hours,
            earth_visible_after_hours=record.earth_visible_after_hours,
        ))
    counts.seeds_valid += len(seeds)
    if seeds:
        counts.cell_months_kept += 1
    return seeds


# ---------------------------------------------------------------------------
# Lighting ranges (section 8)
# ---------------------------------------------------------------------------


def lighting_ranges(seeds: list[Seed], flood_steps: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Before and after flood-grid ranges, inclusive: the before range runs from
    ``flood_steps`` before the earliest before-boundary to the latest one; the after
    range from the earliest after-boundary to ``flood_steps`` past the latest."""
    before = [seed.boundary_before_step for seed in seeds]
    after = [seed.boundary_after_step for seed in seeds]
    return (min(before) - flood_steps, max(before)), (min(after), max(after) + flood_steps)


def lighting_chunks(seeds: list[Seed], grid: FloodGrid, flood_steps: int) -> list[int]:
    """Indices of the 24 h chunks the two ranges touch, rounded out to the chunk grid."""
    chunks: set[int] = set()
    for low, high in lighting_ranges(seeds, flood_steps):
        chunks.update(range(low // grid.steps_per_chunk, high // grid.steps_per_chunk + 1))
    return sorted(chunks)


# ---------------------------------------------------------------------------
# Stage-1 rasters and the pass-1 driver
# ---------------------------------------------------------------------------


class Stage1Rasters:
    """The stage-1 ``hibernation`` and ``joint_total`` products, read as hours.

    Float products hold hours with NaN as NODATA.  Byte products written with a
    two-hour count transform are decoded with ``hours_per_count`` (2.0) and
    ``nodata`` (255): hours = count * hours_per_count, the NODATA value is NaN, and
    a count of 255 is saturated.
    """

    def __init__(self, hibernation_path, joint_total_path, *, hours_per_count: float | None = None,
                 nodata: int | None = None) -> None:
        self.hibernation_path = Path(hibernation_path)
        self.joint_total_path = Path(joint_total_path)
        self.hours_per_count = hours_per_count
        self.nodata = nodata
        if (hours_per_count is None) != (nodata is None):
            raise ValueError("hours_per_count and nodata are given together or not at all")
        import rasterio

        with rasterio.open(self.hibernation_path) as one, rasterio.open(self.joint_total_path) as two:
            if (one.count, one.height, one.width) != (two.count, two.height, two.width):
                raise ValueError("the stage-1 products differ in size or band count")
            self.band_count, self.height, self.width = one.count, one.height, one.width

    def _decode(self, values: npt.NDArray) -> npt.NDArray[np.float64]:
        if self.hours_per_count is None:
            return values.astype(np.float64)
        hours = values.astype(np.float64) * self.hours_per_count
        hours[values == self.nodata] = np.nan
        return hours

    def read_window(self, row0: int, row1: int, col0: int, col1: int):
        """``(hibernation, joint_total)`` hours, each ``(months, rows, columns)``."""
        import rasterio
        from rasterio.windows import Window

        window = Window(col0, row0, col1 - col0, row1 - row0)
        with rasterio.open(self.hibernation_path) as one, rasterio.open(self.joint_total_path) as two:
            return self._decode(one.read(window=window)), self._decode(two.read(window=window))

    def expected(self, hours: float) -> float:
        """The value stage 1 would store for an exact ``hours``, as decoded hours."""
        if self.hours_per_count is None:
            return float(np.float32(hours))
        count = np.clip(np.ceil(np.float32(hours) / np.float32(self.hours_per_count)), 0, 255)
        return float(count) * self.hours_per_count


@dataclass
class PassOneReport:
    """What pass 1 found and what the floods will need (specification, section 9)."""

    flood_step_minutes: int
    flood_steps: int
    cells: int
    months: int
    counts: PlanCounts
    parameters: WindowParameters

    def as_dict(self) -> dict:
        return {
            "flood_step_minutes": self.flood_step_minutes,
            "flood_steps": self.flood_steps,
            "cells": self.cells,
            "months": self.months,
            "counts": asdict(self.counts),
            "parameters": asdict(self.parameters),
        }


def _seed_record(seed: Seed) -> dict:
    record = asdict(seed)
    record["outage_start_utc"] = str(np.datetime64(seed.outage_start_us, "us")) + "Z"
    record["outage_end_utc"] = str(np.datetime64(seed.outage_end_us, "us")) + "Z"
    return record


def iter_cell_months(
    *,
    dem,
    horizon_store: HorizonTileStore,
    stage1: Stage1Rasters,
    times: npt.ArrayLike,
    sun_vectors_at: VectorSource,
    earth_vectors_at: VectorSource,
    parameters: WindowParameters = WindowParameters(),
    observer_height_m: float = 0.0,
    earth_elevation_threshold_deg: float = 2.0,
    sunlight_fraction_threshold: float = 0.2,
    merge_gap_hours: float = 24.0,
    cells: list[tuple[int, int]] | None = None,
    report: list | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[dict]:
    """Yield one record per cell-month that has at least one valid seed.

    ``cells`` lists ``(cell_row, cell_col)`` indices; the default is every cell of
    the raster.  The final :class:`PassOneReport` is appended to ``report``, and
    ``progress(cells_done, cells_total)`` is called after every cell.
    """
    if (stage1.height, stage1.width) != (dem.height, dem.width):
        raise ValueError("the stage-1 products and the DEM differ in size")
    pixel_size = abs(float(dem.geo_transform[1]))
    if not math.isclose(pixel_size, abs(float(dem.geo_transform[5])), rel_tol=1e-6):
        raise ValueError("pass 1 needs square pixels")
    time = TimeContext.build(times, sun_vectors_at, earth_vectors_at)
    if time.month_count != stage1.band_count:
        raise ValueError(
            f"the times span {time.month_count} months but the products have {stage1.band_count} bands"
        )
    grid = FloodGrid(flood_step_minutes(pixel_size, parameters.rover_speed_m_s))
    flood_steps = grid.step_count(parameters.operational_hours)
    halo = halo_pixels(flood_steps)
    signals = PixelSignals(dem, horizon_store, observer_height_m)
    evaluator = CandidateEvaluator(
        signals, time, parameters,
        earth_elevation_threshold_deg=earth_elevation_threshold_deg,
        sunlight_fraction_threshold=sunlight_fraction_threshold,
        merge_gap_hours=merge_gap_hours,
    )
    cell_size = parameters.cell_patches * PATCH_SIZE
    cell_rows = -(-dem.height // cell_size)
    cell_columns = -(-dem.width // cell_size)
    if cells is None:
        cells = [(r, c) for r in range(cell_rows) for c in range(cell_columns)]
    counts = PlanCounts()
    shared_tasks: set[tuple[int, int, int]] = set()
    for cells_done, (cell_row, cell_column) in enumerate(cells, start=1):
        window, core = cell_window(
            cell_row, cell_column, cell_size=cell_size, halo=halo,
            height=dem.height, width=dem.width,
        )
        hibernation, joint_total = stage1.read_window(*window)
        with np.errstate(invalid="ignore"):
            has_haven = (
                hibernation[:, core[0]:core[1], core[2]:core[3]] <= parameters.survival_limit_hours
            ).any(axis=(1, 2))
        window_patches = [
            (pr, pc)
            for pr in range(window[0] // PATCH_SIZE, (window[1] - 1) // PATCH_SIZE + 1)
            for pc in range(window[2] // PATCH_SIZE, (window[3] - 1) // PATCH_SIZE + 1)
        ]
        for month in range(time.month_count):
            counts.cell_months_examined += 1
            if not has_haven[month]:
                continue
            seeds = select_seeds(
                hibernation_hours=hibernation[month], joint_total_hours=joint_total[month],
                window_origin=(window[0], window[2]), core=core, month=month,
                evaluator=evaluator, grid=grid, pixel_size_m=pixel_size,
                expected=stage1.expected, counts=counts,
            )
            if not seeds:
                continue
            chunks = lighting_chunks(seeds, grid, flood_steps)
            counts.tasks_unshared += len(window_patches) * len(chunks)
            shared_tasks.update((pr, pc, chunk) for pr, pc in window_patches for chunk in chunks)
            ranges = lighting_ranges(seeds, flood_steps)
            yield {
                "cell_row": cell_row, "cell_col": cell_column, "month": month,
                "before_range_steps": list(ranges[0]), "after_range_steps": list(ranges[1]),
                "chunks": chunks,
                "seeds": [_seed_record(seed) for seed in seeds],
            }
        if progress is not None:
            progress(cells_done, len(cells))
    counts.tasks_shared = len(shared_tasks)
    if report is not None:
        report.append(PassOneReport(grid.step_minutes, flood_steps, len(cells),
                                    time.month_count, counts, parameters))


def halo_pixels(flood_steps: int) -> int:
    """The halo around a cell, in pixels: whole patches covering the flood reach."""
    return int(math.ceil(flood_steps / PATCH_SIZE)) * PATCH_SIZE


def cell_window(
    cell_row: int, cell_column: int, *, cell_size: int, halo: int, height: int, width: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """The raster window around a cell, clipped to the raster, and the cell inside it.

    Both are ``(row0, row1, col0, col1)``; the cell is relative to the window.
    """
    row0, col0 = cell_row * cell_size, cell_column * cell_size
    row1, col1 = min(height, row0 + cell_size), min(width, col0 + cell_size)
    window = (max(0, row0 - halo), min(height, row1 + halo),
              max(0, col0 - halo), min(width, col1 + halo))
    core = (row0 - window[0], row1 - window[0], col0 - window[2], col1 - window[2])
    return window, core


def spice_vector_source(body: str) -> VectorSource:
    """Sun or Earth Moon-ME positions from SPICE for any UTC times, as pass 1 needs them.

    The conversion is the stage-1 default (``utc2et`` per timestamp).  Calling it
    loads the default SPICE kernels.
    """

    def vectors_at(stamps: npt.ArrayLike) -> npt.NDArray[np.float64]:
        from .product_vectors import generate_moon_me_vectors

        moments = [
            value.replace(tzinfo=timezone.utc)
            for value in np.asarray(stamps).astype("datetime64[us]").astype(object)
        ]
        return generate_moon_me_vectors(body, moments).vectors_m

    return vectors_at


def run_pass_one(*, plan_path: str | Path | None = None, **options) -> PassOneReport:
    """Run :func:`iter_cell_months` to the end and return the report.

    With ``plan_path``, every record is written as one line of JSON for pass 2.
    """
    report: list[PassOneReport] = []
    handle = None if plan_path is None else Path(plan_path).open("w", encoding="utf-8")
    try:
        for record in iter_cell_months(report=report, **options):
            if handle is not None:
                handle.write(json.dumps(record) + "\n")
    finally:
        if handle is not None:
            handle.close()
    return report[0]
