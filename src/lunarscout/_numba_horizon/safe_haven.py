"""Private per-pixel safe-haven product reducer: one forward pass, monthly bands.

The per-pixel state machine is specified in ``docs/safe-haven-state-machine.md``;
the functions below implement it step for step.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from itertools import islice
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt


def _utc_timestamp(value: datetime | str) -> datetime:
    from datetime import timezone

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def build_month_bands(
    times_utc: Sequence[datetime | str],
) -> tuple[tuple[datetime, datetime], ...]:
    """Return calendar-month ``(start, stop)`` intervals covering *times_utc*.

    Each band spans one calendar month in UTC.  Boundaries are inclusive at the
    start and exclusive at the stop (matching the half-open evaluation-interval
    convention).
    """
    timestamps = tuple(_utc_timestamp(value) for value in times_utc)
    if not timestamps:
        return ()

    bands: list[tuple[datetime, datetime]] = []
    month_keys: set[tuple[int, int]] = set()
    for value in timestamps:
        key = (value.year, value.month)
        if key not in month_keys:
            month_keys.add(key)
    from datetime import timedelta

    for year, month in sorted(month_keys):
        start = datetime(year, month, 1, tzinfo=timestamps[0].tzinfo)
        if month == 12:
            stop = datetime(year + 1, 1, 1, tzinfo=start.tzinfo)
        else:
            stop = datetime(year, month + 1, 1, tzinfo=start.tzinfo)
        bands.append((start, stop))
    return tuple(bands)


def _month_index_for_time(
    time: datetime,
    month_bands: tuple[tuple[datetime, datetime], ...],
) -> int | None:
    for index, (start, stop) in enumerate(month_bands):
        if start <= time < stop:
            return index
    return None


def _month_indices_map(
    times_utc: Sequence[datetime | str],
    month_bands: tuple[tuple[datetime, datetime], ...],
) -> npt.NDArray[np.int32]:
    timestamps = tuple(_utc_timestamp(value) for value in times_utc)
    mapping = np.full(len(timestamps), -1, dtype=np.int32)
    for index, value in enumerate(timestamps):
        month_index = _month_index_for_time(value, month_bands)
        if month_index is not None:
            mapping[index] = month_index
    return mapping


# ---------------------------------------------------------------------------
# Per-pixel state machine (see docs/safe-haven-state-machine.md)
# ---------------------------------------------------------------------------

#: Sentinel for "no value yet" in the per-band minimum; never an output value.
UNSET = 2147483647

# Rows of the per-pixel state table ``S[row, pixel]`` (all int32).
RUN_LEN = 0        # samples in the open shadow run, through its last shadow sample
GAP = 1            # sunlit samples since the run's last shadow sample (0 in shadow)
IN_HIB = 2         # 1 while a hibernation (no-communications interval) is active
HIB_BAND = 3       # band of the active hibernation's first sample, -1 if uncredited
HIB_SCORE = 4      # longest closed run intersecting the active hibernation so far
HEAD = 5           # 1 if the open run's chain has a head hibernation
HEAD_BAND = 6      # band of the head hibernation, -1 if uncredited
HEAD_PRIOR = 7     # head's score from runs that closed before the open run began
HEAD_ACTIVE = 8    # 1 while the head hibernation is still the active one
MID_LO = 9         # lowest band of ended non-head chain members (UNSET if none)
MID_HI = 10        # highest band of ended non-head chain members (-1 if none)
TAIL_ACTIVE = 11   # 1 while the active hibernation is a non-head chain member
PEND_LO = 12       # lowest band of hibernations wholly inside the pending gap
PEND_HI = 13       # highest band of those (-1 if none)
PEND_CUR = 14      # 1 while the active hibernation began inside the pending gap
JOINT_RUN = 15     # samples in the current joint-operating run
JOINT_BAND = 16    # band of the current joint-operating run's first sample
DELAY_MASK = 17    # outage flags of the last 2W+1 samples, newest in bit 0
STATE_ROWS = 18

#: Longest half-window of the degenerate-outage rule, in samples (2W+1 flag bits
#: must fit in the int32 mask).
MAX_DEGENERATE_WINDOW = 15
#: Rows of the float ring table: fractions of the last W+1 samples, then flat-horizon
#: elevations of the last W+2 samples, each indexed by sample number modulo its length.
FRACTION_ROW = 0
ELEVATION_ROW = MAX_DEGENERATE_WINDOW + 1
RING_ROWS = ELEVATION_ROW + MAX_DEGENERATE_WINDOW + 2


def degenerate_window_samples(
    window_hours: float | None, time_step_hours: float
) -> int:
    """Whole samples in the degenerate-outage half-window.

    ``None`` is 24 hours, or 15 samples when the step is shorter than 1.6 hours.
    An explicit window must be zero (the rule is off) or one to 15 steps; anything
    else raises ``ValueError``.
    """
    if window_hours is None:
        return min(
            MAX_DEGENERATE_WINDOW, max(1, int(np.floor(24.0 / time_step_hours + 0.5)))
        )
    samples = int(np.floor(window_hours / time_step_hours + 0.5))
    if (window_hours > 0 and samples < 1) or samples > MAX_DEGENERATE_WINDOW:
        raise ValueError(
            "the degenerate-outage window must be zero, or between one and "
            f"{MAX_DEGENERATE_WINDOW} time steps"
        )
    return samples


def new_ring(pixel_count: int) -> npt.NDArray[np.float32]:
    """The float ring table for ``pixel_count`` pixels."""
    return np.zeros((RING_ROWS, pixel_count), dtype=np.float32)


def new_state(pixel_count: int) -> npt.NDArray[np.int32]:
    """Return the initial state table for ``pixel_count`` pixels."""
    state = np.zeros((STATE_ROWS, pixel_count), dtype=np.int32)
    state[HIB_BAND] = -1
    state[HEAD_BAND] = -1
    state[MID_LO] = UNSET
    state[MID_HI] = -1
    state[PEND_LO] = UNSET
    state[PEND_HI] = -1
    state[JOINT_BAND] = -1
    return state


def merge_gap_limit(merge_gap_hours: float, time_step_hours: float) -> int:
    """Smallest sunlit-gap length (in samples) that is *not* bridged.

    Two shadow runs are joined when the sunlit gap between them lasts strictly
    less than ``merge_gap_hours``.  A gap of ``g`` samples lasts ``g * step``.
    """
    ratio = merge_gap_hours / time_step_hours
    return max(0, int(np.ceil(ratio - 1e-9)))


def make_machine(jit):
    """Build the per-pixel machine functions with the decorator ``jit``.

    ``jit`` is ``numba.njit`` for the CPU, ``numba.cuda.jit(device=True)`` for
    the GPU, or the identity for plain Python.  One definition serves all three.
    """

    @jit
    def _commit(best, i, lo, hi, score):
        """Record a hibernation score for every band in ``lo..hi`` (minimum wins)."""
        for band in range(lo, hi + 1):
            if score < best[band, i]:
                best[band, i] = score


    @jit
    def _close_run(S, best, i):
        """The open run is over: give its final length to every chain member."""
        length = S[RUN_LEN, i]
        if S[HEAD, i] == 1:
            score = max(S[HEAD_PRIOR, i], length)
            if S[HEAD_ACTIVE, i] == 1:
                S[HIB_SCORE, i] = score          # head continues as the active hibernation
            elif S[HEAD_BAND, i] >= 0:
                _commit(best, i, S[HEAD_BAND, i], S[HEAD_BAND, i], score)
        if S[MID_HI, i] >= 0:
            _commit(best, i, S[MID_LO, i], S[MID_HI, i], length)
        if S[TAIL_ACTIVE, i] == 1:
            S[HIB_SCORE, i] = length             # prior is 0, so the score is the length
        if S[PEND_HI, i] >= 0:
            _commit(best, i, S[PEND_LO, i], S[PEND_HI, i], 0)  # touched no run
        S[RUN_LEN, i] = 0
        S[GAP, i] = 0
        S[HEAD, i] = 0
        S[HEAD_BAND, i] = -1
        S[HEAD_PRIOR, i] = 0
        S[HEAD_ACTIVE, i] = 0
        S[MID_LO, i] = UNSET
        S[MID_HI, i] = -1
        S[TAIL_ACTIVE, i] = 0
        S[PEND_LO, i] = UNSET
        S[PEND_HI, i] = -1
        S[PEND_CUR, i] = 0


    @jit
    def _widen(S, i, lo_row, hi_row, band):
        if band >= 0:
            if band < S[lo_row, i]:
                S[lo_row, i] = band
            if band > S[hi_row, i]:
                S[hi_row, i] = band


    @jit
    def advance_pixel(
        S, i, best, have, joint_total, joint_best, fraction, margin, time_index,
        band, sunlight_threshold, earth_threshold, operating_threshold, gap_limit,
    ):
        """Advance pixel ``i`` by one time sample.  Steps are in specification order."""
        outage = margin < earth_threshold
        shadow = fraction < sunlight_threshold

        # 1. Hibernation starts or ends.
        if outage:
            if S[IN_HIB, i] == 0:
                S[IN_HIB, i] = 1
                # A hibernation already under way at the first sample has an
                # unknown start and is not credited to any band.
                credited = band if (time_index > 0 and band >= 0) else -1
                S[HIB_BAND, i] = credited
                S[HIB_SCORE, i] = 0
                if credited >= 0:
                    have[credited, i] = 1
        elif S[IN_HIB, i] == 1:
            S[IN_HIB, i] = 0
            credited = S[HIB_BAND, i]
            if S[HEAD_ACTIVE, i] == 1:
                S[HEAD_ACTIVE, i] = 0            # stays the head, now an ended member
            elif S[TAIL_ACTIVE, i] == 1:
                S[TAIL_ACTIVE, i] = 0
                _widen(S, i, MID_LO, MID_HI, credited)
            elif S[PEND_CUR, i] == 1:
                S[PEND_CUR, i] = 0
                _widen(S, i, PEND_LO, PEND_HI, credited)
            elif credited >= 0:
                _commit(best, i, credited, credited, S[HIB_SCORE, i])

        # 2. Shadow runs, with sunlit gaps shorter than the merge limit bridged.
        if shadow:
            if S[RUN_LEN, i] == 0:
                S[RUN_LEN, i] = 1                # a new run opens
                if S[IN_HIB, i] == 1:            # the active hibernation is its head
                    S[HEAD, i] = 1
                    S[HEAD_BAND, i] = S[HIB_BAND, i]
                    S[HEAD_PRIOR, i] = S[HIB_SCORE, i]
                    S[HEAD_ACTIVE, i] = 1
            else:
                if S[GAP, i] > 0:                # the gap is bridged
                    S[RUN_LEN, i] += S[GAP, i] + 1
                    S[GAP, i] = 0
                    if S[PEND_HI, i] >= 0:       # hibernations wholly inside the gap
                        if S[PEND_LO, i] < S[MID_LO, i]:
                            S[MID_LO, i] = S[PEND_LO, i]
                        if S[PEND_HI, i] > S[MID_HI, i]:
                            S[MID_HI, i] = S[PEND_HI, i]
                        S[PEND_LO, i] = UNSET
                        S[PEND_HI, i] = -1
                    S[PEND_CUR, i] = 0           # a hibernation that began in the gap
                                                 # and is still active joins as the tail below
                else:
                    S[RUN_LEN, i] += 1
                if (
                    S[IN_HIB, i] == 1
                    and S[HEAD_ACTIVE, i] == 0
                    and S[TAIL_ACTIVE, i] == 0
                ):
                    S[TAIL_ACTIVE, i] = 1        # began at this shadow sample
        elif S[RUN_LEN, i] > 0:
            S[GAP, i] += 1
            if (
                S[IN_HIB, i] == 1
                and S[HEAD_ACTIVE, i] == 0
                and S[TAIL_ACTIVE, i] == 0
                and S[PEND_CUR, i] == 0
            ):
                S[PEND_CUR, i] = 1               # began inside the gap: unconfirmed
            if S[GAP, i] >= gap_limit:
                _close_run(S, best, i)

        # 3. Joint operating: Earth available and the Sun fully visible.
        if (not outage) and fraction >= operating_threshold:
            if S[JOINT_RUN, i] == 0:
                S[JOINT_BAND, i] = band
            S[JOINT_RUN, i] += 1
            if band >= 0:
                joint_total[band, i] += 1
            start_band = S[JOINT_BAND, i]
            if start_band >= 0 and S[JOINT_RUN, i] > joint_best[start_band, i]:
                joint_best[start_band, i] = S[JOINT_RUN, i]
        else:
            S[JOINT_RUN, i] = 0


    @jit
    def finish_pixel(S, i, best):
        """No more samples: close the open run and the active hibernation."""
        if S[RUN_LEN, i] > 0:
            _close_run(S, best, i)
        if S[IN_HIB, i] == 1 and S[HIB_BAND, i] >= 0:
            _commit(best, i, S[HIB_BAND, i], S[HIB_BAND, i], S[HIB_SCORE, i])
        S[IN_HIB, i] = 0

    @jit
    def advance_delayed(
        S, R, i, best, have, joint_total, joint_best, fraction, margin, elevation,
        time_index, band_of_time, window, sunlight_threshold, earth_threshold,
        operating_threshold, gap_limit,
    ):
        """Take sample ``time_index`` and advance the machine ``window`` samples behind.

        The machine sees sample ``time_index - window`` with its final outage flag:
        the real flag (margin below the threshold), or a degenerate outage at an
        Earth-elevation minimum with no real outage within ``window`` samples
        either side (docs/safe-haven-state-machine.md, section 2.1).
        ``window == 0`` is the plain machine.
        """
        outage = margin < earth_threshold
        R[FRACTION_ROW + time_index % (window + 1), i] = fraction
        R[ELEVATION_ROW + time_index % (window + 2), i] = elevation
        full = (1 << (2 * window + 1)) - 1
        mask = ((S[DELAY_MASK, i] << 1) | (1 if outage else 0)) & full
        S[DELAY_MASK, i] = mask
        sample = time_index - window
        if sample < 0:
            return
        flagged = ((mask >> window) & 1) == 1
        if window >= 1 and sample >= window and mask == 0:
            before = R[ELEVATION_ROW + (sample - 1) % (window + 2), i]
            here = R[ELEVATION_ROW + sample % (window + 2), i]
            after = R[ELEVATION_ROW + (sample + 1) % (window + 2), i]
            if here < before and here <= after:
                flagged = True
        advance_pixel(
            S, i, best, have, joint_total, joint_best,
            R[FRACTION_ROW + sample % (window + 1), i],
            earth_threshold - 1.0 if flagged else earth_threshold + 1.0,
            sample, band_of_time[sample], sunlight_threshold, earth_threshold,
            operating_threshold, gap_limit,
        )

    @jit
    def flush_delayed(
        S, R, i, best, have, joint_total, joint_best, time_count, band_of_time,
        window, sunlight_threshold, earth_threshold, operating_threshold, gap_limit,
    ):
        """Feed the last ``window`` samples, which have no complete window, with
        their real outage flags, then the machine has seen every sample."""
        mask = S[DELAY_MASK, i]
        first = time_count - window
        if first < 0:
            first = 0
        for sample in range(first, time_count):
            flagged = ((mask >> (time_count - 1 - sample)) & 1) == 1
            advance_pixel(
                S, i, best, have, joint_total, joint_best,
                R[FRACTION_ROW + sample % (window + 1), i],
                earth_threshold - 1.0 if flagged else earth_threshold + 1.0,
                sample, band_of_time[sample], sunlight_threshold, earth_threshold,
                operating_threshold, gap_limit,
            )

    return SimpleNamespace(
        advance_pixel=advance_pixel,
        advance_delayed=advance_delayed,
        flush_delayed=flush_delayed,
        finish_pixel=finish_pixel,
    )


def _tile_loops(machine, loop):
    """Build the per-batch loops over pixels from a machine (``loop`` is range/prange).

    Each pixel advances through every time step of a batch before the next
    pixel starts, so one parallel call does many steps of work per thread.
    """
    advance_delayed = machine.advance_delayed
    flush_delayed = machine.flush_delayed
    finish_pixel = machine.finish_pixel

    def advance_batch(
        S, R, best, have, joint_total, joint_best, fractions, margins, elevations,
        first_time_index, band_of_time, window, sunlight_threshold, earth_threshold,
        operating_threshold, gap_limit, valid,
    ):
        steps = fractions.shape[0]
        for i in loop(fractions.shape[1]):
            if valid[i]:
                for step_index in range(steps):
                    advance_delayed(
                        S, R, i, best, have, joint_total, joint_best,
                        fractions[step_index, i], margins[step_index, i],
                        elevations[step_index, i], first_time_index + step_index,
                        band_of_time, window, sunlight_threshold, earth_threshold,
                        operating_threshold, gap_limit,
                    )

    def finish_tile(
        S, R, best, have, joint_total, joint_best, time_count, band_of_time, window,
        sunlight_threshold, earth_threshold, operating_threshold, gap_limit, valid,
    ):
        for i in loop(S.shape[1]):
            if valid[i]:
                flush_delayed(
                    S, R, i, best, have, joint_total, joint_best, time_count,
                    band_of_time, window, sunlight_threshold, earth_threshold,
                    operating_threshold, gap_limit,
                )
                finish_pixel(S, i, best)

    return advance_batch, finish_tile


_COMPILED: dict[str, object] = {}


def _compiled():
    """Compile the machine once with Numba (nogil, parallel over pixels)."""
    if not _COMPILED:
        from numba import njit, prange

        machine = make_machine(njit(cache=False, nogil=True))
        advance_batch, finish_tile = _tile_loops(machine, prange)
        _COMPILED["advance"] = njit(cache=False, nogil=True, parallel=True)(advance_batch)
        _COMPILED["finish"] = njit(cache=False, nogil=True, parallel=True)(finish_tile)
    return _COMPILED["advance"], _COMPILED["finish"]


def _python_tiles():
    """The same machine run as plain Python, for reading and debugging."""
    return _tile_loops(make_machine(lambda function: function), range)


#: Tolerance on the operating-Sun comparison: ``fraction >= threshold - fuzz``.
OPERATING_FUZZ = 1e-6


#: Samples per call of the batch kernel when per-sample tiles are re-batched.
DEFAULT_BATCH_SIZE = 32


def _checked_parameters(
    time_count, month_bands, month_index_of, sunlight_threshold,
    earth_threshold_deg, operating_sunlight_fraction, merge_gap_hours,
    time_step_hours, degenerate_window_samples=0,
):
    if (
        int(degenerate_window_samples) != degenerate_window_samples
        or not 0 <= degenerate_window_samples <= MAX_DEGENERATE_WINDOW
    ):
        raise ValueError(
            f"degenerate_window_samples must be an integer from 0 to {MAX_DEGENERATE_WINDOW}"
        )
    if time_count < 1:
        raise ValueError("time_count must be positive")
    if not 0.0 <= sunlight_threshold <= 1.0:
        raise ValueError("sunlight_threshold must be between zero and one")
    if not np.isfinite(earth_threshold_deg):
        raise ValueError("earth_threshold_deg must be finite")
    if not 0.0 <= operating_sunlight_fraction <= 1.0:
        raise ValueError("operating_sunlight_fraction must be between zero and one")
    if not np.isfinite(merge_gap_hours) or merge_gap_hours < 0.0:
        raise ValueError("merge_gap_hours must be finite and nonnegative")
    if not np.isfinite(time_step_hours) or time_step_hours <= 0.0:
        raise ValueError("time_step_hours must be positive and finite")
    band_count = len(month_bands)
    if month_index_of.shape != (time_count,):
        raise ValueError("month_index_of must have shape (time_count,)")
    if np.any(month_index_of < -1) or np.any(month_index_of >= band_count):
        raise ValueError("month_index_of contains an invalid band index")
    return band_count


def reduce_safe_haven_products_batches(
    sunlight_fraction_batches: Iterable[npt.ArrayLike],
    earth_elevation_batches: Iterable[npt.ArrayLike],
    time_count: int,
    month_bands: tuple[tuple[datetime, datetime], ...],
    *,
    month_index_of: npt.NDArray[np.int32],
    sunlight_threshold: float,
    earth_threshold_deg: float,
    operating_sunlight_fraction: float,
    merge_gap_hours: float,
    time_step_hours: float,
    invalid_pixels: npt.ArrayLike | None = None,
    use_numba: bool = True,
    earth_flat_elevation_batches: Iterable[npt.ArrayLike] | None = None,
    degenerate_window_samples: int = 0,
) -> tuple[
    tuple[npt.NDArray[np.float32], ...],
    tuple[npt.NDArray[np.float32], ...],
    tuple[npt.NDArray[np.float32], ...],
]:
    """Reduce one patch's Sun and Earth signals, given in batches of time steps.

    Each element of the two iterables is a float ``(steps, y, x)`` array; the
    Sun batches and Earth batches must have the same steps, in time order, and
    together cover ``time_count`` samples.  Returns ``(hibernation,
    joint_total, joint_longest)``: each a tuple with one float32 ``(y, x)``
    array in hours per band.  Hibernation is NaN where no hibernation starts
    in the band; the joint layers are never NaN here.  See
    ``docs/safe-haven-state-machine.md`` for the definitions.

    ``invalid_pixels`` is an optional boolean ``(y, x)`` mask of pixels on
    invalid DEM nodes.  Their lighting signals are meaningless (NaN), so they
    are skipped and their outputs stay at the no-event values; the caller
    writes NODATA there.  A non-finite signal at any other pixel is an error.

    ``degenerate_window_samples`` (``W``, 0 to 15) turns on the degenerate-outage
    rule of ``docs/safe-haven-state-machine.md`` section 2.1: an Earth-elevation
    minimum with no outage within ``W`` samples either side becomes a one-sample
    outage.  It needs ``earth_flat_elevation_batches``, Earth's elevation above a
    flat horizon, in the same batches as the other two signals.  With ``W = 0``
    that argument is ignored.
    """
    band_count = _checked_parameters(
        time_count, month_bands, month_index_of, sunlight_threshold,
        earth_threshold_deg, operating_sunlight_fraction, merge_gap_hours,
        time_step_hours, degenerate_window_samples,
    )
    window = int(degenerate_window_samples)
    if window > 0 and earth_flat_elevation_batches is None:
        raise ValueError("the degenerate-outage rule needs earth_flat_elevation_batches")
    gap_limit = merge_gap_limit(merge_gap_hours, time_step_hours)
    operating_threshold = np.float32(operating_sunlight_fraction - OPERATING_FUZZ)
    sun_threshold = np.float32(sunlight_threshold)
    earth_threshold = np.float32(earth_threshold_deg)
    month_index_of = np.ascontiguousarray(month_index_of, dtype=np.int32)
    if use_numba:
        advance, finish = _compiled()
    else:
        advance, finish = _python_tiles()

    fraction_iter = iter(sunlight_fraction_batches)
    earth_iter = iter(earth_elevation_batches)
    flat_iter = (
        iter(earth_flat_elevation_batches)
        if window > 0 and earth_flat_elevation_batches is not None else None
    )
    shape = None
    state = ring = best = have = joint_total = joint_best = valid = checked = None
    time_index = 0
    while time_index < time_count:
        try:
            fractions = np.asarray(next(fraction_iter), dtype=np.float32)
            margins = np.asarray(next(earth_iter), dtype=np.float32)
            flats = None if flat_iter is None else np.asarray(next(flat_iter), dtype=np.float32)
        except StopIteration as exc:
            raise ValueError(
                "fraction or elevation iterator ended before time_count"
            ) from exc
        if fractions.ndim != 3 or margins.ndim != 3:
            raise ValueError("signal batches must be 3D (steps, y, x) arrays")
        if fractions.shape != margins.shape or (flats is not None and flats.shape != margins.shape):
            raise ValueError("fraction and elevation tiles must have the same shape")
        steps = fractions.shape[0]
        if steps < 1 or time_index + steps > time_count:
            raise ValueError("signal batches do not match time_count")
        if shape is None:
            shape = fractions.shape[1:]
            pixel_count = shape[0] * shape[1]
            state = new_state(pixel_count)
            ring = new_ring(pixel_count)
            best = np.full((band_count, pixel_count), UNSET, dtype=np.int32)
            have = np.zeros((band_count, pixel_count), dtype=np.uint8)
            joint_total = np.zeros((band_count, pixel_count), dtype=np.int32)
            joint_best = np.zeros((band_count, pixel_count), dtype=np.int32)
            if invalid_pixels is None:
                checked = np.ones(shape, dtype=np.bool_)
            else:
                checked = ~np.asarray(invalid_pixels, dtype=np.bool_)
                if checked.shape != shape:
                    raise ValueError("invalid_pixels must have the tile shape")
            valid = np.ascontiguousarray(checked.reshape(-1))
        elif fractions.shape[1:] != shape:
            raise ValueError("all tiles must have the same shape")
        # Only pixels on valid DEM nodes must carry finite signals.
        if not np.all(np.isfinite(fractions) | ~checked):
            raise ValueError("sunlight fraction tiles must be finite 2D arrays")
        if not np.all(np.isfinite(margins) | ~checked):
            raise ValueError("earth elevation tiles must be finite 2D arrays")
        if flats is None:
            flats = np.zeros_like(margins)
        elif not np.all(np.isfinite(flats) | ~checked):
            raise ValueError("flat-horizon elevation tiles must be finite 2D arrays")
        advance(
            state, ring, best, have, joint_total, joint_best,
            np.ascontiguousarray(fractions.reshape(steps, -1)),
            np.ascontiguousarray(margins.reshape(steps, -1)),
            np.ascontiguousarray(flats.reshape(steps, -1)),
            time_index, month_index_of, window,
            sun_threshold, earth_threshold, operating_threshold, gap_limit, valid,
        )
        time_index += steps
    for iterator, message in (
        (fraction_iter, "fraction iterator has more entries than time_count"),
        (earth_iter, "elevation iterator has more entries than time_count"),
        (flat_iter, "flat-elevation iterator has more entries than time_count"),
    ):
        if iterator is None:
            continue
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError(message)

    assert state is not None and shape is not None
    finish(
        state, ring, best, have, joint_total, joint_best, time_count, month_index_of,
        window, sun_threshold, earth_threshold, operating_threshold, gap_limit, valid,
    )
    return assemble_outputs(
        best, have, joint_total, joint_best, shape, time_step_hours
    )


def _rebatched(tiles: Iterable[npt.ArrayLike], size: int):
    """Group per-sample ``(y, x)`` tiles into ``(steps, y, x)`` batches."""
    iterator = iter(tiles)
    while True:
        group = list(islice(iterator, size))
        if not group:
            return
        yield np.stack([np.asarray(tile, dtype=np.float32) for tile in group])


def reduce_safe_haven_products_patch_stream(
    sunlight_fraction_tiles: Iterable[npt.ArrayLike],
    earth_elevation_tiles: Iterable[npt.ArrayLike],
    time_count: int,
    month_bands: tuple[tuple[datetime, datetime], ...],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    earth_flat_elevation_tiles: Iterable[npt.ArrayLike] | None = None,
    **options,
):
    """As :func:`reduce_safe_haven_products_batches`, from per-sample tiles.

    Each iterable yields one float ``(y, x)`` tile per time step.  The tiles are
    grouped into batches of ``batch_size`` steps.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return reduce_safe_haven_products_batches(
        _rebatched(sunlight_fraction_tiles, batch_size),
        _rebatched(earth_elevation_tiles, batch_size),
        time_count, month_bands,
        earth_flat_elevation_batches=(
            None if earth_flat_elevation_tiles is None
            else _rebatched(earth_flat_elevation_tiles, batch_size)
        ),
        **options,
    )


def assemble_outputs(best, have, joint_total, joint_best, shape, time_step_hours):
    """Convert the reducer's integer counters to float32 hours per band."""
    step = np.float32(time_step_hours)
    hibernation: list[npt.NDArray[np.float32]] = []
    totals: list[npt.NDArray[np.float32]] = []
    longest: list[npt.NDArray[np.float32]] = []
    for band in range(best.shape[0]):
        started = have[band] == 1
        if np.any(started & (best[band] == UNSET)):
            raise ValueError("internal error: a hibernation was never scored")
        hours = np.where(started, best[band].astype(np.float32) * step, np.nan)
        hibernation.append(np.ascontiguousarray(hours.astype(np.float32).reshape(shape)))
        totals.append(
            np.ascontiguousarray((joint_total[band].astype(np.float32) * step).reshape(shape))
        )
        longest.append(
            np.ascontiguousarray((joint_best[band].astype(np.float32) * step).reshape(shape))
        )
    return tuple(hibernation), tuple(totals), tuple(longest)
