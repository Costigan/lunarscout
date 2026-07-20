"""Private per-pixel safe-haven streaming reducer with monthly bands."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

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


def reduce_safe_haven_patch_stream(
    sunlight_fraction_tiles: Iterable[npt.ArrayLike],
    earth_elevation_tiles: Iterable[npt.ArrayLike],
    time_count: int,
    month_bands: tuple[tuple[datetime, datetime], ...],
    *,
    month_index_of: npt.NDArray[np.int32],
    sunlight_threshold: float,
    earth_threshold_deg: float,
    time_step_hours: float,
) -> tuple[npt.NDArray[np.float32], ...]:
    """Measure per-pixel safe-haven durations with calendar-month bands.

    For each pixel and calendar month, the reducer computes the longest complete
    contiguous low-Sun interval that overlaps any Earth-outage interval detected
    from *that pixel's own terrain horizon* within the month.

    Earth outages are detected per-pixel online (streaming): a pixel enters an
    outage when Earth elevation relative to its local terrain horizon drops
    strictly below *earth_threshold_deg*, and exits when it rises back above.

    A qualifying low-Sun run may begin before an outage, end after it, and span
    multiple months.  The run's full duration is credited to every calendar month
    it touches while overlapping an active Earth outage.

    Pixels where Earth never goes below the threshold during a month, or where
    Earth stays below the threshold for the *entire* month, receive NODATA
    because the safe-haven question is ill-posed for those pixels during that
    month.

    Parameters
    ----------
    sunlight_fraction_tiles:
        Iterator yielding one ``float32 (y, x)`` fraction tile per timestep.
    earth_elevation_tiles:
        Iterator yielding one ``float32 (y, x)`` Earth horizon-relative elevation
        tile per timestep, in degrees.
    time_count:
        Number of timesteps (must equal the length of both iterators).
    month_bands:
        Calendar-month ``(start_utc, stop_utc)`` intervals.
    month_index_of:
        Optional ``int32[time_count]`` mapping each timestep index to its month
        band index, or ``-1`` for timesteps outside all bands.  If omitted it
        is computed from *month_bands*.
    sunlight_threshold:
        Unitless strict-below threshold for a low-Sun pixel.
    earth_threshold_deg:
        Strict-below threshold in degrees for an Earth outage.
    time_step_hours:
        Duration of one sample in decimal hours.
    """
    if time_count < 1:
        raise ValueError("time_count must be positive")
    if not 0.0 <= sunlight_threshold <= 1.0:
        raise ValueError("sunlight_threshold must be between zero and one")
    if not np.isfinite(earth_threshold_deg):
        raise ValueError("earth_threshold_deg must be finite")
    if not np.isfinite(time_step_hours) or time_step_hours <= 0.0:
        raise ValueError("time_step_hours must be positive and finite")
    band_count = len(month_bands)

    if month_index_of.shape != (time_count,):
        raise ValueError("month_index_of must have shape (time_count,)")

    fraction_iter = iter(sunlight_fraction_tiles)
    earth_iter = iter(earth_elevation_tiles)

    shape = None

    run_length = None
    run_touching = None
    best = None
    had_outage = None
    was_above = None

    for time_index in range(time_count):
        try:
            fraction_tile = np.asarray(next(fraction_iter), dtype=np.float32)
            earth_tile = np.asarray(next(earth_iter), dtype=np.float32)
        except StopIteration as exc:
            raise ValueError(
                "fraction or elevation iterator ended before time_count"
            ) from exc

        if fraction_tile.ndim != 2 or not np.all(np.isfinite(fraction_tile)):
            raise ValueError("sunlight fraction tiles must be finite 2D arrays")
        if earth_tile.ndim != 2 or not np.all(np.isfinite(earth_tile)):
            raise ValueError("earth elevation tiles must be finite 2D arrays")

        if shape is None:
            shape = fraction_tile.shape
            if earth_tile.shape != shape:
                raise ValueError(
                    "fraction and elevation tiles must have the same shape"
                )
            run_length = np.zeros(shape, dtype=np.int32)
            run_touching = np.zeros((band_count, *shape), dtype=np.bool_)
            best = np.zeros((band_count, *shape), dtype=np.int32)
            had_outage = np.zeros((band_count, *shape), dtype=np.bool_)
            was_above = np.zeros((band_count, *shape), dtype=np.bool_)
        elif fraction_tile.shape != shape or earth_tile.shape != shape:
            raise ValueError("all tiles must have the same shape")

        month_index = int(month_index_of[time_index]) if time_index < len(month_index_of) else -1

        sun_low = fraction_tile < np.float32(sunlight_threshold)
        earth_below = earth_tile < np.float32(earth_threshold_deg)

        # Update the low-Sun run independently of Earth outage state.  Once a
        # run touches an outage, every later sample in that same run must be
        # credited even after Earth rises above the threshold.
        run_length[:] = np.where(sun_low, run_length + 1, 0)

        # Track whether the monthly safe-haven question is well posed, and
        # mark a low-Sun run when it overlaps an outage in the current month.
        if month_index >= 0:
            band = month_index
            was_above[band] |= ~earth_below
            had_outage[band] |= earth_below
            run_touching[band] |= earth_below & sun_low

        # Extend every marked run, including its portion after the outage and
        # across calendar-month boundaries.  Clear the marker only when the
        # low-Sun run ends.
        for band in range(band_count):
            active = run_touching[band] & sun_low
            best[band] = np.maximum(
                best[band],
                np.where(active, run_length, 0).astype(np.int32),
            )
            run_touching[band] &= sun_low

    # --- right-censor active runs ---
    active_run = run_length > 0
    if np.any(active_run):
        for band in range(band_count):
            mask = active_run & run_touching[band]
            if np.any(mask):
                best[band] = np.maximum(
                    best[band],
                    np.where(mask, run_length, 0).astype(np.int32),
                )

    # --- verify iterators are exhausted ---
    try:
        next(fraction_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("fraction iterator has more entries than time_count")
    try:
        next(earth_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("elevation iterator has more entries than time_count")

    # --- assemble output bands ---
    assert best is not None and had_outage is not None and was_above is not None
    results: list[npt.NDArray[np.float32]] = []
    for band in range(band_count):
        durations = best[band].astype(np.float32) * np.float32(time_step_hours)
        # NODATA for pixels where Earth was never below threshold (no outage
        # question to answer) or always below (permanent Earth shadow).
        nodata_mask = ~had_outage[band] | (had_outage[band] & ~was_above[band])
        durations[nodata_mask] = np.nan
        results.append(np.ascontiguousarray(durations))
    return tuple(results)
