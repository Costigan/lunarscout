"""Private safe-haven interval and bounded patch-reduction semantics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class EarthOutage:
    """Half-open time-index interval with its first minimum-Earth sample."""

    start: int
    stop: int
    minimum_index: int


def find_earth_outages(
    earth_elevation_deg: npt.ArrayLike, *, threshold_deg: float
) -> tuple[EarthOutage, ...]:
    """Return maximal half-open regions where Earth elevation is below threshold."""
    elevations = np.asarray(earth_elevation_deg, dtype=np.float64)
    if elevations.ndim != 1 or elevations.size == 0 or not np.all(np.isfinite(elevations)):
        raise ValueError("Earth elevations must be a non-empty finite 1D array")
    if not np.isfinite(threshold_deg):
        raise ValueError("Earth threshold must be finite")
    below = elevations < float(threshold_deg)
    changes = np.diff(np.pad(below.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple(
        EarthOutage(
            int(start),
            int(stop),
            int(start + np.argmin(elevations[start:stop])),
        )
        for start, stop in zip(starts, stops, strict=True)
    )


def reduce_safe_haven_patch(
    sunlight_fraction: npt.ArrayLike,
    outages: tuple[EarthOutage, ...],
    *,
    sunlight_threshold: float,
    time_step_hours: float,
) -> npt.NDArray[np.float32]:
    """Return complete low-light runs overlapping each outage and pixel."""
    fractions = np.asarray(sunlight_fraction, dtype=np.float32)
    if fractions.ndim != 3 or not np.all(np.isfinite(fractions)):
        raise ValueError("sunlight fractions must be finite (time, y, x) values")
    if not 0.0 <= sunlight_threshold <= 1.0:
        raise ValueError("sunlight_threshold must be between zero and one")
    if not np.isfinite(time_step_hours) or time_step_hours <= 0.0:
        raise ValueError("time_step_hours must be positive and finite")
    return np.stack(
        reduce_safe_haven_patch_stream(
            iter(fractions),
            fractions.shape[0],
            outages,
            sunlight_threshold=sunlight_threshold,
            time_step_hours=time_step_hours,
        )
    ) if outages else np.zeros((0, *fractions.shape[1:]), dtype=np.float32)


def reduce_safe_haven_patch_stream(
    sunlight_fraction_tiles: Iterable[npt.ArrayLike],
    time_count: int,
    outages: tuple[EarthOutage, ...],
    *,
    sunlight_threshold: float,
    time_step_hours: float,
) -> tuple[npt.NDArray[np.float32], ...]:
    """Measure complete low-light runs that overlap each Earth outage.

    A qualifying run may begin before an outage and end after it. The reducer
    remains streaming and retains only per-pixel run and overlap state.
    """
    if time_count < 1:
        raise ValueError("time_count must be positive")
    if not 0.0 <= sunlight_threshold <= 1.0:
        raise ValueError("sunlight_threshold must be between zero and one")
    if not np.isfinite(time_step_hours) or time_step_hours <= 0.0:
        raise ValueError("time_step_hours must be positive and finite")
    previous_stop = 0
    for outage in outages:
        if not previous_stop <= outage.start < outage.stop <= time_count:
            raise ValueError("Earth outages must be ordered, disjoint, and in range")
        previous_stop = outage.stop
    iterator = iter(sunlight_fraction_tiles)
    current = None
    longest = None
    overlaps = None
    shape = None
    for time_index in range(time_count):
        try:
            tile = np.asarray(next(iterator), dtype=np.float32)
        except StopIteration as exc:
            raise ValueError("sunlight fraction iterator ended before time_count") from exc
        if tile.ndim != 2 or not np.all(np.isfinite(tile)):
            raise ValueError("sunlight fraction tiles must be finite 2D arrays")
        if shape is None:
            shape = tile.shape
            current = np.zeros(shape, dtype=np.int32)
            longest = np.zeros((len(outages), *shape), dtype=np.int32)
            overlaps = np.zeros((len(outages), *shape), dtype=np.bool_)
        elif tile.shape != shape:
            raise ValueError("all sunlight fraction tiles must have the same shape")
        low = tile < np.float32(sunlight_threshold)
        current[:] = np.where(low, current + 1, 0)
        for index, outage in enumerate(outages):
            if outage.start <= time_index < outage.stop:
                overlaps[index] |= low
            active = overlaps[index] & low
            longest[index] = np.maximum(
                longest[index], np.where(active, current, 0)
            )
            overlaps[index] &= low
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("sunlight fraction iterator has more entries than time_count")
    assert longest is not None
    durations = longest.astype(np.float32) * np.float32(time_step_hours)
    return tuple(np.ascontiguousarray(tile) for tile in durations)
