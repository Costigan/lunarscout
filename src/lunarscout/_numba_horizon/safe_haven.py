"""Private safe-haven interval and bounded patch-reduction semantics."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Return longest low-light duration for each outage and pixel."""
    fractions = np.asarray(sunlight_fraction, dtype=np.float32)
    if fractions.ndim != 3 or not np.all(np.isfinite(fractions)):
        raise ValueError("sunlight fractions must be finite (time, y, x) values")
    if not 0.0 <= sunlight_threshold <= 1.0:
        raise ValueError("sunlight_threshold must be between zero and one")
    if not np.isfinite(time_step_hours) or time_step_hours <= 0.0:
        raise ValueError("time_step_hours must be positive and finite")
    output = np.zeros((len(outages), *fractions.shape[1:]), dtype=np.float32)
    for outage_index, outage in enumerate(outages):
        if not 0 <= outage.start < outage.stop <= fractions.shape[0]:
            raise ValueError("Earth outage falls outside the sunlight time axis")
        current = np.zeros(fractions.shape[1:], dtype=np.int32)
        longest = np.zeros_like(current)
        for time_index in range(outage.start, outage.stop):
            low = fractions[time_index] < np.float32(sunlight_threshold)
            current = np.where(low, current + 1, 0)
            longest = np.maximum(longest, current)
        output[outage_index] = longest * np.float32(time_step_hours)
    return output
