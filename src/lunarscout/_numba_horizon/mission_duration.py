"""Private landed-mission candidate intervals and streaming reductions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import numpy.typing as npt


DurationUnit = Literal["hours", "days"]


def _utc(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("mission-duration timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class CandidateStartInterval:
    """Half-open interval in which a qualifying mission may start."""

    start_utc: datetime
    stop_utc: datetime

    def __post_init__(self) -> None:
        start = _utc(self.start_utc)
        stop = _utc(self.stop_utc)
        if start >= stop:
            raise ValueError("candidate-start interval must have positive duration")
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "stop_utc", stop)


def candidate_start_intervals(
    values: Sequence[CandidateStartInterval | tuple[datetime | str, datetime | str]],
    *,
    evaluation_start_utc: datetime | str,
    evaluation_stop_utc: datetime | str,
) -> tuple[CandidateStartInterval, ...]:
    """Normalize explicit intervals and require them inside the evaluation range."""
    evaluation_start = _utc(evaluation_start_utc)
    evaluation_stop = _utc(evaluation_stop_utc)
    if evaluation_start >= evaluation_stop:
        raise ValueError("evaluation interval must have positive duration")
    intervals = tuple(
        value
        if isinstance(value, CandidateStartInterval)
        else CandidateStartInterval(_utc(value[0]), _utc(value[1]))
        for value in values
    )
    if not intervals:
        raise ValueError("at least one candidate-start interval is required")
    for interval in intervals:
        if (
            interval.start_utc < evaluation_start
            or interval.stop_utc > evaluation_stop
        ):
            raise ValueError(
                "candidate-start intervals must be inside the evaluation interval"
            )
    return intervals


def monthly_candidate_intervals(
    start_utc: datetime | str, stop_utc: datetime | str
) -> tuple[CandidateStartInterval, ...]:
    """Return UTC calendar-month intervals clipped to ``[start, stop)``."""
    start = _utc(start_utc)
    stop = _utc(stop_utc)
    if start >= stop:
        raise ValueError("interval range must have positive duration")
    boundary = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    output: list[CandidateStartInterval] = []
    while boundary < stop:
        if boundary.month == 12:
            following = datetime(boundary.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            following = datetime(
                boundary.year, boundary.month + 1, 1, tzinfo=timezone.utc
            )
        interval_start = max(start, boundary)
        interval_stop = min(stop, following)
        if interval_start < interval_stop:
            output.append(CandidateStartInterval(interval_start, interval_stop))
        boundary = following
    return tuple(output)


def weekly_candidate_intervals(
    start_utc: datetime | str, stop_utc: datetime | str
) -> tuple[CandidateStartInterval, ...]:
    """Return consecutive seven-day intervals anchored at ``start_utc``."""
    return fixed_candidate_intervals(start_utc, stop_utc, duration=timedelta(days=7))


def fixed_candidate_intervals(
    start_utc: datetime | str,
    stop_utc: datetime | str,
    *,
    duration: timedelta,
) -> tuple[CandidateStartInterval, ...]:
    """Return fixed-duration intervals anchored at ``start_utc`` and clipped."""
    start = _utc(start_utc)
    stop = _utc(stop_utc)
    if start >= stop:
        raise ValueError("interval range must have positive duration")
    if duration <= timedelta(0):
        raise ValueError("fixed interval duration must be positive")
    output: list[CandidateStartInterval] = []
    cursor = start
    while cursor < stop:
        following = min(stop, cursor + duration)
        output.append(CandidateStartInterval(cursor, following))
        cursor = following
    return tuple(output)


def validate_evaluation_samples(
    times_utc: Sequence[datetime | str],
    *,
    evaluation_start_utc: datetime | str,
    evaluation_stop_utc: datetime | str,
) -> tuple[tuple[datetime, ...], npt.NDArray[np.float64]]:
    """Return UTC samples and following-interval durations clipped at stop."""
    evaluation_start = _utc(evaluation_start_utc)
    evaluation_stop = _utc(evaluation_stop_utc)
    if evaluation_start >= evaluation_stop:
        raise ValueError("evaluation interval must have positive duration")
    times = tuple(_utc(value) for value in times_utc)
    if not times:
        raise ValueError("at least one condition sample is required")
    if times[0] != evaluation_start or times[-1] > evaluation_stop:
        raise ValueError(
            "samples must start at evaluation_start and not exceed evaluation_stop"
        )
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("sample timestamps must be strictly increasing")
    durations = np.fromiter(
        (
            (min(times[index + 1], evaluation_stop) - time).total_seconds()
            / 3600.0
                if index + 1 < len(times)
                else (evaluation_stop - time).total_seconds() / 3600.0
            for index, time in enumerate(times)
        ),
        dtype=np.float64,
        count=len(times),
    )
    return times, durations


def reduce_longest_candidate_duration_stream(
    condition_tiles: Iterable[npt.ArrayLike],
    *,
    times_utc: Sequence[datetime | str],
    evaluation_start_utc: datetime | str,
    evaluation_stop_utc: datetime | str,
    start_intervals: Sequence[
        CandidateStartInterval | tuple[datetime | str, datetime | str]
    ],
    output_unit: DurationUnit = "hours",
) -> tuple[npt.NDArray[np.float32], ...]:
    """Reduce a condition stream to longest right-censored candidate durations.

    The condition at sample ``i`` applies to ``[time[i], time[i + 1])``. A
    candidate may start at a true sample inside its half-open start interval.
    Once started it continues beyond that interval while true. An active
    candidate at the evaluation stop receives credit through the stop without
    implying that the real-world condition ends there.
    """
    times, sample_hours = validate_evaluation_samples(
        times_utc,
        evaluation_start_utc=evaluation_start_utc,
        evaluation_stop_utc=evaluation_stop_utc,
    )
    intervals = candidate_start_intervals(
        start_intervals,
        evaluation_start_utc=evaluation_start_utc,
        evaluation_stop_utc=evaluation_stop_utc,
    )
    if output_unit not in ("hours", "days"):
        raise ValueError("output_unit must be 'hours' or 'days'")

    iterator = iter(condition_tiles)
    active = None
    current = None
    longest = None
    has_active = None
    shape = None
    for time_index, time in enumerate(times):
        try:
            condition = np.asarray(next(iterator), dtype=np.bool_)
        except StopIteration as exc:
            raise ValueError("condition iterator ended before the time axis") from exc
        if condition.ndim != 2:
            raise ValueError("condition tiles must be two-dimensional")
        if shape is None:
            shape = condition.shape
            state_shape = (len(intervals), *shape)
            active = np.zeros(state_shape, dtype=np.bool_)
            current = np.zeros(state_shape, dtype=np.float64)
            longest = np.zeros(state_shape, dtype=np.float64)
            has_active = np.zeros(len(intervals), dtype=np.bool_)
        elif condition.shape != shape:
            raise ValueError("all condition tiles must have the same shape")

        assert (
            active is not None
            and current is not None
            and longest is not None
            and has_active is not None
        )
        false_now = ~condition
        for interval_index, interval in enumerate(intervals):
            accepts_start = interval.start_utc <= time < interval.stop_utc
            if not accepts_start and not has_active[interval_index]:
                continue
            band_active = active[interval_index]
            ended = band_active & false_now
            if np.any(ended):
                longest[interval_index][ended] = np.maximum(
                    longest[interval_index][ended], current[interval_index][ended]
                )
                current[interval_index][ended] = 0.0
                band_active[ended] = False
            if accepts_start:
                band_active |= condition
            accruing = band_active & condition
            current[interval_index][accruing] += sample_hours[time_index]
            has_active[interval_index] = np.any(band_active)

    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("condition iterator has more entries than the time axis")
    assert current is not None and longest is not None
    longest = np.maximum(longest, current)
    if output_unit == "days":
        longest /= 24.0
    return tuple(
        np.ascontiguousarray(tile, dtype=np.float32) for tile in longest
    )
