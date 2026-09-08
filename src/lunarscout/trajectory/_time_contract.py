from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.typing import NDArray

from ..errors import TrajectoryInputError


BOUNDARY_SNAP_HOURS = 1.0e-12


def as_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TrajectoryInputError(
            f"{name} must be a timezone-aware datetime.",
            code="trajectory_invalid_dynamic_time",
            details={"name": name},
        )
    if value.utcoffset() is None:
        raise TrajectoryInputError(
            f"{name} must be a timezone-aware datetime.",
            code="trajectory_invalid_dynamic_time",
            details={"name": name},
        )
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True, eq=False)
class IntervalTimeAxis:
    """Private implementation of the trajectory half-open UTC time contract."""

    boundaries: tuple[datetime, ...]
    boundary_hours: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            boundaries = tuple(
                as_utc(value, name=f"boundaries[{index}]")
                for index, value in enumerate(self.boundaries)
            )
        except TypeError as exc:
            raise TrajectoryInputError(
                "Time boundaries must be an iterable of datetimes.",
                code="trajectory_invalid_dynamic_time",
            ) from exc
        if len(boundaries) < 2 or any(
            right <= left for left, right in zip(boundaries, boundaries[1:])
        ):
            raise TrajectoryInputError(
                "Time boundaries must be strictly increasing and contain at "
                "least two values.",
                code="trajectory_invalid_dynamic_time",
                details={"boundary_count": len(boundaries)},
            )
        hours = np.asarray(
            [
                (value - boundaries[0]).total_seconds() / 3600.0
                for value in boundaries
            ],
            dtype=np.float64,
        )
        hours.flags.writeable = False
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "boundary_hours", hours)

    @property
    def interval_count(self) -> int:
        return len(self.boundaries) - 1

    @property
    def duration_hours(self) -> float:
        return float(self.boundary_hours[-1])

    def hours_from_start(self, value: datetime, *, name: str) -> float:
        utc = as_utc(value, name=name)
        return (utc - self.boundaries[0]).total_seconds() / 3600.0

    def datetime_from_hours(self, value: float) -> datetime:
        return self.boundaries[0] + timedelta(hours=value)

    def snap_hour(self, value: float) -> float:
        insertion = int(np.searchsorted(self.boundary_hours, value, side="left"))
        for index in (insertion - 1, insertion):
            if 0 <= index < self.boundary_hours.size:
                boundary = float(self.boundary_hours[index])
                if abs(value - boundary) <= BOUNDARY_SNAP_HOURS:
                    return boundary
        return value

    def interval_index_from_hours(self, value: float) -> int | None:
        value = self.snap_hour(value)
        if value < 0.0 or value >= self.duration_hours:
            return None
        return int(np.searchsorted(self.boundary_hours, value, side="right") - 1)

    def interval_index(self, value: datetime, *, name: str = "time") -> int:
        hours = self.snap_hour(self.hours_from_start(value, name=name))
        index = self.interval_index_from_hours(hours)
        if index is None:
            raise TrajectoryInputError(
                f"{name} is outside the provider time coverage.",
                code="trajectory_provider_time_out_of_range",
                details={
                    "name": name,
                    "time": as_utc(value, name=name).isoformat(),
                    "start": self.boundaries[0].isoformat(),
                    "stop": self.boundaries[-1].isoformat(),
                },
            )
        return index
