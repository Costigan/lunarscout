from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray

from ..alignment import same_grid
from ..errors import PlanningError, TrajectoryInputError
from ..georeference import GeoReference
from ._validation import StaticProblem


_BOUNDARY_SNAP_HOURS = 1.0e-12
_MAX_EXACT_STATES = 1_000_000


def _as_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TrajectoryInputError(
            f"{name} must be a timezone-aware datetime.",
            code="trajectory_invalid_dynamic_time",
            details={"name": name},
        )
    offset = value.utcoffset()
    if offset is None:
        raise TrajectoryInputError(
            f"{name} must be a timezone-aware datetime.",
            code="trajectory_invalid_dynamic_time",
            details={"name": name},
        )
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True, eq=False)
class DynamicOccupancyTimeline:
    boundaries: tuple[datetime, ...]
    allowed: NDArray[np.bool_]
    georef: GeoReference
    _boundary_hours: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.georef, GeoReference):
            raise TrajectoryInputError(
                "Dynamic occupancy georef must be a GeoReference.",
                code="trajectory_invalid_dynamic_grid",
            )
        try:
            boundaries = tuple(
                _as_utc(value, name=f"boundaries[{index}]")
                for index, value in enumerate(self.boundaries)
            )
        except TypeError as exc:
            raise TrajectoryInputError(
                "Dynamic occupancy boundaries must be an iterable of datetimes.",
                code="trajectory_invalid_dynamic_time",
            ) from exc
        if len(boundaries) < 2 or any(
            right <= left for left, right in zip(boundaries, boundaries[1:])
        ):
            raise TrajectoryInputError(
                "Dynamic occupancy boundaries must be strictly increasing and "
                "contain at least two values.",
                code="trajectory_invalid_dynamic_time",
                details={"boundary_count": len(boundaries)},
            )
        allowed_input = np.asarray(self.allowed)
        expected = (
            len(boundaries) - 1,
            self.georef.height,
            self.georef.width,
        )
        if allowed_input.shape != expected:
            raise TrajectoryInputError(
                "Dynamic occupancy must have shape (interval, y, x).",
                code="trajectory_invalid_dynamic_grid",
                details={"shape": list(allowed_input.shape), "expected": list(expected)},
            )
        if not np.issubdtype(allowed_input.dtype, np.bool_):
            raise TrajectoryInputError(
                "Dynamic occupancy must use a Boolean dtype.",
                code="trajectory_invalid_dynamic_grid",
                details={"dtype": str(allowed_input.dtype)},
            )
        allowed = np.array(allowed_input, dtype=np.bool_, copy=True)
        boundary_hours = np.asarray(
            [
                (value - boundaries[0]).total_seconds() / 3600.0
                for value in boundaries
            ],
            dtype=np.float64,
        )
        allowed.flags.writeable = False
        boundary_hours.flags.writeable = False
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "allowed", allowed)
        object.__setattr__(self, "_boundary_hours", boundary_hours)

    @property
    def interval_count(self) -> int:
        return len(self.boundaries) - 1

    @property
    def duration_hours(self) -> float:
        return float(self._boundary_hours[-1])

    def hours_from_start(self, value: datetime, *, name: str) -> float:
        utc = _as_utc(value, name=name)
        return (utc - self.boundaries[0]).total_seconds() / 3600.0

    def datetime_from_hours(self, value: float) -> datetime:
        return self.boundaries[0] + timedelta(hours=value)

    def snap_hour(self, value: float) -> float:
        insertion = int(np.searchsorted(self._boundary_hours, value, side="left"))
        for index in (insertion - 1, insertion):
            if 0 <= index < self._boundary_hours.size:
                boundary = float(self._boundary_hours[index])
                if abs(value - boundary) <= _BOUNDARY_SNAP_HOURS:
                    return boundary
        return value

    def interval_index(self, value: float) -> int | None:
        value = self.snap_hour(value)
        if value < 0.0 or value >= self.duration_hours:
            return None
        return int(np.searchsorted(self._boundary_hours, value, side="right") - 1)


@dataclass(frozen=True, slots=True, eq=False)
class ExactDynamicPathResult:
    reachable: bool
    arrival_time: datetime | None
    cells: NDArray[np.int64] | None
    arrival_times: tuple[datetime, ...] | None
    departure_times: tuple[datetime, ...] | None


def _span_is_allowed(
    timeline: DynamicOccupancyTimeline,
    source: tuple[int, int],
    destination: tuple[int, int],
    departure: float,
    arrival: float,
) -> tuple[bool, int | None]:
    departure = timeline.snap_hour(departure)
    arrival = timeline.snap_hour(arrival)
    departure_interval = timeline.interval_index(departure)
    arrival_interval = timeline.interval_index(arrival)
    if departure_interval is None or arrival_interval is None or arrival <= departure:
        return False, None
    end_interval = int(
        np.searchsorted(timeline._boundary_hours, arrival, side="left") - 1
    )
    source_x, source_y = source
    destination_x, destination_y = destination
    for interval in range(departure_interval, end_interval + 1):
        if not timeline.allowed[interval, source_y, source_x]:
            return False, None
        if not timeline.allowed[interval, destination_y, destination_x]:
            return False, None
    if not timeline.allowed[arrival_interval, destination_y, destination_x]:
        return False, None
    return True, arrival_interval


def _continuous_wait_boundaries(
    timeline: DynamicOccupancyTimeline,
    cell: tuple[int, int],
    arrival: float,
    arrival_interval: int,
) -> tuple[float, ...]:
    x, y = cell
    departures = [arrival]
    interval = arrival_interval + 1
    while interval < timeline.interval_count and timeline.allowed[interval, y, x]:
        departures.append(float(timeline._boundary_hours[interval]))
        interval += 1
    return tuple(departures)


def _reconstruct(
    timeline: DynamicOccupancyTimeline,
    labels: NDArray[np.float64],
    predecessor_x: NDArray[np.int64],
    predecessor_y: NDArray[np.int64],
    predecessor_interval: NDArray[np.int64],
    predecessor_departure: NDArray[np.float64],
    start_state: tuple[int, int, int],
    goal_state: tuple[int, int, int],
) -> ExactDynamicPathResult:
    states = [goal_state]
    departures_reversed: list[float] = []
    interval, y, x = goal_state
    maximum_length = labels.size
    while (interval, y, x) != start_state:
        previous_interval = int(predecessor_interval[interval, y, x])
        previous_x = int(predecessor_x[interval, y, x])
        previous_y = int(predecessor_y[interval, y, x])
        departure = float(predecessor_departure[interval, y, x])
        if (
            previous_interval < 0
            or previous_x < 0
            or previous_y < 0
            or not np.isfinite(departure)
            or len(states) >= maximum_length
        ):
            raise PlanningError(
                "The exact dynamic predecessor chain is invalid.",
                code="trajectory_invalid_dynamic_predecessor",
            )
        departures_reversed.append(departure)
        interval, y, x = previous_interval, previous_y, previous_x
        states.append((interval, y, x))
    states.reverse()
    departures_reversed.reverse()
    cells = np.asarray([(state[2], state[1]) for state in states], dtype=np.int64)
    cells.flags.writeable = False
    arrival_hours = [labels[state] for state in states]
    arrival_times = tuple(timeline.datetime_from_hours(value) for value in arrival_hours)
    departure_times = tuple(
        timeline.datetime_from_hours(value) for value in departures_reversed
    )
    return ExactDynamicPathResult(
        reachable=True,
        arrival_time=arrival_times[-1],
        cells=cells,
        arrival_times=arrival_times,
        departure_times=departure_times,
    )


def exact_dynamic_path(
    problem: StaticProblem,
    timeline: DynamicOccupancyTimeline,
    departure_time: datetime,
) -> ExactDynamicPathResult:
    """Return the private exact earliest-arrival path for a small problem."""

    if problem.goal is None:
        raise TrajectoryInputError(
            "The exact dynamic oracle requires a goal cell.",
            code="trajectory_dynamic_goal_required",
        )
    try:
        grids_match = same_grid(problem.georef, timeline.georef)
    except Exception as exc:
        raise TrajectoryInputError(
            "Unable to compare static and dynamic trajectory grids.",
            code="trajectory_invalid_dynamic_grid",
            details={"error": str(exc)},
        ) from exc
    if not grids_match:
        raise TrajectoryInputError(
            "Static and dynamic trajectory grids must match.",
            code="trajectory_dynamic_grid_mismatch",
        )
    departure = timeline.snap_hour(
        timeline.hours_from_start(departure_time, name="departure_time")
    )
    departure_interval = timeline.interval_index(departure)
    if departure_interval is None:
        raise TrajectoryInputError(
            "departure_time is outside the dynamic occupancy timeline.",
            code="trajectory_departure_outside_timeline",
        )
    start_x, start_y = problem.start
    if not timeline.allowed[departure_interval, start_y, start_x]:
        raise TrajectoryInputError(
            "The start cell is dynamically unavailable at departure_time.",
            code="trajectory_dynamic_start_unavailable",
            details={"start": [start_x, start_y]},
        )

    shape = (timeline.interval_count, problem.georef.height, problem.georef.width)
    state_count = int(np.prod(shape, dtype=np.int64))
    if state_count > _MAX_EXACT_STATES:
        raise PlanningError(
            "The exact dynamic problem exceeds the small-oracle state limit.",
            code="trajectory_exact_dynamic_state_limit",
            details={"state_count": state_count, "maximum": _MAX_EXACT_STATES},
        )
    labels = np.full(shape, np.inf, dtype=np.float64)
    predecessor_x = np.full(shape, -1, dtype=np.int64)
    predecessor_y = np.full(shape, -1, dtype=np.int64)
    predecessor_interval = np.full(shape, -1, dtype=np.int64)
    predecessor_departure = np.full(shape, np.inf, dtype=np.float64)
    start_state = (departure_interval, start_y, start_x)
    labels[start_state] = departure
    frontier: list[tuple[float, int, int, int]] = [
        (departure, departure_interval, start_y, start_x)
    ]

    while frontier:
        arrival, interval, y, x = heappop(frontier)
        state = (interval, y, x)
        if arrival != labels[state]:
            continue
        if (x, y) == problem.goal:
            return _reconstruct(
                timeline,
                labels,
                predecessor_x,
                predecessor_y,
                predecessor_interval,
                predecessor_departure,
                start_state,
                state,
            )
        departures = _continuous_wait_boundaries(
            timeline, (x, y), arrival, interval
        )
        for step in problem.steps:
            duration = problem.transition_time(x, y, step)
            if not np.isfinite(duration):
                continue
            destination = (x + step.dx, y + step.dy)
            for candidate_departure in departures:
                candidate_arrival = timeline.snap_hour(candidate_departure + duration)
                allowed, destination_interval = _span_is_allowed(
                    timeline,
                    (x, y),
                    destination,
                    candidate_departure,
                    candidate_arrival,
                )
                if not allowed or destination_interval is None:
                    continue
                destination_state = (
                    destination_interval,
                    destination[1],
                    destination[0],
                )
                if candidate_arrival < labels[destination_state]:
                    labels[destination_state] = candidate_arrival
                    predecessor_x[destination_state] = x
                    predecessor_y[destination_state] = y
                    predecessor_interval[destination_state] = interval
                    predecessor_departure[destination_state] = candidate_departure
                    heappush(
                        frontier,
                        (
                            candidate_arrival,
                            destination_interval,
                            destination[1],
                            destination[0],
                        ),
                    )

    return ExactDynamicPathResult(
        reachable=False,
        arrival_time=None,
        cells=None,
        arrival_times=None,
        departure_times=None,
    )
