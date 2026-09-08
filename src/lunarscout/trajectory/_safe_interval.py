from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from heapq import heappop, heappush
from itertools import count

import numpy as np

from ..alignment import same_grid
from ..errors import PlanningError, TrajectoryInputError
from ._dynamic_reference import DynamicOccupancyTimeline, ExactDynamicPathResult
from ._environment import occupancy_timeline_from_provider
from ._time_contract import IntervalTimeAxis
from ._validation import StaticProblem
from .providers import ConfigurationSpaceProvider


_MAX_SAFE_INTERVAL_TIMELINE_STATES = 5_000_000


@dataclass(frozen=True, slots=True)
class SafeInterval:
    start_hours: float
    stop_hours: float
    first_interval: int
    stop_interval: int


@dataclass(frozen=True, slots=True)
class SafeIntervalTable:
    cells: tuple[tuple[tuple[SafeInterval, ...], ...], ...]
    safe_interval_count: int

    def intervals(self, x: int, y: int) -> tuple[SafeInterval, ...]:
        return self.cells[y][x]


@dataclass(frozen=True, slots=True)
class SafeIntervalDiagnostics:
    safe_intervals: int
    expanded_states: int
    state_relaxations: int


@dataclass(frozen=True, slots=True)
class SafeIntervalResult:
    path: ExactDynamicPathResult
    diagnostics: SafeIntervalDiagnostics


def _require_timeline_capacity(intervals: int, height: int, width: int) -> None:
    state_count = intervals * height * width
    if state_count > _MAX_SAFE_INTERVAL_TIMELINE_STATES:
        raise PlanningError(
            "The safe-interval occupancy timeline exceeds its configured bound.",
            code="trajectory_safe_interval_state_limit",
            details={
                "state_count": state_count,
                "maximum": _MAX_SAFE_INTERVAL_TIMELINE_STATES,
            },
        )


def build_safe_intervals(
    timeline: DynamicOccupancyTimeline,
) -> SafeIntervalTable:
    """Build maximal half-open allowed intervals for every raster cell."""

    height = timeline.georef.height
    width = timeline.georef.width
    rows: list[tuple[tuple[SafeInterval, ...], ...]] = []
    total = 0
    for y in range(height):
        row: list[tuple[SafeInterval, ...]] = []
        for x in range(width):
            cell: list[SafeInterval] = []
            interval = 0
            while interval < timeline.interval_count:
                if not timeline.allowed[interval, y, x]:
                    interval += 1
                    continue
                first = interval
                interval += 1
                while (
                    interval < timeline.interval_count
                    and timeline.allowed[interval, y, x]
                ):
                    interval += 1
                cell.append(
                    SafeInterval(
                        start_hours=float(timeline._boundary_hours[first]),
                        stop_hours=float(timeline._boundary_hours[interval]),
                        first_interval=first,
                        stop_interval=interval,
                    )
                )
            total += len(cell)
            row.append(tuple(cell))
        rows.append(tuple(row))
    return SafeIntervalTable(tuple(rows), total)


def _validate_problem(
    problem: StaticProblem,
    timeline: DynamicOccupancyTimeline,
    departure_time: datetime,
) -> tuple[float, int]:
    if problem.goal is None:
        raise TrajectoryInputError(
            "Safe-interval planning requires a goal cell.",
            code="trajectory_dynamic_goal_required",
        )
    try:
        grids_match = same_grid(problem.georef, timeline.georef)
    except Exception as exc:
        raise TrajectoryInputError(
            "Unable to compare safe-interval trajectory grids.",
            code="trajectory_invalid_dynamic_grid",
            details={"error": str(exc)},
        ) from exc
    if not grids_match:
        raise TrajectoryInputError(
            "Static and dynamic safe-interval grids must match.",
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
    return departure, departure_interval


def _reconstruct(
    timeline: DynamicOccupancyTimeline,
    labels: dict[tuple[int, int, int], float],
    predecessors: dict[
        tuple[int, int, int], tuple[tuple[int, int, int], float]
    ],
    start_state: tuple[int, int, int],
    goal_state: tuple[int, int, int],
) -> ExactDynamicPathResult:
    states = [goal_state]
    departures: list[float] = []
    state = goal_state
    while state != start_state:
        if state not in predecessors or len(states) > len(labels):
            raise PlanningError(
                "The safe-interval predecessor chain is invalid.",
                code="trajectory_invalid_dynamic_predecessor",
            )
        state, departure = predecessors[state]
        states.append(state)
        departures.append(departure)
    states.reverse()
    departures.reverse()
    cells = np.asarray([(state[1], state[0]) for state in states], dtype=np.int64)
    cells.flags.writeable = False
    arrivals = tuple(
        timeline.datetime_from_hours(labels[state]) for state in states
    )
    return ExactDynamicPathResult(
        True,
        arrivals[-1],
        cells,
        arrivals,
        tuple(timeline.datetime_from_hours(value) for value in departures),
    )


def safe_interval_dynamic_path(
    problem: StaticProblem,
    timeline: DynamicOccupancyTimeline,
    departure_time: datetime,
) -> SafeIntervalResult:
    """Return an exact no-SOC path using maximal safe-interval states."""

    departure, departure_interval = _validate_problem(
        problem, timeline, departure_time
    )
    _require_timeline_capacity(
        timeline.interval_count,
        problem.georef.height,
        problem.georef.width,
    )
    table = build_safe_intervals(timeline)
    start_x, start_y = problem.start
    start_safe_index = next(
        index
        for index, safe in enumerate(table.intervals(start_x, start_y))
        if safe.first_interval <= departure_interval < safe.stop_interval
    )
    start_state = (start_y, start_x, start_safe_index)
    labels = {start_state: departure}
    predecessors: dict[
        tuple[int, int, int], tuple[tuple[int, int, int], float]
    ] = {}
    sequence = count()
    frontier = [(departure, next(sequence), start_state)]
    expanded = 0
    relaxations = 0

    while frontier:
        arrival, _order, state = heappop(frontier)
        if arrival != labels[state]:
            continue
        expanded += 1
        y, x, safe_index = state
        if (x, y) == problem.goal:
            return SafeIntervalResult(
                _reconstruct(
                    timeline,
                    labels,
                    predecessors,
                    start_state,
                    state,
                ),
                SafeIntervalDiagnostics(
                    table.safe_interval_count,
                    expanded,
                    relaxations,
                ),
            )
        source_safe = table.intervals(x, y)[safe_index]
        for step in problem.steps:
            duration = problem.transition_time(x, y, step)
            if not np.isfinite(duration):
                continue
            destination_x = x + step.dx
            destination_y = y + step.dy
            for destination_safe_index, destination_safe in enumerate(
                table.intervals(destination_x, destination_y)
            ):
                candidate_departure = max(arrival, destination_safe.start_hours)
                if candidate_departure >= source_safe.stop_hours:
                    break
                candidate_arrival = timeline.snap_hour(
                    candidate_departure + duration
                )
                if candidate_arrival > source_safe.stop_hours:
                    continue
                if candidate_arrival >= destination_safe.stop_hours:
                    continue
                destination_state = (
                    destination_y,
                    destination_x,
                    destination_safe_index,
                )
                if candidate_arrival >= labels.get(destination_state, np.inf):
                    continue
                labels[destination_state] = candidate_arrival
                predecessors[destination_state] = (state, candidate_departure)
                relaxations += 1
                heappush(
                    frontier,
                    (candidate_arrival, next(sequence), destination_state),
                )

    return SafeIntervalResult(
        ExactDynamicPathResult(False, None, None, None, None),
        SafeIntervalDiagnostics(table.safe_interval_count, expanded, relaxations),
    )


def safe_interval_dynamic_path_from_provider(
    problem: StaticProblem,
    provider: ConfigurationSpaceProvider,
    boundaries: Iterable[datetime],
    departure_time: datetime,
    *,
    read_block_size: int = 128,
) -> SafeIntervalResult:
    """Materialize bounded provider windows, then run safe-interval search."""

    axis = IntervalTimeAxis(boundaries)
    _require_timeline_capacity(
        axis.interval_count,
        problem.georef.height,
        problem.georef.width,
    )
    timeline = occupancy_timeline_from_provider(
        provider,
        axis.boundaries,
        read_block_size=read_block_size,
    )
    return safe_interval_dynamic_path(problem, timeline, departure_time)
