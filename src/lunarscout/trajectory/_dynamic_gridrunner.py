from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from heapq import heappop, heappush
from itertools import count
from numbers import Integral

import numpy as np

from ..alignment import same_grid
from ..errors import PlanningError, TrajectoryInputError
from ._dynamic_mobility import CompiledDynamicTravelModel
from ._dynamic_reference import (
    DynamicOccupancyTimeline,
    ExactDynamicPathResult,
    _reconstruct,
)
from ._environment import occupancy_timeline_from_provider
from ._geometry import cell_distance_m
from .providers import ConfigurationSpaceProvider
from ._validation import StaticProblem


_MAX_GRIDRUNNER_STATES = 5_000_000


@dataclass(frozen=True, slots=True)
class GridRunnerDiagnostics:
    block_activations: int
    block_reactivations: int
    blocks_processed: int
    local_passes: int
    state_relaxations: int


@dataclass(frozen=True, slots=True)
class GridRunnerResult:
    path: ExactDynamicPathResult
    diagnostics: GridRunnerDiagnostics


@dataclass(frozen=True, slots=True)
class SpaceTimePhysics:
    problem: StaticProblem
    timeline: DynamicOccupancyTimeline
    travel_model: CompiledDynamicTravelModel | None

    @property
    def minimum_factor(self) -> float:
        dynamic = (
            1.0
            if self.travel_model is None
            else float(np.min(self.travel_model.interval_factors))
        )
        return self.problem.minimum_factor * dynamic

    def departures(self, x: int, y: int, arrival: float, interval: int):
        return _departures(self.timeline, x, y, arrival, interval)

    def candidate_arrival(
        self,
        x: int,
        y: int,
        direction: int,
        departure: float,
    ) -> float:
        if self.travel_model is None:
            step = self.problem.steps[direction]
            duration = self.problem.transition_time(x, y, step)
            return self.timeline.snap_hour(departure + duration)
        return self.travel_model.arrival_hours(
            self.timeline,
            x,
            y,
            direction,
            departure,
        )

    def span_is_allowed(
        self,
        source_x: int,
        source_y: int,
        destination_x: int,
        destination_y: int,
        departure: float,
        arrival: float,
    ) -> tuple[bool, int | None]:
        return _span_is_allowed(
            self.timeline,
            source_x,
            source_y,
            destination_x,
            destination_y,
            departure,
            arrival,
        )


def _span_is_allowed(
    timeline: DynamicOccupancyTimeline,
    source_x: int,
    source_y: int,
    destination_x: int,
    destination_y: int,
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
    for interval in range(departure_interval, end_interval + 1):
        if not timeline.allowed[interval, source_y, source_x]:
            return False, None
        if not timeline.allowed[interval, destination_y, destination_x]:
            return False, None
    if not timeline.allowed[arrival_interval, destination_y, destination_x]:
        return False, None
    return True, arrival_interval


def _departures(
    timeline: DynamicOccupancyTimeline,
    x: int,
    y: int,
    arrival: float,
    interval: int,
):
    yield arrival
    next_interval = interval + 1
    while (
        next_interval < timeline.interval_count
        and timeline.allowed[next_interval, y, x]
    ):
        yield float(timeline._boundary_hours[next_interval])
        next_interval += 1


def gridrunner_dynamic_path(
    problem: StaticProblem,
    timeline: DynamicOccupancyTimeline,
    departure_time: datetime,
    *,
    travel_model: CompiledDynamicTravelModel | None = None,
    block_size: int = 8,
) -> GridRunnerResult:
    """Run the private exhaustive block-reactivation dynamic planner."""

    if problem.goal is None:
        raise TrajectoryInputError(
            "GridRunner requires a goal cell.",
            code="trajectory_dynamic_goal_required",
        )
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, Integral)
        or int(block_size) < 1
    ):
        raise TrajectoryInputError(
            "block_size must be a positive integer.",
            code="trajectory_invalid_block_size",
            details={"block_size": block_size},
        )
    block_size = int(block_size)
    try:
        grids_match = same_grid(problem.georef, timeline.georef)
    except Exception as exc:
        raise TrajectoryInputError(
            "Unable to compare GridRunner trajectory grids.",
            code="trajectory_invalid_dynamic_grid",
            details={"error": str(exc)},
        ) from exc
    if not grids_match:
        raise TrajectoryInputError(
            "Static and dynamic GridRunner grids must match.",
            code="trajectory_dynamic_grid_mismatch",
        )
    if travel_model is not None:
        travel_model.require_compatible(problem, timeline)
    physics = SpaceTimePhysics(problem, timeline, travel_model)

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

    intervals = timeline.interval_count
    height = problem.georef.height
    width = problem.georef.width
    state_count = intervals * height * width
    if state_count > _MAX_GRIDRUNNER_STATES:
        raise PlanningError(
            "The GridRunner state arrays exceed their configured bound.",
            code="trajectory_gridrunner_state_limit",
            details={"state_count": state_count, "maximum": _MAX_GRIDRUNNER_STATES},
        )
    shape = (intervals, height, width)
    labels = np.full(shape, np.inf, dtype=np.float64)
    predecessor_x = np.full(shape, -1, dtype=np.int64)
    predecessor_y = np.full(shape, -1, dtype=np.int64)
    predecessor_interval = np.full(shape, -1, dtype=np.int64)
    predecessor_departure = np.full(shape, np.inf, dtype=np.float64)
    start_state = (departure_interval, start_y, start_x)
    labels[start_state] = departure

    block_rows = (height + block_size - 1) // block_size
    block_columns = (width + block_size - 1) // block_size
    active = np.zeros((block_rows, block_columns), dtype=np.bool_)
    versions = np.zeros((block_rows, block_columns), dtype=np.int64)
    processed = np.zeros((block_rows, block_columns), dtype=np.int64)
    sequence = count()
    queue: list[tuple[float, int, int, int, int]] = []
    goal_x, goal_y = problem.goal
    minimum_factor = physics.minimum_factor
    block_heuristics = np.empty((block_rows, block_columns), dtype=np.float64)
    for block_y in range(block_rows):
        y0 = block_y * block_size
        y1 = min(y0 + block_size, height)
        for block_x in range(block_columns):
            x0 = block_x * block_size
            x1 = min(x0 + block_size, width)
            distance_lower_bound = min(
                cell_distance_m(
                    problem.georef,
                    (x, y),
                    (goal_x, goal_y),
                    units_to_metres=problem.units_to_metres,
                )
                for y in range(y0, y1)
                for x in range(x0, x1)
            )
            block_heuristics[block_y, block_x] = (
                distance_lower_bound
                / problem.model.speed_m_per_h
                * minimum_factor
            )

    def priority(block_y: int, block_x: int) -> float:
        y0 = block_y * block_size
        y1 = min(y0 + block_size, height)
        x0 = block_x * block_size
        x1 = min(x0 + block_size, width)
        arrival_lower_bound = float(np.min(labels[:, y0:y1, x0:x1]))
        if not np.isfinite(arrival_lower_bound):
            return np.inf
        return arrival_lower_bound + block_heuristics[block_y, block_x]

    activations = 0
    reactivations = 0

    def schedule(block_y: int, block_x: int) -> None:
        nonlocal activations, reactivations
        if not active[block_y, block_x]:
            activations += 1
            if processed[block_y, block_x] > 0:
                reactivations += 1
            active[block_y, block_x] = True
        versions[block_y, block_x] += 1
        version = int(versions[block_y, block_x])
        heappush(
            queue,
            (priority(block_y, block_x), next(sequence), version, block_y, block_x),
        )

    schedule(start_y // block_size, start_x // block_size)
    blocks_processed = 0
    total_local_passes = 0
    state_relaxations = 0

    while queue:
        _priority, _order, version, block_y, block_x = heappop(queue)
        if not active[block_y, block_x] or version != versions[block_y, block_x]:
            continue
        active[block_y, block_x] = False
        processed[block_y, block_x] += 1
        blocks_processed += 1
        y0 = block_y * block_size
        y1 = min(y0 + block_size, height)
        x0 = block_x * block_size
        x1 = min(x0 + block_size, width)
        local_pass_limit = max(64, intervals * (y1 - y0) * (x1 - x0) * 4)
        local_pass = 0
        changed = True
        while changed:
            changed = False
            local_pass += 1
            total_local_passes += 1
            if local_pass > local_pass_limit:
                raise PlanningError(
                    "GridRunner did not reach local block quiescence.",
                    code="trajectory_gridrunner_nonconvergence",
                    details={"block_x": block_x, "block_y": block_y},
                )
            for interval in range(intervals):
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        arrival = float(labels[interval, y, x])
                        if not np.isfinite(arrival):
                            continue
                        for direction, step in enumerate(problem.steps):
                            destination_x = x + step.dx
                            destination_y = y + step.dy
                            if not (
                                0 <= destination_x < width
                                and 0 <= destination_y < height
                            ):
                                continue
                            for candidate_departure in physics.departures(
                                x, y, arrival, interval
                            ):
                                candidate_arrival = physics.candidate_arrival(
                                    x,
                                    y,
                                    direction,
                                    candidate_departure,
                                )
                                if not np.isfinite(candidate_arrival):
                                    continue
                                allowed, destination_interval = physics.span_is_allowed(
                                    x,
                                    y,
                                    destination_x,
                                    destination_y,
                                    candidate_departure,
                                    candidate_arrival,
                                )
                                if not allowed or destination_interval is None:
                                    continue
                                destination_state = (
                                    destination_interval,
                                    destination_y,
                                    destination_x,
                                )
                                if candidate_arrival >= labels[destination_state]:
                                    continue
                                labels[destination_state] = candidate_arrival
                                predecessor_x[destination_state] = x
                                predecessor_y[destination_state] = y
                                predecessor_interval[destination_state] = interval
                                predecessor_departure[
                                    destination_state
                                ] = candidate_departure
                                state_relaxations += 1
                                destination_block_y = destination_y // block_size
                                destination_block_x = destination_x // block_size
                                if (
                                    destination_block_y == block_y
                                    and destination_block_x == block_x
                                ):
                                    changed = True
                                else:
                                    schedule(destination_block_y, destination_block_x)

    best_interval = int(np.argmin(labels[:, goal_y, goal_x]))
    goal_state = (best_interval, goal_y, goal_x)
    if np.isfinite(labels[goal_state]):
        path = _reconstruct(
            timeline,
            labels,
            predecessor_x,
            predecessor_y,
            predecessor_interval,
            predecessor_departure,
            start_state,
            goal_state,
        )
    else:
        path = ExactDynamicPathResult(False, None, None, None, None)
    return GridRunnerResult(
        path=path,
        diagnostics=GridRunnerDiagnostics(
            block_activations=activations,
            block_reactivations=reactivations,
            blocks_processed=blocks_processed,
            local_passes=total_local_passes,
            state_relaxations=state_relaxations,
        ),
    )


def gridrunner_dynamic_path_from_provider(
    problem: StaticProblem,
    provider: ConfigurationSpaceProvider,
    boundaries: Iterable[datetime],
    departure_time: datetime,
    *,
    travel_model: CompiledDynamicTravelModel | None = None,
    block_size: int = 8,
) -> GridRunnerResult:
    """Materialize bounded provider windows, then run private GridRunner."""

    timeline = occupancy_timeline_from_provider(
        provider,
        boundaries,
        read_block_size=block_size,
    )
    return gridrunner_dynamic_path(
        problem,
        timeline,
        departure_time,
        travel_model=travel_model,
        block_size=block_size,
    )
