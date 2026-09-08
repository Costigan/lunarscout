from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from heapq import heappop, heappush
from itertools import count
from numbers import Integral

import numpy as np

from ..alignment import same_grid
from ..errors import PlanningError, TrajectoryInputError
from ._dynamic_reference import (
    DynamicOccupancyTimeline,
    ExactDynamicPathResult,
    _continuous_wait_boundaries,
    _span_is_allowed,
)
from ._power_accounting import (
    PiecewiseSunlightTimeline,
    integrate_timeline_operation,
)
from ._validation import StaticProblem
from .power import (
    BatteryModel,
    EnergySegment,
    EnergyTimelineResult,
    RoverPowerModel,
    SolarPowerModel,
)


_DEFAULT_MAX_LABELS = 100_000
_TIME_TOLERANCE_HOURS = 1.0e-9


@dataclass(slots=True)
class _SocLabel:
    x: int
    y: int
    arrival_hours: float
    energy_wh: float
    predecessor: int | None
    departure_hours: float | None
    active: bool = True


@dataclass(frozen=True, slots=True)
class ExactSocDiagnostics:
    labels_created: int
    labels_expanded: int
    candidates_dominated: int
    labels_retired: int


@dataclass(frozen=True, slots=True, eq=False)
class ExactSocPathResult:
    path: ExactDynamicPathResult
    energy: EnergyTimelineResult | None
    diagnostics: ExactSocDiagnostics

    @property
    def reachable(self) -> bool:
        return self.path.reachable


def _require_models(
    solar: SolarPowerModel,
    battery: BatteryModel,
    rover: RoverPowerModel,
) -> None:
    if not isinstance(solar, SolarPowerModel):
        raise TrajectoryInputError(
            "solar must be a SolarPowerModel.",
            code="trajectory_invalid_power_model",
            details={"name": "solar", "type": type(solar).__name__},
        )
    if not isinstance(battery, BatteryModel):
        raise TrajectoryInputError(
            "battery must be a BatteryModel.",
            code="trajectory_invalid_power_model",
            details={"name": "battery", "type": type(battery).__name__},
        )
    if not isinstance(rover, RoverPowerModel):
        raise TrajectoryInputError(
            "rover must be a RoverPowerModel.",
            code="trajectory_invalid_power_model",
            details={"name": "rover", "type": type(rover).__name__},
        )


def _require_compatible(
    problem: StaticProblem,
    occupancy: DynamicOccupancyTimeline,
    sunlight: PiecewiseSunlightTimeline,
) -> None:
    if problem.goal is None:
        raise TrajectoryInputError(
            "SOC planning requires a goal cell.",
            code="trajectory_dynamic_goal_required",
        )
    try:
        occupancy_matches = same_grid(problem.georef, occupancy.georef)
        sunlight_matches = same_grid(problem.georef, sunlight.georef)
    except Exception as exc:
        raise TrajectoryInputError(
            "Unable to compare SOC trajectory grids.",
            code="trajectory_invalid_dynamic_grid",
            details={"error": str(exc)},
        ) from exc
    if not occupancy_matches or not sunlight_matches:
        raise TrajectoryInputError(
            "Static, occupancy, and sunlight grids must match.",
            code="trajectory_dynamic_grid_mismatch",
        )
    if occupancy.boundaries != sunlight.boundaries:
        raise TrajectoryInputError(
            "Occupancy and sunlight must use identical interval boundaries.",
            code="trajectory_soc_timeline_mismatch",
        )


def _energy_tolerance(battery: BatteryModel) -> float:
    return 1.0e-12 * max(1.0, battery.capacity_wh)


def _canonical_hour(timeline: DynamicOccupancyTimeline, value: float) -> float:
    """Round a computed hour through the public datetime representation."""

    represented = timeline.datetime_from_hours(value)
    return timeline.snap_hour(
        timeline.hours_from_start(represented, name="computed_time")
    )


def _wait_energy(
    label: _SocLabel,
    stop_hours: float,
    *,
    occupancy: DynamicOccupancyTimeline,
    sunlight: PiecewiseSunlightTimeline,
    solar: SolarPowerModel,
    battery: BatteryModel,
    rover: RoverPowerModel,
) -> EnergyTimelineResult | None:
    if stop_hours < label.arrival_hours:
        return None
    if stop_hours == label.arrival_hours:
        return EnergyTimelineResult(True, label.energy_wh, label.energy_wh, ())
    allowed, _interval = _span_is_allowed(
        occupancy,
        (label.x, label.y),
        (label.x, label.y),
        label.arrival_hours,
        stop_hours,
    )
    if not allowed:
        return None
    result = integrate_timeline_operation(
        label.energy_wh,
        occupancy.datetime_from_hours(label.arrival_hours),
        occupancy.datetime_from_hours(stop_hours),
        (label.x, label.y),
        "idle",
        timeline=sunlight,
        solar=solar,
        battery=battery,
        rover=rover,
    )
    return result if result.feasible else None


def _dominates(
    left: _SocLabel,
    right: _SocLabel,
    *,
    occupancy: DynamicOccupancyTimeline,
    sunlight: PiecewiseSunlightTimeline,
    solar: SolarPowerModel,
    battery: BatteryModel,
    rover: RoverPowerModel,
) -> bool:
    """Return whether left can reproduce right's state at no greater cost."""

    if left.x != right.x or left.y != right.y:
        return False
    waited = _wait_energy(
        left,
        right.arrival_hours,
        occupancy=occupancy,
        sunlight=sunlight,
        solar=solar,
        battery=battery,
        rover=rover,
    )
    return waited is not None and waited.final_energy_wh >= (
        right.energy_wh - _energy_tolerance(battery)
    )


def _path_from_label(
    labels: list[_SocLabel],
    goal_label: int,
    timeline: DynamicOccupancyTimeline,
) -> ExactDynamicPathResult:
    chain = [goal_label]
    while labels[chain[-1]].predecessor is not None:
        if len(chain) > len(labels):
            raise PlanningError(
                "The SOC predecessor chain is invalid.",
                code="trajectory_invalid_soc_predecessor",
            )
        predecessor = labels[chain[-1]].predecessor
        assert predecessor is not None
        chain.append(predecessor)
    chain.reverse()
    cells = np.asarray(
        [(labels[index].x, labels[index].y) for index in chain],
        dtype=np.int64,
    )
    cells.flags.writeable = False
    arrivals = tuple(
        timeline.datetime_from_hours(labels[index].arrival_hours) for index in chain
    )
    departures = tuple(
        timeline.datetime_from_hours(labels[index].departure_hours)
        for index in chain[1:]
        if labels[index].departure_hours is not None
    )
    if len(departures) != len(chain) - 1:
        raise PlanningError(
            "The SOC predecessor departures are incomplete.",
            code="trajectory_invalid_soc_predecessor",
        )
    return ExactDynamicPathResult(
        True,
        arrivals[-1],
        cells,
        arrivals,
        departures,
    )


def replay_soc_path(
    problem: StaticProblem,
    occupancy: DynamicOccupancyTimeline,
    sunlight: PiecewiseSunlightTimeline,
    departure_time: datetime,
    path: ExactDynamicPathResult,
    *,
    solar: SolarPowerModel,
    battery: BatteryModel,
    rover: RoverPowerModel,
) -> EnergyTimelineResult:
    """Independently replay occupancy, timing, and energy for one SOC path."""

    _require_models(solar, battery, rover)
    _require_compatible(problem, occupancy, sunlight)
    if (
        not path.reachable
        or path.cells is None
        or path.arrival_times is None
        or path.departure_times is None
        or len(path.cells) == 0
        or len(path.arrival_times) != len(path.cells)
        or len(path.departure_times) != len(path.cells) - 1
    ):
        raise PlanningError(
            "SOC replay requires a complete reachable path.",
            code="trajectory_invalid_soc_replay",
        )
    expected_departure = occupancy._time_axis.hours_from_start(
        departure_time, name="departure_time"
    )
    first_arrival = occupancy.hours_from_start(
        path.arrival_times[0], name="arrival_times[0]"
    )
    first_interval = occupancy.interval_index(first_arrival)
    if (
        tuple(int(value) for value in path.cells[0]) != problem.start
        or tuple(int(value) for value in path.cells[-1]) != problem.goal
        or path.arrival_time != path.arrival_times[-1]
        or abs(first_arrival - expected_departure) > _TIME_TOLERANCE_HOURS
        or first_interval is None
        or not occupancy.allowed[
            first_interval,
            problem.start[1],
            problem.start[0],
        ]
    ):
        raise PlanningError(
            "SOC replay path endpoints or initial time are inconsistent.",
            code="trajectory_invalid_soc_replay",
        )

    energy = battery.initial_energy_wh
    segments: list[EnergySegment] = []
    for index, departure_datetime in enumerate(path.departure_times):
        source = tuple(int(value) for value in path.cells[index])
        destination = tuple(int(value) for value in path.cells[index + 1])
        source_arrival = occupancy.hours_from_start(
            path.arrival_times[index], name=f"arrival_times[{index}]"
        )
        departure = occupancy.hours_from_start(
            departure_datetime, name=f"departure_times[{index}]"
        )
        arrival = occupancy.hours_from_start(
            path.arrival_times[index + 1], name=f"arrival_times[{index + 1}]"
        )
        if departure < source_arrival:
            raise PlanningError(
                "SOC replay found a departure before arrival.",
                code="trajectory_invalid_soc_replay",
                details={"leg": index},
            )
        if departure > source_arrival:
            wait_allowed, _ = _span_is_allowed(
                occupancy,
                source,
                source,
                source_arrival,
                departure,
            )
            if not wait_allowed:
                raise PlanningError(
                    "SOC replay found a wait through unavailable occupancy.",
                    code="trajectory_invalid_soc_replay",
                    details={"leg": index},
                )
            wait = integrate_timeline_operation(
                energy,
                path.arrival_times[index],
                departure_datetime,
                source,
                "idle",
                timeline=sunlight,
                solar=solar,
                battery=battery,
                rover=rover,
            )
            if not wait.feasible:
                raise PlanningError(
                    "SOC replay found an infeasible waiting transition.",
                    code="trajectory_soc_replay_infeasible",
                    details={"leg": index, "mode": "idle"},
                )
            segments.extend(wait.segments)
            energy = wait.final_energy_wh

        dx = destination[0] - source[0]
        dy = destination[1] - source[1]
        step = next(
            (item for item in problem.steps if item.dx == dx and item.dy == dy),
            None,
        )
        if step is None:
            raise PlanningError(
                "SOC replay found non-neighboring path cells.",
                code="trajectory_invalid_soc_replay",
                details={"leg": index},
            )
        duration = problem.transition_time(source[0], source[1], step)
        if not np.isfinite(duration) or abs(
            arrival - (departure + duration)
        ) > _TIME_TOLERANCE_HOURS:
            raise PlanningError(
                "SOC replay found an invalid movement duration.",
                code="trajectory_invalid_soc_replay",
                details={"leg": index},
            )
        move_allowed, _ = _span_is_allowed(
            occupancy,
            source,
            destination,
            departure,
            arrival,
        )
        if not move_allowed:
            raise PlanningError(
                "SOC replay found movement through unavailable occupancy.",
                code="trajectory_invalid_soc_replay",
                details={"leg": index},
            )
        move = integrate_timeline_operation(
            energy,
            departure_datetime,
            path.arrival_times[index + 1],
            source,
            "drive",
            timeline=sunlight,
            solar=solar,
            battery=battery,
            rover=rover,
        )
        if not move.feasible:
            raise PlanningError(
                "SOC replay found an infeasible movement transition.",
                code="trajectory_soc_replay_infeasible",
                details={"leg": index, "mode": "drive"},
            )
        segments.extend(move.segments)
        energy = move.final_energy_wh
    return EnergyTimelineResult(
        True,
        battery.initial_energy_wh,
        energy,
        tuple(segments),
    )


def exact_soc_path(
    problem: StaticProblem,
    occupancy: DynamicOccupancyTimeline,
    sunlight: PiecewiseSunlightTimeline,
    departure_time: datetime,
    *,
    solar: SolarPowerModel,
    battery: BatteryModel,
    rover: RoverPowerModel,
    max_labels: int = _DEFAULT_MAX_LABELS,
) -> ExactSocPathResult:
    """Return the exact earliest feasible path under the event-time contract."""

    _require_models(solar, battery, rover)
    _require_compatible(problem, occupancy, sunlight)
    if (
        isinstance(max_labels, (bool, np.bool_))
        or not isinstance(max_labels, Integral)
        or int(max_labels) < 1
    ):
        raise TrajectoryInputError(
            "max_labels must be a positive integer.",
            code="trajectory_invalid_soc_label_limit",
            details={"max_labels": max_labels},
        )
    max_labels = int(max_labels)
    departure = occupancy.snap_hour(
        occupancy.hours_from_start(departure_time, name="departure_time")
    )
    departure_interval = occupancy.interval_index(departure)
    if departure_interval is None:
        raise TrajectoryInputError(
            "departure_time is outside the SOC timeline.",
            code="trajectory_departure_outside_timeline",
        )
    start_x, start_y = problem.start
    if not occupancy.allowed[departure_interval, start_y, start_x]:
        raise TrajectoryInputError(
            "The start cell is dynamically unavailable at departure_time.",
            code="trajectory_dynamic_start_unavailable",
            details={"start": [start_x, start_y]},
        )

    labels = [
        _SocLabel(
            start_x,
            start_y,
            departure,
            battery.initial_energy_wh,
            None,
            None,
        )
    ]
    labels_by_cell: dict[tuple[int, int], list[int]] = {
        (start_x, start_y): [0]
    }
    sequence = count()
    frontier = [(departure, next(sequence), 0)]
    expanded = 0
    candidates_dominated = 0
    labels_retired = 0

    def add_candidate(candidate: _SocLabel) -> int | None:
        nonlocal candidates_dominated, labels_retired
        cell = (candidate.x, candidate.y)
        cell_labels = labels_by_cell.setdefault(cell, [])
        active = [index for index in cell_labels if labels[index].active]
        if any(
            _dominates(
                labels[index],
                candidate,
                occupancy=occupancy,
                sunlight=sunlight,
                solar=solar,
                battery=battery,
                rover=rover,
            )
            for index in active
        ):
            candidates_dominated += 1
            return None
        retired = [
            index
            for index in active
            if _dominates(
                candidate,
                labels[index],
                occupancy=occupancy,
                sunlight=sunlight,
                solar=solar,
                battery=battery,
                rover=rover,
            )
        ]
        if len(labels) >= max_labels:
            raise PlanningError(
                "The exact SOC oracle exceeded its explicit label bound.",
                code="trajectory_exact_soc_label_limit",
                details={"label_count": len(labels), "maximum": max_labels},
            )
        for index in retired:
            labels[index].active = False
        labels_retired += len(retired)
        identifier = len(labels)
        labels.append(candidate)
        cell_labels.append(identifier)
        heappush(
            frontier,
            (candidate.arrival_hours, next(sequence), identifier),
        )
        return identifier

    while frontier:
        _arrival, _order, identifier = heappop(frontier)
        label = labels[identifier]
        if not label.active:
            continue
        expanded += 1
        if (label.x, label.y) == problem.goal:
            path = _path_from_label(labels, identifier, occupancy)
            energy = replay_soc_path(
                problem,
                occupancy,
                sunlight,
                departure_time,
                path,
                solar=solar,
                battery=battery,
                rover=rover,
            )
            if abs(energy.final_energy_wh - label.energy_wh) > _energy_tolerance(
                battery
            ):
                raise PlanningError(
                    "Exact SOC replay disagrees with the search label.",
                    code="trajectory_soc_replay_mismatch",
                    details={
                        "search_energy_wh": label.energy_wh,
                        "replay_energy_wh": energy.final_energy_wh,
                    },
                )
            return ExactSocPathResult(
                path,
                energy,
                ExactSocDiagnostics(
                    len(labels),
                    expanded,
                    candidates_dominated,
                    labels_retired,
                ),
            )

        interval = occupancy.interval_index(label.arrival_hours)
        assert interval is not None
        departures = _continuous_wait_boundaries(
            occupancy,
            (label.x, label.y),
            label.arrival_hours,
            interval,
        )
        for step in problem.steps:
            duration = problem.transition_time(label.x, label.y, step)
            if not np.isfinite(duration):
                continue
            destination = (label.x + step.dx, label.y + step.dy)
            for candidate_departure in departures:
                waited = _wait_energy(
                    label,
                    candidate_departure,
                    occupancy=occupancy,
                    sunlight=sunlight,
                    solar=solar,
                    battery=battery,
                    rover=rover,
                )
                if waited is None:
                    if candidate_departure > label.arrival_hours:
                        break
                    continue
                candidate_arrival = _canonical_hour(
                    occupancy,
                    candidate_departure + duration,
                )
                allowed, _destination_interval = _span_is_allowed(
                    occupancy,
                    (label.x, label.y),
                    destination,
                    candidate_departure,
                    candidate_arrival,
                )
                if not allowed:
                    continue
                movement = integrate_timeline_operation(
                    waited.final_energy_wh,
                    occupancy.datetime_from_hours(candidate_departure),
                    occupancy.datetime_from_hours(candidate_arrival),
                    (label.x, label.y),
                    "drive",
                    timeline=sunlight,
                    solar=solar,
                    battery=battery,
                    rover=rover,
                )
                if not movement.feasible:
                    continue
                add_candidate(
                    _SocLabel(
                        destination[0],
                        destination[1],
                        candidate_arrival,
                        movement.final_energy_wh,
                        identifier,
                        candidate_departure,
                    )
                )

    return ExactSocPathResult(
        ExactDynamicPathResult(False, None, None, None, None),
        None,
        ExactSocDiagnostics(
            len(labels),
            expanded,
            candidates_dominated,
            labels_retired,
        ),
    )
