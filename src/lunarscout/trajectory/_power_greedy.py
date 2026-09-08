from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from heapq import heappop, heappush
from itertools import count
from numbers import Integral

import numpy as np

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
from ._power_reference import (
    _SocLabel,
    _canonical_hour,
    _energy_tolerance,
    _path_from_label,
    _require_compatible,
    _require_models,
    _wait_energy,
    replay_soc_path,
)
from ._validation import StaticProblem
from .power import (
    BatteryModel,
    EnergyTimelineResult,
    RoverPowerModel,
    SolarPowerModel,
)


_MAX_GREEDY_STATES = 5_000_000
_DEFAULT_MAX_LABELS = 100_000
_TIME_TIE_TOLERANCE_HOURS = 1.0e-9


@dataclass(frozen=True, slots=True)
class GreedySocDiagnostics:
    state_count: int
    labels_created: int
    labels_expanded: int
    candidates_discarded: int
    labels_replaced: int


@dataclass(frozen=True, slots=True, eq=False)
class GreedySocPathResult:
    path: ExactDynamicPathResult
    energy: EnergyTimelineResult | None
    diagnostics: GreedySocDiagnostics

    @property
    def reachable(self) -> bool:
        return self.path.reachable


def _label_limit(value: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) < 1
    ):
        raise TrajectoryInputError(
            "max_labels must be a positive integer.",
            code="trajectory_invalid_soc_label_limit",
            details={"max_labels": value},
        )
    return int(value)


def _state_capacity(occupancy: DynamicOccupancyTimeline) -> int:
    state_count = (
        occupancy.interval_count * occupancy.georef.height * occupancy.georef.width
    )
    if state_count > _MAX_GREEDY_STATES:
        raise PlanningError(
            "The greedy SOC state table exceeds its configured bound.",
            code="trajectory_greedy_soc_state_limit",
            details={"state_count": state_count, "maximum": _MAX_GREEDY_STATES},
        )
    return state_count


def _preferred(
    candidate: _SocLabel,
    incumbent: _SocLabel,
    battery: BatteryModel,
) -> bool:
    if candidate.arrival_hours < (
        incumbent.arrival_hours - _TIME_TIE_TOLERANCE_HOURS
    ):
        return True
    return abs(candidate.arrival_hours - incumbent.arrival_hours) <= (
        _TIME_TIE_TOLERANCE_HOURS
    ) and candidate.energy_wh > (
        incumbent.energy_wh + _energy_tolerance(battery)
    )


def greedy_soc_path(
    problem: StaticProblem,
    occupancy: DynamicOccupancyTimeline,
    sunlight: PiecewiseSunlightTimeline,
    departure_time: datetime,
    *,
    solar: SolarPowerModel,
    battery: BatteryModel,
    rover: RoverPowerModel,
    max_labels: int = _DEFAULT_MAX_LABELS,
) -> GreedySocPathResult:
    """Return the greedy earliest-label SOC path for a materialized timeline."""

    _require_models(solar, battery, rover)
    _require_compatible(problem, occupancy, sunlight)
    maximum_labels = _label_limit(max_labels)
    state_count = _state_capacity(occupancy)
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
    start_state = (departure_interval, start_y, start_x)
    selected: dict[tuple[int, int, int], int] = {start_state: 0}
    sequence = count()
    frontier = [(departure, next(sequence), start_state, 0)]
    expanded = 0
    discarded = 0
    replaced = 0

    def add_candidate(
        candidate: _SocLabel,
        destination_interval: int,
    ) -> None:
        nonlocal discarded, replaced
        state = (destination_interval, candidate.y, candidate.x)
        incumbent_identifier = selected.get(state)
        if incumbent_identifier is not None and not _preferred(
            candidate,
            labels[incumbent_identifier],
            battery,
        ):
            discarded += 1
            return
        if len(labels) >= maximum_labels:
            raise PlanningError(
                "The greedy SOC planner exceeded its explicit label bound.",
                code="trajectory_greedy_soc_label_limit",
                details={"label_count": len(labels), "maximum": maximum_labels},
            )
        if incumbent_identifier is not None:
            replaced += 1
        identifier = len(labels)
        labels.append(candidate)
        selected[state] = identifier
        heappush(
            frontier,
            (candidate.arrival_hours, next(sequence), state, identifier),
        )

    while frontier:
        _arrival, _order, state, identifier = heappop(frontier)
        if selected.get(state) != identifier:
            continue
        label = labels[identifier]
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
                    "Greedy SOC replay disagrees with the selected label.",
                    code="trajectory_soc_replay_mismatch",
                    details={
                        "search_energy_wh": label.energy_wh,
                        "replay_energy_wh": energy.final_energy_wh,
                    },
                )
            return GreedySocPathResult(
                path,
                energy,
                GreedySocDiagnostics(
                    state_count,
                    len(labels),
                    expanded,
                    discarded,
                    replaced,
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
                allowed, destination_interval = _span_is_allowed(
                    occupancy,
                    (label.x, label.y),
                    destination,
                    candidate_departure,
                    candidate_arrival,
                )
                if not allowed or destination_interval is None:
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
                    ),
                    destination_interval,
                )

    return GreedySocPathResult(
        ExactDynamicPathResult(False, None, None, None, None),
        None,
        GreedySocDiagnostics(
            state_count,
            len(labels),
            expanded,
            discarded,
            replaced,
        ),
    )
