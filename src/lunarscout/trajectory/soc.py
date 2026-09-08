from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from ..alignment import same_grid
from ..errors import PlanningError, TrajectoryInputError
from ..georeference import GeoReference
from ..spice_geometry import LonLat
from ._dispatch import resolve_soc_dispatch
from ._environment import (
    occupancy_timeline_from_provider,
    sunlight_timeline_from_provider,
)
from ._power_greedy import greedy_soc_path
from ._power_reference import exact_soc_path
from ._time_contract import IntervalTimeAxis
from ._validation import prepare_static_problem
from .dynamic import DynamicPathResult
from .power import (
    BatteryModel,
    EnergyTimelineResult,
    RoverPowerModel,
    SolarPowerModel,
)
from .providers import ConfigurationSpaceProvider, SunlightProvider
from .static import Cell, StaticTravelModel


SocAlgorithm: TypeAlias = Literal["exact", "greedy"]
SocBackend: TypeAlias = Literal["auto", "cpu", "cuda"]
_MAX_SOC_TIMELINE_STATES = 5_000_000
_DEFAULT_MAX_LABELS = 100_000


def _energy_matches_path(
    path: DynamicPathResult,
    energy: EnergyTimelineResult,
) -> bool:
    assert path.cells is not None
    assert path.arrival_times is not None
    assert path.departure_times is not None
    segment_index = 0

    def consume(
        mode: str,
        start_time: datetime,
        stop_time: datetime,
        cell: tuple[int, int],
    ) -> bool:
        nonlocal segment_index
        cursor = start_time
        while cursor < stop_time:
            if segment_index >= len(energy.segments):
                return False
            segment = energy.segments[segment_index]
            if (
                segment.mode != mode
                or segment.cell != cell
                or segment.start_time != cursor
                or segment.stop_time > stop_time
            ):
                return False
            cursor = segment.stop_time
            segment_index += 1
        return cursor == stop_time

    for index, departure in enumerate(path.departure_times):
        source = tuple(int(value) for value in path.cells[index])
        arrival = path.arrival_times[index]
        if departure > arrival and not consume("idle", arrival, departure, source):
            return False
        if not consume(
            "drive",
            departure,
            path.arrival_times[index + 1],
            source,
        ):
            return False
    return segment_index == len(energy.segments)


@dataclass(frozen=True, slots=True, eq=False)
class SocPathResult:
    """A dynamic path plus replayed battery energy and algorithm identity."""

    path: DynamicPathResult
    energy: EnergyTimelineResult | None
    algorithm: SocAlgorithm
    backend: Literal["cpu"]

    def __post_init__(self) -> None:
        if not isinstance(self.path, DynamicPathResult):
            raise TrajectoryInputError(
                "SocPathResult path must be a DynamicPathResult.",
                code="trajectory_invalid_soc_result",
            )
        if self.algorithm not in ("exact", "greedy") or self.backend != "cpu":
            raise TrajectoryInputError(
                "SocPathResult algorithm or backend is invalid.",
                code="trajectory_invalid_soc_result",
            )
        if not self.path.reachable:
            if self.energy is not None:
                raise TrajectoryInputError(
                    "An unreachable SocPathResult cannot contain energy data.",
                    code="trajectory_invalid_soc_result",
                )
            return
        if (
            not isinstance(self.energy, EnergyTimelineResult)
            or not self.energy.feasible
        ):
            raise TrajectoryInputError(
                "A reachable SocPathResult requires feasible energy data.",
                code="trajectory_invalid_soc_result",
            )
        assert self.path.arrival_times is not None
        assert self.path.arrival_time is not None
        if not _energy_matches_path(self.path, self.energy):
            raise TrajectoryInputError(
                "SOC path and energy segments must describe the same events.",
                code="trajectory_invalid_soc_result",
            )

    @property
    def reachable(self) -> bool:
        return self.path.reachable

    @property
    def arrival_time(self) -> datetime | None:
        return self.path.arrival_time

    @property
    def cells(self) -> NDArray[np.int64] | None:
        return self.path.cells

    @property
    def arrival_times(self) -> tuple[datetime, ...] | None:
        return self.path.arrival_times

    @property
    def departure_times(self) -> tuple[datetime, ...] | None:
        return self.path.departure_times

    @property
    def travel_time_hours(self) -> float | None:
        return self.path.travel_time_hours

    @property
    def wait_intervals(self) -> tuple[tuple[datetime, datetime], ...]:
        return self.path.wait_intervals

    @property
    def final_energy_wh(self) -> float | None:
        return None if self.energy is None else self.energy.final_energy_wh

    @property
    def complete(self) -> bool:
        """Whether an unreachable result proves no feasible event-time path."""

        return self.algorithm == "exact"

    @property
    def optimal(self) -> bool:
        """Whether a reachable result proves minimum event-time arrival."""

        return self.algorithm == "exact"


def _validate_max_labels(value: int) -> int:
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


def _require_provider_grid(
    georef: GeoReference,
    provider: object,
    *,
    name: str,
) -> None:
    other = getattr(provider, "georef", None)
    try:
        matches = isinstance(other, GeoReference) and same_grid(georef, other)
    except Exception as exc:
        raise TrajectoryInputError(
            f"Unable to compare the trajectory and {name} grids.",
            code="trajectory_invalid_provider_grid",
            details={"name": name, "error": str(exc)},
        ) from exc
    if not matches:
        raise TrajectoryInputError(
            f"The {name} grid must match the trajectory grid.",
            code="trajectory_provider_grid_mismatch",
            details={"name": name},
        )


def soc_path(
    traversable: npt.ArrayLike,
    georef: GeoReference,
    start: Cell | LonLat,
    goal: Cell | LonLat,
    boundaries: Iterable[datetime],
    configuration: ConfigurationSpaceProvider,
    sunlight: SunlightProvider,
    departure_time: datetime,
    *,
    solar: SolarPowerModel,
    battery: BatteryModel,
    rover: RoverPowerModel,
    valid: npt.ArrayLike | None = None,
    elevation: npt.ArrayLike | None = None,
    model: StaticTravelModel | None = None,
    algorithm: SocAlgorithm | str = "greedy",
    backend: SocBackend | str = "auto",
    max_labels: int = _DEFAULT_MAX_LABELS,
) -> SocPathResult:
    """Plan a battery-feasible event-time path with a selected SOC algorithm."""

    dispatch = resolve_soc_dispatch(algorithm, backend)
    maximum_labels = _validate_max_labels(max_labels)
    problem = prepare_static_problem(
        traversable,
        georef,
        start,
        goal=goal,
        valid=valid,
        elevation=elevation,
        model=model,
    )
    if not isinstance(solar, SolarPowerModel):
        raise TrajectoryInputError(
            "solar must be a SolarPowerModel.",
            code="trajectory_invalid_power_model",
        )
    if not isinstance(battery, BatteryModel):
        raise TrajectoryInputError(
            "battery must be a BatteryModel.",
            code="trajectory_invalid_power_model",
        )
    if not isinstance(rover, RoverPowerModel):
        raise TrajectoryInputError(
            "rover must be a RoverPowerModel.",
            code="trajectory_invalid_power_model",
        )
    if not isinstance(configuration, ConfigurationSpaceProvider):
        raise TrajectoryInputError(
            "configuration must conform to ConfigurationSpaceProvider.",
            code="trajectory_invalid_configuration_provider",
        )
    if not isinstance(sunlight, SunlightProvider):
        raise TrajectoryInputError(
            "sunlight must conform to SunlightProvider.",
            code="trajectory_invalid_sunlight_provider",
        )
    _require_provider_grid(georef, configuration, name="configuration provider")
    _require_provider_grid(georef, sunlight, name="sunlight provider")
    try:
        boundary_values = tuple(boundaries)
    except TypeError as exc:
        raise TrajectoryInputError(
            "boundaries must be an iterable of timezone-aware datetimes.",
            code="trajectory_invalid_dynamic_time",
        ) from exc
    axis = IntervalTimeAxis(boundary_values)
    state_count = axis.interval_count * georef.height * georef.width
    if state_count > _MAX_SOC_TIMELINE_STATES:
        raise PlanningError(
            "The SOC environment timeline exceeds its configured bound.",
            code="trajectory_soc_timeline_state_limit",
            details={
                "state_count": state_count,
                "maximum": _MAX_SOC_TIMELINE_STATES,
            },
        )
    occupancy = occupancy_timeline_from_provider(
        configuration,
        axis.boundaries,
        read_block_size=128,
    )
    power = sunlight_timeline_from_provider(
        sunlight,
        axis.boundaries,
        read_block_size=128,
    )
    arguments = {
        "solar": solar,
        "battery": battery,
        "rover": rover,
        "max_labels": maximum_labels,
    }
    if dispatch.algorithm == "exact":
        result = exact_soc_path(
            problem,
            occupancy,
            power,
            departure_time,
            **arguments,
        )
    else:
        result = greedy_soc_path(
            problem,
            occupancy,
            power,
            departure_time,
            **arguments,
        )
    if not result.reachable:
        return SocPathResult(
            DynamicPathResult(False, None, None, None, None),
            None,
            dispatch.algorithm,
            dispatch.backend,
        )
    path = result.path
    return SocPathResult(
        DynamicPathResult(
            True,
            path.arrival_time,
            path.cells,
            path.arrival_times,
            path.departure_times,
        ),
        result.energy,
        dispatch.algorithm,
        dispatch.backend,
    )
