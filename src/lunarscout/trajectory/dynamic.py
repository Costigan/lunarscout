from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from ..alignment import same_grid
from ..errors import TrajectoryInputError
from ..georeference import GeoReference
from ..spice_geometry import LonLat
from ._dispatch import resolve_dynamic_dispatch
from ._dynamic_gridrunner import gridrunner_dynamic_path_from_provider
from ._time_contract import as_utc
from ._validation import prepare_static_problem
from .providers import ConfigurationSpaceProvider
from .static import Cell, StaticTravelModel


DynamicAlgorithm: TypeAlias = Literal["gridrunner"]
DynamicBackend: TypeAlias = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True, eq=False)
class DynamicPathResult:
    reachable: bool
    arrival_time: datetime | None
    cells: NDArray[np.int64] | None
    arrival_times: tuple[datetime, ...] | None
    departure_times: tuple[datetime, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.reachable, (bool, np.bool_)):
            raise TrajectoryInputError(
                "DynamicPathResult reachable must be Boolean.",
                code="trajectory_invalid_dynamic_result",
            )
        if not bool(self.reachable):
            if any(
                value is not None
                for value in (
                    self.arrival_time,
                    self.cells,
                    self.arrival_times,
                    self.departure_times,
                )
            ):
                raise TrajectoryInputError(
                    "An unreachable dynamic result cannot contain trajectory data.",
                    code="trajectory_invalid_dynamic_result",
                )
            object.__setattr__(self, "reachable", False)
            return
        if any(
            value is None
            for value in (
                self.arrival_time,
                self.cells,
                self.arrival_times,
                self.departure_times,
            )
        ):
            raise TrajectoryInputError(
                "A reachable dynamic result requires complete trajectory data.",
                code="trajectory_invalid_dynamic_result",
            )
        assert self.arrival_time is not None
        assert self.cells is not None
        assert self.arrival_times is not None
        assert self.departure_times is not None
        cells_input = np.asarray(self.cells)
        if (
            cells_input.ndim != 2
            or cells_input.shape[0] < 1
            or cells_input.shape[1] != 2
            or not np.issubdtype(cells_input.dtype, np.integer)
            or np.issubdtype(cells_input.dtype, np.bool_)
        ):
            raise TrajectoryInputError(
                "Dynamic result cells must be a non-empty integer (N, 2) array.",
                code="trajectory_invalid_dynamic_result",
            )
        int64 = np.iinfo(np.int64)
        if np.any(cells_input < int64.min) or np.any(cells_input > int64.max):
            raise TrajectoryInputError(
                "Dynamic result cells must fit in signed 64-bit integers.",
                code="trajectory_invalid_dynamic_result",
            )
        cells = np.array(cells_input, dtype=np.int64, copy=True, order="C")
        arrivals = tuple(
            as_utc(value, name=f"arrival_times[{index}]")
            for index, value in enumerate(self.arrival_times)
        )
        departures = tuple(
            as_utc(value, name=f"departure_times[{index}]")
            for index, value in enumerate(self.departure_times)
        )
        final_arrival = as_utc(self.arrival_time, name="arrival_time")
        if len(arrivals) != len(cells) or len(departures) != len(cells) - 1:
            raise TrajectoryInputError(
                "Dynamic result time and cell counts are inconsistent.",
                code="trajectory_invalid_dynamic_result",
            )
        if final_arrival != arrivals[-1]:
            raise TrajectoryInputError(
                "Dynamic result arrival_time must equal its final cell arrival.",
                code="trajectory_invalid_dynamic_result",
            )
        for index, departure in enumerate(departures):
            if departure < arrivals[index] or arrivals[index + 1] <= departure:
                raise TrajectoryInputError(
                    "Each dynamic leg must depart after cell arrival and arrive "
                    "strictly after departure.",
                    code="trajectory_invalid_dynamic_result",
                    details={"leg": index},
                )
        cells.flags.writeable = False
        object.__setattr__(self, "reachable", True)
        object.__setattr__(self, "arrival_time", final_arrival)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "arrival_times", arrivals)
        object.__setattr__(self, "departure_times", departures)

    @property
    def travel_time_hours(self) -> float | None:
        if not self.reachable:
            return None
        assert self.arrival_times is not None and self.arrival_time is not None
        return (self.arrival_time - self.arrival_times[0]).total_seconds() / 3600.0

    @property
    def wait_intervals(self) -> tuple[tuple[datetime, datetime], ...]:
        if not self.reachable:
            return ()
        assert self.arrival_times is not None and self.departure_times is not None
        return tuple(
            (arrival, departure)
            for arrival, departure in zip(self.arrival_times, self.departure_times)
            if departure > arrival
        )


def dynamic_path(
    traversable: npt.ArrayLike,
    georef: GeoReference,
    start: Cell | LonLat,
    goal: Cell | LonLat,
    boundaries: Iterable[datetime],
    configuration: ConfigurationSpaceProvider,
    departure_time: datetime,
    *,
    valid: npt.ArrayLike | None = None,
    elevation: npt.ArrayLike | None = None,
    model: StaticTravelModel | None = None,
    algorithm: DynamicAlgorithm | str = "gridrunner",
    backend: DynamicBackend | str = "auto",
) -> DynamicPathResult:
    """Return an exact earliest-arrival path under dynamic occupancy."""

    resolve_dynamic_dispatch(algorithm, backend)
    problem = prepare_static_problem(
        traversable,
        georef,
        start,
        goal=goal,
        valid=valid,
        elevation=elevation,
        model=model,
    )
    if not isinstance(configuration, ConfigurationSpaceProvider):
        raise TrajectoryInputError(
            "configuration must conform to ConfigurationSpaceProvider.",
            code="trajectory_invalid_configuration_provider",
        )
    try:
        grid_matches = same_grid(georef, configuration.georef)
    except Exception as exc:
        raise TrajectoryInputError(
            "Unable to compare the trajectory and configuration grids.",
            code="trajectory_invalid_provider_grid",
            details={"error": str(exc)},
        ) from exc
    if not grid_matches:
        raise TrajectoryInputError(
            "The configuration-provider grid must match the trajectory grid.",
            code="trajectory_provider_grid_mismatch",
        )
    result = gridrunner_dynamic_path_from_provider(
        problem,
        configuration,
        boundaries,
        departure_time,
        block_size=8,
    ).path
    if not result.reachable:
        return DynamicPathResult(False, None, None, None, None)
    return DynamicPathResult(
        True,
        result.arrival_time,
        result.cells,
        result.arrival_times,
        result.departure_times,
    )
