from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from numbers import Integral

import numpy as np

from ..errors import ConfigurationSpaceError, TrajectoryInputError
from ..georeference import GeoReference
from ._dynamic_reference import DynamicOccupancyTimeline
from ._power_accounting import PiecewiseSunlightTimeline
from ._time_contract import IntervalTimeAxis
from .providers import (
    ConfigurationSpaceProvider,
    SunlightProvider,
    _provider_result,
)


def occupancy_timeline_from_provider(
    provider: ConfigurationSpaceProvider,
    boundaries: Iterable[datetime],
    *,
    read_block_size: int = 128,
) -> DynamicOccupancyTimeline:
    """Materialize a small exact-oracle timeline from a public provider."""

    georef = getattr(provider, "georef", None)
    if not isinstance(georef, GeoReference):
        raise TrajectoryInputError(
            "Configuration provider must expose a GeoReference.",
            code="trajectory_invalid_provider_grid",
        )
    if (
        isinstance(read_block_size, bool)
        or not isinstance(read_block_size, Integral)
        or int(read_block_size) < 1
    ):
        raise TrajectoryInputError(
            "read_block_size must be a positive integer.",
            code="trajectory_invalid_block_size",
        )
    read_block_size = int(read_block_size)
    try:
        boundary_values = tuple(boundaries)
    except TypeError as exc:
        raise TrajectoryInputError(
            "boundaries must be an iterable of timezone-aware datetimes.",
            code="trajectory_invalid_dynamic_time",
        ) from exc
    axis = IntervalTimeAxis(boundary_values)
    layers = []
    for time in axis.boundaries[:-1]:
        layer = np.empty((georef.height, georef.width), dtype=np.bool_)
        for y0 in range(0, georef.height, read_block_size):
            height = min(read_block_size, georef.height - y0)
            for x0 in range(0, georef.width, read_block_size):
                width = min(read_block_size, georef.width - x0)
                try:
                    values = _provider_result(
                        provider,
                        np.dtype(np.bool_),
                        (x0, y0, width, height),
                        time,
                        signal="configuration_space",
                    )
                except (TrajectoryInputError, ConfigurationSpaceError):
                    raise
                layer[y0 : y0 + height, x0 : x0 + width] = values
        layers.append(layer)
    return DynamicOccupancyTimeline(
        axis.boundaries,
        np.stack(layers),
        georef,
    )


def sunlight_timeline_from_provider(
    provider: SunlightProvider,
    boundaries: Iterable[datetime],
    *,
    read_block_size: int = 128,
) -> PiecewiseSunlightTimeline:
    """Materialize byte-valued sunlight as a fractional power timeline."""

    georef = getattr(provider, "georef", None)
    if not isinstance(georef, GeoReference):
        raise TrajectoryInputError(
            "Sunlight provider must expose a GeoReference.",
            code="trajectory_invalid_provider_grid",
        )
    if (
        isinstance(read_block_size, bool)
        or not isinstance(read_block_size, Integral)
        or int(read_block_size) < 1
    ):
        raise TrajectoryInputError(
            "read_block_size must be a positive integer.",
            code="trajectory_invalid_block_size",
        )
    read_block_size = int(read_block_size)
    try:
        boundary_values = tuple(boundaries)
    except TypeError as exc:
        raise TrajectoryInputError(
            "boundaries must be an iterable of timezone-aware datetimes.",
            code="trajectory_invalid_dynamic_time",
        ) from exc
    axis = IntervalTimeAxis(boundary_values)
    layers = []
    for time in axis.boundaries[:-1]:
        layer = np.empty((georef.height, georef.width), dtype=np.float64)
        for y0 in range(0, georef.height, read_block_size):
            height = min(read_block_size, georef.height - y0)
            for x0 in range(0, georef.width, read_block_size):
                width = min(read_block_size, georef.width - x0)
                values = _provider_result(
                    provider,
                    np.dtype(np.uint8),
                    (x0, y0, width, height),
                    time,
                    signal="sunlight",
                )
                layer[y0 : y0 + height, x0 : x0 + width] = (
                    values.astype(np.float64) / 255.0
                )
        layers.append(layer)
    return PiecewiseSunlightTimeline(
        axis.boundaries,
        np.stack(layers),
        georef,
    )
