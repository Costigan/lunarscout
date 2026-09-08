from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import numpy as np

from ..errors import ConfigurationSpaceError, TrajectoryInputError
from ._dynamic_reference import DynamicOccupancyTimeline
from ._time_contract import IntervalTimeAxis
from .providers import ConfigurationSpaceProvider, _provider_result


def occupancy_timeline_from_provider(
    provider: ConfigurationSpaceProvider,
    boundaries: Iterable[datetime],
) -> DynamicOccupancyTimeline:
    """Materialize a small exact-oracle timeline from a public provider."""

    georef = getattr(provider, "georef", None)
    if georef is None:
        raise TrajectoryInputError(
            "Configuration provider must expose a GeoReference.",
            code="trajectory_invalid_provider_grid",
        )
    axis = IntervalTimeAxis(tuple(boundaries))
    window = (0, 0, georef.width, georef.height)
    layers = []
    for time in axis.boundaries[:-1]:
        try:
            layer = _provider_result(
                provider,
                np.dtype(np.bool_),
                window,
                time,
                signal="configuration_space",
            )
        except (TrajectoryInputError, ConfigurationSpaceError):
            raise
        layers.append(np.array(layer, dtype=np.bool_, copy=True))
    return DynamicOccupancyTimeline(
        axis.boundaries,
        np.stack(layers),
        georef,
    )
