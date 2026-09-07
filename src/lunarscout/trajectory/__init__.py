"""Public trajectory-planning API.

Importing this namespace performs no raster I/O and initializes neither SPICE
nor CUDA. Optimized and dynamic implementations are added only after their
scientific contracts are frozen.
"""

from .static import (
    PathResult,
    SlipFunction,
    StaticTravelModel,
    TravelTimeResult,
    static_path,
    static_travel_time,
)
from ..errors import (
    ConfigurationSpaceError,
    NoPathError,
    PlanningError,
    TrajectoryError,
    TrajectoryInputError,
)

__all__ = [
    "ConfigurationSpaceError",
    "NoPathError",
    "PathResult",
    "PlanningError",
    "SlipFunction",
    "StaticTravelModel",
    "TrajectoryError",
    "TrajectoryInputError",
    "TravelTimeResult",
    "static_path",
    "static_travel_time",
]
