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
from .providers import (
    AllOfConfigurationSpaceProvider,
    ArrayEarthElevationProvider,
    ArraySunlightProvider,
    ConfigurationSpaceProvider,
    EarthElevationProvider,
    EarthElevationThresholdProvider,
    ExplicitSunVectorProvider,
    HorizonSunlightProvider,
    SpiceSunVectorProvider,
    StaticConfigurationSpaceProvider,
    SunVectorProvider,
    SunlightProvider,
    SunlightThresholdProvider,
)
from ..errors import (
    ConfigurationSpaceError,
    NoPathError,
    PlanningError,
    TrajectoryError,
    TrajectoryInputError,
)

__all__ = [
    "AllOfConfigurationSpaceProvider",
    "ArrayEarthElevationProvider",
    "ArraySunlightProvider",
    "ConfigurationSpaceError",
    "ConfigurationSpaceProvider",
    "EarthElevationProvider",
    "EarthElevationThresholdProvider",
    "ExplicitSunVectorProvider",
    "HorizonSunlightProvider",
    "NoPathError",
    "PathResult",
    "PlanningError",
    "SlipFunction",
    "SpiceSunVectorProvider",
    "StaticConfigurationSpaceProvider",
    "StaticTravelModel",
    "SunVectorProvider",
    "SunlightProvider",
    "SunlightThresholdProvider",
    "TrajectoryError",
    "TrajectoryInputError",
    "TravelTimeResult",
    "static_path",
    "static_travel_time",
]
