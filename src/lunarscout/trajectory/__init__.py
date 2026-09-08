"""Public trajectory-planning API.

Importing this namespace performs no raster I/O and initializes neither SPICE
nor CUDA.
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
from .dynamic import DynamicPathResult, dynamic_path
from .power import (
    BatteryModel,
    EnergySegment,
    EnergyTimelineResult,
    RoverPowerModel,
    SolarPowerModel,
)
from .soc import SocPathResult, soc_path
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
    "BatteryModel",
    "ConfigurationSpaceError",
    "ConfigurationSpaceProvider",
    "DynamicPathResult",
    "EnergySegment",
    "EnergyTimelineResult",
    "EarthElevationProvider",
    "EarthElevationThresholdProvider",
    "ExplicitSunVectorProvider",
    "HorizonSunlightProvider",
    "NoPathError",
    "PathResult",
    "PlanningError",
    "RoverPowerModel",
    "SlipFunction",
    "SolarPowerModel",
    "SocPathResult",
    "SpiceSunVectorProvider",
    "StaticConfigurationSpaceProvider",
    "StaticTravelModel",
    "SunVectorProvider",
    "SunlightProvider",
    "SunlightThresholdProvider",
    "TrajectoryError",
    "TrajectoryInputError",
    "TravelTimeResult",
    "dynamic_path",
    "soc_path",
    "static_path",
    "static_travel_time",
]
