from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_trajectory_namespace_import_is_lazy_and_curated(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    program = """
from pathlib import Path
import sys
import socket
import urllib.request
import rasterio
def forbidden(*_args, **_kwargs):
    raise AssertionError('I/O attempted during trajectory import')
rasterio.open = forbidden
socket.create_connection = forbidden
urllib.request.urlopen = forbidden
before = tuple(Path.cwd().iterdir())
import lunarscout as ls
import lunarscout.trajectory as trajectory
after = tuple(Path.cwd().iterdir())
assert before == after
assert ls.trajectory is trajectory
assert not {'numba', 'numba.cuda', 'spiceypy'} & sys.modules.keys()
assert trajectory.__all__ == [
    'AllOfConfigurationSpaceProvider', 'ArrayEarthElevationProvider',
    'ArraySunlightProvider', 'ConfigurationSpaceError',
    'ConfigurationSpaceProvider', 'DynamicPathResult', 'EarthElevationProvider',
    'EarthElevationThresholdProvider', 'ExplicitSunVectorProvider',
    'HorizonSunlightProvider', 'NoPathError', 'PathResult', 'PlanningError',
    'SlipFunction',
    'SpiceSunVectorProvider', 'StaticConfigurationSpaceProvider',
    'StaticTravelModel', 'SunVectorProvider', 'SunlightProvider',
    'SunlightThresholdProvider', 'TrajectoryError', 'TrajectoryInputError',
    'TravelTimeResult', 'dynamic_path', 'static_path', 'static_travel_time',
]
assert not hasattr(trajectory, 'soc_path')
from datetime import datetime, timedelta, timezone
import numpy as np
from pyproj import CRS
crs = CRS.from_user_input('ESRI:103878')
grid = ls.GeoReference(
    crs.to_wkt(), crs.to_proj4(), (0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    2, 1, 10.0, -10.0, None,
)
field = trajectory.static_travel_time(np.ones((1, 2), dtype=bool), grid, (0, 0))
path = trajectory.static_path(np.ones((1, 2), dtype=bool), grid, (0, 0), (1, 0))
assert field.reached.tolist() == [[True, True]]
assert path.reachable and path.path.tolist() == [[0, 0], [1, 0]]
time0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
configuration = trajectory.StaticConfigurationSpaceProvider(
    np.ones((1, 2), dtype=bool), grid,
)
dynamic = trajectory.dynamic_path(
    np.ones((1, 2), dtype=bool), grid, (0, 0), (1, 0),
    (time0, time0 + timedelta(hours=1)), configuration, time0,
)
assert dynamic.reachable
assert not {'numba', 'numba.cuda'} & sys.modules.keys()
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_private_dynamic_oracle_needs_no_spice_or_numba(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    program = """
from datetime import datetime, timedelta, timezone
import sys
import numpy as np
from pyproj import CRS
import lunarscout as ls
from lunarscout.trajectory._dynamic_reference import (
    DynamicOccupancyTimeline, exact_dynamic_path,
)
from lunarscout.trajectory._validation import prepare_static_problem
crs = CRS.from_user_input('ESRI:103878')
grid = ls.GeoReference(
    crs.to_wkt(), crs.to_proj4(), (0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    2, 1, 10.0, -10.0, None,
)
problem = prepare_static_problem(
    np.ones((1, 2), dtype=bool), grid, (0, 0), goal=(1, 0),
)
t0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
timeline = DynamicOccupancyTimeline(
    (t0, t0 + timedelta(hours=1)), np.ones((1, 1, 2), dtype=bool), grid,
)
result = exact_dynamic_path(problem, timeline, t0)
assert result.reachable
assert not {'numba', 'numba.cuda', 'spiceypy'} & sys.modules.keys()
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_explicit_providers_need_no_spiceypy(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    program = """
from datetime import datetime, timedelta, timezone
import sys
import numpy as np
from pyproj import CRS
import lunarscout as ls
assert 'spiceypy' not in sys.modules
t0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
vectors = ls.trajectory.ExplicitSunVectorProvider(
    (t0,), np.asarray([[1.0, 0.0, 0.0]]),
)
assert vectors.vectors((t0,)).tolist() == [[1.0, 0.0, 0.0]]
crs = CRS.from_user_input('ESRI:103878')
grid = ls.GeoReference(
    crs.to_wkt(), crs.to_proj4(), (0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    1, 1, 10.0, -10.0, None,
)
sun = ls.trajectory.ArraySunlightProvider(
    (t0, t0 + timedelta(hours=1)), np.ones((1, 1, 1), dtype=np.uint8), grid,
)
assert sun.read(0, 0, 1, 1, t0).item() == 1
assert 'spiceypy' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_explicit_dynamic_mobility_needs_no_spice_or_numba(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    program = """
from datetime import datetime, timedelta, timezone
import sys
import numpy as np
from pyproj import CRS
import lunarscout as ls
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout.trajectory._dynamic_mobility import (
    SunDirectionFunction, compile_dynamic_travel_model,
)
from lunarscout.trajectory._dynamic_reference import DynamicOccupancyTimeline
from lunarscout.trajectory._validation import prepare_static_problem
t0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
crs = CRS.from_user_input('ESRI:103878')
grid = ls.GeoReference(
    crs.to_wkt(), crs.to_proj4(), (0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    2, 1, 10.0, -10.0, None,
)
problem = prepare_static_problem(
    np.ones((1, 2), dtype=bool), grid, (0, 0), goal=(1, 0),
)
timeline = DynamicOccupancyTimeline(
    (t0, t0 + timedelta(hours=1)), np.ones((1, 1, 2), dtype=bool), grid,
)
dem = DemGrid(
    np.zeros((1, 2), dtype=np.float32),
    np.asarray(grid.affine_transform, dtype=np.float64),
    ProjectionParameters(1737400.0, -np.pi / 2, 0.0, 1.0, 0.0, 0.0),
)
vectors = ls.trajectory.ExplicitSunVectorProvider(
    (t0,), np.asarray([[1.0e9, 0.0, 0.0]]),
)
compile_dynamic_travel_model(
    problem, timeline, dem=dem, dem_georef=grid, sun_vectors=vectors,
    sun_direction=SunDirectionFunction((-1.0, 1.0), (2.0, 0.5)),
)
assert not {'numba', 'numba.cuda', 'spiceypy'} & sys.modules.keys()
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
