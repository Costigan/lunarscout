#!/usr/bin/env python3
"""Compare exact GridRunner and safe-interval CPU trajectory planners."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import lunarscout as ls  # noqa: E402
from lunarscout.trajectory._dynamic_gridrunner import (  # noqa: E402
    gridrunner_dynamic_path,
)
from lunarscout.trajectory._dynamic_reference import (  # noqa: E402
    DynamicOccupancyTimeline,
    exact_dynamic_path,
)
from lunarscout.trajectory._safe_interval import (  # noqa: E402
    safe_interval_dynamic_path,
)
from lunarscout.trajectory._validation import prepare_static_problem  # noqa: E402


T0 = datetime(2040, 1, 1, tzinfo=timezone.utc)
_WKT = (
    'PROJCS["ESRI:103878",GEOGCS["Moon_2000",DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400,0]],PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["Meter",1]]'
)
_PROJ4 = (
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+R=1737400 +units=m +no_defs"
)


def _georef(width: int, height: int) -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4=_PROJ4,
        affine_transform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
        width=width,
        height=height,
        pixel_size_x=10.0,
        pixel_size_y=-10.0,
        nodata=None,
    )


def _occupancy(
    pattern: str,
    intervals: int,
    height: int,
    width: int,
    seed: int,
) -> np.ndarray:
    allowed = np.ones((intervals, height, width), dtype=np.bool_)
    if pattern == "sparse":
        first = intervals // 3
        last = 2 * intervals // 3
        allowed[first:last, height // 3 : 2 * height // 3, 1:-1] = False
    elif pattern == "frequent":
        generator = np.random.default_rng(seed)
        changing = generator.random((height, width)) < 0.55
        for interval in range(intervals):
            phase = (interval + np.indices((height, width)).sum(axis=0)) % 3
            allowed[interval, changing & (phase == 0)] = False
    elif pattern == "wait_gate":
        allowed[: intervals // 3, :, width // 2 :] = False
    elif pattern != "none":
        raise ValueError(f"Unknown occupancy pattern: {pattern}")

    # Preserve a known feasible top/right corridor except in the dedicated
    # gate case, which deliberately requires a wait before crossing.
    if pattern != "wait_gate":
        allowed[:, 0, :] = True
        allowed[:, :, width - 1] = True
    allowed[0, 0, 0] = True
    allowed[:, height - 1, width - 1] = True
    return allowed


def _measure(function, repeats: int):
    durations = []
    result = None
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - started)
    assert result is not None
    gc.collect()
    tracemalloc.start()
    function()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {
        "median_seconds": statistics.median(durations),
        "minimum_seconds": min(durations),
        "peak_tracemalloc_bytes": peak,
    }


def _hours(result) -> float | None:
    if not result.reachable:
        return None
    assert result.arrival_time is not None
    return (result.arrival_time - T0).total_seconds() / 3600.0


def _run_case(spec: dict[str, object], repeats: int) -> dict[str, object]:
    width = int(spec["width"])
    height = int(spec["height"])
    intervals = int(spec["intervals"])
    spacing = float(spec["spacing_hours"])
    allowed = _occupancy(
        str(spec["pattern"]), intervals, height, width, int(spec["seed"])
    )
    georef = _georef(width, height)
    boundaries = tuple(
        T0 + timedelta(hours=index * spacing) for index in range(intervals + 1)
    )
    timeline = DynamicOccupancyTimeline(boundaries, allowed, georef)
    problem = prepare_static_problem(
        np.any(allowed, axis=0),
        georef,
        (0, 0),
        goal=(width - 1, height - 1),
        model=ls.trajectory.StaticTravelModel(
            speed_m_per_h=40.0,
            include_diagonals=False,
        ),
    )
    exact = exact_dynamic_path(problem, timeline, T0)
    gridrunner, grid_timing = _measure(
        lambda: gridrunner_dynamic_path(problem, timeline, T0, block_size=8),
        repeats,
    )
    safe, safe_timing = _measure(
        lambda: safe_interval_dynamic_path(problem, timeline, T0), repeats
    )
    exact_hours = _hours(exact)
    grid_hours = _hours(gridrunner.path)
    safe_hours = _hours(safe.path)
    agreement = (
        exact.reachable == gridrunner.path.reachable == safe.path.reachable
        and exact_hours == grid_hours == safe_hours
    )
    return {
        **spec,
        "raw_cell_interval_states": int(allowed.size),
        "configuration_changes": int(np.count_nonzero(allowed[1:] != allowed[:-1])),
        "exact": {"reachable": exact.reachable, "travel_time_hours": exact_hours},
        "gridrunner": {
            **grid_timing,
            "travel_time_hours": grid_hours,
            "expanded_states": gridrunner.diagnostics.expanded_states,
            "state_relaxations": gridrunner.diagnostics.state_relaxations,
            "block_activations": gridrunner.diagnostics.block_activations,
            "block_reactivations": gridrunner.diagnostics.block_reactivations,
        },
        "safe_interval": {
            **safe_timing,
            "travel_time_hours": safe_hours,
            "expanded_states": safe.diagnostics.expanded_states,
            "state_relaxations": safe.diagnostics.state_relaxations,
            "safe_intervals": safe.diagnostics.safe_intervals,
        },
        "exact_agreement": agreement,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    cases = (
        dict(name="short_no_wait", width=10, height=10, intervals=12,
             spacing_hours=1.0, pattern="none", seed=1),
        dict(name="long_sparse", width=10, height=10, intervals=48,
             spacing_hours=0.5, pattern="sparse", seed=2),
        dict(name="long_frequent", width=10, height=10, intervals=48,
             spacing_hours=0.5, pattern="frequent", seed=3),
        dict(name="larger_sparse", width=20, height=20, intervals=24,
             spacing_hours=0.5, pattern="sparse", seed=4),
        dict(name="required_wait", width=12, height=1, intervals=24,
             spacing_hours=0.25, pattern="wait_gate", seed=5),
    )
    report = {
        "schema": "lunarscout-trajectory-dynamic-benchmark-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "in-memory planner execution; provider materialization excluded",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "repeats": arguments.repeats,
        "cases": [_run_case(spec, arguments.repeats) for spec in cases],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if all(case["exact_agreement"] for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
