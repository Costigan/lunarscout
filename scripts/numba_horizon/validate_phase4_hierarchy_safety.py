#!/usr/bin/env python3
"""Measure hierarchy approximation differences from dense bilinear sampling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lunarscout._numba_horizon.contract import PyramidArrays  # noqa: E402
from lunarscout._numba_horizon.fixed_step import traverse_level0_fixed_step  # noqa: E402
from lunarscout._numba_horizon.geometry import (  # noqa: E402
    DemGrid,
    ProjectionParameters,
)
from lunarscout._numba_horizon.hierarchy import (  # noqa: E402
    _bilinear_bound,
    traversal_counters,
    traverse_hierarchy,
)
from lunarscout._numba_horizon.pyramid import build_max_pyramid  # noqa: E402


MAP_RESOLUTION_M = 30.0
RADIUS_M = 1_737_400.0
SCIENTIFIC_ANGULAR_COMPARISON_DEGREES = 0.005


def _straight_segment(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    end_km: float,
) -> np.ndarray:
    magnitude = math.hypot(direction_x, direction_y)
    direction_x /= magnitude
    direction_y /= magnitude
    values = np.zeros(18, dtype=np.float32)
    values[0:4] = (origin_x, origin_y, origin_x, origin_y)
    values[4] = direction_x * 1000.0 / MAP_RESOLUTION_M
    values[8] = direction_y * 1000.0 / MAP_RESOLUTION_M
    values[12:15] = (0.001, end_km, 0.001)
    values[15] = 1.0
    return values


def _pyramid(elevation: np.ndarray) -> PyramidArrays:
    transform = np.array(
        (0.0, MAP_RESOLUTION_M, 0.0, 0.0, 0.0, MAP_RESOLUTION_M),
        dtype=np.float64,
    )
    projection = ProjectionParameters(RADIUS_M, 0.0, 0.0, 1.0, 0.0, 0.0)
    return build_max_pyramid(DemGrid(elevation, transform, projection))


def _angle(slope: np.float32) -> float:
    return math.degrees(math.atan(float(slope)))


def _run_case(
    case_id: str,
    elevation: np.ndarray,
    segment: np.ndarray,
    observer_x: int,
    observer_y: int,
) -> dict:
    pyramid = _pyramid(elevation)
    observer_z = float(elevation[observer_y, observer_x])
    dense = traverse_level0_fixed_step(
        segment,
        pyramid.level0,
        observer_z_m=observer_z,
        radius_m=RADIUS_M,
        map_resolution_m=MAP_RESOLUTION_M,
    )
    hierarchy = traverse_hierarchy(
        segment,
        pyramid,
        observer_z_m=observer_z,
        radius_m=RADIUS_M,
        map_resolution_m=MAP_RESOLUTION_M,
        pass_index=0,
    )
    dense_degrees = _angle(dense.maximum_slope)
    hierarchy_degrees = _angle(hierarchy.maximum_slope)
    underestimate = dense_degrees - hierarchy_degrees
    counters = traversal_counters(hierarchy.values)
    return {
        "id": case_id,
        "shape_y_x": list(elevation.shape),
        "dense_fixed_step_degrees": dense_degrees,
        "hierarchy_degrees": hierarchy_degrees,
        "dense_minus_hierarchy_degrees": underestimate,
        "within_scientific_comparison_threshold": (
            underestimate <= SCIENTIFIC_ANGULAR_COMPARISON_DEGREES
        ),
        "hierarchy_trace_rows": len(hierarchy.values),
        "hierarchy_maximum_level": int(np.max(hierarchy.values[:, 2])),
        "counters": {
            "iterations": counters.iterations,
            "level0_samples": counters.level0_samples,
            "culled_blocks": counters.culled_blocks,
            "out_of_bounds": counters.out_of_bounds,
            "nodata_skips": counters.nodata_skips,
        },
    }


def run_matrix() -> dict:
    cases: list[dict] = []
    directions = (
        ("east", 1.0, 0.0),
        ("west", -1.0, 0.0),
        ("south", 0.0, 1.0),
        ("north", 0.0, -1.0),
        ("southeast", 1.0, 1.0),
        ("southwest", -1.0, 1.0),
        ("northeast", 1.0, -1.0),
        ("northwest", -1.0, -1.0),
    )
    size = 129
    center = size // 2
    for name, dx, dy in directions:
        elevation = np.zeros((size, size), dtype=np.float32)
        distance_pixels = 20
        obstacle_x = int(round(center + dx * distance_pixels))
        obstacle_y = int(round(center + dy * distance_pixels))
        elevation[obstacle_y, obstacle_x] = 150.0
        segment = _straight_segment(
            center + 0.25, center + 0.25, dx, dy, 1.7
        )
        cases.append(_run_case(
            f"level0_boundary_{name}", elevation, segment, center, center
        ))

    # An odd-sized long ray crosses enough cells to activate factor-four mips.
    long_size = 2051
    long_center = long_size // 2
    for name, dx, dy in (("east", 1.0, 0.0), ("north", 0.0, -1.0)):
        elevation = np.zeros((long_size, long_size), dtype=np.float32)
        obstacle_x = int(round(long_center + dx * 1000))
        obstacle_y = int(round(long_center + dy * 1000))
        elevation[obstacle_y, obstacle_x] = 5000.0
        segment = _straight_segment(
            long_center + 0.25, long_center + 0.25, dx, dy, 30.6
        )
        cases.append(_run_case(
            f"coarse_mip_boundary_{name}",
            elevation,
            segment,
            long_center,
            long_center,
        ))

    edge_values = np.array(
        ((1.0, 2.0, 3.0), (4.0, np.nan, -32000.0), (7.0, 8.0, 9.0)),
        dtype=np.float32,
    )
    edge_pyramid = PyramidArrays(
        edge_values,
        np.empty(0, dtype=np.float32),
        np.array(((0, 0, 3, 3),), dtype=np.int32),
        np.zeros(11, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
    )
    edge_checks = {
        "top_left_with_nan": float(_bilinear_bound(edge_pyramid, 0, 0, 0)),
        "right_edge": float(_bilinear_bound(edge_pyramid, 0, 2, 1)),
        "bottom_right_corner": float(_bilinear_bound(edge_pyramid, 0, 2, 2)),
    }
    maximum_underestimate = max(
        case["dense_minus_hierarchy_degrees"] for case in cases
    )
    return {
        "schema_version": 1,
        "scope": "hierarchy approximation versus 1.2 m dense bilinear level-0 traversal",
        "diagnostic_comparison": {
            "angular_threshold_degrees": (
                SCIENTIFIC_ANGULAR_COMPARISON_DEGREES
            ),
            "all_cases_within_threshold": all(
                case["within_scientific_comparison_threshold"] for case in cases
            ),
            "is_acceptance_gate": False,
        },
        "configuration": {
            "map_resolution_m": MAP_RESOLUTION_M,
            "moon_radius_m": RADIUS_M,
            "direction_count": len(directions),
            "case_count": len(cases),
        },
        "cases": cases,
        "edge_and_invalid_neighbor_bounds_m": edge_checks,
        "maximum_dense_minus_hierarchy_degrees": maximum_underestimate,
        "qualification": (
            "The dense bilinearly interpolated surface is not treated as terrain "
            "ground truth. Differences in this deliberately adversarial synthetic "
            "matrix characterize the hierarchy approximation but do not gate the "
            "port. C#/Numba parity and downstream illumination error are the gates."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "docs" / "numba-horizon-phase-4-hierarchy-safety.json",
    )
    arguments = parser.parse_args()
    report = run_matrix()
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
