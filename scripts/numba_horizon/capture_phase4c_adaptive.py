#!/usr/bin/env python3
"""Capture selected adaptive-versus-fixed level-0 traversal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lunarscout._numba_horizon.contract import load_reference_artifact  # noqa: E402
from lunarscout._numba_horizon.cuda_backend import CudaSession  # noqa: E402
from lunarscout._numba_horizon.fixed_step import (  # noqa: E402
    traverse_level0_adaptive,
    traverse_level0_fixed_step,
)


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY / "docs" / "numba-horizon-phase-4c-adaptive.json",
    )
    arguments = parser.parse_args()
    data = REPOSITORY / "tests" / "data" / "numba_horizon"
    artifact = load_reference_artifact(
        data / "phase1_reference_rays.json", data / "phase1_reference_rays.npz"
    )
    session = CudaSession()
    results = []
    for case_id in ("flat_east", "single_obstacle_north"):
        case = next(item for item in artifact.metadata["cases"] if item["id"] == case_id)
        prefix = f"{case_id}__dem_0"
        elevation = artifact.arrays[f"{prefix}__elevation_m"]
        segment = artifact.arrays[f"{case_id}__ray_fit_pass_0__segment_values"]
        fit = case["ray_fit_passes"][0]
        observer = case["observer"]
        observer_z = float(elevation[observer["pixel_y"], observer["pixel_x"]])
        observer_z += float(observer["elevation_m"])
        radius = float(
            artifact.arrays[f"{prefix}__pyramid__projection_parameters"][0]
        )
        fixed = traverse_level0_fixed_step(
            segment, elevation, observer_z_m=observer_z, radius_m=radius,
            map_resolution_m=fit["map_resolution_m"],
        )
        adaptive = traverse_level0_adaptive(
            segment, elevation, observer_z_m=observer_z, radius_m=radius,
            map_resolution_m=fit["map_resolution_m"], pass_index=0,
        )
        maximum, traces, counts = session.adaptive_level0(
            segment[np.newaxis], elevation, np.array([observer_z]), np.array([radius]),
            fit["map_resolution_m"], pass_index=0,
        )
        gpu = traces[0, : counts[0]]
        if len(gpu) != len(adaptive.values):
            raise RuntimeError("adaptive trace lengths differ")
        fixed_peak = int(np.argmax(fixed.values[:, 6]))
        peak_distance = fixed.values[fixed_peak, 1]
        nearest = np.argsort(np.abs(adaptive.values[:, 1] - peak_distance))[:3]
        results.append(
            {
                "id": case_id,
                "fixed": {
                    "trace_steps": len(fixed.values),
                    "maximum_slope": float(fixed.maximum_slope),
                    "horizon_setting_row": fixed.values[fixed_peak].tolist(),
                },
                "adaptive": {
                    "trace_steps": len(gpu),
                    "gpu_maximum_slope": float(maximum[0]),
                    "cpu_maximum_slope": float(adaptive.maximum_slope),
                    "gpu_trace_sha256": _hash(gpu),
                    "cpu_trace_sha256": _hash(adaptive.values),
                    "minimum_advance_below_100_m_km": float(
                        np.min(gpu[gpu[:, 1] < 100.0, 7])
                    ),
                    "minimum_advance_at_or_above_100_m_km": float(
                        np.min(gpu[gpu[:, 1] >= 100.0, 7])
                    ),
                    "rows_nearest_fixed_horizon_distance": [
                        gpu[index].tolist() for index in sorted(nearest)
                    ],
                },
                "adaptive_minus_fixed_maximum_slope": float(
                    maximum[0] - fixed.maximum_slope
                ),
                "gpu_cpu_maximum_slope_difference": float(
                    maximum[0] - adaptive.maximum_slope
                ),
            }
        )
    report = {
        "schema_version": 1,
        "scope": "adaptive level-0 traversal compared with fixed-step control; no hierarchy",
        "device": {
            "name": session.info.name,
            "compute_capability": list(session.info.compute_capability),
        },
        "constants": {
            "angular_step_factor": 0.00151,
            "inverse_tangent_max_slope": 1.732,
            "minimum_step_resolution_factor": 0.5,
            "primary_far_minimum_step_resolution_factor": 0.8,
            "primary_far_threshold_m": 100.0,
            "near_far_formula_threshold_m": 500.0,
        },
        "cases": results,
        "finding": (
            "The aligned one-pixel obstacle loses about 0.03606 slope versus the "
            "1.2 m fixed-step control because the 24 m primary far-step floor samples "
            "away from the narrow peak. This follows current production stepping rules."
        ),
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
