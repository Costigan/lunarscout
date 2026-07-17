#!/usr/bin/env python3
"""Capture selected real-GPU fixed-step level-0 traversal traces."""

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
from lunarscout._numba_horizon.direct_reference import direct_reference_trace  # noqa: E402
from lunarscout._numba_horizon.fixed_step import traverse_level0_fixed_step  # noqa: E402
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters  # noqa: E402


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY / "docs" / "numba-horizon-phase-4b-fixed-step.json",
    )
    arguments = parser.parse_args()
    data = REPOSITORY / "tests" / "data" / "numba_horizon"
    artifact = load_reference_artifact(
        data / "phase1_reference_rays.json", data / "phase1_reference_rays.npz"
    )
    session = CudaSession()
    cases = []
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
        dem = DemGrid(
            elevation,
            artifact.arrays[f"{prefix}__geo_transform"],
            ProjectionParameters.from_array(
                artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
            ),
        )
        cpu = traverse_level0_fixed_step(
            segment, elevation, observer_z_m=observer_z, radius_m=radius,
            map_resolution_m=fit["map_resolution_m"],
        )
        maximum, traces, counts = session.fixed_step_level0(
            segment[np.newaxis], elevation, np.array([observer_z]), np.array([radius]),
            fit["map_resolution_m"],
        )
        gpu = traces[0, : counts[0]]
        if len(gpu) != len(cpu.values):
            raise RuntimeError("CPU and GPU trace lengths differ")
        difference = np.abs(gpu.astype(np.float64) - cpu.values.astype(np.float64))
        ray_prefix = f"{case_id}__ray_fit_pass_0"
        direct = direct_reference_trace(
            dem,
            artifact.arrays[f"{ray_prefix}__observer_vector_moon_centered_m"],
            artifact.arrays[f"{ray_prefix}__nominal_direction_moon_centered"],
            gpu[:, 0].astype(np.float64) * 1000.0,
        )
        direct_pixel_error = np.hypot(
            gpu[: len(direct), 2] - direct[:, 1],
            gpu[: len(direct), 3] - direct[:, 2],
        )
        if not (
            np.allclose(gpu[:, 0], cpu.values[:, 0], rtol=1e-5, atol=1e-5)
            and np.allclose(gpu[:, 1], cpu.values[:, 1], rtol=1e-5, atol=1e-2)
            and np.allclose(gpu[:, 2:4], cpu.values[:, 2:4], rtol=3e-6, atol=3e-4)
            and np.allclose(gpu[:, 4], cpu.values[:, 4], rtol=4e-3, atol=5e-3)
            and np.allclose(gpu[:, 5:7], cpu.values[:, 5:7], rtol=1e-5, atol=2e-5)
        ):
            raise RuntimeError(f"fixed-step trace mismatch for {case_id}")
        transition = int(np.searchsorted(gpu[:, 0], np.float32(0.5)))
        horizon_index = int(np.argmax(gpu[:, 6]))
        selected_indices = sorted(set((0, transition - 1, transition, horizon_index, len(gpu) - 1)))
        cases.append(
            {
                "id": case_id,
                "trace_steps": len(gpu),
                "gpu_trace_sha256": _hash(gpu),
                "cpu_trace_sha256": _hash(cpu.values),
                "gpu_maximum_slope": float(maximum[0]),
                "cpu_maximum_slope": float(cpu.maximum_slope),
                "csharp_reference_emulator_maximum_slope": case["result"]["maximum_slope"],
                "gpu_minus_csharp_reference_maximum_slope": (
                    float(maximum[0]) - case["result"]["maximum_slope"]
                ),
                "direct_reference_at_gpu_distances": {
                    "maximum_fitted_path_error_pixels": float(direct_pixel_error.max()),
                    "maximum_elevation_difference_m": float(
                        np.max(np.abs(gpu[: len(direct), 4] - direct[:, 3]))
                    ),
                    "maximum_production_vs_exact_slope_difference": float(
                        np.max(np.abs(gpu[: len(direct), 5] - direct[:, 4]))
                    ),
                    "qualification": (
                        "Position/elevation compare the fitted production ray with "
                        "direct vector geometry. Slope includes the intentional "
                        "production near-field approximation difference."
                    ),
                },
                "maximum_absolute_difference_by_field": {
                    name: float(difference[:, index].max())
                    for index, name in enumerate(
                        ("parameter_distance_km", "true_distance_m", "pixel_x", "pixel_y",
                         "elevation_m", "sample_slope", "running_maximum_slope")
                    )
                },
                "selected_trace_rows": [
                    {
                        "index": index,
                        "parameter_distance_km": float(gpu[index, 0]),
                        "true_distance_m": float(gpu[index, 1]),
                        "pixel_x": float(gpu[index, 2]),
                        "pixel_y": float(gpu[index, 3]),
                        "elevation_m": float(gpu[index, 4]),
                        "sample_slope": float(gpu[index, 5]),
                        "running_maximum_slope": float(gpu[index, 6]),
                    }
                    for index in selected_indices if 0 <= index < len(gpu)
                ],
            }
        )
    report = {
        "schema_version": 1,
        "scope": "fixed 1.2 m step, fitted-ray, level-0 traversal; no adaptive stepping or hierarchy",
        "device": {
            "name": session.info.name,
            "compute_capability": list(session.info.compute_capability),
        },
        "cases": cases,
        "reference_qualification": (
            "The Phase 1 C# ReferenceRayEmulator uses three azimuth offsets and exact "
            "spherical slopes. GPU trace rows are compared sample-by-sample with an "
            "independent CPU implementation of current production fixed-step arithmetic; "
            "C# reference maxima are reported separately, not asserted equal."
        ),
        "known_difference": (
            "Below 500 m the current production kernel uses a flat-earth slope. Flat "
            "terrain therefore produces a zero running maximum, whereas the independent "
            "C# reference reports a small negative curvature slope."
        ),
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
