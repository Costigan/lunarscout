#!/usr/bin/env python3
"""Capture a reproducible real-GPU Phase 4A mechanics report."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lunarscout._numba_horizon.cuda_backend import CudaSession  # noqa: E402
from lunarscout._numba_horizon.kernel_math import (  # noqa: E402
    clamp_subpatch_center,
    evaluate_planar_chord,
    evaluate_quartic,
    evaluate_tangent,
    interpolation_selection,
    interpolate_segments,
    is_valid_elevation,
    sample_bilinear,
)


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _inputs():
    rng = np.random.default_rng(20260716)
    count = 7
    segments = rng.normal(size=(count, 4, 18)).astype(np.float32)
    segments[:, :, 15] = 1.0
    segments[:, :, 16:18] *= np.float32(1e-6)
    elevation = np.arange(36, dtype=np.float32).reshape(6, 6)
    elevation[2, 2] = np.nan
    return dict(
        segments=segments,
        shifts=rng.uniform(-8, 8, size=(count, 4, 2)).astype(np.float32),
        scales=rng.uniform(0.25, 2.0, size=count).astype(np.float32),
        weights=rng.uniform(0, 1, size=(count, 2)).astype(np.float32),
        distances=rng.uniform(0, 5, size=count).astype(np.float32),
        planar_distances=rng.uniform(0, 5000, size=count).astype(np.float32),
        elevation=elevation,
        sample_coordinates=np.array(
            ((0.2, 0.3), (4.7, 4.2), (1.5, 1.5), (-2, 9),
             (3.2, 1.1), (5, 5), (2, 2)), dtype=np.float32,
        ),
        requested_centers=np.array((-4, 0, 4, 8, 12, 16, 99), dtype=np.int32),
        dem_sizes=np.full(count, 16, dtype=np.int32),
        tile_widths=np.full(count, 16, dtype=np.int32),
        subpatch_sizes=np.full(count, 8, dtype=np.int32),
        pixel_coordinates=np.array(
            ((0, 0), (3, 4), (4, 4), (7, 8), (12, 12), (15, 15), (8, 2)),
            dtype=np.int32,
        ),
    )


def _expected(inputs):
    output = np.empty((len(inputs["segments"]), 14), dtype=np.float32)
    for index in range(len(output)):
        segment = interpolate_segments(
            inputs["segments"][index], inputs["shifts"][index],
            inputs["scales"][index], *inputs["weights"][index],
        )
        distance = inputs["distances"][index]
        output[index, 0] = evaluate_quartic(segment[2], segment[4:8], distance)
        output[index, 1] = evaluate_quartic(segment[3], segment[8:12], distance)
        output[index, 2:4] = evaluate_tangent(segment[4:8], segment[8:12], distance)
        output[index, 4] = evaluate_planar_chord(
            segment, inputs["planar_distances"][index]
        )
        output[index, 5] = sample_bilinear(
            inputs["elevation"], *inputs["sample_coordinates"][index]
        )
        output[index, 6] = is_valid_elevation(output[index, 5])
        output[index, 7] = clamp_subpatch_center(
            inputs["requested_centers"][index], inputs["dem_sizes"][index],
            inputs["subpatch_sizes"][index],
        )
        output[index, 8:14] = interpolation_selection(
            *inputs["pixel_coordinates"][index], inputs["tile_widths"][index],
            inputs["subpatch_sizes"][index],
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY / "docs" / "numba-horizon-phase-4a-cuda-mechanics.json",
    )
    arguments = parser.parse_args()
    session = CudaSession()
    mapping = session.index_mapping(35, 37)
    expected_mapping = np.arange(35 * 37, dtype=np.float32).reshape(35, 37)
    inputs = _inputs()
    actual = session.helper_diagnostics(**inputs)
    expected = _expected(inputs)
    maximum_error = float(np.max(np.abs(actual - expected)))
    helpers_accepted = np.allclose(actual, expected, rtol=2e-6, atol=2e-5)
    if not np.array_equal(mapping, expected_mapping) or not helpers_accepted:
        raise RuntimeError("Phase 4A CUDA mechanics differ from CPU oracles")
    from numba import cuda

    report = {
        "schema_version": 1,
        "scope": "CUDA mechanics and arithmetic helpers; no terrain traversal or horizon output",
        "device": {
            "name": session.info.name,
            "compute_capability": list(session.info.compute_capability),
        },
        "packages": {
            name: version(name) for name in (
                "numba", "llvmlite", "numba-cuda", "cuda-toolkit",
                "cuda-bindings", "nvidia-cuda-nvcc-cu12",
            )
        },
        "cuda_module_path": str(Path(cuda.__file__).resolve()),
        "index_mapping": {
            "pixel_count": 35,
            "azimuth_count": 37,
            "layout": "output[pixel, azimuth] = pixel * azimuth_count + azimuth",
            "matches_exactly": True,
            "sha256": _hash(mapping),
        },
        "helper_diagnostics": {
            "case_count": len(actual),
            "field_count": actual.shape[1],
            "maximum_absolute_error": maximum_error,
            "accepted_relative_tolerance": 2e-6,
            "accepted_absolute_tolerance": 2e-5,
            "accepted": True,
            "gpu_sha256": _hash(actual),
            "cpu_sha256": _hash(expected),
            "invalid_elevation_sentinel_present": bool(-32000.0 in actual[:, 5]),
        },
        "qualification": (
            "This proves a real Numba CUDA launch, bounds checks, synchronization, "
            "copies, and selected helper arithmetic. It does not calculate a horizon."
        ),
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
