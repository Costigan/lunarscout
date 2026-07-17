from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lunarscout._numba_horizon.cuda_backend import CudaSession
from lunarscout._numba_horizon.contract import load_reference_artifact
from lunarscout._numba_horizon.direct_reference import direct_reference_trace
from lunarscout._numba_horizon.fixed_step import (
    traverse_level0_adaptive,
    traverse_level0_fixed_step,
)
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.kernel_math import (
    clamp_subpatch_center,
    evaluate_planar_chord,
    evaluate_quartic,
    evaluate_tangent,
    interpolation_selection,
    interpolate_segments,
    is_valid_elevation,
    sample_bilinear,
)


def _diagnostic_inputs():
    rng = np.random.default_rng(20260716)
    count = 7
    segments = rng.normal(size=(count, 4, 18)).astype(np.float32)
    segments[:, :, 15] = 1.0
    segments[:, :, 16:18] *= np.float32(1e-6)
    shifts = rng.uniform(-8, 8, size=(count, 4, 2)).astype(np.float32)
    scales = rng.uniform(0.25, 2.0, size=count).astype(np.float32)
    weights = rng.uniform(0, 1, size=(count, 2)).astype(np.float32)
    distances = rng.uniform(0, 5, size=count).astype(np.float32)
    planar = rng.uniform(0, 5000, size=count).astype(np.float32)
    elevation = np.arange(36, dtype=np.float32).reshape(6, 6)
    elevation[2, 2] = np.nan
    sample_coordinates = np.array(
        ((0.2, 0.3), (4.7, 4.2), (1.5, 1.5), (-2, 9), (3.2, 1.1), (5, 5), (2, 2)),
        dtype=np.float32,
    )
    return dict(
        segments=segments,
        shifts=shifts,
        scales=scales,
        weights=weights,
        distances=distances,
        planar_distances=planar,
        elevation=elevation,
        sample_coordinates=sample_coordinates,
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


def test_cpu_kernel_helpers_cover_interpolation_validity_and_sentinels() -> None:
    inputs = _diagnostic_inputs()
    expected = _expected(inputs)
    assert expected.shape == (7, 14)
    assert -32000.0 in expected[:, 5]
    assert set(expected[:, 6]) == {0.0, 1.0}
    assert np.all(np.isfinite(expected[:, :5]))


def _fixed_step_case(case_id: str):
    data = Path(__file__).resolve().parents[1] / "data" / "numba_horizon"
    artifact = load_reference_artifact(
        data / "phase1_reference_rays.json", data / "phase1_reference_rays.npz"
    )
    case = next(item for item in artifact.metadata["cases"] if item["id"] == case_id)
    prefix = f"{case_id}__dem_0"
    elevation = artifact.arrays[f"{prefix}__elevation_m"]
    segment = artifact.arrays[f"{case_id}__ray_fit_pass_0__segment_values"]
    fit = case["ray_fit_passes"][0]
    observer = case["observer"]
    observer_z = float(elevation[observer["pixel_y"], observer["pixel_x"]])
    observer_z += float(observer["elevation_m"])
    radius = float(artifact.arrays[f"{prefix}__pyramid__projection_parameters"][0])
    return case, elevation, segment, fit, observer_z, radius


def test_fixed_step_cpu_oracle_exposes_near_and_far_behavior() -> None:
    flat_case, elevation, segment, fit, observer_z, radius = _fixed_step_case("flat_east")
    trace = traverse_level0_fixed_step(
        segment, elevation, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=fit["map_resolution_m"],
    )
    assert len(trace.values) == 832
    near = trace.values[:, 0] < 0.5
    assert np.all(trace.values[near, 5] == 0.0)
    assert np.all(trace.values[~near, 5] < 0.0)
    # The production kernel's documented near-field flat approximation is zero;
    # the independent C# reference uses spherical geometry at every distance.
    assert trace.maximum_slope == 0.0
    assert flat_case["result"]["maximum_slope"] < 0.0

    obstacle_case, elevation, segment, fit, observer_z, radius = _fixed_step_case(
        "single_obstacle_north"
    )
    obstacle_trace = traverse_level0_fixed_step(
        segment, elevation, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=fit["map_resolution_m"],
    )
    assert obstacle_trace.maximum_slope > 0.25
    assert obstacle_case["result"]["maximum_slope"] > 0.24


def test_adaptive_cpu_oracle_exposes_step_floors_and_skipped_terrain() -> None:
    _, elevation, segment, fit, observer_z, radius = _fixed_step_case("flat_east")
    adaptive = traverse_level0_adaptive(
        segment, elevation, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=fit["map_resolution_m"], pass_index=0,
    )
    assert np.any(adaptive.values[:, 1] < 100.0)
    assert np.any(adaptive.values[:, 1] >= 100.0)
    before_far_floor = adaptive.values[adaptive.values[:, 1] < 100.0, 7]
    after_far_floor = adaptive.values[adaptive.values[:, 1] >= 100.0, 7]
    assert np.min(before_far_floor) == pytest.approx(0.015, abs=1e-7)
    assert np.min(after_far_floor) == pytest.approx(0.024, abs=1e-7)
    assert np.any(adaptive.values[:, 0] < 0.5)
    assert np.any(adaptive.values[:, 0] >= 0.5)

    _, elevation, segment, fit, observer_z, radius = _fixed_step_case(
        "single_obstacle_north"
    )
    fixed = traverse_level0_fixed_step(
        segment, elevation, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=fit["map_resolution_m"],
    )
    adaptive = traverse_level0_adaptive(
        segment, elevation, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=fit["map_resolution_m"], pass_index=0,
    )
    assert len(adaptive.values) < len(fixed.values)
    assert adaptive.maximum_slope <= fixed.maximum_slope
    missed_peak = float(fixed.maximum_slope - adaptive.maximum_slope)
    assert 0.03 < missed_peak < 0.04


def test_direct_reference_geometry_matches_csharp_trace_sample_by_sample() -> None:
    data = Path(__file__).resolve().parents[1] / "data" / "numba_horizon"
    artifact = load_reference_artifact(
        data / "phase1_reference_rays.json", data / "phase1_reference_rays.npz"
    )
    for case_id in ("flat_east", "single_obstacle_north"):
        case = next(item for item in artifact.metadata["cases"] if item["id"] == case_id)
        prefix = f"{case_id}__dem_0"
        dem = DemGrid(
            artifact.arrays[f"{prefix}__elevation_m"],
            artifact.arrays[f"{prefix}__geo_transform"],
            ProjectionParameters.from_array(
                artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
            ),
        )
        ray_prefix = f"{case_id}__ray_fit_pass_0"
        expected_prefix = f"{case_id}__pass_0"
        expected = np.column_stack(
            (
                artifact.arrays[f"{expected_prefix}__trace_distance_m"],
                artifact.arrays[f"{expected_prefix}__trace_pixel_x"],
                artifact.arrays[f"{expected_prefix}__trace_pixel_y"],
                artifact.arrays[f"{expected_prefix}__trace_elevation_m"],
                artifact.arrays[f"{expected_prefix}__trace_slope"],
            )
        )
        actual = direct_reference_trace(
            dem,
            artifact.arrays[f"{ray_prefix}__observer_vector_moon_centered_m"],
            artifact.arrays[f"{ray_prefix}__nominal_direction_moon_centered"],
            expected[:, 0],
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_real_cuda_mapping_and_helpers_match_cpu_oracles() -> None:
    session = CudaSession()
    mapping = session.index_mapping(35, 37)
    expected_mapping = np.arange(35 * 37, dtype=np.float32).reshape(35, 37)
    np.testing.assert_array_equal(mapping, expected_mapping)

    inputs = _diagnostic_inputs()
    actual = session.helper_diagnostics(**inputs)
    expected = _expected(inputs)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
    assert session.info.compute_capability == (12, 0)

    for case_id in ("flat_east", "single_obstacle_north"):
        _, elevation, segment, fit, observer_z, radius = _fixed_step_case(case_id)
        cpu = traverse_level0_fixed_step(
            segment, elevation, observer_z_m=observer_z, radius_m=radius,
            map_resolution_m=fit["map_resolution_m"],
        )
        maximum, traces, counts = session.fixed_step_level0(
            segment[np.newaxis, :], elevation, np.array([observer_z]),
            np.array([radius]), fit["map_resolution_m"],
        )
        assert counts[0] == len(cpu.values)
        gpu_trace = traces[0, : counts[0]]
        np.testing.assert_allclose(gpu_trace[:, 0], cpu.values[:, 0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(gpu_trace[:, 1], cpu.values[:, 1], rtol=1e-5, atol=1e-2)
        np.testing.assert_allclose(gpu_trace[:, 2:4], cpu.values[:, 2:4], rtol=3e-6, atol=3e-4)
        np.testing.assert_allclose(gpu_trace[:, 4], cpu.values[:, 4], rtol=4e-3, atol=5e-3)
        np.testing.assert_allclose(gpu_trace[:, 5:7], cpu.values[:, 5:7], rtol=1e-5, atol=2e-5)
        np.testing.assert_allclose(maximum[0], cpu.maximum_slope, rtol=2e-6, atol=2e-5)

        adaptive_cpu = traverse_level0_adaptive(
            segment, elevation, observer_z_m=observer_z, radius_m=radius,
            map_resolution_m=fit["map_resolution_m"], pass_index=0,
        )
        adaptive_maximum, adaptive_traces, adaptive_counts = session.adaptive_level0(
            segment[np.newaxis, :], elevation, np.array([observer_z]),
            np.array([radius]), fit["map_resolution_m"], pass_index=0,
        )
        assert adaptive_counts[0] == len(adaptive_cpu.values)
        gpu_adaptive = adaptive_traces[0, : adaptive_counts[0]]
        np.testing.assert_allclose(
            gpu_adaptive[:, 0], adaptive_cpu.values[:, 0], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            gpu_adaptive[:, 1], adaptive_cpu.values[:, 1], rtol=1e-5, atol=1e-2
        )
        np.testing.assert_allclose(
            gpu_adaptive[:, 2:4], adaptive_cpu.values[:, 2:4], rtol=3e-6, atol=3e-4
        )
        np.testing.assert_allclose(
            gpu_adaptive[:, 4], adaptive_cpu.values[:, 4], rtol=4e-3, atol=5e-3
        )
        np.testing.assert_allclose(
            gpu_adaptive[:, 5:8], adaptive_cpu.values[:, 5:8], rtol=1e-5, atol=2e-5
        )
        np.testing.assert_allclose(
            adaptive_maximum[0], adaptive_cpu.maximum_slope, rtol=1e-5, atol=2e-5
        )
