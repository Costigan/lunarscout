from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lunarscout._numba_horizon.contract import (
    ContractConfiguration,
    PyramidArrays,
    SegmentTensor,
    load_reference_artifact,
)
from lunarscout._numba_horizon.cuda_backend import CudaSession
from lunarscout._numba_horizon.geometry import (
    DemGrid,
    GridConvergenceInput,
    ProjectionParameters,
    build_subpatch_segments,
)
from lunarscout._numba_horizon.generator import generate_patch_horizons
from lunarscout._numba_horizon.hierarchy import traverse_hierarchy
from lunarscout._numba_horizon.kernel_math import sample_bilinear
from lunarscout._numba_horizon.pyramid import build_max_pyramid
from lunarscout._numba_horizon.subpatch import interpolate_pixel_segment


DATA = Path(__file__).resolve().parents[1] / "data" / "numba_horizon"


def _production_patch_inputs():
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    base = "single_pixel_multi_dem_production__horizon_buffer_fixture"
    dems = []
    pyramids = []
    for index in range(2):
        prefix = f"{base}__dem_{index}"
        dems.append(
            DemGrid(
                artifact.arrays[f"{prefix}__elevation_m"],
                artifact.arrays[f"{prefix}__geo_transform"],
                ProjectionParameters.from_array(
                    artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
                ),
            )
        )
        pyramids.append(PyramidArrays.from_artifact(artifact.arrays, f"{prefix}__pyramid"))
    convergence = GridConvergenceInput(
        *map(float, artifact.arrays[f"{base}__grid_convergence"])
    )
    segments, _, _ = build_subpatch_segments(
        dems,
        tile_column=20,
        tile_row=20,
        tile_width=1,
        azimuth_count=1440,
        maximum_distance_m=1_000_000,
        observer_elevation_m=0,
        subpatch_size=8,
        grid_convergence=convergence,
    )
    return artifact, base, segments, pyramids


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_cuda_full_subpatch_multi_dem_passes_match_csharp_buffers() -> None:
    artifact, base, segments, pyramids = _production_patch_inputs()
    corrected = json.loads(
        (DATA / "phase4d_production_segments.json").read_text(encoding="utf-8")
    )["production_hierarchy_case"]
    expected_passes = np.asarray(
        corrected["per_dem_slopes"], dtype=np.float32
    ).reshape(2, 1, 1440)
    session = CudaSession()
    pass_zero = session.subpatch_hierarchical_pass(
        segments, pyramids[0], pyramids[0], tile_column=20, tile_row=20,
        tile_width=1, tile_height=1, subpatch_size=8, pass_index=0,
    )
    pass_one = session.subpatch_hierarchical_pass(
        segments, pyramids[0], pyramids[1], tile_column=20, tile_row=20,
        tile_width=1, tile_height=1, subpatch_size=8, pass_index=1,
    )
    np.testing.assert_allclose(pass_zero, expected_passes[0], rtol=0, atol=3e-7)
    np.testing.assert_allclose(pass_one, expected_passes[1], rtol=0, atol=3e-7)

    expected_final = np.asarray(
        corrected["final_slopes"], dtype=np.float32
    ).reshape(1, 1440)
    np.testing.assert_array_equal(
        np.maximum(expected_passes[0], expected_passes[1]), expected_final
    )
    configuration = ContractConfiguration(
        tile_width=1, tile_height=1, azimuth_count=1440, subpatch_size=8,
        dem_count=2, primary_width=41, primary_height=41,
    )
    tensor = SegmentTensor(
        segments,
        np.broadcast_to(
            np.arange(2, dtype=np.int32), segments.shape[:-1]
        ).copy(),
        configuration,
    )
    generated = generate_patch_horizons(
        session, tensor, pyramids, tile_column=20, tile_row=20
    )
    np.testing.assert_allclose(generated.slopes, expected_final, rtol=0, atol=3e-7)
    degrees = generated.degrees()
    np.testing.assert_allclose(
        degrees,
        np.asarray(corrected["final_degrees"], dtype=np.float32).reshape(1, 1440),
        rtol=0,
        atol=1e-5,
    )


def test_generator_carries_accumulated_slopes_between_dem_passes() -> None:
    configuration = ContractConfiguration(
        tile_width=1, tile_height=1, azimuth_count=3, subpatch_size=8,
        dem_count=2, primary_width=2, primary_height=2,
    )
    values = np.zeros((3, 4, 2, 18), dtype=np.float32)
    ids = np.broadcast_to(
        np.arange(2, dtype=np.int32), values.shape[:-1]
    ).copy()
    tensor = SegmentTensor(values, ids, configuration)
    expected_passes = (
        np.array([[5.0, 1.0, -np.inf]], dtype=np.float32),
        np.array([[2.0, 4.0, 3.0]], dtype=np.float32),
    )

    class FakeSession:
        def __init__(self):
            self.calls = []

        def subpatch_hierarchical_all_passes(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return np.maximum(*expected_passes), [0.1, 0.2]

    session = FakeSession()
    generated = generate_patch_horizons(
        session, tensor, [object(), object()], tile_column=0, tile_row=0
    )
    assert len(session.calls) == 1
    assert session.calls[0][0][0] is tensor.values
    assert len(session.calls[0][0][1]) == 2
    np.testing.assert_array_equal(
        generated.slopes, np.array([[5.0, 4.0, 3.0]], dtype=np.float32)
    )


def test_cuda_session_reuses_unchanged_production_pyramids() -> None:
    class FakeCuda:
        def __init__(self):
            self.uploads = []

        def to_device(self, value):
            device = ("device", id(value))
            self.uploads.append(device)
            return device

    def pyramid():
        return SimpleNamespace(
            level0=np.zeros((2, 2), dtype=np.float32),
            mips=np.zeros(1, dtype=np.float32),
            levels=np.zeros((1, 4), dtype=np.int32),
            map_parameters=np.zeros(11, dtype=np.float32),
            projection_parameters=np.zeros(6, dtype=np.float32),
        )

    session = object.__new__(CudaSession)
    session._cuda = FakeCuda()
    session._production_pyramids = None
    session._production_device_pyramids = None
    first = (pyramid(), pyramid())
    first_device = session._prepare_production_pyramids(first)
    assert len(session._cuda.uploads) == 10
    assert session._prepare_production_pyramids(first) is first_device
    assert len(session._cuda.uploads) == 10
    replacement = (first[0], pyramid())
    assert session._prepare_production_pyramids(replacement) is not first_device
    assert len(session._cuda.uploads) == 20


def _boundary_patch_inputs():
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    values = artifact.arrays[
        "boundary_halo_multi_dem_16az__subpatch_fixture__segment_values"
    ]
    pyramids = [
        PyramidArrays.from_artifact(
            artifact.arrays, f"multi_dem_outer_obstacle_east__dem_{index}__pyramid"
        )
        for index in range(2)
    ]
    return values, pyramids


def _case_dem(artifact, case_id: str, index: int) -> DemGrid:
    prefix = f"{case_id}__dem_{index}"
    return DemGrid(
        artifact.arrays[f"{prefix}__elevation_m"],
        artifact.arrays[f"{prefix}__geo_transform"],
        ProjectionParameters.from_array(
            artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
        ),
    )


def _different_resolution_inputs():
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    case_id = "multi_dem_different_resolutions"
    dems = [_case_dem(artifact, case_id, index) for index in range(2)]
    pyramids = [build_max_pyramid(dem) for dem in dems]
    segments, _, _ = build_subpatch_segments(
        dems,
        tile_column=20,
        tile_row=20,
        tile_width=1,
        azimuth_count=16,
        maximum_distance_m=2500,
        observer_elevation_m=0,
        subpatch_size=8,
        grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
    )
    return segments, pyramids


def _full_patch_inputs():
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    projection = ProjectionParameters.from_array(
        artifact.arrays["flat_east__dem_0__pyramid__projection_parameters"]
    )
    dem = DemGrid(
        np.zeros((129, 129), dtype=np.float32),
        np.array((-1920.0, 30.0, 0.0, 1920.0, 0.0, -30.0), dtype=np.float64),
        projection,
    )
    pyramid = build_max_pyramid(dem)
    segments, _, _ = build_subpatch_segments(
        [dem],
        tile_column=0,
        tile_row=0,
        tile_width=128,
        azimuth_count=16,
        maximum_distance_m=2500,
        observer_elevation_m=0,
        subpatch_size=8,
        grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
    )
    return segments, pyramid


def test_cpu_subpatch_oracle_covers_halo_clamping_and_resolution_ratio() -> None:
    values, pyramids = _boundary_patch_inputs()
    selected = []
    for column, row in ((0, 0), (15, 0), (0, 15), (15, 15), (7, 7), (8, 8)):
        for azimuth in (0, 4, 8, 12, 15):
            for pass_index in range(2):
                segment = interpolate_pixel_segment(
                    values, pyramids[0], pyramids[pass_index],
                    tile_column=0, tile_row=0, tile_width=16, subpatch_size=8,
                    pixel_column=column, pixel_row=row, azimuth=azimuth,
                    pass_index=pass_index,
                )
                assert segment.shape == (18,)
                assert np.all(np.isfinite(segment))
                selected.append(segment)
    assert len(selected) == 60


def test_host_layout_covers_full_patch_and_different_resolution_dems() -> None:
    different_segments, different_pyramids = _different_resolution_inputs()
    assert different_segments.shape == (16, 4, 2, 18)
    assert different_pyramids[0].map_parameters[6] == 30.0
    assert different_pyramids[1].map_parameters[6] == 60.0
    full_segments, _ = _full_patch_inputs()
    assert full_segments.shape == (16, 324, 1, 18)


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_cuda_partial_patch_edges_match_selected_cpu_subpatch_rays() -> None:
    values, pyramids = _boundary_patch_inputs()
    session = CudaSession()
    gpu_passes = []
    for pass_index in range(2):
        gpu_passes.append(
            session.subpatch_hierarchical_pass(
                values, pyramids[0], pyramids[pass_index],
                tile_column=0, tile_row=0, tile_width=16, tile_height=16,
                subpatch_size=8, pass_index=pass_index,
            )
        )
    observer_z = sample_bilinear(pyramids[0].level0, 0.0, 0.0)
    for column, row in ((0, 0), (15, 0), (0, 15), (15, 15), (7, 7), (8, 8)):
        pixel = row * 16 + column
        observer_z = sample_bilinear(
            pyramids[0].level0, float(column), float(row)
        )
        for azimuth in (0, 4, 8, 12, 15):
            for pass_index in range(2):
                segment = interpolate_pixel_segment(
                    values, pyramids[0], pyramids[pass_index],
                    tile_column=0, tile_row=0, tile_width=16, subpatch_size=8,
                    pixel_column=column, pixel_row=row, azimuth=azimuth,
                    pass_index=pass_index,
                )
                active_map = pyramids[pass_index].map_parameters
                resolution = 0.5 * (
                    np.hypot(active_map[6], active_map[9])
                    + np.hypot(active_map[7], active_map[10])
                )
                cpu = traverse_hierarchy(
                    segment, pyramids[pass_index], observer_z_m=float(observer_z),
                    radius_m=float(pyramids[pass_index].projection_parameters[0]),
                    map_resolution_m=float(resolution), pass_index=pass_index,
                )
                assert gpu_passes[pass_index][pixel, azimuth] == pytest.approx(
                    float(cpu.maximum_slope), abs=2e-6
                )


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_cuda_full_patch_and_different_resolution_rays_match_cpu_oracle() -> None:
    session = CudaSession()
    full_segments, full_pyramid = _full_patch_inputs()
    full_gpu = session.subpatch_hierarchical_pass(
        full_segments, full_pyramid, full_pyramid,
        tile_column=0, tile_row=0, tile_width=128, tile_height=128,
        subpatch_size=8, pass_index=0,
    )
    for column, row in ((0, 0), (127, 0), (0, 127), (127, 127), (63, 63), (64, 64)):
        pixel = row * 128 + column
        for azimuth in (0, 4, 8, 12, 15):
            segment = interpolate_pixel_segment(
                full_segments, full_pyramid, full_pyramid,
                tile_column=0, tile_row=0, tile_width=128, subpatch_size=8,
                pixel_column=column, pixel_row=row, azimuth=azimuth, pass_index=0,
            )
            cpu = traverse_hierarchy(
                segment, full_pyramid, observer_z_m=0.0,
                radius_m=float(full_pyramid.projection_parameters[0]),
                map_resolution_m=30.0, pass_index=0,
            )
            assert full_gpu[pixel, azimuth] == pytest.approx(
                float(cpu.maximum_slope), abs=2e-6
            )

    segments, pyramids = _different_resolution_inputs()
    pass_zero = session.subpatch_hierarchical_pass(
        segments, pyramids[0], pyramids[0], tile_column=20, tile_row=20,
        tile_width=1, tile_height=1, subpatch_size=8, pass_index=0,
    )
    pass_one = session.subpatch_hierarchical_pass(
        segments, pyramids[0], pyramids[1], tile_column=20, tile_row=20,
        tile_width=1, tile_height=1, subpatch_size=8, pass_index=1,
    )
    for pass_index, gpu in enumerate((pass_zero, pass_one)):
        resolution = float(pyramids[pass_index].map_parameters[6])
        for azimuth in range(16):
            segment = interpolate_pixel_segment(
                segments, pyramids[0], pyramids[pass_index],
                tile_column=20, tile_row=20, tile_width=1, subpatch_size=8,
                pixel_column=0, pixel_row=0, azimuth=azimuth,
                pass_index=pass_index,
            )
            cpu = traverse_hierarchy(
                segment, pyramids[pass_index], observer_z_m=0.0,
                radius_m=float(pyramids[pass_index].projection_parameters[0]),
                map_resolution_m=resolution, pass_index=pass_index,
            )
            assert gpu[0, azimuth] == pytest.approx(
                float(cpu.maximum_slope), abs=2e-6
            )
