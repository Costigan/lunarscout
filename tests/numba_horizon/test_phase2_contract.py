from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lunarscout._numba_horizon.contract import (
    DEVICE_FLOAT_DTYPE,
    DEVICE_INT_DTYPE,
    HOST_FLOAT_DTYPE,
    KERNEL_FLOAT_FIELDS,
    KERNEL_INT_FIELDS,
    MAP_PARAMETER_FIELDS,
    PROJECTION_PARAMETER_FIELDS,
    SEGMENT_FIELDS,
    ContractConfiguration,
    ContractValidationError,
    HorizonBuffers,
    KernelParameters,
    PyramidArrays,
    SegmentTensor,
    device_float32,
    device_int32,
    host_float64,
    load_reference_artifact,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "numba_horizon"
METADATA_PATH = DATA_DIR / "phase1_reference_rays.json"
NPZ_PATH = DATA_DIR / "phase1_reference_rays.npz"


@pytest.fixture(scope="module")
def artifact():
    return load_reference_artifact(METADATA_PATH, NPZ_PATH)


def _configuration(fixture: dict) -> ContractConfiguration:
    config = fixture["configuration"]
    return ContractConfiguration(
        tile_width=config["tile_width"],
        tile_height=config["tile_height"],
        azimuth_count=config["azimuth_count"],
        subpatch_size=config["subpatch_size"],
        dem_count=config["dem_count"],
        primary_width=config["primary_dem_width"],
        primary_height=config["primary_dem_height"],
    )


def test_contract_loads_every_reference_array_without_native_runtime(artifact) -> None:
    assert "clr" not in sys.modules
    assert "numba.cuda" not in sys.modules
    assert len(artifact.arrays) == 822
    assert set(artifact.arrays) == set(artifact.metadata["arrays"])
    assert all(array.flags.c_contiguous for array in artifact.arrays.values())


def test_precision_conversions_are_explicit_and_contiguous() -> None:
    source = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    host = host_float64(source, name="host geometry")
    device = device_float32(host, name="device geometry")
    integers = device_int32([0, 1, 2], name="indices")

    assert host.dtype == HOST_FLOAT_DTYPE and host.flags.c_contiguous
    assert device.dtype == DEVICE_FLOAT_DTYPE and device.flags.c_contiguous
    assert integers.dtype == DEVICE_INT_DTYPE and integers.flags.c_contiguous
    np.testing.assert_array_equal(device, source.astype(np.float32))
    with pytest.raises(ContractValidationError, match="finite integers"):
        device_int32([1.5], name="indices")


def test_host_samples_remain_float64_and_segments_cross_as_float32(artifact) -> None:
    sample_names = [name for name in artifact.arrays if "__sample_" in name]
    segment_names = [name for name in artifact.arrays if name.endswith("__segment_values")]
    assert sample_names and segment_names
    assert all(artifact.arrays[name].dtype == HOST_FLOAT_DTYPE for name in sample_names)
    assert all(artifact.arrays[name].dtype == DEVICE_FLOAT_DTYPE for name in segment_names)


def test_configuration_and_kernel_parameters_freeze_shapes_and_dtypes() -> None:
    configuration = ContractConfiguration(16, 12, 1440, 8, 2, 41, 41)
    assert configuration.pixel_count == 192
    assert configuration.subpatches_per_dimension == 4
    assert configuration.subpatch_count == 16
    assert configuration.output_shape == (192, 1440)

    parameters = KernelParameters.create(
        observer_elevation_m=2.0,
        minimum_traverse_distance_km=0.001,
        gamma_center_rad=0.1,
        d_gamma_dx_rad_per_pixel=1e-4,
        d_gamma_dy_rad_per_pixel=-1e-4,
        debug_azimuth_index=-1,
        debug_flags=0,
        primary_width=41,
        primary_height=41,
    )
    assert parameters.floats.shape == (len(KERNEL_FLOAT_FIELDS),)
    assert parameters.integers.shape == (len(KERNEL_INT_FIELDS),)
    assert parameters.floats.dtype == DEVICE_FLOAT_DTYPE
    assert parameters.integers.dtype == DEVICE_INT_DTYPE

    with pytest.raises(ContractValidationError, match="subpatch_size"):
        ContractConfiguration(16, 16, 16, 3, 1, 16, 16)
    with pytest.raises(ContractValidationError, match="cannot exceed"):
        ContractConfiguration(129, 16, 16, 8, 1, 256, 256)


def test_segment_tensor_matches_csharp_flattening_and_boundary_interpolation(artifact) -> None:
    fixture = next(
        item for item in artifact.metadata["subpatch_fixtures"]
        if item["id"] == "boundary_halo_multi_dem_16az"
    )
    prefix = f"{fixture['id']}__subpatch_fixture"
    tensor = SegmentTensor(
        artifact.arrays[f"{prefix}__segment_values"],
        artifact.arrays[f"{prefix}__segment_dem_ids"],
        _configuration(fixture),
    )
    assert tensor.values.shape == (16, 16, 2, len(SEGMENT_FIELDS))

    for azimuth, subpatch, dem in ((0, 0, 0), (7, 9, 1), (15, 15, 1)):
        expected = np.ravel_multi_index(
            (azimuth, subpatch, dem), tensor.values.shape[:-1], order="C"
        )
        assert tensor.flat_index(azimuth, subpatch, dem) == expected
        np.testing.assert_array_equal(
            tensor.segment(azimuth, subpatch, dem),
            tensor.values[azimuth, subpatch, dem],
        )

    indices, weights = tensor.interpolation_selection(0, 0)
    assert indices == (0, 1, 4, 5)
    assert weights == (np.float32(0.5), np.float32(0.5))
    # All four halo centers clamp to the same C# segment center at this corner.
    np.testing.assert_array_equal(tensor.interpolate(7, 0, 0, 1), tensor.values[7, 0, 1])


def test_segment_validation_rejects_layout_and_dem_axis_mismatches(artifact) -> None:
    fixture = artifact.metadata["subpatch_fixtures"][0]
    prefix = f"{fixture['id']}__subpatch_fixture"
    values = artifact.arrays[f"{prefix}__segment_values"]
    dem_ids = artifact.arrays[f"{prefix}__segment_dem_ids"]
    configuration = _configuration(fixture)

    with pytest.raises(ContractValidationError, match="C-contiguous"):
        SegmentTensor(values[:, ::-1], dem_ids[:, ::-1].copy(), configuration)
    wrong_ids = dem_ids.copy()
    wrong_ids[0, 0, 0] = 1
    with pytest.raises(ContractValidationError, match="DEM axis"):
        SegmentTensor(values, wrong_ids, configuration)


def test_every_captured_pyramid_round_trips_flat_offsets_and_cells(artifact) -> None:
    captured: list[tuple[str, dict]] = []
    for case in artifact.metadata["cases"]:
        for pyramid in case["pyramids"]:
            captured.append((f"{case['id']}__dem_{pyramid['dem_index']}__pyramid", pyramid))
    for fixture in artifact.metadata["pyramid_fixtures"]:
        captured.append((f"{fixture['id']}__pyramid_fixture", fixture["pyramid"]))
    for fixture in artifact.metadata["horizon_buffer_fixtures"]:
        base = f"{fixture['id']}__horizon_buffer_fixture"
        for dem in fixture["dems"]:
            captured.append((f"{base}__dem_{dem['index']}__pyramid", dem["pyramid"]))

    assert captured
    for prefix, metadata in captured:
        pyramid = PyramidArrays.from_artifact(artifact.arrays, prefix)
        assert pyramid.levels.shape == (metadata["level_count"], 4)
        assert pyramid.map_parameters.shape == (len(MAP_PARAMETER_FIELDS),)
        assert pyramid.projection_parameters.shape == (len(PROJECTION_PARAMETER_FIELDS),)
        for level, (_, _, width, height) in enumerate(pyramid.levels):
            stored = artifact.arrays[f"{prefix}__level_{level}"]
            for x, y in ((0, 0), (int(width) - 1, int(height) - 1)):
                np.testing.assert_equal(pyramid.cell(level, x, y), stored[y, x])


def test_pyramid_validation_rejects_bad_offsets(artifact) -> None:
    prefix = "flat_east__dem_0__pyramid"
    pyramid = PyramidArrays.from_artifact(artifact.arrays, prefix)
    levels = pyramid.levels.copy()
    levels[2, 1] += 1
    with pytest.raises(ContractValidationError, match="offsets"):
        PyramidArrays(
            pyramid.level0, pyramid.mips, levels,
            pyramid.map_parameters, pyramid.projection_parameters,
        )


def test_slope_sentinel_merge_boundary_and_single_degree_conversion(artifact) -> None:
    fixture = artifact.metadata["horizon_buffer_fixtures"][0]
    prefix = f"{fixture['id']}__horizon_buffer_fixture"
    configuration = ContractConfiguration(
        tile_width=1,
        tile_height=1,
        azimuth_count=1440,
        subpatch_size=8,
        dem_count=2,
        primary_width=fixture["dems"][0]["width"],
        primary_height=fixture["dems"][0]["height"],
    )
    empty = HorizonBuffers.empty(configuration)
    assert np.all(np.isneginf(empty.slopes))

    per_dem = artifact.arrays[f"{prefix}__per_dem_slopes"]
    expected_slopes = artifact.arrays[f"{prefix}__final_slopes"]
    expected_degrees = artifact.arrays[f"{prefix}__final_degrees"]
    merged = empty
    for pass_slopes in per_dem:
        merged = merged.merge_pass(np.ascontiguousarray(pass_slopes))
    np.testing.assert_array_equal(merged.slopes, expected_slopes)
    actual = merged.degrees()
    np.testing.assert_allclose(actual, expected_degrees, rtol=0, atol=1e-6)

    invalid = expected_slopes.copy()
    invalid[0, 0] = np.inf
    with pytest.raises(ContractValidationError, match="negative infinity"):
        HorizonBuffers(invalid)
