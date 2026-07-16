from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "numba_horizon"
METADATA_PATH = DATA_DIR / "phase1_reference_rays.json"
NPZ_PATH = DATA_DIR / "phase1_reference_rays.npz"


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _downsample_max(source: np.ndarray, factor: int) -> np.ndarray:
    height, width = source.shape
    output = np.full(
        ((height + factor - 1) // factor, (width + factor - 1) // factor),
        -32000.0,
        dtype=np.float32,
    )
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            block = source[
                row * factor : (row + 1) * factor,
                column * factor : (column + 1) * factor,
            ]
            valid = block[np.isfinite(block) & (block > -20000.0)]
            if valid.size:
                output[row, column] = np.max(valid)
    return output


def test_reference_ray_artifact_matches_manifest() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["schema_version"] == 1
    assert metadata["baseline_commit"] == (
        "f3b21b5a7d510162783c8e6a1aa01ca2edc61277"
    )
    assert hashlib.sha256(NPZ_PATH.read_bytes()).hexdigest() == metadata["npz_sha256"]

    with np.load(NPZ_PATH, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(metadata["arrays"])
        for name, expected in metadata["arrays"].items():
            array = arrays[name]
            assert array.dtype.str == expected["dtype"], name
            assert list(array.shape) == expected["shape"], name
            assert _array_sha256(array) == expected["sha256_c_order_data"], name


def test_reference_ray_artifact_preserves_analytical_expectations() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in metadata["cases"]}

    flat = cases["flat_east"]["result"]["elevation_degrees"]
    obstacle_east = cases["single_obstacle_east"]["result"]["elevation_degrees"]
    obstacle_west = cases["single_obstacle_west"]["result"]["elevation_degrees"]
    obstacle_northeast = cases["single_obstacle_northeast"]["result"][
        "elevation_degrees"
    ]
    obstacle_southwest = cases["single_obstacle_southwest"]["result"][
        "elevation_degrees"
    ]
    inner_only = cases["multi_dem_inner_only_east"]["result"]["elevation_degrees"]
    outer_obstacle = cases["multi_dem_outer_obstacle_east"]["result"]

    assert -1.0 < flat < 0.0
    assert obstacle_east > 5.0
    assert obstacle_east > obstacle_west + 5.0
    assert obstacle_northeast > 5.0
    assert obstacle_northeast > obstacle_southwest + 5.0
    assert outer_obstacle["pass_count"] >= 2
    assert outer_obstacle["elevation_degrees"] > 5.0
    assert outer_obstacle["elevation_degrees"] > inner_only + 5.0

    with np.load(NPZ_PATH, allow_pickle=False) as arrays:
        for case in cases.values():
            for pass_index, _ in enumerate(case["passes"]):
                prefix = f"{case['id']}__pass_{pass_index}"
                np.testing.assert_array_equal(
                    arrays[f"{prefix}__slopes"],
                    arrays[f"{prefix}__trace_slope"],
                )


def test_ray_fit_samples_and_coefficients_are_self_consistent() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    with np.load(NPZ_PATH, allow_pickle=False) as arrays:
        for case in metadata["cases"]:
            for fit_index, fit in enumerate(case["ray_fit_passes"]):
                prefix = f"{case['id']}__ray_fit_pass_{fit_index}"
                distance_m = arrays[f"{prefix}__sample_distance_m"]
                pixel_x = arrays[f"{prefix}__sample_pixel_x"]
                pixel_y = arrays[f"{prefix}__sample_pixel_y"]
                values = arrays[f"{prefix}__segment_values"]
                fields = [field["name"] for field in fit["segment_fields"]]
                segment = dict(zip(fields, values, strict=True))

                assert len(distance_m) == fit["sample_count"]
                assert len(distance_m) >= 4
                assert np.all(np.diff(distance_m) > 0)
                assert values.dtype == np.dtype("<f4")
                assert fit["segment_dem_id"] == fit["dem_index"]

                delta_km = distance_m / 1000.0 - segment["s_start_km"]
                fitted_x = segment["x0"] + sum(
                    segment[f"a{power}"] * delta_km**power
                    for power in range(1, 5)
                )
                fitted_y = segment["y0"] + sum(
                    segment[f"b{power}"] * delta_km**power
                    for power in range(1, 5)
                )
                assert np.max(np.abs(fitted_x - pixel_x)) < 0.05
                assert np.max(np.abs(fitted_y - pixel_y)) < 0.05


def test_production_pyramids_match_independent_max_reduction() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["capture_accelerator"]["type"] == "Cuda"
    with np.load(NPZ_PATH, allow_pickle=False) as arrays:
        for case in metadata["cases"]:
            for pyramid in case["pyramids"]:
                dem_index = pyramid["dem_index"]
                prefix = f"{case['id']}__dem_{dem_index}__pyramid"
                expected = arrays[f"{case['id']}__dem_{dem_index}__elevation_m"]
                level_metadata = arrays[f"{prefix}__level_metadata"]

                for level_index in range(pyramid["level_count"]):
                    _, _, width, height = level_metadata[level_index]
                    actual = arrays[f"{prefix}__level_{level_index}"]
                    assert actual.shape == (height, width)
                    np.testing.assert_array_equal(actual, expected)
                    expected = _downsample_max(
                        expected, pyramid["downsample_factor"]
                    )


def test_pyramid_invalid_values_follow_production_nodata_rules() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    [fixture] = metadata["pyramid_fixtures"]
    prefix = f"{fixture['id']}__pyramid_fixture"

    with np.load(NPZ_PATH, allow_pickle=False) as arrays:
        source = arrays[f"{prefix}__elevation_m"]
        assert np.count_nonzero(np.isnan(source)) == 1
        assert np.count_nonzero(np.isposinf(source)) == 1
        assert np.count_nonzero(np.isneginf(source)) == 1
        np.testing.assert_array_equal(
            arrays[f"{prefix}__level_1"],
            np.array(
                [[-19999.0, 42.0], [-32000.0, 100.0]], dtype=np.float32
            ),
        )
        np.testing.assert_array_equal(
            arrays[f"{prefix}__level_2"],
            np.array([[100.0]], dtype=np.float32),
        )


def test_complete_subpatch_grid_preserves_layout_and_boundary_clamping() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    fixture = next(
        item
        for item in metadata["subpatch_fixtures"]
        if item["id"] == "boundary_halo_multi_dem_16az"
    )
    prefix = f"{fixture['id']}__subpatch_fixture"

    with np.load(NPZ_PATH, allow_pickle=False) as arrays:
        centers = arrays[f"{prefix}__centers"]
        segments = arrays[f"{prefix}__segment_values"]
        dem_ids = arrays[f"{prefix}__segment_dem_ids"]
        convergence = arrays[f"{prefix}__grid_convergence"]

        assert centers.shape == (16, 7)
        assert segments.shape == (16, 16, 2, 18)
        assert dem_ids.shape == (16, 16, 2)
        assert np.all(dem_ids[..., 0] == 0)
        assert np.all(dem_ids[..., 1] == 1)
        assert np.all(np.isfinite(segments))
        assert np.any(convergence != 0)

        # The negative interpolation halo clamps onto the first legal center.
        np.testing.assert_array_equal(centers[0, 5:7], [4, 4])
        np.testing.assert_array_equal(centers[1, 5:7], [4, 4])
        np.testing.assert_array_equal(centers[4, 5:7], [4, 4])
        np.testing.assert_array_equal(centers[5, 5:7], [4, 4])
        np.testing.assert_array_equal(segments[:, 0], segments[:, 1])
        np.testing.assert_array_equal(segments[:, 0], segments[:, 4])
        np.testing.assert_array_equal(segments[:, 0], segments[:, 5])


def test_material_grid_convergence_fixture_is_nonzero() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    fixture = next(
        item
        for item in metadata["subpatch_fixtures"]
        if item["id"] == "material_grid_convergence_16az"
    )
    prefix = f"{fixture['id']}__subpatch_fixture"

    with np.load(NPZ_PATH, allow_pickle=False) as arrays:
        convergence = arrays[f"{prefix}__grid_convergence"]
        assert abs(float(convergence[0])) > 0.01


def test_per_dem_and_final_horizon_buffers_preserve_conversion_boundary() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    [fixture] = metadata["horizon_buffer_fixtures"]
    prefix = f"{fixture['id']}__horizon_buffer_fixture"

    with np.load(NPZ_PATH, allow_pickle=False) as arrays:
        per_dem = arrays[f"{prefix}__per_dem_slopes"]
        final_slopes = arrays[f"{prefix}__final_slopes"]
        final_degrees = arrays[f"{prefix}__final_degrees"]

        assert per_dem.shape == (2, 1, 1440)
        assert final_slopes.shape == final_degrees.shape == (1, 1440)
        np.testing.assert_array_equal(final_slopes, np.max(per_dem, axis=0))
        np.testing.assert_allclose(
            final_degrees,
            np.degrees(np.arctan(final_slopes.astype(np.float64))),
            rtol=0,
            atol=1e-6,
        )
        assert per_dem[1, 0, 360] > per_dem[0, 0, 360]
        assert final_degrees[0, 360] > 5.0


def test_selected_cuda_traversal_trace_reproduces_selected_output_bin() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    [fixture] = metadata["horizon_buffer_fixtures"]
    trace_metadata = fixture["traversal_trace"]
    prefix = f"{fixture['id']}__horizon_buffer_fixture"

    with np.load(NPZ_PATH, allow_pickle=False) as arrays:
        trace = arrays[f"{prefix}__traversal_trace"]
        per_dem = arrays[f"{prefix}__per_dem_slopes"]

        fields = {name: index for index, name in enumerate(trace_metadata["fields"])}
        actions = trace[:, fields["action"]].astype(np.int32)
        sample_rows = actions == trace_metadata["action_codes"]["level0_sample"]

        assert len(trace) == trace_metadata["step_count"] > 0
        assert trace_metadata["action_codes"]["descend"] in actions
        assert np.any(sample_rows)
        assert np.all(np.diff(trace[:, fields["parameter_distance_km"]]) >= 0)
        assert np.all(trace[:, fields["advance_km"]] >= 0)
        np.testing.assert_allclose(
            np.max(trace[sample_rows, fields["sample_slope"]]),
            per_dem[
                trace_metadata["dem_pass"],
                trace_metadata["pixel_index"],
                trace_metadata["azimuth_index"],
            ],
            rtol=0,
            atol=1e-7,
        )
