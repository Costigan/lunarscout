#!/usr/bin/env python3
"""Validate Phase 1 reference-ray oracle metadata, arrays, and expectations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = REPO_ROOT / "tests/data/numba_horizon/phase1_reference_rays.npz"
DEFAULT_METADATA = REPO_ROOT / "tests/data/numba_horizon/phase1_reference_rays.json"


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def downsample_max(source: np.ndarray, factor: int) -> np.ndarray:
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


def validate_pyramid(
    arrays: Any,
    pyramid: dict[str, Any],
    prefix: str,
    source: np.ndarray,
) -> None:
    level_metadata = arrays[f"{prefix}__level_metadata"]
    cell_sizes = arrays[f"{prefix}__level_cell_sizes"]
    assert level_metadata.shape == (pyramid["level_count"], 4)
    assert np.all(cell_sizes == 0)
    expected = source
    expected_offset = 0
    for level_index, (stored_level, offset, width, height) in enumerate(
        level_metadata
    ):
        assert stored_level == level_index
        assert offset == (0 if level_index == 0 else expected_offset)
        actual = arrays[f"{prefix}__level_{level_index}"]
        assert actual.shape == (height, width)
        np.testing.assert_array_equal(actual, expected)
        if level_index + 1 < pyramid["level_count"]:
            if level_index > 0:
                expected_offset += int(width * height)
            expected = downsample_max(expected, pyramid["downsample_factor"])
    assert arrays[f"{prefix}__map_parameters"].shape == (11,)
    assert arrays[f"{prefix}__projection_parameters"].shape == (6,)


def validate_expectations(cases: list[dict[str, Any]]) -> None:
    by_id = {case["id"]: case for case in cases}
    for case in cases:
        result = case["result"]
        expected = case["expectations"]
        angle = result["elevation_degrees"]
        if "elevation_degrees_min_exclusive" in expected:
            assert angle > expected["elevation_degrees_min_exclusive"], case["id"]
        if "elevation_degrees_max_exclusive" in expected:
            assert angle < expected["elevation_degrees_max_exclusive"], case["id"]
        if "minimum_pass_count" in expected:
            assert result["pass_count"] >= expected["minimum_pass_count"], case["id"]
        comparison_id = expected.get("comparison_case")
        if comparison_id is None:
            continue
        difference = angle - by_id[comparison_id]["result"]["elevation_degrees"]
        if "minimum_elevation_advantage_degrees" in expected:
            assert difference > expected["minimum_elevation_advantage_degrees"], case["id"]
        if "maximum_elevation_disadvantage_degrees" in expected:
            assert difference < expected["maximum_elevation_disadvantage_degrees"], case["id"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["capture_accelerator"]["type"] == "Cuda"
    assert hashlib.sha256(args.npz.read_bytes()).hexdigest() == metadata["npz_sha256"]
    with np.load(args.npz, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(metadata["arrays"])
        for name, expected in metadata["arrays"].items():
            array = arrays[name]
            assert array.dtype.str == expected["dtype"], name
            assert list(array.shape) == expected["shape"], name
            assert array_sha256(array) == expected["sha256_c_order_data"], name
            if "__ray_fit_pass_" in name or name.endswith("__slopes") or "__trace_" in name:
                assert np.all(np.isfinite(array)), name
        for case in metadata["cases"]:
            for pyramid in case["pyramids"]:
                prefix = f"{case['id']}__dem_{pyramid['dem_index']}__pyramid"
                source = arrays[
                    f"{case['id']}__dem_{pyramid['dem_index']}__elevation_m"
                ]
                validate_pyramid(arrays, pyramid, prefix, source)

            for pass_index, _ in enumerate(case["passes"]):
                slopes = arrays[f"{case['id']}__pass_{pass_index}__slopes"]
                trace_slopes = arrays[f"{case['id']}__pass_{pass_index}__trace_slope"]
                np.testing.assert_array_equal(slopes, trace_slopes)

            previous_final_distance = None
            for fit_index, fit in enumerate(case["ray_fit_passes"]):
                prefix = f"{case['id']}__ray_fit_pass_{fit_index}"
                distance_m = arrays[f"{prefix}__sample_distance_m"]
                pixel_x = arrays[f"{prefix}__sample_pixel_x"]
                pixel_y = arrays[f"{prefix}__sample_pixel_y"]
                direction = arrays[f"{prefix}__nominal_direction_moon_centered"]
                segment_values = arrays[f"{prefix}__segment_values"]
                fields = [field["name"] for field in fit["segment_fields"]]
                segment = dict(zip(fields, segment_values, strict=True))

                assert len(distance_m) == fit["sample_count"]
                assert len(distance_m) >= 4
                assert np.all(np.diff(distance_m) > 0)
                np.testing.assert_allclose(np.linalg.norm(direction), 1.0, rtol=0, atol=1e-12)
                assert fit["segment_dem_id"] == fit["dem_index"]
                if previous_final_distance is not None:
                    np.testing.assert_allclose(
                        fit["requested_start_distance_m"],
                        previous_final_distance,
                        rtol=0,
                        atol=1e-9,
                    )

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
                previous_final_distance = float(distance_m[-1])

        for fixture in metadata["pyramid_fixtures"]:
            prefix = f"{fixture['id']}__pyramid_fixture"
            source = arrays[f"{prefix}__elevation_m"]
            expected = fixture["expectations"]
            assert np.count_nonzero(np.isnan(source)) == expected["expected_nan_count_level0"]
            assert np.count_nonzero(np.isposinf(source)) == expected["expected_positive_infinity_count_level0"]
            assert np.count_nonzero(np.isneginf(source)) == expected["expected_negative_infinity_count_level0"]
            validate_pyramid(arrays, fixture["pyramid"], prefix, source)
            np.testing.assert_array_equal(
                arrays[f"{prefix}__level_1"],
                np.array([[-19999.0, 42.0], [-32000.0, 100.0]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                arrays[f"{prefix}__level_2"],
                np.array([[100.0]], dtype=np.float32),
            )

        for fixture in metadata["subpatch_fixtures"]:
            prefix = f"{fixture['id']}__subpatch_fixture"
            configuration = fixture["configuration"]
            centers = arrays[f"{prefix}__centers"]
            grid_convergence = arrays[f"{prefix}__grid_convergence"]
            segment_values = arrays[f"{prefix}__segment_values"]
            dem_ids = arrays[f"{prefix}__segment_dem_ids"]
            assert centers.shape == (fixture["center_count"], 7)
            assert segment_values.shape == (
                configuration["azimuth_count"],
                fixture["center_count"],
                configuration["dem_count"],
                len(fixture["segment_fields"]),
            )
            assert dem_ids.shape == segment_values.shape[:-1]
            assert np.all(np.isfinite(segment_values))
            assert np.all(np.isfinite(grid_convergence))
            assert np.any(grid_convergence != 0)
            np.testing.assert_array_equal(
                dem_ids,
                np.broadcast_to(
                    np.arange(configuration["dem_count"], dtype=np.int32),
                    dem_ids.shape,
                ),
            )
            expected_requested = np.array([-4, 4, 12, 20], dtype=np.int32)
            half_subpatch = configuration["subpatch_size"] // 2
            expected_segment_x = np.clip(
                expected_requested,
                half_subpatch,
                configuration["primary_dem_width"] - half_subpatch,
            )
            expected_segment_y = np.clip(
                expected_requested,
                half_subpatch,
                configuration["primary_dem_height"] - half_subpatch,
            )
            np.testing.assert_array_equal(
                centers[:, 3].reshape(4, 4),
                np.broadcast_to(expected_requested, (4, 4)),
            )
            np.testing.assert_array_equal(
                centers[:, 4].reshape(4, 4),
                np.broadcast_to(expected_requested[:, None], (4, 4)),
            )
            np.testing.assert_array_equal(
                centers[:, 5].reshape(4, 4),
                np.broadcast_to(expected_segment_x, (4, 4)),
            )
            np.testing.assert_array_equal(
                centers[:, 6].reshape(4, 4),
                np.broadcast_to(expected_segment_y[:, None], (4, 4)),
            )
            for first in range(len(centers)):
                for second in range(first):
                    if np.array_equal(centers[first, 5:7], centers[second, 5:7]):
                        np.testing.assert_array_equal(
                            segment_values[:, first], segment_values[:, second]
                        )

        for fixture in metadata["horizon_buffer_fixtures"]:
            prefix = f"{fixture['id']}__horizon_buffer_fixture"
            configuration = fixture["configuration"]
            for dem in fixture["dems"]:
                dem_prefix = f"{prefix}__dem_{dem['index']}"
                validate_pyramid(
                    arrays,
                    dem["pyramid"],
                    f"{dem_prefix}__pyramid",
                    arrays[f"{dem_prefix}__elevation_m"],
                )
            per_dem = arrays[f"{prefix}__per_dem_slopes"]
            final_slopes = arrays[f"{prefix}__final_slopes"]
            final_degrees = arrays[f"{prefix}__final_degrees"]
            expected_shape = (
                configuration["tile_width"] * configuration["tile_height"],
                configuration["azimuth_count"],
            )
            assert per_dem.shape == (configuration["dem_count"], *expected_shape)
            assert final_slopes.shape == final_degrees.shape == expected_shape
            assert np.all(np.isfinite(per_dem))
            np.testing.assert_array_equal(final_slopes, np.max(per_dem, axis=0))
            np.testing.assert_allclose(
                final_degrees,
                np.degrees(np.arctan(final_slopes.astype(np.float64))),
                rtol=0,
                atol=1e-6,
            )
            east_index = configuration["azimuth_count"] // 4
            assert per_dem[1, 0, east_index] > per_dem[0, 0, east_index]
            assert final_degrees[0, east_index] > 5.0

            trace_metadata = fixture["traversal_trace"]
            trace = arrays[f"{prefix}__traversal_trace"]
            fields = {name: index for index, name in enumerate(trace_metadata["fields"])}
            assert trace.shape == (trace_metadata["step_count"], len(fields))
            assert len(trace) > 0
            assert np.all(np.diff(trace[:, fields["parameter_distance_km"]]) >= 0)
            assert np.all(trace[:, fields["level"]] >= 0)
            assert np.all(trace[:, fields["advance_km"]] >= 0)
            actions = trace[:, fields["action"]].astype(np.int32)
            assert set(actions).issubset(set(trace_metadata["action_codes"].values()))
            assert trace_metadata["action_codes"]["descend"] in actions
            sample_rows = actions == trace_metadata["action_codes"]["level0_sample"]
            assert np.any(sample_rows)
            assert np.all(np.isfinite(trace[sample_rows, fields["sample_elevation_m"]]))
            assert np.all(np.isfinite(trace[sample_rows, fields["sample_slope"]]))
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

    validate_expectations(metadata["cases"])
    print(
        f"validated {len(metadata['cases'])} cases and "
        f"{len(metadata['arrays'])} arrays from {args.npz}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
