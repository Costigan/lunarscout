from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunarscout._numba_horizon.contract import SEGMENT_FIELDS, load_reference_artifact
from lunarscout._numba_horizon.geometry import (
    RAY_SAMPLE_FIELDS,
    DemGrid,
    GridConvergenceInput,
    ProjectionParameters,
    SubpatchSegmentCache,
    azimuth_direction,
    build_dem_segment_contexts,
    build_ray_samples,
    build_subpatch_segments,
    build_subpatch_segments_numba,
    enu_to_moon_matrix,
    fit_ray_segment,
    inverse_stereographic,
    lat_lon_to_vector,
    project_stereographic,
    vector_to_lat_lon,
)
from lunarscout._numba_horizon.geometry_numba import (
    fit_segments_parallel,
    fit_segments_serial,
    generate_segments_parallel,
    generate_segments_serial,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "numba_horizon"


@pytest.fixture(scope="module")
def artifact():
    return load_reference_artifact(
        DATA_DIR / "phase1_reference_rays.json",
        DATA_DIR / "phase1_reference_rays.npz",
    )


def _dem(artifact, case: dict, dem_index: int) -> DemGrid:
    prefix = f"{case['id']}__dem_{dem_index}"
    return DemGrid(
        artifact.arrays[f"{prefix}__elevation_m"],
        artifact.arrays[f"{prefix}__geo_transform"],
        ProjectionParameters.from_array(
            artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
        ),
    )


def _samples(artifact, prefix: str) -> np.ndarray:
    return np.column_stack(
        [artifact.arrays[f"{prefix}__sample_{field}"] for field in RAY_SAMPLE_FIELDS]
    )


def _evaluate_segment(segment: np.ndarray, distance_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_km = distance_m / 1000.0 - segment[12]
    x = segment[2] + sum(segment[3 + power] * delta_km**power for power in range(1, 5))
    y = segment[3] + sum(segment[7 + power] * delta_km**power for power in range(1, 5))
    return x, y


def test_affine_projection_vector_and_direction_round_trips(artifact) -> None:
    case = artifact.metadata["cases"][3]
    dem = _dem(artifact, case, 0)
    for column, row in ((0.0, 0.0), (60.0, 60.0), (120.0, 120.0), (13.25, 87.75)):
        x, y = dem.pixel_to_crs(column, row)
        actual_column, actual_row = dem.crs_to_pixel(x, y)
        np.testing.assert_allclose((actual_column, actual_row), (column, row), atol=1e-12)
        latitude, longitude = inverse_stereographic(x, y, dem.projection)
        actual_x, actual_y = project_stereographic(latitude, longitude, dem.projection)
        np.testing.assert_allclose((actual_x, actual_y), (x, y), atol=1e-9)
        vector = lat_lon_to_vector(latitude, longitude, dem.projection.radius_m)
        actual_latitude, actual_longitude = vector_to_lat_lon(vector)
        longitude_error = np.arctan2(
            np.sin(actual_longitude - longitude),
            np.cos(actual_longitude - longitude),
        )
        np.testing.assert_allclose(actual_latitude, latitude, atol=1e-14)
        np.testing.assert_allclose(longitude_error, 0.0, atol=1e-14)

    matrix = enu_to_moon_matrix(0.2, 1.3)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-15)
    for azimuth in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0):
        np.testing.assert_allclose(np.linalg.norm(azimuth_direction(matrix, azimuth)), 1.0)


def test_all_reference_ray_samples_and_segments_match_csharp(artifact) -> None:
    maximum_sample_error = 0.0
    maximum_segment_error = 0.0
    maximum_path_error = 0.0
    for case in artifact.metadata["cases"]:
        dems = [_dem(artifact, case, dem["index"]) for dem in case["dems"]]
        observer = case["observer"]
        center_terrain = dems[0].elevation(observer["pixel_x"], observer["pixel_y"])
        correction_radius = dems[0].projection.radius_m + center_terrain
        for fit_index, fit in enumerate(case["ray_fit_passes"]):
            prefix = f"{case['id']}__ray_fit_pass_{fit_index}"
            observer_vector = artifact.arrays[
                f"{prefix}__observer_vector_moon_centered_m"
            ]
            direction = artifact.arrays[f"{prefix}__nominal_direction_moon_centered"]
            actual_samples = build_ray_samples(
                observer_vector,
                direction,
                fit["requested_start_distance_m"],
                fit["ray_limit_m"],
                dems[fit["dem_index"]],
            )
            expected_samples = _samples(artifact, prefix)
            maximum_sample_error = max(
                maximum_sample_error,
                float(np.max(np.abs(actual_samples - expected_samples))),
            )
            actual_segment = fit_ray_segment(
                actual_samples,
                fit["map_resolution_m"],
                observer_vector,
                correction_radius,
                fit["requested_start_distance_m"],
            )
            expected_segment = artifact.arrays[f"{prefix}__segment_values"]
            maximum_segment_error = max(
                maximum_segment_error,
                float(np.max(np.abs(actual_segment - expected_segment))),
            )
            actual_x, actual_y = _evaluate_segment(actual_segment, actual_samples[:, 0])
            expected_x, expected_y = _evaluate_segment(expected_segment, actual_samples[:, 0])
            maximum_path_error = max(
                maximum_path_error,
                float(np.max(np.abs(actual_x - expected_x))),
                float(np.max(np.abs(actual_y - expected_y))),
            )

    assert maximum_sample_error < 1e-10
    assert maximum_segment_error < 5e-9
    assert maximum_path_error < 1e-6


def test_segment_contexts_preserve_resolution_and_nested_distance_continuity(artifact) -> None:
    case = next(
        item for item in artifact.metadata["cases"]
        if item["id"] == "multi_dem_different_resolutions"
    )
    contexts = build_dem_segment_contexts(
        [_dem(artifact, case, 0), _dem(artifact, case, 1)], 2500.0
    )
    assert [context.map_resolution_m for context in contexts] == [30.0, 60.0]
    assert all(context.ray_limit_m <= 2500.0 for context in contexts)
    fits = case["ray_fit_passes"]
    assert fits[1]["requested_start_distance_m"] == pytest.approx(
        artifact.arrays[
            f"{case['id']}__ray_fit_pass_0__sample_distance_m"
        ][-1],
        abs=1e-9,
    )


def test_complete_subpatch_generation_matches_csharp_and_reuses_cache(artifact) -> None:
    case = next(
        item for item in artifact.metadata["cases"]
        if item["id"] == "multi_dem_outer_obstacle_east"
    )
    dems = [_dem(artifact, case, 0), _dem(artifact, case, 1)]
    fixture = next(
        item for item in artifact.metadata["subpatch_fixtures"]
        if item["id"] == "boundary_halo_multi_dem_16az"
    )
    prefix = f"{fixture['id']}__subpatch_fixture"
    config = fixture["configuration"]
    convergence = GridConvergenceInput(
        *(float(value) for value in artifact.arrays[f"{prefix}__grid_convergence"])
    )
    actual, centers, returned_convergence = build_subpatch_segments(
        dems,
        tile_column=config["tile_column"],
        tile_row=config["tile_row"],
        tile_width=config["tile_width"],
        azimuth_count=config["azimuth_count"],
        maximum_distance_m=config["max_distance_m"],
        observer_elevation_m=config["observer_elevation_m"],
        subpatch_size=config["subpatch_size"],
        grid_convergence=convergence,
    )
    expected = artifact.arrays[f"{prefix}__segment_values"]
    center_values = np.array(
        [[getattr(center, field) for field in center.__dataclass_fields__] for center in centers],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(center_values, artifact.arrays[f"{prefix}__centers"])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=6e-8)
    assert returned_convergence is convergence

    compiled_serial, _, _ = build_subpatch_segments_numba(
        dems,
        tile_column=config["tile_column"],
        tile_row=config["tile_row"],
        tile_width=config["tile_width"],
        azimuth_count=config["azimuth_count"],
        maximum_distance_m=config["max_distance_m"],
        observer_elevation_m=config["observer_elevation_m"],
        subpatch_size=config["subpatch_size"],
        grid_convergence=convergence,
        parallel=False,
    )
    compiled_parallel, _, _ = build_subpatch_segments_numba(
        dems,
        tile_column=config["tile_column"],
        tile_row=config["tile_row"],
        tile_width=config["tile_width"],
        azimuth_count=config["azimuth_count"],
        maximum_distance_m=config["max_distance_m"],
        observer_elevation_m=config["observer_elevation_m"],
        subpatch_size=config["subpatch_size"],
        grid_convergence=convergence,
        parallel=True,
    )
    np.testing.assert_allclose(compiled_serial, expected, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(compiled_parallel, compiled_serial)
    maximum_compiled_path_error = 0.0
    for compiled_segment, expected_segment in zip(
        compiled_serial.reshape(-1, len(SEGMENT_FIELDS)),
        expected.reshape(-1, len(SEGMENT_FIELDS)),
    ):
        distances = np.linspace(
            max(compiled_segment[12], expected_segment[12]),
            min(compiled_segment[13], expected_segment[13]),
            65,
            dtype=np.float64,
        ) * 1000.0
        compiled_x, compiled_y = _evaluate_segment(compiled_segment, distances)
        expected_x, expected_y = _evaluate_segment(expected_segment, distances)
        maximum_compiled_path_error = max(
            maximum_compiled_path_error,
            float(np.max(np.hypot(compiled_x - expected_x, compiled_y - expected_y))),
        )
    assert maximum_compiled_path_error < 1e-5

    cache = SubpatchSegmentCache(
        dems,
        azimuth_count=config["azimuth_count"],
        maximum_distance_m=config["max_distance_m"],
        observer_elevation_m=config["observer_elevation_m"],
    )
    first = cache.get(4, 4)
    assert cache.get(4, 4) is first
    assert len(cache._segments) == 1


def test_subpatch_segment_paths_match_over_full_fitted_ranges(artifact) -> None:
    fixture = artifact.metadata["subpatch_fixtures"][0]
    prefix = f"{fixture['id']}__subpatch_fixture"
    expected = artifact.arrays[f"{prefix}__segment_values"]
    # Dense evaluation verifies path parity between sample points, not only at fit inputs.
    for azimuth, center, dem in ((0, 0, 0), (3, 10, 1), (7, 15, 0), (15, 6, 1)):
        segment = expected[azimuth, center, dem]
        distances = np.linspace(segment[12], segment[13], 257, dtype=np.float64) * 1000.0
        x, y = _evaluate_segment(segment, distances)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        assert x[0] == pytest.approx(float(segment[2]), abs=2e-5)
        assert y[0] == pytest.approx(float(segment[3]), abs=2e-5)


def test_geometry_field_contract_remains_complete() -> None:
    assert len(RAY_SAMPLE_FIELDS) == 8
    assert len(SEGMENT_FIELDS) == 18


def test_compiled_serial_and_parallel_segment_fits_match_python(artifact) -> None:
    jobs = []
    for case in artifact.metadata["cases"]:
        dems = [_dem(artifact, case, dem["index"]) for dem in case["dems"]]
        observer = case["observer"]
        center_terrain = dems[0].elevation(observer["pixel_x"], observer["pixel_y"])
        for fit_index, fit in enumerate(case["ray_fit_passes"]):
            prefix = f"{case['id']}__ray_fit_pass_{fit_index}"
            samples = _samples(artifact, prefix)
            observer_vector = artifact.arrays[
                f"{prefix}__observer_vector_moon_centered_m"
            ]
            radius = dems[0].projection.radius_m + center_terrain
            expected = fit_ray_segment(
                samples, fit["map_resolution_m"], observer_vector, radius,
                fit["requested_start_distance_m"],
            )
            jobs.append((samples, fit, observer_vector, radius, expected))

    count = len(jobs)
    samples = np.zeros((count, 16, len(RAY_SAMPLE_FIELDS)), dtype=np.float64)
    counts = np.empty(count, dtype=np.int64)
    map_resolutions = np.empty(count, dtype=np.float64)
    observers = np.empty((count, 3), dtype=np.float64)
    radii = np.empty(count, dtype=np.float64)
    starts = np.empty(count, dtype=np.float64)
    expected = np.empty((count, len(SEGMENT_FIELDS)), dtype=np.float32)
    for index, (job_samples, fit, observer, radius, job_expected) in enumerate(jobs):
        counts[index] = len(job_samples)
        samples[index, : counts[index]] = job_samples
        map_resolutions[index] = fit["map_resolution_m"]
        observers[index] = observer
        radii[index] = radius
        starts[index] = fit["requested_start_distance_m"]
        expected[index] = job_expected

    serial = fit_segments_serial(
        samples, counts, map_resolutions, observers, radii, starts
    )
    parallel = fit_segments_parallel(
        samples, counts, map_resolutions, observers, radii, starts
    )
    np.testing.assert_allclose(serial, expected, rtol=0, atol=5e-7)
    np.testing.assert_array_equal(parallel, serial)


def test_compiled_complete_sampling_and_fitting_match_csharp(artifact) -> None:
    for case in artifact.metadata["cases"]:
        dems = [_dem(artifact, case, dem["index"]) for dem in case["dems"]]
        observer = case["observer"]
        terrain = dems[0].elevation(observer["pixel_x"], observer["pixel_y"])
        radius = dems[0].projection.radius_m + terrain
        for dem_index, dem in enumerate(dems):
            passes = [
                (fit_index, fit)
                for fit_index, fit in enumerate(case["ray_fit_passes"])
                if fit["dem_index"] == dem_index
            ]
            if not passes:
                continue
            count = len(passes)
            observers = np.empty((count, 3), dtype=np.float64)
            directions = np.empty((count, 3), dtype=np.float64)
            starts = np.empty(count, dtype=np.float64)
            maximums = np.empty(count, dtype=np.float64)
            radii = np.full(count, radius, dtype=np.float64)
            expected = np.empty((count, len(SEGMENT_FIELDS)), dtype=np.float32)
            for index, (fit_index, fit) in enumerate(passes):
                prefix = f"{case['id']}__ray_fit_pass_{fit_index}"
                observers[index] = artifact.arrays[
                    f"{prefix}__observer_vector_moon_centered_m"
                ]
                directions[index] = artifact.arrays[
                    f"{prefix}__nominal_direction_moon_centered"
                ]
                starts[index] = fit["requested_start_distance_m"]
                maximums[index] = fit["ray_limit_m"]
                expected[index] = artifact.arrays[f"{prefix}__segment_values"]
            projection = np.array(tuple(dem.projection.__dict__.values())) if hasattr(
                dem.projection, "__dict__"
            ) else np.array(
                (
                    dem.projection.radius_m,
                    dem.projection.latitude_origin_rad,
                    dem.projection.longitude_origin_rad,
                    dem.projection.scale,
                    dem.projection.false_easting_m,
                    dem.projection.false_northing_m,
                ),
                dtype=np.float64,
            )
            serial = generate_segments_serial(
                dem.elevation_m, dem.geo_transform, projection, observers, directions,
                starts, maximums, dem.map_resolution_m, radii,
            )
            parallel = generate_segments_parallel(
                dem.elevation_m, dem.geo_transform, projection, observers, directions,
                starts, maximums, dem.map_resolution_m, radii,
            )
            np.testing.assert_allclose(serial, expected, rtol=0, atol=6e-7)
            np.testing.assert_array_equal(parallel, serial)
