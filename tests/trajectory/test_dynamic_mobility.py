from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.trajectory._dynamic_mobility import (
    SunDirectionFunction,
    _compile_sun_direction_factors,
    compile_dynamic_travel_model,
)
from lunarscout.trajectory._dynamic_reference import (
    DynamicOccupancyTimeline,
    exact_dynamic_path,
)
from lunarscout.trajectory._validation import prepare_static_problem


T0 = datetime(2036, 1, 1, tzinfo=timezone.utc)


def _problem_and_timeline(
    make_trajectory_georef,
    *,
    interval_count: int = 3,
    elevation=None,
    model=None,
):
    grid = make_trajectory_georef(width=2, height=1)
    if model is None:
        model = ls.trajectory.StaticTravelModel(
            speed_m_per_h=10.0,
            include_diagonals=False,
        )
    problem = prepare_static_problem(
        np.ones((1, 2), dtype=bool),
        grid,
        (0, 0),
        goal=(1, 0),
        elevation=elevation,
        model=model,
    )
    boundaries = tuple(
        T0 + timedelta(hours=index) for index in range(interval_count + 1)
    )
    timeline = DynamicOccupancyTimeline(
        boundaries,
        np.ones((interval_count, 1, 2), dtype=bool),
        grid,
    )
    return problem, timeline


def test_compiled_mobility_integrates_progress_across_intervals(
    make_trajectory_georef,
) -> None:
    problem, timeline = _problem_and_timeline(make_trajectory_georef)
    factors = np.ones((3, 4), dtype=np.float64)
    factors[0, 0] = 2.0
    compiled = compile_dynamic_travel_model(
        problem, timeline, interval_edge_factors=factors
    )

    result = exact_dynamic_path(problem, timeline, T0, travel_model=compiled)

    assert result.reachable
    assert result.departure_times == (T0,)
    assert result.arrival_time == T0 + timedelta(hours=1.5)


def test_infinite_interval_factor_requires_waiting_at_source(
    make_trajectory_georef,
) -> None:
    problem, timeline = _problem_and_timeline(make_trajectory_georef)
    factors = np.ones((3, 4), dtype=np.float64)
    factors[0, 0] = np.inf
    compiled = compile_dynamic_travel_model(
        problem, timeline, interval_edge_factors=factors
    )

    result = exact_dynamic_path(problem, timeline, T0, travel_model=compiled)

    assert result.reachable
    assert result.departure_times == (T0 + timedelta(hours=1),)
    assert result.arrival_time == T0 + timedelta(hours=2)


def test_compilation_preserves_signed_slope_and_direction(
    make_trajectory_georef,
) -> None:
    slip = ls.trajectory.SlipFunction(
        signed_slopes=(-2.0, 0.0, 2.0),
        factors=(0.5, 1.0, 2.0),
        extrapolation="constant",
    )
    model = ls.trajectory.StaticTravelModel(
        speed_m_per_h=10.0,
        include_diagonals=False,
        slip=slip,
    )
    problem, timeline = _problem_and_timeline(
        make_trajectory_georef,
        interval_count=4,
        elevation=np.asarray([[0.0, 10.0]]),
        model=model,
    )

    compiled = compile_dynamic_travel_model(problem, timeline)

    # Direction 0 is east/uphill; direction 2 is west/downhill.
    assert compiled.base_duration_hours[0, 0, 0] == pytest.approx(1.5)
    assert compiled.base_duration_hours[0, 1, 2] == pytest.approx(0.75)


def test_hazard_factor_is_applied_at_entered_cell(make_trajectory_georef) -> None:
    problem, timeline = _problem_and_timeline(
        make_trajectory_georef, interval_count=4
    )
    compiled = compile_dynamic_travel_model(
        problem,
        timeline,
        hazard_factors=np.asarray([[1.0, 2.0]]),
    )

    outward = exact_dynamic_path(problem, timeline, T0, travel_model=compiled)

    assert outward.arrival_time == T0 + timedelta(hours=2)
    assert compiled.interval_factors[0, 0, 0, 0] == 2.0
    assert compiled.interval_factors[0, 0, 1, 2] == 1.0


@pytest.mark.parametrize(
    "factors",
    [
        np.ones((2, 4), dtype=np.float64),
        np.full((3, 4), np.nan),
        np.zeros((3, 4), dtype=np.float64),
        np.ones((3, 4), dtype=np.complex128),
    ],
)
def test_dynamic_factor_validation_is_structured(
    make_trajectory_georef, factors
) -> None:
    problem, timeline = _problem_and_timeline(make_trajectory_georef)
    with pytest.raises(ls.TrajectoryInputError) as captured:
        compile_dynamic_travel_model(
            problem,
            timeline,
            interval_edge_factors=factors,
        )
    assert captured.value.code == "trajectory_invalid_dynamic_mobility"


def test_compiled_model_must_match_timeline(make_trajectory_georef) -> None:
    problem, timeline = _problem_and_timeline(make_trajectory_georef)
    compiled = compile_dynamic_travel_model(problem, timeline)
    shifted = DynamicOccupancyTimeline(
        tuple(value + timedelta(minutes=1) for value in timeline.boundaries),
        timeline.allowed,
        timeline.georef,
    )

    with pytest.raises(ls.TrajectoryInputError) as captured:
        exact_dynamic_path(problem, shifted, shifted.boundaries[0], travel_model=compiled)
    assert captured.value.code == "trajectory_dynamic_mobility_mismatch"


def test_compiled_arrays_are_owned_and_read_only(make_trajectory_georef) -> None:
    problem, timeline = _problem_and_timeline(make_trajectory_georef)
    factors = np.ones((3, 4), dtype=np.float64)
    compiled = compile_dynamic_travel_model(
        problem, timeline, interval_edge_factors=factors
    )
    factors[:] = 2.0

    assert np.all(compiled.interval_factors == 1.0)
    assert not compiled.interval_factors.flags.writeable
    assert not compiled.base_duration_hours.flags.writeable


def test_finite_dynamic_factor_overflow_is_rejected(make_trajectory_georef) -> None:
    problem, timeline = _problem_and_timeline(make_trajectory_georef)
    factors = np.full((3, 4), np.finfo(np.float64).max)

    with pytest.raises(ls.TrajectoryInputError) as captured:
        compile_dynamic_travel_model(
            problem,
            timeline,
            interval_edge_factors=factors,
            hazard_factors=np.full((1, 2), 2.0),
        )
    assert captured.value.code == "trajectory_dynamic_mobility_overflow"


def _dem_for_problem(problem):
    from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters

    elevation = (
        np.zeros(problem.available.shape, dtype=np.float32)
        if problem.elevation_m is None
        else np.asarray(problem.elevation_m, dtype=np.float32)
    )
    return DemGrid(
        np.ascontiguousarray(elevation),
        np.asarray(problem.georef.affine_transform, dtype=np.float64),
        ProjectionParameters(
            radius_m=1_737_400.0,
            latitude_origin_rad=-np.pi / 2.0,
            longitude_origin_rad=0.0,
            scale=1.0,
            false_easting_m=0.0,
            false_northing_m=0.0,
        ),
    )


def _observer_position(dem, y, x):
    from lunarscout._numba_horizon.psr import _pixel_frame

    rotation, translation = _pixel_frame(dem, y, x)
    return -translation @ rotation.T


def test_explicit_sun_vectors_compile_local_direction_factors(
    make_trajectory_georef,
) -> None:
    problem, timeline = _problem_and_timeline(make_trajectory_georef)
    dem = _dem_for_problem(problem)
    source = _observer_position(dem, 0, 0)
    destination = _observer_position(dem, 0, 1)
    velocity = destination - source
    velocity /= np.linalg.norm(velocity)
    distance = 1.0e9
    vectors = ls.trajectory.ExplicitSunVectorProvider(
        timeline.boundaries[:-1],
        np.asarray(
            [
                source - velocity * distance,
                source + velocity * distance,
                source + velocity * distance,
            ]
        ),
    )
    response = SunDirectionFunction((-1.0, 1.0), (2.0, 0.5))

    factors = _compile_sun_direction_factors(
        problem, timeline, dem, problem.georef, vectors, response
    )
    compiled = compile_dynamic_travel_model(
        problem,
        timeline,
        dem=dem,
        dem_georef=problem.georef,
        sun_vectors=vectors,
        sun_direction=response,
    )
    result = exact_dynamic_path(problem, timeline, T0, travel_model=compiled)

    assert factors[:, 0, 0, 0] == pytest.approx([2.0, 0.5, 0.5])
    assert result.reachable
    assert result.departure_times == (T0,)
    assert result.arrival_time == T0 + timedelta(hours=1.25)


@pytest.mark.parametrize(
    ("cosines", "factors"),
    [
        ((-1.0,), (1.0,)),
        ((-1.1, 1.0), (1.0, 1.0)),
        ((-1.0, -1.0), (1.0, 1.0)),
        ((-1.0, 1.0), (1.0, np.inf)),
        ((-1.0, 1.0), (0.0, 1.0)),
    ],
)
def test_sun_direction_function_validation(cosines, factors) -> None:
    with pytest.raises(ls.TrajectoryInputError) as captured:
        SunDirectionFunction(cosines, factors)
    assert captured.value.code == "trajectory_invalid_sun_direction_function"


def test_sun_mobility_requires_complete_inputs(make_trajectory_georef) -> None:
    problem, timeline = _problem_and_timeline(make_trajectory_georef)
    with pytest.raises(ls.TrajectoryInputError) as captured:
        compile_dynamic_travel_model(
            problem,
            timeline,
            dem=_dem_for_problem(problem),
        )
    assert captured.value.code == "trajectory_incomplete_sun_mobility"
