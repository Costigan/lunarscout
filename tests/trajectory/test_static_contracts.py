from __future__ import annotations

import numpy as np
import pytest
from pyproj import CRS

import lunarscout as ls
from lunarscout.trajectory._geometry import linear_units_to_metres, neighbor_steps


def test_default_model_and_slip_contract() -> None:
    model = ls.trajectory.StaticTravelModel()
    assert model.speed_m_per_h == 36.0
    assert model.include_diagonals is True
    assert model.slip is None

    slip = ls.trajectory.SlipFunction(
        signed_slopes=(-0.5, 0.0, 0.5),
        factors=(2.0, 1.0, 3.0),
    )
    assert slip.factor(-0.25) == pytest.approx(1.5)
    assert slip.factor(0.25) == pytest.approx(2.0)
    assert slip.factor(-0.6) == np.inf
    assert slip.factor(0.6) == np.inf
    assert slip.minimum_factor == 1.0


def test_constant_slip_extrapolation_uses_endpoint_factors() -> None:
    slip = ls.trajectory.SlipFunction(
        (-0.25, 0.25),
        (1.5, 2.5),
        extrapolation="constant",
    )
    assert slip.factor(-1.0) == 1.5
    assert slip.factor(1.0) == 2.5


@pytest.mark.parametrize(
    ("slopes", "factors"),
    [
        ((0.0,), (1.0,)),
        ((0.0, 1.0), (1.0,)),
        ((0.0, 0.0), (1.0, 2.0)),
        ((1.0, 0.0), (1.0, 2.0)),
        ((0.0, np.nan), (1.0, 2.0)),
        ((0.0, 1.0), (1.0, 0.0)),
    ],
)
def test_invalid_slip_tables_raise_structured_error(slopes, factors) -> None:
    with pytest.raises(ls.TrajectoryInputError) as captured:
        ls.trajectory.SlipFunction(slopes, factors)

    assert captured.value.code == "trajectory_invalid_slip_function"


@pytest.mark.parametrize("speed", [False, 0.0, -1.0, np.nan, np.inf])
def test_invalid_speed_raises_structured_error(speed: float) -> None:
    with pytest.raises(ls.TrajectoryInputError) as captured:
        ls.trajectory.StaticTravelModel(speed_m_per_h=speed)

    assert captured.value.code == "trajectory_invalid_speed"


def test_affine_neighbor_distances_use_both_basis_vectors(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(
        width=2,
        height=2,
        affine=(0.0, 3.0, 1.0, 0.0, 4.0, -2.0),
    )
    steps = neighbor_steps(
        georef,
        include_diagonals=True,
        units_to_metres=linear_units_to_metres(georef),
    )
    distances = {(step.dx, step.dy): step.distance_m for step in steps}

    assert distances[(1, 0)] == pytest.approx(5.0)
    assert distances[(0, 1)] == pytest.approx(np.sqrt(5.0))
    assert distances[(1, 1)] == pytest.approx(np.sqrt(20.0))
    assert distances[(-1, 1)] == pytest.approx(np.sqrt(40.0))


def test_projected_foot_units_are_converted_to_metres(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(
        width=2,
        height=1,
        affine=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        crs_input="EPSG:2277",
    )
    result = ls.trajectory.static_path(
        np.ones((1, 2), dtype=bool),
        georef,
        (0, 0),
        (1, 0),
        model=ls.trajectory.StaticTravelModel(speed_m_per_h=1.0),
    )

    assert result.travel_time_hours == pytest.approx(0.3048006096012192)


def test_geographic_crs_is_rejected(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(
        width=2,
        height=1,
        affine=(0.0, 0.01, 0.0, 0.0, 0.0, -0.01),
        crs_input="EPSG:4326",
    )

    with pytest.raises(ls.TrajectoryInputError) as captured:
        ls.trajectory.static_travel_time(np.ones((1, 2), dtype=bool), georef, (0, 0))

    assert captured.value.code == "trajectory_crs_not_projected"


def test_lonlat_selects_containing_cell_and_outer_bound_is_excluded(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=3, height=2)
    easting, northing = georef.pixel_to_projected(1.75, 0.25, anchor="corner")
    longitude, latitude = georef.projected_to_lonlat(easting, northing)
    point = ls.LonLat(longitude, latitude)
    result = ls.trajectory.static_travel_time(
        np.ones((2, 3), dtype=bool), georef, point
    )
    assert result.start == (1, 0)

    outer_easting, outer_northing = georef.pixel_to_projected(
        georef.width, 0.5, anchor="corner"
    )
    outer_lon, outer_lat = georef.projected_to_lonlat(outer_easting, outer_northing)
    with pytest.raises(ls.TrajectoryInputError) as captured:
        ls.trajectory.static_travel_time(
            np.ones((2, 3), dtype=bool),
            georef,
            ls.LonLat(outer_lon, outer_lat),
        )
    assert captured.value.code == "trajectory_cell_out_of_bounds"


@pytest.mark.parametrize(
    ("column", "row", "expected"),
    [
        (0.0, 0.0, (0, 0)),
        (3.0, 0.5, None),
        (0.5, 2.0, None),
        (-0.001, 0.5, None),
        (0.5, -0.001, None),
    ],
)
def test_lonlat_half_open_boundaries(
    make_trajectory_georef,
    column: float,
    row: float,
    expected: tuple[int, int] | None,
) -> None:
    georef = make_trajectory_georef(
        width=3,
        height=2,
        affine=(-1000.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
    )
    easting, northing = georef.pixel_to_projected(column, row, anchor="corner")
    longitude, latitude = georef.projected_to_lonlat(easting, northing)
    point = ls.LonLat(longitude, latitude)
    if expected is None:
        with pytest.raises(ls.TrajectoryInputError) as captured:
            ls.trajectory.static_travel_time(
                np.ones((2, 3), dtype=bool), georef, point
            )
        assert captured.value.code == "trajectory_cell_out_of_bounds"
    else:
        result = ls.trajectory.static_travel_time(
            np.ones((2, 3), dtype=bool), georef, point
        )
        assert result.start == expected


def test_invalid_and_nontraversable_cells_are_distinct_but_unavailable(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=2, height=1)
    with pytest.raises(ls.TrajectoryInputError) as invalid:
        ls.trajectory.static_path(
            np.ones((1, 2), dtype=bool),
            georef,
            (0, 0),
            (1, 0),
            valid=np.asarray([[True, False]]),
        )
    with pytest.raises(ls.TrajectoryInputError) as blocked:
        ls.trajectory.static_path(
            np.asarray([[1, 0]], dtype=np.uint8), georef, (0, 0), (1, 0)
        )

    assert invalid.value.code == "trajectory_cell_unavailable"
    assert blocked.value.code == "trajectory_cell_unavailable"


def test_nonfinite_elevation_is_allowed_only_at_unavailable_cells(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=2, height=1)
    model = ls.trajectory.StaticTravelModel(
        slip=ls.trajectory.SlipFunction((-1.0, 1.0), (1.0, 1.0))
    )
    with pytest.raises(ls.TrajectoryInputError) as captured:
        ls.trajectory.static_travel_time(
            np.ones((1, 2), dtype=bool),
            georef,
            (0, 0),
            elevation=np.asarray([[0.0, np.nan]]),
            model=model,
        )
    assert captured.value.code == "trajectory_nonfinite_elevation"

    with pytest.raises(ls.TrajectoryInputError) as unavailable:
        ls.trajectory.static_path(
            np.asarray([[1, 0]], dtype=np.uint8),
            georef,
            (0, 0),
            (1, 0),
            elevation=np.asarray([[0.0, np.nan]]),
            model=model,
        )
    assert unavailable.value.code == "trajectory_cell_unavailable"


def test_grid_shapes_and_mask_dtypes_are_validated(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=2, height=2)
    with pytest.raises(ls.TrajectoryInputError) as shape:
        ls.trajectory.static_travel_time(np.ones((2, 3), dtype=bool), georef, (0, 0))
    with pytest.raises(ls.TrajectoryInputError) as mask:
        ls.trajectory.static_travel_time(
            np.ones((2, 2), dtype=bool),
            georef,
            (0, 0),
            valid=np.ones((2, 2), dtype=np.uint8),
        )

    assert shape.value.code == "trajectory_grid_shape_mismatch"
    assert mask.value.code == "trajectory_invalid_valid_dtype"


def test_slip_requires_elevation_and_cell_coordinates_are_exact(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=2, height=2)
    model = ls.trajectory.StaticTravelModel(
        slip=ls.trajectory.SlipFunction((-1.0, 1.0), (1.0, 1.0))
    )
    with pytest.raises(ls.TrajectoryInputError) as elevation:
        ls.trajectory.static_travel_time(
            np.ones((2, 2), dtype=bool), georef, (0, 0), model=model
        )
    with pytest.raises(ls.TrajectoryInputError) as fractional:
        ls.trajectory.static_travel_time(
            np.ones((2, 2), dtype=bool), georef, (0.0, 0)
        )

    assert elevation.value.code == "trajectory_elevation_required"
    assert fractional.value.code == "trajectory_invalid_cell"


def test_all_invalid_and_invalid_start_equals_goal_are_rejected(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=2, height=2)
    traversable = np.ones((2, 2), dtype=np.uint64)
    valid = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ls.TrajectoryInputError) as field:
        ls.trajectory.static_travel_time(traversable, georef, (0, 0), valid=valid)
    with pytest.raises(ls.TrajectoryInputError) as path:
        ls.trajectory.static_path(
            traversable, georef, (0, 0), (0, 0), valid=valid
        )

    assert field.value.code == "trajectory_cell_unavailable"
    assert path.value.code == "trajectory_cell_unavailable"


def test_uint64_traversability_is_not_converted_through_float(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=2, height=1)
    traversable = np.asarray([[np.iinfo(np.uint64).max, 1]], dtype=np.uint64)
    result = ls.trajectory.static_path(traversable, georef, (0, 0), (1, 0))

    assert result.reachable is True


def test_result_values_copy_and_validate_their_arrays(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=2, height=1)
    source_path = np.asarray([[0, 0], [1, 0]], dtype=np.int32)
    path = ls.trajectory.PathResult(True, 1.0, source_path)
    source_path[0, 0] = 99
    np.testing.assert_array_equal(path.path, [[0, 0], [1, 0]])
    assert path.path is not None and path.path.dtype == np.int64
    assert not path.path.flags.writeable

    times = np.asarray([[0.0, np.inf]])
    reached = np.asarray([[True, False]])
    field = ls.trajectory.TravelTimeResult(times, reached, georef, (0, 0))
    times[0, 0] = 2.0
    assert field.travel_time_hours[0, 0] == 0.0
    assert not field.travel_time_hours.flags.writeable

    with pytest.raises(ls.TrajectoryInputError) as captured:
        ls.trajectory.PathResult(False, 1.0, None)
    assert captured.value.code == "trajectory_invalid_path_result"

    with pytest.raises(ls.TrajectoryInputError) as inconsistent:
        ls.trajectory.TravelTimeResult(
            np.asarray([[1.0, np.inf]]), reached, georef, (0, 0)
        )
    assert inconsistent.value.code == "trajectory_invalid_travel_time_result"
