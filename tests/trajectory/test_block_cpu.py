from __future__ import annotations

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.trajectory._block_cpu import (
    block_static_travel_time,
    compile_static_problem,
    reconstruct_block_path,
)
from lunarscout.trajectory._static_reference import dijkstra_field
from lunarscout.trajectory._validation import prepare_static_problem


@pytest.mark.parametrize("block_shape", [(1, 1), (2, 3), (4, 4), (32, 32)])
@pytest.mark.parametrize("include_diagonals", [False, True])
def test_block_field_matches_dijkstra(
    make_trajectory_georef,
    block_shape: tuple[int, int],
    include_diagonals: bool,
) -> None:
    georef = make_trajectory_georef(
        width=11,
        height=9,
        affine=(100.0, 7.0, 1.5, 200.0, -0.75, -11.0),
    )
    generator = np.random.default_rng(4501 + int(include_diagonals))
    traversable = generator.random((9, 11)) > 0.22
    traversable[0, 0] = True
    elevation = generator.normal(0.0, 2.0, size=(9, 11))
    model = ls.trajectory.StaticTravelModel(
        speed_m_per_h=31.0,
        include_diagonals=include_diagonals,
        slip=ls.trajectory.SlipFunction(
            (-3.0, 0.0, 3.0),
            (1.3, 1.0, 2.2),
            extrapolation="constant",
        ),
    )
    problem = prepare_static_problem(
        traversable,
        georef,
        (0, 0),
        elevation=elevation,
        model=model,
    )
    expected = dijkstra_field(problem)
    result = block_static_travel_time(
        problem,
        block_width=block_shape[0],
        block_height=block_shape[1],
    )

    np.testing.assert_array_equal(
        np.isfinite(result.travel_time_hours), np.isfinite(expected)
    )
    np.testing.assert_allclose(result.travel_time_hours, expected, rtol=1e-12)


def test_compiled_problem_is_data_only(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=2, height=1)
    model = ls.trajectory.StaticTravelModel(
        slip=ls.trajectory.SlipFunction((-1.0, 1.0), (0.8, 2.0))
    )
    problem = prepare_static_problem(
        np.ones((1, 2), dtype=bool),
        georef,
        (0, 0),
        elevation=np.asarray([[0.0, 1.0]]),
        model=model,
    )
    compiled = compile_static_problem(problem)

    np.testing.assert_array_equal(compiled.step_dx, [1, 0, -1, 0, 1, -1, -1, 1])
    np.testing.assert_array_equal(compiled.slip_slopes, [-1.0, 1.0])
    np.testing.assert_array_equal(compiled.slip_factors, [0.8, 2.0])
    assert compiled.slip_mode == 1


def test_single_cell_start_block_propagates_seed(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=4, height=1)
    problem = prepare_static_problem(
        np.ones((1, 4), dtype=bool), georef, (0, 0)
    )
    result = block_static_travel_time(problem, block_width=1, block_height=1)

    assert np.all(np.isfinite(result.travel_time_hours))
    assert result.block_visits[0, 1] >= 1


def test_neighbor_blocks_are_reactivated_until_global_costs_converge(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=12, height=8)
    traversable = np.ones((8, 12), dtype=bool)
    traversable[1:7, 3] = False
    traversable[1, 3] = True
    traversable[1:7, 7] = False
    traversable[6, 7] = True
    elevation = np.zeros((8, 12), dtype=np.float64)
    elevation[:, 4:8] = 8.0
    model = ls.trajectory.StaticTravelModel(
        slip=ls.trajectory.SlipFunction(
            (-2.0, 0.0, 2.0),
            (0.7, 1.0, 3.0),
            extrapolation="constant",
        )
    )
    problem = prepare_static_problem(
        traversable,
        georef,
        (0, 7),
        elevation=elevation,
        model=model,
    )
    expected = dijkstra_field(problem)
    result = block_static_travel_time(problem, block_width=3, block_height=2)

    np.testing.assert_allclose(result.travel_time_hours, expected, rtol=1e-12)
    assert np.max(result.block_visits) > 1
    assert np.count_nonzero(result.block_improvements > 1) > 0


def test_block_predecessors_reconstruct_a_valid_optimal_path(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=8, height=6)
    traversable = np.ones((6, 8), dtype=bool)
    traversable[1:5, 4] = False
    traversable[3, 4] = True
    problem = prepare_static_problem(traversable, georef, (0, 0), goal=(7, 5))
    result = block_static_travel_time(problem, block_width=3, block_height=2)
    path = reconstruct_block_path(result, problem.start, problem.goal)

    assert path is not None
    np.testing.assert_array_equal(path[0], problem.start)
    np.testing.assert_array_equal(path[-1], problem.goal)
    for left, right in zip(path, path[1:]):
        dx, dy = right - left
        assert abs(dx) <= 1 and abs(dy) <= 1 and (dx != 0 or dy != 0)
        assert traversable[right[1], right[0]]
    reference = dijkstra_field(problem)
    assert result.travel_time_hours[5, 7] == pytest.approx(reference[5, 7])


def test_predecessors_can_be_omitted_for_field_only_execution(
    make_trajectory_georef,
) -> None:
    georef = make_trajectory_georef(width=4, height=3)
    problem = prepare_static_problem(
        np.ones((3, 4), dtype=bool), georef, (0, 0), goal=(3, 2)
    )
    result = block_static_travel_time(
        problem,
        block_width=2,
        block_height=2,
        track_predecessors=False,
    )

    assert result.predecessor_x is None
    assert result.predecessor_y is None
    with pytest.raises(ls.PlanningError) as captured:
        reconstruct_block_path(result, problem.start, problem.goal)
    assert captured.value.code == "trajectory_predecessors_unavailable"


def test_public_large_field_dispatches_to_block_engine(
    make_trajectory_georef,
    monkeypatch,
) -> None:
    from lunarscout.trajectory import static as static_module

    size = 128
    georef = make_trajectory_georef(width=size, height=size)

    def unexpected_reference(_problem):
        raise AssertionError("large field used reference Dijkstra")

    monkeypatch.setattr(static_module, "dijkstra_field", unexpected_reference)
    result = ls.trajectory.static_travel_time(
        np.ones((size, size), dtype=bool), georef, (0, 0)
    )

    assert result.reached.all()
    expected = (size - 1) * np.sqrt(2.0) * 10.0 / 36.0
    assert result.travel_time_hours[-1, -1] == pytest.approx(expected)


def test_public_below_threshold_field_retains_reference_engine(
    make_trajectory_georef,
    monkeypatch,
) -> None:
    from lunarscout.trajectory import _block_cpu

    width = 127
    height = 129  # 16,383 cells: one below the private dispatch threshold.
    georef = make_trajectory_georef(width=width, height=height)

    def unexpected_block(*_args, **_kwargs):
        raise AssertionError("small field used block relaxation")

    monkeypatch.setattr(_block_cpu, "block_static_travel_time", unexpected_block)
    result = ls.trajectory.static_travel_time(
        np.ones((height, width), dtype=bool), georef, (0, 0)
    )

    assert result.reached.all()


@pytest.mark.parametrize(
    ("name", "value"),
    [("block_width", 0), ("block_height", -1), ("block_width", False)],
)
def test_invalid_private_block_geometry_is_structured(
    make_trajectory_georef,
    name: str,
    value: int,
) -> None:
    georef = make_trajectory_georef(width=2, height=2)
    problem = prepare_static_problem(
        np.ones((2, 2), dtype=bool), georef, (0, 0)
    )
    arguments = {"block_width": 2, "block_height": 2}
    arguments[name] = value
    with pytest.raises(ls.TrajectoryInputError) as captured:
        block_static_travel_time(problem, **arguments)

    assert captured.value.code == "trajectory_invalid_block_geometry"


def test_block_activation_limit_is_structured(make_trajectory_georef) -> None:
    georef = make_trajectory_georef(width=4, height=1)
    problem = prepare_static_problem(
        np.ones((1, 4), dtype=bool), georef, (0, 0)
    )
    with pytest.raises(ls.PlanningError) as captured:
        block_static_travel_time(
            problem,
            block_width=1,
            block_height=1,
            maximum_activations=1,
        )

    assert captured.value.code == "trajectory_block_activation_limit"
