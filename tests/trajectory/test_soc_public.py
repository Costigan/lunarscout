from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls
from lunarscout.trajectory._environment import sunlight_timeline_from_provider


T0 = datetime(2042, 1, 1, tzinfo=timezone.utc)


def _models(
    *,
    capacity=500.0,
    initial=300.0,
    minimum=0.0,
    solar=100.0,
    drive=100.0,
    idle=0.0,
):
    return {
        "solar": ls.trajectory.SolarPowerModel(solar),
        "battery": ls.trajectory.BatteryModel(
            capacity, initial, minimum, 1.0, 1.0
        ),
        "rover": ls.trajectory.RoverPowerModel(drive, idle),
    }


def _public_case(make_trajectory_georef):
    grid = make_trajectory_georef(width=3, height=1)
    boundaries = tuple(T0 + timedelta(hours=index) for index in range(4))
    configuration = ls.trajectory.StaticConfigurationSpaceProvider(
        np.ones((1, 3), dtype=bool), grid
    )
    sunlight = ls.trajectory.ArraySunlightProvider(
        boundaries,
        np.asarray([[[255, 0, 0]]] * 3, dtype=np.uint8),
        grid,
    )
    arguments = (
        np.ones((1, 3), dtype=bool),
        grid,
        (0, 0),
        (2, 0),
        boundaries,
        configuration,
        sunlight,
        T0,
    )
    return grid, boundaries, arguments


def test_public_exact_and_greedy_soc_paths_report_algorithm_guarantees(
    make_trajectory_georef,
) -> None:
    _grid, _boundaries, arguments = _public_case(make_trajectory_georef)
    model = ls.trajectory.StaticTravelModel(
        speed_m_per_h=20.0, include_diagonals=False
    )
    models = _models(initial=250.0, drive=200.0, idle=50.0)
    exact = ls.trajectory.soc_path(
        *arguments, model=model, algorithm="exact", backend="cpu", **models
    )
    greedy = ls.trajectory.soc_path(
        *arguments, model=model, algorithm="greedy", backend="auto", **models
    )
    stationary = ls.trajectory.soc_path(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[2],
        *arguments[4:],
        model=model,
        algorithm="exact",
        backend="cpu",
        **models,
    )

    assert isinstance(exact, ls.trajectory.SocPathResult)
    assert exact.reachable and greedy.reachable
    assert exact.cells is not None
    assert exact.cells.tolist() == [[0, 0], [1, 0], [2, 0]]
    assert not exact.cells.flags.writeable
    assert exact.arrival_time == T0 + timedelta(hours=1)
    assert exact.final_energy_wh == 100.0
    assert exact.algorithm == "exact" and exact.backend == "cpu"
    assert exact.complete and exact.optimal
    assert greedy.algorithm == "greedy" and greedy.backend == "cpu"
    assert not greedy.complete and not greedy.optimal
    assert greedy.arrival_time == exact.arrival_time
    assert greedy.final_energy_wh == exact.final_energy_wh
    assert stationary.reachable and stationary.arrival_time == T0
    assert stationary.energy is not None
    assert stationary.energy.segments == ()
    assert stationary.final_energy_wh == models["battery"].initial_energy_wh


def test_public_soc_waits_and_charges_at_environment_boundary(
    make_trajectory_georef,
) -> None:
    grid = make_trajectory_georef(width=2, height=1)
    boundaries = tuple(T0 + timedelta(hours=index) for index in range(4))
    configuration = ls.trajectory.StaticConfigurationSpaceProvider(
        np.ones((1, 2), dtype=bool), grid
    )
    sunlight = ls.trajectory.ArraySunlightProvider(
        boundaries,
        np.asarray([[[255, 0]], [[0, 0]], [[0, 0]]], dtype=np.uint8),
        grid,
    )
    result = ls.trajectory.soc_path(
        np.ones((1, 2), dtype=bool),
        grid,
        (0, 0),
        (1, 0),
        boundaries,
        configuration,
        sunlight,
        T0,
        model=ls.trajectory.StaticTravelModel(
            speed_m_per_h=20.0, include_diagonals=False
        ),
        **_models(
            capacity=300.0,
            initial=100.0,
            minimum=60.0,
            solar=300.0,
            drive=400.0,
            idle=100.0,
        ),
    )

    assert result.reachable
    assert result.departure_times == (T0 + timedelta(hours=1),)
    assert result.wait_intervals == ((T0, T0 + timedelta(hours=1)),)
    assert result.energy is not None
    assert [segment.mode for segment in result.energy.segments] == [
        "idle",
        "drive",
    ]
    assert result.final_energy_wh == 100.0


def test_public_exact_exposes_greedy_false_negative_counterexample(
    make_trajectory_georef,
) -> None:
    grid = make_trajectory_georef(width=3, height=2)
    boundaries = (T0, T0 + timedelta(hours=4))
    available = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=bool)
    configuration = ls.trajectory.StaticConfigurationSpaceProvider(available, grid)
    sunlight_values = np.zeros((1, 2, 3), dtype=np.uint8)
    sunlight_values[:, 1, :2] = 255
    sunlight = ls.trajectory.ArraySunlightProvider(
        boundaries, sunlight_values, grid
    )
    arguments = (
        available,
        grid,
        (0, 0),
        (2, 0),
        boundaries,
        configuration,
        sunlight,
        T0,
    )
    options = {
        "model": ls.trajectory.StaticTravelModel(
            speed_m_per_h=20.0, include_diagonals=False
        ),
        **_models(
            capacity=300.0,
            initial=300.0,
            solar=500.0,
            drive=400.0,
        ),
    }
    exact = ls.trajectory.soc_path(*arguments, algorithm="exact", **options)
    greedy = ls.trajectory.soc_path(*arguments, algorithm="greedy", **options)

    assert exact.reachable and not greedy.reachable
    assert exact.cells is not None
    assert exact.cells.tolist() == [[0, 0], [0, 1], [1, 1], [1, 0], [2, 0]]
    assert exact.complete and not greedy.complete


def test_soc_dispatch_and_bounds_fail_before_provider_reads(
    make_trajectory_georef, monkeypatch
) -> None:
    from lunarscout.trajectory import soc as soc_module

    grid = make_trajectory_georef(width=1, height=1)

    class CountingProvider:
        georef = grid

        def __init__(self, dtype):
            self.dtype = dtype
            self.calls = 0

        def read(self, x0, y0, width, height, time):
            self.calls += 1
            return np.ones((height, width), dtype=self.dtype)

    configuration = CountingProvider(bool)
    sunlight = CountingProvider(np.uint8)
    arguments = (
        np.ones((1, 1), dtype=bool),
        grid,
        (0, 0),
        (0, 0),
        (T0, T0 + timedelta(hours=1)),
        configuration,
        sunlight,
        T0,
    )
    with pytest.raises(ls.TrajectoryInputError) as algorithm:
        ls.trajectory.soc_path(*arguments, algorithm="other", **_models())
    with pytest.raises(ls.TrajectoryInputError) as backend:
        ls.trajectory.soc_path(*arguments, backend="other", **_models())
    with pytest.raises(ls.PlanningError) as cuda:
        ls.trajectory.soc_path(*arguments, backend="cuda", **_models())
    with pytest.raises(ls.TrajectoryInputError) as labels:
        ls.trajectory.soc_path(*arguments, max_labels=True, **_models())
    monkeypatch.setattr(soc_module, "_MAX_SOC_TIMELINE_STATES", 0)
    with pytest.raises(ls.PlanningError) as states:
        ls.trajectory.soc_path(*arguments, **_models())

    assert algorithm.value.code == "trajectory_unknown_algorithm"
    assert backend.value.code == "trajectory_unknown_backend"
    assert cuda.value.code == "trajectory_backend_unavailable"
    assert labels.value.code == "trajectory_invalid_soc_label_limit"
    assert states.value.code == "trajectory_soc_timeline_state_limit"
    assert configuration.calls == sunlight.calls == 0


def test_soc_provider_grid_and_result_validation(make_trajectory_georef) -> None:
    grid = make_trajectory_georef(width=1, height=1)
    other = make_trajectory_georef(
        width=1,
        height=1,
        affine=(1.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    )
    boundaries = (T0, T0 + timedelta(hours=1))
    configuration = ls.trajectory.StaticConfigurationSpaceProvider(
        np.ones((1, 1), dtype=bool), grid
    )
    sunlight = ls.trajectory.ArraySunlightProvider(
        boundaries, np.ones((1, 1, 1), dtype=np.uint8), other
    )
    with pytest.raises(ls.TrajectoryInputError) as mismatch:
        ls.trajectory.soc_path(
            np.ones((1, 1), dtype=bool),
            grid,
            (0, 0),
            (0, 0),
            boundaries,
            configuration,
            sunlight,
            T0,
            **_models(),
        )

    dynamic = ls.trajectory.DynamicPathResult(False, None, None, None, None)
    segment = ls.trajectory.EnergySegment(
        "idle",
        T0,
        T0 + timedelta(hours=1),
        (0, 0),
        1.0,
        100.0,
        100.0,
        10.0,
        10.0,
        0.0,
    )
    energy = ls.trajectory.EnergyTimelineResult(True, 100.0, 100.0, (segment,))
    with pytest.raises(ls.TrajectoryInputError) as invalid:
        ls.trajectory.SocPathResult(dynamic, energy, "greedy", "cpu")

    reachable = ls.trajectory.DynamicPathResult(
        True,
        T0 + timedelta(hours=1),
        np.asarray([[0, 0], [1, 0]], dtype=np.int64),
        (T0, T0 + timedelta(hours=1)),
        (T0,),
    )
    wrong_event = ls.trajectory.EnergySegment(
        "drive",
        T0,
        T0 + timedelta(hours=1),
        (1, 0),
        1.0,
        100.0,
        100.0,
        10.0,
        10.0,
        0.0,
    )
    wrong_energy = ls.trajectory.EnergyTimelineResult(
        True, 100.0, 100.0, (wrong_event,)
    )
    with pytest.raises(ls.TrajectoryInputError) as mismatch_result:
        ls.trajectory.SocPathResult(
            reachable, wrong_energy, "greedy", "cpu"
        )

    assert mismatch.value.code == "trajectory_provider_grid_mismatch"
    assert invalid.value.code == "trajectory_invalid_soc_result"
    assert mismatch_result.value.code == "trajectory_invalid_soc_result"


def test_public_exact_label_exhaustion_is_not_unreachable(
    make_trajectory_georef,
) -> None:
    _grid, _boundaries, arguments = _public_case(make_trajectory_georef)
    with pytest.raises(ls.PlanningError) as error:
        ls.trajectory.soc_path(
            *arguments,
            algorithm="exact",
            max_labels=1,
            model=ls.trajectory.StaticTravelModel(
                speed_m_per_h=20.0, include_diagonals=False
            ),
            **_models(),
        )
    assert error.value.code == "trajectory_exact_soc_label_limit"


def test_sunlight_provider_bytes_materialize_as_owned_fractions(
    make_trajectory_georef,
) -> None:
    grid = make_trajectory_georef(width=3, height=1)
    boundaries = (T0, T0 + timedelta(hours=1))
    values = np.asarray([[[0, 128, 255]]], dtype=np.uint8)
    provider = ls.trajectory.ArraySunlightProvider(boundaries, values, grid)
    timeline = sunlight_timeline_from_provider(provider, boundaries)
    values[:] = 0

    assert timeline.fractions.tolist() == [[[0.0, 128.0 / 255.0, 1.0]]]
    assert not timeline.fractions.flags.writeable


def test_invalid_sunlight_provider_result_is_structured(
    make_trajectory_georef,
) -> None:
    grid = make_trajectory_georef(width=1, height=1)
    boundaries = (T0, T0 + timedelta(hours=1))
    configuration = ls.trajectory.StaticConfigurationSpaceProvider(
        np.ones((1, 1), dtype=bool), grid
    )

    class InvalidSunlight:
        georef = grid

        def read(self, x0, y0, width, height, time):
            return np.ones((height, width), dtype=np.float64)

    with pytest.raises(ls.ConfigurationSpaceError) as error:
        ls.trajectory.soc_path(
            np.ones((1, 1), dtype=bool),
            grid,
            (0, 0),
            (0, 0),
            boundaries,
            configuration,
            InvalidSunlight(),
            T0,
            **_models(),
        )
    assert error.value.code == "trajectory_invalid_provider_result"
    assert error.value.details["signal"] == "sunlight"
