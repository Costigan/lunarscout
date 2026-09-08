from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import lunarscout as ls


T0 = datetime(2035, 1, 1, tzinfo=timezone.utc)


def _boundaries(count: int = 2) -> tuple[datetime, ...]:
    return tuple(T0 + timedelta(hours=index) for index in range(count + 1))


def test_provider_protocols_are_structural(make_trajectory_georef) -> None:
    grid = make_trajectory_georef(width=3, height=2)
    sunlight = ls.trajectory.ArraySunlightProvider(
        _boundaries(), np.zeros((2, 2, 3), dtype=np.uint8), grid
    )
    earth = ls.trajectory.ArrayEarthElevationProvider(
        _boundaries(), np.zeros((2, 2, 3), dtype=np.float32), grid
    )
    occupancy = ls.trajectory.StaticConfigurationSpaceProvider(
        np.ones((2, 3), dtype=bool), grid
    )
    vectors = ls.trajectory.ExplicitSunVectorProvider(
        (T0,), np.asarray([[1.0, 2.0, 3.0]])
    )

    assert isinstance(sunlight, ls.trajectory.SunlightProvider)
    assert isinstance(earth, ls.trajectory.EarthElevationProvider)
    assert isinstance(occupancy, ls.trajectory.ConfigurationSpaceProvider)
    assert isinstance(vectors, ls.trajectory.SunVectorProvider)


def test_interval_signal_partial_windows_and_half_open_lookup(
    make_trajectory_georef,
) -> None:
    grid = make_trajectory_georef(width=4, height=3)
    values = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    provider = ls.trajectory.ArraySunlightProvider(_boundaries(), values, grid)

    first = provider.read(1, 1, 2, 2, T0 + timedelta(minutes=59))
    second = provider.read(1, 1, 2, 2, T0 + timedelta(hours=1))

    assert np.array_equal(first, values[0, 1:3, 1:3])
    assert np.array_equal(second, values[1, 1:3, 1:3])
    assert first.dtype == np.uint8
    assert first.flags.writeable is False
    with pytest.raises(ls.TrajectoryInputError) as final:
        provider.read(0, 0, 1, 1, T0 + timedelta(hours=2))
    assert final.value.code == "trajectory_provider_time_out_of_range"


def test_signal_batch_matches_scalar_and_results_are_isolated(
    make_trajectory_georef,
) -> None:
    grid = make_trajectory_georef(width=2, height=1)
    values = np.asarray([[[1, 2]], [[3, 4]]], dtype=np.uint8)
    provider = ls.trajectory.ArraySunlightProvider(_boundaries(), values, grid)
    times = (T0, T0 + timedelta(hours=1))

    batch = provider.read_many(0, 0, 2, 1, times)
    scalar = np.stack([provider.read(0, 0, 2, 1, value) for value in times])

    assert np.array_equal(batch, scalar)
    assert batch.flags.writeable is False
    original = provider.read(0, 0, 2, 1, T0)
    with pytest.raises(ValueError):
        original[0, 0] = 99
    assert provider.read(0, 0, 2, 1, T0).tolist() == [[1, 2]]


@pytest.mark.parametrize(
    ("factory", "values"),
    [
        (ls.trajectory.ArraySunlightProvider, np.zeros((2, 1, 2), dtype=np.float32)),
        (
            ls.trajectory.ArrayEarthElevationProvider,
            np.asarray([[[0.0, np.nan]], [[0.0, 1.0]]], dtype=np.float32),
        ),
    ],
)
def test_signal_provider_rejects_wrong_dtype_or_nonfinite(
    make_trajectory_georef, factory, values
) -> None:
    grid = make_trajectory_georef(width=2, height=1)
    with pytest.raises(ls.TrajectoryInputError) as captured:
        factory(_boundaries(), values, grid)
    assert captured.value.code == "trajectory_invalid_provider_data"


def test_explicit_vectors_are_exact_copied_and_nonzero() -> None:
    source = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    provider = ls.trajectory.ExplicitSunVectorProvider(
        (T0, T0 + timedelta(hours=1)), source
    )
    source[:] = 9.0

    result = provider.vectors((T0 + timedelta(hours=1), T0))
    assert result.tolist() == [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]
    assert result.dtype == np.float64
    assert result.flags.c_contiguous and not result.flags.writeable
    with pytest.raises(ls.TrajectoryInputError) as missing:
        provider.vectors((T0 + timedelta(hours=2),))
    assert missing.value.code == "trajectory_sun_vector_time_not_found"
    with pytest.raises(ls.TrajectoryInputError) as zero:
        ls.trajectory.ExplicitSunVectorProvider((T0,), np.zeros((1, 3)))
    assert zero.value.code == "trajectory_invalid_sun_vectors"
    with pytest.raises(ls.TrajectoryInputError) as naive:
        provider.vectors((datetime(2035, 1, 1),))
    assert naive.value.code == "trajectory_invalid_dynamic_time"


def test_spice_provider_is_lazy_cached_and_closable(monkeypatch) -> None:
    from lunarscout import spice_geometry

    calls: list[tuple[datetime, ...]] = []

    def fake(_body, times, *, ensure_kernels):
        calls.append(tuple(times))
        return np.tile(np.asarray([[1.0, 2.0, 3.0]]), (len(times), 1))

    monkeypatch.setattr(spice_geometry, "body_vectors_moon_me", fake)
    provider = ls.trajectory.SpiceSunVectorProvider(cache_entries=1)
    assert calls == []
    assert provider.vectors((T0, T0)).shape == (2, 3)
    assert provider.vectors((T0,)).shape == (1, 3)
    assert len(calls) == 1
    provider.vectors((T0 + timedelta(hours=1),))
    provider.vectors((T0,))
    assert len(calls) == 3
    provider.close()
    with pytest.raises(ls.PlanningError) as closed:
        provider.vectors((T0,))
    assert closed.value.code == "trajectory_provider_closed"


def test_horizon_sunlight_batches_vectors_and_reuses_each_tile(
    make_trajectory_georef, monkeypatch, tmp_path
) -> None:
    from lunarscout import products
    from lunarscout._numba_horizon import file_format, lightmap_cpu
    from lunarscout.trajectory import providers as provider_module

    grid = make_trajectory_georef(width=130, height=2)

    class Dem:
        width = 130
        height = 2

    monkeypatch.setattr(products, "_load_dem", lambda _path: (Dem(), grid))
    monkeypatch.setattr(provider_module, "_HORIZON_SHAPE", (2, 2, 3))
    horizon_reads: list[tuple[int, int]] = []

    def fake_horizon_read(_self, tile_y, tile_x, _observer):
        horizon_reads.append((tile_y, tile_x))
        return np.zeros((2, 2, 3), dtype=np.float32)

    monkeypatch.setattr(file_format.HorizonTileStore, "read", fake_horizon_read)
    calculations: list[tuple[int, int, int]] = []

    class FakeSession:
        def __init__(self, *, time_batch_size):
            assert time_batch_size == 4

        def iter_patch_tiles(
            self,
            _dem,
            _horizons,
            vectors,
            *,
            tile_y,
            tile_x,
            valid_height,
            valid_width,
        ):
            calculations.append((tile_y, tile_x, len(vectors)))
            for vector in vectors:
                value = int(vector[0]) * 10 + tile_x // 128
                yield np.full((valid_height, valid_width), value, dtype=np.uint8)

    monkeypatch.setattr(lightmap_cpu, "LightmapCpuSession", FakeSession)
    vectors = ls.trajectory.ExplicitSunVectorProvider(
        (T0, T0 + timedelta(hours=1)),
        np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
    )
    horizons = tmp_path / "horizons"
    horizons.mkdir()
    provider = ls.trajectory.HorizonSunlightProvider(
        "dem.tif",
        horizons,
        vectors,
        horizon_cache_entries=2,
        time_batch_size=4,
    )

    batch = provider.read_many(
        127, 0, 3, 2, (T0, T0 + timedelta(hours=1))
    )
    repeated = provider.read(127, 0, 3, 2, T0)

    assert batch.tolist() == [
        [[10, 11, 11], [10, 11, 11]],
        [[20, 21, 21], [20, 21, 21]],
    ]
    assert np.array_equal(repeated, batch[0])
    assert horizon_reads == [(0, 0), (0, 128)]
    assert calculations == [(0, 0, 2), (0, 128, 2)]
    provider.close()
    assert provider.sun_vectors is vectors
    with pytest.raises(ls.ConfigurationSpaceError) as closed:
        provider.read(127, 0, 3, 2, T0)
    assert closed.value.code == "trajectory_provider_closed"


def test_horizon_sunlight_missing_tile_is_structured(
    make_trajectory_georef, monkeypatch, tmp_path
) -> None:
    from lunarscout import products
    from lunarscout._numba_horizon import file_format, lightmap_cpu

    grid = make_trajectory_georef(width=1, height=1)

    class Dem:
        width = 1
        height = 1

    class FakeSession:
        def __init__(self, *, time_batch_size):
            pass

    monkeypatch.setattr(products, "_load_dem", lambda _path: (Dem(), grid))
    monkeypatch.setattr(file_format.HorizonTileStore, "read", lambda *_args: None)
    monkeypatch.setattr(lightmap_cpu, "LightmapCpuSession", FakeSession)
    vectors = ls.trajectory.ExplicitSunVectorProvider(
        (T0,), np.asarray([[1.0, 1.0, 1.0]])
    )
    horizons = tmp_path / "horizons"
    horizons.mkdir()
    provider = ls.trajectory.HorizonSunlightProvider("dem.tif", horizons, vectors)

    with pytest.raises(ls.ConfigurationSpaceError) as captured:
        provider.read(0, 0, 1, 1, T0)
    assert captured.value.code == "trajectory_horizon_missing"


def test_thresholds_and_composition_are_inclusive(make_trajectory_georef) -> None:
    grid = make_trajectory_georef(width=3, height=1)
    sunlight = ls.trajectory.ArraySunlightProvider(
        _boundaries(1), np.asarray([[[0, 128, 255]]], dtype=np.uint8), grid
    )
    earth = ls.trajectory.ArrayEarthElevationProvider(
        _boundaries(1), np.asarray([[[1.9, 2.0, 3.0]]], dtype=np.float32), grid
    )
    static = ls.trajectory.StaticConfigurationSpaceProvider(
        np.asarray([[True, True, False]]), grid
    )
    sun_rule = ls.trajectory.SunlightThresholdProvider(sunlight, 128 / 255)
    earth_rule = ls.trajectory.EarthElevationThresholdProvider(earth, 2.0)
    combined = ls.trajectory.AllOfConfigurationSpaceProvider(
        (static, sun_rule, earth_rule)
    )

    assert sun_rule.read(0, 0, 3, 1, T0).tolist() == [[False, True, True]]
    assert earth_rule.read(0, 0, 3, 1, T0).tolist() == [[False, True, True]]
    assert combined.read(0, 0, 3, 1, T0).tolist() == [[False, True, False]]


def test_combined_cache_is_bounded_and_repeated_reads_are_deterministic(
    make_trajectory_georef,
) -> None:
    grid = make_trajectory_georef(width=1, height=1)

    class CountingProvider:
        georef = grid

        def __init__(self):
            self.calls = 0

        def read(self, x0, y0, width, height, time):
            self.calls += 1
            return np.ones((height, width), dtype=bool)

    child = CountingProvider()
    combined = ls.trajectory.AllOfConfigurationSpaceProvider(
        (child,), cache_entries=1
    )
    one = combined.read(0, 0, 1, 1, T0)
    two = combined.read(0, 0, 1, 1, T0)
    combined.read(0, 0, 1, 1, T0 + timedelta(minutes=1))
    combined.read(0, 0, 1, 1, T0)

    assert np.array_equal(one, two)
    assert child.calls == 3
    assert len(combined._cache.values) == 1


def test_provider_result_failures_are_structured(make_trajectory_georef) -> None:
    grid = make_trajectory_georef(width=2, height=1)

    class WrongDtype:
        georef = grid

        def read(self, x0, y0, width, height, time):
            return np.ones((height, width), dtype=np.float32)

    combined = ls.trajectory.AllOfConfigurationSpaceProvider((WrongDtype(),))
    with pytest.raises(ls.ConfigurationSpaceError) as captured:
        combined.read(0, 0, 2, 1, T0)
    assert captured.value.code == "trajectory_invalid_provider_result"


def test_exact_oracle_accepts_materialized_configuration_provider(
    make_trajectory_georef,
) -> None:
    from lunarscout.trajectory._dynamic_reference import exact_dynamic_path
    from lunarscout.trajectory._environment import occupancy_timeline_from_provider
    from lunarscout.trajectory._validation import prepare_static_problem

    grid = make_trajectory_georef(width=2, height=1)
    sunlight = ls.trajectory.ArraySunlightProvider(
        _boundaries(),
        np.asarray([[[255, 0]], [[255, 255]]], dtype=np.uint8),
        grid,
    )
    configuration = ls.trajectory.SunlightThresholdProvider(sunlight, 1.0)
    timeline = occupancy_timeline_from_provider(configuration, _boundaries())
    problem = prepare_static_problem(
        np.ones((1, 2), dtype=bool),
        grid,
        (0, 0),
        goal=(1, 0),
        model=ls.trajectory.StaticTravelModel(
            speed_m_per_h=20.0, include_diagonals=False
        ),
    )

    result = exact_dynamic_path(problem, timeline, T0)

    assert result.reachable
    assert result.departure_times == (T0 + timedelta(hours=1),)
    assert result.arrival_time == T0 + timedelta(hours=1.5)


def test_provider_grid_and_window_validation(make_trajectory_georef) -> None:
    first = make_trajectory_georef(width=2, height=1)
    second = make_trajectory_georef(
        width=2,
        height=1,
        affine=(1.0, 10.0, 0.0, 0.0, 0.0, -10.0),
    )
    one = ls.trajectory.StaticConfigurationSpaceProvider(
        np.ones((1, 2), dtype=bool), first
    )
    two = ls.trajectory.StaticConfigurationSpaceProvider(
        np.ones((1, 2), dtype=bool), second
    )
    with pytest.raises(ls.TrajectoryInputError) as grids:
        ls.trajectory.AllOfConfigurationSpaceProvider((one, two))
    assert grids.value.code == "trajectory_provider_grid_mismatch"
    with pytest.raises(ls.TrajectoryInputError) as window:
        one.read(1, 0, 2, 1, T0)
    assert window.value.code == "trajectory_invalid_provider_window"


def test_close_clears_provider_cache_and_prevents_signal_reads(
    make_trajectory_georef,
) -> None:
    grid = make_trajectory_georef(width=1, height=1)
    provider = ls.trajectory.ArraySunlightProvider(
        _boundaries(1), np.ones((1, 1, 1), dtype=np.uint8), grid
    )
    provider.read(0, 0, 1, 1, T0)
    assert len(provider._cache.values) == 1
    provider.close()
    assert len(provider._cache.values) == 0
    with pytest.raises(ls.ConfigurationSpaceError) as closed:
        provider.read(0, 0, 1, 1, T0)
    assert closed.value.code == "trajectory_provider_closed"
