from __future__ import annotations

import numpy as np
import pytest

from lunarscout._numba_horizon.hierarchy import _cull_bound_distance

RADIUS = np.float32(1737400.0)


def _flat_slope(height, observer_z, distance):
    return (height - observer_z) / distance


def _curved_slope(height, observer_z, distance):
    # The production far-field slope used by the cull and level-0 samples,
    # evaluated in float64.
    radius = float(RADIUS)
    squared = distance * distance
    observer_radius = radius + observer_z
    local_z = (
        (height - observer_z) * (2.0 * radius + height + observer_z) - squared
    ) / (2.0 * observer_radius)
    local_x = np.sqrt(max(squared - local_z * local_z, 1e-12))
    return local_z / local_x


def test_block_above_observer_is_bounded_at_its_near_side() -> None:
    bound = _cull_bound_distance(
        np.float32(2000.0), np.float32(320.0), np.float32(50.0),
        np.float32(0.0), RADIUS, False,
    )
    assert bound == pytest.approx(1680.0)


def test_block_below_observer_is_bounded_at_its_far_side_in_the_near_field() -> None:
    bound = _cull_bound_distance(
        np.float32(100.0), np.float32(80.0), np.float32(-20.0),
        np.float32(0.0), RADIUS, True,
    )
    assert bound == pytest.approx(100.0 + 80.0 * np.sqrt(2.0), rel=1e-6)


def test_block_below_observer_is_bounded_at_the_curved_slope_peak() -> None:
    # d* = sqrt(-dh * (2R + h + z0)) = sqrt(1 * 3474799) ~ 1864 m, inside the block.
    bound = _cull_bound_distance(
        np.float32(1500.0), np.float32(640.0), np.float32(-1.0),
        np.float32(0.0), RADIUS, False,
    )
    assert bound == pytest.approx(np.sqrt(2.0 * 1737400.0 - 1.0), rel=1e-5)


@pytest.mark.parametrize("near_field", [True, False])
def test_bound_is_at_least_the_slope_anywhere_in_the_block(near_field) -> None:
    # The H-B review case: a block below the observer must not be culled when
    # a farther point in it rises above the current horizon.
    rng = np.random.default_rng(0)
    slope = _flat_slope if near_field else _curved_slope
    for _ in range(2000):
        true_m = np.float32(rng.uniform(30.0, 400.0 if near_field else 20000.0))
        block_m = np.float32(20.0 * 2 ** int(rng.integers(0, 6)))
        observer_z = np.float32(rng.uniform(-50.0, 50.0))
        height = np.float32(observer_z + rng.uniform(-300.0, 300.0))
        bound = _cull_bound_distance(
            true_m, block_m, height, observer_z, RADIUS, near_field
        )
        near = max(float(true_m - block_m), 1.0)
        far = float(true_m) + float(block_m) * np.sqrt(2.0)
        if abs(float(height - observer_z)) >= 0.9 * near:
            continue  # not a physical geometry: the height difference exceeds the distance
        distances = np.linspace(near, far, 257)
        reachable = max(slope(float(height), float(observer_z), d) for d in distances)
        assert slope(float(height), float(observer_z), float(bound)) >= reachable - 1e-6
