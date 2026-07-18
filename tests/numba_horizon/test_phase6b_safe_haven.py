from __future__ import annotations

import numpy as np

from lunarscout._numba_horizon.safe_haven import (
    EarthOutage,
    find_earth_outages,
    reduce_safe_haven_patch,
)


def test_earth_outages_are_half_open_and_include_edge_regions() -> None:
    elevations = np.asarray((-1.0, -2.0, 3.0, 1.0, 0.0, 3.0, -4.0))

    outages = find_earth_outages(elevations, threshold_deg=2.0)

    assert outages == (
        EarthOutage(0, 2, 1),
        EarthOutage(3, 5, 4),
        EarthOutage(6, 7, 6),
    )


def test_safe_haven_duration_includes_last_sample_and_preserves_float_hours() -> None:
    fractions = np.asarray(
        (
            ((0.1, 0.9),),
            ((0.1, 0.1),),
            ((0.9, 0.1),),
            ((0.1, 0.1),),
            ((0.1, 0.9),),
        ),
        dtype=np.float32,
    )
    outages = (EarthOutage(0, 3, 1), EarthOutage(3, 5, 3))

    actual = reduce_safe_haven_patch(
        fractions,
        outages,
        sunlight_threshold=0.2,
        time_step_hours=2.5,
    )

    np.testing.assert_array_equal(
        actual,
        np.asarray((((5.0, 5.0),), ((5.0, 2.5),)), dtype=np.float32),
    )
