from __future__ import annotations

import numpy as np
import pytest

from scripts.numba_horizon.validate_phase4_hierarchy_safety import run_matrix
from scripts.numba_horizon.validate_phase4_real_terrain import (
    _solar_visible_fraction,
)


def test_dense_bilinear_matrix_is_recorded_as_diagnostic() -> None:
    report = run_matrix()
    assert not report["diagnostic_comparison"]["is_acceptance_gate"]
    assert not report["diagnostic_comparison"]["all_cases_within_threshold"]
    assert report["maximum_dense_minus_hierarchy_degrees"] == pytest.approx(
        0.5714783590002774
    )
    assert report["configuration"]["case_count"] == 10
    assert any(
        case["hierarchy_maximum_level"] > 0 for case in report["cases"]
    )
    assert report["edge_and_invalid_neighbor_bounds_m"] == {
        "top_left_with_nan": 4.0,
        "right_edge": 9.0,
        "bottom_right_corner": 9.0,
    }


def test_uniform_solar_disk_fraction_and_observed_error() -> None:
    values = _solar_visible_fraction(np.array((-0.25, 0.0, 0.25)))
    np.testing.assert_array_equal(values, np.array((0.0, 0.5, 1.0)))

    angular_delta = 7.3909759521484375e-06
    half_delta = angular_delta / 2.0
    worst_fraction_error = float(
        _solar_visible_fraction(np.array((half_delta,)))[0]
        - _solar_visible_fraction(np.array((-half_delta,)))[0]
    )
    assert worst_fraction_error == pytest.approx(1.882e-5, rel=2e-3)
    assert worst_fraction_error < 0.01
