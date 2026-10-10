"""Focal statistics run in bounded strips (review finding H10).

Reductions over a sliding_window_view materialized (height, width, fh, fw)
temporaries, about 100x the raster for a 9x9 footprint. They now run over
horizontal output strips whose temporaries fit a fixed budget.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import lunarscout.map_algebra.focal as focal
from lunarscout.raster import Raster
from tests.map_algebra.test_temporal import _make_georef


def _raster(height: int, width: int, dtype=np.float32, *, seed: int = 0) -> Raster:
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind in "iu":
        values = rng.integers(0, 100, (height, width)).astype(dtype)
    else:
        values = rng.normal(0.0, 10.0, (height, width)).astype(dtype)
    return Raster(
        values=values, georef=_make_georef(width=width, height=height),
        valid=rng.random((height, width)) > 0.1,
    )


_OPERATIONS = ("focal_sum", "focal_mean", "focal_std", "focal_min", "focal_max",
               "focal_count", "focal_median", "focal_range")


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.int16])
@pytest.mark.parametrize("edge", ["invalid", "reflect", "wrap"])
def test_one_row_strips_match_a_single_strip(monkeypatch, operation, dtype, edge) -> None:
    raster = _raster(13, 17, dtype)
    whole = getattr(focal, operation)(raster, size=5, edge=edge, valid_neighbor="ignore_invalid")
    monkeypatch.setattr(focal, "_STRIP_BUDGET_BYTES", 1)
    strips = getattr(focal, operation)(raster, size=5, edge=edge, valid_neighbor="ignore_invalid")
    assert strips.values.dtype == whole.values.dtype
    np.testing.assert_array_equal(strips.valid, whole.valid)
    np.testing.assert_array_equal(strips.values[strips.valid], whole.values[whole.valid])


@pytest.mark.parametrize("operation", ["focal_mean", "focal_std", "focal_sum"])
def test_focal_peak_memory_is_bounded(operation) -> None:
    # 800 x 800 float32 with a 9x9 footprint: whole-raster temporaries were
    # about 800 * 800 * 81 * 9 bytes = 466 MiB.
    raster = _raster(800, 800)
    tracemalloc.start()
    try:
        getattr(focal, operation)(raster, size=9)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 96 * 1024 * 1024, f"peak {peak / 2**20:.0f} MiB"
