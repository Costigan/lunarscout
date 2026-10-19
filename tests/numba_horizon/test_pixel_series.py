"""The single-pixel CPU kernel returns exactly what the patch kernel computes."""

from __future__ import annotations

import numpy as np
import pytest

from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, PATCH_SIZE
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession


def _scene(seed: int = 7):
    rng = np.random.default_rng(seed)
    elevation = rng.uniform(-50.0, 50.0, size=(PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    dem = DemGrid(
        elevation,
        np.asarray((1000.0, 20.0, 0.0, -1000.0, 0.0, -20.0)),
        ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0),
    )
    horizons = rng.uniform(-3.0, 8.0, size=(PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT)).astype(np.float32)
    # Unit directions at lunar distances; the Sun/Earth direction sweeps the sky.
    angles = rng.uniform(0.0, 2.0 * np.pi, size=96)
    lift = rng.uniform(-0.05, 0.12, size=96)
    vectors = 3.8e8 * np.stack([np.cos(angles), np.sin(angles), lift], axis=1)
    return dem, horizons, vectors


@pytest.mark.parametrize("row, column", [(0, 0), (17, 93), (127, 127), (64, 3)])
def test_a_pixel_series_equals_the_patch_kernel_exactly(row: int, column: int) -> None:
    dem, horizons, vectors = _scene()
    session = LightmapCpuSession(time_batch_size=len(vectors))

    (fractions,) = list(session.iter_patch_fraction_batches(
        dem, horizons, vectors, tile_y=0, tile_x=0))
    (margins,) = list(session.iter_patch_margin_batches(
        dem, horizons, vectors, tile_y=0, tile_x=0))

    sun = session.pixel_series(dem, horizons[row, column], vectors, row=row, column=column, margin=False)
    earth = session.pixel_series(dem, horizons[row, column], vectors, row=row, column=column, margin=True)

    assert np.array_equal(sun, fractions[:, row, column])
    assert np.array_equal(earth, margins[:, row, column], equal_nan=True)
    assert sun.dtype == np.float32 and sun.shape == (len(vectors),)
