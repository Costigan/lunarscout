"""Private production-shaped orchestration for the Numba horizon prototype."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from .contract import HorizonBuffers, PyramidArrays, SegmentTensor
from .cuda_backend import CudaSession


def generate_patch_horizons(
    session: CudaSession,
    segments: SegmentTensor,
    pyramids: Sequence[PyramidArrays],
    *,
    tile_column: int,
    tile_row: int,
    observer_elevation_m: float = 0.0,
) -> HorizonBuffers:
    """Accumulate every DEM pass in slope space and convert only on request."""
    configuration = segments.configuration
    if len(pyramids) != configuration.dem_count:
        raise ValueError("pyramid count must match the segment DEM axis")
    slopes = None
    for pass_index, pyramid in enumerate(pyramids):
        pass_slopes = session.subpatch_hierarchical_pass(
            segments.values,
            pyramids[0],
            pyramid,
            tile_column=tile_column,
            tile_row=tile_row,
            tile_width=configuration.tile_width,
            tile_height=configuration.tile_height,
            subpatch_size=configuration.subpatch_size,
            pass_index=pass_index,
            observer_elevation_m=observer_elevation_m,
        )
        slopes = (
            pass_slopes
            if slopes is None
            else np.maximum(slopes, pass_slopes)
        )
    if slopes is None:
        raise ValueError("at least one DEM pyramid is required")
    return HorizonBuffers(slopes)
