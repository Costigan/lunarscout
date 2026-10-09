"""Private production-shaped orchestration for the Numba horizon prototype."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import HorizonGenerationError
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
    if not pyramids:
        raise ValueError("at least one DEM pyramid is required")
    slopes, _ = session.subpatch_hierarchical_all_passes(
        segments.values,
        pyramids,
        tile_column=tile_column,
        tile_row=tile_row,
        tile_width=configuration.tile_width,
        tile_height=configuration.tile_height,
        subpatch_size=configuration.subpatch_size,
        observer_elevation_m=observer_elevation_m,
    )
    capped = int(np.count_nonzero(np.isnan(slopes)))
    if capped:
        raise HorizonGenerationError(
            "Horizon rays exceeded the ray-march step limit.",
            code="horizon_step_limit_exceeded",
            details={
                "tile_column": int(tile_column),
                "tile_row": int(tile_row),
                "ray_count": capped,
            },
        )
    return HorizonBuffers(slopes)
