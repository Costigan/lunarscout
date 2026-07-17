"""Factor-four maximum-elevation pyramids for the Numba horizon prototype."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contract import PyramidArrays
from .geometry import DemGrid


PYRAMID_DOWNSAMPLE_FACTOR = 4
INVALID_BLOCK_ELEVATION_M = np.float32(-32000.0)


def _level_layout(dem: DemGrid) -> tuple[np.ndarray, int]:
    """Return the production factor-four level table and total mip elements."""
    shapes = [(dem.height, dem.width)]
    while shapes[-1] != (1, 1):
        height, width = shapes[-1]
        shapes.append(((height + 3) // 4, (width + 3) // 4))
    levels = np.empty((len(shapes), 4), dtype=np.int32)
    offset = 0
    for level, (height, width) in enumerate(shapes):
        levels[level] = (level, 0 if level == 0 else offset, width, height)
        if level > 0:
            offset += width * height
    return levels, offset


def load_max_pyramid_cache(dem: DemGrid, path: str | Path) -> PyramidArrays:
    """Load the language-neutral float32 mip payload written by moonlib.

    The legacy cache contract validates only the expected array length. Phase 5
    records the DEM and cache hashes separately so benchmark provenance does not
    imply stronger cache invalidation than production currently provides.
    """
    cache_path = Path(path)
    levels, expected_count = _level_layout(dem)
    mips = np.fromfile(cache_path, dtype="<f4")
    if mips.size != expected_count:
        raise ValueError(
            f"pyramid cache has {mips.size} float32 values; expected {expected_count}"
        )
    return _pyramid_arrays(dem, np.ascontiguousarray(mips), levels)


def _pyramid_arrays(
    dem: DemGrid, mips: np.ndarray, levels: np.ndarray
) -> PyramidArrays:
    transform = dem.geo_transform
    determinant = transform[1] * transform[5] - transform[2] * transform[4]
    map_parameters = np.array(
        (
            dem.projection.radius_m,
            dem.projection.scale,
            dem.projection.false_easting_m,
            dem.projection.false_northing_m,
            1.0 / determinant,
            *transform,
        ),
        dtype=np.float32,
    )
    projection_parameters = np.array(
        (
            dem.projection.radius_m,
            dem.projection.latitude_origin_rad,
            dem.projection.longitude_origin_rad,
            dem.projection.scale,
            dem.projection.false_easting_m,
            dem.projection.false_northing_m,
        ),
        dtype=np.float32,
    )
    return PyramidArrays(
        dem.elevation_m, mips, levels, map_parameters, projection_parameters
    )


def build_max_pyramid(dem: DemGrid) -> PyramidArrays:
    """Build the exact production factor-four max pyramid in host memory."""
    arrays = [np.ascontiguousarray(dem.elevation_m, dtype=np.float32)]
    while arrays[-1].shape != (1, 1):
        source = arrays[-1]
        height = (source.shape[0] + PYRAMID_DOWNSAMPLE_FACTOR - 1) // 4
        width = (source.shape[1] + PYRAMID_DOWNSAMPLE_FACTOR - 1) // 4
        destination = np.full(
            (height, width), INVALID_BLOCK_ELEVATION_M, dtype=np.float32
        )
        for row in range(height):
            for column in range(width):
                block = source[
                    row * 4 : min((row + 1) * 4, source.shape[0]),
                    column * 4 : min((column + 1) * 4, source.shape[1]),
                ]
                valid = block[np.isfinite(block) & (block > -20000.0)]
                if valid.size:
                    destination[row, column] = np.max(valid)
        arrays.append(destination)

    levels, expected_count = _level_layout(dem)
    mips = (
        np.ascontiguousarray(np.concatenate([array.ravel() for array in arrays[1:]]))
        if len(arrays) > 1 else np.empty(0, dtype=np.float32)
    )
    if mips.size != expected_count:
        raise AssertionError("constructed pyramid does not match its level layout")
    return _pyramid_arrays(dem, mips, levels)
