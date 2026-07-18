"""Private patch-major CPU safe-haven product pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from lunarscout.georeference import GeoReference

from .file_format import HorizonTileStore
from .geometry import DemGrid
from .lightmap_cpu import LightmapCpuSession
from .pipeline import enumerate_patches
from .product_store import ProductJob, ResumableTiledProduct
from .psr import _azimuth_elevation_deg, _pixel_frame, _validate_vectors
from .psr_pipeline import _inventory_identity
from .safe_haven import find_earth_outages, reduce_safe_haven_patch_stream


FractionCalculator = Callable[..., Iterable[npt.ArrayLike]]


def run_safe_haven_product(
    *,
    dem: DemGrid,
    georef: GeoReference,
    horizon_store: HorizonTileStore,
    output_path: str | Path,
    times_utc: Sequence[datetime | str],
    sun_vectors_m: npt.ArrayLike,
    earth_vectors_m: npt.ArrayLike,
    time_step_hours: float,
    earth_threshold_deg: float = 2.0,
    sunlight_threshold: float = 0.2,
    observer_elevation_m: float = 0.0,
    invalid_value: float = 0.0,
    overwrite: bool = False,
    start_fresh: bool = False,
    time_batch_size: int = 32,
    fraction_calculator: FractionCalculator | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
) -> Path:
    """Write one float32 duration band per center-view Earth outage."""
    if (georef.width, georef.height) != (dem.width, dem.height):
        raise ValueError("GeoReference and DEM dimensions do not match")
    timestamps = tuple(times_utc)
    sun_vectors = _validate_vectors(sun_vectors_m)
    earth_vectors = _validate_vectors(earth_vectors_m)
    if not len(timestamps) == len(sun_vectors) == len(earth_vectors):
        raise ValueError("timestamps, Sun vectors, and Earth vectors must align")
    center_row, center_column = dem.height // 2, dem.width // 2
    rotation, translation = _pixel_frame(dem, center_row, center_column)
    _azimuth, earth_elevation = _azimuth_elevation_deg(
        earth_vectors, rotation, translation
    )
    outages = find_earth_outages(
        earth_elevation, threshold_deg=earth_threshold_deg
    )
    if not outages:
        raise ValueError("no center-view Earth-below-threshold intervals")
    patches = enumerate_patches(dem.width, dem.height)
    inventory = _inventory_identity(
        horizon_store, patches, observer_elevation_m
    )
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    if fraction_calculator is not None:
        calculator = fraction_calculator
    elif backend == "cpu":
        calculator = LightmapCpuSession(
            time_batch_size=time_batch_size
        ).iter_patch_fraction_tiles
    else:
        from .cuda_backend import CudaBackendError
        from .lightmap_cuda import LightmapCudaSession

        try:
            calculator = LightmapCudaSession(
                time_batch_size=time_batch_size
            ).iter_patch_fraction_tiles
        except CudaBackendError:
            if backend == "cuda":
                raise
            calculator = LightmapCpuSession(
                time_batch_size=time_batch_size
            ).iter_patch_fraction_tiles
    product = ResumableTiledProduct(
        output_path,
        ProductJob(
            georef=georef,
            dtype=np.float32,
            band_count=len(outages),
            timestamps_utc=tuple(timestamps[item.minimum_index] for item in outages),
            invalid_value=invalid_value,
            algorithm="safe-haven-longest-low-sun-duration",
            configuration={
                "time_step_hours": float(time_step_hours),
                "earth_threshold_deg": float(earth_threshold_deg),
                "sunlight_threshold": float(sunlight_threshold),
                "outages": [
                    [item.start, item.stop, item.minimum_index] for item in outages
                ],
                "sun_vectors_sha256": hashlib.sha256(
                    sun_vectors.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "earth_vectors_sha256": hashlib.sha256(
                    earth_vectors.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
            },
            horizon_inventory_identity=inventory,
        ),
        overwrite=overwrite,
        start_fresh=start_fresh,
    )
    for patch in patches:
        if product.is_complete(patch.tile_y, patch.tile_x):
            continue
        try:
            horizons = horizon_store.read(
                patch.tile_y, patch.tile_x, observer_elevation_m
            )
        except (OSError, ValueError):
            horizons = None
        if horizons is None:
            product.write_invalid_patch(patch.tile_y, patch.tile_x)
            continue
        fractions = calculator(
            dem,
            horizons,
            sun_vectors,
            tile_y=patch.tile_y,
            tile_x=patch.tile_x,
            valid_height=patch.height,
            valid_width=patch.width,
        )
        duration_tiles = reduce_safe_haven_patch_stream(
            fractions,
            len(timestamps),
            outages,
            sunlight_threshold=sunlight_threshold,
            time_step_hours=time_step_hours,
        )
        product.write_patch(patch.tile_y, patch.tile_x, duration_tiles)
    return product.finalize()


def run_safe_haven_product_cpu(**kwargs) -> Path:
    """Compatibility helper selecting the compiled CPU backend explicitly."""
    return run_safe_haven_product(**kwargs, backend="cpu")
