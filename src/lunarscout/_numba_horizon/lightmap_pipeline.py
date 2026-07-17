"""Private patch-major, resumable byte lightmap pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import TextIO

import numpy as np
import numpy.typing as npt

from lunarscout.georeference import GeoReference

from .file_format import HorizonTileStore
from .geometry import DemGrid
from .lightmap import iter_lightmap_patch_reference
from .pipeline import PatchDescriptor, enumerate_patches
from .product_store import ProductJob, ResumableTiledProduct
from .psr import _validate_vectors
from .psr_pipeline import _inventory_identity


class LightmapPipelineCancelled(RuntimeError):
    """Cancellation observed between bounded lightmap work units."""


@dataclass(frozen=True, slots=True)
class LightmapProgress:
    completed_patches: int
    total_patches: int
    tile_y: int | None
    tile_x: int | None
    state: str


PatchCalculator = Callable[..., Iterable[npt.ArrayLike]]


def run_lightmap_product(
    *,
    dem: DemGrid,
    georef: GeoReference,
    horizon_store: HorizonTileStore,
    output_path: str | Path,
    times_utc: Sequence[datetime | str],
    sun_vectors_m: npt.ArrayLike,
    observer_elevation_m: float = 0.0,
    invalid_value: int = 0,
    overwrite: bool = False,
    start_fresh: bool = False,
    cancellation_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[LightmapProgress], None] | None = None,
    progress_stream: TextIO | None = None,
    patch_calculator: PatchCalculator | None = None,
) -> Path:
    """Write one timestamped uint8 band per vector without a regional cube."""
    if (georef.width, georef.height) != (dem.width, dem.height):
        raise ValueError("GeoReference and DEM dimensions do not match")
    vectors = _validate_vectors(sun_vectors_m)
    timestamps = tuple(times_utc)
    if len(timestamps) != vectors.shape[0]:
        raise ValueError("timestamp count must equal Sun vector count")
    patches = enumerate_patches(dem.width, dem.height)
    calculate_patch = patch_calculator or iter_lightmap_patch_reference
    inventory = _inventory_identity(
        horizon_store, patches, observer_elevation_m
    )
    product = ResumableTiledProduct(
        output_path,
        ProductJob(
            georef=georef,
            dtype=np.uint8,
            band_count=len(timestamps),
            timestamps_utc=timestamps,
            invalid_value=invalid_value,
            algorithm="lightmap-builder-sun-fraction",
            configuration={
                "sun_half_angle_deg": 0.27,
                "solar_disk_slices": 16,
                "vector_sha256": hashlib.sha256(
                    vectors.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "observer_elevation_m": float(observer_elevation_m),
            },
            horizon_inventory_identity=inventory,
        ),
        overwrite=overwrite,
        start_fresh=start_fresh,
    )

    def cancelled() -> bool:
        return bool(cancellation_requested and cancellation_requested())

    completed = len(product.completed_patches)

    def report(patch: PatchDescriptor | None, state: str) -> None:
        event = LightmapProgress(
            completed,
            len(patches),
            None if patch is None else patch.tile_y,
            None if patch is None else patch.tile_x,
            state,
        )
        if progress_callback is not None:
            progress_callback(event)
        if progress_stream is not None:
            location = "" if patch is None else f" row={patch.tile_y} col={patch.tile_x}"
            print(
                f"Lightmap {state}:{location} {completed}/{len(patches)} patches",
                file=progress_stream,
                flush=True,
            )

    report(None, "start")
    for patch in patches:
        if product.is_complete(patch.tile_y, patch.tile_x):
            continue
        if cancelled():
            raise LightmapPipelineCancelled("lightmap generation was cancelled")
        report(patch, "read")
        try:
            horizons = horizon_store.read(
                patch.tile_y, patch.tile_x, observer_elevation_m
            )
        except (OSError, ValueError):
            horizons = None
        if cancelled():
            raise LightmapPipelineCancelled("lightmap generation was cancelled")
        if horizons is None:
            product.write_invalid_patch(patch.tile_y, patch.tile_x)
            state = "invalid"
        else:
            report(patch, "calculate")
            band_tiles = calculate_patch(
                dem,
                horizons,
                vectors,
                tile_y=patch.tile_y,
                tile_x=patch.tile_x,
                valid_height=patch.height,
                valid_width=patch.width,
            )

            def checked_tiles() -> Iterable[npt.ArrayLike]:
                for tile in band_tiles:
                    if cancelled():
                        raise LightmapPipelineCancelled(
                            "lightmap generation was cancelled"
                        )
                    yield tile

            report(patch, "write")
            product.write_patch(
                patch.tile_y, patch.tile_x, checked_tiles()
            )
            state = "valid"
        completed += 1
        report(patch, state)
    if cancelled():
        raise LightmapPipelineCancelled("lightmap generation was cancelled")
    result = product.finalize()
    report(None, "complete")
    return result
