"""Private patch-major per-pixel safe-haven product pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import numpy.typing as npt

from lunarscout.georeference import GeoReference

from .file_format import HorizonTileStore
from .geometry import DemGrid
from .lightmap_cpu import LightmapCpuSession
from .pipeline import PatchDescriptor, enumerate_patches
from .product_store import ProductJob, ResumableTiledProduct, resolve_output_dtype
from .psr import _validate_vectors
from .psr_pipeline import _inventory_identity
from .safe_haven import (
    build_month_bands,
    _month_indices_map,
    reduce_safe_haven_patch_stream,
)


FractionCalculator = Callable[..., Iterable[npt.ArrayLike]]
MarginCalculator = Callable[..., Iterable[npt.ArrayLike]]


class SafeHavenPipelineCancelled(RuntimeError):
    """Cancellation observed between bounded safe-haven work units."""


@dataclass(frozen=True, slots=True)
class SafeHavenProgress:
    completed_patches: int
    total_patches: int
    tile_y: int | None
    tile_x: int | None
    state: str
    backend: Literal["cpu", "cuda"] | None = None


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
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    time_batch_size: int = 32,
    fraction_calculator: FractionCalculator | None = None,
    elevation_calculator: MarginCalculator | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    cancellation_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[SafeHavenProgress], None] | None = None,
    progress_stream: TextIO | None = None,
) -> Path:
    """Write per-pixel safe-haven duration bands, one per calendar month.

    Earth outages are detected per-pixel from each pixel's own terrain horizon.
    Calendar-month bands are computed from the evaluation timespan; the band
    name is the UTC ``[start, stop)`` interval.
    """
    if (georef.width, georef.height) != (dem.width, dem.height):
        raise ValueError("GeoReference and DEM dimensions do not match")
    timestamps = tuple(times_utc)
    sun_vectors = _validate_vectors(sun_vectors_m)
    earth_vectors = _validate_vectors(earth_vectors_m)
    if not len(timestamps) == len(sun_vectors) == len(earth_vectors):
        raise ValueError("timestamps, Sun vectors, and Earth vectors must align")

    month_bands = build_month_bands(timestamps)
    if not month_bands:
        raise ValueError("times_utc must span at least one calendar month")

    month_index_of = _month_indices_map(timestamps, month_bands)

    patches = enumerate_patches(dem.width, dem.height)
    inventory = _inventory_identity(
        horizon_store, patches, observer_elevation_m
    )
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    if time_batch_size < 1:
        raise ValueError("time_batch_size must be positive")
    storage_dtype = resolve_output_dtype(
        np.float32, output_transform, output_dtype, output_transform_id
    )

    selected_backend: Literal["cpu", "cuda"] | None = None

    if fraction_calculator is not None:
        calc_fractions = fraction_calculator
    elif backend == "cpu":
        calc_fractions = LightmapCpuSession(
            time_batch_size=time_batch_size
        ).iter_patch_fraction_tiles
        selected_backend = "cpu"
    else:
        from .cuda_backend import CudaBackendError
        from .lightmap_cuda import LightmapCudaSession

        try:
            calc_fractions = LightmapCudaSession(
                time_batch_size=time_batch_size
            ).iter_patch_fraction_tiles
            selected_backend = "cuda"
        except CudaBackendError:
            if backend == "cuda":
                raise
            calc_fractions = LightmapCpuSession(
                time_batch_size=time_batch_size
            ).iter_patch_fraction_tiles
            selected_backend = "cpu"

    if elevation_calculator is not None:
        calc_elevations = elevation_calculator
    elif backend == "cpu" or selected_backend == "cpu":
        calc_elevations = LightmapCpuSession(
            time_batch_size=time_batch_size
        ).iter_patch_margin_tiles
    else:
        from .cuda_backend import CudaBackendError
        from .lightmap_cuda import LightmapCudaSession

        try:
            calc_elevations = LightmapCudaSession(
                time_batch_size=time_batch_size
            ).iter_patch_margin_tiles
        except CudaBackendError:
            if backend == "cuda":
                raise
            calc_elevations = LightmapCpuSession(
                time_batch_size=time_batch_size
            ).iter_patch_margin_tiles

    band_timestamps: list[datetime | str] = []
    for start, stop in month_bands:
        band_timestamps.append(start.isoformat())

    product = ResumableTiledProduct(
        output_path,
        ProductJob(
            georef=georef,
            dtype=storage_dtype,
            band_count=len(month_bands),
            timestamps_utc=tuple(band_timestamps),
            invalid_value=nodata,
            nodata=nodata,
            compression="deflate" if compress else "none",
            algorithm="safe-haven-per-pixel-monthly-bands",
            configuration={
                "semantics_version": 2,
                "time_step_hours": float(time_step_hours),
                "earth_threshold_deg": float(earth_threshold_deg),
                "sunlight_threshold": float(sunlight_threshold),
                "month_bands": [
                    [start.isoformat(), stop.isoformat()]
                    for start, stop in month_bands
                ],
                "sun_vectors_sha256": hashlib.sha256(
                    sun_vectors.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "earth_vectors_sha256": hashlib.sha256(
                    earth_vectors.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "output_transform_id": output_transform_id,
            },
            horizon_inventory_identity=inventory,
        ),
        overwrite=overwrite,
        start_fresh=start_fresh,
        backend=selected_backend,
        output_transform=output_transform,
    )

    def cancelled() -> bool:
        return bool(cancellation_requested and cancellation_requested())

    completed = len(product.completed_patches)

    def report(patch: PatchDescriptor | None, state: str) -> None:
        event = SafeHavenProgress(
            completed,
            len(patches),
            None if patch is None else patch.tile_y,
            None if patch is None else patch.tile_x,
            state,
            selected_backend,
        )
        if progress_callback is not None:
            progress_callback(event)
        if progress_stream is not None:
            location = "" if patch is None else f" row={patch.tile_y} col={patch.tile_x}"
            print(
                f"Safe haven {state}:{location} {completed}/{len(patches)} patches",
                file=progress_stream,
                flush=True,
            )

    report(None, "start")
    for patch in patches:
        if product.is_complete(patch.tile_y, patch.tile_x):
            continue
        if cancelled():
            raise SafeHavenPipelineCancelled(
                "safe-haven generation was cancelled"
            )
        report(patch, "read")
        try:
            horizons = horizon_store.read(
                patch.tile_y, patch.tile_x, observer_elevation_m
            )
        except (OSError, ValueError):
            horizons = None
        if cancelled():
            raise SafeHavenPipelineCancelled(
                "safe-haven generation was cancelled"
            )
        if horizons is None:
            product.write_invalid_patch(patch.tile_y, patch.tile_x)
            state = "invalid"
        else:
            report(patch, "calculate")

            fractions = calc_fractions(
                dem,
                horizons,
                sun_vectors,
                tile_y=patch.tile_y,
                tile_x=patch.tile_x,
                valid_height=patch.height,
                valid_width=patch.width,
            )
            elevations = calc_elevations(
                dem,
                horizons,
                earth_vectors,
                tile_y=patch.tile_y,
                tile_x=patch.tile_x,
                valid_height=patch.height,
                valid_width=patch.width,
            )

            def checked_fractions() -> Iterable[npt.ArrayLike]:
                for tile in fractions:
                    if cancelled():
                        raise SafeHavenPipelineCancelled(
                            "safe-haven generation was cancelled"
                        )
                    yield tile

            def checked_elevations() -> Iterable[npt.ArrayLike]:
                for tile in elevations:
                    if cancelled():
                        raise SafeHavenPipelineCancelled(
                            "safe-haven generation was cancelled"
                        )
                    yield tile

            duration_tiles = reduce_safe_haven_patch_stream(
                checked_fractions(),
                checked_elevations(),
                len(timestamps),
                month_bands,
                month_index_of=month_index_of,
                sunlight_threshold=sunlight_threshold,
                earth_threshold_deg=earth_threshold_deg,
                time_step_hours=time_step_hours,
            )
            if cancelled():
                raise SafeHavenPipelineCancelled(
                    "safe-haven generation was cancelled"
                )
            report(patch, "write")
            product.write_patch(patch.tile_y, patch.tile_x, duration_tiles)
            state = "valid"
        completed += 1
        report(patch, state)
    if cancelled():
        raise SafeHavenPipelineCancelled("safe-haven generation was cancelled")
    result = product.finalize()
    report(None, "complete")
    return result


def run_safe_haven_product_cpu(**kwargs) -> Path:
    """Compatibility helper selecting the compiled CPU backend explicitly."""
    return run_safe_haven_product(**kwargs, backend="cpu")
