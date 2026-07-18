"""Private patch-major, resumable PSR product pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import numpy.typing as npt

from lunarscout.georeference import GeoReference

from .file_format import HorizonTileStore
from .geometry import DemGrid
from .pipeline import PatchDescriptor, enumerate_patches
from .product_store import ProductJob, ResumableTiledProduct
from .psr import compute_psr_patch_reference, reduce_sun_vectors_for_psr


class PsrPipelineCancelled(RuntimeError):
    """Cancellation observed between bounded PSR work units."""


@dataclass(frozen=True, slots=True)
class PsrProgress:
    completed_patches: int
    total_patches: int
    tile_y: int | None
    tile_x: int | None
    state: str


def _inventory_identity(
    store: HorizonTileStore,
    patches: Sequence[PatchDescriptor],
    observer_elevation_m: float,
) -> str:
    """Fingerprint the bounded input inventory without rereading every payload."""
    records = []
    for patch in patches:
        path = store.find_existing_path(
            patch.tile_y, patch.tile_x, observer_elevation_m
        )
        if path is None:
            records.append((patch.tile_y, patch.tile_x, None))
        else:
            stat = path.stat()
            records.append(
                (
                    patch.tile_y,
                    patch.tile_x,
                    str(path.resolve()),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                )
            )
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return f"stat-sha256:{hashlib.sha256(payload).hexdigest()}"


def run_psr_product(
    *,
    dem: DemGrid,
    georef: GeoReference,
    horizon_store: HorizonTileStore,
    output_path: str | Path,
    sun_vectors_m: npt.ArrayLike,
    observer_elevation_m: float = 0.0,
    invalid_value: int = 0,
    overwrite: bool = False,
    start_fresh: bool = False,
    cancellation_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[float], None] | None = None,
    progress_event_callback: Callable[[PsrProgress], None] | None = None,
    progress_stream: TextIO | None = None,
    patch_calculator: Callable[..., npt.NDArray[np.uint8]] | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
) -> Path:
    """Generate a single-band PSR GeoTIFF with explicit backend selection."""
    if (georef.width, georef.height) != (dem.width, dem.height):
        raise ValueError("GeoReference and DEM dimensions do not match")
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    patches = enumerate_patches(dem.width, dem.height)
    if patch_calculator is not None:
        calculate_patch = patch_calculator
    elif backend == "cpu":
        calculate_patch = compute_psr_patch_reference
    else:
        from .cuda_backend import CudaBackendError
        from .psr_cuda import PsrCudaSession

        try:
            calculate_patch = PsrCudaSession().compute_patch
        except CudaBackendError:
            if backend == "cuda":
                raise
            calculate_patch = compute_psr_patch_reference
    reduced_vectors, reduced_indices = reduce_sun_vectors_for_psr(
        dem, sun_vectors_m
    )
    inventory = _inventory_identity(
        horizon_store, patches, observer_elevation_m
    )
    product = ResumableTiledProduct(
        output_path,
        ProductJob(
            georef=georef,
            dtype=np.uint8,
            band_count=1,
            invalid_value=invalid_value,
            algorithm="psr-upper-solar-limb",
            configuration={
                "sun_angular_size_deg": 0.545,
                "input_vector_count": int(np.asarray(sun_vectors_m).shape[0]),
                "reduced_vector_count": int(reduced_vectors.shape[0]),
                "reduced_vector_indices_sha256": hashlib.sha256(
                    reduced_indices.astype("<i8", copy=False).tobytes()
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
        event = PsrProgress(
            completed,
            len(patches),
            None if patch is None else patch.tile_y,
            None if patch is None else patch.tile_x,
            state,
        )
        if progress_event_callback is not None:
            progress_event_callback(event)
        if progress_stream is not None:
            if patch is None:
                line = f"PSR {state}: {completed}/{len(patches)} patches"
            else:
                line = (
                    f"PSR {state}: row={patch.tile_y} col={patch.tile_x} "
                    f"{completed}/{len(patches)} patches"
                )
            print(line, file=progress_stream, flush=True)

    report(None, "start")
    if progress_callback is not None:
        progress_callback(completed / len(patches))
    for patch in patches:
        if product.is_complete(patch.tile_y, patch.tile_x):
            continue
        if cancelled():
            raise PsrPipelineCancelled("PSR generation was cancelled")
        report(patch, "read")
        try:
            horizons = horizon_store.read(
                patch.tile_y, patch.tile_x, observer_elevation_m
            )
        except (OSError, ValueError):
            horizons = None
        if cancelled():
            raise PsrPipelineCancelled("PSR generation was cancelled")
        if horizons is None:
            product.write_invalid_patch(patch.tile_y, patch.tile_x)
            state = "invalid"
        else:
            report(patch, "calculate")
            tile = calculate_patch(
                dem,
                horizons,
                reduced_vectors,
                tile_y=patch.tile_y,
                tile_x=patch.tile_x,
                valid_height=patch.height,
                valid_width=patch.width,
            )
            if cancelled():
                raise PsrPipelineCancelled("PSR generation was cancelled")
            report(patch, "write")
            product.write_patch(patch.tile_y, patch.tile_x, (tile,))
            state = "valid"
        completed += 1
        report(patch, state)
        if progress_callback is not None:
            progress_callback(completed / len(patches))
    if cancelled():
        raise PsrPipelineCancelled("PSR generation was cancelled")
    result = product.finalize()
    report(None, "complete")
    return result
