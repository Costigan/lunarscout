"""Private patch-major Sun/Earth local-horizon elevation products."""

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
from .pipeline import PatchDescriptor, enumerate_patches
from .product_store import ProductJob, ResumableTiledProduct
from .psr import _validate_vectors
from .psr_pipeline import _inventory_identity


Backend = Literal["auto", "cpu", "cuda"]
MarginCalculator = Callable[..., Iterable[npt.ArrayLike]]


class BodyElevationPipelineCancelled(RuntimeError):
    """Cancellation observed between bounded body-elevation work units."""


@dataclass(frozen=True, slots=True)
class BodyElevationProgress:
    body: Literal["sun", "earth"]
    completed_patches: int
    total_patches: int
    tile_y: int | None
    tile_x: int | None
    state: str


def _run_body_elevation_product(
    *,
    body: Literal["sun", "earth"],
    dem: DemGrid,
    georef: GeoReference,
    horizon_store: HorizonTileStore,
    output_path: str | Path,
    times_utc: Sequence[datetime | str],
    body_vectors_m: npt.ArrayLike,
    observer_elevation_m: float = 0.0,
    invalid_value: float = 0.0,
    overwrite: bool = False,
    start_fresh: bool = False,
    cancellation_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[BodyElevationProgress], None] | None = None,
    progress_stream: TextIO | None = None,
    backend: Backend = "auto",
    time_batch_size: int = 32,
    _margin_calculator: MarginCalculator | None = None,
) -> Path:
    """Write one Float32 terrain-relative body-center elevation band per time."""
    if (georef.width, georef.height) != (dem.width, dem.height):
        raise ValueError("GeoReference and DEM dimensions do not match")
    if body not in ("sun", "earth"):
        raise ValueError("body must be 'sun' or 'earth'")
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    if time_batch_size < 1:
        raise ValueError("time_batch_size must be positive")
    vectors = _validate_vectors(body_vectors_m)
    timestamps = tuple(times_utc)
    if len(timestamps) != len(vectors):
        raise ValueError("timestamp count must equal body vector count")

    selected_backend = None
    if _margin_calculator is not None:
        calculate_patch = _margin_calculator
    elif backend == "cpu":
        from .lightmap_cpu import LightmapCpuSession

        calculate_patch = LightmapCpuSession(
            time_batch_size=time_batch_size
        ).iter_patch_margin_tiles
        selected_backend = "cpu"
    else:
        from .cuda_backend import CudaBackendError
        from .lightmap_cuda import LightmapCudaSession

        try:
            calculate_patch = LightmapCudaSession(
                time_batch_size=time_batch_size
            ).iter_patch_margin_tiles
            selected_backend = "cuda"
        except CudaBackendError:
            if backend == "cuda":
                raise
            from .lightmap_cpu import LightmapCpuSession

            calculate_patch = LightmapCpuSession(
                time_batch_size=time_batch_size
            ).iter_patch_margin_tiles
            selected_backend = "cpu"

    patches = enumerate_patches(dem.width, dem.height)
    inventory = _inventory_identity(horizon_store, patches, observer_elevation_m)
    product = ResumableTiledProduct(
        output_path,
        ProductJob(
            georef=georef,
            dtype=np.float32,
            band_count=len(timestamps),
            timestamps_utc=timestamps,
            invalid_value=invalid_value,
            algorithm=f"{body}-center-local-horizon-elevation",
            configuration={
                "body": body,
                "body_vectors_sha256": hashlib.sha256(
                    vectors.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "observer_elevation_m": float(observer_elevation_m),
            },
            horizon_inventory_identity=inventory,
        ),
        overwrite=overwrite,
        start_fresh=start_fresh,
        backend=selected_backend,
    )

    def cancelled() -> bool:
        return bool(cancellation_requested and cancellation_requested())

    completed = len(product.completed_patches)

    def report(patch: PatchDescriptor | None, state: str) -> None:
        event = BodyElevationProgress(
            body,
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
                f"{body.title()} elevation {state}:{location} "
                f"{completed}/{len(patches)} patches",
                file=progress_stream,
                flush=True,
            )

    report(None, "start")
    for patch in patches:
        if product.is_complete(patch.tile_y, patch.tile_x):
            continue
        if cancelled():
            raise BodyElevationPipelineCancelled(
                f"{body} elevation generation was cancelled"
            )
        report(patch, "read")
        try:
            horizons = horizon_store.read(
                patch.tile_y, patch.tile_x, observer_elevation_m
            )
        except (OSError, ValueError):
            horizons = None
        if cancelled():
            raise BodyElevationPipelineCancelled(
                f"{body} elevation generation was cancelled"
            )
        if horizons is None:
            product.write_invalid_patch(patch.tile_y, patch.tile_x)
            state = "invalid"
        else:
            report(patch, "calculate")
            tiles = calculate_patch(
                dem,
                horizons,
                vectors,
                tile_y=patch.tile_y,
                tile_x=patch.tile_x,
                valid_height=patch.height,
                valid_width=patch.width,
            )

            def checked_tiles() -> Iterable[npt.ArrayLike]:
                for tile in tiles:
                    if cancelled():
                        raise BodyElevationPipelineCancelled(
                            f"{body} elevation generation was cancelled"
                        )
                    yield tile

            report(patch, "write")
            product.write_patch(patch.tile_y, patch.tile_x, checked_tiles())
            state = "valid"
        completed += 1
        report(patch, state)
    if cancelled():
        raise BodyElevationPipelineCancelled(
            f"{body} elevation generation was cancelled"
        )
    result = product.finalize()
    report(None, "complete")
    return result


def run_sun_elevation_product(*, sun_vectors_m: npt.ArrayLike, **kwargs) -> Path:
    """Write Sun-center elevation relative to the local terrain horizon."""
    return _run_body_elevation_product(
        body="sun", body_vectors_m=sun_vectors_m, **kwargs
    )


def run_earth_elevation_product(*, earth_vectors_m: npt.ArrayLike, **kwargs) -> Path:
    """Write Earth-center elevation relative to the local terrain horizon."""
    return _run_body_elevation_product(
        body="earth", body_vectors_m=earth_vectors_m, **kwargs
    )
