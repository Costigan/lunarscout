"""Private patch-major pipeline writing the three per-pixel safe-haven products."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import queue
import threading
from typing import Literal, TextIO

import numpy as np
import numpy.typing as npt

from lunarscout.georeference import GeoReference

from .file_format import HorizonTileStore, read_patch_horizons
from .geometry import DemGrid
from .lightmap_cpu import LightmapCpuSession
from .pipeline import PatchDescriptor, enumerate_patches
from .product_store import ProductJob, ResumableTiledProduct, resolve_output_dtype
from .psr import _validate_vectors
from .psr_pipeline import _inventory_identity
from .safe_haven import (
    build_month_bands,
    _month_indices_map,
    reduce_safe_haven_products_batches,
    reduce_safe_haven_products_patch_stream,
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
    invalid_patches: int | None = None


#: Horizon tiles read and decoded ahead of the compute stage, and the threads
#: doing it. Decoding a tile takes about as long as computing a patch on the GPU,
#: so the compute stage should never wait for it.
PREFETCH_TILES = 3
READ_WORKERS = 2
#: Computed patches waiting for the writer thread.
WRITE_QUEUE_SIZE = 2

#: Output layers, in the order the reducer returns them.
PRODUCT_KINDS = ("hibernation", "joint_total", "joint_longest")
_ALGORITHMS = {
    "hibernation": "safe-haven-hibernation-by-outage-start",
    "joint_total": "safe-haven-joint-operating-total",
    "joint_longest": "safe-haven-joint-operating-longest-run",
}


def patch_slope_exceeds(
    dem: DemGrid,
    georef: GeoReference,
    patch: PatchDescriptor,
    max_slope_deg: float,
) -> npt.NDArray[np.bool_]:
    """Mask of patch pixels whose Horn slope is above ``max_slope_deg``.

    The slope uses the same Horn kernel as ``lunarscout.slope`` on a window
    with a one-pixel halo, so patch edges see their true neighbours; at the DEM
    border missing neighbours are extrapolated (gdaldem ``-compute_edges``).
    Pixels whose slope cannot be computed (an invalid DEM node in the 3x3
    window) are also masked.
    """
    from lunarscout.terrain import _gradient, _require_supported_grid

    _require_supported_grid(georef, scale=1.0)
    y0 = max(patch.tile_y - 1, 0)
    x0 = max(patch.tile_x - 1, 0)
    y1 = min(patch.tile_y + patch.height + 1, dem.height)
    x1 = min(patch.tile_x + patch.width + 1, dem.width)
    window = np.asarray(dem.elevation_m[y0:y1, x0:x1], dtype=np.float64)
    dz_east, dz_south = _gradient(window, georef)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_east, dz_south)))
    inner = slope_deg[
        patch.tile_y - y0 : patch.tile_y - y0 + patch.height,
        patch.tile_x - x0 : patch.tile_x - x0 + patch.width,
    ]
    return ~(inner <= max_slope_deg)


def run_safe_haven_products(
    *,
    dem: DemGrid,
    georef: GeoReference,
    horizon_store: HorizonTileStore,
    output_paths: Sequence[str | Path],
    times_utc: Sequence[datetime | str],
    sun_vectors_m: npt.ArrayLike,
    earth_vectors_m: npt.ArrayLike,
    time_step_hours: float,
    earth_threshold_deg: float = 2.0,
    sunlight_threshold: float = 0.2,
    operating_sunlight_fraction: float = 1.0,
    merge_gap_hours: float = 24.0,
    max_slope_deg: float | None = 20.0,
    observer_elevation_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    time_batch_size: int = 32,
    fraction_calculator: FractionCalculator | None = None,
    elevation_calculator: MarginCalculator | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    cancellation_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[SafeHavenProgress], None] | None = None,
    progress_stream: TextIO | None = None,
) -> tuple[Path, Path, Path]:
    """Write the hibernation, joint-total and joint-longest products in one pass.

    Each product has one band per calendar month, written as its own resumable
    GeoTIFF.  A patch is calculated once and skipped only when all three
    products already hold it.  ``output_paths`` are in ``PRODUCT_KINDS`` order.
    """
    if (georef.width, georef.height) != (dem.width, dem.height):
        raise ValueError("GeoReference and DEM dimensions do not match")
    if len(output_paths) != len(PRODUCT_KINDS):
        raise ValueError("output_paths must name one file per product kind")
    timestamps = tuple(times_utc)
    sun_vectors = _validate_vectors(sun_vectors_m)
    earth_vectors = _validate_vectors(earth_vectors_m)
    if not len(timestamps) == len(sun_vectors) == len(earth_vectors):
        raise ValueError("timestamps, Sun vectors, and Earth vectors must align")
    if max_slope_deg is not None and not (np.isfinite(max_slope_deg) and max_slope_deg > 0.0):
        raise ValueError("max_slope_deg must be positive and finite, or None")

    month_bands = build_month_bands(timestamps)
    if not month_bands:
        raise ValueError("times_utc must span at least one calendar month")
    month_index_of = _month_indices_map(timestamps, month_bands)

    patches = enumerate_patches(dem.width, dem.height)
    inventory = _inventory_identity(horizon_store, patches, observer_elevation_m)
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    if time_batch_size < 1:
        raise ValueError("time_batch_size must be positive")
    storage_dtype = resolve_output_dtype(
        np.float32, output_transform, output_dtype, output_transform_id
    )

    selected_backend: Literal["cpu", "cuda"] | None = None
    cuda_session = None
    cpu_session = None
    if fraction_calculator is not None or elevation_calculator is not None:
        # Any injected calculator selects the reference streaming path.  Fill
        # the other signal with the compiled CPU implementation below.
        if fraction_calculator is not None:
            calc_fractions = fraction_calculator
        else:
            calc_fractions = LightmapCpuSession(time_batch_size=time_batch_size).iter_patch_fraction_tiles
        selected_backend = "cpu"
    elif backend == "cpu":
        cpu_session = LightmapCpuSession(time_batch_size=time_batch_size)
        selected_backend = "cpu"
    else:
        from .cuda_backend import CudaBackendError
        from .safe_haven_cuda import SafeHavenCudaSession, SafeHavenCudaCancelled

        try:
            cuda_session = SafeHavenCudaSession(time_batch_size=time_batch_size)
            selected_backend = "cuda"
        except CudaBackendError:
            if backend == "cuda":
                raise
            cpu_session = LightmapCpuSession(time_batch_size=time_batch_size)
            selected_backend = "cpu"

    if elevation_calculator is not None:
        calc_elevations = elevation_calculator
    elif fraction_calculator is not None:
        calc_elevations = LightmapCpuSession(time_batch_size=time_batch_size).iter_patch_margin_tiles
    else:
        calc_elevations = None   # the CUDA session or the CPU batch session supplies it

    band_timestamps = tuple(start.isoformat() for start, _stop in month_bands)
    configuration = {
        "semantics_version": 3,
        "time_step_hours": float(time_step_hours),
        "earth_threshold_deg": float(earth_threshold_deg),
        "sunlight_threshold": float(sunlight_threshold),
        "operating_sunlight_fraction": float(operating_sunlight_fraction),
        "merge_gap_hours": float(merge_gap_hours),
        "max_slope_deg": None if max_slope_deg is None else float(max_slope_deg),
        "month_bands": [
            [start.isoformat(), stop.isoformat()] for start, stop in month_bands
        ],
        "sun_vectors_sha256": hashlib.sha256(
            sun_vectors.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "earth_vectors_sha256": hashlib.sha256(
            earth_vectors.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "output_transform_id": output_transform_id,
    }
    products = [
        ResumableTiledProduct(
            path,
            ProductJob(
                georef=georef,
                dtype=storage_dtype,
                band_count=len(month_bands),
                timestamps_utc=band_timestamps,
                invalid_value=nodata,
                nodata=nodata,
                compression="deflate" if compress else "none",
                algorithm=_ALGORITHMS[kind],
                configuration=configuration,
                horizon_inventory_identity=inventory,
            ),
            overwrite=overwrite,
            start_fresh=start_fresh,
            backend=selected_backend,
            output_transform=output_transform,
            dem_elevation=None if skip_dem_nodata_check else dem.elevation_m,
        )
        for kind, path in zip(PRODUCT_KINDS, output_paths)
    ]

    def cancelled() -> bool:
        return bool(cancellation_requested and cancellation_requested())

    def complete(patch: PatchDescriptor) -> bool:
        return all(product.is_complete(patch.tile_y, patch.tile_x) for product in products)

    completed = sum(1 for patch in patches if complete(patch))
    # The compute stage runs on the calling thread (the CUDA context belongs to
    # it); tile reads and product writes run on helper threads. Progress events
    # and the completed count are serialised by one lock, so a progress
    # callback is never called concurrently with itself.
    progress_lock = threading.Lock()

    def report(
        patch: PatchDescriptor | None, state: str, invalid_patches: int | None = None
    ) -> None:
        with progress_lock:
            event = SafeHavenProgress(
                completed,
                len(patches),
                None if patch is None else patch.tile_y,
                None if patch is None else patch.tile_x,
                state,
                selected_backend,
                invalid_patches,
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

    def read_tile(patch: PatchDescriptor):
        report(patch, "read")
        # A missing tile is an invalid patch; an existing tile that cannot be
        # read raises HorizonFormatError (re-raised, unchanged, by future.result()).
        return read_patch_horizons(
            horizon_store, patch.tile_y, patch.tile_x, observer_elevation_m
        )

    def calculate(patch: PatchDescriptor, horizons):
        """The three layers of one patch, as a tuple of per-band tile tuples."""
        report(patch, "calculate")
        # Pixels on invalid DEM nodes have NaN lighting signals; the stores
        # write them as nodata, so the reducers skip them.
        invalid_pixels = None
        if not skip_dem_nodata_check:
            with np.errstate(invalid="ignore"):
                window = dem.elevation_m[
                    patch.tile_y : patch.tile_y + patch.height,
                    patch.tile_x : patch.tile_x + patch.width,
                ]
                invalid_pixels = ~(window > -20000.0)
            if not invalid_pixels.any():
                invalid_pixels = None
        options = dict(
            month_index_of=month_index_of,
            sunlight_threshold=sunlight_threshold,
            earth_threshold_deg=earth_threshold_deg,
            operating_sunlight_fraction=operating_sunlight_fraction,
            merge_gap_hours=merge_gap_hours,
            time_step_hours=time_step_hours,
            invalid_pixels=invalid_pixels,
        )
        window_arguments = dict(
            tile_y=patch.tile_y, tile_x=patch.tile_x,
            valid_height=patch.height, valid_width=patch.width,
        )

        def checked(tiles: Iterable[npt.ArrayLike]) -> Iterable[npt.ArrayLike]:
            for tile in tiles:
                if cancelled():
                    raise SafeHavenPipelineCancelled("safe-haven generation was cancelled")
                yield tile

        if cuda_session is not None and fraction_calculator is None and elevation_calculator is None:
            try:
                return cuda_session.reduce_patch(
                    dem, horizons, sun_vectors, earth_vectors,
                    month_bands=month_bands, time_count=len(timestamps),
                    cancellation_requested=cancelled, **window_arguments,
                    **{key: options[key] for key in (
                        "month_index_of", "sunlight_threshold", "earth_threshold_deg",
                        "operating_sunlight_fraction", "merge_gap_hours",
                        "time_step_hours", "invalid_pixels",
                    )},
                )
            except SafeHavenCudaCancelled as error:
                raise SafeHavenPipelineCancelled(str(error)) from error
        if cpu_session is not None and fraction_calculator is None and elevation_calculator is None:
            # Whole batches of time steps go to the reducer, so each parallel
            # call does many steps of work per thread.
            return reduce_safe_haven_products_batches(
                checked(cpu_session.iter_patch_fraction_batches(
                    dem, horizons, sun_vectors, **window_arguments)),
                checked(cpu_session.iter_patch_margin_batches(
                    dem, horizons, earth_vectors, **window_arguments)),
                len(timestamps), month_bands, **options,
            )
        fractions = calc_fractions(dem, horizons, sun_vectors, **window_arguments)
        elevations = calc_elevations(dem, horizons, earth_vectors, **window_arguments)
        return reduce_safe_haven_products_patch_stream(
            checked(fractions), checked(elevations), len(timestamps),
            month_bands, **options,
        )

    write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
    finished = object()
    writer_errors: list[BaseException] = []

    def writer() -> None:
        nonlocal completed
        while True:
            job = write_queue.get()
            if job is finished:
                return
            if writer_errors:
                continue          # a write failed: drop the work queued behind it
            patch, layers = job
            try:
                if layers is None:
                    for product in products:
                        product.write_invalid_patch(patch.tile_y, patch.tile_x)
                    state = "invalid"
                else:
                    report(patch, "write")
                    for product, kind_tiles in zip(products, layers):
                        product.write_patch(patch.tile_y, patch.tile_x, kind_tiles)
                    state = "valid"
                with progress_lock:
                    completed += 1
                report(patch, state)
            except BaseException as error:  # reported to the calling thread
                writer_errors.append(error)

    def hand_to_writer(job) -> None:
        while True:
            if writer_errors:
                raise writer_errors[0]
            try:
                write_queue.put(job, timeout=0.05)
                return
            except queue.Full:
                continue

    report(None, "start")
    pending = [patch for patch in patches if not complete(patch)]
    writer_thread = threading.Thread(target=writer, name="safe-haven-writer")
    writer_thread.start()
    futures: deque = deque()
    try:
        with ThreadPoolExecutor(max_workers=READ_WORKERS, thread_name_prefix="safe-haven-read") as readers:
            upcoming = iter(pending)

            def refill() -> None:
                while len(futures) < PREFETCH_TILES:
                    patch = next(upcoming, None)
                    if patch is None:
                        return
                    futures.append((patch, readers.submit(read_tile, patch)))

            try:
                refill()
                while futures:
                    if cancelled():
                        raise SafeHavenPipelineCancelled("safe-haven generation was cancelled")
                    patch, future = futures.popleft()
                    refill()
                    horizons = future.result()
                    if cancelled():
                        raise SafeHavenPipelineCancelled("safe-haven generation was cancelled")
                    if horizons is None:
                        hand_to_writer((patch, None))
                        continue
                    layers = calculate(patch, horizons)
                    del horizons
                    if cancelled():
                        raise SafeHavenPipelineCancelled("safe-haven generation was cancelled")
                    if max_slope_deg is not None:
                        steep = patch_slope_exceeds(dem, georef, patch, max_slope_deg)
                        if steep.any():
                            layers = tuple(
                                tuple(np.where(steep, np.float32(np.nan), tile) for tile in kind_tiles)
                                for kind_tiles in layers
                            )
                    hand_to_writer((patch, layers))
            finally:
                for _patch, future in futures:
                    future.cancel()
    finally:
        # Let the writer finish what is already queued, so completed work is kept.
        while True:
            try:
                write_queue.put(finished, timeout=0.05)
                break
            except queue.Full:
                continue
        writer_thread.join()
    if writer_errors:
        raise writer_errors[0]
    if cancelled():
        raise SafeHavenPipelineCancelled("safe-haven generation was cancelled")
    results = tuple(product.finalize() for product in products)
    report(
        None,
        "complete",
        sum(state == "invalid" for state in products[0].completed_patches.values()),
    )
    return results  # type: ignore[return-value]
