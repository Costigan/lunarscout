"""Private patch-major, resumable PSR product pipeline."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from queue import Full, Queue
import threading
import time
from typing import Literal, TextIO

import numpy as np
import numpy.typing as npt

from lunarscout.georeference import GeoReference

from .file_format import HorizonTileStore, read_horizon_tile_with_timings
from .geometry import DemGrid
from .pipeline import PatchDescriptor, enumerate_patches
from .product_store import (
    ProductCheckpointTimings,
    ProductJob,
    ProductBatchWriter,
    ResumableTiledProduct,
)
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


@dataclass(frozen=True, slots=True)
class PsrTiming:
    """One opt-in timing observation from the private PSR pipeline."""

    stage: str
    seconds: float
    tile_y: int | None = None
    tile_x: int | None = None


@dataclass(frozen=True, slots=True)
class PsrPipelineMetrics:
    """Observed queue and decoded-buffer bounds for one PSR run."""

    mode: str
    reader_worker_count: int
    decoded_horizon_capacity: int
    writer_queue_capacity: int
    maximum_live_decoded_horizons: int
    maximum_reader_queue_depth: int
    maximum_writer_queue_depth: int
    reader_enqueue_wait_seconds: tuple[float, ...]
    cuda_dequeue_wait_seconds: tuple[float, ...]
    writer_enqueue_wait_seconds: tuple[float, ...]
    writer_dequeue_wait_seconds: tuple[float, ...]
    durable_batch_size: int
    maximum_uncheckpointed_patches: int
    host_horizon_buffer_kind: str
    preallocated_host_horizon_bytes: int


@dataclass(slots=True)
class _ReadPatch:
    patch: PatchDescriptor
    horizons: npt.NDArray[np.float32] | None
    patch_started: float
    owns_decoded_slot: bool


@dataclass(slots=True)
class _ComputedPatch:
    patch: PatchDescriptor
    tile: npt.NDArray[np.uint8] | None
    state: str
    patch_started: float


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
    timing_callback: Callable[[PsrTiming], None] | None = None,
    metrics_callback: Callable[[PsrPipelineMetrics], None] | None = None,
    progress_stream: TextIO | None = None,
    patch_calculator: Callable[..., npt.NDArray[np.uint8]] | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    pipeline_mode: Literal["serial", "bounded"] = "bounded",
    decoded_horizon_capacity: int = 5,
    writer_queue_capacity: int = 1,
    reader_worker_count: int = 4,
    durable_batch_size: int = 16,
    host_horizon_buffers: Literal[
        "auto", "allocated", "pageable", "pinned"
    ] = "auto",
) -> Path:
    """Generate a single-band PSR GeoTIFF with explicit backend selection."""
    pipeline_started = time.perf_counter()
    callback_lock = threading.Lock()

    def timing(
        stage: str,
        seconds: float,
        patch: PatchDescriptor | None = None,
    ) -> None:
        if timing_callback is not None:
            with callback_lock:
                timing_callback(
                    PsrTiming(
                        stage=stage,
                        seconds=float(seconds),
                        tile_y=None if patch is None else patch.tile_y,
                        tile_x=None if patch is None else patch.tile_x,
                    )
                )

    if (georef.width, georef.height) != (dem.width, dem.height):
        raise ValueError("GeoReference and DEM dimensions do not match")
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    if pipeline_mode not in ("serial", "bounded"):
        raise ValueError("pipeline_mode must be 'serial' or 'bounded'")
    if host_horizon_buffers not in ("auto", "allocated", "pageable", "pinned"):
        raise ValueError(
            "host_horizon_buffers must be 'auto', 'allocated', 'pageable', or 'pinned'"
        )
    if (
        host_horizon_buffers in ("pageable", "pinned")
        and pipeline_mode != "bounded"
    ):
        raise ValueError("preallocated horizon buffers require the bounded pipeline")
    if decoded_horizon_capacity < 1:
        raise ValueError("decoded_horizon_capacity must be positive")
    if pipeline_mode == "bounded" and decoded_horizon_capacity < 2:
        raise ValueError(
            "bounded pipeline decoded_horizon_capacity must be at least 2"
        )
    if writer_queue_capacity < 1:
        raise ValueError("writer_queue_capacity must be positive")
    if reader_worker_count < 1:
        raise ValueError("reader_worker_count must be positive")
    if (
        isinstance(durable_batch_size, bool)
        or not isinstance(durable_batch_size, int)
        or durable_batch_size < 1
    ):
        raise ValueError("durable_batch_size must be a positive integer")
    if pipeline_mode == "bounded" and reader_worker_count > decoded_horizon_capacity:
        raise ValueError(
            "reader_worker_count cannot exceed decoded_horizon_capacity"
        )
    patches = enumerate_patches(dem.width, dem.height)
    patches_by_origin = {(patch.tile_y, patch.tile_x): patch for patch in patches}
    cuda_session = None
    selected_backend = None
    if patch_calculator is not None:
        calculate_patch = patch_calculator
    elif backend == "cpu":
        calculate_patch = compute_psr_patch_reference
        selected_backend = "cpu"
    else:
        from .cuda_backend import CudaBackendError
        from .psr_cuda import PsrCudaPatchTimings, PsrCudaSession

        def cuda_timing(values: PsrCudaPatchTimings) -> None:
            patch = patches_by_origin[(values.tile_y, values.tile_x)]
            for stage, seconds in (
                ("host_patch_preparation", values.host_preparation_seconds),
                ("h2d_dem", values.h2d_dem_seconds),
                ("h2d_metadata", values.h2d_metadata_seconds),
                ("h2d_horizon", values.h2d_horizon_seconds),
                ("h2d_vectors", values.h2d_vectors_seconds),
                ("kernel_launch", values.kernel_launch_seconds),
                ("kernel_execution_gpu", values.kernel_execution_seconds),
                (
                    "cuda_synchronization_boundary",
                    values.synchronization_boundary_seconds,
                ),
                ("d2h_result", values.d2h_result_seconds),
                ("cuda_calculation_boundary", values.total_seconds),
            ):
                timing(stage, seconds, patch)

        try:
            cuda_session = PsrCudaSession(
                timing_callback=cuda_timing if timing_callback is not None else None
            )
            calculate_patch = cuda_session.compute_patch
            selected_backend = "cuda"
        except CudaBackendError:
            if backend == "cuda":
                raise
            calculate_patch = compute_psr_patch_reference
            selected_backend = "cpu"
    reduction_started = time.perf_counter()
    reduced_vectors, reduced_indices = reduce_sun_vectors_for_psr(
        dem, sun_vectors_m
    )
    timing("vector_reduction", time.perf_counter() - reduction_started)
    inventory_started = time.perf_counter()
    inventory = _inventory_identity(
        horizon_store, patches, observer_elevation_m
    )
    timing("horizon_inventory", time.perf_counter() - inventory_started)
    product_started = time.perf_counter()
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
        backend=selected_backend,
    )
    timing("product_initialize", time.perf_counter() - product_started)

    def cancelled() -> bool:
        return bool(cancellation_requested and cancellation_requested())

    completed = len(product.completed_patches)
    horizon_buffer_pool: Queue[npt.NDArray[np.float32]] | None = None
    selected_host_horizon_buffers = host_horizon_buffers
    if selected_host_horizon_buffers == "auto":
        selected_host_horizon_buffers = (
            "pinned"
            if pipeline_mode == "bounded" and cuda_session is not None
            else "allocated"
        )
    elif selected_host_horizon_buffers == "pinned" and cuda_session is None:
        selected_host_horizon_buffers = "allocated"

    def report(patch: PatchDescriptor | None, state: str) -> None:
        event = PsrProgress(
            completed,
            len(patches),
            None if patch is None else patch.tile_y,
            None if patch is None else patch.tile_x,
            state,
        )
        with callback_lock:
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

    def read_patch(
        patch: PatchDescriptor,
        *,
        acquire_decoded_slot: Callable[[], bool] | None = None,
        release_decoded_slot: Callable[
            [npt.NDArray[np.float32] | None], None
        ]
        | None = None,
    ) -> _ReadPatch:
        patch_started = time.perf_counter()
        report(patch, "read")
        lookup_started = time.perf_counter()
        try:
            horizon_path = horizon_store.find_existing_path(
                patch.tile_y, patch.tile_x, observer_elevation_m
            )
        except (OSError, ValueError):
            horizon_path = None
        timing("horizon_lookup", time.perf_counter() - lookup_started, patch)
        owns_decoded_slot = False
        decode_output = None
        if horizon_path is None:
            horizons = None
        else:
            try:
                if acquire_decoded_slot is not None:
                    if not acquire_decoded_slot():
                        return _ReadPatch(patch, None, patch_started, False)
                    owns_decoded_slot = True
                    if horizon_buffer_pool is not None:
                        decode_output = horizon_buffer_pool.get_nowait()
                horizons, read_timings = read_horizon_tile_with_timings(
                    horizon_path,
                    output=decode_output,
                )
                timing(
                    "compressed_file_read",
                    read_timings.file_read_seconds,
                    patch,
                )
                timing(
                    "cbin_decompression",
                    read_timings.decompression_seconds,
                    patch,
                )
            except (OSError, ValueError):
                if owns_decoded_slot:
                    assert release_decoded_slot is not None
                    release_decoded_slot(decode_output)
                    owns_decoded_slot = False
                horizons = None
        return _ReadPatch(patch, horizons, patch_started, owns_decoded_slot)

    def compute_patch(item: _ReadPatch) -> _ComputedPatch:
        patch = item.patch
        if item.horizons is None:
            return _ComputedPatch(patch, None, "invalid", item.patch_started)
        report(patch, "calculate")
        calculation_started = time.perf_counter()
        tile = calculate_patch(
            dem,
            item.horizons,
            reduced_vectors,
            tile_y=patch.tile_y,
            tile_x=patch.tile_x,
            valid_height=patch.height,
            valid_width=patch.width,
        )
        timing(
            "calculation_boundary",
            time.perf_counter() - calculation_started,
            patch,
        )
        return _ComputedPatch(patch, tile, "valid", item.patch_started)

    uncheckpointed: dict[str, _ComputedPatch] = {}
    maximum_uncheckpointed_patches = 0

    def record_checkpoint(values: ProductCheckpointTimings | None) -> None:
        nonlocal completed
        if values is None:
            return
        last_item = uncheckpointed[values.completed_patch_keys[-1]]
        timing("tiff_checkpoint_close", values.tiff_close_seconds, last_item.patch)
        timing("tiff_synchronize", values.tiff_synchronize_seconds, last_item.patch)
        timing(
            "journal_persistence",
            values.journal_persistence_seconds,
            last_item.patch,
        )
        for key in values.completed_patch_keys:
            item = uncheckpointed.pop(key)
            patch = item.patch
            timing("patch_total", time.perf_counter() - item.patch_started, patch)
            completed += 1
            report(patch, item.state)
            if progress_callback is not None:
                progress_callback(completed / len(patches))

    def write_patch(writer: ProductBatchWriter, item: _ComputedPatch) -> None:
        nonlocal maximum_uncheckpointed_patches
        patch = item.patch
        report(patch, "write")
        key = product._patch_key(patch.tile_y, patch.tile_x)
        uncheckpointed[key] = item
        maximum_uncheckpointed_patches = max(
            maximum_uncheckpointed_patches, len(uncheckpointed)
        )
        try:
            if item.state == "invalid":
                write_timings = writer.write_patch_with_timings(
                    patch.tile_y, patch.tile_x, (), valid=False
                )
            else:
                assert item.tile is not None
                write_timings = writer.write_patch_with_timings(
                    patch.tile_y, patch.tile_x, (item.tile,)
                )
        except BaseException:
            uncheckpointed.pop(key, None)
            raise
        timing("tiff_write", write_timings.tiff_write_seconds, patch)
        record_checkpoint(write_timings.checkpoint)

    pending = [
        patch
        for patch in patches
        if not product.is_complete(patch.tile_y, patch.tile_x)
    ]
    if selected_host_horizon_buffers != "allocated" and pending:
        allocation_started = time.perf_counter()
        horizon_buffer_pool = Queue(maxsize=decoded_horizon_capacity)
        for _ in range(decoded_horizon_capacity):
            if selected_host_horizon_buffers == "pinned":
                if cuda_session is None:
                    break
                buffer = cuda_session.allocate_pinned_horizon_buffer()
            else:
                buffer = np.empty((128, 128, 1440), dtype=np.float32)
            horizon_buffer_pool.put_nowait(buffer)
        if horizon_buffer_pool.qsize() != decoded_horizon_capacity:
            horizon_buffer_pool = None
        timing(
            "host_horizon_buffer_allocation",
            time.perf_counter() - allocation_started,
        )
    if pipeline_mode == "serial":
        with product.batch_writer(durable_batch_size) as batch_writer:
            for patch in pending:
                if cancelled():
                    raise PsrPipelineCancelled("PSR generation was cancelled")
                read_item = read_patch(patch)
                if cancelled():
                    raise PsrPipelineCancelled("PSR generation was cancelled")
                computed_item = compute_patch(read_item)
                if cancelled():
                    raise PsrPipelineCancelled("PSR generation was cancelled")
                write_patch(batch_writer, computed_item)
            record_checkpoint(batch_writer.checkpoint_with_timings())
        metrics = PsrPipelineMetrics(
            mode="serial",
            reader_worker_count=0,
            decoded_horizon_capacity=1,
            writer_queue_capacity=0,
            maximum_live_decoded_horizons=1 if pending else 0,
            maximum_reader_queue_depth=0,
            maximum_writer_queue_depth=0,
            reader_enqueue_wait_seconds=(),
            cuda_dequeue_wait_seconds=(),
            writer_enqueue_wait_seconds=(),
            writer_dequeue_wait_seconds=(),
            durable_batch_size=durable_batch_size,
            maximum_uncheckpointed_patches=maximum_uncheckpointed_patches,
            host_horizon_buffer_kind=(
                selected_host_horizon_buffers
                if horizon_buffer_pool is not None
                else "allocated"
            ),
            preallocated_host_horizon_bytes=(
                decoded_horizon_capacity * 128 * 128 * 1440 * 4
                if horizon_buffer_pool is not None
                else 0
            ),
        )
    else:
        reader_queue: Queue[_ReadPatch | object] = Queue(
            maxsize=decoded_horizon_capacity - 1
        )
        writer_queue: Queue[_ComputedPatch | object] = Queue(
            maxsize=writer_queue_capacity
        )
        reader_sentinel = object()
        writer_sentinel = object()
        decoded_slots = threading.BoundedSemaphore(decoded_horizon_capacity)
        stop = threading.Event()
        errors: list[BaseException] = []
        error_lock = threading.Lock()
        metrics_lock = threading.Lock()
        live_decoded_horizons = 0
        maximum_live_decoded_horizons = 0
        maximum_reader_queue_depth = 0
        maximum_writer_queue_depth = 0
        reader_enqueue_wait_seconds: list[float] = []
        cuda_dequeue_wait_seconds: list[float] = []
        writer_enqueue_wait_seconds: list[float] = []
        writer_dequeue_wait_seconds: list[float] = []

        def fail(error: BaseException) -> None:
            with error_lock:
                if not errors:
                    errors.append(error)
            stop.set()

        def acquire_slot() -> bool:
            nonlocal live_decoded_horizons, maximum_live_decoded_horizons
            while not stop.is_set():
                if decoded_slots.acquire(timeout=0.05):
                    with metrics_lock:
                        live_decoded_horizons += 1
                        maximum_live_decoded_horizons = max(
                            maximum_live_decoded_horizons,
                            live_decoded_horizons,
                        )
                    return True
            return False

        def release_slot(
            horizons: npt.NDArray[np.float32] | None = None,
        ) -> None:
            nonlocal live_decoded_horizons
            if horizon_buffer_pool is not None:
                assert horizons is not None
                horizon_buffer_pool.put_nowait(horizons)
            with metrics_lock:
                live_decoded_horizons -= 1
            decoded_slots.release()

        def put_reader(item: _ReadPatch) -> bool:
            nonlocal maximum_reader_queue_depth
            wait_started = time.perf_counter()
            while True:
                try:
                    reader_queue.put(item, timeout=0.05)
                    wait_seconds = time.perf_counter() - wait_started
                    with metrics_lock:
                        reader_enqueue_wait_seconds.append(wait_seconds)
                        maximum_reader_queue_depth = max(
                            maximum_reader_queue_depth,
                            reader_queue.qsize(),
                        )
                    timing("reader_enqueue_wait", wait_seconds, item.patch)
                    return True
                except Full:
                    if stop.is_set():
                        return False

        def put_reader_sentinel() -> None:
            while True:
                try:
                    reader_queue.put(reader_sentinel, timeout=0.05)
                    return
                except Full:
                    continue

        def put_writer(item: _ComputedPatch) -> bool:
            nonlocal maximum_writer_queue_depth
            wait_started = time.perf_counter()
            while True:
                try:
                    writer_queue.put(item, timeout=0.05)
                    wait_seconds = time.perf_counter() - wait_started
                    with metrics_lock:
                        writer_enqueue_wait_seconds.append(wait_seconds)
                        maximum_writer_queue_depth = max(
                            maximum_writer_queue_depth,
                            writer_queue.qsize(),
                        )
                    timing("writer_enqueue_wait", wait_seconds, item.patch)
                    return True
                except Full:
                    if stop.is_set():
                        return False

        def put_writer_sentinel() -> None:
            while True:
                try:
                    writer_queue.put(writer_sentinel, timeout=0.05)
                    return
                except Full:
                    continue

        def reader() -> None:
            try:
                if reader_worker_count == 1:
                    for patch in pending:
                        if stop.is_set():
                            break
                        if cancelled():
                            raise PsrPipelineCancelled(
                                "PSR generation was cancelled"
                            )
                        item = read_patch(
                            patch,
                            acquire_decoded_slot=acquire_slot,
                            release_decoded_slot=release_slot,
                        )
                        if stop.is_set():
                            if item.owns_decoded_slot:
                                release_slot(item.horizons)
                            break
                        if not put_reader(item):
                            if item.owns_decoded_slot:
                                release_slot(item.horizons)
                            break
                else:
                    patch_iterator = iter(pending)
                    futures: deque[Future[_ReadPatch]] = deque()
                    with ThreadPoolExecutor(
                        max_workers=reader_worker_count,
                        thread_name_prefix="psr-decompressor",
                    ) as executor:

                        def submit_one() -> bool:
                            try:
                                patch = next(patch_iterator)
                            except StopIteration:
                                return False
                            futures.append(
                                executor.submit(
                                    read_patch,
                                    patch,
                                    acquire_decoded_slot=acquire_slot,
                                    release_decoded_slot=release_slot,
                                )
                            )
                            return True

                        for _ in range(reader_worker_count):
                            if not submit_one():
                                break
                        try:
                            while futures:
                                item = futures.popleft().result()
                                if stop.is_set():
                                    if item.owns_decoded_slot:
                                        release_slot(item.horizons)
                                    break
                                if not put_reader(item):
                                    if item.owns_decoded_slot:
                                        release_slot(item.horizons)
                                    break
                                submit_one()
                        finally:
                            while futures:
                                try:
                                    unused = futures.popleft().result()
                                except BaseException:
                                    continue
                                if unused.owns_decoded_slot:
                                    release_slot(unused.horizons)
            except BaseException as error:
                fail(error)
            finally:
                put_reader_sentinel()

        def writer() -> None:
            try:
                with product.batch_writer(durable_batch_size) as batch_writer:
                    while True:
                        wait_started = time.perf_counter()
                        item = writer_queue.get()
                        wait_seconds = time.perf_counter() - wait_started
                        try:
                            if item is writer_sentinel:
                                record_checkpoint(
                                    batch_writer.checkpoint_with_timings()
                                )
                                return
                            assert isinstance(item, _ComputedPatch)
                            with metrics_lock:
                                writer_dequeue_wait_seconds.append(wait_seconds)
                            timing("writer_dequeue_wait", wait_seconds, item.patch)
                            if stop.is_set():
                                continue
                            if cancelled():
                                raise PsrPipelineCancelled(
                                    "PSR generation was cancelled"
                                )
                            write_patch(batch_writer, item)
                        finally:
                            writer_queue.task_done()
            except BaseException as error:
                fail(error)

        reader_thread = threading.Thread(target=reader, name="psr-horizon-reader")
        writer_thread = threading.Thread(target=writer, name="psr-output-writer")
        writer_thread.start()
        reader_thread.start()
        while True:
            wait_started = time.perf_counter()
            item = reader_queue.get()
            wait_seconds = time.perf_counter() - wait_started
            try:
                if item is reader_sentinel:
                    break
                assert isinstance(item, _ReadPatch)
                with metrics_lock:
                    cuda_dequeue_wait_seconds.append(wait_seconds)
                timing("cuda_dequeue_wait", wait_seconds, item.patch)
                if stop.is_set():
                    if item.owns_decoded_slot:
                        release_slot(item.horizons)
                    continue
                try:
                    if cancelled():
                        raise PsrPipelineCancelled("PSR generation was cancelled")
                    computed = compute_patch(item)
                    if cancelled():
                        raise PsrPipelineCancelled("PSR generation was cancelled")
                    put_writer(computed)
                except BaseException as error:
                    fail(error)
                finally:
                    if item.owns_decoded_slot:
                        release_slot(item.horizons)
            finally:
                reader_queue.task_done()
        reader_thread.join()
        put_writer_sentinel()
        writer_thread.join()
        metrics = PsrPipelineMetrics(
            mode="bounded",
            reader_worker_count=reader_worker_count,
            decoded_horizon_capacity=decoded_horizon_capacity,
            writer_queue_capacity=writer_queue_capacity,
            maximum_live_decoded_horizons=maximum_live_decoded_horizons,
            maximum_reader_queue_depth=maximum_reader_queue_depth,
            maximum_writer_queue_depth=maximum_writer_queue_depth,
            reader_enqueue_wait_seconds=tuple(reader_enqueue_wait_seconds),
            cuda_dequeue_wait_seconds=tuple(cuda_dequeue_wait_seconds),
            writer_enqueue_wait_seconds=tuple(writer_enqueue_wait_seconds),
            writer_dequeue_wait_seconds=tuple(writer_dequeue_wait_seconds),
            durable_batch_size=durable_batch_size,
            maximum_uncheckpointed_patches=maximum_uncheckpointed_patches,
            host_horizon_buffer_kind=(
                selected_host_horizon_buffers
                if horizon_buffer_pool is not None
                else "allocated"
            ),
            preallocated_host_horizon_bytes=(
                decoded_horizon_capacity * 128 * 128 * 1440 * 4
                if horizon_buffer_pool is not None
                else 0
            ),
        )
        if errors:
            if metrics_callback is not None:
                metrics_callback(metrics)
            raise errors[0]
    if metrics_callback is not None:
        metrics_callback(metrics)
    if cancelled():
        raise PsrPipelineCancelled("PSR generation was cancelled")
    finalize_started = time.perf_counter()
    result = product.finalize()
    timing("finalize", time.perf_counter() - finalize_started)
    timing("pipeline_total", time.perf_counter() - pipeline_started)
    report(None, "complete")
    return result
