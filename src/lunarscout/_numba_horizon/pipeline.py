"""Bounded, streaming production-pipeline prototype for Numba horizons."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
import threading
import time
from typing import Any, TextIO, TypeAlias

import numpy as np
import numpy.typing as npt

from .file_format import HorizonTileStore, PATCH_SIZE


CancellationCheck: TypeAlias = Callable[[], bool]


class HorizonPipelineCancelled(RuntimeError):
    """Raised when cancellation is observed between bounded pipeline units."""


@dataclass(frozen=True, slots=True)
class PatchDescriptor:
    index: int
    tile_x: int
    tile_y: int
    patch_x: int
    patch_y: int
    width: int
    height: int

    @property
    def kernel_width(self) -> int:
        """Return the fixed production kernel/file width."""
        return PATCH_SIZE

    @property
    def kernel_height(self) -> int:
        """Return the fixed production kernel/file height."""
        return PATCH_SIZE


@dataclass(frozen=True, slots=True)
class HorizonProgress:
    processed_patches: int
    total_patches: int
    percent: float
    stage: str
    message: str
    file_name: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_paths: tuple[Path, ...]
    skipped_patches: int
    wall_seconds: float
    preparation_seconds: float
    compute_seconds: float
    finalization_seconds: float
    write_seconds: float
    maximum_prepared_queue_depth: int
    maximum_writer_queue_depth: int
    producer_enqueue_wait_seconds: tuple[float, ...]
    consumer_dequeue_wait_seconds: tuple[float, ...]
    writer_enqueue_wait_seconds: tuple[float, ...]
    writer_dequeue_wait_seconds: tuple[float, ...]


@dataclass(slots=True)
class _PreparedPatch:
    patch: PatchDescriptor
    payload: Any
    preparation_seconds: float


@dataclass(slots=True)
class _ComputedPatch:
    patch: PatchDescriptor
    payload: Any
    compute_seconds: float


def enumerate_patches(
    width: int,
    height: int,
    *,
    include_partial_edges: bool = True,
) -> list[PatchDescriptor]:
    """Enumerate row-major 128-pixel patches, retaining partial DEM edges."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("DEM dimensions must be positive integers")
    if not include_partial_edges and (width % PATCH_SIZE or height % PATCH_SIZE):
        raise ValueError("DEM dimensions must be even multiples of 128")
    columns = (width + PATCH_SIZE - 1) // PATCH_SIZE
    rows = (height + PATCH_SIZE - 1) // PATCH_SIZE
    patches = []
    for patch_y in range(rows):
        for patch_x in range(columns):
            tile_x = patch_x * PATCH_SIZE
            tile_y = patch_y * PATCH_SIZE
            patches.append(
                PatchDescriptor(
                    index=len(patches),
                    tile_x=tile_x,
                    tile_y=tile_y,
                    patch_x=patch_x,
                    patch_y=patch_y,
                    width=min(PATCH_SIZE, width - tile_x),
                    height=min(PATCH_SIZE, height - tile_y),
                )
            )
    return patches


def run_bounded_pipeline(
    patches: Sequence[PatchDescriptor],
    *,
    store: HorizonTileStore,
    prepare_patch: Callable[[PatchDescriptor], Any],
    processor_factory: Callable[
        [int], Callable[[PatchDescriptor, Any], Any]
    ],
    finalize_patch: Callable[
        [PatchDescriptor, Any], npt.NDArray[np.float32]
    ] | None = None,
    observer_elevation_m: float = 0.0,
    compress: bool = False,
    skip_existing: bool = True,
    prepared_queue_capacity: int = 1,
    writer_queue_capacity: int | None = None,
    worker_count: int = 1,
    progress_callback: Callable[[HorizonProgress], None] | None = None,
    cancellation_requested: CancellationCheck | None = None,
    progress_stream: TextIO | None = None,
) -> PipelineResult:
    """Prepare, compute, and immediately stage/write patches with bounded memory."""
    if prepared_queue_capacity < 1:
        raise ValueError("prepared_queue_capacity must be positive")
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if writer_queue_capacity is not None and writer_queue_capacity < 1:
        raise ValueError("writer_queue_capacity must be positive or None")
    started = time.perf_counter()
    cancellation_requested = cancellation_requested or (lambda: False)
    if cancellation_requested():
        raise HorizonPipelineCancelled("Horizon generation was cancelled.")

    pending = []
    skipped = 0
    for patch in patches:
        existing = store.find_existing_path(
            patch.tile_y, patch.tile_x, observer_elevation_m
        ) if skip_existing else None
        if existing is None:
            pending.append(patch)
        else:
            skipped += 1

    progress_lock = threading.Lock()
    processed = 0
    output_paths: list[Path] = []
    preparation_seconds = 0.0
    compute_seconds = 0.0
    finalization_seconds = 0.0
    write_seconds = 0.0
    maximum_queue_depth = 0
    maximum_writer_queue_depth = 0
    producer_enqueue_wait_seconds: list[float] = []
    consumer_dequeue_wait_seconds: list[float] = []
    writer_enqueue_wait_seconds: list[float] = []
    writer_dequeue_wait_seconds: list[float] = []
    metrics_lock = threading.Lock()
    finalize_patch = finalize_patch or (lambda _patch, payload: payload)

    def emit(item: HorizonProgress) -> None:
        if progress_callback is not None:
            progress_callback(item)
        if progress_stream is not None:
            progress_stream.write(
                f"[{item.stage}] {item.percent:5.1f}% {item.message}\n"
            )
            progress_stream.flush()

    total = len(pending)
    emit(HorizonProgress(0, total, 10.0, "prepare_patches", "Preparing horizon patch pipeline."))
    if total == 0:
        emit(HorizonProgress(0, 0, 100.0, "complete", "No horizon patches need to be generated."))
        return PipelineResult(
            (), skipped, time.perf_counter() - started,
            0.0, 0.0, 0.0, 0.0, 0, 0, (), (), (), (),
        )
    emit(HorizonProgress(0, total, 15.0, "process_patches", "Starting horizon patch generation."))

    queue: Queue[_PreparedPatch | object] = Queue(maxsize=prepared_queue_capacity)
    sentinel = object()
    writer_queue: Queue[_ComputedPatch | object] | None = None
    writer_sentinel = object()
    if writer_queue_capacity is not None:
        writer_queue = Queue(maxsize=writer_queue_capacity)
    stop = threading.Event()
    error_lock = threading.Lock()
    errors: list[BaseException] = []

    def fail(error: BaseException) -> None:
        with error_lock:
            if not errors:
                errors.append(error)
        stop.set()

    def put(item: _PreparedPatch | object) -> tuple[bool, float]:
        started_wait = time.perf_counter()
        while True:
            try:
                queue.put(item, timeout=0.05)
                return True, time.perf_counter() - started_wait
            except Full:
                if stop.is_set():
                    return False, time.perf_counter() - started_wait

    def put_sentinel() -> None:
        while True:
            try:
                queue.put(sentinel, timeout=0.05)
                return
            except Full:
                # Workers remain alive after an item failure specifically so they
                # can drain bounded work and observe these termination markers.
                continue

    def put_writer(item: _ComputedPatch) -> bool:
        nonlocal maximum_writer_queue_depth
        assert writer_queue is not None
        started_wait = time.perf_counter()
        while True:
            try:
                writer_queue.put(item, timeout=0.05)
                with metrics_lock:
                    writer_enqueue_wait_seconds.append(
                        time.perf_counter() - started_wait
                    )
                    maximum_writer_queue_depth = max(
                        maximum_writer_queue_depth, writer_queue.qsize()
                    )
                return True
            except Full:
                if stop.is_set():
                    return False

    def put_writer_sentinel() -> None:
        assert writer_queue is not None
        while True:
            try:
                writer_queue.put(writer_sentinel, timeout=0.05)
                return
            except Full:
                continue

    def producer() -> None:
        nonlocal preparation_seconds, maximum_queue_depth
        try:
            for patch in pending:
                if stop.is_set():
                    break
                if cancellation_requested():
                    raise HorizonPipelineCancelled("Horizon generation was cancelled.")
                item_started = time.perf_counter()
                payload = prepare_patch(patch)
                elapsed = time.perf_counter() - item_started
                preparation_seconds += elapsed
                enqueued, enqueue_wait = put(_PreparedPatch(patch, payload, elapsed))
                producer_enqueue_wait_seconds.append(enqueue_wait)
                if not enqueued:
                    break
                maximum_queue_depth = max(maximum_queue_depth, queue.qsize())
        except BaseException as error:
            fail(error)
        finally:
            # Workers drain after failure, so bounded sentinel insertion terminates.
            for _ in range(worker_count):
                put_sentinel()

    def worker(worker_id: int) -> None:
        nonlocal processed, compute_seconds, finalization_seconds, write_seconds
        try:
            processor = processor_factory(worker_id)
        except BaseException as error:
            fail(error)
            processor = None
        while True:
            dequeue_started = time.perf_counter()
            item = queue.get()
            dequeue_wait = time.perf_counter() - dequeue_started
            try:
                if item is sentinel:
                    return
                assert isinstance(item, _PreparedPatch)
                with metrics_lock:
                    consumer_dequeue_wait_seconds.append(dequeue_wait)
                if stop.is_set() or processor is None:
                    continue
                try:
                    if cancellation_requested():
                        raise HorizonPipelineCancelled("Horizon generation was cancelled.")
                    compute_started = time.perf_counter()
                    computed = processor(item.patch, item.payload)
                    compute_elapsed = time.perf_counter() - compute_started
                    if cancellation_requested():
                        raise HorizonPipelineCancelled("Horizon generation was cancelled.")
                    if writer_queue is not None:
                        if not put_writer(
                            _ComputedPatch(item.patch, computed, compute_elapsed)
                        ):
                            continue
                        with metrics_lock:
                            compute_seconds += compute_elapsed
                        continue
                    finalize_started = time.perf_counter()
                    degrees = finalize_patch(item.patch, computed)
                    finalize_elapsed = time.perf_counter() - finalize_started
                    write_started = time.perf_counter()
                    path = store.write(
                        item.patch.tile_y,
                        item.patch.tile_x,
                        observer_elevation_m,
                        degrees,
                        compress=compress,
                        valid_width=item.patch.width,
                        valid_height=item.patch.height,
                    )
                    write_elapsed = time.perf_counter() - write_started
                    with progress_lock:
                        compute_seconds += compute_elapsed
                        finalization_seconds += finalize_elapsed
                        write_seconds += write_elapsed
                        output_paths.append(path)
                        processed += 1
                        percent = processed * 100.0 / total
                        emit(
                            HorizonProgress(
                                processed,
                                total,
                                percent,
                                "process_patches",
                                f"Generated {processed}/{total} horizon patches.",
                                path.name,
                            )
                        )
                except BaseException as error:
                    fail(error)
            finally:
                queue.task_done()

    def writer() -> None:
        nonlocal processed, finalization_seconds, write_seconds
        assert writer_queue is not None
        while True:
            dequeue_started = time.perf_counter()
            item = writer_queue.get()
            dequeue_wait = time.perf_counter() - dequeue_started
            try:
                if item is writer_sentinel:
                    return
                assert isinstance(item, _ComputedPatch)
                with metrics_lock:
                    writer_dequeue_wait_seconds.append(dequeue_wait)
                if stop.is_set():
                    continue
                try:
                    if cancellation_requested():
                        raise HorizonPipelineCancelled("Horizon generation was cancelled.")
                    finalize_started = time.perf_counter()
                    degrees = finalize_patch(item.patch, item.payload)
                    finalize_elapsed = time.perf_counter() - finalize_started
                    if cancellation_requested():
                        raise HorizonPipelineCancelled("Horizon generation was cancelled.")
                    write_started = time.perf_counter()
                    path = store.write(
                        item.patch.tile_y,
                        item.patch.tile_x,
                        observer_elevation_m,
                        degrees,
                        compress=compress,
                        valid_width=item.patch.width,
                        valid_height=item.patch.height,
                    )
                    write_elapsed = time.perf_counter() - write_started
                    with progress_lock:
                        finalization_seconds += finalize_elapsed
                        write_seconds += write_elapsed
                        output_paths.append(path)
                        processed += 1
                        percent = processed * 100.0 / total
                        emit(
                            HorizonProgress(
                                processed,
                                total,
                                percent,
                                "process_patches",
                                f"Generated {processed}/{total} horizon patches.",
                                path.name,
                            )
                        )
                except BaseException as error:
                    fail(error)
            finally:
                writer_queue.task_done()

    producer_thread = threading.Thread(target=producer, name="horizon-segment-producer")
    worker_threads = [
        threading.Thread(target=worker, args=(index,), name=f"horizon-gpu-worker-{index}")
        for index in range(worker_count)
    ]
    writer_thread = None
    if writer_queue is not None:
        writer_thread = threading.Thread(target=writer, name="horizon-output-writer")
        writer_thread.start()
    for thread in worker_threads:
        thread.start()
    producer_thread.start()
    producer_thread.join()
    for thread in worker_threads:
        thread.join()
    if writer_queue is not None:
        put_writer_sentinel()
        assert writer_thread is not None
        writer_thread.join()
    if errors:
        raise errors[0]
    emit(HorizonProgress(total, total, 100.0, "complete", "Horizons generation complete."))
    return PipelineResult(
        tuple(output_paths), skipped, time.perf_counter() - started,
        preparation_seconds, compute_seconds, finalization_seconds, write_seconds,
        maximum_queue_depth, maximum_writer_queue_depth,
        tuple(producer_enqueue_wait_seconds),
        tuple(consumer_dequeue_wait_seconds),
        tuple(writer_enqueue_wait_seconds),
        tuple(writer_dequeue_wait_seconds),
    )


def make_numba_processor_factory(
    pyramids: Sequence[Any],
    *,
    observer_elevation_m: float = 0.0,
    device_id: int = 0,
) -> Callable[[int], Callable[[PatchDescriptor, Any], npt.NDArray[np.float32]]]:
    """Bind prepared ``SegmentTensor`` items to the correctness-approved kernel."""

    def factory(_worker_id: int):
        # Imports and CUDA selection remain explicit and occur only when a worker
        # starts, never during ordinary Lunarscout or pipeline-module import.
        from .cuda_backend import CudaSession
        from .generator import generate_patch_horizons

        session = CudaSession(device_id=device_id)

        def process(patch: PatchDescriptor, segments: Any) -> npt.NDArray[np.float32]:
            configuration = segments.configuration
            if (configuration.tile_width, configuration.tile_height) != (
                patch.kernel_width,
                patch.kernel_height,
            ):
                raise ValueError(
                    "prepared segments must use the fixed 128x128 production shape"
                )
            result = generate_patch_horizons(
                session,
                segments,
                pyramids,
                tile_column=patch.tile_x,
                tile_row=patch.tile_y,
                observer_elevation_m=observer_elevation_m,
            )
            return result.degrees()

        return process

    return factory
