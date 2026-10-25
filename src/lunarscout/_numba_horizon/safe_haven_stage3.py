"""Stage 3 as a pipeline with bounded queues (private).

``safe_haven_blocks.run_pass_two`` plugs this in as its executor (``pipeline=PipelineSettings(...)``).
Where the block loop does plan, decode, light and flood one after another, here every step is a
stage with its own workers, and the stages are joined by bounded queues, so each can be tuned by
its worker count:

    cells -> PLAN (processes) -> units requested -> LIGHT (threads) -> units ready
                                      |                  ^  decode cache
                                      +-> jobs waiting --+-> FLOOD (threads) -> rows -> blocks -> checkpoint

* **PLAN** is pass 1 for one cell: seeds, flood boundaries, the (patch, chunk) lighting units each
  cell-month needs, and the passable mask. It is Python, so it runs in forked processes.
* **LIGHT** turns a unit, 24 hours of flood steps of one 128 x 128 patch, into packed lit bits
  (CPU lightmap with threshold and bit packing fused in one compiled pass; the slot a GPU kernel
  can take). A patch's horizons are decoded on demand into a small cache and freed, instead of a
  whole block's staying resident.
* **FLOOD** floods a cell-month's seeds once every unit of its window is lit.

Units are shared by every cell-month that needs them and freed when the last one has been flooded
(reference counted), so memory is the units in flight, not a block of horizons. The number of cells
in flight and the bytes of units requested are bounded, which is the back pressure on PLAN.

The table is the same as the block loop's for any worker counts and any bounds: cells are planned
and flooded independently and rows are sorted at the end.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import signal
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np

from .file_format import PATCH_SIZE, read_patch_horizons
from .safe_haven_flood import OPERATING_FUZZ
from .safe_haven_litstore import LitStore, light_patch_chunks_fused
from .safe_haven_windows import PlanCounts, cell_window, iter_cell_months

logger = logging.getLogger(__name__)

Cell = tuple[int, int]
Block = tuple[int, int]
Patch = tuple[int, int]
UnitKey = tuple[int, int, int]

_PENDING, _INFLIGHT, _READY = 0, 1, 2


@dataclass(frozen=True)
class PipelineSettings:
    """Worker counts and bounds of the stage-3 pipeline."""

    plan_workers: int = 4
    light_workers: int = 12
    numba_threads_per_light_worker: int | None = None     # default: the CPU count divided by light_workers
    flood_workers: int = 2
    decode_cache_patches: int = 32                          # decoded patches kept (94 MB each, float32)
    cells_in_flight: int = 24                               # cells planned and not yet flooded
    store_budget_bytes: int = 8 * 2**30                     # lit units requested, in bytes
    status_seconds: float = 0.0                             # log a status line this often (0: never)
    #: Where lighting runs: ``"cpu"`` (never probes CUDA), ``"cuda"`` (an unavailable device is an error)
    #: or ``"auto"`` (CUDA when it can be set up, otherwise the CPU, with a warning).
    lighting: str = "auto"
    gpu_device_id: int = 0
    gpu_light_workers: int = 6          # threads feeding the GPU: they decode horizons while it computes
    gpu_chunks_per_batch: int = 4       # chunks (days) per kernel launch

    def __post_init__(self) -> None:
        for name in ("plan_workers", "light_workers", "flood_workers", "decode_cache_patches",
                     "cells_in_flight", "store_budget_bytes"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.status_seconds >= 0.0:
            raise ValueError("status_seconds must not be negative")
        if self.lighting not in ("auto", "cpu", "cuda"):
            raise ValueError("lighting must be 'auto', 'cpu' or 'cuda'")
        for name in ("gpu_light_workers", "gpu_chunks_per_batch"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if int(self.gpu_device_id) != self.gpu_device_id or self.gpu_device_id < 0:
            raise ValueError("gpu_device_id must be a non-negative integer")
        threads = self.numba_threads_per_light_worker
        if threads is not None and (int(threads) != threads or threads < 1):
            raise ValueError("numba_threads_per_light_worker must be a positive integer")


def _make_gpu_session(settings: PipelineSettings, steps_per_chunk: int):
    from .safe_haven_lit_cuda import LitCudaSession

    return LitCudaSession(device_id=settings.gpu_device_id, steps_per_chunk=steps_per_chunk,
                          chunks_per_batch=settings.gpu_chunks_per_batch)


def select_lighting(settings: PipelineSettings, steps_per_chunk: int):
    """The CUDA lighting session, or ``None`` for the CPU. ``"cpu"`` never touches CUDA; ``"cuda"``
    raises when the device cannot be set up; ``"auto"`` falls back to the CPU with a warning."""
    from .cuda_backend import CudaBackendError

    if settings.lighting == "cpu":
        return None
    try:
        return _make_gpu_session(settings, steps_per_chunk)
    except (CudaBackendError, ImportError) as error:
        if settings.lighting == "cuda":
            raise
        logger.warning("CUDA lighting is not available (%s: %s); lighting on the CPU.",
                       type(error).__name__, error)
        return None


def default_numba_threads(light_workers: int, requested: int | None = None) -> int:
    """Numba threads for each LIGHT worker: the request, or the CPU count divided among them."""
    if requested is not None:
        return int(requested)
    return max(1, (os.cpu_count() or light_workers) // light_workers)


# ---------------------------------------------------------------------------
# Decoded-horizon cache
# ---------------------------------------------------------------------------


class HorizonCache:
    """Decoded patches kept for a short while, least recently used out first. Concurrent callers
    asking for the same patch share one decode."""

    def __init__(self, loader, capacity: int) -> None:
        self._loader = loader
        self._capacity = int(capacity)
        self._lock = threading.Lock()
        self._items: OrderedDict[Patch, object] = OrderedDict()
        self._patch_locks: dict[Patch, threading.Lock] = {}
        self.loads = 0
        self.peak = 0

    def get(self, patch: Patch):
        with self._lock:
            if patch in self._items:
                self._items.move_to_end(patch)
                return self._items[patch]
            patch_lock = self._patch_locks.setdefault(patch, threading.Lock())
        with patch_lock:
            with self._lock:
                if patch in self._items:
                    self._items.move_to_end(patch)
                    return self._items[patch]
            value = self._loader(patch)
            with self._lock:
                self._items[patch] = value
                self.loads += 1
                while len(self._items) > self._capacity:
                    evicted, _ = self._items.popitem(last=False)
                    self._patch_locks.pop(evicted, None)
                self.peak = max(self.peak, len(self._items))
            return value


class _Locked:
    """Serialises a vector source: CSPICE is not thread safe."""

    def __init__(self, source) -> None:
        self._source = source
        self._lock = threading.Lock()

    def __call__(self, stamps):
        with self._lock:
            return self._source(stamps)


# ---------------------------------------------------------------------------
# PLAN: pass 1 for one cell, in forked processes
# ---------------------------------------------------------------------------

_PLAN: dict = {}


def _plan_init(spice_kernels: list[str]) -> None:
    from .safe_haven_blocks import _refurnish_spice

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _refurnish_spice(spice_kernels)


def _plan_cell(cell: Cell):
    """``(cell, window, patch_window, [(record, packed passable)], plan counts)`` for one cell."""
    from .safe_haven_blocks import _passable_packed, _window_patches

    a = _PLAN["arguments"]
    report: list = []
    records = list(iter_cell_months(
        dem=a["dem"], horizon_store=a["horizon_store"], stage1=a["stage1"],
        times=a["time"].times_us.astype("datetime64[us]"),
        sun_vectors_at=a["sun_vectors_at"], earth_vectors_at=a["earth_vectors_at"],
        parameters=a["parameters"], observer_height_m=a["observer_height_m"],
        earth_elevation_threshold_deg=a["earth_elevation_threshold_deg"],
        sunlight_fraction_threshold=a["sunlight_fraction_threshold"], merge_gap_hours=a["merge_gap_hours"],
        cells=[cell], report=report, time_context=a["time"],
    ))
    window, _core = cell_window(cell[0], cell[1], cell_size=a["cell_size"], halo=a["halo"],
                                height=a["dem"].height, width=a["dem"].width)
    patch_window = _window_patches(window)
    packed = []
    if records:
        valid = a["stage1"].read_valid_window(*window)
        packed = [(record, _passable_packed(valid[record["month"]], patch_window)) for record in records]
    return cell, window, patch_window, packed, asdict(report[0].counts)


# ---------------------------------------------------------------------------
# The coordinator: units, jobs, cells and blocks
# ---------------------------------------------------------------------------


class _Unit:
    __slots__ = ("state", "refs", "waiters")

    def __init__(self) -> None:
        self.state = _PENDING
        self.refs = 0
        self.waiters: list = []


class _Job:
    """One cell-month: flood its seeds when ``waiting`` reaches zero."""

    __slots__ = ("cell", "month", "record", "window", "patch_window", "passable", "keys", "waiting", "block")

    def __init__(self, cell, month, record, window, patch_window, passable, keys, block) -> None:
        self.cell, self.month, self.record = cell, month, record
        self.window, self.patch_window, self.passable = window, patch_window, passable
        self.keys, self.waiting, self.block = keys, 0, block


class _BlockState:
    def __init__(self, cells_left: int) -> None:
        from .safe_haven_blocks import PassTwoCounts

        self.cells_left = cells_left
        self.rows: list[dict] = []
        self.plan = PlanCounts()
        self.flood = PassTwoCounts(blocks_run=1)


class _Coordinator:
    def __init__(self, *, store: LitStore, settings: PipelineSettings, by_block, finish) -> None:
        self.lock = threading.RLock()
        self.light_cv = threading.Condition(self.lock)
        self.budget_cv = threading.Condition(self.lock)
        self.store = store
        self.settings = settings
        self.finish = finish
        self.finish_lock = threading.Lock()
        self.unit_bytes = int(np.prod(store.chunk_shape)) * 8
        self.units: dict[UnitKey, _Unit] = {}
        self.pending: OrderedDict[Patch, set[int]] = OrderedDict()
        self.ready_jobs: queue.Queue = queue.Queue()
        self.blocks = {block: _BlockState(len(cells)) for block, cells in by_block.items()}
        self.cell_jobs: dict[Cell, int] = {}
        self.block_of: dict[Cell, Block] = {cell: block for block, cells in by_block.items() for cell in cells}
        self.requested_bytes = 0
        self.cells_in_flight = 0
        self.cell_slots = threading.Semaphore(settings.cells_in_flight)
        self.error: BaseException | None = None
        self.stop = False
        self.outstanding_cells = 0
        self.feeding_done = False
        self.all_done = threading.Event()
        self.units_computed = 0
        self.peak_requested_bytes = 0

    def status(self) -> dict:
        """A snapshot of where work is waiting, for the status line."""
        with self.lock:
            return {
                "cells_in_flight": self.cells_in_flight,
                "patches_waiting_to_be_lit": len(self.pending),
                "units_waiting_to_be_lit": sum(len(c) for c in self.pending.values()),
                "jobs_ready_to_flood": self.ready_jobs.qsize(),
                "unit_gb_requested": self.requested_bytes / 2**30,
                "units_lit": self.units_computed,
                "blocks_left": sum(1 for b in self.blocks.values() if b.cells_left > 0),
            }

    # ---- failure and shutdown ----

    def fail(self, error: BaseException) -> None:
        with self.lock:
            if self.error is None:
                self.error = error
            self.stop = True
            self.light_cv.notify_all()
            self.budget_cv.notify_all()
        self.all_done.set()

    def _check_done_locked(self) -> None:
        if self.feeding_done and self.outstanding_cells == 0:
            self.all_done.set()
            self.light_cv.notify_all()

    # ---- feeding (back pressure) ----

    def acquire_cell_slot(self) -> bool:
        """Wait for a free slot and for the unit memory to fit; ``False`` if stopping."""
        while True:
            if self.stop:
                return False
            if self.cell_slots.acquire(timeout=0.2):
                break
        with self.lock:
            while (not self.stop and self.requested_bytes >= self.settings.store_budget_bytes
                   and self.cells_in_flight > 0):
                self.budget_cv.wait(0.5)
            if self.stop:
                self.cell_slots.release()
                return False
            self.cells_in_flight += 1
            self.outstanding_cells += 1
        return True

    def feeding_finished(self) -> None:
        with self.lock:
            self.feeding_done = True
            self._check_done_locked()

    # ---- a planned cell arrives ----

    def add_cell(self, result) -> None:
        cell, window, patch_window, packed, plan_counts = result
        actions = []
        with self.lock:
            block = self.blocks[self.block_of[cell]]
            for name, value in plan_counts.items():
                setattr(block.plan, name, getattr(block.plan, name) + value)
            self.cell_jobs[cell] = len(packed)
            if not packed:
                actions.extend(self._cell_done_locked(cell))
            for record, passable in packed:
                keys = [
                    (pr, pc, chunk)
                    for pr in range(patch_window[0], patch_window[0] + patch_window[2])
                    for pc in range(patch_window[1], patch_window[1] + patch_window[3])
                    for chunk in record["chunks"]
                ]
                job = _Job(cell, record["month"], record, window, patch_window, passable, keys, block)
                self._request_locked(job)
        self._run(actions)

    def _request_locked(self, job: _Job) -> None:
        waiting = 0
        for key in job.keys:
            unit = self.units.get(key)
            if unit is None:
                unit = _Unit()
                unit.refs = 1
                self.units[key] = unit
                self.pending.setdefault((key[0], key[1]), set()).add(key[2])
                self.requested_bytes += self.unit_bytes
                self.peak_requested_bytes = max(self.peak_requested_bytes, self.requested_bytes)
                job.block.flood.lit_units_computed += 1
                unit.waiters.append(job)
                waiting += 1
            else:
                unit.refs += 1
                job.block.flood.lit_units_reused += 1
                if unit.state != _READY:
                    unit.waiters.append(job)
                    waiting += 1
        job.waiting = waiting
        if waiting == 0:
            self.ready_jobs.put(job)
        self.light_cv.notify_all()

    # ---- LIGHT ----

    def take_lighting_task(self):
        """The next patch with units to light, and its chunks; ``None`` when there is no more work."""
        with self.lock:
            while not self.pending:
                if self.stop or self.all_done.is_set():
                    return None
                self.light_cv.wait(0.5)
            patch, chunks = self.pending.popitem(last=False)
            for chunk in chunks:
                self.units[(patch[0], patch[1], chunk)].state = _INFLIGHT
            return patch, sorted(chunks)

    def publish(self, patch: Patch, bits_by_chunk) -> None:
        with self.lock:
            self.units_computed += len(bits_by_chunk)
            for chunk, bits in bits_by_chunk.items():
                key = (patch[0], patch[1], chunk)
                unit = self.units.get(key)
                if unit is None:
                    continue
                self.store.put(patch[0], patch[1], chunk, bits)
                unit.state = _READY
                for job in unit.waiters:
                    job.waiting -= 1
                    if job.waiting == 0:
                        self.ready_jobs.put(job)
                unit.waiters = []

    # ---- FLOOD results ----

    def job_done(self, job: _Job, rows: list[dict], counts) -> None:
        actions = []
        with self.lock:
            block = job.block
            block.rows.extend(rows)
            for name in ("cell_months_flooded", "seeds_flooded", "seeds_boundary_mismatch",
                         "seeds_without_both_sides", "cell_months_without_valid_seed"):
                setattr(block.flood, name, getattr(block.flood, name) + getattr(counts, name))
            for key in job.keys:
                unit = self.units[key]
                unit.refs -= 1
                if unit.refs == 0:
                    del self.units[key]
                    self.store.discard(*key)
                    self.requested_bytes -= self.unit_bytes
            self.cell_jobs[job.cell] -= 1
            if self.cell_jobs[job.cell] == 0:
                actions.extend(self._cell_done_locked(job.cell))
            self.budget_cv.notify_all()
        self._run(actions)

    def _cell_done_locked(self, cell: Cell) -> list:
        self.cells_in_flight -= 1
        self.outstanding_cells -= 1
        self.cell_slots.release()
        block_key = self.block_of[cell]
        block = self.blocks[block_key]
        block.cells_left -= 1
        actions = []
        if block.cells_left == 0:
            block.flood.rows = len(block.rows)
            actions.append((block_key, asdict(block.plan), asdict(block.flood), block.rows))
        self._check_done_locked()
        self.budget_cv.notify_all()
        return actions

    def _run(self, actions) -> None:
        for block_key, plan, flood, rows in actions:
            with self.finish_lock:
                self.finish(block_key, plan, flood, rows)


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------


def _run_blocks_in_pipeline(pending, by_block, run_arguments, finish, *, settings: PipelineSettings,
                            stop_after_blocks, details: dict | None = None):
    """Run the ``pending`` blocks through the pipeline, calling ``finish`` for each completed block.

    Returns ``(stopped_early, horizon_loads, peak_resident)``. The first interrupt stops feeding after
    the cells of the block being fed and lets everything in flight finish; a second stops at once.
    """
    if not pending or stop_after_blocks == 0:
        if details is not None:
            details["lighting"] = "none (every block was already in the checkpoint)" if not pending else "none"
        return bool(pending), 0, 0

    import numba

    from .lightmap_cpu import LightmapCpuSession
    from .safe_haven_blocks import (
        PassTwoCounts,
        _flood_record,
        _graceful_interrupt,
        _loaded_spice_kernels,
    )

    arguments = dict(run_arguments)
    dem, grid = arguments["dem"], arguments["grid"]
    parameters, flood_steps = arguments["parameters"], arguments["flood_steps"]
    plan_blocks = list(pending if stop_after_blocks is None else pending[:stop_after_blocks])
    stopped_early = len(plan_blocks) < len(pending)
    plan_by_block = {block: by_block[block] for block in plan_blocks}
    # PLAN processes first, before any thread exists (forking a threaded process is unsafe).
    _PLAN.clear()
    _PLAN["arguments"] = dict(run_arguments)
    pool = multiprocessing.get_context("fork").Pool(
        settings.plan_workers, initializer=_plan_init, initargs=(_loaded_spice_kernels(),))
    light_vectors = _Locked(arguments["sun_vectors_at"])                    # the LIGHT threads' source
    gpu = select_lighting(settings, grid.steps_per_chunk)                   # after the fork: CUDA is not fork safe
    if details is not None:
        details["lighting"] = f"cuda device {settings.gpu_device_id}" if gpu is not None else "cpu"
    logger.info("Pipeline lighting: %s", "CUDA device %d, %d feeder threads, %d chunks per launch" % (
        settings.gpu_device_id, settings.gpu_light_workers, settings.gpu_chunks_per_batch)
        if gpu is not None else "CPU, %d threads" % settings.light_workers)

    store = LitStore(grid, (dem.height, dem.width))
    coordinator = _Coordinator(store=store, settings=settings, by_block=plan_by_block, finish=finish)
    cache = HorizonCache(
        lambda patch: read_patch_horizons(
            arguments["horizon_store"], patch[0] * PATCH_SIZE, patch[1] * PATCH_SIZE,
            arguments["observer_height_m"]),
        settings.decode_cache_patches,
    )
    session = LightmapCpuSession()
    light_threads = default_numba_threads(settings.light_workers, settings.numba_threads_per_light_worker)

    def guarded(function):
        def run():
            try:
                function()
            except BaseException as error:             # noqa: BLE001  (shut the pipeline down, raise in the caller)
                coordinator.fail(error)
        return run

    def light_worker() -> None:
        numba.set_num_threads(light_threads)
        while True:
            task = coordinator.take_lighting_task()
            if task is None:
                return
            patch, chunks = task
            horizons = cache.get(patch)
            if horizons is None:                        # a patch without a horizon tile is never lit
                bits = {chunk: np.zeros(store.chunk_shape, dtype=np.uint64) for chunk in chunks}
            else:
                bits = light_patch_chunks_fused(
                    session, dem, horizons, patch[0], patch[1], grid, chunks, light_vectors,
                    parameters.operating_sunlight_fraction, compiled=True)
            coordinator.publish(patch, bits)

    def gpu_light_worker() -> None:
        """Decode a patch's horizons here (in parallel with the other feeders) and light its units on the
        GPU, which runs one patch at a time."""
        spc = grid.steps_per_chunk
        threshold = parameters.operating_sunlight_fraction - OPERATING_FUZZ
        while True:
            task = coordinator.take_lighting_task()
            if task is None:
                return
            patch, chunks = task
            horizons = cache.get(patch)
            if horizons is None:
                bits = {chunk: np.zeros(store.chunk_shape, dtype=np.uint64) for chunk in chunks}
            else:
                tile_y, tile_x = patch[0] * PATCH_SIZE, patch[1] * PATCH_SIZE
                vectors = [light_vectors(grid.times(np.arange(c * spc, (c + 1) * spc, dtype=np.int64)))
                           for c in chunks]
                units = gpu.light_patch(
                    dem, horizons, vectors, tile_y=tile_y, tile_x=tile_x,
                    valid_height=min(PATCH_SIZE, dem.elevation_m.shape[0] - tile_y),
                    valid_width=min(PATCH_SIZE, dem.elevation_m.shape[1] - tile_x), threshold=threshold)
                bits = dict(zip(chunks, units))
            coordinator.publish(patch, bits)

    def flood_worker() -> None:
        while True:
            try:
                job = coordinator.ready_jobs.get(timeout=0.5)
            except queue.Empty:
                if coordinator.stop or coordinator.all_done.is_set():
                    return
                continue
            counts = PassTwoCounts()
            rows = _flood_record(
                store, job.record, job.window, job.patch_window, job.passable, counts, cell=job.cell,
                month_label=arguments["months"][job.month], grid=grid, parameters=parameters,
                flood_steps=flood_steps, compiled=arguments["compiled"], cell_size=arguments["cell_size"],
                transform=arguments["transform"], dem_height=dem.height, dem_width=dem.width)
            coordinator.job_done(job, rows, counts)

    interrupts_seen: list = []

    def feeder():
        """Cells in snake order, a block at a time; after an interrupt it finishes the block it is in."""
        for block in plan_blocks:
            for cell in plan_by_block[block]:
                if not coordinator.acquire_cell_slot():
                    return
                yield cell
            if interrupts_seen or coordinator.stop:
                return

    def collect() -> None:
        try:
            for result in pool.imap_unordered(_plan_cell, feeder()):
                coordinator.add_cell(result)
        finally:
            coordinator.feeding_finished()

    def status_reporter() -> None:
        while not coordinator.all_done.wait(settings.status_seconds):
            snapshot = coordinator.status()
            logger.info(
                "pipeline: %d cells in flight, %d units (%d patches) waiting to be lit, %d jobs ready to "
                "flood, %.1f GB of units requested, %d units lit, %d decoded patches cached, %d blocks left",
                snapshot["cells_in_flight"], snapshot["units_waiting_to_be_lit"],
                snapshot["patches_waiting_to_be_lit"], snapshot["jobs_ready_to_flood"],
                snapshot["unit_gb_requested"], snapshot["units_lit"], len(cache._items), snapshot["blocks_left"])

    threads = (
        [threading.Thread(target=guarded(light_worker if gpu is None else gpu_light_worker),
                          name=f"light-{i}", daemon=True)
         for i in range(settings.light_workers if gpu is None else settings.gpu_light_workers)]
        + [threading.Thread(target=guarded(flood_worker), name=f"flood-{i}", daemon=True)
           for i in range(settings.flood_workers)]
        + [threading.Thread(target=guarded(collect), name="collect", daemon=True)]
        + ([threading.Thread(target=guarded(status_reporter), name="status", daemon=True)]
           if settings.status_seconds > 0 else [])
    )
    try:
        with _graceful_interrupt() as interrupts:
            interrupts_seen = interrupts
            for thread in threads:
                thread.start()
            while not coordinator.all_done.wait(0.5):
                pass
        if coordinator.error is not None:
            raise coordinator.error
        logger.info(
            "Pipeline finished: %d units lit, peak %d units (%.2f GB) requested at once, %d units still held "
            "(%d in the store), %d patches decoded (at most %d cached).", coordinator.units_computed,
            coordinator.peak_requested_bytes // coordinator.unit_bytes, coordinator.peak_requested_bytes / 2**30,
            len(coordinator.units), len(store), cache.loads, cache.peak)
        unfinished = any(state.cells_left > 0 for state in coordinator.blocks.values())
        return stopped_early or unfinished, cache.loads, cache.peak
    finally:
        coordinator.stop = True
        with coordinator.lock:
            coordinator.light_cv.notify_all()
            coordinator.budget_cv.notify_all()
        for thread in threads:
            thread.join(timeout=30)
        pool.terminate()
        pool.join()
        _PLAN.clear()
