"""Stage 0 of the safe-haven analysis: light every patch once and store it (private).

Stage 0 reads each horizon patch once, lights it for the whole time range and writes the three
thresholded planes of ``safe_haven_lightstore`` to one file per patch. Stages 1 and 3 then read those
files and never touch the horizons. The store is a directory::

    light_store.json              manifest: the settings, the time range and the DEM's identity
    light_store.report.json       what the last run did
    <row:03d>/light_<row:03d>_<column:03d>.lsl    one file per patch that has a horizon tile

A patch without a horizon tile has no file and is never lit. The run is restartable: a file is renamed
into place only when complete, so a rerun skips every patch whose file is present and valid and does
only the rest. A manifest that disagrees with the settings, the time range or the DEM refuses the run
rather than mixing two stores.

Each patch is one task: decode the horizons, light the whole range, compress and write. A pool of
``workers`` threads runs the tasks, with at most twice that many in flight, so memory is bounded by
``workers`` patches (about half a gigabyte each). Lighting is serialized on one lock: the CUDA session
is one device, and the CPU lightmap already uses every core, so the other threads decode horizons and
compress and write files meanwhile. Lighting is on the GPU when ``lighting`` is ``"auto"`` and a device
can be set up, or ``"cuda"`` (no fallback); ``"cpu"`` never probes CUDA.

A first Ctrl-C stops starting patches and lets those in flight finish; a second stops at once.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import threading
import time as wall_clock
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt

from ..errors import InputError, ProductStorageError
from . import safe_haven_lightstore as lightstore
from .file_format import PATCH_SIZE, read_patch_horizons
from .lightmap_cpu import LightmapCpuSession
from .safe_haven_lightstore import LightStore, LightStoreSettings
from .safe_haven_windows import FloodGrid

logger = logging.getLogger(__name__)

MANIFEST_FORMAT = "lunarscout-light-store-manifest-v1"
REPORT_FORMAT = "lunarscout-light-store-report-v1"
MANIFEST_NAME = "light_store.json"
REPORT_NAME = "light_store.report.json"
LIGHTING_CHOICES = ("auto", "cuda", "cpu")

Patch = tuple[int, int]
VectorSource = Callable[[npt.NDArray[np.datetime64]], npt.ArrayLike]


@dataclass(frozen=True)
class Stage0Settings:
    """The store's contents (``store``, ``start_utc``, ``stop_utc``) and how a run is carried out."""

    store: LightStoreSettings = field(default_factory=LightStoreSettings)
    start_utc: str = "2024-01-01T00:00:00"
    stop_utc: str = "2032-01-01T00:00:00"                 # exclusive
    lighting: str = "auto"
    gpu_device_id: int = 0
    gpu_chunks_per_batch: int = 4
    workers: int = 8
    status_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.lighting not in LIGHTING_CHOICES:
            raise InputError(f"lighting must be one of {LIGHTING_CHOICES}", code="light_store_lighting_invalid",
                             details={"lighting": self.lighting})
        if self.workers < 1 or self.gpu_chunks_per_batch < 1:
            raise InputError("workers and gpu_chunks_per_batch must be positive", code="light_store_workers_invalid")
        if self.status_seconds <= 0:
            raise InputError("status_seconds must be positive", code="light_store_status_invalid")
        first, stop = self._step_range()
        if stop <= first:
            raise InputError("the time range holds no steps", code="light_store_time_range",
                             details={"start_utc": self.start_utc, "stop_utc": self.stop_utc})

    def _step_range(self) -> tuple[int, int]:
        grid = FloodGrid(self.store.step_minutes)
        return (grid.ceil_index(int(np.datetime64(self.start_utc, "us").astype("int64"))),
                grid.ceil_index(int(np.datetime64(self.stop_utc, "us").astype("int64"))))

    @property
    def first_step(self) -> int:
        return self._step_range()[0]

    @property
    def step_count(self) -> int:
        first, stop = self._step_range()
        return stop - first


def patch_file(directory: str | Path, patch_row: int, patch_column: int) -> Path:
    return Path(directory) / f"{patch_row:03d}" / f"light_{patch_row:03d}_{patch_column:03d}.lsl"


def patch_grid_shape(dem) -> tuple[int, int]:
    height, width = dem.elevation_m.shape
    return -(-height // PATCH_SIZE), -(-width // PATCH_SIZE)


def _graceful_interrupt():
    """The block driver's interrupt handler, imported when used: that module imports this one."""
    from .safe_haven_blocks import _graceful_interrupt as handler

    return handler()


def dem_identity(dem) -> dict:
    """What identifies a DEM for the store: its shape, georeferencing and a hash of its heights."""
    heights = np.ascontiguousarray(dem.elevation_m)
    digest = hashlib.blake2b(memoryview(heights).cast("B"), digest_size=16).hexdigest()
    projection = dem.projection
    return {
        "shape": [int(value) for value in heights.shape],
        "geo_transform": [float(value) for value in dem.geo_transform],
        "projection": [float(value) for value in (
            projection.radius_m, projection.latitude_origin_rad, projection.longitude_origin_rad,
            projection.scale, projection.false_easting_m, projection.false_northing_m)],
        "elevation_blake2b": digest,
    }


def manifest_document(settings: Stage0Settings, dem) -> dict:
    return {
        "format": MANIFEST_FORMAT,
        "store_settings": asdict(settings.store),
        "first_step": settings.first_step,
        "step_count": settings.step_count,
        "start_utc": settings.start_utc,
        "stop_utc": settings.stop_utc,
        "patch_grid": list(patch_grid_shape(dem)),
        "dem": dem_identity(dem),
    }


def _write_json(path: Path, document: dict) -> None:
    staging = path.with_name(path.name + f".partial-{os.getpid()}")
    try:
        staging.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _check_manifest(directory: Path, expected: dict) -> None:
    path = directory / MANIFEST_NAME
    if not path.exists():
        return
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProductStorageError(f"the light-store manifest {path} is unreadable", code="light_store_manifest_unreadable",
                                  details={"path": str(path)}) from error
    differing = sorted(key for key in set(found) | set(expected) if found.get(key) != expected.get(key))
    if differing:
        raise ProductStorageError(
            f"the light store in {directory} was made with different settings ({', '.join(differing)}); "
            "use a new directory or delete the old one",
            code="light_store_manifest_mismatch", details={"directory": str(directory), "differing": differing},
        )


# ---------------------------------------------------------------------------
# Lighting backend
# ---------------------------------------------------------------------------


def _make_gpu_session(settings: Stage0Settings):
    from .safe_haven_lit_cuda import LitCudaSession

    return LitCudaSession(device_id=settings.gpu_device_id, steps_per_chunk=settings.store.steps_per_chunk,
                          chunks_per_batch=settings.gpu_chunks_per_batch)


def select_lighting(settings: Stage0Settings):
    """``(session, name)``: the CUDA session or the CPU lightmap session. ``"cpu"`` never touches
    CUDA, ``"cuda"`` raises when the device cannot be set up, ``"auto"`` falls back with a warning."""
    from .cuda_backend import CudaBackendError

    if settings.lighting != "cpu":
        try:
            return _make_gpu_session(settings), "cuda"
        except (CudaBackendError, ImportError) as error:
            if settings.lighting == "cuda":
                raise
            logger.warning("CUDA lighting is not available (%s: %s); lighting on the CPU.",
                           type(error).__name__, error)
    return LightmapCpuSession(time_batch_size=64), "cpu"


# ---------------------------------------------------------------------------
# Reading a store
# ---------------------------------------------------------------------------


class LightStoreDirectory:
    """A store directory opened for reading, with its manifest checked against the DEM.

    ``open_patch(row, column)`` returns the patch's :class:`LightStore`, or ``None`` when the patch has
    no file (it had no horizon tile, so it is never lit). Settings, time range and DEM identity come
    from the manifest; pass ``dem`` to check that the store was made for it.
    """

    def __init__(self, directory: str | Path, dem=None) -> None:
        self.directory = Path(directory)
        path = self.directory / MANIFEST_NAME
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["format"] != MANIFEST_FORMAT:
                raise ValueError(manifest["format"])
            self.settings = LightStoreSettings(**manifest["store_settings"])
            self.first_step = int(manifest["first_step"])
            self.step_count = int(manifest["step_count"])
            self.patch_grid = tuple(int(value) for value in manifest["patch_grid"])
            self.dem = manifest["dem"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ProductStorageError(f"{path} is not a readable light-store manifest",
                                      code="light_store_manifest_unreadable", details={"path": str(path)}) from error
        if dem is not None and dem_identity(dem) != self.dem:
            raise ProductStorageError("the light store was made for a different DEM",
                                      code="light_store_manifest_mismatch",
                                      details={"directory": str(self.directory), "differing": ["dem"]})

    @property
    def stop_step(self) -> int:
        return self.first_step + self.step_count

    def open_patch(self, patch_row: int, patch_column: int) -> LightStore | None:
        path = patch_file(self.directory, patch_row, patch_column)
        if not path.exists():
            return None
        opened = LightStore(path, self.settings)
        if (opened.patch != (patch_row, patch_column) or opened.first_step != self.first_step
                or opened.step_count != self.step_count):
            opened.close()
            raise ProductStorageError(f"{path} does not belong to this store at this position",
                                      code="light_store_manifest_mismatch", details={"path": str(path)})
        return opened


# ---------------------------------------------------------------------------
# Running stage 0
# ---------------------------------------------------------------------------


@dataclass
class Stage0Report:
    status: str = "running"                   # running, completed, interrupted or failed
    patches_total: int = 0
    patches_already_done: int = 0
    patches_written: int = 0
    patches_without_horizons: list[list[int]] = field(default_factory=list)
    bytes_written: int = 0
    elapsed_seconds: float = 0.0
    lighting: str = ""
    decode_seconds: float = 0.0               # summed over patches and threads
    light_seconds: float = 0.0
    write_seconds: float = 0.0
    error: str = ""


def _usable(path: Path, settings: Stage0Settings, patch: Patch) -> bool:
    try:
        with LightStore(path, settings.store) as opened:
            return (opened.patch == patch and opened.first_step == settings.first_step
                    and opened.step_count == settings.step_count)
    except ProductStorageError:
        return False


def _clear_staging(directory: Path) -> int:
    removed = 0
    for entry in directory.glob("*/*.partial-*"):
        entry.unlink(missing_ok=True)
        removed += 1
    for entry in directory.glob("*.partial-*"):
        entry.unlink(missing_ok=True)
        removed += 1
    return removed


def generate_light_store(
    directory: str | Path,
    *,
    dem,
    horizon_store,
    settings: Stage0Settings,
    sun_vectors_at: VectorSource,
    earth_vectors_at: VectorSource,
    patches: Iterable[Patch] | None = None,
    report_path: str | Path | None = None,
) -> Stage0Report:
    """Light every patch of the DEM (or just ``patches``) and write the store in ``directory``.

    Returns the report, also written to ``report_path`` (default ``light_store.report.json`` in the
    directory). An interrupt returns a report with status ``"interrupted"``; any other failure writes
    a ``"failed"`` report and raises after the patches in flight have finished.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ProductStorageError(f"the light-store directory {directory} does not exist",
                                  code="light_store_directory_missing", details={"directory": str(directory)})
    rows, columns = patch_grid_shape(dem)
    wanted = ([(row, column) for row in range(rows) for column in range(columns)] if patches is None
              else [(int(row), int(column)) for row, column in patches])
    for row, column in wanted:
        lightstore.patch_geometry(dem, row, column)             # a patch outside the DEM fails here
    if len(set(wanted)) != len(wanted):
        raise InputError("a patch is listed twice", code="light_store_patch_duplicate")

    expected = manifest_document(settings, dem)
    _check_manifest(directory, expected)
    report_path = Path(report_path) if report_path is not None else directory / REPORT_NAME
    report = Stage0Report(patches_total=len(wanted))
    started = wall_clock.monotonic()
    try:
        _write_json(directory / MANIFEST_NAME, expected)
        stale = _clear_staging(directory)
        if stale:
            logger.info("removed %d staging files left by an earlier run", stale)

        pending: list[Patch] = []
        for patch in wanted:
            path = patch_file(directory, *patch)
            if path.exists():
                if _usable(path, settings, patch):
                    report.patches_already_done += 1
                    continue
                logger.warning("replacing the unusable light-store file %s", path)
            pending.append(patch)
        logger.info("stage 0: %d patches, %d already stored, %d to light", len(wanted),
                    report.patches_already_done, len(pending))

        if pending:
            _run(directory, pending, dem, horizon_store, settings, sun_vectors_at, earth_vectors_at, report, started)
        else:
            report.lighting = "none needed"
        if report.status == "running":
            report.status = "completed"
    except BaseException as error:
        report.status = "failed"
        report.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.elapsed_seconds = wall_clock.monotonic() - started
        _write_json(report_path, {"format": REPORT_FORMAT, "settings": asdict(settings), **asdict(report)})
    logger.info("stage 0 %s: %d patches written, %d without horizons, %.1f GB, %.0f s", report.status,
                report.patches_written, len(report.patches_without_horizons), report.bytes_written / 1e9,
                report.elapsed_seconds)
    return report


def _run(directory, pending, dem, horizon_store, settings, sun_vectors_at, earth_vectors_at, report, started) -> None:
    grid = FloodGrid(settings.store.step_minutes)
    steps = np.arange(settings.first_step, settings.first_step + settings.step_count, dtype=np.int64)
    stamps = grid.times(steps)
    sun = np.asarray(sun_vectors_at(stamps), dtype=np.float64)
    earth = np.asarray(earth_vectors_at(stamps), dtype=np.float64)
    session, report.lighting = select_lighting(settings)
    logger.info("lighting on the %s; %d workers; %d steps of %d minutes", report.lighting.upper(),
                settings.workers, steps.size, settings.store.step_minutes)
    light_lock = threading.Lock()
    observer = settings.store.observer_height_m

    def process(patch: Patch) -> dict:
        row, column = patch
        began = wall_clock.monotonic()
        horizons = read_patch_horizons(horizon_store, row * PATCH_SIZE, column * PATCH_SIZE, observer)
        decoded = wall_clock.monotonic()
        if horizons is None:
            return {"patch": patch, "none": True, "decode": decoded - began}
        with light_lock:
            lit_began = wall_clock.monotonic()
            planes = lightstore.light_patch_planes(session, dem, horizons, row, column, sun, earth, settings.store)
            lit = wall_clock.monotonic()
        path = patch_file(directory, row, column)
        path.parent.mkdir(exist_ok=True)
        size = lightstore.write_patch(path, planes, settings.store, patch_row=row, patch_column=column,
                                      first_step=settings.first_step)
        return {"patch": patch, "none": False, "bytes": size, "decode": decoded - began,
                "light": lit - lit_began, "write": wall_clock.monotonic() - lit}

    failure: BaseException | None = None
    queue = iter(pending)
    outstanding: set[concurrent.futures.Future] = set()
    finished = 0
    last_status = wall_clock.monotonic()
    with concurrent.futures.ThreadPoolExecutor(settings.workers) as pool, _graceful_interrupt() as interrupts:
        while True:
            while len(outstanding) < 2 * settings.workers and failure is None and not interrupts:
                patch = next(queue, None)
                if patch is None:
                    break
                outstanding.add(pool.submit(process, patch))
            if not outstanding:
                break
            done, outstanding = concurrent.futures.wait(
                outstanding, timeout=settings.status_seconds, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    result = future.result()
                except BaseException as error:
                    failure = failure or error
                    continue
                finished += 1
                report.decode_seconds += result["decode"]
                if result["none"]:
                    report.patches_without_horizons.append(list(result["patch"]))
                else:
                    report.patches_written += 1
                    report.bytes_written += result["bytes"]
                    report.light_seconds += result["light"]
                    report.write_seconds += result["write"]
            now = wall_clock.monotonic()
            if now - last_status >= settings.status_seconds:
                last_status = now
                rate = finished / max(now - started, 1e-9)
                remaining = len(pending) - finished
                logger.info("stage 0: %d of %d patches (%.2f/s, about %.0f min left), %.1f GB written",
                            finished, len(pending), rate, remaining / rate / 60.0 if rate else float("inf"),
                            report.bytes_written / 1e9)
        interrupted = bool(interrupts)
    if failure is not None:
        raise failure
    if interrupted and finished < len(pending):
        report.status = "interrupted"
        logger.warning("interrupted after %d of %d patches; run again to continue", finished, len(pending))
