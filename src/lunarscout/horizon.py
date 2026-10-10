"""Public horizon generation through the Python/Numba CUDA pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from numbers import Real
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from .errors import (
    CudaError,
    GridError,
    HorizonGenerationError,
    InputError,
    OperationCancelledError,
)
from .progress import ProgressEvent


ProgressCallback = Callable[[float], None]
ProgressEventCallback = Callable[[ProgressEvent], None]
CancellationCheck = Callable[[], bool]


def _is_cuda_runtime_failure(error: BaseException) -> bool:
    """Recognize CUDA-stack exceptions without importing CUDA to classify them."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        module = type(current).__module__.lower()
        if module == "cuda" or module.startswith(("cuda.", "numba.cuda")):
            return True
        current = current.__cause__ or current.__context__
    return False


class _HorizonProgressAdapter:
    def __init__(
        self,
        output_directory: Path,
        *,
        verbose: bool,
        progress_callback: ProgressCallback | None,
        progress_event_callback: ProgressEventCallback | None,
    ) -> None:
        self.output_directory = output_directory
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.progress_event_callback = progress_event_callback
        self._last_fraction: float | None = None
        self._process_started_at: float | None = None
        self.callback_error: BaseException | None = None

    def _timing_status(self, event: ProgressEvent) -> str:
        if event.stage == "process_patches" and self._process_started_at is None:
            self._process_started_at = time.monotonic()
        if self._process_started_at is None or event.completed <= 0:
            return "    --.-- s/patch | ETA -------------------"
        seconds_per_patch = (time.monotonic() - self._process_started_at) / event.completed
        remaining = max(0, event.total - event.completed)
        completion = datetime.now().astimezone() + timedelta(
            seconds=seconds_per_patch * remaining
        )
        return (
            f"{seconds_per_patch:8.2f} s/patch | "
            f"ETA {completion:%Y-%m-%d %H:%M:%S}"
        )

    def __call__(self, private_event: Any) -> None:
        stage = str(private_event.stage)
        completed = int(private_event.processed_patches)
        total = int(private_event.total_patches)
        if stage == "prepare_patches":
            fraction = 0.1
        elif stage == "process_patches":
            patch_fraction = 0.0 if total == 0 else completed / total
            fraction = 0.15 + 0.85 * patch_fraction
        elif stage == "complete":
            fraction = 1.0
        else:
            fraction = min(1.0, max(0.0, float(private_event.percent) / 100.0))
        if self._last_fraction is not None:
            fraction = max(self._last_fraction, fraction)
        file_name = private_event.file_name
        tile_y: int | None = None
        tile_x: int | None = None
        if file_name:
            parts = str(file_name).split("_")
            if len(parts) >= 3:
                try:
                    tile_y, tile_x = int(parts[1]), int(parts[2])
                except ValueError:
                    pass
        event = ProgressEvent(
            operation="horizons",
            stage=stage,
            completed=completed,
            total=total,
            fraction=fraction,
            backend="cuda",
            message=str(private_event.message),
            tile_y=tile_y,
            tile_x=tile_x,
            path=self.output_directory,
        )
        if self.verbose:
            if event.stage == "prepare_patches":
                print("horizons: using cuda backend", file=sys.stdout, flush=True)
            message = f" | {event.message}" if event.message else ""
            print(
                f"horizons: {event.stage:<16} "
                f"{event.completed:6d}/{event.total:<6d} | "
                f"{self._timing_status(event)}{message}",
                file=sys.stdout,
                flush=True,
            )
        if self.progress_event_callback is not None:
            try:
                self.progress_event_callback(event)
            except BaseException as exc:
                self.callback_error = exc
                raise
        if self.progress_callback is not None and fraction != self._last_fraction:
            try:
                self.progress_callback(fraction)
            except BaseException as exc:
                self.callback_error = exc
                raise
            self._last_fraction = fraction


# Patches with any pixel this close to a geographic pole get exact rays for
# every pixel (subpatch size 1).  Elsewhere rays are exact only every 8 px and
# are translated, not rotated, in between; near a pole grid north turns fast
# enough that the translated rays point the wrong way (review finding M-C).
# On the 20 m LOLA fixture (review_scripts/s10_polar_rays.py) the mean error
# is 6 degrees at the pole, about 0.03 degrees at 50-60 px, and at the
# background level by about 100 px.
_POLE_EXACT_RAY_RADIUS_PX = 160


def _pole_patch_origins(
    primary: Any, patches: Sequence[Any], radius_px: float = _POLE_EXACT_RAY_RADIUS_PX
) -> frozenset[tuple[int, int]]:
    """Return ``(tile_x, tile_y)`` of patches within ``radius_px`` of a pole."""
    poles = []
    for latitude in (0.5 * np.pi, -0.5 * np.pi):
        column, row = primary.lon_lat_to_pixel(0.0, latitude)
        if np.isfinite(column) and np.isfinite(row):
            poles.append((column, row))
    selected = set()
    for patch in patches:
        for column, row in poles:
            dx = max(patch.tile_x - column, 0.0, column - (patch.tile_x + patch.width - 1))
            dy = max(patch.tile_y - row, 0.0, row - (patch.tile_y + patch.height - 1))
            if dx * dx + dy * dy <= radius_px * radius_px:
                selected.add((patch.tile_x, patch.tile_y))
    return frozenset(selected)


def _run_horizon_pipeline(
    dems: Sequence[Any],
    output_directory: Path,
    *,
    observer_height_m: float,
    compress: bool,
    overwrite: bool,
    progress_callback: Callable[[Any], None] | None,
    cancellation_requested: CancellationCheck | None,
    patch_offset: int = 0,
    patch_stride: int = 1,
    verbose: bool = False,
    skip_failed_patches: bool = False,
) -> None:
    """Bind validated public inputs to the selected private production pipeline."""
    from ._numba_horizon.contract import (
        ContractConfiguration,
        HorizonBuffers,
        SegmentTensor,
    )
    from ._numba_horizon.cuda_backend import CudaSession
    from ._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore, PATCH_SIZE
    from ._numba_horizon.generator import generate_patch_horizons
    from ._numba_horizon.geometry import (
        GridConvergenceInput,
        build_subpatch_segments_numba,
    )
    from ._numba_horizon.pipeline import (
        HorizonPipelineCancelled,
        enumerate_patches,
        partition_patches,
        run_bounded_pipeline,
    )

    primary = dems[0]
    session = CudaSession(device_id=0, production_concurrency=2)
    # Building each max pyramid on the GPU takes about 0.3 s for a 30,400 px
    # DEM, mostly the host-to-device copy, so no on-disk cache is kept.
    pyramids = tuple(session.build_max_pyramid(dem) for dem in dems)
    configurations = {
        subpatch_size: ContractConfiguration(
            PATCH_SIZE,
            PATCH_SIZE,
            AZIMUTH_COUNT,
            subpatch_size,
            len(dems),
            primary.width,
            primary.height,
        )
        for subpatch_size in (1, 8)
    }
    patches = partition_patches(
        enumerate_patches(primary.width, primary.height),
        offset=patch_offset,
        stride=patch_stride,
    )
    pole_patches = _pole_patch_origins(primary, patches)

    def prepare(patch: Any) -> SegmentTensor:
        # Subpatch size 1 needs about 1.6 GiB of segments per DEM per patch
        # (64x the default) and about 10 s of CPU time to build.
        subpatch_size = 1 if (patch.tile_x, patch.tile_y) in pole_patches else 8
        values, _centers, _convergence = build_subpatch_segments_numba(
            dems,
            tile_column=patch.tile_x,
            tile_row=patch.tile_y,
            tile_width=patch.kernel_width,
            azimuth_count=AZIMUTH_COUNT,
            maximum_distance_m=1_000_000.0,
            observer_elevation_m=observer_height_m,
            subpatch_size=subpatch_size,
            grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
            parallel=True,
        )
        dem_ids = np.broadcast_to(
            np.arange(len(dems), dtype=np.int32), values.shape[:-1]
        ).copy()
        return SegmentTensor(values, dem_ids, configurations[subpatch_size])

    def processor_factory(_worker_id: int):
        def process(patch: Any, segments: SegmentTensor) -> np.ndarray:
            return generate_patch_horizons(
                session,
                segments,
                pyramids,
                tile_column=patch.tile_x,
                tile_row=patch.tile_y,
                observer_elevation_m=observer_height_m,
            ).slopes

        return process

    def finalize(_patch: Any, slopes: np.ndarray) -> np.ndarray:
        return HorizonBuffers(slopes).degrees()

    if patches:
        # Warm up the lazy-compiled CUDA kernels on a single thread so the
        # concurrent pipeline workers do not race on numba's first-launch path.
        warmup_tensor = prepare(patches[0])
        generate_patch_horizons(
            session,
            warmup_tensor,
            pyramids,
            tile_column=patches[0].tile_x,
            tile_row=patches[0].tile_y,
            observer_elevation_m=observer_height_m,
        )
    def skippable(error: BaseException) -> bool:
        # A CUDA fault poisons the context for every later patch.
        return (
            skip_failed_patches
            and not isinstance(error, HorizonPipelineCancelled)
            and not _is_cuda_runtime_failure(error)
        )

    result = run_bounded_pipeline(
        patches,
        store=HorizonTileStore(output_directory),
        prepare_patch=prepare,
        processor_factory=processor_factory,
        finalize_patch=finalize,
        observer_elevation_m=observer_height_m,
        compress=compress,
        skip_existing=not overwrite,
        prepared_queue_capacity=2,
        writer_queue_capacity=2,
        worker_count=2,
        progress_callback=progress_callback,
        cancellation_requested=cancellation_requested,
        timing_stream=sys.stdout if verbose else None,
        patch_error_filter=skippable,
    )
    if result.failed_patches:
        failures = [
            {
                "tile_x": patch.tile_x,
                "tile_y": patch.tile_y,
                "error": f"{type(error).__name__}: {error}",
            }
            for patch, error in result.failed_patches
        ]
        raise HorizonGenerationError(
            f"{len(failures)} horizon patches failed; the others were generated.",
            code="horizon_patches_failed",
            details={"patches": failures},
        ) from result.failed_patches[0][1]


def generate_horizons(
    output_directory: str | Path,
    dem_paths: Sequence[str | Path],
    *,
    observer_height_m: float = 0.0,
    compress: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
    patch_offset: int = 0,
    patch_stride: int = 1,
    skip_failed_patches: bool = False,
) -> Path:
    """Generate compatible 128-pixel horizon tiles with NVIDIA CUDA.

    Parameters
    ----------
    output_directory:
        The directory where horizon tiles will be written.  Created if it
        does not exist.
    dem_paths:
        An ordered sequence of DEM files.  The first DEM defines the output
        grid.  Later DEMs extend surrounding terrain coverage.  The
        cumulative horizon from earlier DEMs participates in hierarchy
        culling for later DEMs.  Each DEM is read with its declared
        scale/offset applied and normalized to metres above the lunar
        reference sphere (1737.4 km); DEMs stored as radius-from-centre are
        converted to elevation-from-sphere automatically.
    observer_height_m:
        Observer height above the DEM surface, in meters.  Must be finite
        and in ``[0, 100)``.
    compress:
        When ``True`` (the default) tiles are written as ``.cbin``;
        ``compress=False`` writes uncompressed ``.bin``.
    overwrite:
        When ``False`` (the default), structurally complete existing tiles
        are skipped.  ``overwrite=True`` regenerates every tile.
    verbose:
        When ``True``, writes backend and progress messages to stdout, plus
        mean elapsed times and completion counts for each pipeline stage every
        ten written patches. Compute time includes host-side overhead and is
        not a measurement of GPU utilization.
    progress_callback:
        Optional callable receiving a monotonic durable fraction ``[0, 1]``.
    progress_event_callback:
        Optional callable receiving an immutable :class:`ProgressEvent` with
        stage, tile coordinates, and backend detail.
    cancellation_requested:
        Optional callable checked at bounded work boundaries.  When it
        returns ``True``, generation raises
        :class:`~lunarscout.OperationCancelledError` and leaves resumable
        staging state.
    patch_offset:
        Partition index (0-based) when sharding the patch list across
        independent workers.  ``patch_stride=1`` (the default) and
        ``patch_offset=0`` process every patch.  With ``patch_stride=N``,
        worker ``patch_offset`` processes patches ``patch_offset,
        patch_offset+N, ...`` so ``N`` workers cover every patch exactly once.
    patch_stride:
        Number of workers sharing the patch list (see ``patch_offset``).  Must
        be a positive integer; the default ``1`` processes all patches.
    skip_failed_patches:
        If true, a patch whose own calculation raises an ordinary error is
        recorded and the run continues with the remaining patches; at the end
        :class:`~lunarscout.HorizonGenerationError` with code
        ``horizon_patches_failed`` lists the failed patches.  CUDA faults,
        cancellation and storage errors still stop the run at once.  Batch
        workers use this so one deterministic bad patch does not block the
        rest of its partition on every retry.

    Returns
    -------
    pathlib.Path
        The resolved completed output directory.

    Notes
    -----
    Horizon generation is CUDA-only and has no CPU fallback.  If Numba CUDA
    cannot initialize a compatible NVIDIA device, :class:`~lunarscout.CudaError`
    is raised before any horizon tile is modified.  Existing stored horizons
    remain usable by CPU downstream products.

    Each output tile is a fixed 128 by 128 pixel patch with 1,440 ``float32``
    azimuth samples per pixel at 0.25-degree spacing.  Sample 0 is north.
    Tiles are staged beside their destination and atomically published.  A
    failed calculation or cancelled overwrite preserves the prior complete
    tile.  Each DEM's maximum-elevation pyramid is built on the GPU for every
    run; nothing is written beside the DEMs.
    """
    if isinstance(dem_paths, (str, bytes, Path)):
        raise InputError(
            "dem_paths must be a sequence ordered primary DEM first.",
            code="horizon_dem_paths_invalid",
        )
    try:
        resolved_dems = tuple(Path(path).expanduser().resolve() for path in dem_paths)
    except (TypeError, ValueError, OSError) as exc:
        raise InputError(
            "dem_paths must contain filesystem paths.",
            code="horizon_dem_paths_invalid",
        ) from exc
    if not resolved_dems:
        raise InputError(
            "At least one DEM path is required.",
            code="horizon_dem_paths_empty",
        )
    missing = tuple(str(path) for path in resolved_dems if not path.is_file())
    if missing:
        raise InputError(
            "A horizon DEM path does not identify a file.",
            code="horizon_dem_not_found",
            details={"paths": missing},
        )
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise InputError(
            "Horizon output must be a directory path.",
            code="horizon_output_is_file",
            details={"path": str(output)},
        )
    if (
        isinstance(observer_height_m, bool)
        or not isinstance(observer_height_m, Real)
        or not np.isfinite(observer_height_m)
        or not 0.0 <= float(observer_height_m) < 100.0
    ):
        raise InputError(
            "observer_height_m must be finite and between 0.0 and 100.0 meters.",
            code="horizon_observer_height_invalid",
            details={"observer_height_m": observer_height_m},
        )
    for name, value in (
        ("compress", compress),
        ("overwrite", overwrite),
        ("verbose", verbose),
    ):
        if not isinstance(value, bool):
            raise InputError(
                f"{name} must be a bool.",
                code="horizon_option_invalid",
                details={"argument": name},
            )
    for name, callback in (
        ("progress_callback", progress_callback),
        ("progress_event_callback", progress_event_callback),
        ("cancellation_requested", cancellation_requested),
    ):
        if callback is not None and not callable(callback):
            raise InputError(
                f"{name} must be callable or None.",
                code="horizon_callback_invalid",
                details={"argument": name},
            )
    if isinstance(patch_stride, bool) or not isinstance(patch_stride, int) or patch_stride < 1:
        raise InputError(
            "patch_stride must be a positive integer.",
            code="horizon_patch_stride_invalid",
            details={"patch_stride": patch_stride},
        )
    if not isinstance(skip_failed_patches, bool):
        raise InputError(
            "skip_failed_patches must be a boolean.",
            code="horizon_skip_failed_patches_invalid",
            details={"skip_failed_patches": skip_failed_patches},
        )
    if (
        isinstance(patch_offset, bool)
        or not isinstance(patch_offset, int)
        or not 0 <= patch_offset < patch_stride
    ):
        raise InputError(
            "patch_offset must be an integer in [0, patch_stride).",
            code="horizon_patch_offset_invalid",
            details={"patch_offset": patch_offset, "patch_stride": patch_stride},
        )
    if cancellation_requested is not None and cancellation_requested():
        raise OperationCancelledError(
            "Horizon generation was cancelled.",
            code="horizon_generation_cancelled",
            details={"path": str(output)},
        )

    from .products import _load_dem

    dems = tuple(_load_dem(path)[0] for path in resolved_dems)
    primary = dems[0]
    # Observer heights are measured from the primary DEM's sphere, and each
    # pass measures its own terrain from its DEM's sphere, so the spheres must
    # agree (review finding M-J).  Extents and resolutions may differ: each
    # DEM continues the ray where the previous DEM's pass ended.
    radii = [dem.projection.radius_m for dem in dems]
    if any(abs(radius - radii[0]) > 1e-3 for radius in radii[1:]):
        raise GridError(
            "All horizon DEMs must use the same reference sphere radius.",
            code="horizon_dem_radius_mismatch",
            details={
                "paths": [str(path) for path in resolved_dems],
                "radii_m": radii,
            },
        )
    last_tile_x = ((primary.width - 1) // 128) * 128
    last_tile_y = ((primary.height - 1) // 128) * 128
    if last_tile_x > 99_999 or last_tile_y > 99_999:
        raise InputError(
            "The primary DEM exceeds the horizon filename coordinate range.",
            code="horizon_grid_too_large",
            details={"width": primary.width, "height": primary.height},
        )

    adapter = _HorizonProgressAdapter(
        output,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
    )
    try:
        _run_horizon_pipeline(
            dems,
            output,
            observer_height_m=float(observer_height_m),
            compress=compress,
            overwrite=overwrite,
            progress_callback=adapter,
            cancellation_requested=cancellation_requested,
            patch_offset=patch_offset,
            patch_stride=patch_stride,
            verbose=verbose,
            skip_failed_patches=skip_failed_patches,
        )
        return output
    except Exception as exc:
        if adapter.callback_error is exc:
            raise
        if isinstance(exc, HorizonGenerationError):
            raise
        from ._numba_horizon.cuda_backend import CudaBackendError
        from ._numba_horizon.pipeline import HorizonPipelineCancelled

        if isinstance(exc, HorizonPipelineCancelled):
            raise OperationCancelledError(
                "Horizon generation was cancelled.",
                code="horizon_generation_cancelled",
                details={"path": str(output)},
            ) from exc
        if isinstance(exc, CudaBackendError):
            raise CudaError(
                "A compatible NVIDIA CUDA device is required for horizon generation.",
                code="cuda_horizon_unavailable",
                details={"error": str(exc)},
            ) from exc
        if _is_cuda_runtime_failure(exc):
            raise CudaError(
                "CUDA horizon generation failed during execution.",
                code="cuda_horizon_execution_failed",
                details={"error": str(exc)},
            ) from exc
        raise HorizonGenerationError(
            "Horizon generation failed.",
            code=(
                "horizon_storage_failed"
                if isinstance(exc, OSError)
                else "horizon_generation_failed"
            ),
            details={"path": str(output), "error": str(exc)},
        ) from exc
