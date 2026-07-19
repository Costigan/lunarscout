"""Public horizon generation through the Python/Numba CUDA pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Real
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .errors import (
    CudaError,
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
        self.callback_error: BaseException | None = None

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
            print(
                f"horizons: {event.stage} {event.completed}/{event.total}",
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


def _run_horizon_pipeline(
    dems: Sequence[Any],
    output_directory: Path,
    *,
    observer_height_m: float,
    compress: bool,
    overwrite: bool,
    progress_callback: Callable[[Any], None] | None,
    cancellation_requested: CancellationCheck | None,
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
    from ._numba_horizon.pipeline import enumerate_patches, run_bounded_pipeline
    from ._numba_horizon.pyramid import build_max_pyramid

    primary = dems[0]
    pyramids = tuple(build_max_pyramid(dem) for dem in dems)
    configuration = ContractConfiguration(
        PATCH_SIZE,
        PATCH_SIZE,
        AZIMUTH_COUNT,
        8,
        len(dems),
        primary.width,
        primary.height,
    )

    def prepare(patch: Any) -> SegmentTensor:
        values, _centers, _convergence = build_subpatch_segments_numba(
            dems,
            tile_column=patch.tile_x,
            tile_row=patch.tile_y,
            tile_width=patch.kernel_width,
            azimuth_count=AZIMUTH_COUNT,
            maximum_distance_m=1_000_000.0,
            observer_elevation_m=observer_height_m,
            subpatch_size=8,
            grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
            parallel=True,
        )
        dem_ids = np.broadcast_to(
            np.arange(len(dems), dtype=np.int32), values.shape[:-1]
        ).copy()
        return SegmentTensor(values, dem_ids, configuration)

    def processor_factory(_worker_id: int):
        session = CudaSession(device_id=0, production_concurrency=1)

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

    run_bounded_pipeline(
        enumerate_patches(primary.width, primary.height),
        store=HorizonTileStore(output_directory),
        prepare_patch=prepare,
        processor_factory=processor_factory,
        finalize_patch=finalize,
        observer_elevation_m=observer_height_m,
        compress=compress,
        skip_existing=not overwrite,
        prepared_queue_capacity=1,
        writer_queue_capacity=1,
        worker_count=1,
        progress_callback=progress_callback,
        cancellation_requested=cancellation_requested,
    )


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
        culling for later DEMs.
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
        When ``True``, writes backend and progress messages to stdout.
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
    tile.
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
    if cancellation_requested is not None and cancellation_requested():
        raise OperationCancelledError(
            "Horizon generation was cancelled.",
            code="horizon_generation_cancelled",
            details={"path": str(output)},
        )

    from .products import _load_dem

    dems = tuple(_load_dem(path)[0] for path in resolved_dems)
    primary = dems[0]
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
        )
        return output
    except Exception as exc:
        if adapter.callback_error is exc:
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
