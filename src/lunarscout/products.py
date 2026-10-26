"""Public Python-only horizon-derived product functions."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from typing import Any, Literal
import warnings

import numpy as np
import numpy.typing as npt
from pyproj import CRS

from .errors import (
    CudaError,
    GridError,
    HorizonFormatError,
    InputError,
    OperationCancelledError,
    ProductCalculationError,
    ProductStorageError,
    ProductTimeError,
    VectorError,
)
from .geotiff import read_dem_raster
from .progress import Backend, ProgressEvent
from .temporal import TimeInput, TimeRange


ProgressCallback = Callable[[float], None]
ProgressEventCallback = Callable[[ProgressEvent], None]
CancellationCheck = Callable[[], bool]


def _validate_output_conversion(
    output_transform: Callable[[np.ndarray], np.ndarray] | None,
    output_dtype: npt.DTypeLike | None,
    output_transform_id: str | None,
) -> None:
    if output_transform is None:
        if output_dtype is not None or output_transform_id is not None:
            raise InputError(
                "output_dtype and output_transform_id require output_transform.",
                code="product_output_conversion_invalid",
            )
        return
    if not callable(output_transform):
        raise InputError(
            "output_transform must be callable or None.",
            code="product_output_conversion_invalid",
        )
    if output_dtype is None:
        raise InputError(
            "output_dtype is required with output_transform.",
            code="product_output_conversion_invalid",
        )
    try:
        np.dtype(output_dtype)
    except (TypeError, ValueError) as exc:
        raise InputError(
            "output_dtype is not a valid NumPy dtype.",
            code="product_output_dtype_invalid",
            details={"output_dtype": repr(output_dtype)},
        ) from exc
    if output_transform_id is not None and not isinstance(output_transform_id, str):
        raise InputError(
            "output_transform_id must be a string or None.",
            code="product_output_conversion_invalid",
        )


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


def _projection_value(values: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in values:
            try:
                return float(values[name])
            except (TypeError, ValueError, OverflowError):
                return None
    return None


def _normalize_dem_elevations(
    values: npt.ArrayLike,
    *,
    scale: float | None,
    offset: float | None,
    radius_m: float,
    nodata: float | None = None,
) -> npt.NDArray[np.float32]:
    """Convert raw DEM samples to metres above the reference sphere.

    Samples equal to the declared ``nodata`` value (compared in raw units,
    before scale and offset) and non-finite samples become NaN, which every
    horizon stage treats as invalid terrain.  Applies the raster's declared
    scale and offset.  An offset that encodes the reference-sphere radius
    (i.e. values stored as radius-from-centre) is undone so the result is
    elevation relative to the sphere.  When no scale or offset is declared,
    radius-from-centre values are detected by the magnitude of the valid
    samples and shifted down by the reference radius.  The input array is
    never modified.
    """
    raw = np.asarray(values)
    result = np.array(raw, dtype=np.float32, order="C", copy=True)
    invalid = ~np.isfinite(result)
    if nodata is not None and np.isfinite(nodata):
        invalid |= raw == nodata
    scale_f = 1.0 if scale is None else float(scale)
    offset_f = 0.0 if offset is None else float(offset)
    radius = float(radius_m)
    transformed = False
    if scale_f != 1.0:
        result *= scale_f
        transformed = True
    if offset_f != 0.0:
        transformed = True
        if abs(offset_f - radius) < abs(offset_f):
            result += offset_f - radius
        else:
            result += offset_f
    result[invalid] = np.nan
    if not transformed and not invalid.all():
        finite_min = float(np.min(result[~invalid]))
        if finite_min > 0.5 * radius:
            result -= radius
    return result


def _load_dem(path: str | Path):
    """Read a stereographic lunar DEM normalized to metres above the reference sphere.

    Applies the raster's declared scale/offset and undoes any radius-from-centre
    encoding so the returned :class:`DemGrid` ``elevation_m`` is elevation
    relative to the reference sphere of radius ``projection.radius_m``.
    """
    from ._numba_horizon.geometry import DemGrid, ProjectionParameters

    dem_path = Path(path).expanduser().resolve()
    if not dem_path.is_file():
        raise InputError(
            "The DEM path does not identify a file.",
            code="product_dem_not_found",
            details={"path": str(dem_path)},
        )
    values, georef, band_scale, band_offset = read_dem_raster(dem_path)
    if georef is None:
        raise GridError(
            "The DEM must have complete geospatial metadata.",
            code="product_dem_not_georeferenced",
            details={"path": str(dem_path)},
        )
    try:
        crs = CRS.from_wkt(georef.projection_wkt)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You will likely lose important projection information",
                category=UserWarning,
            )
            parameters = crs.to_dict()
    except Exception as exc:
        raise GridError(
            "The DEM coordinate reference system cannot be interpreted.",
            code="product_dem_crs_invalid",
            details={"path": str(dem_path), "error": str(exc)},
        ) from exc
    if str(parameters.get("proj", "")).lower() != "stere":
        raise GridError(
            "Horizon-derived products require a stereographic lunar DEM.",
            code="product_dem_projection_unsupported",
            details={"path": str(dem_path), "projection": parameters.get("proj")},
        )
    latitude_origin_deg = _projection_value(parameters, "lat_0")
    longitude_origin_deg = _projection_value(parameters, "lon_0")
    scale = _projection_value(parameters, "k", "k_0")
    if scale is None:
        standard_parallel_deg = _projection_value(parameters, "lat_ts")
        if (
            standard_parallel_deg is not None
            and abs(abs(standard_parallel_deg) - 90.0) <= 1e-10
        ):
            scale = 1.0
    false_easting_m = _projection_value(parameters, "x_0")
    false_northing_m = _projection_value(parameters, "y_0")
    radius_m = _projection_value(parameters, "R", "a")
    if radius_m is None:
        radius_m = float(crs.ellipsoid.semi_major_metre)
    required = (
        latitude_origin_deg,
        longitude_origin_deg,
        scale,
        false_easting_m,
        false_northing_m,
        radius_m,
    )
    if any(value is None or not np.isfinite(value) for value in required):
        raise GridError(
            "The DEM stereographic projection parameters are incomplete.",
            code="product_dem_projection_invalid",
            details={"path": str(dem_path)},
        )
    projection = ProjectionParameters(
        radius_m=float(radius_m),
        latitude_origin_rad=float(np.deg2rad(latitude_origin_deg)),
        longitude_origin_rad=float(np.deg2rad(longitude_origin_deg)),
        scale=float(scale),
        false_easting_m=float(false_easting_m),
        false_northing_m=float(false_northing_m),
    )
    dem = DemGrid(
        _normalize_dem_elevations(
            values, scale=band_scale, offset=band_offset,
            radius_m=float(radius_m), nodata=georef.nodata,
        ),
        np.ascontiguousarray(georef.affine_transform, dtype=np.float64),
        projection,
    )
    return dem, georef


def _preflight_product_paths(
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool,
    input_name: str = "horizons",
) -> tuple[Path, Path]:
    horizons = Path(horizons_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not horizons.is_dir():
        raise InputError(
            f"The {input_name} path does not identify a directory.",
            code="product_horizons_not_found",
            details={"path": str(horizons)},
        )
    if output.suffix.lower() not in (".tif", ".tiff"):
        raise InputError(
            "Product output must use a .tif or .tiff extension.",
            code="product_output_extension_invalid",
            details={"path": str(output)},
        )
    if output.exists() and not overwrite:
        raise ProductStorageError(
            f"The product output already exists: {output}. "
            "Delete the file or pass overwrite=True to replace it.",
            code="product_output_exists",
            details={"path": str(output)},
        )
    if output.exists() and not output.is_file():
        raise ProductStorageError(
            "The product output path is not a file.",
            code="product_output_invalid",
            details={"path": str(output)},
        )
    return horizons, output


def _resolve_vectors(
    body: str,
    *,
    vectors_m: npt.ArrayLike | None,
    times: Iterable[TimeInput] | TimeRange,
):
    from ._numba_horizon.product_vectors import resolve_moon_me_vectors

    try:
        return resolve_moon_me_vectors(
            body,
            explicit_vectors_m=vectors_m,
            explicit_times=times if vectors_m is not None else None,
            times=times if vectors_m is None else None,
            start=None,
            stop=None,
            step=None,
        )
    except ValueError as exc:
        error_type = VectorError if vectors_m is not None else ProductTimeError
        raise error_type(
            str(exc),
            code=(
                "product_vectors_invalid"
                if vectors_m is not None
                else "product_times_invalid"
            ),
            details={"body": body},
        ) from exc


def _format_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60.0))
    if minutes < 1:
        return f"{int(round(seconds))}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


class _PatchRate:
    """Recent patch completion rate, for an ETA that is right after a resume.

    Only patches finished by this run count: the first sample is the run's own
    start, so patches completed by an earlier run do not inflate the rate.  The
    rate is taken over the most recent samples, so a slow start (kernel
    compilation, filling the read-ahead) drops out of the estimate.
    """

    WINDOW = 1000
    MINIMUM_SAMPLES = 5

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._samples: deque[tuple[float, int]] = deque(maxlen=self.WINDOW)

    def add(self, completed: int) -> None:
        self._samples.append((self._clock(), completed))

    def seconds_per_patch(self) -> float | None:
        if len(self._samples) < self.MINIMUM_SAMPLES:
            return None
        (first_time, first_count), (last_time, last_count) = self._samples[0], self._samples[-1]
        if last_count <= first_count or last_time <= first_time:
            return None
        return (last_time - first_time) / (last_count - first_count)


class _ProgressAdapter:
    def __init__(
        self,
        operation: str,
        output_path: Path,
        *,
        verbose: bool,
        progress_callback: ProgressCallback | None,
        progress_event_callback: ProgressEventCallback | None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.operation = operation
        self.output_path = output_path
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.progress_event_callback = progress_event_callback
        self._last_completed: int | None = None
        self.callback_error: BaseException | None = None
        self._rate = _PatchRate(clock)
        self._wall_clock = wall_clock or (lambda: datetime.now().astimezone())

    def _timing_status(self, completed: int, total: int) -> str:
        seconds = self._rate.seconds_per_patch()
        if seconds is None:
            return " | ETA pending"
        remaining = max(0, total - completed)
        finish = self._wall_clock() + timedelta(seconds=seconds * remaining)
        rate = f"{1.0 / seconds:.2f} patches/s" if seconds <= 1.0 else f"{seconds:.2f} s/patch"
        return (
            f" | {rate} | ETA {finish:%Y-%m-%d %H:%M:%S} "
            f"({_format_duration(seconds * remaining)} left)"
        )

    def __call__(self, private_event: Any) -> None:
        total = int(private_event.total_patches)
        completed = int(private_event.completed_patches)
        if str(private_event.state) in ("start", "valid", "invalid"):
            self._rate.add(completed)
        fraction = 1.0 if total == 0 else completed / total
        state = str(private_event.state)
        backend = getattr(private_event, "backend", None)
        message = f"{self.operation} {state}"
        invalid_patches = getattr(private_event, "invalid_patches", None)
        if state == "complete" and invalid_patches is not None:
            message += f" ({invalid_patches} invalid patches)"
        event = ProgressEvent(
            operation=self.operation,
            stage=state,
            completed=completed,
            total=total,
            fraction=fraction,
            backend=backend,
            message=message,
            tile_y=private_event.tile_y,
            tile_x=private_event.tile_x,
            path=self.output_path,
        )
        if self.verbose:
            if state == "start":
                selected = "unknown" if backend is None else backend
                print(
                    f"{self.operation}: using {selected} backend",
                    file=sys.stdout,
                    flush=True,
                )
            elif state in ("valid", "invalid", "complete"):
                suffix = (
                    f" ({invalid_patches} invalid patches)"
                    if state == "complete" and invalid_patches is not None
                    else ""
                )
                timing = "" if state == "complete" else self._timing_status(completed, total)
                print(
                    f"{self.operation}: {state} {completed}/{total}{suffix}{timing}",
                    file=sys.stdout,
                    flush=True,
                )
        if self.progress_event_callback is not None:
            try:
                self.progress_event_callback(event)
            except BaseException as exc:
                self.callback_error = exc
                raise
        if (
            self.progress_callback is not None
            and completed != self._last_completed
        ):
            try:
                self.progress_callback(fraction)
            except BaseException as exc:
                self.callback_error = exc
                raise
            self._last_completed = completed


def generate_lightmap(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    sun_vectors_m: npt.ArrayLike | None = None,
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    invalid_value: int = 0,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path:
    """Generate a timestamped, tiled uint8 visible-solar-fraction BigTIFF.

    Parameters
    ----------
    times:
        The UTC time domain as a :class:`TimeRange`.  Each sample becomes one
        band.  When ``sun_vectors_m`` is not supplied, Lunarscout generates
        geometric Moon-ME vectors for the Sun from this time range using
        SpiceyPy.
    sun_vectors_m:
        Optional explicit Moon-ME Sun vectors in meters, shape ``(time, 3)``.
        When supplied, they take precedence over SPICE generation and the first
        dimension must match the time count in ``times``.  Supplying vectors
        avoids SPICE import and kernel loading.
    backend:
        ``"auto"`` defaults to CUDA when a session can be initialized and
        otherwise uses CPU; ``"cpu"`` never probes CUDA; ``"cuda"`` never
        falls back and fails with a structured error if CUDA is unavailable.
    observer_height_m:
        Observer height above the DEM surface, in meters.
    invalid_value:
        The deterministic physical payload written into invalid pixels.  This
        value is stored alongside an authoritative dataset validity mask.  The
        default is zero, which may also be a valid science value for fully
        obscured pixels.
    output_transform:
        Optional callable applied patch-by-patch to valid calculated pixels
        before writing.  Must preserve the patch shape.  When supplied,
        ``output_dtype`` is required and the result must have exactly that
        dtype.
    output_dtype:
        Required when ``output_transform`` is supplied; must be a valid NumPy
        dtype.
    output_transform_id:
        Optional string identity that becomes part of the staged-job
        compatibility check.  Omitted on both original and restart runs means
        the jobs match.  Mismatched IDs reject restart.
    compress:
        When ``True`` (the default) tiles are compressed.  ``compress=False``
        produces tiled but uncompressed output.
    overwrite:
        When ``False`` (the default) an existing completed product raises
        :class:`ProductStorageError` before any DEM loading, SPICE, or CUDA
        work begins.
    start_fresh:
        When ``True``, discards any compatible staged work and begins again.
        Does not substitute for ``overwrite`` permission.
    skip_dem_nodata_check:
        When ``False`` (the default), pixels whose DEM node is invalid (NaN,
        the DEM's declared nodata value, or at or below -20000 m) are written
        as invalid, with the product's invalid value and dataset mask 0,
        because horizons computed there are meaningless.  Only the DEM window
        of each written patch is checked.  ``True`` skips the check, for DEMs
        known to contain no invalid nodes.
    verbose:
        When ``True``, writes concise backend and progress messages to stdout.
    progress_callback / progress_event_callback / cancellation_requested:
        See :ref:`shared-public-contracts` in the user guide.

    Returns
    -------
    pathlib.Path
        The resolved completed output path.  Backend information is recorded
        in progress events, staged metadata, and the final GeoTIFF
        ``LUNARSCOUT_COMPUTE_BACKENDS`` field, not in the return value.

    Notes
    -----
    Values are ``trunc(255 * visible_fraction)`` using a 16-slice solar-disk
    model with a 0.27-degree solar half-angle.  Invalid pixels carry
    ``invalid_value`` and are distinguished by the dataset validity mask.
    Compatible staged work resumes by default.  Cancellation leaves resumable
    staging state and never publishes an incomplete file.  The input DEM is
    read with its declared scale/offset applied and normalized to metres above
    the lunar reference sphere (1737.4 km); DEMs stored as radius-from-centre
    are converted to elevation-from-sphere automatically.
    """

    _validate_output_conversion(
        output_transform, output_dtype, output_transform_id
    )
    if backend not in ("auto", "cpu", "cuda"):
        raise InputError(
            "backend must be 'auto', 'cpu', or 'cuda'.",
            code="product_backend_invalid",
            details={"backend": backend},
        )
    if progress_callback is not None and not callable(progress_callback):
        raise InputError(
            "progress_callback must be callable or None.",
            code="product_callback_invalid",
        )
    if progress_event_callback is not None and not callable(progress_event_callback):
        raise InputError(
            "progress_event_callback must be callable or None.",
            code="product_callback_invalid",
        )
    if cancellation_requested is not None and not callable(cancellation_requested):
        raise InputError(
            "cancellation_requested must be callable or None.",
            code="product_callback_invalid",
        )
    horizons, output = _preflight_product_paths(
        horizons_path, output_path, overwrite=overwrite
    )
    dem, georef = _load_dem(dem_path)
    vectors = _resolve_vectors(
        "sun",
        vectors_m=sun_vectors_m,
        times=times,
    )
    from ._numba_horizon.cuda_backend import CudaBackendError
    from ._numba_horizon.file_format import HorizonTileStore
    from ._numba_horizon.lightmap_pipeline import (
        LightmapPipelineCancelled,
        run_lightmap_product,
    )
    from ._numba_horizon.product_store import ProductStoreError

    adapter = _ProgressAdapter(
        "lightmap",
        output,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
    )
    try:
        return run_lightmap_product(
            dem=dem,
            georef=georef,
            horizon_store=HorizonTileStore(horizons),
            output_path=output,
            times_utc=vectors.times_utc,
            sun_vectors_m=vectors.vectors_m,
            observer_elevation_m=observer_height_m,
            invalid_value=invalid_value,
            output_transform=output_transform,
            output_dtype=output_dtype,
            output_transform_id=output_transform_id,
            compress=compress,
            overwrite=overwrite,
            start_fresh=start_fresh,
            skip_dem_nodata_check=skip_dem_nodata_check,
            cancellation_requested=cancellation_requested,
            progress_callback=adapter,
            backend=backend,
        )
    except LightmapPipelineCancelled as exc:
        raise OperationCancelledError(
            "Lightmap generation was cancelled.",
            code="lightmap_cancelled",
            details={"path": str(output)},
        ) from exc
    except CudaBackendError as exc:
        raise CudaError(
            "The CUDA lightmap backend is unavailable.",
            code="cuda_lightmap_unavailable",
            details={"error": str(exc)},
        ) from exc
    except HorizonFormatError:
        raise
    except ProductStoreError as exc:
        raise ProductStorageError(
            "Lightmap storage failed.",
            code="lightmap_storage_failed",
            details={"path": str(output), "error": str(exc)},
        ) from exc
    except OSError as exc:
        raise ProductStorageError(
            "Lightmap file access failed.",
            code="lightmap_file_access_failed",
            details={"path": str(output), "error": str(exc)},
        ) from exc
    except ValueError as exc:
        if adapter.callback_error is exc:
            raise
        raise ProductCalculationError(
            "Lightmap calculation failed.",
            code="lightmap_calculation_failed",
            details={"error": str(exc)},
        ) from exc
    except Exception as exc:
        if adapter.callback_error is exc:
            raise
        if _is_cuda_runtime_failure(exc):
            raise CudaError(
                "The CUDA lightmap backend failed during execution.",
                code="cuda_lightmap_execution_failed",
                details={"error": str(exc)},
            ) from exc
        raise ProductCalculationError(
            "Lightmap calculation failed.",
            code="lightmap_calculation_failed",
            details={"error": str(exc)},
        ) from exc


def generate_psr(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    sun_vectors_m: npt.ArrayLike | None = None,
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    invalid_value: int = 0,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path:
    """Generate a single-band tiled permanent-shadow classification GeoTIFF.

    Parameters
    ----------
    times:
        The UTC time domain as a :class:`TimeRange`.  When ``sun_vectors_m``
        is not supplied, Lunarscout generates geometric Moon-ME Sun vectors
        using SpiceyPy.  PSR reduces the candidate vectors rather than
        materializing a full Metonic lightmap cube.
    sun_vectors_m:
        Optional explicit Moon-ME Sun vectors in meters, shape ``(time, 3)``.
        When supplied, they take precedence and avoid SPICE import.
    backend:
        ``"auto"`` defaults to CUDA when available and otherwise uses CPU;
        ``"cpu"`` never probes CUDA; ``"cuda"`` never falls back.
    observer_height_m:
        Observer height above the DEM surface, in meters.
    invalid_value:
        The deterministic physical payload written into invalid pixels.  The
        default is zero.  An authoritative dataset validity mask distinguishes
        invalid pixels from valid data, so zeros are not treated as nodata.
    output_transform / output_dtype / output_transform_id:
        Optional per-patch conversion applied to valid pixels before writing.
        See :func:`generate_lightmap` for the shared contract.
    compress:
        When ``True`` (the default) tiles are compressed.  ``compress=False``
        produces tiled but uncompressed output.
    overwrite:
        When ``False`` (the default) an existing completed product is rejected
        before any expensive work begins.
    start_fresh:
        When ``True``, discards compatible staged work.  Does not grant
        ``overwrite`` permission.
    skip_dem_nodata_check:
        See :func:`generate_lightmap`.  Invalid DEM pixels are written with
        ``invalid_value`` and dataset mask 0 unless this is ``True``.

    Returns
    -------
    pathlib.Path
        The completed output path.  Backend selection is recorded in progress
        and file metadata, not in the return value.

    Notes
    -----
    Value 255 means that the upper solar limb never clears the interpolated
    terrain horizon for the supplied samples; value 0 means that it clears at
    least once.  Both values are valid science data.  The calculation uses a
    five-viewpoint vector-reduction heuristic and a 0.27-degree solar
    half-angle.  In QGIS, render both 0 and 255 as valid classes and use the
    dataset mask for transparency.  The input DEM is read with its declared
    scale/offset applied and normalized to metres above the lunar reference
    sphere (1737.4 km); DEMs stored as radius-from-centre are converted to
    elevation-from-sphere automatically.
    """

    _validate_output_conversion(
        output_transform, output_dtype, output_transform_id
    )
    if backend not in ("auto", "cpu", "cuda"):
        raise InputError(
            "backend must be 'auto', 'cpu', or 'cuda'.",
            code="product_backend_invalid",
            details={"backend": backend},
        )
    for name, callback in (
        ("progress_callback", progress_callback),
        ("progress_event_callback", progress_event_callback),
        ("cancellation_requested", cancellation_requested),
    ):
        if callback is not None and not callable(callback):
            raise InputError(
                f"{name} must be callable or None.",
                code="product_callback_invalid",
                details={"callback": name},
            )
    horizons, output = _preflight_product_paths(
        horizons_path, output_path, overwrite=overwrite
    )
    dem, georef = _load_dem(dem_path)
    vectors = _resolve_vectors(
        "sun",
        vectors_m=sun_vectors_m,
        times=times,
    )
    from ._numba_horizon.cuda_backend import CudaBackendError
    from ._numba_horizon.file_format import HorizonTileStore
    from ._numba_horizon.product_store import ProductStoreError
    from ._numba_horizon.psr_pipeline import PsrPipelineCancelled, run_psr_product

    adapter = _ProgressAdapter(
        "psr",
        output,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
    )
    try:
        return run_psr_product(
            dem=dem,
            georef=georef,
            horizon_store=HorizonTileStore(horizons),
            output_path=output,
            sun_vectors_m=vectors.vectors_m,
            observer_elevation_m=observer_height_m,
            invalid_value=invalid_value,
            output_transform=output_transform,
            output_dtype=output_dtype,
            output_transform_id=output_transform_id,
            compress=compress,
            overwrite=overwrite,
            start_fresh=start_fresh,
            skip_dem_nodata_check=skip_dem_nodata_check,
            cancellation_requested=cancellation_requested,
            progress_event_callback=adapter,
            backend=backend,
        )
    except PsrPipelineCancelled as exc:
        raise OperationCancelledError(
            "PSR generation was cancelled.",
            code="psr_cancelled",
            details={"path": str(output)},
        ) from exc
    except CudaBackendError as exc:
        raise CudaError(
            "The CUDA PSR backend is unavailable.",
            code="cuda_psr_unavailable",
            details={"error": str(exc)},
        ) from exc
    except HorizonFormatError:
        raise
    except ProductStoreError as exc:
        raise ProductStorageError(
            "PSR storage failed.",
            code="psr_storage_failed",
            details={"path": str(output), "error": str(exc)},
        ) from exc
    except OSError as exc:
        raise ProductStorageError(
            "PSR file access failed.",
            code="psr_file_access_failed",
            details={"path": str(output), "error": str(exc)},
        ) from exc
    except ValueError as exc:
        if adapter.callback_error is exc:
            raise
        raise ProductCalculationError(
            "PSR calculation failed.",
            code="psr_calculation_failed",
            details={"error": str(exc)},
        ) from exc
    except Exception as exc:
        if adapter.callback_error is exc:
            raise
        if _is_cuda_runtime_failure(exc):
            raise CudaError(
                "The CUDA PSR backend failed during execution.",
                code="cuda_psr_execution_failed",
                details={"error": str(exc)},
            ) from exc
        raise ProductCalculationError(
            "PSR calculation failed.",
            code="psr_calculation_failed",
            details={"error": str(exc)},
        ) from exc


def _generate_body_elevation(
    body: str,
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    vectors_m: npt.ArrayLike | None,
    backend: Backend,
    observer_height_m: float,
    nodata: float,
    output_transform: Callable[[np.ndarray], np.ndarray] | None,
    output_dtype: npt.DTypeLike | None,
    output_transform_id: str | None,
    compress: bool,
    overwrite: bool,
    start_fresh: bool,
    skip_dem_nodata_check: bool,
    verbose: bool,
    progress_callback: ProgressCallback | None,
    progress_event_callback: ProgressEventCallback | None,
    cancellation_requested: CancellationCheck | None,
) -> Path:
    _validate_output_conversion(
        output_transform, output_dtype, output_transform_id
    )
    if backend not in ("auto", "cpu", "cuda"):
        raise InputError(
            "backend must be 'auto', 'cpu', or 'cuda'.",
            code="product_backend_invalid",
            details={"backend": backend},
        )
    for name, callback in (
        ("progress_callback", progress_callback),
        ("progress_event_callback", progress_event_callback),
        ("cancellation_requested", cancellation_requested),
    ):
        if callback is not None and not callable(callback):
            raise InputError(
                f"{name} must be callable or None.",
                code="product_callback_invalid",
                details={"callback": name},
            )
    horizons, output = _preflight_product_paths(
        horizons_path, output_path, overwrite=overwrite
    )
    dem, georef = _load_dem(dem_path)
    vectors = _resolve_vectors(
        body,
        vectors_m=vectors_m,
        times=times,
    )
    from ._numba_horizon.cuda_backend import CudaBackendError
    from ._numba_horizon.elevation_pipeline import (
        BodyElevationPipelineCancelled,
        run_earth_elevation_product,
        run_sun_elevation_product,
    )
    from ._numba_horizon.file_format import HorizonTileStore
    from ._numba_horizon.product_store import ProductStoreError

    operation = f"{body}_elevation"
    adapter = _ProgressAdapter(
        operation,
        output,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
    )
    runner = (
        run_sun_elevation_product
        if body == "sun"
        else run_earth_elevation_product
    )
    vector_keyword = (
        {"sun_vectors_m": vectors.vectors_m}
        if body == "sun"
        else {"earth_vectors_m": vectors.vectors_m}
    )
    try:
        return runner(
            dem=dem,
            georef=georef,
            horizon_store=HorizonTileStore(horizons),
            output_path=output,
            times_utc=vectors.times_utc,
            observer_elevation_m=observer_height_m,
            nodata=nodata,
            output_transform=output_transform,
            output_dtype=output_dtype,
            output_transform_id=output_transform_id,
            compress=compress,
            overwrite=overwrite,
            start_fresh=start_fresh,
            skip_dem_nodata_check=skip_dem_nodata_check,
            cancellation_requested=cancellation_requested,
            progress_callback=adapter,
            backend=backend,
            **vector_keyword,
        )
    except BodyElevationPipelineCancelled as exc:
        raise OperationCancelledError(
            f"{body.title()} elevation generation was cancelled.",
            code=f"{body}_elevation_cancelled",
            details={"path": str(output)},
        ) from exc
    except CudaBackendError as exc:
        raise CudaError(
            f"The CUDA {body} elevation backend is unavailable.",
            code=f"cuda_{body}_elevation_unavailable",
            details={"error": str(exc)},
        ) from exc
    except HorizonFormatError:
        raise
    except ProductStoreError as exc:
        raise ProductStorageError(
            f"{body.title()} elevation storage failed.",
            code=f"{body}_elevation_storage_failed",
            details={"path": str(output), "error": str(exc)},
        ) from exc
    except OSError as exc:
        raise ProductStorageError(
            f"{body.title()} elevation file access failed.",
            code=f"{body}_elevation_file_access_failed",
            details={"path": str(output), "error": str(exc)},
        ) from exc
    except ValueError as exc:
        if adapter.callback_error is exc:
            raise
        raise ProductCalculationError(
            f"{body.title()} elevation calculation failed.",
            code=f"{body}_elevation_calculation_failed",
            details={"error": str(exc)},
        ) from exc
    except Exception as exc:
        if adapter.callback_error is exc:
            raise
        if _is_cuda_runtime_failure(exc):
            raise CudaError(
                f"The CUDA {body} elevation backend failed during execution.",
                code=f"cuda_{body}_elevation_execution_failed",
                details={"error": str(exc)},
            ) from exc
        raise ProductCalculationError(
            f"{body.title()} elevation calculation failed.",
            code=f"{body}_elevation_calculation_failed",
            details={"error": str(exc)},
        ) from exc


def generate_sun_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    sun_vectors_m: npt.ArrayLike | None = None,
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path:
    """Generate Sun-center elevation relative to the terrain horizon.

    Parameters
    ----------
    times:
        The UTC time domain as a :class:`TimeRange`.  Each sample becomes one
        tiled ``float32`` BigTIFF band in degrees.  When ``sun_vectors_m`` is
        not supplied, Lunarscout generates geometric Moon-ME Sun vectors from
        this time range.
    sun_vectors_m:
        Optional explicit Moon-ME Sun vectors in meters, shape ``(time, 3)``.
        Supplied vectors take precedence and avoid SPICE import.
    backend:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.  See :func:`generate_lightmap`.
    observer_height_m:
        Observer height above the DEM surface, in meters.
    nodata:
        The value stored in invalid pixels.  Defaults to ``NaN``, stored as
        float32 NaN in the TIFF.  An authoritative dataset mask is always
        written and is the preferred validity representation.
    output_transform / output_dtype / output_transform_id:
        Optional per-patch conversion.  See :func:`generate_lightmap`.
    compress:
        ``True`` (default) compresses tiles; ``False`` disables compression
        while preserving tiling.
    overwrite / start_fresh / skip_dem_nodata_check:
        See :func:`generate_lightmap` for the shared overwrite, restart, and
        DEM nodata contract.

    Returns
    -------
    pathlib.Path
        The completed output path.

    Notes
    -----
    Values are the Sun center's elevation relative to the interpolated terrain
    horizon at its azimuth, not elevation above a smooth local horizontal
    plane.  Compatible staged work resumes by default.  Cancellation leaves
    resumable staging state.  The input DEM is read with its declared
    scale/offset applied and normalized to metres above the lunar reference
    sphere (1737.4 km); DEMs stored as radius-from-centre are converted to
    elevation-from-sphere automatically.
    """

    return _generate_body_elevation(
        "sun",
        dem_path,
        horizons_path,
        output_path,
        times=times,
        vectors_m=sun_vectors_m,
        backend=backend,
        observer_height_m=observer_height_m,
        nodata=nodata,
        output_transform=output_transform,
        output_dtype=output_dtype,
        output_transform_id=output_transform_id,
        compress=compress,
        overwrite=overwrite,
        start_fresh=start_fresh,
        skip_dem_nodata_check=skip_dem_nodata_check,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        cancellation_requested=cancellation_requested,
    )


def generate_earth_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    earth_vectors_m: npt.ArrayLike | None = None,
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path:
    """Generate Earth-center elevation relative to the terrain horizon.

    Parameters
    ----------
    times:
        The UTC time domain as a :class:`TimeRange`.  Each sample becomes one
        tiled ``float32`` BigTIFF band in degrees.  When ``earth_vectors_m``
        is not supplied, Lunarscout generates geometric Moon-ME Earth vectors
        from this time range.
    earth_vectors_m:
        Optional explicit Moon-ME Earth vectors in meters, shape ``(time, 3)``.
        Supplied vectors take precedence and avoid SPICE import.
    backend:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.  See :func:`generate_lightmap`.
    observer_height_m:
        Observer height above the DEM surface, in meters.
    nodata:
        The value stored in invalid pixels.  Defaults to ``NaN``.
        An authoritative dataset mask is always written.
    output_transform / output_dtype / output_transform_id:
        Optional per-patch conversion.  See :func:`generate_lightmap`.
    compress:
        ``True`` (default) compresses tiles; ``False`` disables compression
        while preserving tiling.
    overwrite / start_fresh / skip_dem_nodata_check:
        See :func:`generate_lightmap`.

    Returns
    -------
    pathlib.Path
        The completed output path.

    Notes
    -----
    Values are the Earth center's elevation relative to the interpolated
    terrain horizon at its azimuth.  Compatible staged work resumes by
    default.  The input DEM is read with its declared scale/offset applied and
    normalized to metres above the lunar reference sphere (1737.4 km); DEMs
    stored as radius-from-centre are converted to elevation-from-sphere
    automatically.
    """

    return _generate_body_elevation(
        "earth",
        dem_path,
        horizons_path,
        output_path,
        times=times,
        vectors_m=earth_vectors_m,
        backend=backend,
        observer_height_m=observer_height_m,
        nodata=nodata,
        output_transform=output_transform,
        output_dtype=output_dtype,
        output_transform_id=output_transform_id,
        compress=compress,
        overwrite=overwrite,
        start_fresh=start_fresh,
        skip_dem_nodata_check=skip_dem_nodata_check,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        cancellation_requested=cancellation_requested,
    )


@dataclass(frozen=True, slots=True)
class SafeHavenProducts:
    """Paths of the three safe-haven GeoTIFFs written by one run."""

    hibernation: Path
    joint_total: Path
    joint_longest: Path

    def __iter__(self):
        return iter((self.hibernation, self.joint_total, self.joint_longest))


def generate_safe_haven_products(
    dem_path: str | Path,
    horizons_path: str | Path | None,
    output_directory: str | Path,
    *,
    times: TimeRange,
    light_store_path: str | Path | None = None,
    sun_vectors_m: npt.ArrayLike | None = None,
    earth_vectors_m: npt.ArrayLike | None = None,
    earth_elevation_threshold_deg: float = 2.0,
    sunlight_fraction_threshold: float = 0.2,
    operating_sunlight_fraction: float = 1.0,
    merge_gap_hours: float = 24.0,
    degenerate_outage_window_hours: float | None = None,
    max_slope_deg: float | None = 20.0,
    name_prefix: str = "safe_haven",
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> SafeHavenProducts:
    """Generate monthly safe-haven screening maps for solar-powered rovers.

    Writes three GeoTIFFs into ``output_directory`` from one pass over the Sun
    and Earth, each with one float32 band per UTC calendar month, in hours:

    ``<name_prefix>_hibernation_hours.tif``
        For each month, the best (shortest) power-negative time among the
        hibernations that start in that month.  See Notes.
    ``<name_prefix>_joint_operating_total_hours.tif``
        Hours in the month with Earth available and the Sun at least
        ``operating_sunlight_fraction`` visible.
    ``<name_prefix>_joint_operating_longest_hours.tif``
        Longest uninterrupted stretch of such hours, credited to the month in
        which the stretch starts.

    The products are screening aids for choosing where to study more closely;
    they do not model a particular rover, its battery, or its route.

    Parameters
    ----------
    dem_path : str or pathlib.Path
        Primary DEM GeoTIFF defining the output grid. Its declared scale and
        offset are applied and elevations are normalized to metres above the
        1737.4-km lunar reference sphere.
    horizons_path : str or pathlib.Path
        Directory of precomputed 128-pixel horizon tiles for the DEM grid and
        ``observer_height_m``. This call does not generate horizons.
    output_directory : str or pathlib.Path
        Directory for the three GeoTIFFs. Created as needed. Each completed
        file is published atomically.
    times : TimeRange
        UTC sampling domain with at least two strictly increasing, uniformly
        spaced timestamps. Bands are created for months containing samples,
        including partially sampled months.
    sun_vectors_m, earth_vectors_m : array-like, optional
        Geometric Moon-centred Sun and Earth positions in the Moon-ME frame,
        in metres, shape ``(times.time_count, 3)``. ``None`` generates them
        with SPICE; supplying both avoids importing SpiceyPy.
    earth_elevation_threshold_deg : float, default 2.0
        Earth-centre elevation above the pixel's own terrain horizon below
        which there is no reliable line of sight (strictly below; equality is
        available). The rover hibernates for as long as this holds.
    sunlight_fraction_threshold : float, default 0.2
        A sample is shadow when the visible fraction of the solar disk is
        strictly below this value; the rover is power-negative in shadow.
    operating_sunlight_fraction : float, default 1.0
        Visible solar fraction at or above which the rover can operate. The
        comparison allows a tolerance of 1e-6 so that a fully lit disk, whose
        float32 fraction can fall a rounding step short of 1, qualifies.
    merge_gap_hours : float, default 24.0
        Two shadow runs separated by a sunlit gap shorter than this are one
        run, and the gap counts toward its length: a rover cannot recharge in
        a short gap. Zero disables merging.
    degenerate_outage_window_hours : float or None, default None
        Where Earth never goes below ``earth_elevation_threshold_deg`` in an
        Earth cycle (the Earth side of the map), the cycle has no outage. The
        rule gives it one: a single outage sample at the minimum of Earth's
        elevation above a flat horizon, if Earth stays at or above the threshold
        for this many hours either side of it. Rounded to whole time steps; an
        explicit value must be zero (rule off) or one to 15 steps. ``None`` means
        24 hours, or 15 steps when the step is shorter than 1.6 hours. See Notes.
    max_slope_deg : float or None, default 20.0
        Pixels steeper than this (Horn slope, as ``lunarscout.slope``), or whose
        slope cannot be computed, are NODATA in all three products. ``None``
        disables the mask.
    name_prefix : str, default "safe_haven"
        Prefix of the three file names.
    backend : {"auto", "cpu", "cuda"}, default "auto"
        Calculation backend. ``"cpu"`` does not probe CUDA. ``"auto"`` uses
        CUDA when it can initialize a session, otherwise CPU. Explicit
        ``"cuda"`` raises a structured CUDA error if unavailable and never
        falls back.
    observer_height_m : float, default 0.0
        Observer height above the DEM surface; selects the stored horizon
        tiles. Horizons for a surface observer are never lower than for a raised
        solar array, so surface horizons err toward more shadow and less
        operating time.
    nodata : float, default numpy.nan
        GeoTIFF nodata metadata and the fill value for patches with missing or
        unreadable horizons. The reducers always emit NaN where a value is
        undefined; use ``output_transform`` to encode these for other dtypes.
    output_transform, output_dtype, output_transform_id
        Optional conversion of each band's patch before writing, applied to all
        three products; see the product-storage documentation.
    compress : bool, default True
        DEFLATE-compress the tiled output.
    overwrite : bool, default False
        Permit replacing existing completed files.
    start_fresh : bool, default False
        Discard staged output and restart. Otherwise compatible staged work
        is resumed; a patch is skipped only when all three products hold it.
    skip_dem_nodata_check : bool, default False
        Skip writing NODATA where the DEM node is invalid.
    verbose : bool, default False
        Print backend and patch-completion messages.
    progress_callback, progress_event_callback, cancellation_requested
        As for the other products. Horizon tiles are read ahead and products are
        written behind the calculation on helper threads, so a progress callback
        can be called from a helper thread; calls are serialised, never
        concurrent, and an exception raised by a callback reaches the caller
        unchanged. On cancellation, patches already calculated are still written
        before the call raises, and the run resumes from them.

    Returns
    -------
    SafeHavenProducts
        The three completed file paths.

    Raises
    ------
    InputError
        Invalid backend, callback, parameter, or output-conversion arguments.
    ProductTimeError
        Insufficient, nonuniform, or inconsistent sampling timestamps.
    VectorError
        Invalid explicitly supplied Sun or Earth vectors.
    CudaError
        Explicit CUDA selection cannot initialize or CUDA execution fails.
    OperationCancelledError
        The cancellation callback requests cancellation.
    ProductStorageError
        Output exists without overwrite permission, incompatible staged work,
        or a storage failure.
    ProductCalculationError
        Calculation or patch-conversion failure.

    Notes
    -----
    **Hibernation.** The rover hibernates whenever Earth is below
    ``earth_elevation_threshold_deg`` above the local horizon, because there is
    no reliable radio link. A *hibernation* is one maximal such interval. In
    shadow (``sunlight_fraction_threshold``) the hibernating rover is
    power-negative. The *score* of a hibernation is the length of the longest
    shadow run that intersects it, counting the whole run, including the parts
    before and after the outage, and bridging sunlit gaps shorter than
    ``merge_gap_hours``. A hibernation with no shadow on it scores 0, the best
    possible. Durations are ``samples * step`` in hours.

    **Degenerate outage.** An Earth cycle (about 27 days) in which Earth stays
    above the threshold has no outage, so it would have no hibernation. Earth's
    elevation above a flat horizon has one minimum per cycle; if there is no
    outage within ``degenerate_outage_window_hours`` of it, that one sample is
    defined as an outage (a hibernation of one sample, scored like any other,
    with Earth unavailable for the joint-operating layers at that sample).
    Minima within the window of either end of the series are not credited.

    **Months.** Each hibernation belongs to the calendar month containing its
    first sample, and nothing else decides the month: a shadow run that starts
    later, or continues into the next month, does not move it. A hibernation
    is therefore counted exactly once. When two hibernations start in one
    month, the band holds the smaller (better) of their scores.

    **NODATA.** A hibernation band is NaN where no hibernation starts in the
    month (Earth always below the threshold, no cycle minimum falls in the month
    with the rule satisfied, or the month begins inside a hibernation that
    started earlier), where the slope exceeds
    ``max_slope_deg``, and where the DEM node is invalid. A hibernation already
    under way at the first sample has an unknown start and is not credited.
    Zero means a hibernation starts in the month and no shadow touches it. The
    joint-operating bands are NaN only for steep or invalid pixels.

    **Sampling and truncation.** Runs touching the first or last sample are
    limited to the available samples. Six-hour sampling is coarse: a sunlit
    gap of three samples is between 12 and 24 hours long, so the merge
    threshold is accurate only to about one sample.

    The per-pixel state machine that does this in one forward pass over time,
    on both CPU and CUDA, is specified in ``docs/safe-haven-state-machine.md``.
    ``backend="cuda"`` keeps all lighting batches and run state on the device;
    memory is bounded by the patch, time-batch and month counts, not by the
    number of samples.

    ``light_store_path`` replaces ``horizons_path`` (pass ``None`` for it) with a
    stage-0 light store: a directory of per-patch thresholded lighting made by
    ``scripts/run_safe_haven_stage0.py``.  The patches are then reduced from the
    stored Sun and Earth bits, with no horizon read and no Sun vectors
    (``sun_vectors_m`` is ignored); Earth vectors are still used for the
    degenerate-outage rule.  ``times`` must lie on the store's step grid and the
    thresholds must be the ones the store was made with.  ``backend`` chooses where
    the stored bits are reduced (the device, or all CPU cores).  See ``docs/safe-haven-light-store.md``.
    """

    _validate_output_conversion(
        output_transform, output_dtype, output_transform_id
    )
    if backend not in ("auto", "cpu", "cuda"):
        raise InputError(
            "backend must be 'auto', 'cpu', or 'cuda'.",
            code="product_backend_invalid",
            details={"backend": backend},
        )
    for name, callback in (
        ("progress_callback", progress_callback),
        ("progress_event_callback", progress_event_callback),
        ("cancellation_requested", cancellation_requested),
    ):
        if callback is not None and not callable(callback):
            raise InputError(
                f"{name} must be callable or None.",
                code="product_callback_invalid",
                details={"callback": name},
            )
    if (
        isinstance(merge_gap_hours, bool)
        or not isinstance(merge_gap_hours, (int, float))
        or not np.isfinite(merge_gap_hours)
        or merge_gap_hours < 0
    ):
        raise InputError(
            "merge_gap_hours must be a finite, nonnegative number of hours.",
            code="safe_haven_merge_gap_invalid",
            details={"merge_gap_hours": merge_gap_hours},
        )
    if degenerate_outage_window_hours is not None and (
        isinstance(degenerate_outage_window_hours, bool)
        or not isinstance(degenerate_outage_window_hours, (int, float))
        or not np.isfinite(degenerate_outage_window_hours)
        or degenerate_outage_window_hours < 0
    ):
        raise InputError(
            "degenerate_outage_window_hours must be a finite, nonnegative number of hours.",
            code="safe_haven_degenerate_window_invalid",
            details={"degenerate_outage_window_hours": degenerate_outage_window_hours},
        )
    if (
        isinstance(operating_sunlight_fraction, bool)
        or not isinstance(operating_sunlight_fraction, (int, float))
        or not 0.0 <= operating_sunlight_fraction <= 1.0
    ):
        raise InputError(
            "operating_sunlight_fraction must be between zero and one.",
            code="safe_haven_operating_fraction_invalid",
            details={"operating_sunlight_fraction": operating_sunlight_fraction},
        )
    if max_slope_deg is not None and (
        isinstance(max_slope_deg, bool)
        or not isinstance(max_slope_deg, (int, float))
        or not np.isfinite(max_slope_deg)
        or max_slope_deg <= 0
    ):
        raise InputError(
            "max_slope_deg must be a positive, finite angle in degrees, or None.",
            code="safe_haven_max_slope_invalid",
            details={"max_slope_deg": max_slope_deg},
        )
    if (
        not isinstance(name_prefix, str)
        or not name_prefix
        or any(separator in name_prefix for separator in ("/", "\\", "\0"))
    ):
        raise InputError(
            "name_prefix must be a non-empty string without path separators.",
            code="safe_haven_name_prefix_invalid",
            details={"name_prefix": name_prefix},
        )
    directory = Path(output_directory).expanduser().resolve()
    output_paths = (
        directory / f"{name_prefix}_hibernation_hours.tif",
        directory / f"{name_prefix}_joint_operating_total_hours.tif",
        directory / f"{name_prefix}_joint_operating_longest_hours.tif",
    )
    if (horizons_path is None) == (light_store_path is None):
        raise InputError(
            "Give exactly one of horizons_path and light_store_path.",
            code="safe_haven_input_ambiguous",
        )
    horizons = None
    for candidate in output_paths:
        horizons, _checked = _preflight_product_paths(
            horizons_path if light_store_path is None else light_store_path,
            candidate,
            overwrite=overwrite,
            input_name="horizons" if light_store_path is None else "light store",
        )
    dem, georef = _load_dem(dem_path)
    time_values = times
    sun = None
    if light_store_path is None:
        sun = _resolve_vectors(
            "sun",
            vectors_m=sun_vectors_m,
            times=time_values,
        )
    earth = _resolve_vectors(
        "earth",
        vectors_m=earth_vectors_m,
        times=time_values,
    )
    if sun is not None and sun.times_utc != earth.times_utc:
        raise ProductTimeError(
            "Sun and Earth vectors must use the same UTC timestamps.",
            code="safe_haven_times_mismatch",
        )
    if len(earth.times_utc) < 2:
        raise ProductTimeError(
            "Safe-haven generation requires at least two time samples.",
            code="safe_haven_times_insufficient",
        )
    steps = np.asarray(
        [
            (right - left).total_seconds() / 3600.0
            for left, right in zip(earth.times_utc, earth.times_utc[1:])
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(steps)) or steps[0] <= 0.0 or not np.allclose(
        steps, steps[0], rtol=0.0, atol=1e-9
    ):
        raise ProductTimeError(
            "Safe-haven timestamps must be strictly increasing and uniformly spaced.",
            code="safe_haven_times_not_uniform",
        )
    from ._numba_horizon.cuda_backend import CudaBackendError
    from ._numba_horizon.file_format import HorizonTileStore
    from ._numba_horizon.product_store import ProductStoreError
    from ._numba_horizon.safe_haven import degenerate_window_samples
    from ._numba_horizon.safe_haven_pipeline import (
        SafeHavenPipelineCancelled,
        run_safe_haven_products,
    )

    light_store = None
    if light_store_path is not None:
        from ._numba_horizon.safe_haven_stage0 import LightStoreDirectory

        from ._numba_horizon import safe_haven_stage1_store as from_store

        light_store = LightStoreDirectory(horizons, dem)
        # Refuse a store that cannot answer this request here, as input errors, rather than from
        # inside the run.
        from_store.check_thresholds(
            light_store,
            sunlight_threshold=sunlight_fraction_threshold,
            operating_sunlight_fraction=float(operating_sunlight_fraction),
            earth_threshold_deg=earth_elevation_threshold_deg,
        )
        from_store.first_store_step(light_store, earth.times_utc, float(steps[0]))
    try:
        window_samples = degenerate_window_samples(
            degenerate_outage_window_hours, float(steps[0])
        )
    except ValueError as exc:
        raise InputError(
            str(exc),
            code="safe_haven_degenerate_window_invalid",
            details={
                "degenerate_outage_window_hours": degenerate_outage_window_hours,
                "time_step_hours": float(steps[0]),
            },
        ) from exc
    adapter = _ProgressAdapter(
        "safe_havens",
        directory,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
    )
    try:
        return SafeHavenProducts(*run_safe_haven_products(
            dem=dem,
            georef=georef,
            horizon_store=None if light_store is not None else HorizonTileStore(horizons),
            light_store=light_store,
            output_paths=output_paths,
            times_utc=earth.times_utc,
            sun_vectors_m=None if sun is None else sun.vectors_m,
            earth_vectors_m=earth.vectors_m,
            time_step_hours=float(steps[0]),
            earth_threshold_deg=earth_elevation_threshold_deg,
            sunlight_threshold=sunlight_fraction_threshold,
            operating_sunlight_fraction=float(operating_sunlight_fraction),
            merge_gap_hours=float(merge_gap_hours),
            degenerate_window_samples=window_samples,
            max_slope_deg=None if max_slope_deg is None else float(max_slope_deg),
            observer_elevation_m=observer_height_m,
            nodata=nodata,
            output_transform=output_transform,
            output_dtype=output_dtype,
            output_transform_id=output_transform_id,
            compress=compress,
            overwrite=overwrite,
            start_fresh=start_fresh,
            skip_dem_nodata_check=skip_dem_nodata_check,
            backend=backend,
            cancellation_requested=cancellation_requested,
            progress_callback=adapter,
        ))
    except SafeHavenPipelineCancelled as exc:
        raise OperationCancelledError(
            "Safe-haven generation was cancelled.",
            code="safe_haven_cancelled",
            details={"path": str(directory)},
        ) from exc
    except CudaBackendError as exc:
        raise CudaError(
            "The CUDA safe-haven backend is unavailable.",
            code="cuda_safe_haven_unavailable",
            details={"error": str(exc)},
        ) from exc
    except HorizonFormatError:
        raise
    except ProductStoreError as exc:
        raise ProductStorageError(
            "Safe-haven storage failed.",
            code="safe_haven_storage_failed",
            details={"path": str(directory), "error": str(exc)},
        ) from exc
    except OSError as exc:
        raise ProductStorageError(
            "Safe-haven file access failed.",
            code="safe_haven_file_access_failed",
            details={"path": str(directory), "error": str(exc)},
        ) from exc
    except ValueError as exc:
        if adapter.callback_error is exc:
            raise
        raise ProductCalculationError(
            "Safe-haven calculation failed.",
            code="safe_haven_calculation_failed",
            details={"error": str(exc)},
        ) from exc
    except Exception as exc:
        if adapter.callback_error is exc:
            raise
        if _is_cuda_runtime_failure(exc):
            raise CudaError(
                "The CUDA safe-haven backend failed during execution.",
                code="cuda_safe_haven_execution_failed",
                details={"error": str(exc)},
            ) from exc
        raise ProductCalculationError(
            "Safe-haven calculation failed.",
            code="safe_haven_calculation_failed",
            details={"error": str(exc)},
        ) from exc
def _generate_mission_duration(
    mode: str,
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: TimeInput,
    evaluation_stop: TimeInput,
    step: timedelta,
    candidate_start_intervals: Iterable[tuple[TimeInput, TimeInput]],
    sun_vectors_m: npt.ArrayLike | None,
    earth_vectors_m: npt.ArrayLike | None,
    sunlight_fraction_threshold: float | None,
    sun_elevation_threshold_deg: float | None,
    earth_elevation_threshold_deg: float | None,
    output_unit: Literal["hours", "days"],
    backend: Backend,
    observer_height_m: float,
    nodata: float,
    output_transform: Callable[[np.ndarray], np.ndarray] | None,
    output_dtype: npt.DTypeLike | None,
    output_transform_id: str | None,
    compress: bool,
    overwrite: bool,
    start_fresh: bool,
    skip_dem_nodata_check: bool,
    verbose: bool,
    progress_callback: ProgressCallback | None,
    progress_event_callback: ProgressEventCallback | None,
    cancellation_requested: CancellationCheck | None,
) -> Path:
    _validate_output_conversion(
        output_transform, output_dtype, output_transform_id
    )
    if backend not in ("auto", "cpu", "cuda"):
        raise InputError(
            "backend must be 'auto', 'cpu', or 'cuda'.",
            code="product_backend_invalid",
            details={"backend": backend},
        )
    if output_unit not in ("hours", "days"):
        raise InputError(
            "output_unit must be 'hours' or 'days'.",
            code="mission_duration_unit_invalid",
            details={"output_unit": output_unit},
        )
    for name, callback in (
        ("progress_callback", progress_callback),
        ("progress_event_callback", progress_event_callback),
        ("cancellation_requested", cancellation_requested),
    ):
        if callback is not None and not callable(callback):
            raise InputError(
                f"{name} must be callable or None.",
                code="product_callback_invalid",
                details={"callback": name},
            )
    horizons, output = _preflight_product_paths(
        horizons_path, output_path, overwrite=overwrite
    )
    dem, georef = _load_dem(dem_path)
    from .spice_geometry import iter_times

    try:
        time_values = tuple(iter_times(evaluation_start, evaluation_stop, step))
    except Exception as exc:
        raise ProductTimeError(
            "Mission-duration evaluation times are invalid.",
            code="mission_duration_times_invalid",
            details={"error": str(exc)},
        ) from exc
    intervals = tuple(candidate_start_intervals)
    sun = _resolve_vectors(
        "sun",
        vectors_m=sun_vectors_m,
        times=time_values,
    )
    earth = None
    if earth_elevation_threshold_deg is not None:
        earth = _resolve_vectors(
            "earth",
            vectors_m=earth_vectors_m,
            times=time_values,
        )
        if earth.times_utc != sun.times_utc:
            raise ProductTimeError(
                "Sun and Earth vectors must use the same UTC timestamps.",
                code="mission_duration_times_mismatch",
            )
    from ._numba_horizon.cuda_backend import CudaBackendError
    from ._numba_horizon.file_format import HorizonTileStore
    from ._numba_horizon.mission_duration_pipeline import (
        MissionDurationPipelineCancelled,
        run_sun_elevation_duration_product,
        run_sun_elevation_earth_elevation_duration_product,
        run_sunlight_duration_product,
        run_sunlight_earth_elevation_duration_product,
    )
    from ._numba_horizon.product_store import ProductStoreError

    runners = {
        "sunlight": run_sunlight_duration_product,
        "sun_elevation": run_sun_elevation_duration_product,
        "sunlight_earth": run_sunlight_earth_elevation_duration_product,
        "sun_earth_elevation": run_sun_elevation_earth_elevation_duration_product,
    }
    threshold_kwargs: dict[str, Any] = {}
    if sunlight_fraction_threshold is not None:
        threshold_kwargs["sunlight_fraction_threshold"] = sunlight_fraction_threshold
    if sun_elevation_threshold_deg is not None:
        threshold_kwargs["sun_elevation_threshold_deg"] = sun_elevation_threshold_deg
    if earth_elevation_threshold_deg is not None:
        assert earth is not None
        threshold_kwargs["earth_elevation_threshold_deg"] = earth_elevation_threshold_deg
        threshold_kwargs["earth_vectors_m"] = earth.vectors_m
    operation = f"mission_duration_{mode}"
    adapter = _ProgressAdapter(
        operation,
        output,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
    )
    try:
        return runners[mode](
            dem=dem,
            georef=georef,
            horizon_store=HorizonTileStore(horizons),
            output_path=output,
            times_utc=sun.times_utc,
            evaluation_start_utc=evaluation_start,
            evaluation_stop_utc=evaluation_stop,
            start_intervals=intervals,
            sun_vectors_m=sun.vectors_m,
            output_unit=output_unit,
            observer_elevation_m=observer_height_m,
            nodata=nodata,
            output_transform=output_transform,
            output_dtype=output_dtype,
            output_transform_id=output_transform_id,
            compress=compress,
            overwrite=overwrite,
            start_fresh=start_fresh,
            skip_dem_nodata_check=skip_dem_nodata_check,
            cancellation_requested=cancellation_requested,
            progress_callback=adapter,
            backend=backend,
            **threshold_kwargs,
        )
    except MissionDurationPipelineCancelled as exc:
        raise OperationCancelledError(
            "Mission-duration generation was cancelled.",
            code="mission_duration_cancelled",
            details={"path": str(output), "mode": mode},
        ) from exc
    except CudaBackendError as exc:
        raise CudaError(
            "The CUDA mission-duration backend is unavailable.",
            code="cuda_mission_duration_unavailable",
            details={"error": str(exc), "mode": mode},
        ) from exc
    except HorizonFormatError:
        raise
    except ProductStoreError as exc:
        raise ProductStorageError(
            "Mission-duration storage failed.",
            code="mission_duration_storage_failed",
            details={"path": str(output), "error": str(exc), "mode": mode},
        ) from exc
    except OSError as exc:
        raise ProductStorageError(
            "Mission-duration file access failed.",
            code="mission_duration_file_access_failed",
            details={"path": str(output), "error": str(exc), "mode": mode},
        ) from exc
    except ValueError as exc:
        if adapter.callback_error is exc:
            raise
        raise ProductTimeError(
            str(exc),
            code="mission_duration_inputs_invalid",
            details={"mode": mode},
        ) from exc
    except Exception as exc:
        if adapter.callback_error is exc:
            raise
        if _is_cuda_runtime_failure(exc):
            raise CudaError(
                "The CUDA mission-duration backend failed during execution.",
                code="cuda_mission_duration_execution_failed",
                details={"error": str(exc), "mode": mode},
            ) from exc
        raise ProductCalculationError(
            "Mission-duration calculation failed.",
            code="mission_duration_calculation_failed",
            details={"error": str(exc), "mode": mode},
        ) from exc


def mission_duration_from_sunlight(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: TimeInput,
    evaluation_stop: TimeInput,
    step: timedelta,
    candidate_start_intervals: Iterable[tuple[TimeInput, TimeInput]],
    sunlight_fraction_threshold: float,
    sun_vectors_m: npt.ArrayLike | None = None,
    output_unit: Literal["hours", "days"] = "hours",
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path:
    """Longest inclusive sunlight-fraction duration for each start interval.

    Parameters
    ----------
    evaluation_start / evaluation_stop:
        The overall half-open evaluation interval defining the sample domain.
    step:
        The sampling step as a ``datetime.timedelta``.  Timestamps are
        generated from ``evaluation_start`` to ``evaluation_stop`` inclusive
        with this step.
    candidate_start_intervals:
        An iterable of half-open ``(start, stop)`` interval pairs.  Each
        interval controls where a qualifying run may start and becomes one
        output band.
    sunlight_fraction_threshold:
        Unitless inclusive lower bound.  A candidate run is valid while the
        sunlight fraction is ``>=`` this threshold.
    sun_vectors_m:
        Optional explicit Moon-ME Sun vectors, shape ``(time, 3)``.  When
        supplied, they take precedence and avoid SPICE import.
    output_unit:
        ``"hours"`` (default) or ``"days"``.  Days aggregate actual
        sample-to-sample durations and divide by 24 after reduction.
    backend:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.  See :func:`generate_lightmap`.
    observer_height_m:
        Observer height above the DEM surface, in meters.
    nodata:
        Value stored in invalid pixels.  Defaults to ``NaN``.
    output_transform / output_dtype / output_transform_id:
        Optional per-patch conversion.  See :func:`generate_lightmap`.
    compress:
        ``True`` (default) compresses tiles; ``False`` disables compression
        while preserving tiling.
    overwrite / start_fresh / skip_dem_nodata_check:
        See :func:`generate_lightmap`.

    Returns
    -------
    pathlib.Path
        The completed output path with one ``float32`` band per
        candidate-start interval.

    Notes
    -----
    The condition sampled at ``times[i]`` applies over
    ``[times[i], times[i+1])``, clipped to ``evaluation_stop``.  A run may
    begin at any qualifying sample inside a candidate-start interval and may
    continue beyond it but never beyond the overall evaluation stop.  A run
    still active at the evaluation stop is right-censored.  The input DEM is
    read with its declared scale/offset applied and normalized to metres above
    the lunar reference sphere (1737.4 km); DEMs stored as radius-from-centre
    are converted to elevation-from-sphere automatically.
    """

    return _generate_mission_duration(
        "sunlight", dem_path, horizons_path, output_path,
        evaluation_start=evaluation_start, evaluation_stop=evaluation_stop,
        step=step,
        candidate_start_intervals=candidate_start_intervals,
        sun_vectors_m=sun_vectors_m, earth_vectors_m=None,
        sunlight_fraction_threshold=sunlight_fraction_threshold,
        sun_elevation_threshold_deg=None, earth_elevation_threshold_deg=None,
        output_unit=output_unit, backend=backend,
        observer_height_m=observer_height_m, nodata=nodata,
        output_transform=output_transform, output_dtype=output_dtype,
        output_transform_id=output_transform_id,
        compress=compress,
        overwrite=overwrite, start_fresh=start_fresh, verbose=verbose,
        skip_dem_nodata_check=skip_dem_nodata_check,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        cancellation_requested=cancellation_requested,
    )


def mission_duration_from_sun_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: TimeInput,
    evaluation_stop: TimeInput,
    step: timedelta,
    candidate_start_intervals: Iterable[tuple[TimeInput, TimeInput]],
    sun_elevation_threshold_deg: float,
    sun_vectors_m: npt.ArrayLike | None = None,
    output_unit: Literal["hours", "days"] = "hours",
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path:
    """Longest inclusive Sun terrain-relative elevation duration.

    ``sun_elevation_threshold_deg`` is an inclusive lower bound in degrees
    for the Sun center's elevation relative to the terrain horizon.

    All other parameters and semantics match
    :func:`mission_duration_from_sunlight`; see its docstring for the
    complete evaluation-interval, candidate-start, duration, backend, and
    output lifecycle contract.
    """

    return _generate_mission_duration(
        "sun_elevation", dem_path, horizons_path, output_path,
        evaluation_start=evaluation_start, evaluation_stop=evaluation_stop,
        step=step,
        candidate_start_intervals=candidate_start_intervals,
        sun_vectors_m=sun_vectors_m, earth_vectors_m=None,
        sunlight_fraction_threshold=None,
        sun_elevation_threshold_deg=sun_elevation_threshold_deg,
        earth_elevation_threshold_deg=None, output_unit=output_unit,
        backend=backend, observer_height_m=observer_height_m,
        nodata=nodata,
        output_transform=output_transform,
        output_dtype=output_dtype,
        output_transform_id=output_transform_id,
        compress=compress, overwrite=overwrite,
        start_fresh=start_fresh, verbose=verbose,
        skip_dem_nodata_check=skip_dem_nodata_check,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        cancellation_requested=cancellation_requested,
    )


def mission_duration_from_sunlight_and_earth_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: TimeInput,
    evaluation_stop: TimeInput,
    step: timedelta,
    candidate_start_intervals: Iterable[tuple[TimeInput, TimeInput]],
    sunlight_fraction_threshold: float,
    earth_elevation_threshold_deg: float,
    sun_vectors_m: npt.ArrayLike | None = None,
    earth_vectors_m: npt.ArrayLike | None = None,
    output_unit: Literal["hours", "days"] = "hours",
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path:
    """Longest duration meeting inclusive sunlight and Earth thresholds.

    ``sunlight_fraction_threshold`` is a unitless inclusive lower bound.
    ``earth_elevation_threshold_deg`` is an inclusive lower bound in degrees
    for the Earth center's elevation relative to the terrain horizon.

    All other parameters and semantics match
    :func:`mission_duration_from_sunlight`; see its docstring for the
    complete evaluation-interval, candidate-start, duration, backend, and
    output lifecycle contract.
    """

    return _generate_mission_duration(
        "sunlight_earth", dem_path, horizons_path, output_path,
        evaluation_start=evaluation_start, evaluation_stop=evaluation_stop,
        step=step,
        candidate_start_intervals=candidate_start_intervals,
        sun_vectors_m=sun_vectors_m, earth_vectors_m=earth_vectors_m,
        sunlight_fraction_threshold=sunlight_fraction_threshold,
        sun_elevation_threshold_deg=None,
        earth_elevation_threshold_deg=earth_elevation_threshold_deg,
        output_unit=output_unit, backend=backend,
        observer_height_m=observer_height_m, nodata=nodata,
        output_transform=output_transform, output_dtype=output_dtype,
        output_transform_id=output_transform_id,
        compress=compress,
        overwrite=overwrite, start_fresh=start_fresh, verbose=verbose,
        skip_dem_nodata_check=skip_dem_nodata_check,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        cancellation_requested=cancellation_requested,
    )


def mission_duration_from_sun_and_earth_elevation(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_start: TimeInput,
    evaluation_stop: TimeInput,
    step: timedelta,
    candidate_start_intervals: Iterable[tuple[TimeInput, TimeInput]],
    sun_elevation_threshold_deg: float,
    earth_elevation_threshold_deg: float,
    sun_vectors_m: npt.ArrayLike | None = None,
    earth_vectors_m: npt.ArrayLike | None = None,
    output_unit: Literal["hours", "days"] = "hours",
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    skip_dem_nodata_check: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path:
    """Longest duration meeting inclusive Sun and Earth elevation thresholds.

    Both terrain-relative elevation thresholds are inclusive lower bounds in
    degrees.

    All other parameters and semantics match
    :func:`mission_duration_from_sunlight`; see its docstring for the
    complete evaluation-interval, candidate-start, duration, backend, and
    output lifecycle contract.
    """

    return _generate_mission_duration(
        "sun_earth_elevation", dem_path, horizons_path, output_path,
        evaluation_start=evaluation_start, evaluation_stop=evaluation_stop,
        step=step,
        candidate_start_intervals=candidate_start_intervals,
        sun_vectors_m=sun_vectors_m, earth_vectors_m=earth_vectors_m,
        sunlight_fraction_threshold=None,
        sun_elevation_threshold_deg=sun_elevation_threshold_deg,
        earth_elevation_threshold_deg=earth_elevation_threshold_deg,
        output_unit=output_unit, backend=backend,
        observer_height_m=observer_height_m, nodata=nodata,
        output_transform=output_transform, output_dtype=output_dtype,
        output_transform_id=output_transform_id,
        compress=compress,
        overwrite=overwrite, start_fresh=start_fresh, verbose=verbose,
        skip_dem_nodata_check=skip_dem_nodata_check,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        cancellation_requested=cancellation_requested,
    )
