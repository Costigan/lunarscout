"""Public Python-only horizon-derived product functions."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import timedelta
from pathlib import Path
import sys
from typing import Any, Literal
import warnings

import numpy as np
import numpy.typing as npt
from pyproj import CRS

from .errors import (
    CudaError,
    GridError,
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
) -> npt.NDArray[np.float32]:
    """Convert raw DEM samples to metres above the reference sphere.

    Applies the raster's declared scale and offset.  An offset that encodes
    the reference-sphere radius (i.e. values stored as radius-from-centre) is
    undone so the result is elevation relative to the sphere.  When no scale
    or offset is declared, radius-from-centre values are detected by magnitude
    and shifted down by the reference radius.
    """
    result = np.asarray(values, dtype=np.float32)
    if not result.flags.c_contiguous:
        result = np.ascontiguousarray(result, dtype=np.float32)
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
    if not transformed:
        finite_min = float(np.nanmin(result))
        if np.isfinite(finite_min) and finite_min > 0.5 * radius:
            result -= radius
    return np.ascontiguousarray(result, dtype=np.float32)


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
            values, scale=band_scale, offset=band_offset, radius_m=float(radius_m)
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
) -> tuple[Path, Path]:
    horizons = Path(horizons_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not horizons.is_dir():
        raise InputError(
            "The horizons path does not identify a directory.",
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


class _ProgressAdapter:
    def __init__(
        self,
        operation: str,
        output_path: Path,
        *,
        verbose: bool,
        progress_callback: ProgressCallback | None,
        progress_event_callback: ProgressEventCallback | None,
    ) -> None:
        self.operation = operation
        self.output_path = output_path
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.progress_event_callback = progress_event_callback
        self._last_completed: int | None = None
        self.callback_error: BaseException | None = None

    def __call__(self, private_event: Any) -> None:
        total = int(private_event.total_patches)
        completed = int(private_event.completed_patches)
        fraction = 1.0 if total == 0 else completed / total
        state = str(private_event.state)
        backend = getattr(private_event, "backend", None)
        message = f"{self.operation} {state}"
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
                print(
                    f"{self.operation}: {state} {completed}/{total}",
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
    overwrite / start_fresh:
        See :func:`generate_lightmap` for the shared overwrite and restart
        contract.

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
    overwrite / start_fresh:
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
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        cancellation_requested=cancellation_requested,
    )


def generate_safe_havens(
    dem_path: str | Path,
    horizons_path: str | Path,
    output_path: str | Path,
    *,
    times: TimeRange,
    sun_vectors_m: npt.ArrayLike | None = None,
    earth_vectors_m: npt.ArrayLike | None = None,
    earth_elevation_threshold_deg: float = 2.0,
    sunlight_fraction_threshold: float = 0.2,
    backend: Backend = "auto",
    observer_height_m: float = 0.0,
    nodata: float = np.nan,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    output_dtype: npt.DTypeLike | None = None,
    output_transform_id: str | None = None,
    compress: bool = True,
    overwrite: bool = False,
    start_fresh: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_event_callback: ProgressEventCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> Path:
    """Generate monthly maps of low-sunlight duration during Earth outages.

    The default output contains float32 durations in hours, one band per
    sampled UTC calendar month. See Notes for the scientific definition
    and allocation of low-Sun runs to bands.

    Parameters
    ----------
    dem_path : str or pathlib.Path
        Primary DEM GeoTIFF defining the output grid. Its declared scale and
        offset are applied and elevations are normalized to metres above the
        1737.4-km lunar reference sphere. Radius-from-centre DEMs are converted
        to elevations above that sphere.
    horizons_path : str or pathlib.Path
        Directory containing precomputed 128-pixel horizon tiles for the DEM
        grid and ``observer_height_m``. This call does not generate horizons.
    output_path : str or pathlib.Path
        Destination for the tiled, multi-band GeoTIFF. Parent directories are
        created as needed. Completed output is published atomically.
    times : TimeRange
        UTC sampling domain, with at least two strictly increasing, uniformly
        spaced timestamps. Use ``lunarscout.times`` to construct it. Its stop
        is inclusive when it falls on the sampling grid. Bands are created for
        months containing samples, including partially sampled months.
    sun_vectors_m : array-like, optional
        Geometric Moon-centered Sun positions in the Moon-ME frame, in metres,
        with finite values and shape ``(times.time_count, 3)``. Rows correspond
        to the timestamps in ``times``. Default ``None`` generates Sun vectors
        with SPICE. Explicit Sun vectors bypass SPICE for the Sun only.
    earth_vectors_m : array-like, optional
        Geometric Moon-centered Earth positions in the Moon-ME frame, in
        metres, with finite values and shape ``(times.time_count, 3)``. Default
        ``None`` generates Earth vectors with SPICE. Supply both vector arrays
        to avoid importing SpiceyPy and loading kernels for this operation.
    earth_elevation_threshold_deg : float, default 2.0
        Earth-center elevation threshold in degrees relative to each pixel's
        terrain horizon at the Earth's azimuth. A sample is an Earth outage
        when this elevation is strictly below the threshold. Equality counts
        as Earth being available. This is a geometric communication criterion;
        it does not model antenna, link-budget, or ground-station constraints.
    sunlight_fraction_threshold : float, default 0.2
        Unitless threshold for the visible fraction of the solar disk, whose
        physical range is zero to one. A sample is low-Sun when its fraction
        is strictly below the threshold; equality is not low-Sun. The default
        means less than 20 percent of the solar disk is visible. This fraction
        does not represent electrical power or a battery state of charge.
    backend : {"auto", "cpu", "cuda"}, default "auto"
        Calculation backend. ``"cpu"`` does not probe CUDA. ``"auto"`` uses
        CUDA when it can initialize a session, otherwise CPU. Explicit
        ``"cuda"`` raises a structured CUDA error if unavailable and never
        falls back to CPU.
    observer_height_m : float, default 0.0
        Observer height above the DEM surface, in metres. Selects the stored
        horizon tiles for that height; matching horizons must already exist.
        This call does not generate or adjust horizon tiles for a new height.
    nodata : float, default numpy.nan
        GeoTIFF nodata metadata and the fill value for patches whose horizons
        are missing or unreadable. The monthly reducer itself always emits
        NaN where Earth is never below the threshold or is below it at every
        evaluated sample in that month. A finite ``nodata`` does not replace
        those reducer NaNs automatically; use ``output_transform`` to encode
        them if a finite sentinel or integer storage is required.
    output_transform : callable, optional
        Optional conversion of each calculated band's two-dimensional patch
        before writing. Receives a duration array in hours that can contain
        NaNs. Must preserve its shape and return exactly ``output_dtype``.
        Handle NaNs explicitly and keep the chosen encoding consistent with
        ``nodata``. Missing-horizon patches are filled directly with ``nodata``
        and do not pass through this callable. Default ``None`` writes float32
        durations without conversion.
    output_dtype : numpy dtype-like, optional
        Required storage dtype when ``output_transform`` is provided; omitted
        otherwise. Accepts forms such as ``numpy.float32`` or ``"uint16"``.
        The configured ``nodata`` must be representable in the storage dtype.
    output_transform_id : str, optional
        Optional identity for the conversion, recorded in the staged job's
        compatibility metadata. Requires ``output_transform``. Reusing staged
        output requires the same ID; omitting the ID on both runs also matches.
        Choose an ID that changes when the conversion's meaning changes.
    compress : bool, default True
        Use DEFLATE compression for the tiled output. ``False`` retains tiling
        but disables compression.
    overwrite : bool, default False
        Permit replacement of an existing completed output. Without permission,
        an existing output raises ``ProductStorageError`` before calculation.
        A failed overwrite preserves the previous completed product; the new
        file is published only after all patches finish.
    start_fresh : bool, default False
        Discard staged output for this destination and restart calculation.
        Otherwise compatible staged work is resumed, skipping completed
        patches. This is separate from permission to overwrite a completed
        output. Do not discard staging while another process is writing it.
    verbose : bool, default False
        Print selected-backend and patch-completion messages to standard output.
    progress_callback : callable, optional
        Callable receiving a durable completion fraction in ``[0, 1]`` when
        the completed-patch count changes, including resumed work. Default
        ``None`` disables this callback.
    progress_event_callback : callable, optional
        Callable receiving immutable ``lunarscout.ProgressEvent`` objects with
        operation, stage, completed and total patch counts, fraction, backend,
        output path, and tile coordinates where applicable. Default ``None``
        disables this callback.
    cancellation_requested : callable, optional
        Zero-argument callable checked between bounded work units, including
        CPU time samples or CUDA time batches. Returning ``True`` raises
        ``OperationCancelledError``
        with code ``safe_haven_cancelled`` and leaves resumable staging state.
        Default ``None`` disables cancellation checks.

    Returns
    -------
    pathlib.Path
        Resolved completed output path, containing chronological monthly bands.

    Raises
    ------
    InputError
        Invalid backend, callback, or output-conversion arguments.
    ProductTimeError
        Insufficient, nonuniform, or inconsistent sampling timestamps.
    VectorError
        Invalid explicitly supplied Sun or Earth vectors.
    CudaError
        Explicit CUDA selection cannot initialize or CUDA execution fails.
    OperationCancelledError
        The cancellation callback requests cancellation.
    ProductStorageError
        Output already exists without overwrite permission, incompatible staged
        work, or a storage failure.
    ProductCalculationError
        Calculation or patch-conversion failure. Exceptions from the progress
        callbacks propagate unchanged.

    Notes
    -----
    Here a safe-haven product characterizes the low-sunlight duration that
    coincides with losing the geometric Earth link at a fixed terrain location.
    For each pixel and month, it reports the longest contiguous low-Sun run
    that includes at least one Earth-outage sample in that month. Shorter
    values indicate shorter low-sunlight runs associated with those outages.
    The product does not certify survival or select suitable landing sites:
    it does not include thermal, battery, terrain-slope, or rover models.

    With ``backend="cuda"``, Sun visibility, Earth elevation, and monthly
    run-duration reduction are calculated on the GPU. Lighting batches and
    run state remain in device memory; only the completed monthly duration
    patches are copied to the host for output conversion, compression, and
    writing. Memory use is bounded by the patch, time-batch, and month counts,
    rather than a full ``(time, y, x)`` lighting cube. ``backend="cpu"`` uses
    the streaming CPU reducer; ``"auto"`` selects one complete calculation
    backend for both bodies and the reduction.

    Bands are ordered chronologically, starting at GeoTIFF band 1. Each band
    corresponds to the UTC calendar interval ``[month start, next month start)``
    and is timestamped at the first day of that month at 00:00 UTC. Only months
    containing evaluation samples get bands. A partial month uses only its
    available samples, even though the band's timestamp names the full month.

    Earth visibility and solar fraction are calculated from each pixel's own
    terrain horizon. A low-Sun run is tracked independently of Earth visibility.
    If the run overlaps an outage in a month, its entire sampled duration is
    credited to that month, including portions before and after the outage
    and outside the month. Crossing a month boundary alone does not qualify
    it for the other month: it must also overlap an Earth outage there.
    For example, a 36-hour low-Sun run crossing January and February contributes
    36 hours to each band if it overlaps an outage in each month and both
    months have some samples where Earth is available. The duration is not
    split into January and February portions. Multiple qualifying runs are
    reduced by taking their maximum, not their sum.

    Duration is ``number of consecutive low-Sun samples * sampling step`` in
    hours, including the last sample. A run already active at the first sample
    or still active at the last sample is counted only over the sampled data;
    its true start or end outside the evaluation domain is unknown. In
    particular, this sample-count convention can yield one step more than the
    elapsed time between the first and last timestamps in a run.

    A month's result is NaN when Earth is below the threshold at no samples
    or at every sample, regardless of sunlight. These are undefined results,
    not zero-duration havens. Zero is a valid result when that month contains
    both outage and available-Earth samples but no low-Sun run overlaps an
    outage. With partial-month input, these classifications describe the
    sampled portion only. Patches with missing or unreadable horizons are
    instead written as invalid patches filled with ``nodata``.
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
    time_values = times
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
    if sun.times_utc != earth.times_utc:
        raise ProductTimeError(
            "Sun and Earth vectors must use the same UTC timestamps.",
            code="safe_haven_times_mismatch",
        )
    if len(sun.times_utc) < 2:
        raise ProductTimeError(
            "Safe-haven generation requires at least two time samples.",
            code="safe_haven_times_insufficient",
        )
    steps = np.asarray(
        [
            (right - left).total_seconds() / 3600.0
            for left, right in zip(sun.times_utc, sun.times_utc[1:])
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
    from ._numba_horizon.safe_haven_pipeline import (
        SafeHavenPipelineCancelled,
        run_safe_haven_product,
    )

    adapter = _ProgressAdapter(
        "safe_havens",
        output,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
    )
    try:
        return run_safe_haven_product(
            dem=dem,
            georef=georef,
            horizon_store=HorizonTileStore(horizons),
            output_path=output,
            times_utc=sun.times_utc,
            sun_vectors_m=sun.vectors_m,
            earth_vectors_m=earth.vectors_m,
            time_step_hours=float(steps[0]),
            earth_threshold_deg=earth_elevation_threshold_deg,
            sunlight_threshold=sunlight_fraction_threshold,
            observer_elevation_m=observer_height_m,
            nodata=nodata,
            output_transform=output_transform,
            output_dtype=output_dtype,
            output_transform_id=output_transform_id,
            compress=compress,
            overwrite=overwrite,
            start_fresh=start_fresh,
            backend=backend,
            cancellation_requested=cancellation_requested,
            progress_callback=adapter,
        )
    except SafeHavenPipelineCancelled as exc:
        raise OperationCancelledError(
            "Safe-haven generation was cancelled.",
            code="safe_haven_cancelled",
            details={"path": str(output)},
        ) from exc
    except CudaBackendError as exc:
        raise CudaError(
            "The CUDA safe-haven backend is unavailable.",
            code="cuda_safe_haven_unavailable",
            details={"error": str(exc)},
        ) from exc
    except ProductStoreError as exc:
        raise ProductStorageError(
            "Safe-haven storage failed.",
            code="safe_haven_storage_failed",
            details={"path": str(output), "error": str(exc)},
        ) from exc
    except OSError as exc:
        raise ProductStorageError(
            "Safe-haven file access failed.",
            code="safe_haven_file_access_failed",
            details={"path": str(output), "error": str(exc)},
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
    overwrite / start_fresh:
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
        progress_callback=progress_callback,
        progress_event_callback=progress_event_callback,
        cancellation_requested=cancellation_requested,
    )
