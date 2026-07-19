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
from .geotiff import read_geotiff
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


def _load_dem(path: str | Path):
    from ._numba_horizon.geometry import DemGrid, ProjectionParameters

    dem_path = Path(path).expanduser().resolve()
    if not dem_path.is_file():
        raise InputError(
            "The DEM path does not identify a file.",
            code="product_dem_not_found",
            details={"path": str(dem_path)},
        )
    values, georef = read_geotiff(dem_path)
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
        np.ascontiguousarray(values, dtype=np.float32),
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
            "The product output already exists.",
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
    staging state and never publishes an incomplete file.
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
    dataset mask for transparency.
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
    resumable staging state.
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
    default.
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
    """Generate longest low-Sun durations for center-view Earth outages.

    Parameters
    ----------
    times:
        The UTC time domain as a :class:`TimeRange`.  Must be strictly
        increasing and uniformly spaced.  When explicit vectors are not
        supplied, Lunarscout generates geometric Moon-ME Sun and Earth
        vectors from this time range.
    sun_vectors_m / earth_vectors_m:
        Optional explicit Moon-ME vectors in meters, shape ``(time, 3)``.
        Supplied vectors take precedence and avoid SPICE import.
    earth_elevation_threshold_deg:
        Earth-center elevation threshold in degrees.  An Earth outage is a
        maximal half-open interval during which the Earth-center elevation
        relative to the terrain horizon is strictly *below* this threshold.
        Default 2.0 degrees.
    sunlight_fraction_threshold:
        Unitless sunlight-fraction threshold.  The reducer finds the longest
        complete contiguous interval during which the sunlight fraction is
        strictly *below* this threshold.  Default 0.2.
    backend:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.  See :func:`generate_lightmap`.
    observer_height_m:
        Observer height above the DEM surface, in meters.
    nodata:
        Value stored in invalid pixels.  Defaults to ``NaN``.  A dataset
        validity mask is always written.
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
        The completed output path.  Each band represents one Earth outage.

    Notes
    -----
    Each ``float32`` output band stores the longest complete contiguous
    low-Sun interval that overlaps an Earth outage, in hours.  The low-Sun
    interval may begin before the Earth outage or end after it.  The band is
    timestamped with the first occurrence of the minimum Earth elevation
    within that outage.
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
    still active at the evaluation stop is right-censored.
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
