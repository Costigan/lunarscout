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

    Values are ``trunc(255 * visible_fraction)``. Invalid pixels carry
    ``invalid_value`` and are distinguished by the dataset validity mask.
    Compatible staged work resumes by default; ``start_fresh=True`` discards
    it. ``backend="cpu"`` never probes CUDA, explicit ``"cuda"`` never falls
    back, and ``"auto"`` falls back only when CUDA cannot be initialized.
    Tiles are compressed by default; pass ``compress=False`` to disable
    compression without disabling tiling.
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

    Value 255 means that the upper solar limb never clears the interpolated
    terrain horizon for the supplied samples; value 0 means that it clears at
    least once. Both are valid science values. Invalid pixels are represented
    by the dataset mask and carry ``invalid_value`` deterministically. Tiles
    are compressed unless ``compress=False``.
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

    Each UTC sample becomes one tiled ``float32`` BigTIFF band in degrees. Invalid
    pixels carry ``nodata`` (NaN by default) and are distinguished by the
    dataset mask.
    The returned :class:`Path` identifies the completed output. Tiles are
    compressed unless ``compress=False``.
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

    Each UTC sample becomes one tiled ``float32`` BigTIFF band in degrees. Invalid
    pixels carry ``nodata`` (NaN by default) and are distinguished by the
    dataset mask.
    The returned :class:`Path` identifies the completed output. Tiles are
    compressed unless ``compress=False``.
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

    Earth outages are maximal half-open intervals strictly below
    ``earth_elevation_threshold_deg``. Each float32 output band stores the
    longest complete contiguous sunlight interval strictly below
    ``sunlight_fraction_threshold`` that overlaps that outage, in hours. The
    low-Sun interval may begin before the outage or end after it. Input samples
    must be uniformly spaced. Invalid pixels carry ``nodata`` (NaN by default) and are
    distinguished by the dataset mask. The returned :class:`Path` identifies
    the completed tiled output. Tiles are compressed unless
    ``compress=False``.
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

    ``sunlight_fraction_threshold`` is a unitless inclusive lower bound.
    Evaluation and candidate-start intervals are half-open. Durations use
    actual UTC sample spacing and are stored as ``float32`` hours or days as
    selected by ``output_unit``. Invalid pixels carry ``nodata`` and are
    distinguished by the dataset mask. The returned :class:`Path` identifies
    the completed output.
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

    ``sun_elevation_threshold_deg`` is an inclusive lower bound in degrees.
    Evaluation and candidate-start intervals are half-open. Durations use
    actual UTC sample spacing and are stored as ``float32`` hours or days as
    selected by ``output_unit``. Invalid pixels carry ``nodata`` and are
    distinguished by the dataset mask. The returned :class:`Path` identifies
    the completed output.
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

    The sunlight fraction is unitless and Earth terrain-relative elevation is
    in degrees. Both thresholds are inclusive lower bounds. Evaluation and
    candidate-start intervals are half-open. Durations use actual UTC sample
    spacing and are stored as ``float32`` hours or days as selected by
    ``output_unit``. Invalid pixels carry ``nodata`` and are
    distinguished by the dataset mask. The returned :class:`Path` identifies
    the completed output.
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
    degrees. Evaluation and candidate-start intervals are half-open. Durations
    use actual UTC sample spacing and are stored as ``float32`` hours or days
    as selected by ``output_unit``. Invalid pixels carry ``nodata`` and
    are distinguished by the dataset mask. The returned :class:`Path`
    identifies the completed output.
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
