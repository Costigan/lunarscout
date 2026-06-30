from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import rasterio

from .alignment import same_grid
from .errors import NativeInputError, NativeProductError
from .geotiff import read_geotiff
from .temporal_store import _layer_metadata


NativeProductProgressCallback: TypeAlias = Callable[["NativeProductProgress"], None]
CancellationCheck: TypeAlias = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class NativeProductProgress:
    stage: str
    percent: float
    message: str


def _report(
    callback: NativeProductProgressCallback | None,
    *,
    stage: str,
    percent: float,
    message: str,
) -> None:
    if callback is not None:
        callback(
            NativeProductProgress(
                stage=str(stage),
                percent=float(percent),
                message=str(message),
            )
        )


def _raise_if_cancelled(check: CancellationCheck | None) -> None:
    if check is not None and check():
        raise NativeProductError(
            "Native PSR generation was cancelled.",
            code="native_psr_cancelled",
        )


def _dotnet_string_list(values: tuple[Path, ...]) -> Any:
    try:
        from System import String  # type: ignore
        from System.Collections.Generic import List as DotNetList  # type: ignore

        result = DotNetList[String]()
        for value in values:
            result.Add(String(str(value)))
        return result
    except ModuleNotFoundError:
        return [str(value) for value in values]


def _native_progress_fields(progress: Any) -> tuple[str, float, str]:
    stage = str(getattr(progress, "Stage", "native_execution"))
    message = str(
        getattr(progress, "Message", "Generating permanent shadow raster.")
    )
    try:
        percent = float(getattr(progress, "Percent", 10.0))
    except (TypeError, ValueError, OverflowError):
        percent = 10.0
    return stage, min(95.0, max(1.0, percent)), message


def _read_validity_mask(path: Path) -> np.ndarray:
    try:
        with rasterio.open(path) as dataset:
            if dataset.driver != "GTiff" or int(dataset.count) < 1:
                raise RuntimeError("PSR GeoTIFF band is unavailable")
            return np.asarray(dataset.read_masks(1), dtype=np.uint8)
    except NativeProductError:
        raise
    except Exception as exc:
        raise NativeProductError(
            "Unable to read the native PSR validity mask.",
            code="native_psr_validity_unreadable",
            details={"error": str(exc)},
        ) from exc


def generate_psr_product(
    *,
    scenario_root: Path,
    dem_path: Path,
    horizons_path: Path,
    output_path: Path,
    surrounding_dem_paths: tuple[Path, ...] = (),
    overwrite: bool = False,
    progress_callback: NativeProductProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
    _bridge: Any | None = None,
) -> Path:
    """Generate one native byte PSR mask and publish it atomically."""

    if progress_callback is not None and not callable(progress_callback):
        raise NativeInputError(
            "progress_callback must be callable or None.",
            code="native_psr_invalid_callback",
        )
    if cancellation_requested is not None and not callable(cancellation_requested):
        raise NativeInputError(
            "cancellation_requested must be callable or None.",
            code="native_psr_invalid_callback",
        )
    if not scenario_root.is_dir() or not dem_path.is_file() or not horizons_path.is_dir():
        raise NativeInputError(
            "Scenario root, DEM, and horizons must exist before PSR generation.",
            code="native_psr_input_missing",
            details={
                "scenario_root": str(scenario_root),
                "dem_path": str(dem_path),
                "horizons_path": str(horizons_path),
            },
        )
    missing_surrounding = [str(path) for path in surrounding_dem_paths if not path.is_file()]
    if missing_surrounding:
        raise NativeInputError(
            "A surrounding DEM does not exist.",
            code="native_psr_input_missing",
            details={"paths": missing_surrounding},
        )
    if output_path.suffix.lower() not in {".tif", ".tiff"}:
        raise NativeInputError(
            "Native PSR output must use a .tif or .tiff extension.",
            code="native_psr_output_extension_invalid",
            details={"path": str(output_path)},
        )
    if output_path.is_symlink():
        raise NativeInputError(
            "Native PSR output cannot be a symbolic link.",
            code="native_psr_output_symlink",
            details={"path": str(output_path)},
        )
    if output_path.exists() and output_path.is_dir():
        raise NativeInputError(
            "Native PSR output must be a file path.",
            code="native_psr_output_is_directory",
            details={"path": str(output_path)},
        )
    if output_path.exists() and not overwrite:
        raise NativeInputError(
            "Native PSR output already exists.",
            code="native_psr_output_exists",
            details={"path": str(output_path)},
        )

    _raise_if_cancelled(cancellation_requested)
    _report(
        progress_callback,
        stage="preflight",
        percent=1.0,
        message="Native PSR inputs and output path validated.",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.staging-",
        suffix=".tif",
        dir=output_path.parent,
    )
    os.close(descriptor)
    staging_path = Path(temporary_name)
    staging_path.unlink()

    try:
        if _bridge is None:
            from .native import _bootstrap_module, _create_moonlib_bridge

            bridge = _create_moonlib_bridge(force=True, verify=False)
            moonlib = _bootstrap_module().import_moonlib(
                force_bootstrap=True,
                verify_bridge_smoke=False,
            )
        else:
            bridge = _bridge
            moonlib = None

        progress_lock = threading.Lock()
        last_native_percent = 1.0

        def emit_native_progress(progress: Any) -> None:
            nonlocal last_native_percent
            stage, percent, message = _native_progress_fields(progress)
            with progress_lock:
                percent = max(last_native_percent, percent)
                last_native_percent = percent
                _report(
                    progress_callback,
                    stage=stage,
                    percent=percent,
                    message=message,
                )

        def is_cancelled() -> bool:
            return bool(
                cancellation_requested is not None and cancellation_requested()
            )

        try:
            progress_argument = (
                moonlib.PsrProgressCallback(emit_native_progress)
                if moonlib is not None
                else emit_native_progress
            )
            cancellation_argument = (
                moonlib.PsrCancellationCallback(is_cancelled)
                if moonlib is not None
                else is_cancelled
            )
            bridge.GeneratePermanentShadowMap(
                str(scenario_root),
                str(dem_path),
                _dotnet_string_list(surrounding_dem_paths),
                str(horizons_path),
                str(staging_path),
                progress_argument,
                cancellation_argument,
            )
        except Exception as exc:
            if is_cancelled():
                raise NativeProductError(
                    "Native PSR generation was cancelled.",
                    code="native_psr_cancelled",
                ) from exc
            raise NativeProductError(
                "Native PSR generation failed.",
                code="native_psr_generation_failed",
                details={"error": str(exc)},
            ) from exc

        _raise_if_cancelled(cancellation_requested)
        _report(
            progress_callback,
            stage="validate_output",
            percent=96.0,
            message="Validating native PSR GeoTIFF.",
        )
        values, output_georef = read_geotiff(staging_path)
        _dem_dtype, dem_georef = _layer_metadata(dem_path)
        if output_georef is None or not same_grid(output_georef, dem_georef):
            raise NativeProductError(
                "Native PSR output grid does not match the scenario DEM.",
                code="native_psr_grid_mismatch",
            )
        if output_georef.nodata is not None:
            raise NativeProductError(
                "Native PSR output must use a validity mask instead of a nodata sentinel.",
                code="native_psr_nodata_invalid",
                details={"nodata": output_georef.nodata},
            )
        validity = _read_validity_mask(staging_path)
        if validity.shape != values.shape or not np.all(
            np.isin(validity, np.asarray([0, 255], dtype=np.uint8))
        ):
            raise NativeProductError(
                "Native PSR validity must contain only 0 and 255 on the output grid.",
                code="native_psr_validity_invalid",
                details={
                    "data_shape": list(values.shape),
                    "validity_shape": list(validity.shape),
                    "unique_values": np.unique(validity)[:16].tolist(),
                },
            )
        unique_values = np.unique(values)
        if values.dtype != np.dtype(np.uint8) or not np.all(
            np.isin(unique_values, np.asarray([0, 255], dtype=np.uint8))
        ):
            raise NativeProductError(
                "Native PSR output must be a uint8 mask containing only 0 and 255.",
                code="native_psr_values_invalid",
                details={
                    "dtype": str(values.dtype),
                    "unique_values": unique_values[:16].tolist(),
                },
            )
        if np.any(values[validity == 0] != 0):
            raise NativeProductError(
                "Invalid PSR pixels must have a deterministic zero payload.",
                code="native_psr_invalid_payload",
            )
        _raise_if_cancelled(cancellation_requested)
        if output_path.exists() and not overwrite:
            raise NativeInputError(
                "Native PSR output was created concurrently.",
                code="native_psr_output_exists",
                details={"path": str(output_path)},
            )
        os.replace(staging_path, output_path)
        _report(
            progress_callback,
            stage="complete",
            percent=100.0,
            message="Native PSR GeoTIFF complete.",
        )
        return output_path
    finally:
        staging_path.unlink(missing_ok=True)
