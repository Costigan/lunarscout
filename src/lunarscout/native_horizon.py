from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from .errors import NativeInputError, NativeProductError


NativeHorizonProgressCallback: TypeAlias = Callable[["NativeHorizonProgress"], None]
CancellationCheck: TypeAlias = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class NativeHorizonProgress:
    processed_patches: int
    total_patches: int
    percent: float
    stage: str
    message: str
    file_name: str | None = None


def _dotnet_list(cls: Any, values: Sequence[Any]) -> Any:
    try:
        from System.Collections.Generic import List as DotNetList  # type: ignore

        result = DotNetList[cls]()
        for value in values:
            result.Add(value)
        return result
    except Exception:
        return list(values)


def _native_progress_fields(progress: Any) -> NativeHorizonProgress:
    return NativeHorizonProgress(
        processed_patches=int(getattr(progress, "ProcessedPatches", 0)),
        total_patches=int(getattr(progress, "TotalPatches", 0)),
        percent=float(getattr(progress, "Percent", 0.0)),
        stage=str(getattr(progress, "Stage", "native_execution")),
        message=str(getattr(progress, "Message", "Generating horizon patches.")),
        file_name=(
            None
            if getattr(progress, "FileName", None) is None
            else str(getattr(progress, "FileName"))
        ),
    )


def _raise_if_cancelled(check: CancellationCheck | None) -> None:
    if check is not None and check():
        raise NativeProductError(
            "Native horizon generation was cancelled.",
            code="native_horizons_cancelled",
        )


def _wait_for_task(task: Any) -> None:
    awaiter = task.GetAwaiter()
    awaiter.GetResult()


def _create_horizon_components() -> tuple[Any, Any]:
    from .native import _bootstrap_module

    try:
        bootstrap = _bootstrap_module()
        moonlib = bootstrap.import_moonlib(
            force_bootstrap=True,
            verify_bridge_smoke=False,
        )
        return moonlib, moonlib.horizon
    except Exception as exc:
        raise NativeProductError(
            "Unable to create native horizon generator components.",
            code="native_horizons_creation_failed",
            details={"error": str(exc)},
        ) from exc


def GenerateHorizons(
    output_dir: str | Path,
    dem_paths: Sequence[str | Path],
    observer_elevation: float = 0.0,
    *,
    skip_existing: bool = True,
    compress_horizons: bool = False,
    disable_hierarchy: bool = False,
    progress_callback: NativeHorizonProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
    _moonlib: Any | None = None,
    _horizon_module: Any | None = None,
) -> Path:
    """Generate native horizon tiles for all 128x128 patches in the primary DEM.

    ``dem_paths`` must be ordered primary DEM first, followed by any surrounding
    DEMs. The implementation calls the C# ``QuadTreeHorizonGenerator`` directly:
    load ``ElevationMap`` objects, build the full patch list from the primary
    DEM, optionally skip existing horizon tiles, then run
    ``GenerateHorizonsForPatches``.
    """

    if progress_callback is not None and not callable(progress_callback):
        raise NativeInputError(
            "progress_callback must be callable or None.",
            code="native_horizons_invalid_callback",
        )
    if cancellation_requested is not None and not callable(cancellation_requested):
        raise NativeInputError(
            "cancellation_requested must be callable or None.",
            code="native_horizons_invalid_callback",
        )

    output_path = Path(output_dir).expanduser().resolve()
    dem_path_values = tuple(Path(path).expanduser().resolve() for path in dem_paths)
    if not dem_path_values:
        raise NativeInputError(
            "At least one DEM path is required.",
            code="native_horizons_input_missing",
        )
    missing = [str(path) for path in dem_path_values if not path.is_file()]
    if missing:
        raise NativeInputError(
            "A DEM path does not exist.",
            code="native_horizons_input_missing",
            details={"paths": missing},
        )
    if output_path.exists() and not output_path.is_dir():
        raise NativeInputError(
            "Native horizons output must be a directory path.",
            code="native_horizons_output_is_file",
            details={"path": str(output_path)},
        )

    _raise_if_cancelled(cancellation_requested)
    output_path.mkdir(parents=True, exist_ok=True)

    if _moonlib is None or _horizon_module is None:
        moonlib, horizon = _create_horizon_components()
    else:
        moonlib, horizon = _moonlib, _horizon_module

    progress_argument = None
    if progress_callback is not None:

        def emit_native_progress(progress: Any) -> None:
            progress_callback(_native_progress_fields(progress))

        progress_argument = (
            moonlib.HorizonProgressCallback(emit_native_progress)
            if hasattr(moonlib, "HorizonProgressCallback")
            else emit_native_progress
        )

    cancellation_argument = None
    if cancellation_requested is not None:

        def is_cancelled() -> bool:
            return bool(cancellation_requested())

        cancellation_argument = (
            moonlib.HorizonCancellationCallback(is_cancelled)
            if hasattr(moonlib, "HorizonCancellationCallback")
            else is_cancelled
        )

    generator = None
    try:
        bridge_type = getattr(moonlib, "MoonlibBridge", None)
        if bridge_type is not None and hasattr(bridge_type, "EnsureGdalInitialized"):
            bridge_type.EnsureGdalInitialized()

        dems = [horizon.ElevationMap(str(path)) for path in dem_path_values]
        dem_list = _dotnet_list(horizon.ElevationMap, dems)
        patches = horizon.QuadTreeHorizonGenerator.GeneratePatchList(dems[0])
        if skip_existing:
            patches = horizon.QuadTreeHorizonGenerator.RemoveCompletedPatches(
                patches,
                str(output_path),
                float(observer_elevation),
            )
        patch_count = getattr(patches, "Count", None)
        if patch_count is None:
            patch_count = len(patches)
        if int(patch_count) < 1:
            return output_path

        generator = horizon.QuadTreeHorizonGenerator(bool(disable_hierarchy))
        task = generator.GenerateHorizonsForPatches(
            str(output_path),
            dem_list,
            patches,
            float(observer_elevation),
            bool(compress_horizons),
            progress_argument,
            cancellation_argument,
        )
        _wait_for_task(task)
        _raise_if_cancelled(cancellation_requested)
        return output_path
    except NativeProductError:
        raise
    except Exception as exc:
        if cancellation_requested is not None and cancellation_requested():
            raise NativeProductError(
                "Native horizon generation was cancelled.",
                code="native_horizons_cancelled",
            ) from exc
        raise NativeProductError(
            "Native horizon generation failed.",
            code="native_horizons_generation_failed",
            details={"error": str(exc)},
        ) from exc
    finally:
        if generator is not None:
            generator.Dispose()
