from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .errors import NativeAllocationError, NativeInputError, NativeTemporalError
from .georeference import GeoReference
from .temporal import TemporalCube, TimeRange
from .temporal_store import TemporalGeoTiffSeries, TemporalGeoTiffSeriesWriter


TemporalStorage: TypeAlias = Literal["memory", "geotiff_series"]
NativeProgressCallback: TypeAlias = Callable[["NativeTemporalProgress"], None]
CancellationCheck: TypeAlias = Callable[[], bool]
_DEFAULT_MAX_IN_MEMORY_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NativeTemporalProgress:
    stage: str
    percent: float
    message: str
    completed_spatial_pixels: int
    total_spatial_pixels: int


@dataclass(frozen=True, slots=True)
class TemporalAllocationEstimate:
    signal: str
    storage: TemporalStorage
    dtype: np.dtype[Any]
    shape: tuple[int, int, int]
    estimated_bytes: int
    limit_bytes: int | None


@dataclass(frozen=True, slots=True)
class NativeLightmapBufferPatch:
    """Borrowed lightmap patch buffer returned by ``stream_lightmap_buffers``."""

    buffer_id: int
    tile_id: int
    patch_row: int
    patch_col: int
    width: int
    height: int
    time_count: int
    values: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _SignalSpec:
    native_name: str
    dtype: np.dtype[Any]
    units: str
    scale: float


_SIGNALS = {
    "sun_fraction": _SignalSpec(
        native_name="sun_fraction_u8",
        dtype=np.dtype(np.float32),
        units="fraction",
        scale=1.0 / 255.0,
    ),
    "sun_over_horizon_deg": _SignalSpec(
        native_name="sun_center_margin_deg_f32",
        dtype=np.dtype(np.float32),
        units="degrees",
        scale=1.0,
    ),
    "earth_over_horizon_deg": _SignalSpec(
        native_name="earth_center_margin_deg_f32",
        dtype=np.dtype(np.float32),
        units="degrees",
        scale=1.0,
    ),
}


def _signal_spec(signal: str) -> _SignalSpec:
    try:
        return _SIGNALS[str(signal)]
    except KeyError as exc:
        raise NativeInputError(
            "Unsupported native temporal signal.",
            code="native_temporal_signal_unsupported",
            details={"signal": signal, "supported": sorted(_SIGNALS)},
        ) from exc


def estimate_temporal_allocation(
    *,
    signal: str,
    times: TimeRange,
    georef: GeoReference,
    storage: TemporalStorage,
    max_in_memory_bytes: int = _DEFAULT_MAX_IN_MEMORY_BYTES,
) -> TemporalAllocationEstimate:
    """Estimate the full result or scratch allocation before native startup."""

    if not isinstance(times, TimeRange):
        raise NativeInputError(
            "times must be a TimeRange.", code="native_temporal_invalid_times"
        )
    if not isinstance(georef, GeoReference):
        raise NativeInputError(
            "georef must be a GeoReference.", code="native_temporal_invalid_georef"
        )
    if storage not in {"memory", "geotiff_series"}:
        raise NativeInputError(
            "storage must be 'memory' or 'geotiff_series'.",
            code="native_temporal_invalid_storage",
            details={"storage": storage},
        )
    spec = _signal_spec(signal)
    try:
        memory_limit = int(max_in_memory_bytes)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativeInputError(
            "max_in_memory_bytes must be a positive integer.",
            code="native_temporal_invalid_memory_limit",
        ) from exc
    if memory_limit < 1:
        raise NativeInputError(
            "max_in_memory_bytes must be a positive integer.",
            code="native_temporal_invalid_memory_limit",
        )
    shape = (times.time_count, georef.height, georef.width)
    estimated_bytes = int(np.prod(shape, dtype=object)) * spec.dtype.itemsize
    return TemporalAllocationEstimate(
        signal=signal,
        storage=storage,
        dtype=spec.dtype,
        shape=shape,
        estimated_bytes=estimated_bytes,
        limit_bytes=memory_limit if storage == "memory" else None,
    )


def _streaming_components() -> SimpleNamespace:
    raise NativeTemporalError(
        "The legacy Lunar Analyst temporal streaming adapter is not part of standalone Lunarscout. "
        "Use stream_lightmap_buffers or provide explicit test components until the standalone native "
        "temporal reducer API is implemented.",
        code="native_temporal_adapter_removed",
    )


def _create_streaming_client(client_type: type[Any]) -> Any:
    from .native import _bootstrap_module

    try:
        bootstrap = _bootstrap_module()
        moonlib = bootstrap.import_moonlib(
            force_bootstrap=True, verify_bridge_smoke=False
        )
        bridge = moonlib.MoonlibBridge()
        return client_type(bridge=bridge, moonlib_module=moonlib)
    except Exception as exc:
        raise NativeTemporalError(
            "Unable to create the MoonlibBridge temporal streaming client.",
            code="native_temporal_bridge_creation_failed",
            details={"error": str(exc)},
        ) from exc


def _timestamp_text(value: np.datetime64) -> str:
    return f"{np.datetime_as_string(value.astype('datetime64[us]'), unit='us')}Z"


def _dotnet_timestamps(times: TimeRange) -> Any:
    try:
        from System import Array, DateTime
    except Exception as exc:
        raise NativeTemporalError(
            "Python.NET System types are unavailable.",
            code="native_temporal_pythonnet_unavailable",
            details={"error": str(exc)},
        ) from exc
    return Array[DateTime]([DateTime.Parse(_timestamp_text(value)[:-1]) for value in times.values])


def _dotnet_array(cls: Any, values: list[Any]) -> Any:
    from System import Array

    return Array[cls](values)


def _create_fill_lightmap_buffers() -> tuple[Any, Any]:
    from .native import _bootstrap_module

    try:
        bootstrap = _bootstrap_module()
        moonlib = bootstrap.import_moonlib(
            force_bootstrap=True, verify_bridge_smoke=False
        )
        pipeline = moonlib.pipeline
        return pipeline, pipeline.FillLightmapBuffers()
    except Exception as exc:
        raise NativeTemporalError(
            "Unable to create FillLightmapBuffers.",
            code="native_temporal_fill_buffers_creation_failed",
            details={"error": str(exc)},
        ) from exc


def _enum_name(value: Any) -> str:
    return str(value).split(".")[-1]


def _report(
    callback: NativeProgressCallback | None,
    *,
    stage: str,
    percent: float,
    message: str,
    completed: int,
    total: int,
) -> None:
    if callback is not None:
        callback(
            NativeTemporalProgress(
                stage=stage,
                percent=float(percent),
                message=message,
                completed_spatial_pixels=int(completed),
                total_spatial_pixels=int(total),
            )
        )


def stream_lightmap_buffers(
    *,
    dem_path: str | Path,
    horizons_path: str | Path,
    times: TimeRange,
    buffer_count: int = 8,
    patch_width: int = 128,
    patch_height: int = 128,
    max_read_parallelism: int = 12,
    max_compute_parallelism: int = 20,
    queue_capacity: int = 40,
    poll_timeout_ms: int = 250,
    use_spice_sun_vectors: bool = True,
    progress_callback: NativeProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
    _pipeline: Any | None = None,
    _filler: Any | None = None,
    _array_factory: Callable[[Any, list[Any]], Any] | None = None,
) -> Iterator[NativeLightmapBufferPatch]:
    """Stream C#-filled lightmap patches through a reusable NumPy buffer pool.

    Each yielded ``values`` array is borrowed from the internal pool and has
    shape ``(patch_height, patch_width, time_count)`` with dtype ``uint8``.
    The caller must finish reading it before advancing the iterator again. Copy
    the array if the data must outlive the current iteration.
    """

    if not isinstance(times, TimeRange):
        raise NativeInputError(
            "times must be a TimeRange.", code="native_temporal_invalid_times"
        )
    try:
        pool_size = int(buffer_count)
        timeout = int(poll_timeout_ms)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativeInputError(
            "buffer_count and poll_timeout_ms must be integers.",
            code="native_temporal_fill_buffers_invalid_options",
        ) from exc
    if pool_size < 1:
        raise NativeInputError(
            "buffer_count must be at least one.",
            code="native_temporal_fill_buffers_invalid_options",
        )
    if timeout < 1:
        raise NativeInputError(
            "poll_timeout_ms must be at least one.",
            code="native_temporal_fill_buffers_invalid_options",
        )

    dem = Path(dem_path).expanduser().resolve()
    horizons = Path(horizons_path).expanduser().resolve()
    if not dem.is_file() or not horizons.is_dir():
        raise NativeInputError(
            "DEM and horizons directory must exist before native buffer filling.",
            code="native_temporal_input_missing",
            details={"dem_path": str(dem), "horizons_path": str(horizons)},
        )

    if _pipeline is None or _filler is None:
        pipeline, filler = _create_fill_lightmap_buffers()
    else:
        pipeline, filler = _pipeline, _filler
    array_factory = _array_factory or _dotnet_array

    request = pipeline.FillLightmapBuffersRequest(
        str(dem),
        str(horizons),
        _dotnet_timestamps(times) if _pipeline is None else times,
        int(patch_width),
        int(patch_height),
        int(max_read_parallelism),
        int(max_compute_parallelism),
        int(queue_capacity),
        bool(use_spice_sun_vectors),
    )

    buffers = {
        buffer_id: np.empty(
            (int(patch_height), int(patch_width), times.time_count), dtype=np.uint8
        )
        for buffer_id in range(pool_size)
    }
    available_ids = list(buffers)
    completed_patches = 0
    horizon_count = sum(1 for _ in horizons.glob("**/horizon_*.*"))
    total_pixels = max(1, horizon_count * int(patch_width) * int(patch_height))

    try:
        filler.Start(request)
        while True:
            _raise_if_cancelled(cancellation_requested)
            offered = [
                pipeline.FillLightmapAvailableBuffer(
                    int(buffer_id),
                    int(buffers[buffer_id].ctypes.data),
                    int(buffers[buffer_id].nbytes),
                )
                for buffer_id in available_ids
            ]
            available_ids = []
            result = filler.Poll(
                array_factory(pipeline.FillLightmapAvailableBuffer, offered),
                timeout,
            )

            filled_buffers = list(result.FilledBuffers)
            for filled in filled_buffers:
                buffer_id = int(filled.BufferId)
                if buffer_id not in buffers:
                    raise NativeTemporalError(
                        "Native buffer fill returned an unknown buffer id.",
                        code="native_temporal_fill_buffers_unknown_buffer",
                        details={"buffer_id": buffer_id},
                    )
                state = _enum_name(filled.State)
                if state != "Filled":
                    raise NativeTemporalError(
                        str(filled.Message or "Native buffer fill returned an error."),
                        code="native_temporal_fill_buffers_tile_failed",
                        details={
                            "buffer_id": buffer_id,
                            "tile_id": int(filled.TileId),
                            "patch_row": int(filled.PatchRow),
                            "patch_col": int(filled.PatchCol),
                        },
                    )
                completed_patches += 1
                _report(
                    progress_callback,
                    stage="native_fill_buffers",
                    percent=min(99.0, 100.0 * completed_patches / max(1, horizon_count)),
                    message="Receiving native lightmap patch buffers.",
                    completed=completed_patches * int(patch_width) * int(patch_height),
                    total=total_pixels,
                )
                yield NativeLightmapBufferPatch(
                    buffer_id=buffer_id,
                    tile_id=int(filled.TileId),
                    patch_row=int(filled.PatchRow),
                    patch_col=int(filled.PatchCol),
                    width=int(filled.Width),
                    height=int(filled.Height),
                    time_count=int(filled.TimeCount),
                    values=buffers[buffer_id],
                )
                available_ids.append(buffer_id)

            run_state = _enum_name(result.State)
            if run_state == "Completed":
                _report(
                    progress_callback,
                    stage="complete",
                    percent=100.0,
                    message="Native lightmap buffer stream complete.",
                    completed=total_pixels,
                    total=total_pixels,
                )
                break
            if run_state in {"Cancelled", "Failed"}:
                raise NativeTemporalError(
                    str(result.Message or f"Native buffer fill ended with state {run_state}."),
                    code="native_temporal_fill_buffers_failed",
                    details={"state": run_state},
                )
    except NativeTemporalError:
        cancel = getattr(filler, "Cancel", None)
        if callable(cancel):
            cancel()
        raise
    finally:
        dispose = getattr(filler, "Dispose", None)
        if callable(dispose):
            dispose()


def _raise_if_cancelled(check: CancellationCheck | None) -> None:
    if check is not None and check():
        raise NativeTemporalError(
            "Native temporal generation was cancelled.",
            code="native_temporal_cancelled",
        )


def _request(
    components: SimpleNamespace,
    *,
    signal_spec: _SignalSpec,
    scenario_root: Path,
    dem_path: Path,
    surrounding_dem_paths: tuple[Path, ...],
    horizons_path: Path,
    times: TimeRange,
    observer_elevation_meters: float,
    patch_width: int,
    patch_height: int,
    chunk_time_count: int,
) -> Any:
    return components.Request(
        scenario_root_dir=scenario_root,
        dem_path=dem_path,
        surrounding_dem_paths=list(surrounding_dem_paths),
        horizon_dir=horizons_path,
        start_utc=_timestamp_text(times.values[0]),
        stop_utc=_timestamp_text(times.values[-1]),
        time_step_hours=times.step_hours,
        observer_elevation_meters=float(observer_elevation_meters),
        patch_width=int(patch_width),
        patch_height=int(patch_height),
        max_read_parallelism=4,
        max_compute_parallelism=24,
        ready_queue_capacity=64,
        use_spice_sun_vectors=True,
        mode="signal_stream",
        signals=[components.SignalSpec(signal=signal_spec.native_name)],
        chunk_time_count=int(chunk_time_count),
        reducers=None,
        use_spice_earth_vectors=True,
    )


def _fill_from_tiles(
    target: NDArray[Any],
    *,
    stream: Iterator[tuple[Any, NDArray[Any]]],
    spec: _SignalSpec,
    cancellation_requested: CancellationCheck | None,
    progress_callback: NativeProgressCallback | None,
) -> None:
    time_count, height, width = target.shape
    coverage = np.zeros((height, width), dtype=bool)
    tile_states: dict[tuple[int, int], int] = {}
    completed_pixels = 0
    total_pixels = height * width
    for tile, chunk in stream:
        _raise_if_cancelled(cancellation_requested)
        if int(tile.rank) != 4 or np.ndim(chunk) != 4:
            raise NativeTemporalError(
                "Native temporal stream returned an unexpected tensor rank.",
                code="native_temporal_stream_shape_invalid",
            )
        yoff, xoff = int(tile.patch_row), int(tile.patch_col)
        tile_height, tile_width = int(tile.height), int(tile.width)
        offset, count = int(tile.time_offset), int(tile.time_count)
        key = (yoff, xoff)
        expected_offset = tile_states.get(key, 0)
        if offset != expected_offset or offset < 0 or offset + count > time_count:
            raise NativeTemporalError(
                "Native temporal chunks arrived out of order or outside the time axis.",
                code="native_temporal_stream_order_invalid",
                details={"tile": key, "expected_offset": expected_offset, "offset": offset},
            )
        if (
            yoff < 0
            or xoff < 0
            or tile_height < 1
            or tile_width < 1
            or yoff + tile_height > height
            or xoff + tile_width > width
        ):
            raise NativeTemporalError(
                "Native temporal tile falls outside the target grid.",
                code="native_temporal_stream_window_invalid",
                details={"tile": key, "height": tile_height, "width": tile_width},
            )
        view = np.asarray(chunk[:count, 0, :tile_height, :tile_width])
        if view.shape != (count, tile_height, tile_width):
            raise NativeTemporalError(
                "Native temporal chunk shape does not match its metadata.",
                code="native_temporal_stream_shape_invalid",
            )
        destination = target[
            offset : offset + count,
            yoff : yoff + tile_height,
            xoff : xoff + tile_width,
        ]
        if spec.scale == 1.0:
            destination[...] = view
        else:
            np.multiply(view, spec.scale, out=destination, casting="unsafe")
        next_offset = offset + count
        tile_states[key] = next_offset
        if next_offset == time_count:
            spatial = coverage[yoff : yoff + tile_height, xoff : xoff + tile_width]
            if np.any(spatial):
                raise NativeTemporalError(
                    "Native temporal stream returned overlapping spatial tiles.",
                    code="native_temporal_stream_overlap",
                    details={"tile": key},
                )
            spatial[...] = True
            completed_pixels += tile_height * tile_width
            _report(
                progress_callback,
                stage="native_stream",
                percent=5.0 + 75.0 * completed_pixels / total_pixels,
                message="Streaming native temporal tiles.",
                completed=completed_pixels,
                total=total_pixels,
            )
    incomplete = [key for key, count in tile_states.items() if count != time_count]
    if incomplete or not np.all(coverage):
        raise NativeTemporalError(
            "Native temporal stream did not cover the complete output grid and time axis.",
            code="native_temporal_stream_incomplete",
            details={
                "incomplete_tile_count": len(incomplete),
                "missing_spatial_pixels": int(np.count_nonzero(~coverage)),
            },
        )


def generate_temporal_signal(
    *,
    signal: str,
    scenario_root: str | Path,
    dem_path: str | Path,
    horizons_path: str | Path,
    times: TimeRange,
    georef: GeoReference,
    storage: TemporalStorage,
    output_path: str | Path | None = None,
    surrounding_dem_paths: tuple[str | Path, ...] = (),
    observer_elevation_meters: float = 0.0,
    overwrite: bool = False,
    max_in_memory_bytes: int = _DEFAULT_MAX_IN_MEMORY_BYTES,
    scratch_directory: str | Path | None = None,
    patch_width: int = 128,
    patch_height: int = 128,
    chunk_time_count: int = 256,
    buffer_count: int = 6,
    poll_timeout_ms: int = 250,
    progress_callback: NativeProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
    _client: Any | None = None,
    _components: SimpleNamespace | None = None,
) -> TemporalCube | TemporalGeoTiffSeries:
    """Generate one native temporal signal with explicit result storage."""

    estimate = estimate_temporal_allocation(
        signal=signal,
        times=times,
        georef=georef,
        storage=storage,
        max_in_memory_bytes=max_in_memory_bytes,
    )
    if storage == "memory" and estimate.estimated_bytes > int(max_in_memory_bytes):
        raise NativeAllocationError(
            "Native temporal result exceeds the configured in-memory limit; select storage='geotiff_series'.",
            code="native_temporal_memory_limit_exceeded",
            details={
                "estimated_bytes": estimate.estimated_bytes,
                "limit_bytes": int(max_in_memory_bytes),
            },
        )
    if storage == "geotiff_series" and output_path is None:
        raise NativeInputError(
            "output_path is required for geotiff_series storage.",
            code="native_temporal_output_required",
        )
    if storage == "memory" and output_path is not None:
        raise NativeInputError(
            "output_path is only valid for geotiff_series storage.",
            code="native_temporal_output_not_allowed",
        )

    root = Path(scenario_root).expanduser().resolve()
    dem = Path(dem_path).expanduser().resolve()
    horizons = Path(horizons_path).expanduser().resolve()
    surrounding = tuple(Path(path).expanduser().resolve() for path in surrounding_dem_paths)
    if not root.is_dir() or not dem.is_file() or not horizons.is_dir():
        raise NativeInputError(
            "Scenario root, DEM, and horizons must exist before native generation.",
            code="native_temporal_input_missing",
            details={"scenario_root": str(root), "dem_path": str(dem), "horizons_path": str(horizons)},
        )
    for path in surrounding:
        if not path.is_file():
            raise NativeInputError(
                "A surrounding DEM does not exist.",
                code="native_temporal_input_missing",
                details={"path": str(path)},
            )

    spec = _signal_spec(signal)
    components = _components or _streaming_components()
    client = _client or _create_streaming_client(components.Client)
    request = _request(
        components,
        signal_spec=spec,
        scenario_root=root,
        dem_path=dem,
        surrounding_dem_paths=surrounding,
        horizons_path=horizons,
        times=times,
        observer_elevation_meters=observer_elevation_meters,
        patch_width=patch_width,
        patch_height=patch_height,
        chunk_time_count=chunk_time_count,
    )
    _raise_if_cancelled(cancellation_requested)
    _report(
        progress_callback,
        stage="preflight",
        percent=1.0,
        message="Native temporal allocation preflight complete.",
        completed=0,
        total=georef.height * georef.width,
    )

    writer: TemporalGeoTiffSeriesWriter | None = None
    scratch_path: Path | None = None
    target: NDArray[Any] | None = None
    stream: Iterator[tuple[Any, NDArray[Any]]] | None = None
    try:
        if storage == "memory":
            target = np.empty(estimate.shape, dtype=spec.dtype)
        else:
            assert output_path is not None
            writer = TemporalGeoTiffSeriesWriter(
                output_path,
                georef=georef.with_nodata(None),
                dtype=spec.dtype,
                signal_name=signal,
                units=spec.units,
                provenance={"generator": "MoonlibBridge", "native_signal": spec.native_name},
                overwrite=overwrite,
            )
            scratch_root = Path(scratch_directory).expanduser().resolve() if scratch_directory else writer.destination.parent
            scratch_root.mkdir(parents=True, exist_ok=True)
            scratch_available = shutil.disk_usage(scratch_root).free
            output_available = shutil.disk_usage(writer.destination.parent).free
            same_filesystem = scratch_root.stat().st_dev == writer.destination.parent.stat().st_dev
            required_scratch = estimate.estimated_bytes
            required_output = estimate.estimated_bytes
            if same_filesystem:
                enough_space = scratch_available >= required_scratch + required_output
            else:
                enough_space = (
                    scratch_available >= required_scratch
                    and output_available >= required_output
                )
            if not enough_space:
                raise NativeAllocationError(
                    "Native temporal scratch and output allocation exceeds available disk space.",
                    code="native_temporal_scratch_space_exceeded",
                    details={
                        "scratch_bytes": required_scratch,
                        "estimated_output_bytes": required_output,
                        "scratch_available_bytes": scratch_available,
                        "output_available_bytes": output_available,
                        "same_filesystem": same_filesystem,
                    },
                )
            descriptor, raw_scratch = tempfile.mkstemp(
                prefix=".lunarscout-native-temporal-", suffix=".dat", dir=scratch_root
            )
            os.close(descriptor)
            scratch_path = Path(raw_scratch)
            target = np.memmap(scratch_path, mode="w+", dtype=spec.dtype, shape=estimate.shape)

        stream = components.stream(
            client,
            request,
            buffer_count=max(1, int(buffer_count)),
            poll_timeout_ms=max(1, int(poll_timeout_ms)),
        )
        _fill_from_tiles(
            target,
            stream=stream,
            spec=spec,
            cancellation_requested=cancellation_requested,
            progress_callback=progress_callback,
        )
        _raise_if_cancelled(cancellation_requested)
        if storage == "memory":
            _report(
                progress_callback,
                stage="complete",
                percent=100.0,
                message="Native temporal in-memory result complete.",
                completed=georef.height * georef.width,
                total=georef.height * georef.width,
            )
            return TemporalCube(target, times, georef.with_nodata(None))

        assert writer is not None
        assert target is not None
        for index, time_value in enumerate(times.values):
            _raise_if_cancelled(cancellation_requested)
            writer.write_layer(time_value, np.asarray(target[index]))
            _report(
                progress_callback,
                stage="write_series",
                percent=80.0 + 19.0 * (index + 1) / times.time_count,
                message="Writing timestamped GeoTIFF layers.",
                completed=georef.height * georef.width,
                total=georef.height * georef.width,
            )
        _raise_if_cancelled(cancellation_requested)
        result = writer.finalize()
        writer = None
        _report(
            progress_callback,
            stage="complete",
            percent=100.0,
            message="Native temporal GeoTIFF series complete.",
            completed=georef.height * georef.width,
            total=georef.height * georef.width,
        )
        return result
    except (NativeInputError, NativeAllocationError, NativeTemporalError):
        raise
    except Exception as exc:
        raise NativeTemporalError(
            "Native temporal generation failed.",
            code="native_temporal_generation_failed",
            details={"signal": signal, "storage": storage, "error": str(exc)},
        ) from exc
    finally:
        close_stream = getattr(stream, "close", None)
        if callable(close_stream):
            close_stream()
        if writer is not None:
            writer.abort()
        if isinstance(target, np.memmap):
            target.flush()
        target = None
        if scratch_path is not None:
            scratch_path.unlink(missing_ok=True)
