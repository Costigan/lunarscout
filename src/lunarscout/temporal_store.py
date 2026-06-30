from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
import rasterio

from .alignment import same_grid
from .errors import (
    GeoTiffDataTypeError,
    TemporalLookupError,
    TemporalOperationError,
    TemporalSeriesOpenError,
    TemporalSeriesWriteError,
)
from .georeference import GeoReference
from .geotiff import _validate_geotiff_dtype, read_geotiff, write_geotiff
from .temporal import (
    Nodata,
    TemporalCube,
    TimeInput,
    TimeRange,
    _datetime64_utc,
    _parse_time,
)


TimeLookupMethod: TypeAlias = Literal["exact", "nearest", "before", "after"]
CancellationCheck: TypeAlias = Callable[[], bool]

_FORMAT = "lunarscout.temporal_geotiff_series"
_FORMAT_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_VRT_NAME = "series.vrt"
_COMPLETE_NAME = "COMPLETE"
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_DEFAULT_LAYER_CACHE_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_OPEN_DATASETS = 32


def _time_text(value: np.datetime64) -> str:
    return f"{np.datetime_as_string(value.astype('datetime64[us]'), unit='us')}Z"


def _time_filename(value: np.datetime64) -> str:
    return _time_text(value).replace("-", "").replace(":", "") + ".tif"


def _reject_nonstandard_json(value: str) -> None:
    raise ValueError(f"non-standard JSON value: {value}")


def _parse_manifest_time(value: Any) -> np.datetime64:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise TemporalSeriesOpenError(
            "Manifest time must be a UTC timestamp with microsecond precision.",
            code="temporal_series_invalid_time",
            details={"time": value},
        )
    try:
        return np.datetime64(value[:-1], "us")
    except ValueError as exc:
        raise TemporalSeriesOpenError(
            "Manifest contains an invalid UTC timestamp.",
            code="temporal_series_invalid_time",
            details={"time": value},
        ) from exc


def _encode_nodata(nodata: int | float | None) -> dict[str, Any]:
    if nodata is None:
        return {"kind": "none"}
    if isinstance(nodata, (np.integer, np.floating)):
        nodata = nodata.item()
    if isinstance(nodata, float):
        if np.isnan(nodata):
            return {"kind": "nan"}
        if np.isposinf(nodata):
            return {"kind": "positive_infinity"}
        if np.isneginf(nodata):
            return {"kind": "negative_infinity"}
    return {"kind": "value", "value": nodata}


def _decode_nodata(payload: Any) -> int | float | None:
    if not isinstance(payload, dict):
        raise TemporalSeriesOpenError(
            "Manifest nodata must use the tagged object encoding.",
            code="temporal_series_invalid_nodata",
        )
    kind = payload.get("kind")
    if kind == "none":
        return None
    if kind == "nan":
        return float("nan")
    if kind == "positive_infinity":
        return float("inf")
    if kind == "negative_infinity":
        return float("-inf")
    if kind == "value":
        value = payload.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TemporalSeriesOpenError(
                "Manifest nodata value must be numeric.",
                code="temporal_series_invalid_nodata",
                details={"nodata": value},
            )
        return value
    raise TemporalSeriesOpenError(
        "Manifest contains an unknown nodata encoding.",
        code="temporal_series_invalid_nodata",
        details={"kind": kind},
    )


def _nodata_equal(left: int | float | None, right: int | float | None) -> bool:
    if left is None or right is None:
        return left is right
    try:
        if np.isnan(left) and np.isnan(right):
            return True
    except TypeError:
        pass
    return left == right


def _canonical_json(payload: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TemporalSeriesWriteError(
            "Temporal series metadata must be standard JSON data.",
            code="temporal_series_metadata_not_json",
            details={"error": str(exc)},
        ) from exc


def _safe_relative_path(root: Path, raw: Any, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise TemporalSeriesOpenError(
            f"Manifest {field} must be a non-empty relative path.",
            code="temporal_series_invalid_path",
            details={"field": field, "path": raw},
        )
    relative = Path(raw)
    if relative.is_absolute():
        raise TemporalSeriesOpenError(
            f"Manifest {field} cannot be absolute.",
            code="temporal_series_absolute_path",
            details={"field": field, "path": raw},
        )
    try:
        resolved = (root / relative).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise TemporalSeriesOpenError(
            f"Manifest {field} cannot be resolved.",
            code="temporal_series_invalid_path",
            details={"field": field, "path": raw, "error": str(exc)},
        ) from exc
    if not resolved.is_relative_to(root):
        raise TemporalSeriesOpenError(
            f"Manifest {field} escapes the series root.",
            code="temporal_series_path_escape",
            details={"field": field, "path": raw},
        )
    return resolved


def _manifest_georef(payload: Any) -> GeoReference:
    if not isinstance(payload, dict):
        raise TemporalSeriesOpenError(
            "Manifest georeference must be an object.",
            code="temporal_series_invalid_georef",
        )
    try:
        affine = tuple(float(value) for value in payload["affine_transform"])
        return GeoReference(
            projection_wkt=str(payload["projection_wkt"]),
            projection_proj4=str(payload["projection_proj4"]),
            affine_transform=affine,  # type: ignore[arg-type]
            width=int(payload["width"]),
            height=int(payload["height"]),
            pixel_size_x=float(affine[1]),
            pixel_size_y=float(affine[5]),
            nodata=_decode_nodata(payload["nodata"]),
        )
    except TemporalSeriesOpenError:
        raise
    except Exception as exc:
        raise TemporalSeriesOpenError(
            "Manifest contains invalid georeferencing.",
            code="temporal_series_invalid_georef",
            details={"error": str(exc)},
        ) from exc


def _georef_payload(georef: GeoReference) -> dict[str, Any]:
    return {
        "projection_wkt": georef.projection_wkt,
        "projection_proj4": georef.projection_proj4,
        "affine_transform": list(georef.affine_transform),
        "width": georef.width,
        "height": georef.height,
        "nodata": _encode_nodata(georef.nodata),
    }


def _layer_metadata(path: Path) -> tuple[np.dtype[Any], GeoReference]:
    try:
        with rasterio.open(path) as dataset:
            if dataset.driver != "GTiff" or int(dataset.count) != 1:
                raise ValueError("layer must be a readable single-band GeoTIFF")
            dtype = np.dtype(dataset.dtypes[0])
        _values, georef = read_geotiff(path)
        if georef is None:
            raise ValueError("layer must be a readable single-band GeoTIFF")
        return dtype, georef
    except TemporalSeriesOpenError:
        raise
    except Exception as exc:
        raise TemporalSeriesOpenError(
            "Unable to validate temporal series layer.",
            code="temporal_series_invalid_layer",
            details={"path": str(path), "error": str(exc)},
        ) from exc


_VRT_DTYPE_NAMES = {
    np.dtype(np.uint8): "Byte",
    np.dtype(np.int8): "Int8",
    np.dtype(np.uint16): "UInt16",
    np.dtype(np.int16): "Int16",
    np.dtype(np.uint32): "UInt32",
    np.dtype(np.int32): "Int32",
    np.dtype(np.uint64): "UInt64",
    np.dtype(np.int64): "Int64",
    np.dtype(np.float32): "Float32",
    np.dtype(np.float64): "Float64",
}


def _vrt_datatype_name(dtype: np.dtype[Any]) -> str:
    _validate_geotiff_dtype(np.empty(0, dtype=dtype))
    try:
        return _VRT_DTYPE_NAMES[np.dtype(dtype)]
    except KeyError as exc:
        raise TemporalSeriesWriteError(
            "Temporal series dtype cannot be represented in a VRT.",
            code="temporal_series_invalid_dtype",
            details={"dtype": str(dtype)},
        ) from exc


def _same_layer_metadata(
    actual_dtype: np.dtype[Any],
    actual_georef: GeoReference,
    expected_dtype: np.dtype[Any],
    expected_georef: GeoReference,
) -> bool:
    if actual_dtype != expected_dtype:
        return False
    if not _nodata_equal(actual_georef.nodata, expected_georef.nodata):
        return False
    if actual_georef.projection_wkt == expected_georef.projection_wkt:
        return (
            actual_georef.width == expected_georef.width
            and actual_georef.height == expected_georef.height
            and actual_georef.affine_transform == expected_georef.affine_transform
        )
    return same_grid(actual_georef, expected_georef)


def _write_vrt(
    path: Path,
    *,
    georef: GeoReference,
    dtype: np.dtype[Any],
    layers: list[dict[str, Any]],
) -> None:
    datatype_name = _vrt_datatype_name(dtype)
    root = ET.Element(
        "VRTDataset",
        rasterXSize=str(georef.width),
        rasterYSize=str(georef.height),
    )
    ET.SubElement(root, "SRS").text = georef.projection_wkt
    ET.SubElement(root, "GeoTransform").text = ", ".join(
        repr(value) for value in georef.affine_transform
    )
    for layer in layers:
        band = ET.SubElement(
            root,
            "VRTRasterBand",
            dataType=datatype_name,
            band=str(int(layer["index"]) + 1),
        )
        ET.SubElement(band, "Description").text = str(layer["time_utc"])
        metadata = ET.SubElement(band, "Metadata")
        ET.SubElement(metadata, "MDI", key="TIMESTAMP_UTC").text = str(
            layer["time_utc"]
        )
        if georef.nodata is not None:
            ET.SubElement(band, "NoDataValue").text = str(georef.nodata)
        source = ET.SubElement(band, "SimpleSource")
        ET.SubElement(source, "SourceFilename", relativeToVRT="1").text = str(
            layer["relative_path"]
        )
        ET.SubElement(source, "SourceBand").text = "1"
        rect = {
            "xOff": "0",
            "yOff": "0",
            "xSize": str(georef.width),
            "ySize": str(georef.height),
        }
        ET.SubElement(source, "SrcRect", **rect)
        ET.SubElement(source, "DstRect", **rect)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _commit_staging_directory(
    staging: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if destination.is_symlink():
        raise TemporalSeriesWriteError(
            "Temporal series destination cannot be a symbolic link.",
            code="temporal_series_destination_symlink",
            details={"path": str(destination)},
        )
    if destination.exists() and not destination.is_dir():
        raise TemporalSeriesWriteError(
            "Temporal series destination exists and is not a directory.",
            code="temporal_series_destination_not_directory",
            details={"path": str(destination)},
        )
    if destination.exists() and not overwrite:
        raise TemporalSeriesWriteError(
            "Temporal series destination already exists.",
            code="temporal_series_output_exists",
            details={"path": str(destination)},
        )
    if not destination.exists():
        os.replace(staging, destination)
        return

    backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class TemporalWriteProgress:
    """Progress emitted after one temporal GeoTIFF layer is durable."""

    layers_written: int
    last_index: int
    last_time: np.datetime64
    layer_path: Path


ProgressCallback: TypeAlias = Callable[[TemporalWriteProgress], None]


class TemporalGeoTiffSeriesWriter:
    """Incrementally build a completed temporal GeoTIFF series."""

    def __init__(
        self,
        path: str | Path,
        *,
        georef: GeoReference,
        dtype: np.dtype[Any] | type[Any] | str,
        signal_name: str | None = None,
        units: str | None = None,
        provenance: dict[str, Any] | None = None,
        overwrite: bool = False,
        create_vrt: bool = True,
        progress_callback: ProgressCallback | None = None,
        cancellation_requested: CancellationCheck | None = None,
    ) -> None:
        if not isinstance(georef, GeoReference):
            raise TemporalSeriesWriteError(
                "Temporal series georef must be a GeoReference.",
                code="temporal_series_invalid_georef",
        )
        try:
            resolved_dtype = np.dtype(dtype)
            _validate_geotiff_dtype(np.empty(0, dtype=resolved_dtype))
        except (TypeError, ValueError, GeoTiffDataTypeError) as exc:
            raise TemporalSeriesWriteError(
                "Temporal series dtype must be supported by GeoTIFF.",
                code="temporal_series_invalid_dtype",
                details={"dtype": str(dtype)},
            ) from exc
        if signal_name is not None and not isinstance(signal_name, str):
            raise TemporalSeriesWriteError(
                "signal_name must be a string or None.",
                code="temporal_series_invalid_metadata",
            )
        if units is not None and not isinstance(units, str):
            raise TemporalSeriesWriteError(
                "units must be a string or None.",
                code="temporal_series_invalid_metadata",
            )
        if provenance is not None and not isinstance(provenance, dict):
            raise TemporalSeriesWriteError(
                "provenance must be a JSON object or None.",
                code="temporal_series_invalid_metadata",
            )
        metadata = {
            "signal_name": signal_name,
            "units": units,
            "provenance": dict(provenance or {}),
        }
        _canonical_json(metadata)
        if progress_callback is not None and not callable(progress_callback):
            raise TemporalSeriesWriteError(
                "progress_callback must be callable or None.",
                code="temporal_series_invalid_callback",
            )
        if cancellation_requested is not None and not callable(cancellation_requested):
            raise TemporalSeriesWriteError(
                "cancellation_requested must be callable or None.",
                code="temporal_series_invalid_callback",
            )

        raw_destination = Path(path).expanduser()
        if raw_destination.name in {"", ".", ".."}:
            raise TemporalSeriesWriteError(
                "Temporal series destination must name a directory.",
                code="temporal_series_invalid_destination",
                details={"path": str(path)},
            )
        parent = raw_destination.parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
        destination = parent / raw_destination.name
        if destination.is_symlink():
            raise TemporalSeriesWriteError(
                "Temporal series destination cannot be a symbolic link.",
                code="temporal_series_destination_symlink",
                details={"path": str(destination)},
            )
        if destination.exists() and not destination.is_dir():
            raise TemporalSeriesWriteError(
                "Temporal series destination exists and is not a directory.",
                code="temporal_series_destination_not_directory",
                details={"path": str(destination)},
            )
        if destination.exists() and not overwrite:
            raise TemporalSeriesWriteError(
                "Temporal series destination already exists.",
                code="temporal_series_output_exists",
                details={"path": str(destination)},
            )

        staging = parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
        try:
            staging.mkdir()
            (staging / "layers").mkdir()
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise TemporalSeriesWriteError(
                "Unable to create temporal series staging directory.",
                code="temporal_series_staging_failed",
                details={"path": str(staging), "error": str(exc)},
            ) from exc

        self.destination = destination
        self.georef = georef
        self.dtype = resolved_dtype
        self.signal_name = signal_name
        self.units = units
        self.provenance = metadata["provenance"]
        self.overwrite = overwrite
        self.create_vrt = bool(create_vrt)
        self.progress_callback = progress_callback
        self.cancellation_requested = cancellation_requested
        self._staging: Path | None = staging
        self._layers: list[dict[str, Any]] = []
        self._last_time: np.datetime64 | None = None
        self._state = "active"
        self._result: TemporalGeoTiffSeries | None = None

    @property
    def layers_written(self) -> int:
        return len(self._layers)

    @property
    def result(self) -> TemporalGeoTiffSeries | None:
        return self._result

    def _ensure_active(self) -> Path:
        if self._state != "active" or self._staging is None:
            raise TemporalSeriesWriteError(
                "Temporal series writer is no longer active.",
                code="temporal_series_writer_inactive",
                details={"state": self._state},
            )
        return self._staging

    def _check_cancelled(self) -> None:
        if self.cancellation_requested is not None and self.cancellation_requested():
            raise TemporalSeriesWriteError(
                "Temporal series write was cancelled.",
                code="temporal_series_write_cancelled",
                details={"layers_written": self.layers_written},
            )

    def write_layer(self, time: TimeInput, array: NDArray[Any]) -> Path:
        """Write one strictly increasing UTC layer to the staging series."""

        staging = self._ensure_active()
        try:
            self._check_cancelled()
            time_value = _datetime64_utc(
                _parse_time(time, source_timezone=None)
            ).astype("datetime64[us]")
            if np.isnat(time_value):
                raise ValueError("time cannot be NaT")
            if self._last_time is not None and time_value <= self._last_time:
                raise TemporalSeriesWriteError(
                    "Temporal layer times must be unique and strictly increasing.",
                    code="temporal_series_times_not_increasing",
                    details={"time": str(time)},
                )
            values = np.asarray(array)
            expected_shape = (self.georef.height, self.georef.width)
            if values.ndim != 2 or values.shape != expected_shape:
                raise TemporalSeriesWriteError(
                    "Temporal layer shape does not match its GeoReference.",
                    code="temporal_series_layer_shape_mismatch",
                    details={
                        "shape": list(values.shape),
                        "expected_shape": list(expected_shape),
                    },
                )
            if values.dtype != self.dtype:
                raise TemporalSeriesWriteError(
                    "Temporal layer dtype does not match the series dtype.",
                    code="temporal_series_layer_dtype_mismatch",
                    details={"dtype": str(values.dtype), "expected_dtype": str(self.dtype)},
                )

            index = self.layers_written
            relative_path = Path("layers") / _time_filename(time_value)
            staged_path = staging / relative_path
            write_geotiff(staged_path, values, self.georef)
            entry = {
                "index": index,
                "time_utc": _time_text(time_value),
                "relative_path": relative_path.as_posix(),
            }
            self._layers.append(entry)
            self._last_time = time_value
            if self.progress_callback is not None:
                self.progress_callback(
                    TemporalWriteProgress(
                        layers_written=self.layers_written,
                        last_index=index,
                        last_time=time_value,
                        layer_path=self.destination / relative_path,
                    )
                )
            return self.destination / relative_path
        except TemporalSeriesWriteError:
            self.abort()
            raise
        except Exception as exc:
            self.abort()
            raise TemporalSeriesWriteError(
                "Unable to write temporal GeoTIFF layer.",
                code="temporal_series_layer_write_failed",
                details={"time": str(time), "error": str(exc)},
            ) from exc

    def finalize(self) -> TemporalGeoTiffSeries:
        """Publish the staged series atomically and return its open reader."""

        staging = self._ensure_active()
        try:
            self._check_cancelled()
            if not self._layers:
                raise TemporalSeriesWriteError(
                    "Temporal series must contain at least one layer.",
                    code="temporal_series_empty",
                )
            vrt_relative_path: str | None = None
            if self.create_vrt:
                _write_vrt(
                    staging / _VRT_NAME,
                    georef=self.georef,
                    dtype=self.dtype,
                    layers=self._layers,
                )
                vrt_relative_path = _VRT_NAME
            manifest = {
                "format": _FORMAT,
                "format_version": _FORMAT_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "signal_name": self.signal_name,
                "units": self.units,
                "dtype": self.dtype.name,
                "shape": [
                    self.layers_written,
                    self.georef.height,
                    self.georef.width,
                ],
                "georeference": _georef_payload(self.georef),
                "layers": self._layers,
                "vrt_relative_path": vrt_relative_path,
                "provenance": self.provenance,
            }
            manifest_bytes = _canonical_json(manifest)
            (staging / _MANIFEST_NAME).write_bytes(manifest_bytes)
            completion = {
                "format_version": _FORMAT_VERSION,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }
            (staging / _COMPLETE_NAME).write_bytes(_canonical_json(completion))
            self._check_cancelled()
            _commit_staging_directory(
                staging, self.destination, overwrite=self.overwrite
            )
            self._staging = None
            self._state = "finalized"
            self._result = open_temporal_cube(self.destination)
            return self._result
        except TemporalSeriesWriteError:
            self.abort()
            raise
        except Exception as exc:
            self.abort()
            raise TemporalSeriesWriteError(
                "Unable to finalize temporal GeoTIFF series.",
                code="temporal_series_finalize_failed",
                details={"path": str(self.destination), "error": str(exc)},
            ) from exc

    def abort(self) -> None:
        """Discard an unfinished staging series; completed output is untouched."""

        staging = self._staging
        self._staging = None
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if self._state == "active":
            self._state = "aborted"

    close = abort

    def __enter__(self) -> TemporalGeoTiffSeriesWriter:
        self._ensure_active()
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        if exc_type is not None:
            self.abort()
        elif self._state == "active":
            self.finalize()


def write_temporal_cube(
    path: str | Path,
    cube: TemporalCube,
    *,
    signal_name: str | None = None,
    units: str | None = None,
    provenance: dict[str, Any] | None = None,
    overwrite: bool = False,
    create_vrt: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> TemporalGeoTiffSeries:
    """Persist an in-memory cube as a completed timestamped GeoTIFF series."""

    if not isinstance(cube, TemporalCube):
        raise TemporalSeriesWriteError(
            "write_temporal_cube requires a TemporalCube.",
            code="temporal_series_invalid_cube",
        )
    writer = TemporalGeoTiffSeriesWriter(
        path,
        georef=cube.georef,
        dtype=cube.dtype,
        signal_name=signal_name,
        units=units,
        provenance=provenance,
        overwrite=overwrite,
        create_vrt=create_vrt,
        progress_callback=progress_callback,
        cancellation_requested=cancellation_requested,
    )
    try:
        for index, time_value in enumerate(cube.times):
            writer.write_layer(time_value, cube.values[index])
        return writer.finalize()
    except TemporalSeriesWriteError:
        raise
    except Exception as exc:
        raise TemporalSeriesWriteError(
            "Unable to write temporal GeoTIFF series.",
            code="temporal_series_write_failed",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    finally:
        writer.abort()


class TemporalGeoTiffSeries:
    """Safe, file-backed temporal raster series indexed by layer or UTC time."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: dict[str, Any],
        georef: GeoReference,
        dtype: np.dtype[Any],
        times: NDArray[np.datetime64],
        layer_paths: tuple[Path, ...],
        vrt_path: Path | None,
        layer_cache_bytes: int,
        max_open_datasets: int,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.georef = georef
        self.dtype = dtype
        self.times = times
        self.layer_paths = layer_paths
        self.vrt_path = vrt_path
        self.layer_cache_bytes = layer_cache_bytes
        self.max_open_datasets = max_open_datasets
        self._layer_cache: OrderedDict[int, NDArray[Any]] = OrderedDict()
        self._layer_cache_size = 0
        self._dataset_cache: OrderedDict[int, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False

    @property
    def time_count(self) -> int:
        return len(self.layer_paths)

    @property
    def height(self) -> int:
        return self.georef.height

    @property
    def width(self) -> int:
        return self.georef.width

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.time_count, self.height, self.width)

    @property
    def dimensions(self) -> tuple[str, str, str]:
        return ("time", "y", "x")

    @property
    def signal_name(self) -> str | None:
        value = self.manifest.get("signal_name")
        return value if isinstance(value, str) else None

    @property
    def units(self) -> str | None:
        value = self.manifest.get("units")
        return value if isinstance(value, str) else None

    def _ensure_open(self) -> None:
        if self._closed:
            raise TemporalSeriesOpenError(
                "Temporal series is closed.",
                code="temporal_series_closed",
                details={"path": str(self.root)},
            )

    def _layer_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, Integral):
            index_value = -1
        else:
            index_value = int(index)
        if index_value < 0 or index_value >= self.time_count:
            raise TemporalLookupError(
                "Temporal layer index is out of range.",
                code="temporal_layer_out_of_range",
                details={"index": index, "time_count": self.time_count},
            )
        return index_value

    def time_for_layer(self, index: int) -> np.datetime64:
        return self.times[self._layer_index(index)]

    def layer_for_time(
        self,
        time: TimeInput,
        *,
        method: TimeLookupMethod = "exact",
    ) -> int:
        self._ensure_open()
        if method not in {"exact", "nearest", "before", "after"}:
            raise TemporalLookupError(
                "Unknown temporal lookup method.",
                code="temporal_lookup_invalid_method",
                details={"method": method},
            )
        target = _datetime64_utc(_parse_time(time, source_timezone=None)).astype("datetime64[us]")
        insertion = int(np.searchsorted(self.times, target, side="left"))
        if method == "exact":
            if insertion < self.time_count and self.times[insertion] == target:
                return insertion
        elif method == "after":
            if insertion < self.time_count:
                return insertion
        elif method == "before":
            candidate = int(np.searchsorted(self.times, target, side="right")) - 1
            if candidate >= 0:
                return candidate
        else:
            if insertion == 0 and self.times[0] != target:
                pass
            elif insertion == self.time_count:
                pass
            elif insertion == 0:
                return 0
            else:
                before = self.times[insertion - 1].astype(np.int64)
                after = self.times[insertion].astype(np.int64)
                target_integer = target.astype(np.int64)
                return (
                    insertion - 1
                    if target_integer - before <= after - target_integer
                    else insertion
                )
        raise TemporalLookupError(
            "No temporal layer satisfies the requested lookup.",
            code="temporal_time_not_found",
            details={"time": str(time), "method": method},
        )

    def _dataset(self, index: int):
        self._ensure_open()
        with self._lock:
            cached = self._dataset_cache.pop(index, None)
            if cached is not None:
                self._dataset_cache[index] = cached
                return cached
            try:
                dataset = rasterio.open(self.layer_paths[index])
            except Exception as exc:
                raise TemporalSeriesOpenError(
                    "Unable to open temporal series layer.",
                    code="temporal_series_layer_open_failed",
                    details={
                        "path": str(self.layer_paths[index]),
                        "index": index,
                        "error": str(exc),
                    },
                ) from exc
            if dataset.driver != "GTiff" or int(dataset.count) < 1:
                dataset.close()
                raise TemporalSeriesOpenError(
                    "Temporal series layer is not a readable GeoTIFF.",
                    code="temporal_series_layer_open_failed",
                    details={"path": str(self.layer_paths[index]), "index": index},
                )
            self._dataset_cache[index] = dataset
            while len(self._dataset_cache) > self.max_open_datasets:
                _old_index, old_dataset = self._dataset_cache.popitem(last=False)
                old_dataset.close()
            return dataset

    def _read_layer(self, index: int, *, use_layer_cache: bool) -> NDArray[Any]:
        index = self._layer_index(index)
        with self._lock:
            if use_layer_cache:
                cached = self._layer_cache.pop(index, None)
                if cached is not None:
                    self._layer_cache[index] = cached
                    return cached
            dataset = self._dataset(index)
            try:
                values = dataset.read(1)
            except Exception as exc:
                raise TemporalSeriesOpenError(
                    "Unable to read temporal series layer.",
                    code="temporal_series_layer_read_failed",
                    details={
                        "path": str(self.layer_paths[index]),
                        "index": index,
                        "error": str(exc),
                    },
                ) from exc
            result = np.asarray(values)
            result.flags.writeable = False
            if use_layer_cache and result.nbytes <= self.layer_cache_bytes:
                self._layer_cache[index] = result
                self._layer_cache_size += result.nbytes
                while self._layer_cache_size > self.layer_cache_bytes:
                    _evicted_index, evicted = self._layer_cache.popitem(last=False)
                    self._layer_cache_size -= evicted.nbytes
            return result

    def read_layer(self, index: int) -> tuple[NDArray[Any], GeoReference]:
        return self._read_layer(index, use_layer_cache=True), self.georef

    def read_time(
        self,
        time: TimeInput,
        *,
        method: TimeLookupMethod = "exact",
    ) -> tuple[NDArray[Any], GeoReference]:
        return self.read_layer(self.layer_for_time(time, method=method))

    def close(self) -> None:
        with self._lock:
            for dataset in self._dataset_cache.values():
                dataset.close()
            self._dataset_cache.clear()
            self._layer_cache.clear()
            self._layer_cache_size = 0
            self._closed = True

    def __enter__(self) -> TemporalGeoTiffSeries:
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _temporal_reduce(
        self,
        operation: str,
        *,
        nodata: Nodata,
        dtype: np.dtype[Any] | type[Any] | str | None = None,
        ddof: float = 0,
    ) -> tuple[NDArray[Any], GeoReference]:
        self._ensure_open()
        if operation == "std":
            try:
                ddof_value = float(ddof)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TemporalOperationError(
                    "ddof must be a finite non-negative number.",
                    code="temporal_invalid_ddof",
                    details={"ddof": ddof},
                ) from exc
            if not np.isfinite(ddof_value) or ddof_value < 0:
                raise TemporalOperationError(
                    "ddof must be a finite non-negative number.",
                    code="temporal_invalid_ddof",
                    details={"ddof": ddof},
                )
        else:
            ddof_value = 0.0
        if nodata == "auto":
            resolved_nodata = self.georef.nodata
        elif nodata is None or isinstance(
            nodata, (int, float, np.integer, np.floating)
        ):
            resolved_nodata = nodata.item() if isinstance(
                nodata, (np.integer, np.floating)
            ) else nodata
        else:
            raise TemporalOperationError(
                "nodata must be 'auto', None, or a numeric value.",
                code="temporal_invalid_nodata",
                details={"nodata": nodata},
            )

        if dtype is None:
            accumulator_dtype = (
                np.dtype(np.float64)
                if np.issubdtype(self.dtype, np.integer)
                else self.dtype
            )
        else:
            try:
                accumulator_dtype = np.dtype(dtype)
            except (TypeError, ValueError) as exc:
                raise TemporalOperationError(
                    "Invalid temporal reduction dtype.",
                    code="temporal_invalid_reduction_dtype",
                    details={"dtype": str(dtype)},
                ) from exc

        spatial_shape = (self.height, self.width)
        count = np.zeros(spatial_shape, dtype=np.uint32)
        if operation in {"mean", "std"}:
            aggregate = np.zeros(spatial_shape, dtype=accumulator_dtype)
        else:
            aggregate = np.zeros(spatial_shape, dtype=self.dtype)
        m2 = np.zeros(spatial_shape, dtype=accumulator_dtype) if operation == "std" else None

        for index in range(self.time_count):
            layer = self._read_layer(index, use_layer_cache=False)
            if resolved_nodata is None:
                valid = np.ones(spatial_shape, dtype=bool)
            elif isinstance(resolved_nodata, float) and np.isnan(resolved_nodata):
                valid = ~np.isnan(layer)
            else:
                valid = layer != resolved_nodata
            if operation == "mean":
                aggregate[valid] += layer[valid]
            elif operation == "std":
                previous = count[valid].astype(accumulator_dtype, copy=False)
                values = layer[valid].astype(accumulator_dtype, copy=False)
                delta = values - aggregate[valid]
                new_count = previous + 1
                aggregate[valid] += delta / new_count
                assert m2 is not None
                m2[valid] += delta * (values - aggregate[valid])
            elif operation == "min":
                first = valid & (count == 0)
                subsequent = valid & (count > 0)
                aggregate[first] = layer[first]
                aggregate[subsequent] = np.minimum(
                    aggregate[subsequent], layer[subsequent]
                )
            elif operation == "max":
                first = valid & (count == 0)
                subsequent = valid & (count > 0)
                aggregate[first] = layer[first]
                aggregate[subsequent] = np.maximum(
                    aggregate[subsequent], layer[subsequent]
                )
            else:  # pragma: no cover - private dispatch is controlled
                raise AssertionError(operation)
            count[valid] += 1

        if operation == "mean":
            output = np.zeros(spatial_shape, dtype=accumulator_dtype)
            np.divide(aggregate, count, out=output, where=count > 0)
        elif operation == "std":
            output = np.zeros(spatial_shape, dtype=accumulator_dtype)
            valid_count = count > ddof_value
            assert m2 is not None
            np.divide(m2, count - ddof_value, out=output, where=valid_count)
            np.sqrt(output, out=output)
            count = np.where(valid_count, count, 0).astype(np.uint32)
        else:
            output = aggregate

        invalid_output = count == 0
        if np.any(invalid_output):
            if resolved_nodata is None:
                if np.issubdtype(output.dtype, np.floating):
                    output[invalid_output] = np.nan
                else:
                    raise TemporalOperationError(
                        "Temporal reduction has no valid value for some pixels.",
                        code="temporal_reduction_empty_pixel",
                    )
            else:
                try:
                    output[invalid_output] = resolved_nodata
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TemporalOperationError(
                        "Output nodata cannot be represented by the reduced dtype.",
                        code="temporal_unrepresentable_nodata",
                        details={"dtype": str(output.dtype), "nodata": resolved_nodata},
                    ) from exc
        return output, self.georef.with_nodata(resolved_nodata)


def open_temporal_cube(
    path: str | Path,
    *,
    layer_cache_bytes: int = _DEFAULT_LAYER_CACHE_BYTES,
    max_open_datasets: int = _DEFAULT_MAX_OPEN_DATASETS,
    validate_layers: bool = True,
) -> TemporalGeoTiffSeries:
    """Open and validate a completed timestamped GeoTIFF temporal series."""

    try:
        cache_bytes = int(layer_cache_bytes)
        open_limit = int(max_open_datasets)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TemporalSeriesOpenError(
            "Temporal cache limits must be integers.",
            code="temporal_series_invalid_cache_config",
        ) from exc
    if cache_bytes < 0 or open_limit < 1:
        raise TemporalSeriesOpenError(
            "layer_cache_bytes must be non-negative and max_open_datasets positive.",
            code="temporal_series_invalid_cache_config",
            details={
                "layer_cache_bytes": layer_cache_bytes,
                "max_open_datasets": max_open_datasets,
            },
        )
    try:
        root = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise TemporalSeriesOpenError(
            "Temporal series directory does not exist.",
            code="temporal_series_not_found",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not root.is_dir():
        raise TemporalSeriesOpenError(
            "Temporal series path must be a directory.",
            code="temporal_series_not_directory",
            details={"path": str(root)},
        )
    manifest_path = root / _MANIFEST_NAME
    completion_path = root / _COMPLETE_NAME
    if not manifest_path.is_file() or not completion_path.is_file():
        raise TemporalSeriesOpenError(
            "Temporal series is incomplete.",
            code="temporal_series_incomplete",
            details={"path": str(root)},
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes, parse_constant=_reject_nonstandard_json)
        completion = json.loads(
            completion_path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TemporalSeriesOpenError(
            "Temporal series metadata is unreadable.",
            code="temporal_series_metadata_unreadable",
            details={"path": str(root), "error": str(exc)},
        ) from exc
    expected_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if not isinstance(completion, dict) or completion.get("manifest_sha256") != expected_digest:
        raise TemporalSeriesOpenError(
            "Temporal series completion digest does not match its manifest.",
            code="temporal_series_completion_mismatch",
            details={"path": str(root)},
        )
    if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT:
        raise TemporalSeriesOpenError(
            "Directory is not a Lunarscout temporal GeoTIFF series.",
            code="temporal_series_unknown_format",
            details={"path": str(root)},
        )
    if manifest.get("format_version") != _FORMAT_VERSION:
        raise TemporalSeriesOpenError(
            "Temporal series format version is unsupported.",
            code="temporal_series_unsupported_version",
            details={"format_version": manifest.get("format_version")},
        )
    if completion.get("format_version") != _FORMAT_VERSION:
        raise TemporalSeriesOpenError(
            "Completion record format version does not match the reader.",
            code="temporal_series_completion_mismatch",
        )
    try:
        dtype = np.dtype(manifest["dtype"])
        _validate_geotiff_dtype(np.empty(0, dtype=dtype))
    except (KeyError, TypeError, ValueError, GeoTiffDataTypeError) as exc:
        raise TemporalSeriesOpenError(
            "Manifest contains an invalid dtype.",
            code="temporal_series_invalid_dtype",
            details={"dtype": manifest.get("dtype")},
        ) from exc
    georef = _manifest_georef(manifest.get("georeference"))
    layers = manifest.get("layers")
    shape = manifest.get("shape")
    if not isinstance(layers, list) or not layers:
        raise TemporalSeriesOpenError(
            "Manifest must contain at least one layer.",
            code="temporal_series_invalid_layers",
        )
    if shape != [len(layers), georef.height, georef.width]:
        raise TemporalSeriesOpenError(
            "Manifest shape does not match its layers or georeference.",
            code="temporal_series_shape_mismatch",
            details={"shape": shape},
        )

    time_values: list[np.datetime64] = []
    layer_paths: list[Path] = []
    for expected_index, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("index") != expected_index:
            raise TemporalSeriesOpenError(
                "Manifest layer indexes must be contiguous and zero-based.",
                code="temporal_series_invalid_layer_index",
                details={"expected_index": expected_index},
            )
        time_value = _parse_manifest_time(layer.get("time_utc"))
        layer_path = _safe_relative_path(
            root, layer.get("relative_path"), field="layer path"
        )
        expected_filename = _time_filename(time_value)
        if layer_path.name != expected_filename:
            raise TemporalSeriesOpenError(
                "Layer filename does not match its UTC timestamp.",
                code="temporal_series_filename_mismatch",
                details={
                    "index": expected_index,
                    "filename": layer_path.name,
                    "expected_filename": expected_filename,
                },
            )
        if not layer_path.is_file():
            raise TemporalSeriesOpenError(
                "Temporal series layer is missing.",
                code="temporal_series_layer_missing",
                details={"index": expected_index, "path": str(layer_path)},
            )
        time_values.append(time_value)
        layer_paths.append(layer_path)
    times_array = np.asarray(time_values, dtype="datetime64[us]")
    if np.any(np.diff(times_array).astype(np.int64) <= 0):
        raise TemporalSeriesOpenError(
            "Temporal series times must be unique and strictly increasing.",
            code="temporal_series_times_not_increasing",
        )
    times_array.flags.writeable = False

    vrt_path = None
    raw_vrt_path = manifest.get("vrt_relative_path")
    if raw_vrt_path is not None:
        vrt_path = _safe_relative_path(root, raw_vrt_path, field="VRT path")
        if not vrt_path.is_file():
            raise TemporalSeriesOpenError(
                "Manifest VRT is missing.",
                code="temporal_series_vrt_missing",
                details={"path": str(vrt_path)},
            )

    if validate_layers:
        for index, layer_path in enumerate(layer_paths):
            actual_dtype, actual_georef = _layer_metadata(layer_path)
            if not _same_layer_metadata(actual_dtype, actual_georef, dtype, georef):
                raise TemporalSeriesOpenError(
                    "Temporal layer metadata does not match the manifest.",
                    code="temporal_series_layer_metadata_mismatch",
                    details={"index": index, "path": str(layer_path)},
                )

    return TemporalGeoTiffSeries(
        root=root,
        manifest=manifest,
        georef=georef,
        dtype=dtype,
        times=times_array,
        layer_paths=tuple(layer_paths),
        vrt_path=vrt_path,
        layer_cache_bytes=cache_bytes,
        max_open_datasets=open_limit,
    )
