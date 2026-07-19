"""Private resumable tiled-GeoTIFF storage for downstream horizon products."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from rasterio import Env, open as open_raster
from rasterio.transform import Affine
from rasterio.windows import Window

from lunarscout.georeference import GeoReference

from .file_format import PATCH_SIZE


PRODUCT_MANIFEST_SCHEMA = "lunarscout-numba-product-job-v1"
PRODUCT_ALGORITHM_VERSION = "phase6b-v1"
TIMESTAMPS_TAG = "LUNARSCOUT_TIMESTAMPS_UTC"
TIMESTAMP_TAG = "TIMESTAMP_UTC"
COMPUTE_BACKENDS_TAG = "LUNARSCOUT_COMPUTE_BACKENDS"
_SUPPORTED_DTYPES = frozenset(
    np.dtype(value)
    for value in (
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float32,
        np.float64,
    )
)


class ProductStoreError(RuntimeError):
    """A private staged-product contract or durability failure."""


class IncompatibleProductJobError(ProductStoreError):
    """An existing staged product belongs to a different calculation."""


@dataclass(frozen=True, slots=True)
class ProductWriteTimings:
    """Durability-stage timings for one patch write."""

    tiff_write_close_seconds: float
    tiff_synchronize_seconds: float
    journal_persistence_seconds: float


@dataclass(frozen=True, slots=True)
class ProductBatchWriteTimings:
    """TIFF-write timing plus any durable checkpoint triggered by that write."""

    tiff_write_seconds: float
    checkpoint: ProductCheckpointTimings | None


@dataclass(frozen=True, slots=True)
class ProductCheckpointTimings:
    """Close/sync/journal timings for one bounded durable checkpoint."""

    tiff_close_seconds: float
    tiff_synchronize_seconds: float
    journal_persistence_seconds: float
    completed_patch_keys: tuple[str, ...]


def _utc_timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 timestamp: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    utc = parsed.astimezone(timezone.utc)
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ProductJob:
    """Identity and storage contract for one resumable tiled product."""

    georef: GeoReference
    dtype: np.dtype[Any] | type[np.generic] | str
    band_count: int = 1
    timestamps_utc: Sequence[datetime | str] = ()
    band_metadata: Sequence[Mapping[str, Any]] = ()
    invalid_value: int | float = 0
    compression: str = "deflate"
    algorithm: str = "unspecified"
    configuration: Mapping[str, Any] | None = None
    horizon_inventory_identity: str = "unspecified"

    def manifest(self) -> dict[str, Any]:
        dtype = np.dtype(self.dtype)
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported GeoTIFF dtype: {dtype}")
        if (
            isinstance(self.band_count, bool)
            or not isinstance(self.band_count, Integral)
            or not 1 <= int(self.band_count) <= 65535
        ):
            raise ValueError("band_count must be between 1 and 65535")
        timestamps = tuple(_utc_timestamp(value) for value in self.timestamps_utc)
        if timestamps and len(timestamps) != int(self.band_count):
            raise ValueError("timestamp count must equal band_count")
        if self.band_metadata and len(self.band_metadata) != int(self.band_count):
            raise ValueError("band metadata count must equal band_count")
        band_metadata = (
            [dict(item) for item in self.band_metadata]
            if self.band_metadata
            else [{} for _ in range(int(self.band_count))]
        )
        try:
            _json_bytes(band_metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("band metadata must be JSON-serializable") from exc
        if not np.isfinite(self.invalid_value):
            raise ValueError("invalid_value must be finite")
        try:
            invalid = np.asarray(self.invalid_value, dtype=dtype).item()
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("invalid_value is not representable by the dtype") from exc
        if np.issubdtype(dtype, np.integer):
            if not np.isfinite(self.invalid_value) or float(self.invalid_value) != float(invalid):
                raise ValueError("invalid_value is not exactly representable by the dtype")
            invalid = int(invalid)
        else:
            if np.isfinite(self.invalid_value) and not np.isfinite(invalid):
                raise ValueError("invalid_value is not representable by the dtype")
            invalid = float(invalid)
        configuration = dict(self.configuration or {})
        try:
            _json_bytes(configuration)
        except (TypeError, ValueError) as exc:
            raise ValueError("configuration must be JSON-serializable") from exc
        return {
            "schema": PRODUCT_MANIFEST_SCHEMA,
            "algorithm_version": PRODUCT_ALGORITHM_VERSION,
            "algorithm": str(self.algorithm),
            "width": int(self.georef.width),
            "height": int(self.georef.height),
            "dtype": dtype.str,
            "band_count": int(self.band_count),
            "timestamps_utc": list(timestamps),
            "band_metadata": band_metadata,
            "invalid_value": invalid,
            "compression": str(self.compression).lower(),
            "projection_wkt": self.georef.projection_wkt,
            "affine_transform": [float(value) for value in self.georef.affine_transform],
            "configuration": configuration,
            "horizon_inventory_identity": str(self.horizon_inventory_identity),
            "tile_size": PATCH_SIZE,
            "interleave": "band",
        }


class ResumableTiledProduct:
    """Stable staged BigTIFF plus durable per-patch completion journal."""

    def __init__(
        self,
        output_path: str | Path,
        job: ProductJob,
        *,
        overwrite: bool = False,
        start_fresh: bool = False,
        backend: str | None = None,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.staging_path = self.output_path.with_name(
            f".{self.output_path.name}.lunarscout-partial.tif"
        )
        self.manifest_path = self.staging_path.with_suffix(".manifest.json")
        self.journal_path = self.staging_path.with_suffix(".journal.json")
        self._mask_sidecar = Path(f"{self.staging_path}.msk")
        self._manifest = job.manifest()
        self._fingerprint = hashlib.sha256(_json_bytes(self._manifest)).hexdigest()
        self._dtype = np.dtype(self._manifest["dtype"])
        self._overwrite = bool(overwrite)
        if backend not in (None, "cpu", "cuda"):
            raise ValueError("backend must be 'cpu', 'cuda', or None")
        self._backend = backend

        if self.output_path.exists() and not overwrite:
            raise ProductStoreError(f"output already exists: {self.output_path}")
        if start_fresh:
            self._remove_staging_artifacts()
        existing = (
            self.staging_path.exists(),
            self.manifest_path.exists(),
            self.journal_path.exists(),
        )
        if any(existing) and not all(existing):
            raise IncompatibleProductJobError(
                "staged product is incomplete; use start_fresh=True to discard it"
            )
        if all(existing):
            self._resume()
        else:
            self._create()

    @property
    def completed_patches(self) -> Mapping[str, str]:
        return dict(self._completed)

    @property
    def compute_backends(self) -> tuple[str, ...]:
        """Return backends with durable valid patches, in first-use order."""

        return tuple(self._compute_backends)

    @staticmethod
    def _patch_key(tile_y: int, tile_x: int) -> str:
        return f"{tile_y},{tile_x}"

    def _expected_patch_keys(self) -> set[str]:
        return {
            self._patch_key(y, x)
            for y in range(0, int(self._manifest["height"]), PATCH_SIZE)
            for x in range(0, int(self._manifest["width"]), PATCH_SIZE)
        }

    def _remove_staging_artifacts(self) -> None:
        for path in (
            self.staging_path,
            self.manifest_path,
            self.journal_path,
            self._mask_sidecar,
        ):
            path.unlink(missing_ok=True)

    def _profile(self) -> dict[str, Any]:
        predictor = 3 if np.issubdtype(self._dtype, np.floating) else 2
        return {
            "driver": "GTiff",
            "width": int(self._manifest["width"]),
            "height": int(self._manifest["height"]),
            "count": int(self._manifest["band_count"]),
            "dtype": self._dtype.name,
            "crs": self._manifest["projection_wkt"],
            "transform": Affine.from_gdal(*self._manifest["affine_transform"]),
            "tiled": True,
            "blockxsize": PATCH_SIZE,
            "blockysize": PATCH_SIZE,
            "compress": self._manifest["compression"],
            "predictor": predictor,
            "BIGTIFF": "YES",
            "SPARSE_OK": "TRUE",
            "interleave": "band",
        }

    def _create(self) -> None:
        with Env(GDAL_TIFF_INTERNAL_MASK=True):
            with open_raster(self.staging_path, "w", **self._profile()) as dataset:
                timestamps = self._manifest["timestamps_utc"]
                dataset.update_tags(
                    **{
                        "LUNARSCOUT_PRODUCT_SCHEMA": PRODUCT_MANIFEST_SCHEMA,
                        TIMESTAMPS_TAG: json.dumps(timestamps, separators=(",", ":")),
                    }
                )
                for band_index in range(1, int(self._manifest["band_count"]) + 1):
                    metadata = {
                        str(key): str(value)
                        for key, value in self._manifest["band_metadata"][
                            band_index - 1
                        ].items()
                    }
                    if timestamps:
                        metadata[TIMESTAMP_TAG] = timestamps[band_index - 1]
                    if metadata:
                        dataset.update_tags(band_index, **metadata)
        self._sync_staging()
        self._completed: dict[str, str] = {}
        self._patch_backends: dict[str, str] = {}
        self._compute_backends: list[str] = []
        self._write_manifest()
        self._write_journal()

    def _write_manifest(self) -> None:
        manifest = dict(self._manifest)
        manifest["compute_backends"] = list(self._compute_backends)
        _atomic_json(self.manifest_path, manifest)

    def _resume(self) -> None:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="ascii"))
            journal = json.loads(self.journal_path.read_text(encoding="ascii"))
        except (OSError, ValueError) as exc:
            raise IncompatibleProductJobError("unable to read staged product metadata") from exc
        manifest_backends = manifest.pop("compute_backends", [])
        if manifest != self._manifest:
            raise IncompatibleProductJobError(
                "staged product does not match the requested job"
            )
        if journal.get("manifest_sha256") != self._fingerprint:
            raise IncompatibleProductJobError("completion journal fingerprint is invalid")
        completed = journal.get("completed_patches")
        if not isinstance(completed, dict) or any(
            state not in ("valid", "invalid") for state in completed.values()
        ):
            raise IncompatibleProductJobError("completion journal is invalid")
        if not set(completed).issubset(self._expected_patch_keys()):
            raise IncompatibleProductJobError("completion journal contains unknown patches")
        patch_backends = journal.get("patch_backends", {})
        compute_backends = journal.get("compute_backends", [])
        if (
            not isinstance(patch_backends, dict)
            or any(value not in ("cpu", "cuda") for value in patch_backends.values())
            or not set(patch_backends).issubset(completed)
            or not isinstance(compute_backends, list)
            or any(value not in ("cpu", "cuda") for value in compute_backends)
            or len(set(compute_backends)) != len(compute_backends)
            or set(patch_backends.values()) != set(compute_backends)
            or not isinstance(manifest_backends, list)
            or any(value not in ("cpu", "cuda") for value in manifest_backends)
        ):
            raise IncompatibleProductJobError(
                "staged backend provenance is invalid"
            )
        try:
            with open_raster(self.staging_path) as dataset:
                if (
                    dataset.width != self._manifest["width"]
                    or dataset.height != self._manifest["height"]
                    or dataset.count != self._manifest["band_count"]
                    or np.dtype(dataset.dtypes[0]) != self._dtype
                ):
                    raise IncompatibleProductJobError(
                        "staged GeoTIFF does not match its manifest"
                    )
        except IncompatibleProductJobError:
            raise
        except Exception as exc:
            raise IncompatibleProductJobError("staged GeoTIFF is unreadable") from exc
        self._completed = dict(completed)
        self._patch_backends = dict(patch_backends)
        self._compute_backends = list(compute_backends)
        if manifest_backends != compute_backends:
            self._write_manifest()

    def _sync_staging(self) -> None:
        with self.staging_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if self._mask_sidecar.exists():
            with self._mask_sidecar.open("rb") as handle:
                os.fsync(handle.fileno())
        _sync_directory(self.staging_path.parent)

    def _write_journal(
        self,
        completed: Mapping[str, str] | None = None,
        patch_backends: Mapping[str, str] | None = None,
        compute_backends: Sequence[str] | None = None,
    ) -> None:
        completed_values = self._completed if completed is None else completed
        patch_backend_values = (
            self._patch_backends if patch_backends is None else patch_backends
        )
        compute_backend_values = (
            self._compute_backends if compute_backends is None else compute_backends
        )
        _atomic_json(
            self.journal_path,
            {
                "schema": PRODUCT_MANIFEST_SCHEMA,
                "manifest_sha256": self._fingerprint,
                "completed_patches": completed_values,
                "patch_backends": patch_backend_values,
                "compute_backends": list(compute_backend_values),
            },
        )

    def _commit_checkpoint_metadata(
        self,
        completed: Mapping[str, str],
        new_patch_backends: Mapping[str, str],
    ) -> None:
        patch_backends = dict(self._patch_backends)
        patch_backends.update(new_patch_backends)
        compute_backends = list(self._compute_backends)
        for value in new_patch_backends.values():
            if value not in compute_backends:
                compute_backends.append(value)
        self._write_journal(completed, patch_backends, compute_backends)
        self._completed = dict(completed)
        self._patch_backends = patch_backends
        self._compute_backends = compute_backends
        self._write_manifest()

    def is_complete(self, tile_y: int, tile_x: int) -> bool:
        return self._patch_key(tile_y, tile_x) in self._completed

    def _window(self, tile_y: int, tile_x: int) -> tuple[Window, int, int, str]:
        if (
            isinstance(tile_x, bool)
            or isinstance(tile_y, bool)
            or not isinstance(tile_x, int)
            or not isinstance(tile_y, int)
            or tile_x < 0
            or tile_y < 0
            or tile_x % PATCH_SIZE
            or tile_y % PATCH_SIZE
        ):
            raise ValueError("patch origins must be nonnegative multiples of 128")
        width = min(PATCH_SIZE, int(self._manifest["width"]) - tile_x)
        height = min(PATCH_SIZE, int(self._manifest["height"]) - tile_y)
        if width <= 0 or height <= 0:
            raise ValueError("patch origin is outside the output grid")
        return Window(tile_x, tile_y, width, height), width, height, self._patch_key(
            tile_y, tile_x
        )

    def write_patch(
        self,
        tile_y: int,
        tile_x: int,
        band_tiles: Iterable[npt.ArrayLike],
        *,
        valid: bool = True,
    ) -> None:
        """Durably write all bands for a patch, then journal it as one work unit."""
        self.write_patch_with_timings(
            tile_y, tile_x, band_tiles, valid=valid
        )

    def write_patch_with_timings(
        self,
        tile_y: int,
        tile_x: int,
        band_tiles: Iterable[npt.ArrayLike],
        *,
        valid: bool = True,
    ) -> ProductWriteTimings:
        """Durably write a patch and expose its compression/sync/journal costs."""
        window, width, height, key = self._window(tile_y, tile_x)
        if key in self._completed:
            return ProductWriteTimings(0.0, 0.0, 0.0)
        iterator = iter(band_tiles)
        invalid_tile = np.full(
            (height, width), self._manifest["invalid_value"], dtype=self._dtype
        )
        write_started = time.perf_counter()
        with Env(GDAL_TIFF_INTERNAL_MASK=True):
            with open_raster(self.staging_path, "r+") as dataset:
                for band_index in range(1, int(self._manifest["band_count"]) + 1):
                    if valid:
                        try:
                            source = next(iterator)
                        except StopIteration as exc:
                            raise ValueError("band_tiles has fewer entries than band_count") from exc
                        tile = np.asarray(source, dtype=self._dtype)
                        if tile.shape != (height, width):
                            raise ValueError(
                                f"band tile must have shape {(height, width)}, got {tile.shape}"
                            )
                        tile = np.ascontiguousarray(tile)
                    else:
                        tile = invalid_tile
                    dataset.write(tile, band_index, window=window)
                if valid:
                    try:
                        next(iterator)
                    except StopIteration:
                        pass
                    else:
                        raise ValueError("band_tiles has more entries than band_count")
                mask = np.full((height, width), 255 if valid else 0, dtype=np.uint8)
                dataset.write_mask(mask, window=window)
        tiff_write_close_seconds = time.perf_counter() - write_started
        sync_started = time.perf_counter()
        self._sync_staging()
        tiff_synchronize_seconds = time.perf_counter() - sync_started
        completed = dict(self._completed)
        completed[key] = "valid" if valid else "invalid"
        new_patch_backends = (
            {key: self._backend} if valid and self._backend is not None else {}
        )
        journal_started = time.perf_counter()
        self._commit_checkpoint_metadata(completed, new_patch_backends)
        journal_persistence_seconds = time.perf_counter() - journal_started
        return ProductWriteTimings(
            tiff_write_close_seconds=tiff_write_close_seconds,
            tiff_synchronize_seconds=tiff_synchronize_seconds,
            journal_persistence_seconds=journal_persistence_seconds,
        )

    def batch_writer(self, checkpoint_patch_count: int) -> ProductBatchWriter:
        """Open a writer that checkpoints no more than this many patches at once."""
        return ProductBatchWriter(self, checkpoint_patch_count)

    def write_invalid_patch(self, tile_y: int, tile_x: int) -> None:
        self.write_patch(tile_y, tile_x, (), valid=False)

    def finalize(self) -> Path:
        missing = self._expected_patch_keys() - set(self._completed)
        if missing:
            raise ProductStoreError(
                f"cannot finalize with {len(missing)} incomplete patches"
            )
        if self.output_path.exists() and not self._overwrite:
            raise ProductStoreError(f"output already exists: {self.output_path}")
        with Env(GDAL_TIFF_INTERNAL_MASK=True):
            with open_raster(self.staging_path, "r+") as dataset:
                dataset.update_tags(
                    **{
                        COMPUTE_BACKENDS_TAG: json.dumps(
                            self._compute_backends, separators=(",", ":")
                        )
                    }
                )
        self._sync_staging()
        if self._mask_sidecar.exists():
            raise ProductStoreError(
                "staged validity mask is external and cannot be published atomically"
            )
        os.replace(self.staging_path, self.output_path)
        _sync_directory(self.output_path.parent)
        self.manifest_path.unlink(missing_ok=True)
        self.journal_path.unlink(missing_ok=True)
        self._mask_sidecar.unlink(missing_ok=True)
        _sync_directory(self.output_path.parent)
        return self.output_path


class ProductBatchWriter:
    """Keep GDAL open within a bounded batch and journal only flushed patches."""

    def __init__(
        self,
        product: ResumableTiledProduct,
        checkpoint_patch_count: int,
    ) -> None:
        if (
            isinstance(checkpoint_patch_count, bool)
            or not isinstance(checkpoint_patch_count, int)
            or checkpoint_patch_count < 1
        ):
            raise ValueError("checkpoint_patch_count must be a positive integer")
        self._product = product
        self.checkpoint_patch_count = checkpoint_patch_count
        self._environment: Env | None = None
        self._dataset: Any | None = None
        self._pending: dict[str, str] = {}
        self._pending_backends: dict[str, str] = {}
        self._entered = False

    @property
    def pending_patch_count(self) -> int:
        return len(self._pending)

    def __enter__(self) -> ProductBatchWriter:
        if self._entered:
            raise ProductStoreError("batch writer is already open")
        self._environment = Env(GDAL_TIFF_INTERNAL_MASK=True)
        self._environment.__enter__()
        self._entered = True
        return self

    def _require_open(self) -> None:
        if not self._entered:
            raise ProductStoreError("batch writer is not open")

    def _open_dataset(self) -> Any:
        self._require_open()
        if self._dataset is None:
            self._dataset = open_raster(self._product.staging_path, "r+")
        return self._dataset

    def _close_dataset(self) -> float:
        started = time.perf_counter()
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
        return time.perf_counter() - started

    def write_patch_with_timings(
        self,
        tile_y: int,
        tile_x: int,
        band_tiles: Iterable[npt.ArrayLike],
        *,
        valid: bool = True,
    ) -> ProductBatchWriteTimings:
        """Write one tile, checkpointing when the configured bound is reached."""
        self._require_open()
        product = self._product
        window, width, height, key = product._window(tile_y, tile_x)
        if key in product._completed or key in self._pending:
            return ProductBatchWriteTimings(0.0, None)
        iterator = iter(band_tiles)
        invalid_tile = np.full(
            (height, width),
            product._manifest["invalid_value"],
            dtype=product._dtype,
        )
        write_started = time.perf_counter()
        dataset = self._open_dataset()
        for band_index in range(1, int(product._manifest["band_count"]) + 1):
            if valid:
                try:
                    source = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "band_tiles has fewer entries than band_count"
                    ) from exc
                tile = np.asarray(source, dtype=product._dtype)
                if tile.shape != (height, width):
                    raise ValueError(
                        f"band tile must have shape {(height, width)}, got {tile.shape}"
                    )
                tile = np.ascontiguousarray(tile)
            else:
                tile = invalid_tile
            dataset.write(tile, band_index, window=window)
        if valid:
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("band_tiles has more entries than band_count")
        mask = np.full((height, width), 255 if valid else 0, dtype=np.uint8)
        dataset.write_mask(mask, window=window)
        tiff_write_seconds = time.perf_counter() - write_started
        self._pending[key] = "valid" if valid else "invalid"
        if valid and product._backend is not None:
            self._pending_backends[key] = product._backend
        checkpoint = None
        if len(self._pending) >= self.checkpoint_patch_count:
            checkpoint = self.checkpoint_with_timings()
        return ProductBatchWriteTimings(tiff_write_seconds, checkpoint)

    def checkpoint_with_timings(self) -> ProductCheckpointTimings | None:
        """Close and sync TIFF data before atomically advancing the journal."""
        self._require_open()
        if not self._pending:
            return None
        tiff_close_seconds = self._close_dataset()
        sync_started = time.perf_counter()
        self._product._sync_staging()
        tiff_synchronize_seconds = time.perf_counter() - sync_started
        completed = dict(self._product._completed)
        completed.update(self._pending)
        journal_started = time.perf_counter()
        self._product._commit_checkpoint_metadata(
            completed, self._pending_backends
        )
        journal_persistence_seconds = time.perf_counter() - journal_started
        completed_patch_keys = tuple(self._pending)
        self._pending.clear()
        self._pending_backends.clear()
        return ProductCheckpointTimings(
            tiff_close_seconds=tiff_close_seconds,
            tiff_synchronize_seconds=tiff_synchronize_seconds,
            journal_persistence_seconds=journal_persistence_seconds,
            completed_patch_keys=completed_patch_keys,
        )

    def __exit__(self, exception_type, _exception, _traceback) -> None:
        try:
            if exception_type is None:
                self.checkpoint_with_timings()
            else:
                self._close_dataset()
        finally:
            self._entered = False
            if self._environment is not None:
                self._environment.__exit__(exception_type, _exception, _traceback)
                self._environment = None
