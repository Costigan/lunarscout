from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import (
    MapAlgebraExpressionError,
    MapAlgebraStorageError,
    OperationCancelledError,
    OutputExistsError,
)
from ..raster import _validate_nodata_representable
from ._model import RasterExpression
from ._planner import (
    _GEOTIFF_WRITE_OPTIONS,
    _build_journal_identity,
    plan_expression,
)
from ._windows import SourceWindowCache, enumerate_windows
from ._windowed import (
    CancellationCheck,
    ProgressCallback,
    _execute_window,
)

_JOURNAL_FORMAT_VERSION = 2
_DEFAULT_CHECKPOINT_INTERVAL = 16
_JOURNAL_IDENTITY_TAG = "LUNARSCOUT_JOURNAL_IDENTITY"


def _staging_tiff_path(output_path: Path) -> Path:
    return output_path.with_name(
        "." + output_path.name + ".lunarscout-partial.tif"
    )


def _staging_journal_path(output_path: Path) -> Path:
    return output_path.with_name(
        "." + output_path.name + ".lunarscout-partial.journal.json"
    )


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def _staging_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(
        "." + output_path.name + ".lunarscout-partial.manifest.json"
    )


def _rollback_interrupted_swap(
    current: Path,
    backup: Path,
    staging: Path,
) -> None:
    """Restore a deterministic backup left by an interrupted publication."""
    if not backup.exists():
        return
    if current.exists():
        if staging.exists():
            raise MapAlgebraStorageError(
                f"Cannot recover interrupted publication for {current}.",
                code="map_algebra_ambiguous_publication_state",
                details={
                    "current": str(current),
                    "backup": str(backup),
                    "staging": str(staging),
                },
            )
        os.replace(current, staging)
    os.replace(backup, current)


def _build_restart_identity(
    expression: RasterExpression,
    output_dtype: np.dtype[Any],
    fill: int | float,
) -> dict[str, Any]:
    grid = expression.grid
    return {
        "scientific_identity": expression.scientific_identity(),
        "output_dtype": str(output_dtype),
        "invalid_fill": repr(fill),
        "grid_width": grid.width if grid else None,
        "grid_height": grid.height if grid else None,
    }


def _atomic_json(target: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        prefix="." + target.name + ".",
        suffix=".tmp",
        dir=target.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        try:
            view = memoryview(content)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("Unable to write staged JSON content.")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    dir_fd: int | None = None
    try:
        dir_fd = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _journal_identity_matches(
    journal: dict[str, Any],
    identity: str,
) -> bool:
    return journal.get("journal_format") == _JOURNAL_FORMAT_VERSION and journal.get(
        "identity"
    ) == identity


def _validate_journal_completed(
    journal: dict[str, Any],
    plan: Any,
) -> int:
    completed = journal.get("completed_windows")
    if (
        not isinstance(completed, int)
        or isinstance(completed, bool)
        or completed < 0
        or completed > plan.total_windows
    ):
        return 0
    return completed


def _write_journal(
    journal_path: Path,
    journal_identity: str,
    completed_windows: int,
    total_windows: int,
) -> None:
    _atomic_json(journal_path, {
        "journal_format": _JOURNAL_FORMAT_VERSION,
        "identity": journal_identity,
        "layout": "row_major_contiguous_prefix",
        "completed_windows": completed_windows,
        "total_windows": total_windows,
    })


def _open_staged_tiff(path: Path) -> Any:
    import rasterio
    return rasterio.open(path, "r+")


def _create_staged_tiff(
    path: Path,
    width: int,
    height: int,
    output_dtype: np.dtype[Any],
    georef: Any,
    nodata: int | float,
    journal_identity: str,
) -> Any:
    import rasterio
    from rasterio.transform import Affine

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": output_dtype,
        "crs": georef.projection_wkt,
        "transform": Affine.from_gdal(*georef.affine_transform),
        "tiled": _GEOTIFF_WRITE_OPTIONS["tiled"],
        "blockxsize": _GEOTIFF_WRITE_OPTIONS["block_width"],
        "blockysize": _GEOTIFF_WRITE_OPTIONS["block_height"],
        "compress": _GEOTIFF_WRITE_OPTIONS["compression"],
        "predictor": (
            _GEOTIFF_WRITE_OPTIONS["float_predictor"]
            if np.issubdtype(output_dtype, np.floating)
            else _GEOTIFF_WRITE_OPTIONS["integer_predictor"]
        ),
        "BIGTIFF": _GEOTIFF_WRITE_OPTIONS["bigtiff"],
        "nodata": nodata,
    }

    path.unlink(missing_ok=True)
    ds = rasterio.open(path, "w", **profile)
    ds.update_tags(**{_JOURNAL_IDENTITY_TAG: journal_identity})
    if np.issubdtype(output_dtype, np.integer):
        ds.update_tags(1, **{LUNARSCOUT_NODATA_VALUE: str(int(nodata))})
    return ds


def _same_nodata(actual: int | float | None, expected: int | float) -> bool:
    if actual is None:
        return False
    if isinstance(expected, float) and np.isnan(expected):
        return bool(np.isnan(actual))
    return actual == expected


def _staged_tiff_matches(
    ds: Any,
    *,
    grid: Any,
    output_dtype: np.dtype[Any],
    nodata: int | float,
    journal_identity: str,
) -> bool:
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    expected_crs = CRS.from_wkt(grid.projection_wkt)
    expected_transform = Affine.from_gdal(*grid.affine_transform)
    image_structure = ds.tags(ns="IMAGE_STRUCTURE")
    expected_predictor = (
        _GEOTIFF_WRITE_OPTIONS["float_predictor"]
        if np.issubdtype(output_dtype, np.floating)
        else _GEOTIFF_WRITE_OPTIONS["integer_predictor"]
    )
    return bool(
        ds.driver == _GEOTIFF_WRITE_OPTIONS["driver"]
        and ds.count == 1
        and ds.width == grid.width
        and ds.height == grid.height
        and np.dtype(ds.dtypes[0]) == output_dtype
        and ds.crs == expected_crs
        and ds.transform.almost_equals(expected_transform)
        and _same_nodata(ds.nodata, nodata)
        and ds.tags().get(_JOURNAL_IDENTITY_TAG) == journal_identity
        and ds.compression is not None
        and ds.compression.value.lower() == _GEOTIFF_WRITE_OPTIONS["compression"]
        and image_structure.get("PREDICTOR") == str(expected_predictor)
        and ds.block_shapes == [(
            _GEOTIFF_WRITE_OPTIONS["block_height"],
            _GEOTIFF_WRITE_OPTIONS["block_width"],
        )]
    )


LUNARSCOUT_NODATA_VALUE = "LUNARSCOUT_NODATA_VALUE"


def write(
    path: str | Path,
    expression: RasterExpression,
    *,
    overwrite: bool = False,
    start_fresh: bool = False,
    dtype: np.dtype[Any] | str | None = None,
    invalid_value: int | float | None = None,
    window_width: int = 128,
    window_height: int = 128,
    progress_callback: ProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
    checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
) -> Path:
    """Evaluate *expression* in bounded windows and publish a GeoTIFF.

    ``progress_callback`` receives ``(completed, total, window_index)`` after
    each newly completed window is written. ``cancellation_requested`` is
    polled before execution and at window boundaries. Restart state is durably
    checkpointed every ``checkpoint_interval`` windows; matching state is
    resumed automatically unless ``start_fresh`` is true.
    """
    output_path = Path(path).expanduser().resolve()

    # ---------- preflight validation ----------
    for name, callback in (
        ("progress_callback", progress_callback),
        ("cancellation_requested", cancellation_requested),
    ):
        if callback is not None and not callable(callback):
            raise MapAlgebraExpressionError(
                f"{name} must be callable or None.",
                code="map_algebra_invalid_lifecycle_callback",
                details={"argument": name},
            )
    if (
        not isinstance(checkpoint_interval, int)
        or isinstance(checkpoint_interval, bool)
        or checkpoint_interval < 1
    ):
        raise MapAlgebraExpressionError(
            "checkpoint_interval must be a positive integer.",
            code="map_algebra_invalid_checkpoint_interval",
            details={"checkpoint_interval": checkpoint_interval},
        )

    if expression.grid is None:
        raise MapAlgebraExpressionError(
            "Cannot write an expression without an inferred output grid.",
            code="map_algebra_missing_output_grid",
        )

    if expression.dtype is None:
        raise MapAlgebraExpressionError(
            "Cannot write an expression without an inferred output dtype.",
            code="map_algebra_missing_output_dtype",
        )

    output_dtype = np.dtype(dtype) if dtype is not None else expression.dtype
    if output_dtype == np.dtype(np.bool_):
        output_dtype = np.dtype(np.uint8)

    if output_dtype not in {
        np.dtype(np.uint8), np.dtype(np.int8),
        np.dtype(np.uint16), np.dtype(np.int16),
        np.dtype(np.uint32), np.dtype(np.int32),
        np.dtype(np.uint64), np.dtype(np.int64),
        np.dtype(np.float32), np.dtype(np.float64),
    }:
        raise MapAlgebraStorageError(
            f"Unsupported output dtype: {output_dtype}",
            code="map_algebra_unsupported_output_dtype",
            details={"dtype": str(output_dtype)},
        )

    if invalid_value is not None:
        tmp_val: int | float = (
            int(invalid_value)
            if isinstance(invalid_value, (int, np.integer))
            else float(invalid_value)
        )
        tmp_val = _validate_nodata_representable(tmp_val, output_dtype)  # type: ignore[assignment]
    elif np.issubdtype(output_dtype, np.floating):
        tmp_val = float(np.nan)
    else:
        tmp_val = 0
    fill_val: int | float = _validate_nodata_representable(tmp_val, output_dtype)  # type: ignore[assignment]

    restart_id = _build_restart_identity(expression, output_dtype, fill_val)
    journal_id_str = _build_journal_identity(
        expression, output_dtype, fill_val,
        window_width, window_height, checkpoint_interval,
    )

    # Plan and validate the complete graph before modifying any existing output.
    plan = plan_expression(
        expression,
        window_width=window_width,
        window_height=window_height,
    )
    if plan.grid is None:
        raise MapAlgebraExpressionError(
            "Plan has no output grid.",
            code="map_algebra_missing_output_grid",
        )
    if output_dtype != plan.output_dtype:
        _validate_output_dtype_conversion(output_dtype, plan)
        expression = _make_cast_expression(expression, output_dtype)
        plan = plan_expression(
            expression,
            window_width=window_width,
            window_height=window_height,
        )
        journal_id_str = _build_journal_identity(
            expression, output_dtype, fill_val,
            window_width, window_height, checkpoint_interval,
        )

    grid = plan.grid
    if grid is None:
        raise MapAlgebraExpressionError(
            "Plan has no output grid.",
            code="map_algebra_missing_output_grid",
        )

    # ---------- staging paths ----------
    staging_tiff = _staging_tiff_path(output_path)
    staging_journal = _staging_journal_path(output_path)
    manifest_path = _manifest_path(output_path)
    staging_manifest = _staging_manifest_path(output_path)
    backup_tiff = staging_tiff.with_name(staging_tiff.name + ".previous")
    backup_manifest = staging_manifest.with_name(staging_manifest.name + ".previous")

    # ---------- handle start_fresh ----------
    if start_fresh:
        staging_tiff.unlink(missing_ok=True)
        staging_journal.unlink(missing_ok=True)
        staging_manifest.unlink(missing_ok=True)
        backup_tiff.unlink(missing_ok=True)
        backup_manifest.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
    else:
        _rollback_interrupted_swap(output_path, backup_tiff, staging_tiff)
        _rollback_interrupted_swap(manifest_path, backup_manifest, staging_manifest)

    # ---------- check complete output ----------
    existing_manifest = _load_json(manifest_path)
    if existing_manifest is not None and _manifest_matches(existing_manifest, restart_id):
        if output_path.exists():
            return output_path

    # ---------- check overwrite ----------
    if output_path.exists() and not overwrite and not start_fresh:
        raise OutputExistsError(
            f"Output already exists: {output_path}. Use overwrite=True.",
            code="map_algebra_output_exists",
            details={"path": str(output_path)},
        )

    # ---------- load journal for resume ----------
    total = plan.total_windows
    existing_journal = _load_json(staging_journal)
    journal_matches = bool(
        existing_journal is not None
        and _journal_identity_matches(existing_journal, journal_id_str)
        and existing_journal.get("layout") == "row_major_contiguous_prefix"
        and existing_journal.get("total_windows") == total
    )
    completed_windows = (
        _validate_journal_completed(existing_journal, plan)
        if journal_matches and existing_journal is not None
        else 0
    )

    # ---------- cancellation check before execution ----------
    if cancellation_requested is not None and cancellation_requested():
        raise OperationCancelledError(
            "Windowed write cancelled before execution began.",
            code="map_algebra_cancelled",
            details={
                "completed_windows": completed_windows,
                "total_windows": total,
            },
        )

    # A staged TIFF and journal are one restart unit. Never trust either alone.
    if not journal_matches or not staging_tiff.exists():
        staging_tiff.unlink(missing_ok=True)
        staging_journal.unlink(missing_ok=True)
        completed_windows = 0

    # ---------- open or create staged TIFF ----------
    import rasterio

    ds: Any | None = None
    if staging_tiff.exists():
        try:
            ds = rasterio.open(staging_tiff, "r+")
        except rasterio.errors.RasterioIOError:
            staging_tiff.unlink(missing_ok=True)
            staging_journal.unlink(missing_ok=True)
            completed_windows = 0
        if ds is not None:
            if not _staged_tiff_matches(
                ds,
                grid=grid,
                output_dtype=output_dtype,
                nodata=fill_val,
                journal_identity=journal_id_str,
            ):
                ds.close()
                ds = None
                staging_tiff.unlink(missing_ok=True)
                staging_journal.unlink(missing_ok=True)
                completed_windows = 0

    if ds is None:
        ds = _create_staged_tiff(
            staging_tiff,
            grid.width,
            grid.height,
            output_dtype,
            grid,
            nodata=fill_val,
            journal_identity=journal_id_str,
        )

    # ---------- stage management ----------
    backup_tiff.unlink(missing_ok=True)
    backup_manifest.unlink(missing_ok=True)

    def _check_cancelled() -> bool:
        return bool(cancellation_requested and cancellation_requested())

    try:
        n_completed = completed_windows
        checkpointed_completed = completed_windows
        checkpoint_count = 0
        executed_any = False

        def _checkpoint(*, reopen: bool) -> None:
            nonlocal ds, checkpoint_count, checkpointed_completed
            if ds is not None:
                ds.close()
                ds = None
            _write_journal(
                staging_journal,
                journal_id_str,
                n_completed,
                total,
            )
            checkpointed_completed = n_completed
            checkpoint_count = 0
            if reopen:
                ds = _open_staged_tiff(staging_tiff)

        with SourceWindowCache(
            max_datasets=16,
            max_windows=max(1, min(64, plan.n_sources)),
        ) as cache:
            for idx, x0, y0, width, height, _ncols in enumerate_windows(
                grid.width,
                grid.height,
                plan.window_width,
                plan.window_height,
            ):
                if idx < n_completed:
                    continue

                if _check_cancelled():
                    if n_completed > checkpointed_completed:
                        _checkpoint(reopen=False)
                    raise OperationCancelledError(
                        f"Windowed write cancelled before window "
                        f"{n_completed + 1} of {total}.",
                        code="map_algebra_cancelled",
                        details={
                            "completed_windows": n_completed,
                            "total_windows": total,
                            "window_index": idx,
                        },
                    )

                values, valid = _execute_window(
                    plan, cache, idx, x0, y0, width, height,
                )
                _write_window(ds, idx, x0, y0, width, height, values, valid,
                              fill_val, output_dtype)
                n_completed += 1
                checkpoint_count += 1
                cache.discard_window(idx)
                executed_any = True

                is_last = n_completed == total
                should_checkpoint = (
                    checkpoint_count >= checkpoint_interval
                    or is_last
                )

                if should_checkpoint:
                    _checkpoint(reopen=not is_last)

                if progress_callback is not None:
                    progress_callback(n_completed, total, idx)

        if n_completed > checkpointed_completed:
            _checkpoint(reopen=False)
        if ds is not None:
            ds.close()
            ds = None

        if n_completed != total:
            raise MapAlgebraStorageError(
                f"Write completed with {n_completed} of {total} windows.",
                code="map_algebra_incomplete_write",
                details={
                    "completed_windows": n_completed,
                    "total_windows": total,
                },
            )

        # A fully checkpointed resume has no newly completed window index.
        if progress_callback is not None and not executed_any:
            progress_callback(total, total, -1)

        # ---------- publish ----------
        _write_manifest_atomic(staging_manifest, restart_id)
        output_backed_up = False
        manifest_backed_up = False
        output_published = False
        manifest_published = False
        try:
            if output_path.exists():
                os.replace(output_path, backup_tiff)
                output_backed_up = True
            if manifest_path.exists():
                os.replace(manifest_path, backup_manifest)
                manifest_backed_up = True
            os.replace(staging_tiff, output_path)
            output_published = True
            os.replace(staging_manifest, manifest_path)
            manifest_published = True
        except Exception:
            if manifest_published and manifest_path.exists():
                os.replace(manifest_path, staging_manifest)
            if output_published and output_path.exists():
                os.replace(output_path, staging_tiff)
            if manifest_backed_up and backup_manifest.exists():
                os.replace(backup_manifest, manifest_path)
            if output_backed_up and backup_tiff.exists():
                os.replace(backup_tiff, output_path)
            raise

        backup_tiff.unlink(missing_ok=True)
        backup_manifest.unlink(missing_ok=True)
        staging_journal.unlink(missing_ok=True)

    except Exception:
        if ds is not None:
            ds.close()
            ds = None
        raise
    finally:
        if ds is not None:
            ds.close()

    return output_path


def _write_window(
    ds: Any,
    idx: int,
    x0: int,
    y0: int,
    w: int,
    h: int,
    values: np.ndarray[Any, Any],
    valid: np.ndarray[Any, Any],
    fill: int | float,
    output_dtype: np.dtype[Any],
) -> None:
    window = ((y0, y0 + h), (x0, x0 + w))
    block_values = values.astype(output_dtype, copy=False)
    if not np.all(valid):
        block_values = block_values.copy()
        block_values[~valid] = fill
    ds.write(block_values, 1, window=window)
    ds.write_mask(valid.astype(np.uint8) * 255, window=window)


def _make_cast_expression(expression: RasterExpression, target_dtype: np.dtype[Any]) -> RasterExpression:
    from ._model import _make_expr_node
    return _make_expr_node(
        "local.cast",
        (expression, target_dtype),
        grid=expression._inferred_grid,
        dtype=target_dtype,
        units=expression._inferred_units,
        params={"casting": "unsafe"},
    )


def _validate_output_dtype_conversion(
    output_dtype: np.dtype[Any],
    plan: Any,
) -> None:
    src_dtype = plan.output_dtype
    if src_dtype is None or src_dtype == output_dtype:
        return

    ok = bool(np.can_cast(src_dtype, output_dtype, casting="safe"))
    if not ok:
        raise MapAlgebraStorageError(
            f"Cannot safely convert output from {src_dtype} to {output_dtype}. "
            f"Use ma.cast() explicitly before writing.",
            code="map_algebra_unsafe_output_cast",
            details={"source_dtype": str(src_dtype), "target_dtype": str(output_dtype)},
        )


def _manifest_matches(
    existing: dict[str, Any],
    restart_id: dict[str, Any],
) -> bool:
    for key in ("scientific_identity", "output_dtype", "invalid_fill",
                "grid_width", "grid_height"):
        if existing.get(key) != restart_id.get(key):
            return False
    return True


def _write_manifest_atomic(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    _atomic_json(manifest_path, manifest)
