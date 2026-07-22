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
    OutputExistsError,
)
from ..raster import _validate_nodata_representable
from ._model import RasterExpression
from ._planner import plan_expression
from ._windows import SourceWindowCache
from ._windowed import execute_windowed


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


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


def _write_manifest_atomic(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix="." + manifest_path.name + ".",
        suffix=".tmp",
        dir=manifest_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
        os.replace(tmp_path, manifest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _create_staged_tiff(
    path: Path,
    width: int,
    height: int,
    output_dtype: np.dtype[Any],
    georef: Any,
    compress: bool,
    nodata: int | float,
) -> tuple[Any, Path]:
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
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "compress": "deflate" if compress else None,
        "predictor": 3 if np.issubdtype(output_dtype, np.floating) else 2,
        "BIGTIFF": "IF_SAFER",
        "nodata": nodata,
    }

    fd, tmp_name = tempfile.mkstemp(
        prefix="." + path.name + ".",
        suffix=".tmp.tif",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink()

    ds = rasterio.open(tmp_path, "w", **profile)
    return ds, tmp_path


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
) -> Path:
    output_path = Path(path).expanduser().resolve()

    # ---------- preflight validation ----------
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
        fill = _validate_nodata_representable(invalid_value, output_dtype)
    elif np.issubdtype(output_dtype, np.floating):
        fill = float(np.nan)
    else:
        fill = 0
    fill = _validate_nodata_representable(fill, output_dtype)

    restart_id = _build_restart_identity(expression, output_dtype, fill if fill is not None else 0)

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

    # ---------- check existing output ----------
    manifest_path = _manifest_path(output_path)

    if start_fresh:
        manifest_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    existing_manifest = _load_manifest(manifest_path)
    if existing_manifest is not None and _manifest_matches(existing_manifest, restart_id):
        if output_path.exists():
            return output_path

    if output_path.exists() and not overwrite and not start_fresh:
        raise OutputExistsError(
            f"Output already exists: {output_path}. Use overwrite=True.",
            code="map_algebra_output_exists",
            details={"path": str(output_path)},
        )

    grid = plan.grid
    if grid is None:
        raise MapAlgebraExpressionError("Plan has no output grid.", code="map_algebra_missing_output_grid")
    g_w = grid.width
    g_h = grid.height

    # ---------- windowed execution ----------
    staging_dir = Path(tempfile.mkdtemp(prefix=".lunarscout_write.", dir=output_path.parent))
    staging_tiff = staging_dir / output_path.name
    staging_manifest = staging_dir / manifest_path.name
    backup_tiff = staging_dir / (output_path.name + ".previous")
    backup_manifest = staging_dir / (manifest_path.name + ".previous")
    committed = False
    published_tiff = False
    published_manifest = False

    ds: Any | None = None
    ds_tmp_path: Path | None = None
    try:
        ds, ds_tmp_path = _create_staged_tiff(
            staging_tiff,
            g_w,
            g_h,
            output_dtype,
            grid,
            compress=True,
            nodata=fill,
        )

        def _write_block(
            idx: int, x0: int, y0: int, w: int, h: int,
            values: np.ndarray[Any, Any],
            valid: np.ndarray[Any, Any],
        ) -> None:
            window = ((y0, y0 + h), (x0, x0 + w))
            block_values = values.astype(output_dtype, copy=False)
            if not np.all(valid):
                block_values = block_values.copy()
                block_values[~valid] = fill
            ds.write(block_values, 1, window=window)
            ds.write_mask(valid.astype(np.uint8) * 255, window=window)

        with ds:
            with SourceWindowCache(
                max_datasets=16,
                max_windows=max(1, min(64, plan.n_sources)),
            ) as cache:
                execute_windowed(plan, cache, write_block=_write_block)
            if np.issubdtype(output_dtype, np.integer):
                ds.update_tags(1, LUNARSCOUT_NODATA_VALUE=str(int(fill)))
        ds = None

        # Move from temp to staging
        assert ds_tmp_path is not None
        os.replace(ds_tmp_path, staging_tiff)
        ds_tmp_path = None

        _write_manifest_atomic(staging_manifest, restart_id)

        if output_path.exists():
            os.replace(output_path, backup_tiff)
        if manifest_path.exists():
            os.replace(manifest_path, backup_manifest)

        os.replace(staging_tiff, output_path)
        published_tiff = True
        os.replace(staging_manifest, manifest_path)
        published_manifest = True
        committed = True
        backup_tiff.unlink(missing_ok=True)
        backup_manifest.unlink(missing_ok=True)
    finally:
        if ds is not None:
            ds.close()
        if ds_tmp_path is not None:
            ds_tmp_path.unlink(missing_ok=True)
        if not committed:
            if published_tiff:
                output_path.unlink(missing_ok=True)
            if published_manifest:
                manifest_path.unlink(missing_ok=True)
            if backup_tiff.exists():
                os.replace(backup_tiff, output_path)
            if backup_manifest.exists():
                os.replace(backup_manifest, manifest_path)
            staging_tiff.unlink(missing_ok=True)
            staging_manifest.unlink(missing_ok=True)
        try:
            staging_dir.rmdir()
        except OSError:
            pass

    return output_path


def _make_cast_expression(expression: RasterExpression, target_dtype: np.dtype[Any]) -> RasterExpression:
    """Create a cast node wrapping the expression."""
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


def _load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, KeyError):
        return None


def _manifest_matches(
    existing: dict[str, Any],
    restart_id: dict[str, Any],
) -> bool:
    for key in ("scientific_identity", "output_dtype", "invalid_fill",
                "grid_width", "grid_height"):
        if existing.get(key) != restart_id.get(key):
            return False
    return True
