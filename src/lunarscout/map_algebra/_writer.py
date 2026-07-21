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
from ..raster import Raster, _validate_nodata_representable
from ._model import RasterExpression
from .expression import compute


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


def _write_raster_with_mask(
    output_path: Path,
    raster: Raster,
    *,
    compress: bool = True,
) -> None:
    import rasterio
    from rasterio.transform import Affine

    georef = raster.georef
    values = raster.values
    valid = raster.valid

    profile = {
        "driver": "GTiff",
        "width": int(georef.width),
        "height": int(georef.height),
        "count": 1,
        "dtype": values.dtype,
        "crs": georef.projection_wkt,
        "transform": Affine.from_gdal(*georef.affine_transform),
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "compress": "deflate" if compress else None,
        "predictor": 3 if np.issubdtype(values.dtype, np.floating) else 2,
        "BIGTIFF": "IF_SAFER",
    }
    if georef.nodata is not None:
        profile["nodata"] = georef.nodata

    fd, tmp_name = tempfile.mkstemp(
        prefix="." + output_path.name + ".",
        suffix=".tmp.tif",
        dir=output_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink()

    try:
        with rasterio.open(tmp_path, "w", **profile) as ds:
            ds.write(values, 1)
            mask = valid.astype(np.uint8) * 255
            ds.write_mask(mask)
            if georef.nodata is not None and np.issubdtype(values.dtype, np.integer):
                ds.update_tags(1, **{"LUNARSCOUT_NODATA_VALUE": str(int(georef.nodata))})

        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write(
    path: str | Path,
    expression: RasterExpression,
    *,
    overwrite: bool = False,
    start_fresh: bool = False,
    dtype: np.dtype[Any] | str | None = None,
    invalid_value: int | float | None = None,
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

    # ---------- compute ----------
    raster = compute(expression)

    if output_dtype != raster.dtype:
        ok = False
        if np.issubdtype(raster.dtype, np.bool_) and output_dtype == np.dtype(np.uint8):
            ok = True
        elif np.issubdtype(raster.dtype, np.floating) and np.issubdtype(output_dtype, np.floating):
            ok = True
        elif np.issubdtype(raster.dtype, np.integer) and np.issubdtype(output_dtype, np.integer):
            if np.iinfo(output_dtype).max >= np.iinfo(raster.dtype).max:
                ok = True
        elif np.issubdtype(output_dtype, np.floating) and (
            np.issubdtype(raster.dtype, np.integer) or np.issubdtype(raster.dtype, np.bool_)
        ):
            ok = True
        if not ok:
            raise MapAlgebraStorageError(
                f"Cannot safely convert output from {raster.dtype} to {output_dtype}. "
                f"Use ma.cast() explicitly before writing.",
                code="map_algebra_unsafe_output_cast",
                details={"source_dtype": str(raster.dtype), "target_dtype": str(output_dtype)},
            )
        from .local import cast
        raster = cast(raster, output_dtype, casting="unsafe")

    fill = _validate_nodata_representable(fill, output_dtype)

    values = raster.filled(fill) if fill is not None else raster.values.copy()
    georef = raster.georef.with_nodata(fill)
    out_raster = Raster(
        values=values,
        georef=georef,
        valid=raster.valid,
        name=raster.name,
        units=raster.units,
    )

    # ---------- atomic two-file commit ----------
    staging_dir = Path(tempfile.mkdtemp(prefix=".lunarscout_write.", dir=output_path.parent))
    staging_tiff = staging_dir / output_path.name
    staging_manifest = staging_dir / manifest_path.name
    committed = False
    try:
        _write_raster_with_mask(staging_tiff, out_raster)
        _write_manifest_atomic(staging_manifest, restart_id)

        if output_path.exists():
            output_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

        os.replace(staging_tiff, output_path)
        os.replace(staging_manifest, manifest_path)
        committed = True
    finally:
        if not committed:
            staging_tiff.unlink(missing_ok=True)
            staging_manifest.unlink(missing_ok=True)
        try:
            staging_dir.rmdir()
        except OSError:
            pass

    return output_path


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
