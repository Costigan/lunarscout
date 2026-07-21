from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import (
    MapAlgebraExpressionError,
    MapAlgebraStorageError,
    OutputExistsError,
)
from ..georeference import GeoReference
from ..raster import Raster, _validate_nodata_representable
from ._model import RasterExpression
from .expression import compute


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def _check_restart(
    output_path: Path,
    expression: RasterExpression,
    *,
    overwrite: bool,
    start_fresh: bool,
) -> tuple[bool, dict[str, Any] | None]:
    manifest_file = _manifest_path(output_path)

    if start_fresh:
        manifest_file.unlink(missing_ok=True)
        return False, None

    if output_path.exists():
        if not overwrite:
            raise OutputExistsError(
                f"Output already exists: {output_path}. Use overwrite=True.",
                code="map_algebra_output_exists",
                details={"path": str(output_path)},
            )
        manifest_file.unlink(missing_ok=True)
        return False, None

    if manifest_file.exists():
        try:
            existing = json.loads(manifest_file.read_text())
            stored_id = existing.get("scientific_identity", "")
            current_id = expression.scientific_identity()
            if stored_id == current_id and output_path.exists():
                return True, existing
            else:
                manifest_file.unlink(missing_ok=True)
        except (json.JSONDecodeError, KeyError):
            manifest_file.unlink(missing_ok=True)

    return False, None


def _write_manifest(
    output_path: Path,
    expression: RasterExpression,
    georef: GeoReference,
    dtype: np.dtype[Any],
    invalid_value: Any,
    backend: str,
) -> None:
    manifest = {
        "scientific_identity": expression.scientific_identity(),
        "expression_json": expression.to_json(),
        "grid": {
            "width": georef.width,
            "height": georef.height,
            "crs": georef.projection_wkt,
            "affine": [float(v) for v in georef.affine_transform],
        },
        "dtype": str(dtype),
        "invalid_value": repr(invalid_value),
        "backend": backend,
        "lunarscout_version": _get_version(),
    }
    manifest_file = _manifest_path(output_path)
    manifest_data = json.dumps(manifest, sort_keys=True, indent=2)
    tmp = manifest_file.with_suffix(manifest_file.suffix + ".tmp")
    tmp.write_text(manifest_data)
    os.replace(tmp, manifest_file)


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("lunarscout")
    except Exception:
        return "0+unknown"


def write(
    path: str | Path,
    expression: RasterExpression,
    *,
    overwrite: bool = False,
    start_fresh: bool = False,
    dtype: np.dtype[Any] | str | None = None,
    invalid_value: int | float | None = None,
    backend: str = "cpu",
) -> Path:
    output_path = Path(path).expanduser().resolve()

    should_skip, _ = _check_restart(
        output_path, expression, overwrite=overwrite, start_fresh=start_fresh,
    )
    if should_skip:
        return output_path

    raster = compute(expression)
    if raster.georef is None:
        raise MapAlgebraExpressionError(
            "Cannot write an expression without an inferred output grid.",
            code="map_algebra_missing_output_grid",
        )

    output_dtype = np.dtype(dtype) if dtype is not None else raster.dtype
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

    if output_dtype != raster.dtype:
        from .local import cast
        raster = cast(raster, output_dtype, casting="unsafe")

    values = raster.filled(fill) if fill is not None else raster.values.copy()
    georef = raster.georef.with_nodata(fill)

    from ..geotiff import write_geotiff as _write
    result = _write(output_path, values, georef, overwrite=True)

    _write_manifest(output_path, expression, georef, output_dtype, fill, backend)
    return result
