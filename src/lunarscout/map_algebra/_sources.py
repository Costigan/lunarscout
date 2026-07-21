from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geotiff import read_geotiff as _read_geotiff
from ..raster import Raster
from ._model import RasterExpression, _new_node_id


def _source_descriptor(path: Path, band: int, georef, dtype, nodata) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "band": band,
        "width": georef.width,
        "height": georef.height,
        "dtype": str(dtype),
        "nodata": repr(nodata),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def source(path: str | Path, *, band: int = 1, units: str | None = None) -> RasterExpression:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    values, georef = _read_geotiff(path, band=band)
    if georef is None:
        raise ValueError(f"GeoTIFF is not georeferenced: {path}")

    params = _source_descriptor(path, band, georef, values.dtype, georef.nodata)
    return RasterExpression(
        _node_id=_new_node_id(),
        _operation_id="source",
        _operands=(),
        _params=params,
        _inferred_grid=georef,
        _inferred_dtype=np.dtype(values.dtype),
        _inferred_units=units,
        _halo=0,
    )


def constant(raster: Raster) -> RasterExpression:
    return RasterExpression(
        _node_id=_new_node_id(),
        _operation_id="constant",
        _operands=(raster,),
        _params={"name": raster.name or ""},
        _inferred_grid=raster.georef,
        _inferred_dtype=raster.dtype,
        _inferred_units=raster.units,
        _halo=0,
    )
