from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..errors import MapAlgebraExpressionError, MapAlgebraUnitError, GeoTiffOpenError
from ..georeference import GeoReference
from ..raster import Raster
from ._model import RasterExpression, _make_expr_node, _next_id


def _read_source_metadata(path: Path, band: int) -> tuple[GeoReference, np.dtype[Any], int | float | None]:
    import rasterio as _rasterio

    if not path.exists() or not path.is_file():
        raise GeoTiffOpenError(
            f"Source file does not exist: {path}",
            code="geotiff_file_not_found",
            details={"path": str(path)},
        )
    try:
        dataset = _rasterio.open(path)
    except Exception as exc:
        raise GeoTiffOpenError(
            f"File is not a readable GeoTIFF: {path}",
            code="geotiff_unreadable_or_unsupported",
            details={"path": str(path), "error": str(exc)},
        ) from exc

    with dataset:
        if dataset.driver != "GTiff":
            raise GeoTiffOpenError(
                f"File is not a GeoTIFF: {path}",
                code="geotiff_unreadable_or_unsupported",
                details={"path": str(path), "driver": dataset.driver},
            )
        if band > int(dataset.count):
            raise GeoTiffOpenError(
                f"Band {band} is out of range.",
                code="geotiff_band_out_of_range",
                details={"band": band, "band_count": int(dataset.count)},
            )

        crs_wkt = dataset.crs.to_wkt() if dataset.crs is not None else ""
        if not crs_wkt:
            raise MapAlgebraExpressionError(
                f"GeoTIFF is not georeferenced: {path}",
                code="map_algebra_unreferenced_source",
                details={"path": str(path)},
            )

        transform = dataset.transform
        if transform.is_identity:
            raise MapAlgebraExpressionError(
                f"GeoTIFF has no valid geotransform: {path}",
                code="map_algebra_unreferenced_source",
                details={"path": str(path)},
            )

        from pyproj import CRS
        proj4 = str(CRS.from_wkt(crs_wkt).to_proj4() or "").strip()

        affine_tuple = tuple(float(v) for v in transform.to_gdal())
        dtype = np.dtype(dataset.dtypes[band - 1])
        nodata = dataset.nodatavals[band - 1]

        georef = GeoReference(
            projection_wkt=crs_wkt,
            projection_proj4=proj4,
            affine_transform=affine_tuple,  # type: ignore[arg-type]
            width=int(dataset.width),
            height=int(dataset.height),
            pixel_size_x=float(affine_tuple[1]),
            pixel_size_y=float(affine_tuple[5]),
            nodata=nodata,
        )
        return georef, dtype, nodata


def source(
    path: str | Path,
    *,
    band: int = 1,
    units: str | None = None,
    identity: Literal["stat", "sha256"] = "stat",
) -> RasterExpression:
    if not isinstance(band, int) or isinstance(band, bool) or band < 1:
        raise GeoTiffOpenError(
            "GeoTIFF band must be a positive one-based integer.",
            code="geotiff_band_out_of_range",
            details={"band": band},
        )
    if identity not in {"stat", "sha256"}:
        raise MapAlgebraExpressionError(
            "Source identity must be 'stat' or 'sha256'.",
            code="map_algebra_invalid_source_identity",
            details={"identity": identity},
        )
    if units is not None:
        if not isinstance(units, str) or not units.strip():
            raise MapAlgebraUnitError(
                "Source units must be a non-empty string or None.",
                code="map_algebra_invalid_units",
                details={"units": repr(units)},
            )
        units = units.strip()
    path = Path(path).expanduser().resolve()
    georef, dtype, nodata = _read_source_metadata(path, band)

    stat = path.stat()
    params: dict[str, Any] = {
        "path": str(path),
        "band": band,
        "width": georef.width,
        "height": georef.height,
        "dtype": str(dtype),
        "nodata": repr(nodata),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "identity_mode": identity,
    }

    import hashlib
    if identity == "sha256":
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                sha.update(chunk)
        params["sha256"] = sha.hexdigest()

    return _make_expr_node(
        "source", (),
        grid=georef, dtype=dtype, units=units,
        params=params,
    )


def constant(raster: Raster) -> RasterExpression:
    return _make_expr_node(
        "constant", (raster,),
        grid=raster.georef, dtype=raster.dtype,
        units=raster.units,
        params={"name": raster.name or ""},
    )
