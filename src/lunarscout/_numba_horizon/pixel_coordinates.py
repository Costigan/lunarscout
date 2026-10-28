"""Pixel-centre coordinates in the DEM's CRS for tables that name pixels by raster row and column (private).

``x_m`` and ``y_m`` are the projected coordinates of the centre of the pixel (the DEM's transform is GDAL's
transform of the pixel corner, so the centre is half a pixel in, as ``DemGrid.pixel_to_crs``). They are
written with three decimals.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError

COORDINATE_COLUMNS = ("x_m", "y_m")


def pixel_centres(geo_transform: npt.ArrayLike, rows: npt.ArrayLike, columns: npt.ArrayLike):
    """``(x, y)`` float64 arrays: the CRS coordinates of the centres of pixels ``(rows, columns)``."""
    transform = np.asarray(geo_transform, dtype=np.float64)
    if transform.shape != (6,) or not np.isfinite(transform).all():
        raise InputError("the geotransform must be six finite numbers", code="coordinates_transform_invalid")
    column = np.asarray(columns, dtype=np.float64) + 0.5
    row = np.asarray(rows, dtype=np.float64) + 0.5
    return (transform[0] + transform[1] * column + transform[2] * row,
            transform[3] + transform[4] * column + transform[5] * row)


def format_coordinate(value: float) -> str:
    return f"{value:.3f}"


def dem_geo_transform(dem_path: str | Path) -> tuple[float, ...]:
    """GDAL's six-number transform of a DEM GeoTIFF (read from its header; no pixels are loaded)."""
    import rasterio

    from ..geotiff import _resolve_dem_path

    path = _resolve_dem_path(Path(dem_path).expanduser().resolve())
    with rasterio.open(path) as dataset:
        return tuple(float(v) for v in dataset.transform.to_gdal())


def add_coordinate_columns(
    table_path: str | Path, geo_transform: npt.ArrayLike, *, output_path: str | Path | None = None,
    row_column: str = "row", column_column: str = "column",
) -> int:
    """Add ``x_m`` and ``y_m`` to the CSV table at ``table_path`` from its ``row`` and ``column`` columns and
    return the number of rows written. The columns go right after ``column``; where the table already has
    them they are overwritten in place. The table is written to ``output_path`` (default: replaced in
    place) through a staging file, so a failure leaves the original untouched."""
    table_path = Path(table_path)
    output_path = table_path if output_path is None else Path(output_path)
    staging = output_path.with_name(output_path.name + ".partial")
    count = 0
    try:
        with table_path.open(encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            header = next(reader, None)
            if header is None or row_column not in header or column_column not in header:
                raise InputError(f"{table_path} has no '{row_column}' and '{column_column}' columns",
                                 code="coordinates_columns_missing", details={"header": header})
            row_at, column_at = header.index(row_column), header.index(column_column)
            if all(name in header for name in COORDINATE_COLUMNS):
                producers = [("x" if name == "x_m" else "y" if name == "y_m" else i) for i, name in enumerate(header)]
            else:                                  # none (or one of the two, which is dropped): add both after the later
                base = [i for i, name in enumerate(header) if name not in COORDINATE_COLUMNS]
                after = max(base.index(row_at), base.index(column_at)) + 1
                producers = base[:after] + ["x", "y"] + base[after:]
            names = [COORDINATE_COLUMNS[0] if p == "x" else COORDINATE_COLUMNS[1] if p == "y" else header[p]
                     for p in producers]
            transform = np.asarray(geo_transform, dtype=np.float64)
            with staging.open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target, lineterminator="\n")
                writer.writerow(names)
                for record in reader:
                    if not record:
                        continue
                    if record[row_at] == "" or record[column_at] == "":
                        x = y = ""
                    else:
                        cx, cy = pixel_centres(transform, int(record[row_at]), int(record[column_at]))
                        x, y = format_coordinate(float(cx)), format_coordinate(float(cy))
                    writer.writerow([x if p == "x" else y if p == "y" else record[p] for p in producers])
                    count += 1
        os.replace(staging, output_path)
    finally:
        staging.unlink(missing_ok=True)
    return count
