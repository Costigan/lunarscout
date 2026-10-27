"""Permanently shadowed regions, one id each, with their sizes (private).

A PSR map marks the pixels that never see the Sun. This module turns it into two reusable products:

* an id raster: every connected PSR region has its own id (1, 2, 3, ... in raster-scan order of the
  region's first pixel) on every one of its pixels, and every other pixel is 0 (not a PSR);
* a region table, one row per id: the size in pixels and in square metres, the bounding box in pixels and
  the centroid in the map's coordinates.

The regions are connected the way the flood moves: eight neighbours by default. Only PSR pixels are
labelled. A map that does not look like a PSR map (other values than "not PSR" and "PSR", or most of the
raster marked) is refused instead of labelling the wrong pixels.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from ..regions import label_regions

PSR_VALUE = 255
NOT_PSR_VALUE = 0
#: A PSR map marks well under half of a polar raster (about 7% of the 70 degree south raster).
MAX_PSR_FRACTION = 0.5
REGION_COLUMNS = ("psr_id", "pixels", "area_m2", "row_min", "row_max", "col_min", "col_max", "x_m", "y_m")


def psr_mask(values: npt.ArrayLike, *, psr_value: int = PSR_VALUE, max_fraction: float = MAX_PSR_FRACTION):
    """The boolean PSR mask of a PSR map: true exactly where the map holds ``psr_value``.

    The map may hold only ``psr_value`` and ``NOT_PSR_VALUE`` (0), and ``psr_value`` must mark at most
    ``max_fraction`` of the pixels: an inverted or foreign map is refused.
    """
    array = np.asarray(values)
    if array.ndim != 2:
        raise InputError("a PSR map must be two-dimensional", code="psr_map_shape", details={"shape": list(array.shape)})
    if psr_value == NOT_PSR_VALUE:
        raise InputError("the PSR value cannot be the not-PSR value 0", code="psr_value_invalid")
    mask = array == psr_value
    other = ~(mask | (array == NOT_PSR_VALUE))
    if other.any():
        found = np.unique(array[other])[:8].tolist()
        raise InputError(
            f"a PSR map holds only 0 (not PSR) and {psr_value} (PSR); it also holds {found}",
            code="psr_map_values", details={"values": found})
    fraction = float(mask.mean()) if mask.size else 0.0
    if fraction > max_fraction:
        raise InputError(
            f"{100 * fraction:.0f}% of the map is marked PSR: is it inverted?", code="psr_map_inverted",
            details={"psr_fraction": fraction, "max_fraction": max_fraction})
    return mask


def label_psr(mask: npt.NDArray[np.bool_], connectivity: int = 8) -> tuple[npt.NDArray[np.int32], int]:
    """``(ids, count)``: ids 1..count on the PSR pixels of ``mask`` and 0 on every other pixel."""
    ids, _georef = label_regions(mask, None, connectivity=connectivity, nodata=None)
    ids = np.asarray(ids, dtype=np.int32)
    return ids, int(ids.max()) if ids.size else 0


@dataclass(frozen=True)
class PsrRegions:
    """The region table as parallel arrays, row ``i`` describing id ``ids[i]`` (ids are 1..count)."""

    psr_id: npt.NDArray[np.int32]
    pixels: npt.NDArray[np.int64]
    area_m2: npt.NDArray[np.float64]
    row_min: npt.NDArray[np.int32]
    row_max: npt.NDArray[np.int32]
    col_min: npt.NDArray[np.int32]
    col_max: npt.NDArray[np.int32]
    x_m: npt.NDArray[np.float64]
    y_m: npt.NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.psr_id)

    def pixels_by_id(self) -> npt.NDArray[np.int64]:
        """Pixels of every id, indexed by id (entry 0, "not a PSR", is 0): ``pixels_by_id()[ids_raster]``."""
        table = np.zeros(len(self) + 1, dtype=np.int64)
        table[self.psr_id] = self.pixels
        return table


def region_table(ids: npt.NDArray[np.int32], count: int, affine_transform) -> PsrRegions:
    """Sizes, bounding boxes and centroids of ids 1..``count`` of an id raster.

    ``affine_transform`` is the GDAL geotransform ``(x0, dx, 0, y0, 0, dy)``; a pixel's area is
    ``|dx * dy|`` and the centroid is the mean of the pixels' centres.
    """
    from scipy import ndimage

    x0, dx, rx, y0, ry, dy = (float(v) for v in affine_transform)
    if rx != 0.0 or ry != 0.0:
        raise InputError("a rotated raster has no simple pixel area", code="psr_rotated_grid")
    rows, columns = np.nonzero(ids)
    labels = ids[rows, columns]
    pixels = np.bincount(labels, minlength=count + 1)[1:].astype(np.int64)
    mean_row = np.bincount(labels, weights=rows, minlength=count + 1)[1:] / np.maximum(pixels, 1)
    mean_column = np.bincount(labels, weights=columns, minlength=count + 1)[1:] / np.maximum(pixels, 1)
    del rows, columns, labels
    boxes = ndimage.find_objects(ids, max_label=count)
    row_min = np.array([box[0].start for box in boxes], dtype=np.int32)
    row_max = np.array([box[0].stop - 1 for box in boxes], dtype=np.int32)
    col_min = np.array([box[1].start for box in boxes], dtype=np.int32)
    col_max = np.array([box[1].stop - 1 for box in boxes], dtype=np.int32)
    return PsrRegions(
        psr_id=np.arange(1, count + 1, dtype=np.int32), pixels=pixels, area_m2=pixels * abs(dx * dy),
        row_min=row_min, row_max=row_max, col_min=col_min, col_max=col_max,
        x_m=x0 + (mean_column + 0.5) * dx, y_m=y0 + (mean_row + 0.5) * dy)


def write_region_table(path: str | Path, regions: PsrRegions, *, overwrite: bool = False) -> Path:
    """The table as CSV (written to a staging name and renamed). Refuses an existing file unless ``overwrite``."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise InputError(f"{path} exists; pass overwrite=True to replace it", code="psr_output_exists",
                         details={"path": str(path)})
    staging = path.with_name(path.name + ".partial")
    try:
        with staging.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(REGION_COLUMNS)
            writer.writerows(zip(
                regions.psr_id.tolist(), regions.pixels.tolist(), (regions.area_m2).tolist(),
                regions.row_min.tolist(), regions.row_max.tolist(), regions.col_min.tolist(),
                regions.col_max.tolist(), np.round(regions.x_m, 2).tolist(), np.round(regions.y_m, 2).tolist()))
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)
    return path


def read_region_table(path: str | Path) -> PsrRegions:
    """The table written by :func:`write_region_table`."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if tuple(header or ()) != REGION_COLUMNS:
            raise InputError(f"{path} is not a PSR region table", code="psr_table_unreadable", details={"header": header})
        rows = list(reader)
    columns = list(zip(*rows)) if rows else [()] * len(REGION_COLUMNS)
    kinds = (np.int32, np.int64, np.float64, np.int32, np.int32, np.int32, np.int32, np.float64, np.float64)
    arrays = [np.array(column, dtype=kind) for column, kind in zip(columns, kinds)]
    regions = PsrRegions(*arrays)
    if len(regions) and not np.array_equal(regions.psr_id, np.arange(1, len(regions) + 1)):
        raise InputError(f"{path}: the ids are not 1..{len(regions)} in order", code="psr_table_unreadable")
    return regions


def make_psr_products(
    psr_path: str | Path, ids_path: str | Path, table_path: str | Path, *,
    psr_value: int = PSR_VALUE, connectivity: int = 8, overwrite: bool = False,
) -> tuple[int, int]:
    """Read a PSR map, write the id raster (int32 GeoTIFF on the same grid) and the region table.

    Returns ``(regions, psr_pixels)``. Nothing is written when an output exists and ``overwrite`` is false.
    """
    from ..geotiff import read_geotiff, write_geotiff

    for output in (ids_path, table_path):
        if Path(output).exists() and not overwrite:
            raise InputError(f"{output} exists; pass overwrite=True to replace it", code="psr_output_exists",
                             details={"path": str(output)})
    values, georef = read_geotiff(psr_path)
    if georef is None:
        raise InputError(f"{psr_path} has no georeferencing", code="psr_map_georeference")
    mask = psr_mask(values, psr_value=psr_value)
    del values
    ids, count = label_psr(mask, connectivity)
    psr_pixels = int(mask.sum())
    del mask
    regions = region_table(ids, count, georef.affine_transform)
    write_geotiff(ids_path, ids, georef.with_nodata(None), overwrite=overwrite)
    write_region_table(table_path, regions, overwrite=overwrite)
    return count, psr_pixels
