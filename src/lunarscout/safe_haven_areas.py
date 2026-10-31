"""Seed haven areas, month by month: the havens that matter for the chains still ahead in each month.

A chain is drawn each month as the part still ahead: its seeds that are not passed (a seed is passed once its own month is
over). Its **current start** in a month is the first of them, and the chain is drawn only while 2 or more seeds are left.
The area of a month is the current start of each chain drawn in the month, every haven of the month, a seed or not, whose
centre is within a radius of one of them (200 m by default; the DEM's pixel size makes it a number of pixels), and the
**lone seed** of each chain with only its final seed left, as that pixel alone. The havens are read from the monthly haven
images (``scripts/threshold_safe_havens.py``: a pixel is a haven when its hibernation is at most the survival limit). The
other seeds (isolated ones, passed ones, those still ahead that are not a current start) are not part of an area. The
rules for every monthly layer are in ``docs/safe-haven-monthly-products.md``.

Two products are written, from one computation:

* ``safe_haven_stage_4_seed_haven_areas/safe_haven_stage_4_seed_haven_areas_<YYYY-MM>.tif``, one GeoTIFF per month on
  the DEM's grid for a GIS: uint8, the pixel's hibernation in counts of ``hours_per_count`` hours (255 is NODATA and
  everything outside the area), coloured by a palette in the shadow-period classes of the viewer (white for none,
  yellow up to 12 h, cyan 12 to 24 h, violet above), tiled and DEFLATE-compressed with internal overviews in which each
  pixel is the minimum of its block, so a haven survives at every zoom;
* ``safe_haven_stage_4_seed_haven_areas.npz`` beside the chain table: the same pixels as lists (row, column, hours) by
  month with each month's start offset, the chain starts and lone seeds per month and the radius, for the viewer.

Both are written through staging, so a failure leaves earlier products untouched. A month without a haven image has its
chain starts alone.
"""

from __future__ import annotations

import os
import re
import shutil
import time as clock
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ._numba_horizon.chain_map_styles import HAVEN_SHADOW_CLASSES, haven_shadow_classes
from ._numba_horizon.pixel_coordinates import dem_header
from ._numba_horizon.safe_haven_chain_maps import build_chain_sets, current_chain_starts, seed_month_spans
from ._numba_horizon.safe_haven_chains import read_chain_table
from .errors import InputError, ProductStorageError

__all__ = ["AREA_RADIUS_METERS", "SeedHavenAreasReport", "generate_seed_haven_areas", "read_seed_areas"]

AREA_RADIUS_METERS = 200.0
AREA_DIRECTORY = "safe_haven_stage_4_seed_haven_areas"
AREA_IMAGE = "safe_haven_stage_4_seed_haven_areas_{month}.tif"
AREA_PIXELS = "safe_haven_stage_4_seed_haven_areas.npz"
HAVEN_MONTH_IMAGE = re.compile(r"^safe_haven_stage_1_haven_months_(\d{4}-\d{2})\.tif$")
NODATA = 255
STRIP = 1024
BLOCK = 512
MIN_OVERVIEW = 512

Progress = Callable[[str], None]


def _quiet(_message: str) -> None:
    return None


@dataclass
class SeedHavenAreasReport:
    """What :func:`generate_seed_haven_areas` wrote: the directory of the monthly GeoTIFFs, the pixel file, the number of
    months, the chain starts (current starts, counted in each month they are one), haven pixels and lone seeds in all
    months, how many months had a haven image, the radius used (m) and the seconds taken."""

    directory: str
    file: str
    months: int
    chain_starts: int
    pixels: int
    months_with_images: int
    radius_meters: float
    seconds: float = 0.0
    lone_seeds: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def areas_path(chains_directory: str | Path) -> Path:
    """The pixel file of the seed haven areas beside the chain table."""
    return Path(chains_directory) / AREA_PIXELS


def chain_starts(rows: list[dict]):
    """``(start_seed, start_month, lone_seed, lone_month, months)``: the 0-based table index of the current start of each
    chain in each month it is drawn in and the index in ``months`` of that month, the same for the lone seeds, and the month
    names (as ``seed_month_spans``). A seed is the current start of its chain in a month when it is the first one not passed
    and 2 or more seeds are left; it is a lone seed when it is the last seed and the only one left."""
    start_index, last_index, months = seed_month_spans(rows)
    _arrays, _edges, chain_set = build_chain_sets(rows)
    return (*current_chain_starts(chain_set.lines, start_index, last_index), months)


def haven_month_images(directory: str | Path) -> dict[str, Path]:
    """The monthly haven images in ``directory`` as ``{"YYYY-MM": path}``; empty when there are none."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {m.group(1): path for path in sorted(directory.iterdir()) if (m := HAVEN_MONTH_IMAGE.match(path.name))}


def area_havens(image: Path | None, rows, columns, hours, radius: float, width: int, height: int,
                hours_per_count: float | None = None, alone=None):
    """``(rows, columns, hours)`` of the seed haven area of one month: the chain starts (``rows``, ``columns``, ``hours``)
    and, from the month's haven image when there is one, every haven pixel whose centre is within ``radius`` pixels of a
    start, its hours the image's count times the image's hours per count. A start keeps the hours of the chain table.
    ``alone`` is a ``(rows, columns, hours)`` triple of lone seeds, added as their pixels alone (no haven around them,
    hours of the chain table). Sorted by pixel. ``hours_per_count``, when given, must be the image's."""
    import rasterio
    from rasterio.windows import Window

    found: dict[int, float] = {}
    if image is not None and len(rows):
        reach = int(np.floor(radius))
        with rasterio.open(image) as dataset:
            tags = dataset.tags()
            if "SAFE_HAVEN_BYTE_THRESHOLD" not in tags:
                raise InputError(f"{image} is not a monthly haven image (no SAFE_HAVEN_BYTE_THRESHOLD tag)",
                                 code="seed_areas_haven_image_invalid", details={"path": str(image)})
            threshold = int(tags["SAFE_HAVEN_BYTE_THRESHOLD"])
            per_count = float(tags.get("SAFE_HAVEN_HOURS_PER_COUNT", 2.0))
            if hours_per_count is not None and abs(per_count - hours_per_count) > 1e-9:
                raise InputError(f"{image} counts {per_count:g} h, not {hours_per_count:g} h",
                                 code="seed_areas_hours_per_count_mismatch",
                                 details={"path": str(image), "image": per_count, "setting": hours_per_count})
            nodata = dataset.nodata
            for row, column in zip(rows.tolist(), columns.tolist()):
                top, left = max(row - reach, 0), max(column - reach, 0)
                bottom, right = min(row + reach + 1, height), min(column + reach + 1, width)
                block = dataset.read(1, window=Window(left, top, right - left, bottom - top))
                rr, cc = np.mgrid[top:bottom, left:right]
                near = (rr - row) ** 2 + (cc - column) ** 2 <= radius * radius
                haven = near & (block <= threshold) & ((block != nodata) if nodata is not None else True)
                for r, c, count in zip(rr[haven].tolist(), cc[haven].tolist(), block[haven].tolist()):
                    found[r * width + c] = count * per_count
    for row, column, hour in zip(rows.tolist(), columns.tolist(), hours.tolist()):
        found[row * width + column] = hour
    if alone is not None:
        for row, column, hour in zip(*(np.asarray(v).tolist() for v in alone)):
            found[row * width + column] = hour
    keys = np.array(sorted(found), dtype=np.int64)
    return ((keys // width).astype(np.int32), (keys % width).astype(np.int32),
            np.array([found[k] for k in keys.tolist()], dtype=np.float32))


def _palette(hours_per_count: float) -> dict[int, tuple[int, int, int, int]]:
    """Palette entries for the counts 0..254 by shadow-period class, and a transparent NODATA."""
    colours = [_hex(colour) for _bound, colour in HAVEN_SHADOW_CLASSES]
    classes = haven_shadow_classes(np.arange(NODATA, dtype=np.float64) * hours_per_count)
    table = {count: (*colours[int(k)], 255) for count, k in enumerate(classes)}
    table[NODATA] = (0, 0, 0, 0)
    return table


def _hex(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def _level_arrays(rows, columns, counts, level: int, width: int):
    """The minimum count of each ``2 ** level`` block holding a pixel: ``(keys, values, level_width)``, keys sorted."""
    level_width = -(-width // 2 ** level)
    keys = (rows.astype(np.int64) >> level) * level_width + (columns.astype(np.int64) >> level)
    order = np.argsort(keys, kind="stable")
    keys, counts = keys[order], counts[order]
    first = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.zeros(0, dtype=np.int64)
    return keys[first], (np.minimum.reduceat(counts, first) if len(keys) else counts), level_width


def _write_level(dataset, keys, values, level_width: int, level_height: int) -> None:
    """Write one level in strips: NODATA everywhere but the pixels at ``keys`` (row * ``level_width`` + column)."""
    from rasterio.windows import Window

    for first in range(0, level_height, STRIP):
        last = min(first + STRIP, level_height)
        strip = np.full((last - first, level_width), NODATA, dtype=np.uint8)
        low, high = np.searchsorted(keys, [first * level_width, last * level_width])
        local = keys[low:high] - first * level_width
        strip[local // level_width, local % level_width] = values[low:high]
        dataset.write(strip, 1, window=Window(0, first, level_width, last - first))


def write_area_image(path: Path, rows, columns, hours, *, transform_gdal, crs_wkt: str, width: int, height: int,
                     hours_per_count: float, tags: dict[str, str]) -> None:
    """The GeoTIFF of one month's area: counts (hours / ``hours_per_count``, rounded), NODATA 255, palette, overviews."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.enums import Resampling

    counts = np.clip(np.rint(hours / hours_per_count), 0, NODATA - 1).astype(np.uint8)
    factors, size = [], max(width, height)
    factor = 2
    while size > MIN_OVERVIEW:
        factors.append(factor)
        size, factor = (size + 1) // 2, factor * 2
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype="uint8", nodata=NODATA,
                       crs=crs_wkt or None, transform=Affine.from_gdal(*transform_gdal), tiled=True, blockxsize=BLOCK,
                       blockysize=BLOCK, compress="deflate", BIGTIFF="YES") as dataset:
        keys, values, level_width = _level_arrays(rows, columns, counts, 0, width)
        _write_level(dataset, keys, values, level_width, height)
        dataset.write_colormap(1, _palette(hours_per_count))
        dataset.update_tags(**tags)
        if factors:
            dataset.build_overviews(factors, Resampling.nearest)
    for level in range(1, len(factors) + 1):
        keys, values, level_width = _level_arrays(rows, columns, counts, level, width)
        with rasterio.open(path, "r+", overview_level=level - 1) as overview:
            _write_level(overview, keys, values, level_width, overview.height)


def generate_seed_haven_areas(chains_path: str | Path, dem_path: str | Path, output_directory: str | Path, *,
                              haven_months_directory: str | Path | None = None,
                              radius_meters: float = AREA_RADIUS_METERS, hours_per_count: float = 2.0,
                              overwrite: bool = False, progress: Progress | None = None) -> SeedHavenAreasReport:
    """Write the seed haven areas of every month.

    The area of a month is the current start of each chain drawn in the month (its first seed that is not passed, while 2
    or more seeds are left), every haven of the month, a seed or not, whose centre is within ``radius_meters`` of one of
    them, and each lone last seed as its pixel alone (see ``docs/safe-haven-monthly-products.md``). Two products are written into
    ``output_directory``: ``safe_haven_stage_4_seed_haven_areas/`` (one GeoTIFF per month on the DEM's grid, uint8
    hibernation counts, 255 NODATA, a palette in the shadow-period classes, minimum overviews) and
    ``safe_haven_stage_4_seed_haven_areas.npz`` (the same pixels as lists, for the viewer; read it with
    :func:`read_seed_areas`). Both are written through staging; a failure leaves earlier products untouched.

    Parameters
    ----------
    chains_path : str or Path
        The final stage-4 chain table (CSV).
    dem_path : str or Path
        The DEM GeoTIFF the table's rows and columns refer to; its pixel size turns the radius into pixels.
    output_directory : str or Path
        Where the two products are written.
    haven_months_directory : str or Path, optional
        The monthly haven images (``safe_haven_stage_1_haven_months_<YYYY-MM>.tif``); by default
        ``safe_haven_stage_1_haven_months`` beside the chain table. A month without an image has its chain starts alone.
    radius_meters : float
        How far from a chain start a haven belongs to its area, in metres (the default is 10 pixels on a 20 m DEM).
    hours_per_count : float
        Hours per stored hibernation count; it must be the images' (2 for the stage-1 product).
    overwrite : bool
        Replace existing products; when false an existing product raises ``InputError``.
    progress : callable, optional
        Called with a message for each month.

    Returns
    -------
    SeedHavenAreasReport
        The paths, months, chain starts, lone seeds, pixels and the radius used.

    Raises
    ------
    InputError
        For a negative or non-finite radius, a seed outside the DEM, a haven image that is not one of the monthly
        images or counts other hours, or an existing product without ``overwrite``.
    ProductStorageError
        When the products cannot be written.
    """
    progress = progress or _quiet
    output_directory = Path(output_directory)
    started = clock.monotonic()
    if not np.isfinite(radius_meters) or radius_meters < 0:
        raise InputError("radius_meters must be 0 or more", code="seed_areas_setting_invalid",
                         details={"radius_meters": radius_meters})
    if not np.isfinite(hours_per_count) or hours_per_count <= 0:
        raise InputError("hours_per_count must be positive", code="seed_areas_setting_invalid",
                         details={"hours_per_count": hours_per_count})
    chains_path, directory = Path(chains_path), output_directory
    images_directory = Path(haven_months_directory) if haven_months_directory is not None \
        else chains_path.parent / "safe_haven_stage_1_haven_months"
    final_directory, final_file = directory / AREA_DIRECTORY, directory / AREA_PIXELS
    for path in (final_directory, final_file):
        if path.exists() and not overwrite:
            raise InputError(f"{path} exists; pass overwrite=True to replace it", code="seed_areas_output_exists",
                             details={"path": str(path)})
    progress("Reading the chain table")
    rows = read_chain_table(chains_path)
    geo_transform, width, height, crs_wkt = dem_header(dem_path)
    if max(r["row"] for r in rows) >= height or max(r["column"] for r in rows) >= width:
        raise InputError("a seed lies outside the DEM", code="seed_areas_grid_mismatch")
    pixel_rows = np.array([r["row"] for r in rows], dtype=np.int64)
    pixel_columns = np.array([r["column"] for r in rows], dtype=np.int64)
    hours = np.array([r["hibernation_hours"] or 0.0 for r in rows], dtype=np.float32)
    start_seed, start_month, lone_seed, lone_month, months = chain_starts(rows)
    radius = radius_meters / abs(geo_transform[1])
    images = haven_month_images(images_directory)
    staging_directory = final_directory.with_name(final_directory.name + ".partial")
    staging_file = final_file.with_name(final_file.name + ".partial")
    shutil.rmtree(staging_directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    staging_directory.mkdir()
    area_rows, area_columns, area_hours, offsets, starts, lone, with_image = [], [], [], [0], [], [], []
    try:
        for index, name in enumerate(months):
            progress(f"Seed haven areas: {name}")
            roots = start_seed[start_month == index]
            singles = lone_seed[lone_month == index]
            image = images.get(name)
            r, c, h = area_havens(image, pixel_rows[roots], pixel_columns[roots], hours[roots], radius, width, height,
                                  hours_per_count, alone=(pixel_rows[singles], pixel_columns[singles], hours[singles]))
            area_rows.append(r)
            area_columns.append(c)
            area_hours.append(h)
            offsets.append(offsets[-1] + len(r))
            starts.append(len(roots))
            lone.append(len(singles))
            with_image.append(image is not None)
            write_area_image(
                staging_directory / AREA_IMAGE.format(month=name), r, c, h, transform_gdal=geo_transform, crs_wkt=crs_wkt,
                width=width, height=height, hours_per_count=hours_per_count,
                tags={"SAFE_HAVEN_MONTH": name, "SAFE_HAVEN_AREA_RADIUS_METERS": f"{radius_meters:g}",
                      "SAFE_HAVEN_AREA_STARTS": str(len(roots)), "SAFE_HAVEN_AREA_LONE_SEEDS": str(len(singles)),
                      "SAFE_HAVEN_PIXELS": str(len(r)),
                      "SAFE_HAVEN_HOURS_PER_COUNT": f"{hours_per_count:g}",
                      "SAFE_HAVEN_HAVEN_IMAGE": image.name if image is not None else ""})
        with staging_file.open("wb") as handle:
            np.savez_compressed(
                handle, months=np.array(months), offsets=np.array(offsets, dtype=np.int64),
                row=np.concatenate(area_rows), column=np.concatenate(area_columns), hours=np.concatenate(area_hours),
                starts=np.array(starts, dtype=np.int64), alone=np.array(lone, dtype=np.int64), with_image=np.array(with_image, dtype=bool),
                radius_meters=np.array(float(radius_meters)))
        if final_directory.exists():
            shutil.rmtree(final_directory)
        os.replace(staging_directory, final_directory)
        os.replace(staging_file, final_file)
    except OSError as error:
        raise ProductStorageError(f"could not write the seed haven areas: {error}", code="seed_areas_write_failed",
                                  details={"directory": str(directory)}) from error
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)
        staging_file.unlink(missing_ok=True)
    return SeedHavenAreasReport(directory=str(final_directory), file=str(final_file), months=len(months),
                           chain_starts=int(sum(starts)), pixels=int(offsets[-1]), months_with_images=int(sum(with_image)),
                           radius_meters=float(radius_meters), seconds=clock.monotonic() - started,
                           lone_seeds=int(sum(lone)))


def read_seed_areas(path: str | Path) -> dict:
    """The pixel file of the seed haven areas: ``months`` (names), ``offsets``, ``row``, ``column``, ``hours``, ``starts``,
    ``alone`` (the lone seeds per month), ``with_image`` and ``radius_meters`` (a float)."""
    with np.load(path) as data:
        found = {name: data[name] for name in ("offsets", "row", "column", "hours", "starts", "alone", "with_image")}
        found["months"] = [str(m) for m in data["months"]]
        found["radius_meters"] = float(data["radius_meters"])
    return found
