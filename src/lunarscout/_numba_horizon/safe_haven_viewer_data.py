"""The data behind the safe-haven pole viewer (private): the pregenerated boxes and haven pixels, and the image pyramids.

The viewer (``scripts/view_safe_haven.py``) shows the pole as one bitmap that is panned and zoomed. This module makes
everything it draws, with no user interface in it:

* ``viewer.json`` in the viewer directory: the months of the dataset, the DEM grid, and for every month the bounding
  box of each cluster of connected havens whose chain starts in the month (``[row_min, row_max, column_min, column_max,
  havens]`` in DEM pixels, as in ``safe_haven_stage_4_cluster_boxes.csv``), plus the yearly chain images;
* ``havens.npz`` in the same directory: the DEM pixels of the seed haven areas, month by month, copied from
  ``safe_haven_stage_4_seed_haven_areas.npz`` beside the chain table (driver step ``seed-haven-areas``; see
  ``safe_haven_seed_areas.py``: the first seed of each chain of 2 or more seeds that starts in the month and the havens of
  the month within the step's radius of one), or the chain starts alone when that file is missing; ordered by month with
  the start of each month's slice in ``offsets``;
* a cache directory (by default under ``/tmp``) with image pyramids: the dark hillshade at every zoom level and, for
  each year, the chain image at every zoom level but the first (the first is read from the GeoTIFF itself). A pyramid
  is built once; the next run finds it and starts at once. A pyramid is a set of raw files read through ``numpy.memmap``,
  so a tile of the picture is a slice;
* the monthly haven images (``safe_haven_stage_1_haven_months/``, ``scripts/threshold_safe_havens.py``) when they are
  beside the chain table: every pixel that is a haven in a month, one GeoTIFF per month. They are not copied: a tile is
  read from the month's file, from its own overview at the coarser levels (each overview pixel is the minimum of its
  block, so a haven is never lost), and coloured by its shadow period.

A level ``k`` of a pyramid is the full-size grid reduced by ``2**k`` in both directions (mean for the hillshade, the
largest value of each block for the chain images, which keeps their thin lines), cut into tiles of ``TILE`` pixels.
The boxes and haven dots of a month are not stored as bitmaps: a month has a few thousand of them, so the tiles in view
are drawn from the lists in milliseconds, and the 96 months need no bitmaps of their own.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time as clock
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from ..errors import InputError, ProductStorageError
from ..safe_haven_areas import areas_path, chain_starts, read_seed_areas
from .pixel_coordinates import dem_header
from .safe_haven_chain_maps import monthly_path, seed_month_spans
from .safe_haven_chains import read_chain_table
from .safe_haven_cluster_maps import COUNT_LABELS, cluster_classes, cluster_colours, find_clusters
from .safe_haven_floods import read_edge_table
from .chain_map_styles import HAVEN_SHADOW_CLASSES as _SHADOW_CLASSES
from .chain_map_styles import haven_shadow_classes, haven_shadow_labels
from .safe_haven_overview import SURFACE, _rgb, raw_shade

TILE = 512
STRIP = 1024                                  # DEM rows per strip while building; a multiple of 2 ** the top level
PRODUCT = "safe-haven-viewer-v6"
CHAIN_IMAGE = re.compile(r"^safe_haven_stage_4_longest_chain_per_root_(\d{4})\.tif$")
HAVEN_MONTHS_DIRECTORY = "safe_haven_stage_1_haven_months"      # beside the chain table: scripts/threshold_safe_havens.py
HAVEN_MONTH_IMAGE = re.compile(r"^safe_haven_stage_1_haven_months_(\d{4}-\d{2})\.tif$")
# Every layer that draws havens colours each by its shadow period (its hibernation; a chain's start keeps its own colour):
# the classes are those of the chain images (``chain_map_styles.HAVEN_SHADOW_CLASSES``), which carry them baked in. The seed
# layers and the all-havens layer draw them here.
HAVEN_SHADOW_CLASSES = tuple((bound, _rgb(colour)) for bound, colour in _SHADOW_CLASSES)
SHADOW_RGBA = np.array([(*colour, 255) for _bound, colour in HAVEN_SHADOW_CLASSES], dtype=np.uint8)
_LOW, _HIGH = 0.07, 0.24                      # the hillshade's grey range, as in the overview maps

Progress = Callable[[str], None]


def haven_shadow_class(hours: float) -> int:
    """The index in ``HAVEN_SHADOW_CLASSES`` of one shadow period (see ``haven_shadow_classes``)."""
    return int(haven_shadow_classes(hours))


def haven_shadow_legend(limit_hours: float | None) -> list[tuple[tuple[int, int, int], str]]:
    """``(colour, label)`` for each class that can occur under ``limit_hours`` (None: the last bound), the last class
    running to the limit."""
    labels = haven_shadow_labels(limit_hours)
    return [(HAVEN_SHADOW_CLASSES[k][1], label) for k, label in enumerate(labels)]


def _quiet(_message: str) -> None:
    return None


def level_sizes(size: int) -> list[int]:
    """The side of the grid at every pyramid level, from the full size down to one that fits in a tile."""
    sizes = [size]
    while sizes[-1] > TILE:
        sizes.append((sizes[-1] + 1) // 2)
    return sizes


def reduce2(block: npt.NDArray, how: str) -> npt.NDArray:
    """``block`` (rows, columns[, channels]) with each 2 x 2 group of pixels reduced to one: ``"mean"`` or ``"max"``. An odd
    edge is padded by repeating its last row or column."""
    rows, columns = block.shape[:2]
    if rows % 2 or columns % 2:
        block = np.pad(block, ((0, rows % 2), (0, columns % 2)) + ((0, 0),) * (block.ndim - 2), mode="edge")
    shape = (block.shape[0] // 2, 2, block.shape[1] // 2, 2) + block.shape[2:]
    grouped = block.reshape(shape)
    if how == "max":
        return grouped.max(axis=(1, 3))
    return np.rint(grouped.mean(axis=(1, 3), dtype=np.float32)).astype(block.dtype)


# ---------------------------------------------------------------------------
# The pregenerated lists: viewer.json and havens.npz
# ---------------------------------------------------------------------------


@dataclass
class ViewerReport:
    months: int
    seeds: int
    clusters: int
    connected_havens: int
    chain_years: list
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _monthly_chain_name(chains_path: str | Path) -> str | None:
    """The name of the monthly chain pixel file beside the chain table (scripts/run_safe_haven_monthly_chain_maps.py),
    or None when there is none."""
    path = monthly_path(Path(chains_path).parent)
    return path.name if path.is_file() else None


def _haven_month_images(chains_path: str | Path) -> dict[str, str]:
    """The monthly haven images beside the chain table (scripts/threshold_safe_havens.py), as ``{"YYYY-MM": path relative to
    the chain table's directory}``; empty when there are none."""
    directory = Path(chains_path).parent / HAVEN_MONTHS_DIRECTORY
    if not directory.is_dir():
        return {}
    found = {}
    for path in sorted(directory.iterdir()):
        match = HAVEN_MONTH_IMAGE.match(path.name)
        if match:
            found[match.group(1)] = f"{HAVEN_MONTHS_DIRECTORY}/{path.name}"
    return found


def viewer_paths(viewer_directory: str | Path) -> dict:
    directory = Path(viewer_directory)
    return {"json": directory / "viewer.json", "havens": directory / "havens.npz"}


def _seed_areas_signature(chains_path: Path) -> dict | None:
    """Name, size and time of the seed haven area pixel file beside the chain table, or None when there is none."""
    path = areas_path(chains_path.parent)
    if not path.is_file():
        return None
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_viewer_lists(viewer_directory: str | Path, *, chains_path: str | Path, dem_path: str | Path,
                       connect: str = "chains", edges_path: str | Path | None = None,
                       progress: Progress = _quiet) -> ViewerReport:
    """Write ``viewer.json`` and ``havens.npz`` into ``viewer_directory`` (through staging files). A cluster belongs to the
    month its chain starts in (its lowest seed's ``root_start_utc``). The havens drawn in a month are its seed haven
    area, read from the pixel file ``safe_haven_stage_4_seed_haven_areas.npz`` beside the chain table (driver step
    ``seed-haven-areas``): the first seed of each chain of 2 or more seeds that starts in the month and the havens of the
    month near it. Without that file a month has its chain starts alone. The other seeds are not drawn."""
    started = clock.monotonic()
    chains_path = Path(chains_path)
    progress("Reading the chain table")
    rows = read_chain_table(chains_path)
    edges = read_edge_table(edges_path) if connect == "edges" and edges_path is not None else None
    transform, width, height, _crs = dem_header(dem_path)
    if max(r["row"] for r in rows) >= height or max(r["column"] for r in rows) >= width:
        raise InputError("a seed lies outside the DEM", code="viewer_table_invalid")
    progress("Clustering the connected havens")
    clusters = find_clusters(rows, connect=connect, edges=edges)
    _start_index, _last_index, months = seed_month_spans(rows)
    cluster_month = [rows[int(first) - 1]["root_start_utc"][:7] for first in clusters.first_seed]
    boxes = {name: [] for name in months}
    order = np.argsort(clusters.havens, kind="stable")                # more havens last: drawn on top
    for k in order:
        boxes[cluster_month[k]].append([int(clusters.row_min[k]), int(clusters.row_max[k]), int(clusters.column_min[k]),
                                        int(clusters.column_max[k]), int(clusters.havens[k])])
    progress("Seed haven areas")
    signature = _seed_areas_signature(chains_path)
    seed_rows, seed_columns, seed_hours, offsets, chain_starts_by_month = [], [], [], [0], {}
    if signature is not None:
        areas = read_seed_areas(areas_path(chains_path.parent))
        where = {name: k for k, name in enumerate(areas["months"])}
        for name in months:
            k = where.get(name)
            low, high = (int(areas["offsets"][k]), int(areas["offsets"][k + 1])) if k is not None else (0, 0)
            seed_rows.append(areas["row"][low:high])
            seed_columns.append(areas["column"][low:high])
            seed_hours.append(areas["hours"][low:high])
            offsets.append(offsets[-1] + high - low)
            chain_starts_by_month[name] = int(areas["starts"][k]) if k is not None else 0
        radius = areas["radius_meters"]
    else:
        hours = np.array([r["hibernation_hours"] or 0.0 for r in rows], dtype=np.float32)
        start_seed, start_month, lone_seed, lone_month, table_months = chain_starts(rows)
        pixel_rows = np.array([r["row"] for r in rows], dtype=np.int32)
        pixel_columns = np.array([r["column"] for r in rows], dtype=np.int32)
        for index, name in enumerate(months):
            roots = start_seed[start_month == index]
            shown = np.unique(np.concatenate([roots, lone_seed[lone_month == index]]))   # the starts and lone seeds
            seed_rows.append(pixel_rows[shown])
            seed_columns.append(pixel_columns[shown])
            seed_hours.append(hours[shown])
            offsets.append(offsets[-1] + len(shown))
            chain_starts_by_month[name] = int(len(roots))
        radius = None
    seed_rows, seed_columns = np.concatenate(seed_rows), np.concatenate(seed_columns)
    offsets, seed_hours = np.array(offsets, dtype=np.int64), np.concatenate(seed_hours)
    directory = Path(viewer_directory)
    directory.mkdir(parents=True, exist_ok=True)
    chain_images = {}
    for path in sorted(chains_path.parent.iterdir()):
        match = CHAIN_IMAGE.match(path.name)
        if match:
            chain_images[match.group(1)] = path.name
    meta = {
        "product": PRODUCT, "connect": connect, "months": months, "tile": TILE,
        "dem": {"path": str(dem_path), "width": width, "height": height, "transform": list(transform)},
        "chain_directory": str(chains_path.parent), "chain_images": chain_images,
        "hours_max": float(seed_hours.max()) if len(seed_hours) else 0.0,
        "seed_areas": signature, "area_radius_meters": radius, "chain_starts": chain_starts_by_month,
        "monthly_chains": _monthly_chain_name(chains_path), "haven_months": _haven_month_images(chains_path),
        "source": {"chains": chains_path.name, "size": chains_path.stat().st_size,
                   "mtime_ns": chains_path.stat().st_mtime_ns},
        "box_columns": ["row_min", "row_max", "column_min", "column_max", "havens"],
        "box_classes": list(COUNT_LABELS), "boxes": boxes,
    }
    paths = viewer_paths(directory)
    staging_json, staging_npz = (paths[k].with_name(paths[k].name + ".partial") for k in ("json", "havens"))
    try:
        staging_json.write_text(json.dumps(meta, separators=(",", ":")) + "\n", encoding="utf-8")
        with staging_npz.open("wb") as handle:
            np.savez_compressed(handle, row=seed_rows, column=seed_columns, hours=seed_hours, offsets=offsets)
        os.replace(staging_json, paths["json"])
        os.replace(staging_npz, paths["havens"])
    finally:
        staging_json.unlink(missing_ok=True)
        staging_npz.unlink(missing_ok=True)
    return ViewerReport(months=len(months), seeds=len(rows), clusters=len(clusters),
                        connected_havens=int(clusters.havens.sum()), chain_years=sorted(chain_images),
                        seconds=clock.monotonic() - started)


def lists_current(viewer_directory: str | Path, *, chains_path: str | Path, dem_path: str | Path,
                  connect: str) -> bool:
    """Whether ``viewer.json`` and ``havens.npz`` exist and were made from this chain table, DEM, connection rule and
    seed haven area file."""
    paths = viewer_paths(viewer_directory)
    if not (paths["json"].is_file() and paths["havens"].is_file()):
        return False
    try:
        meta = json.loads(paths["json"].read_text(encoding="utf-8"))
        stat = Path(chains_path).stat()
        return (meta.get("product") == PRODUCT and meta["connect"] == connect
                and meta.get("seed_areas") == _seed_areas_signature(Path(chains_path))
                and meta["source"] == {"chains": Path(chains_path).name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                and meta["dem"]["path"] == str(dem_path) and meta.get("monthly_chains") == _monthly_chain_name(chains_path)
                and meta.get("haven_months", {}) == _haven_month_images(chains_path))
    except (OSError, ValueError, KeyError):
        return False


# ---------------------------------------------------------------------------
# Pyramids
# ---------------------------------------------------------------------------


def _signature(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def _file_signature(path: Path) -> str:
    stat = path.stat()
    return _signature(path, stat.st_size, stat.st_mtime_ns)


def _publish(staging: Path, final: Path) -> None:
    """Make the finished staging directory the pyramid (replacing an older one)."""
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)


def _level_files(directory: Path, sizes: list[int], channels: int, first: int, mode: str):
    return [np.lib.format.open_memmap(directory / f"level{k}.npy", mode=mode, dtype=np.uint8,
                                      shape=(sizes[k], sizes[k]) + ((channels,) if channels > 1 else ()))
            for k in range(first, len(sizes))]


def hillshade_directory(cache_directory: Path, dem_path: Path) -> Path:
    return Path(cache_directory) / f"hillshade_{_file_signature(Path(dem_path))}"


def build_hillshade(cache_directory: str | Path, dem_path: str | Path, progress: Progress = _quiet) -> Path:
    """The dark hillshade of the DEM as an 8-bit pyramid in the cache (nothing is done when it is there already).

    Pass one writes each strip's sun illumination, the same formula as the overview maps, as 1 + 254 x illumination
    (0 where there is no DEM); the grey range is then fixed from the 2nd and 98th percentiles of the whole pole and pass
    two maps the strips to it; the coarser levels are the means of 2 x 2 blocks."""
    import rasterio

    final = hillshade_directory(Path(cache_directory), Path(dem_path))
    if (final / "done.json").is_file():
        return final
    staging = final.with_name(final.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    pixel_metres = abs(dem_header(dem_path)[0][1])
    try:
        with rasterio.open(dem_path) as dem:
            size = dem.width
            if dem.height != dem.width:
                raise InputError("the viewer needs a square DEM", code="viewer_dem_invalid",
                                 details={"height": dem.height, "width": dem.width})
            sizes = level_sizes(size)
            levels = _level_files(staging, sizes, 1, 0, "w+")
            nodata = dem.nodata
            histogram = np.zeros(256, dtype=np.int64)
            for first in range(0, dem.height, STRIP):
                progress(f"Hillshade, pass 1: rows {first:,} of {dem.height:,}")
                top, bottom = max(first - 1, 0), min(first + STRIP + 1, dem.height)
                block = dem.read(1, window=rasterio.windows.Window(0, top, dem.width, bottom - top)).astype(np.float32)
                bad = ~np.isfinite(block) if nodata is None else (block == np.float32(nodata)) | ~np.isfinite(block)
                block[bad] = np.nan
                shade = raw_shade(block, pixel_metres)
                keep = slice(first - top, first - top + min(STRIP, dem.height - first))
                raw = np.where(bad[keep], 0, 1 + np.rint(shade[keep] * 254.0)).astype(np.uint8)
                levels[0][first:first + raw.shape[0]] = raw
                histogram += np.bincount(raw.ravel(), minlength=256)
        valid = histogram[1:]
        cumulative = np.cumsum(valid)
        lo = int(np.searchsorted(cumulative, 0.02 * cumulative[-1])) + 1
        hi = int(np.searchsorted(cumulative, 0.98 * cumulative[-1])) + 1
        surface = _rgb(SURFACE)[0]
        lut = np.full(256, surface, dtype=np.uint8)
        ramp = np.clip((np.arange(256) - lo) / max(hi - lo, 1), 0.0, 1.0)
        lut[1:] = np.rint(255.0 * (_LOW + (_HIGH - _LOW) * ramp[1:])).astype(np.uint8)
        for first in range(0, size, STRIP):
            progress(f"Hillshade, pass 2: rows {first:,} of {size:,}")
            strip = lut[np.asarray(levels[0][first:first + STRIP])]
            levels[0][first:first + strip.shape[0]] = strip
            block = strip
            for k in range(1, len(sizes)):
                block = reduce2(block, "mean")
                target = first // 2 ** k
                levels[k][target:target + block.shape[0]] = block
        for level in levels:
            level.flush()
        del levels
        (staging / "done.json").write_text(json.dumps({"sizes": sizes, "lo": lo, "hi": hi}), encoding="utf-8")
        _publish(staging, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return final


def chain_directory(cache_directory: Path, tif_path: Path) -> Path:
    return Path(cache_directory) / f"chains_{tif_path.stem[-4:]}_{_file_signature(tif_path)}"


def _build_chain_year(args) -> str:
    """Reduce one yearly RGBA chain image into pyramid levels 1 and up (a worker of the process pool)."""
    import rasterio

    cache_directory, tif_path = Path(args[0]), Path(args[1])
    final = chain_directory(cache_directory, tif_path)
    if (final / "done.json").is_file():
        return str(final)
    staging = final.with_name(final.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(tif_path) as image:
            if image.count != 4 or image.height != image.width:
                raise InputError(f"{tif_path.name} is not a square RGBA image", code="viewer_chain_image_invalid")
            sizes = level_sizes(image.width)
            levels = _level_files(staging, sizes, 4, 1, "w+")
            for first in range(0, image.height, STRIP):
                rows = min(STRIP, image.height - first)
                block = np.moveaxis(image.read(window=rasterio.windows.Window(0, first, image.width, rows)), 0, -1)
                for k in range(1, len(sizes)):
                    block = reduce2(block, "max")
                    target = first // 2 ** k
                    levels[k - 1][target:target + block.shape[0]] = block
        for level in levels:
            level.flush()
        del levels
        (staging / "done.json").write_text(json.dumps({"sizes": sizes}), encoding="utf-8")
        _publish(staging, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(final)


def build_chain_pyramids(cache_directory: str | Path, tif_paths: dict, progress: Progress = _quiet,
                         workers: int = 6) -> dict:
    """The pyramids of the yearly chain images in the cache, the missing ones in parallel; ``{year: directory}``."""
    import multiprocessing

    todo = {y: p for y, p in tif_paths.items() if not (chain_directory(Path(cache_directory), Path(p)) / "done.json").is_file()}
    if todo:
        progress(f"Chain images: building {len(todo)} pyramids ({', '.join(sorted(todo))})")
        with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=multiprocessing.get_context("spawn")) as pool:
            for _ in pool.map(_build_chain_year, [(str(cache_directory), str(p)) for p in todo.values()]):
                pass
    return {y: chain_directory(Path(cache_directory), Path(p)) for y, p in tif_paths.items()}


def prepare_viewer(viewer_directory: str | Path, cache_directory: str | Path, *, chains_path: str | Path,
                   dem_path: str | Path, connect: str = "chains", edges_path: str | Path | None = None,
                   rebuild: bool = False, progress: Progress = _quiet) -> None:
    """Make whatever the viewer needs and does not have yet: the lists, the hillshade pyramid and the chain pyramids.
    ``rebuild`` makes the lists again (the pyramids are rebuilt only when their source file changed)."""
    if rebuild or not lists_current(viewer_directory, chains_path=chains_path, dem_path=dem_path, connect=connect):
        build_viewer_lists(viewer_directory, chains_path=chains_path, dem_path=dem_path, connect=connect,
                           edges_path=edges_path, progress=progress)
    meta = json.loads(viewer_paths(viewer_directory)["json"].read_text(encoding="utf-8"))
    cache = Path(cache_directory)
    cache.mkdir(parents=True, exist_ok=True)
    build_hillshade(cache, dem_path, progress)
    tifs = {y: Path(meta["chain_directory"]) / name for y, name in meta["chain_images"].items()}
    build_chain_pyramids(cache, tifs, progress)
    progress("Ready")


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------


def _outline(image, x0: int, y0: int, x1: int, y1: int, colour) -> None:
    """The outline of the rectangle ``x0..x1`` by ``y0..y1`` (inclusive), clipped to the tile; solid when it is thin."""
    height, width = image.shape[:2]
    if x1 - x0 < 3 or y1 - y0 < 3:
        image[max(y0, 0):max(min(y1 + 1, height), 0), max(x0, 0):max(min(x1 + 1, width), 0)] = colour
        return
    xa, xb = max(x0, 0), min(x1 + 1, width)
    if xa < xb:
        for y in (y0, y1):
            if 0 <= y < height:
                image[y, xa:xb] = colour
    ya, yb = max(y0, 0), min(y1 + 1, height)
    if ya < yb:
        for x in (x0, x1):
            if 0 <= x < width:
                image[ya:yb, x] = colour


class ViewerData:
    """Opens a prepared viewer directory and its cache and gives the 512-pixel tiles of every layer."""

    def __init__(self, viewer_directory: str | Path, cache_directory: str | Path):
        paths = viewer_paths(viewer_directory)
        meta = json.loads(paths["json"].read_text(encoding="utf-8"))
        if meta.get("product") != PRODUCT:
            raise ProductStorageError("not a viewer file", code="viewer_file_invalid", details={"path": str(paths["json"])})
        self.meta = meta
        self.months: list[str] = meta["months"]
        self.size = int(meta["dem"]["width"])
        self.sizes = level_sizes(self.size)
        self.levels = len(self.sizes)
        self.boxes = [np.array(meta["boxes"][m], dtype=np.int64).reshape(-1, 5) for m in self.months]
        with np.load(paths["havens"]) as arrays:
            self._rows, self._columns, self._offsets = arrays["row"], arrays["column"], arrays["offsets"]
            self._hours = arrays["hours"]
        self.hours_max = float(meta.get("hours_max", 0.0))
        self.cache = Path(cache_directory)
        dem = Path(meta["dem"]["path"])
        self._hillshade = [np.load(p, mmap_mode="r") for p in self._sorted_levels(hillshade_directory(self.cache, dem))]
        self._root_code = -1                        # the colour class of a chain's start in the monthly file, if it has one
        self.monthly_chains = None                  # (rows, columns, codes, table, per-month slices, chains, started) or None
        name = meta.get("monthly_chains")
        if name and (Path(meta["chain_directory"]) / name).is_file():
            self._load_monthly_chains(Path(meta["chain_directory"]) / name)
        self.chain_years = {int(y): Path(meta["chain_directory"]) / name for y, name in meta["chain_images"].items()}
        self.haven_months = {self.months.index(m): Path(meta["chain_directory"]) / name
                             for m, name in meta.get("haven_months", {}).items()
                             if m in self.months and (Path(meta["chain_directory"]) / name).is_file()}
        self._haven_month_info: dict[int, dict] = {}
        self._haven_luts: dict[tuple, npt.NDArray[np.uint8]] = {}
        self._chain_levels: dict[int, list] = {}
        self._chain_datasets: dict[int, object] = {}

    def _load_monthly_chains(self, path: Path) -> None:
        with np.load(path) as data:
            names = [str(m) for m in data["months"]]
            offsets, chains = data["offsets"], data["chains"]
            slices = np.zeros((len(self.months), 2), dtype=np.int64)
            counts = np.zeros(len(self.months), dtype=np.int64)
            started = np.zeros(len(self.months), dtype=np.int64)
            for k, name in enumerate(names):
                if name in self.months:
                    index = self.months.index(name)
                    slices[index] = (offsets[k], offsets[k + 1])
                    counts[index] = chains[k]
                    started[index] = data["started"][k] if "started" in data else 0
            labels = [str(x) for x in data["labels"]] if "labels" in data else []
            self._root_code = len(data["table"]) - 1 if labels and labels[-1] == "start of a chain" else -1
            self.monthly_chains = (data["row"], data["column"], data["code"], data["table"], slices, counts, started)

    @staticmethod
    def _sorted_levels(directory: Path) -> list[Path]:
        if not (directory / "done.json").is_file():
            raise ProductStorageError("the pyramid is not built (run prepare_viewer)", code="viewer_cache_missing",
                                      details={"directory": str(directory)})
        return sorted(directory.glob("level*.npy"), key=lambda p: int(p.stem[5:]))

    def tiles_across(self, level: int) -> int:
        return -(-self.sizes[level] // TILE)

    def _crop(self, array, level: int, tx: int, ty: int, channels: int):
        """The tile of ``array`` (a level), padded with zeros past the image."""
        size = self.sizes[level]
        y0, x0 = ty * TILE, tx * TILE
        piece = np.asarray(array[y0:min(y0 + TILE, size), x0:min(x0 + TILE, size)])
        if piece.shape[0] == TILE and piece.shape[1] == TILE:
            return np.ascontiguousarray(piece)
        out = np.zeros((TILE, TILE) + ((channels,) if channels > 1 else ()), dtype=np.uint8)
        out[:piece.shape[0], :piece.shape[1]] = piece
        return out

    def hillshade_tile(self, level: int, tx: int, ty: int) -> npt.NDArray[np.uint8]:
        """``(512, 512)`` grey."""
        return self._crop(self._hillshade[level], level, tx, ty, 1)

    def chain_tile(self, year: int, level: int, tx: int, ty: int):
        """``(512, 512, 4)`` RGBA of the year's chain image, or None when there is no image for the year."""
        if year not in self.chain_years:
            return None
        if level == 0:
            import rasterio

            dataset = self._chain_datasets.get(year)
            if dataset is None:
                dataset = self._chain_datasets[year] = rasterio.open(self.chain_years[year])
            window = rasterio.windows.Window(tx * TILE, ty * TILE, min(TILE, self.size - tx * TILE),
                                             min(TILE, self.size - ty * TILE))
            piece = np.moveaxis(dataset.read(window=window), 0, -1)
            out = np.zeros((TILE, TILE, 4), dtype=np.uint8)
            out[:piece.shape[0], :piece.shape[1]] = piece
            return out
        if year not in self._chain_levels:
            directory = chain_directory(self.cache, self.chain_years[year])
            if not (directory / "done.json").is_file():
                raise ProductStorageError("the chain pyramid is not built", code="viewer_cache_missing",
                                          details={"directory": str(directory)})
            files = sorted(directory.glob("level*.npy"), key=lambda p: int(p.stem[5:]))
            self._chain_levels[year] = [np.load(p, mmap_mode="r") for p in files]
        return self._crop(self._chain_levels[year][level - 1], level, tx, ty, 4)

    def haven_month_info(self, month: int) -> dict:
        """``{"pixels", "limit_hours", "threshold", "hours_per_count"}`` of the month's haven image (each None when its file
        does not say; the hours per count default to 2), or an empty dict without an image."""
        if month not in self.haven_months:
            return {}
        if month not in self._haven_month_info:
            import rasterio

            with rasterio.open(self.haven_months[month]) as dataset:
                tags = dataset.tags()

            def number(key, kind):
                try:
                    return kind(float(tags[key]))
                except (KeyError, ValueError):
                    return None
            self._haven_month_info[month] = {
                "pixels": number("SAFE_HAVEN_PIXELS", int), "limit_hours": number("SAFE_HAVEN_SURVIVAL_LIMIT_HOURS", float),
                "threshold": number("SAFE_HAVEN_BYTE_THRESHOLD", int),
                "hours_per_count": number("SAFE_HAVEN_HOURS_PER_COUNT", float) or 2.0}
        return self._haven_month_info[month]

    def _haven_lut(self, threshold: int, hours_per_count: float = 2.0) -> npt.NDArray[np.uint8]:
        """``(256, 4)`` RGBA by stored count: the colour of the shadow-period class of ``count * hours_per_count`` hours for
        the counts up to ``threshold``; everything above (NODATA 255 among it) is transparent."""
        key = (threshold, hours_per_count)
        if key not in self._haven_luts:
            lut = np.zeros((256, 4), dtype=np.uint8)
            for count in range(threshold + 1):
                lut[count] = SHADOW_RGBA[haven_shadow_class(count * hours_per_count)]
            self._haven_luts[key] = lut
        return self._haven_luts[key]

    def haven_month_legend(self, month: int) -> list[tuple[tuple[int, int, int], str]]:
        """``(colour, label)`` of each shadow-period class the month's haven image can show; empty without an image."""
        info = self.haven_month_info(month)
        if not info:
            return []
        limit = info["limit_hours"]
        if limit is None and info["threshold"] is not None:
            limit = info["threshold"] * info["hours_per_count"]
        return haven_shadow_legend(limit)

    def haven_months_tile(self, month: int, level: int, tx: int, ty: int):
        """``(512, 512, 4)`` RGBA of every haven pixel of the month, from the month's haven image (the file's own overview
        at the coarser levels, each pixel the best of its block), coloured by its shadow period (``HAVEN_SHADOW_CLASSES``);
        None without an image for the month."""
        info = self.haven_month_info(month)
        if not info:
            return None
        import rasterio
        from rasterio.errors import RasterioIOError
        from rasterio.windows import Window

        threshold = info["threshold"] if info["threshold"] is not None else 254
        try:
            dataset = rasterio.open(self.haven_months[month], **({"overview_level": level - 1} if level > 0 else {}))
        except (RasterioIOError, IndexError, ValueError):
            return None                                                  # the file has no overview this coarse
        with dataset:
            width, height = min(TILE, dataset.width - tx * TILE), min(TILE, dataset.height - ty * TILE)
            if width <= 0 or height <= 0:
                return None
            values = dataset.read(1, window=Window(tx * TILE, ty * TILE, width, height))
        out = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        out[:height, :width] = self._haven_lut(threshold, info["hours_per_count"])[values]
        return out

    def _in_tile(self, rows, columns, level: int, tx: int, ty: int, margin: int):
        """Mask of the entries whose level pixel lies within ``margin`` pixels of the tile."""
        x, y = columns >> level, rows >> level
        return (x >= tx * TILE - margin) & (x < (tx + 1) * TILE + margin) & (y >= ty * TILE - margin) & (y < (ty + 1) * TILE + margin)

    def havens_count(self, month: int) -> int:
        """The havens of the month's seed haven area (the chain starts and the havens near them)."""
        return int(self._offsets[month + 1] - self._offsets[month])

    def chain_starts_count(self, month: int) -> int:
        """The chains of 2 or more seeds that start in the month: the centres of its seed haven areas."""
        return int(self.meta.get("chain_starts", {}).get(self.months[month], 0))

    def monthly_chain_count(self, month: int) -> int:
        """The chains of 2 or more seeds drawn in the month (the part of each that is still ahead)."""
        return int(self.monthly_chains[5][month]) if self.monthly_chains is not None else 0

    def monthly_chain_started(self, month: int) -> int:
        """How many of those start in the month."""
        return int(self.monthly_chains[6][month]) if self.monthly_chains is not None else 0

    def monthly_chain_tile(self, month: int, level: int, tx: int, ty: int):
        """``(512, 512, 4)`` RGBA of the chains of the month, each as the part still ahead (the pixels of the monthly chain
        file, in their colour classes; 1 pixel at the two finest levels, 2 at the coarser ones) with the start of each
        chain that starts in the month marked by a 3 x 3 square of the start colour, or None without the file."""
        if self.monthly_chains is None:
            return None
        rows, columns, codes, table, slices = self.monthly_chains[:5]
        low, high = int(slices[month][0]), int(slices[month][1])
        image = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        rows, columns, codes = rows[low:high], columns[low:high], codes[low:high]
        mask = self._in_tile(rows, columns, level, tx, ty, 2)
        x = (columns[mask] >> level) - tx * TILE
        y = (rows[mask] >> level) - ty * TILE
        codes = codes[mask]
        colours = table[codes]                                     # in class order: the last write is the one on top
        marker = codes == self._root_code
        dot = 1 if level < 2 else 2
        for dy in range(dot):
            for dx in range(dot):
                inside = ~marker & (x + dx >= 0) & (x + dx < TILE) & (y + dy >= 0) & (y + dy < TILE)
                image[(y + dy)[inside], (x + dx)[inside]] = colours[inside]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                inside = marker & (x + dx >= 0) & (x + dx < TILE) & (y + dy >= 0) & (y + dy < TILE)
                image[(y + dy)[inside], (x + dx)[inside]] = colours[inside]
        return image

    def boxes_havens_tile(self, month: int, level: int, tx: int, ty: int) -> npt.NDArray[np.uint8]:
        """``(512, 512, 4)`` RGBA: the cluster boxes of the month with the havens of the month drawn over them."""
        image = self.boxes_tile(month, level, tx, ty)
        dots = self.havens_tile(month, level, tx, ty)
        drawn = dots[..., 3] > 0
        image[drawn] = dots[drawn]
        return image

    def boxes_count(self, month: int) -> int:
        return len(self.boxes[month])

    def havens_tile(self, month: int, level: int, tx: int, ty: int) -> npt.NDArray[np.uint8]:
        """``(512, 512, 4)`` RGBA: one dot per haven of the month in the colour of its shadow period (1 pixel at the full
        size, 2 at the coarser levels), the brightest class on top where dots overlap."""
        image = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        low, high = int(self._offsets[month]), int(self._offsets[month + 1])
        rows, columns, hours = self._rows[low:high], self._columns[low:high], self._hours[low:high]
        mask = self._in_tile(rows, columns, level, tx, ty, 1)
        x = (columns[mask] >> level) - tx * TILE
        y = (rows[mask] >> level) - ty * TILE
        classes = haven_shadow_classes(hours[mask])
        dot = 1 if level == 0 else 2
        for shadow in range(len(HAVEN_SHADOW_CLASSES) - 1, -1, -1):
            chosen = classes == shadow
            for dy in range(dot):
                for dx in range(dot):
                    inside = chosen & (x + dx >= 0) & (x + dx < TILE) & (y + dy >= 0) & (y + dy < TILE)
                    image[(y + dy)[inside], (x + dx)[inside]] = SHADOW_RGBA[shadow]
        return image

    def shadow_legend(self) -> list[tuple[tuple[int, int, int], str]]:
        """``(colour, label)`` of the shadow-period classes of the seeds: the last runs to the survival limit, 48 hours
        unless a seed's shadow period is longer."""
        return haven_shadow_legend(max(HAVEN_SHADOW_CLASSES[-1][0], self.hours_max))

    def chain_start_colour(self):
        """The colour of the mark of a chain's start in the monthly chain layer, or None without that layer."""
        if self.monthly_chains is None or self._root_code < 0:
            return None
        return tuple(int(v) for v in self.monthly_chains[3][self._root_code][:3])

    def boxes_tile(self, month: int, level: int, tx: int, ty: int) -> npt.NDArray[np.uint8]:
        """``(512, 512, 4)`` RGBA: the outline of each cluster's box, one level pixel wide just outside the box and solid when
        the box is only a few pixels across at this level, coloured by the number of havens (more havens on top)."""
        image = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        boxes = self.boxes[month]
        if not len(boxes):
            return image
        row_min, row_max, column_min, column_max, havens = boxes.T
        x0, x1 = (column_min >> level) - 1, (column_max >> level) + 1
        y0, y1 = (row_min >> level) - 1, (row_max >> level) + 1
        left, top = tx * TILE, ty * TILE
        visible = (x1 >= left) & (x0 < left + TILE) & (y1 >= top) & (y0 < top + TILE)
        colours = cluster_colours()
        classes = cluster_classes(havens)
        for k in np.flatnonzero(visible):
            colour = (*[int(v) for v in colours[classes[k]][:3]], 255)
            _outline(image, int(x0[k]) - left, int(y0[k]) - top, int(x1[k]) - left, int(y1[k]) - top, colour)
        return image

