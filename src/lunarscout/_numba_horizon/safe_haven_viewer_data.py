"""The data behind the safe-haven pole viewer (private): the pregenerated boxes and haven pixels, and the image pyramids.

The viewer (``scripts/view_safe_haven.py``) shows the pole as one bitmap that is panned and zoomed. This module makes
everything it draws, with no user interface in it:

* ``viewer.json`` in the viewer directory: the months of the dataset, the DEM grid, and for every month the bounding
  box of each cluster of connected havens whose chain starts in the month (``[row_min, row_max, column_min, column_max,
  havens]`` in DEM pixels, as in ``safe_haven_stage_4_cluster_boxes.csv``), plus the yearly chain images;
* ``havens.npz`` in the same directory: the DEM pixel of every seed, ordered by month with the start of each month's
  slice in ``offsets``;
* a cache directory (by default under ``/tmp``) with image pyramids: the dark hillshade at every zoom level and, for
  each year, the chain image at every zoom level but the first (the first is read from the GeoTIFF itself). A pyramid
  is built once; the next run finds it and starts at once. A pyramid is a set of raw files read through ``numpy.memmap``,
  so a tile of the picture is a slice.

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
from .pixel_coordinates import dem_header
from .safe_haven_chain_maps import month_range, monthly_path
from .safe_haven_chains import read_chain_table
from .safe_haven_cluster_maps import COUNT_LABELS, cluster_classes, cluster_colours, find_clusters
from .safe_haven_floods import read_edge_table
from .safe_haven_overview import SURFACE, _rgb, raw_shade

TILE = 512
STRIP = 1024                                  # DEM rows per strip while building; a multiple of 2 ** the top level
PRODUCT = "safe-haven-viewer-v1"
CHAIN_IMAGE = re.compile(r"^safe_haven_stage_4_longest_chain_per_root_(\d{4})\.tif$")
HAVEN_COLOUR = (238, 143, 102, 255)           # one orange for the havens, bright against the dark terrain
_LOW, _HIGH = 0.07, 0.24                      # the hillshade's grey range, as in the overview maps

Progress = Callable[[str], None]


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


def viewer_paths(viewer_directory: str | Path) -> dict:
    directory = Path(viewer_directory)
    return {"json": directory / "viewer.json", "havens": directory / "havens.npz"}


def build_viewer_lists(viewer_directory: str | Path, *, chains_path: str | Path, dem_path: str | Path,
                       connect: str = "chains", edges_path: str | Path | None = None,
                       progress: Progress = _quiet) -> ViewerReport:
    """Write ``viewer.json`` and ``havens.npz`` into ``viewer_directory`` (through staging files). A cluster belongs to the
    month its chain starts in (its lowest seed's ``root_start_utc``), a seed to its own month."""
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
    seed_month = [str(r["month"]) for r in rows]
    cluster_month = [rows[int(first) - 1]["root_start_utc"][:7] for first in clusters.first_seed]
    months = month_range(min(seed_month + cluster_month), max(seed_month + cluster_month))
    index = {name: k for k, name in enumerate(months)}
    boxes = {name: [] for name in months}
    order = np.argsort(clusters.havens, kind="stable")                # more havens last: drawn on top
    for k in order:
        boxes[cluster_month[k]].append([int(clusters.row_min[k]), int(clusters.row_max[k]), int(clusters.column_min[k]),
                                        int(clusters.column_max[k]), int(clusters.havens[k])])
    month_of_seed = np.array([index[m] for m in seed_month], dtype=np.int32)
    sort = np.argsort(month_of_seed, kind="stable")
    seed_rows = np.array([r["row"] for r in rows], dtype=np.int32)[sort]
    seed_columns = np.array([r["column"] for r in rows], dtype=np.int32)[sort]
    offsets = np.searchsorted(month_of_seed[sort], np.arange(len(months) + 1)).astype(np.int64)
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
        "monthly_chains": _monthly_chain_name(chains_path),
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
            np.savez_compressed(handle, row=seed_rows, column=seed_columns, offsets=offsets)
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
    """Whether ``viewer.json`` and ``havens.npz`` exist and were made from this chain table, DEM and connection rule."""
    paths = viewer_paths(viewer_directory)
    if not (paths["json"].is_file() and paths["havens"].is_file()):
        return False
    try:
        meta = json.loads(paths["json"].read_text(encoding="utf-8"))
        stat = Path(chains_path).stat()
        return (meta.get("product") == PRODUCT and meta["connect"] == connect
                and meta["source"] == {"chains": Path(chains_path).name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                and meta["dem"]["path"] == str(dem_path) and meta.get("monthly_chains") == _monthly_chain_name(chains_path))
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
        self.cache = Path(cache_directory)
        dem = Path(meta["dem"]["path"])
        self._hillshade = [np.load(p, mmap_mode="r") for p in self._sorted_levels(hillshade_directory(self.cache, dem))]
        self.monthly_chains = None                  # (rows, columns, codes, table, per-month slices, chains) or None
        name = meta.get("monthly_chains")
        if name and (Path(meta["chain_directory"]) / name).is_file():
            self._load_monthly_chains(Path(meta["chain_directory"]) / name)
        self.chain_years = {int(y): Path(meta["chain_directory"]) / name for y, name in meta["chain_images"].items()}
        self._chain_levels: dict[int, list] = {}
        self._chain_datasets: dict[int, object] = {}

    def _load_monthly_chains(self, path: Path) -> None:
        with np.load(path) as data:
            names = [str(m) for m in data["months"]]
            offsets, chains = data["offsets"], data["chains"]
            slices = np.zeros((len(self.months), 2), dtype=np.int64)
            counts = np.zeros(len(self.months), dtype=np.int64)
            for k, name in enumerate(names):
                if name in self.months:
                    index = self.months.index(name)
                    slices[index] = (offsets[k], offsets[k + 1])
                    counts[index] = chains[k]
            self.monthly_chains = (data["row"], data["column"], data["code"], data["table"], slices, counts)

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

    def _in_tile(self, rows, columns, level: int, tx: int, ty: int, margin: int):
        """Mask of the entries whose level pixel lies within ``margin`` pixels of the tile."""
        x, y = columns >> level, rows >> level
        return (x >= tx * TILE - margin) & (x < (tx + 1) * TILE + margin) & (y >= ty * TILE - margin) & (y < (ty + 1) * TILE + margin)

    def havens_count(self, month: int) -> int:
        return int(self._offsets[month + 1] - self._offsets[month])

    def monthly_chain_count(self, month: int) -> int:
        """The chains of 2 or more seeds that start in the month."""
        return int(self.monthly_chains[5][month]) if self.monthly_chains is not None else 0

    def monthly_chain_tile(self, month: int, level: int, tx: int, ty: int):
        """``(512, 512, 4)`` RGBA of the chains that start in the month (the pixels of the monthly chain file, in their
        colour classes; 1 pixel at the two finest levels, 2 at the coarser ones), or None without the file."""
        if self.monthly_chains is None:
            return None
        rows, columns, codes, table, slices, _counts = self.monthly_chains
        low, high = int(slices[month][0]), int(slices[month][1])
        image = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        rows, columns, codes = rows[low:high], columns[low:high], codes[low:high]
        mask = self._in_tile(rows, columns, level, tx, ty, 1)
        x = (columns[mask] >> level) - tx * TILE
        y = (rows[mask] >> level) - ty * TILE
        colours = table[codes[mask]]                               # in class order: the last write is the one on top
        dot = 1 if level < 2 else 2
        for dy in range(dot):
            for dx in range(dot):
                inside = (x + dx >= 0) & (x + dx < TILE) & (y + dy >= 0) & (y + dy < TILE)
                image[(y + dy)[inside], (x + dx)[inside]] = colours[inside]
        return image

    def boxes_count(self, month: int) -> int:
        return len(self.boxes[month])

    def havens_tile(self, month: int, level: int, tx: int, ty: int) -> npt.NDArray[np.uint8]:
        """``(512, 512, 4)`` RGBA: one dot per haven of the month (1 pixel at the full size, 2 at the coarser levels)."""
        image = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        low, high = int(self._offsets[month]), int(self._offsets[month + 1])
        rows, columns = self._rows[low:high], self._columns[low:high]
        mask = self._in_tile(rows, columns, level, tx, ty, 1)
        x = (columns[mask] >> level) - tx * TILE
        y = (rows[mask] >> level) - ty * TILE
        dot = 1 if level == 0 else 2
        for dy in range(dot):
            for dx in range(dot):
                inside = (x + dx >= 0) & (x + dx < TILE) & (y + dy >= 0) & (y + dy < TILE)
                image[(y + dy)[inside], (x + dx)[inside]] = HAVEN_COLOUR
        return image

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

