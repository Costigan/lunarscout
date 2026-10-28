"""Maps of the stage-4 chains: colour images and GeoPackages (private).

Two chain sets are drawn from the stage-4 table, each as a colour image, a copy with dilated lines and a GeoPackage:

* **predecessor edges**: one segment from every seed that has a predecessor to the seed (the label forest of stage 4:
  a root's chains branch, and every edge is drawn once), coloured by the longest chain that passes through it (the
  deepest seed at or below its target), so a chain keeps one colour along its length.
* **longest chain per root**: for every root that has descendants, the chain with the most seeds (ties: the longer
  mission, then the lowest seed id), drawn root to end in the colour of its length.

The images are for people, not data (the table is the data): RGBA GeoTIFFs on the DEM's grid with a transparent
background. Chain lengths are put into intervals: chains of 2 seeds, then four intervals cut at the quartiles of the
lengths of the chains of 3 or more (the same cut points for both sets), each a distinct, saturated hue, brighter for
longer chains (``chain_map_styles``). Every haven (seed) is a white pixel; route pixels that pass next to a permanently
shadowed region (PSR) are aqua. Paint order, bottom to top: routes (longer chains over shorter), aqua, havens.

In the images a segment is the rover's route: stage 3 keeps frontiers, not paths, so the flood of the edge's predecessor
is run again with its frontiers kept and traced back from the seed (``safe_haven_paths``), giving one feasible,
time-respecting route. Without a light store the segment is the straight line between the centres of its two seed
pixels, a schematic that may cross terrain the rover cannot. The dilated copy stamps a square of ``2 * dilate + 1``
pixels along every segment (havens one pixel larger).

The GeoPackages hold the same sets as vectors, not dilated and always straight (a route would be thousands of short
segments), with attributes and the same colours: a ``predecessor_edges`` line layer (one segment per edge) or a
``chains`` line layer (one line through the seeds of each chain, with the numbers of small, medium and large PSRs it
passes) and an ``isolated_seeds`` point layer (white: havens that are alone).
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import time as clock
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .chain_map_styles import (
    HAVEN_COLOUR, PSR_COLOUR, build_intervals, class_of, line_style_qml, lookup_table, point_style_qml,
)
from .pixel_coordinates import dem_header, pixel_centres
from .safe_haven_chains import read_chain_table

CHAIN_SETS = ("predecessor_edges", "longest_chain_per_root")
_SRS_ID = 100000
_GPKG_APPLICATION_ID = 0x47504B47
_STRIP_ROWS = 1024
_PLANES = 3                     # per pixel: the interval class of the route on top, a PSR mark, a haven mark


@dataclass
class ChainMapsReport:
    seeds: int
    predecessor_edges: int
    chains: int                      # roots with descendants: one longest chain each
    isolated_seeds: int
    intervals: list                  # the chain-length intervals: label, first, last, colour, chains
    files: dict
    raster_lines: str = "straight"   # "routes" when the images follow reconstructed rover routes
    route_pixels: int = 0
    psr_marked_pixels: dict | None = None    # per set, the pixels of the full-resolution image that are aqua
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# The chain sets
# ---------------------------------------------------------------------------


@dataclass
class Segments:
    """Segments to draw: pixel ends and the chain length (which decides the colour and the drawing order)."""

    row0: npt.NDArray[np.int64]
    column0: npt.NDArray[np.int64]
    row1: npt.NDArray[np.int64]
    column1: npt.NDArray[np.int64]
    length: npt.NDArray[np.int64]

    def ascending(self) -> npt.NDArray[np.int64]:
        """The order to draw in: by length, so that the longer chain is on top."""
        return np.argsort(self.length, kind="stable")


@dataclass
class ChainSet:
    """What one product shows: segments for the images; ``lines`` (vertex lists as seed-id arrays with their
    attributes) and ``dots`` (the seeds that are alone) for the vectors."""

    segments: Segments
    segment_attributes: dict          # columns of the segment layer, one array each
    segment_seeds: npt.NDArray        # (n, 2) 1-based seed ids of each segment's ends
    lines: list                       # longest-chain set only: seed-id lists
    line_attributes: dict
    dots: npt.NDArray                 # 1-based seed ids of the seeds that are alone


def build_chain_sets(rows: list[dict]):
    """``(arrays, predecessor_edges, longest_chain_per_root)`` from the stage-4 table: the per-seed arrays used and
    the two :class:`ChainSet` s."""
    count = len(rows)
    if count == 0 or [r["seed_id"] for r in rows] != list(range(1, count + 1)):
        raise InputError("the chain table's seeds are not 1..N in order", code="chain_maps_table_invalid")

    def field(name, dtype=np.int64):
        return np.array([r[name] for r in rows], dtype=dtype)

    pixel_row, pixel_column = field("row"), field("column")
    predecessor, root, length = field("predecessor_id"), field("root_id"), field("chain_length")
    mission = field("mission_hours", float)
    root_start = [r["root_start_utc"] for r in rows]
    mission_u16 = np.minimum(np.rint(mission), 65535).astype(np.int64)

    # The longest chain through each seed and the longest mission at or below it, from the leaves up.
    through_length, through_mission = length.copy(), mission_u16.copy()
    for level in range(int(length.max()), 1, -1):
        members = np.flatnonzero(length == level)
        np.maximum.at(through_length, predecessor[members] - 1, through_length[members])
        np.maximum.at(through_mission, predecessor[members] - 1, through_mission[members])

    # --- predecessor edges
    heads = np.flatnonzero(predecessor > 0)
    tails = predecessor[heads] - 1
    edge_set = ChainSet(
        segments=Segments(pixel_row[tails], pixel_column[tails], pixel_row[heads], pixel_column[heads],
                          through_length[heads]),
        segment_attributes={
            "seed_id": heads + 1, "predecessor_id": predecessor[heads], "root_id": root[heads],
            "chain_length": through_length[heads], "seed_chain_length": length[heads],
            "mission_hours": through_mission[heads], "root_start_utc": [root_start[i] for i in heads]},
        segment_seeds=np.stack([tails + 1, heads + 1], axis=1) if len(heads) else np.zeros((0, 2), dtype=np.int64),
        lines=[], line_attributes={}, dots=np.zeros(0, dtype=np.int64))

    # --- the longest chain of each root: most seeds, then the longer mission, then the lowest seed id
    ids = np.arange(1, count + 1)
    order = np.lexsort((-ids, mission_u16, length, root))
    last = np.flatnonzero(np.r_[root[order][1:] != root[order][:-1], True])
    ends = order[last]                                         # 0-based index of each root's best end seed
    long_ends = ends[length[ends] >= 2]
    isolated = ends[length[ends] == 1]
    lines = []
    if len(long_ends):
        steps = int(length[long_ends].max())
        paths = np.zeros((len(long_ends), steps), dtype=np.int64)       # seed ids, root first, 0-padded at the end
        cursor = long_ends + 1
        for k in range(steps):
            alive = cursor > 0
            position = length[cursor[alive] - 1] - 1          # the seed's place in its chain
            paths[np.flatnonzero(alive), position] = cursor[alive]
            cursor = np.where(alive, predecessor[np.maximum(cursor, 1) - 1], 0)
        lines = [paths[i, :length[e]] for i, e in enumerate(long_ends)]
    segment_ends, segment_length = [], []
    for line, end in zip(lines, long_ends):
        for a, b in zip(line[:-1], line[1:]):
            segment_ends.append((a - 1, b - 1))
            segment_length.append(length[end])                 # all of a chain at the chain's own length
    pair = np.array(segment_ends, dtype=np.int64).reshape(-1, 2)
    psr = {name: [rows[e][name] for e in long_ends] for name in ("chain_psr_small", "chain_psr_medium", "chain_psr_large")}
    chain_set = ChainSet(
        segments=Segments(pixel_row[pair[:, 0]], pixel_column[pair[:, 0]], pixel_row[pair[:, 1]],
                          pixel_column[pair[:, 1]], np.array(segment_length, dtype=np.int64)),
        segment_attributes={},
        segment_seeds=pair + 1,
        lines=lines,
        line_attributes={
            "root_id": np.array([line[0] for line in lines], dtype=np.int64), "end_seed_id": long_ends + 1,
            "chain_length": length[long_ends], "mission_hours": mission[long_ends],
            "root_start_utc": [root_start[i] for i in long_ends],
            "seed_ids": [",".join(str(s) for s in line) for line in lines], **psr},
        dots=isolated + 1)
    arrays = dict(row=pixel_row, column=pixel_column, root=root, predecessor=predecessor, length=length,
                  mission=mission, through_length=through_length, through_mission=through_mission, rows=rows)
    return arrays, edge_set, chain_set


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _build_kernels(jit):
    """``(draw, draw_polylines, draw_havens)`` with ``jit`` applied to them and their helpers."""

    @jit
    def next_to_psr(psr, r, c, height, width):
        for pr in range(max(r - 1, 0), min(r + 1, height - 1) + 1):
            for pc in range(max(c - 1, 0), min(c + 1, width - 1) + 1):
                if psr[pr, pc]:
                    return True
        return False

    @jit
    def stamp(planes, plane, r, c, radius, height, width, value):
        for rr in range(max(r - radius, 0), min(r + radius, height - 1) + 1):
            for cc in range(max(c - radius, 0), min(c + radius, width - 1) + 1):
                planes[plane, rr, cc] = value

    @jit
    def draw(planes, row0, column0, row1, column1, klass, order, radius, psr, use_psr):
        height, width = planes.shape[1], planes.shape[2]
        for k in range(order.shape[0]):
            i = order[k]
            r, c = row0[i], column0[i]
            dr, dc = abs(row1[i] - r), abs(column1[i] - c)
            sr = 1 if row1[i] > r else -1
            sc = 1 if column1[i] > c else -1
            err = dc - dr
            while True:
                stamp(planes, 0, r, c, radius, height, width, klass[i])
                if use_psr and r >= 0 and r < height and c >= 0 and c < width and next_to_psr(psr, r, c, height, width):
                    stamp(planes, 1, r, c, radius, height, width, 1)
                if r == row1[i] and c == column1[i]:
                    break
                e2 = 2 * err
                if e2 > -dr:
                    err -= dr
                    c += sc
                if e2 < dc:
                    err += dc
                    r += sr

    @jit
    def draw_polylines(planes, starts, stops, rows, columns, klass, order, radius, psr, use_psr):
        height, width = planes.shape[1], planes.shape[2]
        for k in range(order.shape[0]):
            i = order[k]
            for p in range(starts[i], stops[i]):
                r, c = rows[p], columns[p]
                stamp(planes, 0, r, c, radius, height, width, klass[i])
                if use_psr and r >= 0 and r < height and c >= 0 and c < width and next_to_psr(psr, r, c, height, width):
                    stamp(planes, 1, r, c, radius, height, width, 1)

    @jit
    def draw_havens(planes, rows, columns, radius):
        height, width = planes.shape[1], planes.shape[2]
        for i in range(rows.shape[0]):
            stamp(planes, 2, rows[i], columns[i], radius, height, width, 1)

    return draw, draw_polylines, draw_havens


_PLAIN = _build_kernels(lambda function: function)
_KERNEL: dict = {}


def _kernel(index: int, compiled: bool):
    """The plain-Python or the compiled (built on first use) ``draw`` (0), ``draw_polylines`` (1) or ``draw_havens`` (2)."""
    if not compiled:
        return _PLAIN[index]
    if "compiled" not in _KERNEL:
        from numba import njit

        _KERNEL["compiled"] = _build_kernels(njit(cache=False, nogil=True))
    return _KERNEL["compiled"][index]


def _psr_arguments(psr):
    """``(array, use)`` for the kernels: the PSR mask, or a dummy array and False."""
    if psr is None:
        return np.zeros((1, 1), dtype=np.bool_), False
    return np.ascontiguousarray(psr, dtype=np.bool_), True


def draw_segments(planes, segments: Segments, intervals, radius: int = 0, *, psr=None, compiled: bool = True) -> None:
    """Draw ``segments`` as straight lines into ``planes`` ((3, height, width) uint8: the interval class of the route on
    top, the PSR mark, the haven mark), shortest chain first, each with a square stamp of ``2 * radius + 1`` pixels along
    a Bresenham line between its pixel ends; pixels next to ``psr`` (a boolean mask of the PSR pixels) set the PSR mark.
    Ends outside the raster are clipped."""
    if not len(segments.length):
        return
    mask, use = _psr_arguments(psr)
    _kernel(0, compiled)(
        planes, *[np.ascontiguousarray(a, dtype=np.int64) for a in (segments.row0, segments.column0, segments.row1,
                                                                    segments.column1)],
        np.ascontiguousarray(class_of(segments.length, intervals), dtype=np.uint8),
        np.ascontiguousarray(segments.ascending(), dtype=np.int64), int(radius), mask, use)


def draw_polylines(planes, segments: Segments, intervals, paths, heads, radius: int = 0, *, psr=None,
                   compiled: bool = True) -> None:
    """Draw each segment as the route ``paths`` (a ``PathSet``) holds for its head seed (``heads``, one id per
    segment), shortest chain first, a square stamp of ``2 * radius + 1`` pixels on every pixel of the route; pixels
    next to ``psr`` set the PSR mark."""
    if not len(segments.length):
        return
    mask, use = _psr_arguments(psr)
    index = paths.index(heads)
    _kernel(1, compiled)(
        planes, np.ascontiguousarray(paths.offsets[index]), np.ascontiguousarray(paths.offsets[index + 1]),
        np.ascontiguousarray(paths.rows, dtype=np.int32), np.ascontiguousarray(paths.columns, dtype=np.int32),
        np.ascontiguousarray(class_of(segments.length, intervals), dtype=np.uint8),
        np.ascontiguousarray(segments.ascending(), dtype=np.int64), int(radius), mask, use)


def render(chain_set: ChainSet, havens, intervals, height: int, width: int, radius: int, *, paths=None, psr=None,
           compiled: bool = True):
    """The three planes of a set's image: its segments (routes when ``paths`` is given, straight lines otherwise),
    the route pixels next to ``psr`` and every haven (``havens`` is a ``(rows, columns)`` pair), the havens one pixel
    larger than the stamp when there is one."""
    planes = np.zeros((_PLANES, height, width), dtype=np.uint8)
    if paths is None:
        draw_segments(planes, chain_set.segments, intervals, radius, psr=psr, compiled=compiled)
    else:
        draw_polylines(planes, chain_set.segments, intervals, paths, chain_set.segment_seeds[:, 1], radius, psr=psr,
                       compiled=compiled)
    _kernel(2, compiled)(planes, np.ascontiguousarray(havens[0], dtype=np.int64),
                         np.ascontiguousarray(havens[1], dtype=np.int64), radius + (1 if radius > 0 else 0))
    return planes


def colour_image(planes, intervals) -> npt.NDArray[np.uint8]:
    """``(4, rows, columns)`` RGBA of planes: the route's interval colour, aqua over it where the PSR mark is set,
    white over both where there is a haven; 0 (nothing) is transparent."""
    table = lookup_table(intervals)
    index = planes[0].astype(np.int64)
    index[planes[1] > 0] = len(intervals) + 1
    index[planes[2] > 0] = len(intervals) + 2
    return np.moveaxis(table[index], -1, 0)


def write_image(path: Path, planes, intervals, *, transform, crs_wkt: str, tags: dict) -> None:
    """Write the colour image of ``planes`` as a tiled RGBA GeoTIFF (transparent where nothing is drawn)."""
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import Affine
    from rasterio.windows import Window

    staging = path.with_name(path.name + ".partial")
    try:
        with rasterio.open(staging, "w", driver="GTiff", height=planes.shape[1], width=planes.shape[2], count=4,
                           dtype="uint8", photometric="RGB", transform=Affine.from_gdal(*transform),
                           crs=crs_wkt or None, tiled=True, blockxsize=128, blockysize=128, compress="deflate",
                           predictor=2, BIGTIFF="IF_SAFER") as dataset:
            dataset.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
            for row in range(0, planes.shape[1], _STRIP_ROWS):
                strip = planes[:, row:row + _STRIP_ROWS, :]
                dataset.write(colour_image(strip, intervals), window=Window(0, row, planes.shape[2], strip.shape[1]))
            dataset.update_tags(**tags)
        os.replace(staging, path)
        # A sidecar left by the file just replaced (band names, statistics) would make QGIS stretch the new colours.
        path.with_name(path.name + ".aux.xml").unlink(missing_ok=True)
    finally:
        staging.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# GeoPackage
# ---------------------------------------------------------------------------


def _blob(wkb: bytes) -> bytes:
    return b"GP" + bytes([0, 0x01]) + struct.pack("<i", _SRS_ID) + wkb      # version 0; little endian, no envelope


def _wkb_point(x: float, y: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, x, y)


def _wkb_line(xs, ys) -> bytes:
    body = b"".join(struct.pack("<dd", float(x), float(y)) for x, y in zip(xs, ys))
    return struct.pack("<BII", 1, 2, len(xs)) + body


def write_geopackage(path: Path, crs_wkt: str, layers: list[dict]) -> None:
    """A GeoPackage 1.3 with the given layers: ``{"name", "geometry": "LINESTRING" | "POINT", "columns":
    [(name, sql_type)], "features": [(wkb, values...)], "bounds": (xmin, ymin, xmax, ymax), "style": QGIS QML text
    (optional)}``. Layers with a style get it in QGIS's ``layer_styles`` table as the default style of the layer."""
    from pyproj import CRS

    staging = path.with_name(path.name + ".partial")
    staging.unlink(missing_ok=True)
    try:
        connection = sqlite3.connect(staging)
        with connection:
            connection.execute(f"PRAGMA application_id = {_GPKG_APPLICATION_ID}")
            connection.execute("PRAGMA user_version = 10300")
            connection.execute("""CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
                organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL,
                description TEXT)""")
            connection.execute("""CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
                identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL DEFAULT
                (strftime('%Y-%m-%dT%H:%M:%fZ','now')), min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
                srs_id INTEGER, CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id))""")
            connection.execute("""CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,
                CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
                CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
                CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id))""")
            connection.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", [
                ("Undefined Cartesian SRS", -1, "NONE", -1, "undefined", "undefined Cartesian coordinate reference system"),
                ("Undefined geographic SRS", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"),
                ("WGS 84 geodetic", 4326, "EPSG", 4326, CRS.from_epsg(4326).to_wkt(version="WKT1_GDAL"),
                 "longitude/latitude coordinates in decimal degrees on the WGS 84 spheroid"),
                ("DEM projection", _SRS_ID, "NONE", _SRS_ID, crs_wkt or "undefined", "the CRS of the DEM")])
            for layer in layers:
                name = layer["name"]
                columns = "".join(f', "{c}" {t}' for c, t in layer["columns"])
                connection.execute(f'CREATE TABLE "{name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB{columns})')
                marks = ",".join("?" * (1 + len(layer["columns"])))
                connection.executemany(f'INSERT INTO "{name}" (geom{"".join(f", {chr(34)}{c}{chr(34)}" for c, _ in layer["columns"])}) '
                                       f"VALUES ({marks})", layer["features"])
                if layer.get("style"):
                    _add_style(connection, name, layer["style"])
                x0, y0, x1, y1 = layer["bounds"]
                connection.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y, max_x, max_y, srs_id) "
                                   "VALUES (?,?,?,?,?,?,?,?)", (name, "features", name, x0, y0, x1, y1, _SRS_ID))
                connection.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,0,0)",
                                   (name, "geom", layer["geometry"], _SRS_ID))
        connection.close()
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _add_style(connection, table: str, qml: str) -> None:
    """QGIS's ``layer_styles`` table (created on first use) with ``qml`` as the default style of ``table``."""
    connection.execute("""CREATE TABLE IF NOT EXISTS layer_styles (id INTEGER PRIMARY KEY AUTOINCREMENT,
        f_table_catalog TEXT(256), f_table_schema TEXT(256), f_table_name TEXT(256), f_geometry_column TEXT(256),
        styleName TEXT(30), styleQML TEXT, styleSLD TEXT, useAsDefault BOOLEAN, description TEXT, owner TEXT(30),
        ui TEXT(30), update_time DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""")
    connection.execute("INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column, "
                       "styleName, styleQML, styleSLD, useAsDefault, description, owner) VALUES ('', '', ?, 'geom', ?, ?, '', 1, ?, '')",
                       (table, "chain length", qml, "Colour and width by chain length (lunarscout)"))


def _bounds(xs, ys):
    if not len(xs):
        return (0.0, 0.0, 0.0, 0.0)
    return (float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def output_paths(directory: str | Path, prefix: str = "safe_haven_stage_4") -> dict[str, dict[str, Path]]:
    directory = Path(directory)
    return {name: {"raster": directory / f"{prefix}_{name}.tif", "dilated": directory / f"{prefix}_{name}_dilated.tif",
                   "geopackage": directory / f"{prefix}_{name}.gpkg"} for name in CHAIN_SETS}


def run_chain_maps(
    output_directory: str | Path, *, chains_path: str | Path, dem_path: str | Path, dilate_pixels: int = 2,
    prefix: str = "safe_haven_stage_4", overwrite: bool = False, compiled: bool = True,
    light_store_path: str | Path | None = None, passable_path: str | Path | None = None,
    flood_settings=None, workers: int = 1, progress=None, psr_ids_path: str | Path | None = None,
) -> ChainMapsReport:
    """Write the six products and ``<prefix>_chain_maps.report.json`` into ``output_directory``.

    ``psr_ids_path`` is the PSR id raster (``generate_psr_regions.py``): route pixels with a PSR pixel among their
    eight neighbours are aqua in the images. Without it none are. With ``light_store_path`` (and the stage-3
    ``passable_path``) the images follow the rover's route along every predecessor edge, found by flooding the
    predecessors again (``workers`` processes) and tracing back; without it they are straight lines between the seed
    pixels. ``flood_settings`` must be the ones stage 3 ran with."""
    started = clock.monotonic()
    if int(dilate_pixels) != dilate_pixels or dilate_pixels < 1:
        raise InputError("dilate_pixels must be a positive integer", code="chain_maps_setting_invalid",
                         details={"dilate_pixels": dilate_pixels})
    directory = Path(output_directory)
    paths = output_paths(directory, prefix)
    report_path = directory / f"{prefix}_chain_maps.report.json"
    targets = [p for group in paths.values() for p in group.values()] + [report_path]
    for path in targets:
        if path.exists() and not overwrite:
            raise InputError(f"{path} exists; pass overwrite=True to replace it", code="chain_maps_output_exists",
                             details={"path": str(path)})
    rows = read_chain_table(chains_path)
    transform, width, height, crs_wkt = dem_header(dem_path)
    arrays, edge_set, chain_set = build_chain_sets(rows)
    if arrays["row"].min() < 0 or arrays["row"].max() >= height or arrays["column"].min() < 0 \
            or arrays["column"].max() >= width:
        raise InputError("a seed lies outside the DEM's grid", code="chain_maps_grid_mismatch",
                         details={"height": height, "width": width})
    chain_lengths = chain_set.line_attributes["chain_length"]
    intervals = build_intervals(chain_lengths)
    route_set = None
    if light_store_path is not None:
        if passable_path is None:
            raise InputError("routes need the passable mask stage 3 cached", code="chain_maps_setting_invalid",
                             details={"passable_path": None})
        from .safe_haven_floods import FloodSettings
        from .safe_haven_paths import reconstruct_paths
        from .safe_haven_stage0 import LightStoreDirectory

        route_set = reconstruct_paths(
            rows, directory=LightStoreDirectory(light_store_path), passable_path=passable_path,
            dem_shape=(height, width), settings=flood_settings or FloodSettings(), workers=workers,
            compiled=compiled, progress=progress)
    psr_mask = None
    if psr_ids_path is not None:
        import rasterio

        with rasterio.open(psr_ids_path) as dataset:
            if (dataset.height, dataset.width) != (height, width):
                raise InputError("the PSR id raster and the DEM differ in size", code="chain_maps_psr_mismatch",
                                 details={"psr": [dataset.height, dataset.width], "dem": [height, width]})
            psr_mask = dataset.read(1) > 0
    directory.mkdir(parents=True, exist_ok=True)
    tags = {
        "LUNARSCOUT_PRODUCT": "safe-haven-chain-image-v1",
        "COLOURS": "intervals of chain length (seeds): " + "; ".join(f"{i.label} {i.colour}" for i in intervals)
                   + f"; havens {HAVEN_COLOUR}; route pixels next to a PSR {PSR_COLOUR}; transparent background",
        "SEGMENTS": ("rover routes: one feasible time-respecting route per edge, from the predecessor's pixel to "
                     "the seed's" if route_set is not None else
                     "straight lines between seed pixel centres: a schematic, not the rover's path"),
        "SOURCE_TABLE": Path(chains_path).name,
    }
    havens = (arrays["row"], arrays["column"])
    marked: dict[str, int] = {}
    for set_name, chain in zip(CHAIN_SETS, (edge_set, chain_set)):
        description = {"predecessor_edges": "every seed's edge from its predecessor, coloured by the longest chain through it",
                       "longest_chain_per_root": "each root's longest chain, coloured by its length"}[set_name]
        for kind, radius in (("raster", 0), ("dilated", int(dilate_pixels))):
            planes = render(chain, havens, intervals, height, width, radius, paths=route_set, psr=psr_mask,
                            compiled=compiled)
            write_image(paths[set_name][kind], planes, intervals, transform=transform, crs_wkt=crs_wkt,
                        tags={**tags, "SET": set_name, "DESCRIPTION": description, "DILATE_PIXELS": str(radius)})
            if kind == "raster":
                marked[set_name] = int((planes[1] > 0).sum())
            del planes
    _write_vectors(paths, arrays, edge_set, chain_set, intervals, transform, crs_wkt)
    report = ChainMapsReport(
        seeds=len(rows), predecessor_edges=int(len(edge_set.segments.length)), chains=len(chain_set.lines),
        isolated_seeds=int(len(chain_set.dots)),
        intervals=[{"label": i.label, "first": i.first, "last": i.last, "colour": i.colour,
                    "chains": int(((chain_lengths >= i.first) & (chain_lengths <= (i.last or 10**9))).sum())}
                   for i in intervals],
        files={n: {k: str(p) for k, p in g.items()} for n, g in paths.items()},
        raster_lines="routes" if route_set is not None else "straight",
        route_pixels=int(len(route_set.rows)) if route_set is not None else 0, psr_marked_pixels=marked,
        seconds=clock.monotonic() - started)
    report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return report


def _write_vectors(paths, arrays, edge_set: ChainSet, chain_set: ChainSet, intervals, transform, crs_wkt) -> None:
    x, y = pixel_centres(transform, arrays["row"], arrays["column"])
    attributes = edge_set.segment_attributes
    ends = edge_set.segment_seeds - 1
    features = [
        (_blob(_wkb_line((x[a], x[b]), (y[a], y[b]))), int(attributes["seed_id"][k]), int(attributes["predecessor_id"][k]),
         int(attributes["root_id"][k]), int(attributes["chain_length"][k]), int(attributes["seed_chain_length"][k]),
         float(attributes["mission_hours"][k]), attributes["root_start_utc"][k])
        for k, (a, b) in enumerate(ends)]
    columns = [("seed_id", "INTEGER"), ("predecessor_id", "INTEGER"), ("root_id", "INTEGER"), ("chain_length", "INTEGER"),
               ("seed_chain_length", "INTEGER"), ("mission_hours", "REAL"), ("root_start_utc", "TEXT")]
    write_geopackage(paths["predecessor_edges"]["geopackage"], crs_wkt, [{
        "name": "predecessor_edges", "geometry": "LINESTRING", "columns": columns, "features": features,
        "style": line_style_qml(intervals),
        "bounds": _bounds(x[ends].ravel() if len(ends) else [], y[ends].ravel() if len(ends) else [])}])

    lines, line_attributes = chain_set.lines, chain_set.line_attributes

    def count(name, k):
        value = line_attributes[name][k]
        return None if value is None else int(value)

    chain_features = [
        (_blob(_wkb_line(x[line - 1], y[line - 1])), int(line_attributes["root_id"][k]), int(line_attributes["end_seed_id"][k]),
         int(line_attributes["chain_length"][k]), float(line_attributes["mission_hours"][k]),
         line_attributes["root_start_utc"][k], line_attributes["seed_ids"][k], count("chain_psr_small", k),
         count("chain_psr_medium", k), count("chain_psr_large", k)) for k, line in enumerate(lines)]
    dots = chain_set.dots - 1
    rows = arrays["rows"]
    dot_features = [(_blob(_wkb_point(x[i], y[i])), int(i + 1), rows[i]["root_start_utc"], float(arrays["mission"][i]))
                    for i in dots]
    every = np.concatenate([x[line - 1] for line in lines]) if lines else np.zeros(0)
    every_y = np.concatenate([y[line - 1] for line in lines]) if lines else np.zeros(0)
    write_geopackage(paths["longest_chain_per_root"]["geopackage"], crs_wkt, [
        # The dots come first: QGIS adds the layers of a file in this order, each above the one before, so the lines end
        # up above the dots.
        {"name": "isolated_seeds", "geometry": "POINT", "features": dot_features, "bounds": _bounds(x[dots], y[dots]),
         "columns": [("seed_id", "INTEGER"), ("root_start_utc", "TEXT"), ("mission_hours", "REAL")],
         "style": point_style_qml()},
        {"name": "chains", "geometry": "LINESTRING", "features": chain_features, "bounds": _bounds(every, every_y),
         "columns": [("root_id", "INTEGER"), ("end_seed_id", "INTEGER"), ("chain_length", "INTEGER"),
                     ("mission_hours", "REAL"), ("root_start_utc", "TEXT"), ("seed_ids", "TEXT"),
                     ("psr_small", "INTEGER"), ("psr_medium", "INTEGER"), ("psr_large", "INTEGER")],
         "style": line_style_qml(intervals)}])
