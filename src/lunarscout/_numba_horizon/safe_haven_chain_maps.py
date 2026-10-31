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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .chain_map_styles import (
    HAVEN_SHADOW_CLASSES, PSR_COLOUR, ROOT_COLOUR, build_intervals, class_of, haven_shadow_classes, haven_shadow_labels,
    line_style_qml, lookup_table, point_style_qml,
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
    hours = np.array([r.get("hibernation_hours") or 0.0 for r in rows], dtype=float)       # the seed's shadow period
    arrays = dict(row=pixel_row, column=pixel_column, root=root, predecessor=predecessor, length=length,
                  mission=mission, through_length=through_length, through_mission=through_mission, rows=rows,
                  hours=hours, shadow=haven_shadow_classes(hours))
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
    def draw_havens(planes, rows, columns, radius, shadow):
        height, width = planes.shape[1], planes.shape[2]
        for i in range(rows.shape[0]):
            stamp(planes, 2, rows[i], columns[i], radius, height, width, shadow[i] + 1)

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
           compiled: bool = True, out=None):
    """The three planes of a set's image: its segments (routes when ``paths`` is given, straight lines otherwise),
    the route pixels next to ``psr`` and every haven (``havens`` is a ``(rows, columns)`` pair), the havens one pixel
    larger than the stamp when there is one, each marked with its shadow-period class + 1 (``havens`` is
    ``(rows, columns)`` or ``(rows, columns, classes)``; without classes every haven is class 0). ``out`` is a ``(3, height, width)`` uint8 array to clear and draw into
    instead of a new one."""
    if out is None:
        planes = np.zeros((_PLANES, height, width), dtype=np.uint8)
    else:
        planes = out
        planes[:] = 0
    if paths is None:
        draw_segments(planes, chain_set.segments, intervals, radius, psr=psr, compiled=compiled)
    else:
        draw_polylines(planes, chain_set.segments, intervals, paths, chain_set.segment_seeds[:, 1], radius, psr=psr,
                       compiled=compiled)
    shadow = np.zeros(len(havens[0]), dtype=np.uint8) if len(havens) < 3 else np.asarray(havens[2], dtype=np.uint8)
    order = np.argsort(-shadow.astype(np.int64), kind="stable")        # the longest shadow first: no shadow (white) ends on top
    _kernel(2, compiled)(planes, np.ascontiguousarray(np.asarray(havens[0])[order], dtype=np.int64),
                         np.ascontiguousarray(np.asarray(havens[1])[order], dtype=np.int64),
                         radius + (1 if radius > 0 else 0), np.ascontiguousarray(shadow[order], dtype=np.uint8))
    return planes


def haven_code(interval_count: int, shadow):
    """The class of ``lookup_table`` of havens of shadow-period class ``shadow`` (0 none, ... 3): the classes run from the
    longest shadow, at ``interval_count + 2``, to none at ``interval_count + 5``."""
    return interval_count + 2 + (len(HAVEN_SHADOW_CLASSES) - 1 - np.asarray(shadow))


def colour_image(planes, intervals) -> npt.NDArray[np.uint8]:
    """``(4, rows, columns)`` RGBA of planes: the route's interval colour, aqua over it where the PSR mark is set, the
    haven's shadow-period colour over both where there is a haven; 0 (nothing) is transparent."""
    table = lookup_table(intervals)
    index = planes[0].astype(np.int64)
    index[planes[1] > 0] = len(intervals) + 1
    haven = planes[2] > 0
    index[haven] = haven_code(len(intervals), planes[2][haven].astype(np.int64) - 1)
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


def _reconstruct_routes(rows, light_store_path, passable_path, dem_shape, flood_settings, workers, compiled,
                        progress):
    """The rover's routes along every predecessor edge (None without ``light_store_path``: straight lines)."""
    if light_store_path is None:
        return None
    if passable_path is None:
        raise InputError("routes need the passable mask stage 3 cached", code="chain_maps_setting_invalid",
                         details={"passable_path": None})
    from .safe_haven_floods import FloodSettings
    from .safe_haven_paths import reconstruct_paths
    from .safe_haven_stage0 import LightStoreDirectory

    return reconstruct_paths(
        rows, directory=LightStoreDirectory(light_store_path), passable_path=passable_path, dem_shape=dem_shape,
        settings=flood_settings or FloodSettings(), workers=workers, compiled=compiled, progress=progress)


def _read_psr_mask(psr_ids_path, height: int, width: int):
    """True where the PSR id raster has a region (None without ``psr_ids_path``)."""
    if psr_ids_path is None:
        return None
    import rasterio

    with rasterio.open(psr_ids_path) as dataset:
        if (dataset.height, dataset.width) != (height, width):
            raise InputError("the PSR id raster and the DEM differ in size", code="chain_maps_psr_mismatch",
                             details={"psr": [dataset.height, dataset.width], "dem": [height, width]})
        return dataset.read(1) > 0


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
    route_set = _reconstruct_routes(rows, light_store_path, passable_path, (height, width), flood_settings, workers,
                                    compiled, progress)
    psr_mask = _read_psr_mask(psr_ids_path, height, width)
    directory.mkdir(parents=True, exist_ok=True)
    tags = {
        "LUNARSCOUT_PRODUCT": "safe-haven-chain-image-v1",
        "COLOURS": "intervals of chain length (seeds): " + "; ".join(f"{i.label} {i.colour}" for i in intervals)
                   + "; havens by shadow period: " + "; ".join(f"{label} {colour}" for label, (_b, colour) in zip(
                       haven_shadow_labels(), HAVEN_SHADOW_CLASSES))
                   + f"; route pixels next to a PSR {PSR_COLOUR}; transparent background",
        "SEGMENTS": ("rover routes: one feasible time-respecting route per edge, from the predecessor's pixel to "
                     "the seed's" if route_set is not None else
                     "straight lines between seed pixel centres: a schematic, not the rover's path"),
        "SOURCE_TABLE": Path(chains_path).name,
    }
    havens = (arrays["row"], arrays["column"], arrays["shadow"])
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


@dataclass
class YearlyChainMapsReport:
    years: dict                      # per year: the file, the chains and the havens drawn
    intervals: list
    raster_lines: str
    route_pixels: int
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def yearly_path(directory: str | Path, year: int, prefix: str = "safe_haven_stage_4") -> Path:
    return Path(directory) / f"{prefix}_longest_chain_per_root_{year}.tif"


def run_yearly_chain_maps(
    output_directory: str | Path, *, chains_path: str | Path, dem_path: str | Path, years=None,
    prefix: str = "safe_haven_stage_4", overwrite: bool = False, compiled: bool = True,
    light_store_path: str | Path | None = None, passable_path: str | Path | None = None,
    flood_settings=None, workers: int = 1, progress=None, psr_ids_path: str | Path | None = None,
) -> YearlyChainMapsReport:
    """Write ``<prefix>_longest_chain_per_root_<year>.tif`` for each of ``years`` (None: every UTC year from the first
    to the last in which a chain's root starts) and ``<prefix>_yearly_chain_maps.report.json``.

    Each image is the non-dilated ``longest_chain_per_root`` image restricted to one year: a chain belongs to the UTC
    year in which its root starts (``root_start_utc``; a chain that carries on into the next year stays whole in the
    year it began), and the havens drawn are the seeds of the chains of that year, isolated ones too. The colours are
    the intervals of all the chains, so a colour means the same length in every year. A year with no chains is a
    transparent image. The routes, PSR marks and other arguments are those of :func:`run_chain_maps`."""
    started = clock.monotonic()
    directory = Path(output_directory)
    rows = read_chain_table(chains_path)
    transform, width, height, crs_wkt = dem_header(dem_path)
    arrays, _edge_set, chain_set = build_chain_sets(rows)
    if arrays["row"].min() < 0 or arrays["row"].max() >= height or arrays["column"].min() < 0 \
            or arrays["column"].max() >= width:
        raise InputError("a seed lies outside the DEM's grid", code="chain_maps_grid_mismatch",
                         details={"height": height, "width": width})
    seed_year = np.array([int(str(r["root_start_utc"])[:4]) for r in rows], dtype=np.int64)
    line_year = np.array([int(str(t)[:4]) for t in chain_set.line_attributes["root_start_utc"]], dtype=np.int64)
    if years is None:
        years = range(int(seed_year.min()), int(seed_year.max()) + 1)
    years = [int(y) for y in years]
    if not years or len(set(years)) != len(years):
        raise InputError("years must be a non-empty list of distinct years", code="chain_maps_setting_invalid",
                         details={"years": years})
    targets = [yearly_path(directory, y, prefix) for y in years] + [directory / f"{prefix}_yearly_chain_maps.report.json"]
    for path in targets:
        if path.exists() and not overwrite:
            raise InputError(f"{path} exists; pass overwrite=True to replace it", code="chain_maps_output_exists",
                             details={"path": str(path)})
    intervals = build_intervals(chain_set.line_attributes["chain_length"])
    route_set = _reconstruct_routes(rows, light_store_path, passable_path, (height, width), flood_settings, workers,
                                    compiled, progress)
    psr_mask = _read_psr_mask(psr_ids_path, height, width)
    directory.mkdir(parents=True, exist_ok=True)
    segment_year = np.repeat(line_year, [len(line) - 1 for line in chain_set.lines])
    tags = {
        "LUNARSCOUT_PRODUCT": "safe-haven-chain-image-v1",
        "COLOURS": "intervals of chain length (seeds): " + "; ".join(f"{i.label} {i.colour}" for i in intervals)
                   + "; havens by shadow period: " + "; ".join(f"{label} {colour}" for label, (_b, colour) in zip(
                       haven_shadow_labels(), HAVEN_SHADOW_CLASSES))
                   + f"; route pixels next to a PSR {PSR_COLOUR}; transparent background",
        "SEGMENTS": ("rover routes: one feasible time-respecting route per edge, from the predecessor's pixel to "
                     "the seed's" if route_set is not None else
                     "straight lines between seed pixel centres: a schematic, not the rover's path"),
        "SOURCE_TABLE": Path(chains_path).name, "SET": "longest_chain_per_root", "DILATE_PIXELS": "0",
    }
    result: dict = {}
    for year in years:
        keep = segment_year == year
        segments = chain_set.segments
        subset = replace(
            chain_set, segments=Segments(segments.row0[keep], segments.column0[keep], segments.row1[keep],
                                         segments.column1[keep], segments.length[keep]),
            segment_seeds=chain_set.segment_seeds[keep])
        in_year = seed_year == year
        planes = render(subset, (arrays["row"][in_year], arrays["column"][in_year], arrays["shadow"][in_year]), intervals,
                        height, width, 0, paths=route_set, psr=psr_mask, compiled=compiled)
        path = yearly_path(directory, year, prefix)
        write_image(path, planes, intervals, transform=transform, crs_wkt=crs_wkt,
                    tags={**tags, "YEAR": str(year),
                          "DESCRIPTION": f"the longest chain of each root that starts in {year}, coloured by its length"})
        del planes
        result[year] = {"file": str(path), "chains": int((line_year == year).sum()), "segments": int(keep.sum()),
                        "havens": int(in_year.sum())}
    report = YearlyChainMapsReport(
        years=result,
        intervals=[{"label": i.label, "first": i.first, "last": i.last, "colour": i.colour} for i in intervals],
        raster_lines="routes" if route_set is not None else "straight",
        route_pixels=int(len(route_set.rows)) if route_set is not None else 0, seconds=clock.monotonic() - started)
    (directory / f"{prefix}_yearly_chain_maps.report.json").write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return report


def _rgb_of(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def month_range(first: str, last: str) -> list[str]:
    """``["YYYY-MM", ...]`` from ``first`` to ``last`` inclusive."""
    year, month = int(first[:4]), int(first[5:7])
    stop = int(last[:4]) * 12 + int(last[5:7])
    names = []
    while year * 12 + month <= stop:
        names.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return names


def seed_month_spans(rows: list[dict]):
    """``(start_index, last_index, months)`` for the seeds of the stage-4 table: the month names from the first in which a
    chain starts to the last month of any seed, and for each seed the index of the month its chain starts in
    (``root_start_utc``) and of the last month it is shown in: its own ``month``, never before the chain's start. A seed
    is shown from the start month to the last month; the table's months never go back along a chain."""
    def number(name: str) -> int:
        return int(name[:4]) * 12 + int(name[5:7]) - 1

    start = np.array([number(str(r["root_start_utc"])[:7]) for r in rows], dtype=np.int64)
    last = np.maximum(np.array([number(str(r["month"])) for r in rows], dtype=np.int64), start)
    first = int(start.min())
    final = int(last.max())
    names = month_range(f"{first // 12:04d}-{first % 12 + 1:02d}", f"{final // 12:04d}-{final % 12 + 1:02d}")
    return start - first, last - first, names


def current_chain_starts(lines, start_index, last_index):
    """``(start_seed, start_month, lone_seed, lone_month)`` for the longest chains ``lines`` (arrays of 1-based seed ids,
    root first) and the seeds' ``start_index`` and ``last_index`` from ``seed_month_spans``: the 0-based table index of the
    seed that is the current start of a chain in a month, and the month's index, then the same for the lone seeds.

    A seed is passed once its last month is over. In month M a chain's current start is its first seed that is not passed;
    it is a start only while the chain has 2 or more seeds left, so it is also the head of a route drawn that month. A chain
    with only its final seed left in M has that seed as its lone seed in M. The chain's root is the current start of the
    month it starts in. Each list is ordered by month, then by seed."""
    empty = np.zeros(0, dtype=np.int64)
    if not len(lines):
        return empty, empty, empty, empty
    lengths = np.array([len(line) for line in lines], dtype=np.int64)
    seed = np.concatenate(lines).astype(np.int64) - 1
    first = np.cumsum(lengths) - lengths
    position = np.arange(len(seed)) - np.repeat(first, lengths)
    last = last_index[seed]
    # The months in which a seed is the first one not passed: after the seed before it is passed, from the chain's start.
    previous = np.where(position == 0, -1, np.roll(last, 1))
    low = np.maximum(np.repeat(start_index[seed[first]], lengths), previous + 1)
    count = np.maximum(last - low + 1, 0)
    pairs_seed = np.repeat(seed, count)
    pairs_month = np.repeat(low, count) + (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count))
    final = position == np.repeat(lengths, lengths) - 1
    keep = np.repeat(~final, count)
    out = []
    for selected in (keep, ~keep):
        order = np.lexsort((pairs_seed[selected], pairs_month[selected]))
        out += [pairs_seed[selected][order], pairs_month[selected][order]]
    return tuple(out)


PSR_SIZES = ("small", "medium", "large")


def line_psr_levels(chain_set):
    """``(levels, known)`` for the longest chains of ``chain_set``: the largest PSR size class each chain's routes pass next
    to (0 none, 1 small, 2 medium, 3 large, from ``chain_psr_small``, ``chain_psr_medium`` and ``chain_psr_large`` of the
    stage-4 table, which cover the whole chain from its root to its last seed), and whether every count is known (they are
    blank when stage 4 ran without the routes; ``levels`` is then 0 for a blank chain)."""
    counts = [chain_set.line_attributes[f"chain_psr_{size}"] for size in PSR_SIZES]
    known = all(value is not None for column in counts for value in column)
    small, medium, large = (np.array([0 if value is None else int(value) for value in column], dtype=np.int64) for column in counts)
    return np.where(large > 0, 3, np.where(medium > 0, 2, np.where(small > 0, 1, 0))).astype(np.uint8), known


@dataclass
class MonthlyChainMapsReport:
    file: str
    months: dict                     # per month: the chains drawn and the ones that start in it, the pixels and the havens
    intervals: list
    raster_lines: str
    route_pixels: int
    pixels: int
    seconds: float = 0.0
    psr_files: dict | None = None    # per PSR size: the file of the chains that touch a PSR of at least that size, its pixels

    def as_dict(self) -> dict:
        return asdict(self)


def monthly_path(directory: str | Path, prefix: str = "safe_haven_stage_4", psr_size: str | None = None) -> Path:
    """The monthly chain pixel file; with ``psr_size`` (``"small"``, ``"medium"`` or ``"large"``) the file of the chains that
    touch a PSR of at least that size."""
    if psr_size is None:
        return Path(directory) / f"{prefix}_monthly_chain_pixels.npz"
    if psr_size not in PSR_SIZES:
        raise InputError(f"psr_size must be one of {PSR_SIZES}", code="chain_maps_setting_invalid", details={"psr_size": psr_size})
    return Path(directory) / f"{prefix}_monthly_chain_pixels_psr_{psr_size}.npz"


def run_monthly_chain_maps(
    output_directory: str | Path, *, chains_path: str | Path, dem_path: str | Path, prefix: str = "safe_haven_stage_4",
    overwrite: bool = False, compiled: bool = True, light_store_path: str | Path | None = None,
    passable_path: str | Path | None = None, flood_settings=None, workers: int = 1, progress=None,
    psr_ids_path: str | Path | None = None, seed_areas_path: str | Path | None = None,
    psr_sizes: tuple[str, ...] = (),
) -> MonthlyChainMapsReport:
    """Write ``<prefix>_monthly_chain_pixels.npz`` and ``<prefix>_monthly_chain_maps.report.json``: the pixels of the
    longest-chain image of each month, for the viewer (``scripts/view_safe_haven.py``).

    A chain is drawn from the month its root starts in (``root_start_utc``) to the month of its last seed, each month as
    the part still ahead: its seeds from that month on, so a chain that carries on into later months is shorter in each
    of them and takes the colour of its remaining length (the intervals of the whole chains, as in the yearly images).
    A seed's month is its own ``month`` in the stage-4 table, but never before the month its chain starts, and a seed is
    passed once its month is over. The havens drawn in a month are those of the seed haven areas of the month
    (``safe_haven_stage_4_seed_haven_areas.npz``, from ``seed_areas_path`` or beside the chain table): the current start
    of each chain (its first seed not passed), the havens near it and the lone last seeds, in their shadow colours; an
    ``InputError`` with code ``chain_maps_seed_areas_missing`` is raised when that file does not exist. The routes and
    PSR marks are those of the yearly images, and the chains and routes of the month a chain starts in are what the
    yearly image shows when every chain starts in one month. A month is stored as only the pixels drawn: ``row``,
    ``column`` and the colour class ``code`` (a row of ``table``, RGBA: 0 is transparent, then the length intervals, aqua
    for a PSR pass, white for a haven and magenta for the start of a chain, one pixel on the current start of each chain
    drawn in the month, in addition to the haven there), ordered by code so that drawing them in order puts the longer
    chains, then aqua, then the havens, then the starts on top; ``offsets`` holds the start of each month's slice
    (``months`` are the names, ``chains`` the chains of 2 or more seeds drawn in each, and ``started`` those that start in
    it).

    ``psr_sizes`` (some of ``"small"``, ``"medium"`` and ``"large"``) also writes, for each, ``<prefix>_monthly_chain_pixels_psr_
    <size>.npz``: the same picture of only the chains whose routes pass next to a PSR of at least that size (small: any;
    medium: a medium or large one; large: a large one), by the whole chain's ``chain_psr_*`` counts in the stage-4 table, so
    a chain qualifies in every month it is drawn. The havens drawn are those of the areas file whose ``psr_level`` is at
    least the size (the havens near a qualifying chain's start and its lone last seed). An ``InputError`` with code
    ``chain_maps_psr_counts_missing`` is raised when the table's counts are blank (stage 4 ran without the routes)."""
    started = clock.monotonic()
    directory = Path(output_directory)
    rows = read_chain_table(chains_path)
    transform, width, height, _crs_wkt = dem_header(dem_path)
    arrays, _edge_set, chain_set = build_chain_sets(rows)
    if arrays["row"].min() < 0 or arrays["row"].max() >= height or arrays["column"].min() < 0 \
            or arrays["column"].max() >= width:
        raise InputError("a seed lies outside the DEM's grid", code="chain_maps_grid_mismatch",
                         details={"height": height, "width": width})
    unknown = [size for size in psr_sizes if size not in PSR_SIZES]
    if unknown:
        raise InputError(f"psr_sizes must be among {PSR_SIZES}", code="chain_maps_setting_invalid",
                         details={"psr_sizes": list(psr_sizes)})
    line_level, counts_known = line_psr_levels(chain_set)
    if psr_sizes and not counts_known:
        raise InputError("the chain table has no PSR counts (stage 4 ran without the routes), so the chains that touch a "
                         "PSR cannot be told", code="chain_maps_psr_counts_missing")
    target, report_path = monthly_path(directory, prefix), directory / f"{prefix}_monthly_chain_maps.report.json"
    psr_targets = {size: monthly_path(directory, prefix, size) for size in psr_sizes}
    for path in (target, report_path, *psr_targets.values()):
        if path.exists() and not overwrite:
            raise InputError(f"{path} exists; pass overwrite=True to replace it", code="chain_maps_output_exists",
                             details={"path": str(path)})
    start_index, last_index, months = seed_month_spans(rows)
    from ..safe_haven_areas import areas_path, read_seed_areas
    areas_file = Path(seed_areas_path) if seed_areas_path is not None else areas_path(Path(chains_path).parent)
    if not areas_file.is_file():
        raise InputError(f"{areas_file} does not exist; the monthly chain pixels draw the seed haven areas (run "
                         "generate_seed_haven_areas first)", code="chain_maps_seed_areas_missing",
                         details={"path": str(areas_file)})
    areas = read_seed_areas(areas_file)
    if areas["months"] != list(months):
        raise InputError(f"{areas_file} is for other months than the chain table", code="chain_maps_seed_areas_mismatch",
                         details={"path": str(areas_file), "areas": [areas["months"][0], areas["months"][-1]],
                                  "chains": [months[0], months[-1]]})
    if len(areas["row"]) and (areas["row"].max() >= height or areas["column"].max() >= width):
        raise InputError("a seed haven area pixel lies outside the DEM's grid", code="chain_maps_grid_mismatch",
                         details={"height": height, "width": width})
    area_class = haven_shadow_classes(areas["hours"])
    intervals = build_intervals(chain_set.line_attributes["chain_length"])
    table = np.concatenate([lookup_table(intervals), [(*_rgb_of(ROOT_COLOUR), 255)]]).astype(np.uint8)
    route_set = _reconstruct_routes(rows, light_store_path, passable_path, (height, width), flood_settings, workers,
                                    compiled, progress)
    psr_mask = _read_psr_mask(psr_ids_path, height, width)
    # The longest chain of each root: its seeds (root first) with their last months, and its segments with the line they
    # belong to and the last month of their tail seed.
    lines = chain_set.lines
    line_start = np.array([start_index[line[0] - 1] for line in lines], dtype=np.int64)
    entry_line = np.repeat(np.arange(len(lines)), [len(line) for line in lines])
    entry_last = np.concatenate([last_index[line - 1] for line in lines]) if lines else np.zeros(0, dtype=np.int64)
    segment_line = np.repeat(np.arange(len(lines)), [len(line) - 1 for line in lines])
    is_tail = np.ones(len(entry_line), dtype=bool)
    if lines:
        is_tail[np.cumsum([len(line) for line in lines]) - 1] = False
    segment_last = entry_last[is_tail]
    planes = np.zeros((_PLANES, height, width), dtype=np.uint8)
    aqua_code, root_code = len(intervals) + 1, len(intervals) + 2 + len(HAVEN_SHADOW_CLASSES)
    start_seed, start_month, _lone_seed, _lone_month = current_chain_starts(lines, start_index, last_index)
    level_of_root = dict(zip((int(line[0]) for line in lines), line_level.tolist()))
    start_level = np.array([level_of_root[int(rows[i]["root_id"])] for i in start_seed], dtype=np.uint8)
    # The pictures: the one of every chain and, for each PSR size, the one of the chains that touch a PSR of at least that size.
    variants = {None: 0, **{size: PSR_SIZES.index(size) + 1 for size in psr_sizes}}
    found = {key: dict(rows=[], columns=[], codes=[], offsets=[0], chains=[], started=[]) for key in variants}
    result = {}
    segments = chain_set.segments
    for index, month in enumerate(months):
        window = slice(areas["offsets"][index], areas["offsets"][index + 1])            # the havens of the month
        remaining = np.bincount(entry_line, weights=(entry_last >= index), minlength=len(lines)).astype(np.int64)
        for key, minimum in variants.items():
            line_ok = line_level >= minimum
            active = (line_start[segment_line] <= index) & (segment_last >= index) & line_ok[segment_line]
            shown = np.arange(window.start, window.stop)[areas["psr_level"][window] >= minimum]
            subset = replace(
                chain_set, segments=Segments(segments.row0[active], segments.column0[active], segments.row1[active],
                                             segments.column1[active], remaining[segment_line[active]]),
                segment_seeds=chain_set.segment_seeds[active])
            month_rows, month_columns, month_codes = [], [], []
            if active.any() or len(shown):
                render(subset, (areas["row"][shown], areas["column"][shown], area_class[shown]), intervals, height,
                       width, 0, paths=route_set, psr=psr_mask, compiled=compiled, out=planes)
                for first in range(0, height, _STRIP_ROWS):
                    strip = planes[:, first:first + _STRIP_ROWS]
                    code = strip[0].copy()
                    code[strip[1] > 0] = aqua_code
                    code[strip[2] > 0] = haven_code(len(intervals), strip[2][strip[2] > 0].astype(np.int64) - 1)
                    found_r, found_c = np.nonzero(code)
                    month_rows.append((found_r + first).astype(np.int32))
                    month_columns.append(found_c.astype(np.int32))
                    month_codes.append(code[found_r, found_c])
                starting = start_seed[(start_month == index) & (start_level >= minimum)]    # the current start of each chain drawn
                month_rows.append(arrays["row"][starting].astype(np.int32))
                month_columns.append(arrays["column"][starting].astype(np.int32))
                month_codes.append(np.full(len(starting), root_code, dtype=np.uint8))
            month_r = np.concatenate(month_rows) if month_rows else np.zeros(0, dtype=np.int32)
            month_c = np.concatenate(month_columns) if month_columns else np.zeros(0, dtype=np.int32)
            month_k = np.concatenate(month_codes) if month_codes else np.zeros(0, dtype=np.uint8)
            order = np.argsort(month_k, kind="stable")
            held = found[key]
            held["rows"].append(month_r[order])
            held["columns"].append(month_c[order])
            held["codes"].append(month_k[order])
            held["offsets"].append(held["offsets"][-1] + len(order))
            held["chains"].append(int(np.unique(segment_line[active]).size))
            held["started"].append(int(((line_start == index) & line_ok).sum()))
            if key is None:
                result[month] = {"chains": held["chains"][-1], "started": held["started"][-1], "pixels": len(order),
                                 "havens": int(len(shown))}
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.array([i.label for i in intervals] + ["next to a PSR"]
                      + [f"haven, shadow {label}" for label in reversed(haven_shadow_labels())] + ["start of a chain"])
    stagings = {}
    try:
        for key, held in found.items():
            path = target if key is None else psr_targets[key]
            stagings[path] = path.with_name(path.name + ".partial")
            with stagings[path].open("wb") as handle:
                extra = {} if key is None else {"psr_size": np.array(key)}
                np.savez_compressed(
                    handle, months=np.array(months), offsets=np.array(held["offsets"], dtype=np.int64),
                    row=np.concatenate(held["rows"]), column=np.concatenate(held["columns"]),
                    code=np.concatenate(held["codes"]), table=table, chains=np.array(held["chains"], dtype=np.int64),
                    started=np.array(held["started"], dtype=np.int64), labels=labels, **extra)
        for path, staging in stagings.items():
            os.replace(staging, path)
    finally:
        for staging in stagings.values():
            staging.unlink(missing_ok=True)
    report = MonthlyChainMapsReport(
        file=str(target), months=result,
        intervals=[{"label": i.label, "first": i.first, "last": i.last, "colour": i.colour} for i in intervals],
        raster_lines="routes" if route_set is not None else "straight",
        route_pixels=int(len(route_set.rows)) if route_set is not None else 0, pixels=int(found[None]["offsets"][-1]),
        seconds=clock.monotonic() - started,
        psr_files={size: {"file": str(psr_targets[size]), "pixels": int(found[size]["offsets"][-1]),
                          "chains": int(sum(found[size]["chains"]))} for size in psr_sizes} or None)
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
    dot_features = [(_blob(_wkb_point(x[i], y[i])), int(i + 1), rows[i]["root_start_utc"], float(arrays["mission"][i]),
                     float(arrays["hours"][i])) for i in dots]
    every = np.concatenate([x[line - 1] for line in lines]) if lines else np.zeros(0)
    every_y = np.concatenate([y[line - 1] for line in lines]) if lines else np.zeros(0)
    write_geopackage(paths["longest_chain_per_root"]["geopackage"], crs_wkt, [
        # The dots come first: QGIS adds the layers of a file in this order, each above the one before, so the lines end
        # up above the dots.
        {"name": "isolated_seeds", "geometry": "POINT", "features": dot_features, "bounds": _bounds(x[dots], y[dots]),
         "columns": [("seed_id", "INTEGER"), ("root_start_utc", "TEXT"), ("mission_hours", "REAL"),
                     ("hibernation_hours", "REAL")],
         "style": point_style_qml()},
        {"name": "chains", "geometry": "LINESTRING", "features": chain_features, "bounds": _bounds(every, every_y),
         "columns": [("root_id", "INTEGER"), ("end_seed_id", "INTEGER"), ("chain_length", "INTEGER"),
                     ("mission_hours", "REAL"), ("root_start_utc", "TEXT"), ("seed_ids", "TEXT"),
                     ("psr_small", "INTEGER"), ("psr_medium", "INTEGER"), ("psr_large", "INTEGER")],
         "style": line_style_qml(intervals)}])
