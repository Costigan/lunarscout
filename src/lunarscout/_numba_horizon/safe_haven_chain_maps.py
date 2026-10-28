"""Maps of the stage-4 chains: rasters and GeoPackages (private).

Two chain sets are drawn from the stage-4 table, each as a full-resolution raster, a copy with dilated lines
and a GeoPackage:

* **predecessor edges**: one straight segment from every seed that has a predecessor to the seed (the label forest
  of stage 4: a root's chains branch, and every edge is drawn once). An edge takes the value of the longest chain
  that passes through it (the deepest seed at or below its target), so a chain does not change colour along its
  length.
* **longest chain per root**: for every root that has descendants, the chain with the most seeds (ties: the longer
  mission, then the lowest seed id), drawn root to end, all of it at the chain's length. Seeds that are alone (a
  root with no descendants) are drawn as dots at the value 1.

The rasters have the DEM's grid and three uint16 bands: the chain length (seeds; 0 is background), the month in
which the chain's root starts operations (1 is the table's first month) and the mission hours of the chain (the
chain's end seed; for predecessor edges the longest mission among the seeds at or below the edge's target).
Segments are drawn in ascending order of (length, mission), so where lines cross, all three bands come from the
longer chain. A segment is straight between the centres of its two seed pixels: it is a schematic, not the
rover's path (stage 3 keeps frontiers, not paths), and it may cross terrain the rover cannot. The dilated copy
stamps a square of ``2 * dilate + 1`` pixels along every segment (dots one pixel larger).

The GeoPackages hold the same sets as vectors, not dilated, with attributes: a ``predecessor_edges`` line layer
(one segment per edge) or a ``chains`` line layer (one line through the seeds of each chain) and an
``isolated_seeds`` point layer.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import time as clock
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .pixel_coordinates import dem_header, pixel_centres
from .safe_haven_chains import read_chain_table

BANDS = ("chain_length_seeds", "root_start_month", "mission_hours")
CHAIN_SETS = ("predecessor_edges", "longest_chain_per_root")
_UINT16_MAX = 65535
_SRS_ID = 100000
_GPKG_APPLICATION_ID = 0x47504B47
_STRIP_ROWS = 1024


@dataclass
class ChainMapsReport:
    seeds: int
    predecessor_edges: int
    chains: int                      # roots with descendants: one longest chain each
    isolated_seeds: int
    files: dict
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# The chain sets
# ---------------------------------------------------------------------------


def month_origin(rows: list[dict]) -> tuple[int, int]:
    """``(year, month)`` of the earliest seed month or root start in the table."""
    first = min(min(r["month"] for r in rows), min(r["root_start_utc"][:7] for r in rows))
    return int(first[:4]), int(first[5:7])


def month_index(stamps: list[str], origin: tuple[int, int]) -> npt.NDArray[np.int64]:
    """1-based months since ``origin`` of ISO UTC stamps."""
    return np.array([(int(t[:4]) - origin[0]) * 12 + int(t[5:7]) - origin[1] + 1 for t in stamps], dtype=np.int64)


@dataclass
class Segments:
    """Segments to draw: pixel ends and the three band values, in arbitrary order."""

    row0: npt.NDArray[np.int64]
    column0: npt.NDArray[np.int64]
    row1: npt.NDArray[np.int64]
    column1: npt.NDArray[np.int64]
    length: npt.NDArray[np.int64]
    month: npt.NDArray[np.int64]
    mission: npt.NDArray[np.int64]

    def ascending(self) -> npt.NDArray[np.int64]:
        """The order to draw in: by length, then mission, so that the longer chain is on top."""
        return np.lexsort((self.mission, self.length))


@dataclass
class ChainSet:
    """What one product shows: segments for the rasters; ``lines`` (vertex lists as seed-id arrays with their
    attributes) and ``dots`` for the vectors."""

    segments: Segments
    segment_attributes: dict          # columns of the segment layer, one array each
    segment_seeds: npt.NDArray        # (n, 2) 1-based seed ids of each segment's ends
    lines: list                       # longest-chain set only: seed-id lists
    line_attributes: dict
    dots: npt.NDArray                 # 1-based seed ids drawn as dots
    dot_segments: Segments | None = None      # the dots as zero-length segments (rasters)


def build_chain_sets(rows: list[dict]):
    """``(arrays, predecessor_edges, longest_chain_per_root)`` from the stage-4 table: the per-seed arrays used and
    the two :class:`ChainSet` s."""
    count = len(rows)
    if [r["seed_id"] for r in rows] != list(range(1, count + 1)):
        raise InputError("the chain table's seeds are not 1..N in order", code="chain_maps_table_invalid")
    def field(name, dtype=np.int64):
        return np.array([r[name] for r in rows], dtype=dtype)

    pixel_row, pixel_column = field("row"), field("column")
    predecessor, root, length = field("predecessor_id"), field("root_id"), field("chain_length")
    mission = field("mission_hours", float)
    origin = month_origin(rows)
    month = month_index([r["root_start_utc"] for r in rows], origin)
    root_start = [r["root_start_utc"] for r in rows]
    if count == 0:
        raise InputError("the chain table has no seeds", code="chain_maps_table_invalid")
    mission_u16 = np.minimum(np.rint(mission), _UINT16_MAX).astype(np.int64)

    # The longest chain through each seed and the longest mission at or below it, from the leaves up.
    through_length, through_mission = length.copy(), mission_u16.copy()
    for level in range(int(length.max()), 1, -1):
        members = np.flatnonzero(length == level)
        np.maximum.at(through_length, predecessor[members] - 1, through_length[members])
        np.maximum.at(through_mission, predecessor[members] - 1, through_mission[members])

    # --- predecessor edges
    heads = np.flatnonzero(predecessor > 0)
    tails = predecessor[heads] - 1
    edge_segments = Segments(pixel_row[tails], pixel_column[tails], pixel_row[heads], pixel_column[heads],
                             through_length[heads], month[heads], through_mission[heads])
    edge_set = ChainSet(
        segments=edge_segments,
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
    chain_ids = np.array([line[0] for line in lines], dtype=np.int64)
    segment_ends, segment_length, segment_month, segment_mission = [], [], [], []
    for line, end in zip(lines, long_ends):
        for a, b in zip(line[:-1], line[1:]):
            segment_ends.append((a - 1, b - 1))
            segment_length.append(length[end])                 # all of a chain at the chain's own length
            segment_month.append(month[b - 1])
            segment_mission.append(mission_u16[end])
    pair = np.array(segment_ends, dtype=np.int64).reshape(-1, 2)
    chain_segments = Segments(pixel_row[pair[:, 0]], pixel_column[pair[:, 0]], pixel_row[pair[:, 1]],
                              pixel_column[pair[:, 1]], np.array(segment_length, dtype=np.int64),
                              np.array(segment_month, dtype=np.int64), np.array(segment_mission, dtype=np.int64))
    dots = Segments(pixel_row[isolated], pixel_column[isolated], pixel_row[isolated], pixel_column[isolated],
                    length[isolated], month[isolated], mission_u16[isolated])
    chain_set = ChainSet(
        segments=chain_segments,
        segment_attributes={},
        segment_seeds=pair + 1,
        lines=lines,
        line_attributes={
            "root_id": chain_ids, "end_seed_id": long_ends + 1, "chain_length": length[long_ends],
            "mission_hours": mission[long_ends], "root_start_utc": [root_start[i] for i in long_ends],
            "seed_ids": [",".join(str(s) for s in line) for line in lines]},
        dots=isolated + 1, dot_segments=dots)
    arrays = dict(row=pixel_row, column=pixel_column, root=root, predecessor=predecessor, length=length,
                  mission=mission, month=month, through_length=through_length, through_mission=through_mission,
                  origin=origin, rows=rows)
    return arrays, edge_set, chain_set


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------


def _draw_python(bands, row0, column0, row1, column1, length, month, mission, order, radius):
    height, width = bands.shape[1], bands.shape[2]
    for k in range(order.shape[0]):
        i = order[k]
        r, c = row0[i], column0[i]
        dr, dc = abs(row1[i] - r), abs(column1[i] - c)
        sr = 1 if row1[i] > r else -1
        sc = 1 if column1[i] > c else -1
        err = dc - dr
        while True:
            for rr in range(max(r - radius, 0), min(r + radius, height - 1) + 1):
                for cc in range(max(c - radius, 0), min(c + radius, width - 1) + 1):
                    bands[0, rr, cc] = length[i]
                    bands[1, rr, cc] = month[i]
                    bands[2, rr, cc] = mission[i]
            if r == row1[i] and c == column1[i]:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr


_KERNEL = {}


def draw_segments(bands, segments: Segments, radius: int = 0, *, compiled: bool = True) -> None:
    """Draw ``segments`` into ``bands`` (3, height, width) uint16, in ascending order of (length, mission), each
    with a square stamp of ``2 * radius + 1`` pixels along a Bresenham line between its pixel ends. Ends outside
    the raster are clipped."""
    if not len(segments.length):
        return
    draw = _draw_python
    if compiled:
        if "draw" not in _KERNEL:
            from numba import njit

            _KERNEL["draw"] = njit(cache=False, nogil=True)(_draw_python)
        draw = _KERNEL["draw"]
    values = [np.ascontiguousarray(a, dtype=np.int64) for a in (
        segments.row0, segments.column0, segments.row1, segments.column1)]
    draw(bands, *values, np.ascontiguousarray(np.minimum(segments.length, _UINT16_MAX), dtype=np.uint16),
         np.ascontiguousarray(np.minimum(segments.month, _UINT16_MAX), dtype=np.uint16),
         np.ascontiguousarray(np.minimum(segments.mission, _UINT16_MAX), dtype=np.uint16),
         np.ascontiguousarray(segments.ascending(), dtype=np.int64), int(radius))


def _concatenate(first: Segments, second: Segments) -> Segments:
    return Segments(*(np.concatenate([getattr(first, f), getattr(second, f)]) for f in (
        "row0", "column0", "row1", "column1", "length", "month", "mission")))


def render(chain_set: ChainSet, height: int, width: int, radius: int, *, dots_extra: int = 1):
    """The three bands of a set: its segments with ``radius`` and its dots (if any) one pixel larger."""
    bands = np.zeros((3, height, width), dtype=np.uint16)
    dots = chain_set.dot_segments
    if dots is not None and len(dots.length):
        draw_segments(bands, dots, radius + (dots_extra if radius > 0 else 0))      # a dot is never a chain's seed
    draw_segments(bands, chain_set.segments, radius)
    return bands


def write_raster(path: Path, bands, *, transform, crs_wkt: str, tags: dict) -> None:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window

    staging = path.with_name(path.name + ".partial")
    try:
        with rasterio.open(staging, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2], count=3,
                           dtype="uint16", transform=Affine.from_gdal(*transform), crs=crs_wkt or None,
                           tiled=True, blockxsize=128, blockysize=128, compress="deflate", predictor=2,
                           BIGTIFF="IF_SAFER") as dataset:
            for row in range(0, bands.shape[1], _STRIP_ROWS):
                strip = bands[:, row:row + _STRIP_ROWS, :]
                dataset.write(strip, window=Window(0, row, bands.shape[2], strip.shape[1]))
            dataset.update_tags(**tags)
            for number, name in enumerate(BANDS, start=1):
                dataset.set_band_description(number, name)
        os.replace(staging, path)
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
    [(name, sql_type)], "features": [(wkb, values...)], "bounds": (xmin, ymin, xmax, ymax)}``."""
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
                x0, y0, x1, y1 = layer["bounds"]
                connection.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y, max_x, max_y, srs_id) "
                                   "VALUES (?,?,?,?,?,?,?,?)", (name, "features", name, x0, y0, x1, y1, _SRS_ID))
                connection.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,0,0)",
                                   (name, "geom", layer["geometry"], _SRS_ID))
        connection.close()
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


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
) -> ChainMapsReport:
    """Write the six products and ``<prefix>_chain_maps.report.json`` into ``output_directory``."""
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
    directory.mkdir(parents=True, exist_ok=True)
    origin = arrays["origin"]
    tags = {
        "LUNARSCOUT_PRODUCT": "safe-haven-chain-map-v1",
        "BANDS": "1 chain_length_seeds (0 = none); 2 root_start_month (1 = %04d-%02d); 3 mission_hours" % origin,
        "SEGMENTS": "straight lines between seed pixel centres: a schematic, not the rover's path",
        "SOURCE_TABLE": Path(chains_path).name,
    }
    for set_name, chain in zip(CHAIN_SETS, (edge_set, chain_set)):
        description = {"predecessor_edges": "every seed's edge from its predecessor; value = the longest chain through the edge",
                       "longest_chain_per_root": "each root's longest chain; isolated seeds are dots of value 1"}[set_name]
        for kind, radius in (("raster", 0), ("dilated", int(dilate_pixels))):
            bands = render(chain, height, width, radius)
            write_raster(paths[set_name][kind], bands, transform=transform, crs_wkt=crs_wkt,
                         tags={**tags, "SET": set_name, "DESCRIPTION": description, "DILATE_PIXELS": str(radius)})
            del bands
    _write_vectors(paths, arrays, edge_set, chain_set, transform, crs_wkt)
    report = ChainMapsReport(
        seeds=len(rows), predecessor_edges=int(len(edge_set.segments.length)), chains=len(chain_set.lines),
        isolated_seeds=int(len(chain_set.dots)), files={n: {k: str(p) for k, p in g.items()} for n, g in paths.items()},
        seconds=clock.monotonic() - started)
    report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return report


def _write_vectors(paths, arrays, edge_set: ChainSet, chain_set: ChainSet, transform, crs_wkt) -> None:
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
        "bounds": _bounds(x[ends].ravel() if len(ends) else [], y[ends].ravel() if len(ends) else [])}])

    lines, line_attributes = chain_set.lines, chain_set.line_attributes
    chain_features = [
        (_blob(_wkb_line(x[line - 1], y[line - 1])), int(line_attributes["root_id"][k]), int(line_attributes["end_seed_id"][k]),
         int(line_attributes["chain_length"][k]), float(line_attributes["mission_hours"][k]),
         line_attributes["root_start_utc"][k], line_attributes["seed_ids"][k]) for k, line in enumerate(lines)]
    dots = chain_set.dots - 1
    rows = arrays["rows"]
    dot_features = [(_blob(_wkb_point(x[i], y[i])), int(i + 1), rows[i]["root_start_utc"], float(arrays["mission"][i]))
                    for i in dots]
    every = np.concatenate([x[line - 1] for line in lines]) if lines else np.zeros(0)
    every_y = np.concatenate([y[line - 1] for line in lines]) if lines else np.zeros(0)
    write_geopackage(paths["longest_chain_per_root"]["geopackage"], crs_wkt, [
        {"name": "chains", "geometry": "LINESTRING", "features": chain_features, "bounds": _bounds(every, every_y),
         "columns": [("root_id", "INTEGER"), ("end_seed_id", "INTEGER"), ("chain_length", "INTEGER"),
                     ("mission_hours", "REAL"), ("root_start_utc", "TEXT"), ("seed_ids", "TEXT")]},
        {"name": "isolated_seeds", "geometry": "POINT", "features": dot_features, "bounds": _bounds(x[dots], y[dots]),
         "columns": [("seed_id", "INTEGER"), ("root_start_utc", "TEXT"), ("mission_hours", "REAL")]}])
