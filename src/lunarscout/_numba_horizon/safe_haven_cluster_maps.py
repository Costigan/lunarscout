"""Bounding boxes of the groups of connected safe havens, one map per year (private).

A **connected haven** is a seed that is joined to at least one other seed by a chain; a seed that is alone is dropped.
The connected havens are clustered by connection: two havens are in one cluster when a chain of any length joins them.
Each cluster gets the bounding box of its havens' pixels, and the maps draw the boxes coloured by the number of connected
havens in the cluster.

What "joined by a chain" means is the ``connect`` setting:

* ``"chains"`` (default): the chains of stage 4. Every seed has one predecessor at most (the label forest), so a cluster
  is one root's whole tree: the root and every seed that descends from it. Its havens are exactly the seeds with the
  same ``root_id`` in a tree of two or more.
* ``"edges"``: every stage-3 edge, whether or not stage 4 kept it as a predecessor. A seed that several chains could
  reach joins their trees into one cluster, so the clusters are larger and there are fewer of them.

A cluster belongs to the UTC year of its root's operations start (``root_start_utc``), as in the yearly chain maps. With
``connect="edges"`` an edge whose two seeds have roots that start in different years is not used (the count is
reported), so a cluster never spans two years.

Per year the run writes an RGBA GeoTIFF on the DEM's grid (``<prefix>_cluster_boxes_<year>.tif``: the outline of each
box, ``outline_pixels`` wide just outside the box so that the havens stay visible, in the colour of its class, transparent
elsewhere), a PNG of the pole (``<prefix>_cluster_boxes_<year>.png``: the hillshade, PSRs and graticule of the other
overview maps with the boxes on top; a box smaller than a few overview pixels is drawn at that minimum size so it can be
seen) and, for all years together, a contact sheet. ``<prefix>_cluster_boxes.csv`` has one row per cluster (its year,
havens, pixel and CRS bounds, longest mission) and is the data behind the pictures.

Colour classes by havens in the cluster: 2, 3, 4, 5-6, 7-9, 10-19, 20-49 and 50 or more, one aqua hue, brighter for more.
"""

from __future__ import annotations

import csv
import json
import os
import time as clock
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .pixel_coordinates import dem_header
from .safe_haven_chains import read_chain_table
from .safe_haven_floods import read_edge_table
from .safe_haven_overview import (
    Geometry, Legend, backdrop_rgb, cell_grid_shape, check_overview_settings, read_backdrop, refuse_existing,
    render_contact_sheet, render_year_png, _rgb,
)

CONNECTIONS = ("chains", "edges")
# Class k (1-based) holds the clusters with at least COUNT_EDGES[k-2] havens (class 1: from 2) and fewer than COUNT_EDGES[k-1].
COUNT_EDGES = (3, 4, 5, 7, 10, 20, 50)
COUNT_LABELS = ("2", "3", "4", "5-6", "7-9", "10-19", "20-49", "50+")
_AQUA_STEPS = ("#127a56", "#168a62", "#199e70", "#3cb386", "#62c59c", "#8ad6b6", "#b1e5cf", "#d7f3e7")
_STRIP_ROWS = 1024
_MIN_BOX_CANVAS = 4.0              # the smallest box drawn in the PNG, in overview pixels
CLUSTER_COLUMNS = (
    "cluster_id", "year", "havens", "first_seed_id", "row_min", "row_max", "column_min", "column_max",
    "x_min_m", "x_max_m", "y_min_m", "y_max_m", "longest_mission_hours",
)


def cluster_colours() -> npt.NDArray[np.uint8]:
    """``(9, 3)`` RGB: row 0 is unused, rows 1 to 8 the classes from dimmer to brighter aqua."""
    return np.array([(0, 0, 0)] + [_rgb(c) for c in _AQUA_STEPS], dtype=np.uint8)


def cluster_legend() -> Legend:
    return Legend(cluster_colours(), COUNT_LABELS, "connected havens in the cluster (the box encloses them)")


def cluster_classes(havens: npt.NDArray) -> npt.NDArray[np.uint8]:
    """The colour class 1 to 8 of clusters of ``havens`` (2 or more) connected havens."""
    return (np.searchsorted(np.array(COUNT_EDGES), havens, side="right") + 1).astype(np.uint8)


@dataclass
class Clusters:
    """The clusters of connected havens, one entry per cluster, ordered by year, then by the lowest seed id."""

    year: npt.NDArray[np.int64]
    havens: npt.NDArray[np.int64]
    first_seed: npt.NDArray[np.int64]           # 1-based id of the lowest seed id of the cluster
    row_min: npt.NDArray[np.int64]
    row_max: npt.NDArray[np.int64]
    column_min: npt.NDArray[np.int64]
    column_max: npt.NDArray[np.int64]
    longest_mission: npt.NDArray[np.float64]
    seed_years: npt.NDArray[np.int64]           # the year of every seed's root
    edges_not_used: int = 0                     # connect="edges": edges between seeds whose roots start in different years

    def __len__(self) -> int:
        return len(self.year)


def _components(count: int, sources: npt.NDArray[np.int64], targets: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """The lowest 0-based seed index of the connected component of every seed (union-find over the edges)."""
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in zip(sources.tolist(), targets.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(count)], dtype=np.int64)


def find_clusters(rows: list[dict], *, connect: str = "chains", edges=None) -> Clusters:
    """The clusters of connected havens of the stage-4 table ``rows`` (seeds 1..N in order). ``edges`` are the stage-3
    ``(source, target, hours)`` tuples and are needed for ``connect="edges"``."""
    if connect not in CONNECTIONS:
        raise InputError(f"connect must be one of {CONNECTIONS}", code="cluster_setting_invalid", details={"connect": connect})
    count = len(rows)
    if count == 0 or [r["seed_id"] for r in rows] != list(range(1, count + 1)):
        raise InputError("the chain table's seeds are not 1..N in order", code="cluster_table_invalid")
    root = np.array([r["root_id"] for r in rows], dtype=np.int64)
    seed_years = np.array([int(r["root_start_utc"][:4]) for r in rows], dtype=np.int64)
    pixel_row = np.array([r["row"] for r in rows], dtype=np.int64)
    pixel_column = np.array([r["column"] for r in rows], dtype=np.int64)
    mission = np.array([r["mission_hours"] for r in rows], dtype=np.float64)
    not_used = 0
    if connect == "chains":
        label = root - 1
    else:
        if edges is None:
            raise InputError("connect='edges' needs the stage-3 edge table", code="cluster_setting_invalid")
        sources = np.array([e[0] for e in edges], dtype=np.int64)
        targets = np.array([e[1] for e in edges], dtype=np.int64)
        if len(sources) and (min(sources.min(), targets.min()) < 1 or max(sources.max(), targets.max()) > count):
            raise InputError("an edge names a seed that is not in the chain table", code="cluster_table_invalid")
        same_year = seed_years[sources - 1] == seed_years[targets - 1]
        not_used = int((~same_year).sum())
        label = _components(count, sources[same_year] - 1, targets[same_year] - 1)
    order = np.argsort(label, kind="stable")            # members of a cluster together, lowest seed id first
    sorted_label = label[order]
    starts = np.flatnonzero(np.r_[True, sorted_label[1:] != sorted_label[:-1]])
    ends = np.r_[starts[1:], count]
    keep = (ends - starts) >= 2
    starts, ends = starts[keep], ends[keep]
    firsts = order[starts]
    ranking = np.lexsort((firsts, seed_years[firsts]))
    starts, ends, firsts = starts[ranking], ends[ranking], firsts[ranking]
    fields = {name: np.zeros(len(starts), dtype=dtype) for name, dtype in (
        ("row_min", np.int64), ("row_max", np.int64), ("column_min", np.int64), ("column_max", np.int64),
        ("longest_mission", np.float64))}
    for k, (s, e) in enumerate(zip(starts, ends)):
        members = order[s:e]
        fields["row_min"][k], fields["row_max"][k] = pixel_row[members].min(), pixel_row[members].max()
        fields["column_min"][k], fields["column_max"][k] = pixel_column[members].min(), pixel_column[members].max()
        fields["longest_mission"][k] = mission[members].max()
    return Clusters(year=seed_years[firsts], havens=ends - starts, first_seed=firsts + 1, seed_years=seed_years,
                    edges_not_used=not_used, **fields)


def box_bounds_m(clusters: Clusters, transform) -> dict:
    """The CRS bounds of each cluster's box: the outer edges of its pixels."""
    x0 = transform[0] + transform[1] * clusters.column_min
    x1 = transform[0] + transform[1] * (clusters.column_max + 1)
    y0 = transform[3] + transform[5] * (clusters.row_max + 1)           # rows run south
    y1 = transform[3] + transform[5] * clusters.row_min
    return {"x_min_m": np.minimum(x0, x1), "x_max_m": np.maximum(x0, x1),
            "y_min_m": np.minimum(y0, y1), "y_max_m": np.maximum(y0, y1)}


def write_cluster_table(path: Path, clusters: Clusters, transform) -> None:
    bounds = box_bounds_m(clusters, transform)
    staging = path.with_name(path.name + ".partial")
    try:
        with staging.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(CLUSTER_COLUMNS)
            for k in range(len(clusters)):
                writer.writerow([k + 1, int(clusters.year[k]), int(clusters.havens[k]), int(clusters.first_seed[k]),
                                 int(clusters.row_min[k]), int(clusters.row_max[k]), int(clusters.column_min[k]),
                                 int(clusters.column_max[k]), *(f"{bounds[n][k]:.3f}" for n in
                                                                ("x_min_m", "x_max_m", "y_min_m", "y_max_m")),
                                 f"{clusters.longest_mission[k]:g}"])
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The raster
# ---------------------------------------------------------------------------


def draw_outlines(strip, first_row: int, boxes, outline: int) -> None:
    """Draw into ``strip`` (rows ``first_row`` on of the DEM grid) the outline of each box, ``outline`` pixels wide just
    outside it, in the order given (the later over the earlier). ``boxes`` are ``(row_min, row_max, column_min,
    column_max, class)``."""
    height, width = strip.shape

    def clip(value: int) -> int:
        return min(max(value, 0), height)

    for row_min, row_max, column_min, column_max, klass in boxes:
        top, bottom = row_min - outline - first_row, row_max + outline + 1 - first_row          # the outer rectangle
        left, right = max(column_min - outline, 0), min(column_max + outline + 1, width)
        inner_top, inner_bottom = row_min - first_row, row_max + 1 - first_row
        inner_left, inner_right = max(column_min, 0), min(column_max + 1, width)
        strip[clip(top):clip(inner_top), left:right] = klass                                    # the band above
        strip[clip(inner_bottom):clip(bottom), left:right] = klass                              # below
        sides = slice(clip(inner_top), clip(inner_bottom))
        strip[sides, left:inner_left] = klass                                                   # left
        strip[sides, inner_right:right] = klass                                                 # right


def write_boxes_image(path: Path, boxes, *, height: int, width: int, outline: int, transform, crs_wkt: str,
                      tags: dict) -> None:
    """The outlines of ``boxes`` as a tiled RGBA GeoTIFF on the DEM's grid, written in strips (a strip no box touches is
    left unwritten: transparent). ``boxes`` are ``(row_min, row_max, column_min, column_max, class)`` in drawing order."""
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import Affine
    from rasterio.windows import Window

    table = np.zeros((9, 4), dtype=np.uint8)
    table[1:, :3] = cluster_colours()[1:]
    table[1:, 3] = 255
    array = np.array(boxes, dtype=np.int64).reshape(-1, 5)
    staging = path.with_name(path.name + ".partial")
    try:
        with rasterio.open(staging, "w", driver="GTiff", height=height, width=width, count=4, dtype="uint8",
                           photometric="RGB", transform=Affine.from_gdal(*transform), crs=crs_wkt or None, tiled=True,
                           blockxsize=128, blockysize=128, compress="deflate", predictor=2, BIGTIFF="IF_SAFER") as dataset:
            dataset.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
            for first in range(0, height, _STRIP_ROWS):
                rows_here = min(_STRIP_ROWS, height - first)
                touching = array[(array[:, 0] - outline < first + rows_here) & (array[:, 1] + outline + 1 > first)]
                if not len(touching):
                    continue
                strip = np.zeros((rows_here, width), dtype=np.uint8)
                draw_outlines(strip, first, touching.tolist(), outline)
                dataset.write(np.moveaxis(table[strip], -1, 0), window=Window(0, first, width, rows_here))
            dataset.update_tags(**tags)
        os.replace(staging, path)
        path.with_name(path.name + ".aux.xml").unlink(missing_ok=True)       # a sidecar of the file just replaced
    finally:
        staging.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The pictures
# ---------------------------------------------------------------------------


def box_patches(clusters: Clusters, selection, *, factor: float, line_width: float):
    """A matplotlib collection of the selected boxes in overview pixels, fewer havens first (more on top). A box smaller
    than ``_MIN_BOX_CANVAS`` is drawn at that size around its centre."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    colours = cluster_colours() / 255.0
    order = selection[np.argsort(clusters.havens[selection], kind="stable")]
    patches, edge = [], []
    for k in order:
        x0, x1 = clusters.column_min[k] / factor, (clusters.column_max[k] + 1) / factor
        y0, y1 = clusters.row_min[k] / factor, (clusters.row_max[k] + 1) / factor
        width, height = max(x1 - x0, _MIN_BOX_CANVAS), max(y1 - y0, _MIN_BOX_CANVAS)
        patches.append(Rectangle(((x0 + x1) / 2 - width / 2, (y0 + y1) / 2 - height / 2), width, height))
        edge.append(tuple(colours[cluster_classes(clusters.havens[k:k + 1])[0]]))
    return PatchCollection(patches, facecolor="none", edgecolor=edge, linewidth=line_width)


def _stats_text(clusters: Clusters, selection) -> str:
    if not len(selection):
        return "no connected havens"
    havens = clusters.havens[selection]
    return (f"{len(selection):,} clusters of {int(havens.sum()):,} connected havens; the largest has {int(havens.max()):,}; "
            f"the median has {int(np.median(havens))}")


def cluster_paths(directory: str | Path, years, prefix: str = "safe_haven_stage_4") -> dict:
    directory = Path(directory)
    return {"years": {y: {"geotiff": directory / f"{prefix}_cluster_boxes_{y}.tif", "png": directory / f"{prefix}_cluster_boxes_{y}.png"}
                      for y in years},
            "contact_sheet": directory / f"{prefix}_cluster_boxes_contact_sheet.png",
            "table": directory / f"{prefix}_cluster_boxes.csv",
            "report": directory / f"{prefix}_cluster_boxes.report.json"}


@dataclass
class ClusterReport:
    connect: str
    seeds: int
    clusters: int                    # of all years
    connected_havens: int
    edges_not_used: int              # connect="edges": edges between seeds whose roots start in different years
    years: dict                      # per year: the files, the clusters, the connected havens, the clusters in each class
    classes: list
    outline_pixels: int
    table: str
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def run_cluster_maps(
    output_directory: str | Path, *, chains_path: str | Path, dem_path: str | Path, edges_path: str | Path | None = None,
    psr_ids_path: str | Path | None = None, connect: str = "chains", years=None, cell_patches: int = 2,
    overview_pixels_per_cell: int = 16, outline_pixels: int = 6, prefix: str = "safe_haven_stage_4",
    overwrite: bool = False,
) -> ClusterReport:
    """Cluster the connected havens of the stage-4 table, and write the box GeoTIFF and PNG of each of ``years`` (None:
    every year in which a chain starts), the contact sheet, ``<prefix>_cluster_boxes.csv`` and
    ``<prefix>_cluster_boxes.report.json`` into ``output_directory``. ``edges_path`` (the stage-3 edge table) is needed
    for ``connect="edges"``. ``cell_patches`` must be stage 2's and sets only the pole pictures' backdrop grid;
    ``outline_pixels`` is the width of the raster outlines in DEM pixels."""
    started = clock.monotonic()
    if connect not in CONNECTIONS or outline_pixels < 1:
        raise InputError("connect must be 'chains' or 'edges' and outline_pixels at least 1", code="cluster_setting_invalid",
                         details={"connect": connect, "outline_pixels": outline_pixels})
    cell_pixels = check_overview_settings(cell_patches, overview_pixels_per_cell)
    rows = read_chain_table(chains_path)
    edges = read_edge_table(edges_path) if connect == "edges" and edges_path is not None else None
    clusters = find_clusters(rows, connect=connect, edges=edges)
    in_table = sorted({int(y) for y in np.unique(clusters.seed_years)})
    years = list(range(in_table[0], in_table[-1] + 1)) if years is None else [int(y) for y in years]
    if not years or len(set(years)) != len(years):
        raise InputError("years must be a non-empty list of distinct years", code="cluster_setting_invalid",
                         details={"years": years})
    transform, width, height, crs_wkt = dem_header(dem_path)
    if (max(r["row"] for r in rows) >= height) or (max(r["column"] for r in rows) >= width):
        raise InputError("a seed lies outside the DEM", code="cluster_table_invalid")
    cell_shape = cell_grid_shape(height, width, cell_pixels)
    directory = Path(output_directory)
    paths = cluster_paths(directory, years, prefix)
    refuse_existing([p for y in paths["years"].values() for p in y.values()]
                    + [paths["contact_sheet"], paths["table"], paths["report"]], overwrite)
    geometry = Geometry(transform, overview_pixels_per_cell, cell_pixels, cell_shape, crs_wkt)
    elevation, psr = read_backdrop(dem_path, psr_ids_path, cell_pixels=cell_pixels, scale=overview_pixels_per_cell,
                                   cell_shape=cell_shape)
    backdrop = np.rint(backdrop_rgb(elevation, psr, geometry.pixel_metres) * 255.0).astype(np.uint8)
    del elevation, psr
    directory.mkdir(parents=True, exist_ok=True)
    write_cluster_table(paths["table"], clusters, transform)
    factor = cell_pixels / overview_pixels_per_cell
    legend = cluster_legend()
    selections, result = {}, {}
    for year in years:
        selection = np.flatnonzero(clusters.year == year)
        selections[year] = selection
        by_havens = selection[np.argsort(clusters.havens[selection], kind="stable")]      # more havens drawn later: on top
        boxes = [(int(clusters.row_min[k]), int(clusters.row_max[k]), int(clusters.column_min[k]),
                  int(clusters.column_max[k]), int(cluster_classes(clusters.havens[k:k + 1])[0])) for k in by_havens]
        write_boxes_image(
            paths["years"][year]["geotiff"], boxes, height=height, width=width, outline=outline_pixels,
            transform=transform, crs_wkt=crs_wkt,
            tags={"LUNARSCOUT_PRODUCT": "safe-haven-cluster-boxes-v1", "YEAR": str(year), "CONNECT": connect,
                  "DESCRIPTION": f"bounding boxes of the clusters of connected havens whose chains start in {year}; "
                                 "the colour is the number of havens in the cluster",
                  "SOURCE_TABLE": Path(chains_path).name, "OUTLINE_PIXELS": str(outline_pixels)})
        render_year_png(
            paths["years"][year]["png"], backdrop, geometry, title=f"Clusters of connected havens, {year}",
            subtitle=f"Bounding box of each group of havens joined by {'chains' if connect == 'chains' else 'stage-3 edges'};"
                     " colour: havens in the group",
            stats=_stats_text(clusters, selection), legend=legend,
            footnote="boxes under 4 overview pixels are drawn at 4; dark terrain: hillshade; dark outlined areas: PSRs",
            decorate=lambda axes, s=selection: axes.add_collection(box_patches(clusters, s, factor=factor, line_width=0.8)))
        classes = cluster_classes(clusters.havens[selection]) if len(selection) else np.zeros(0, dtype=np.uint8)
        result[year] = {"geotiff": str(paths["years"][year]["geotiff"]), "png": str(paths["years"][year]["png"]),
                        "clusters": int(len(selection)), "connected_havens": int(clusters.havens[selection].sum()),
                        "largest_cluster": int(clusters.havens[selection].max()) if len(selection) else 0,
                        "clusters_by_class": [int(v) for v in np.bincount(classes, minlength=len(COUNT_LABELS) + 1)]}
    images = {year: backdrop for year in years}
    render_contact_sheet(
        paths["contact_sheet"], images, {year: f"{len(selections[year]):,} clusters" for year in years}, geometry,
        title="Clusters of connected havens by year",
        subtitle="Bounding box of each group of connected havens, coloured by the havens in it; the same scale in every panel",
        legend=legend,
        decorate=lambda year, axes: axes.add_collection(box_patches(clusters, selections[year], factor=factor, line_width=0.5)))
    report = ClusterReport(
        connect=connect, seeds=len(rows), clusters=len(clusters), connected_havens=int(clusters.havens.sum()),
        edges_not_used=clusters.edges_not_used, years=result, classes=list(COUNT_LABELS), outline_pixels=outline_pixels,
        table=str(paths["table"]), seconds=clock.monotonic() - started)
    paths["report"].write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return report
