"""Overview maps of where the safe havens are around the pole, one per year (private).

The stage-2 seed table names, for every seed, its 5.12 km cell and its month. The **persistence** of a cell in a year is
the number of months of that year (0 to 12) in which the cell has at least one valid seed. It comes straight from the
table (no lighting, flooding or routes) and, unlike a count of seeds, it does not depend on stage 2's cap of five seeds
per cell and month.

Per year the run writes a one-band uint8 GeoTIFF on the cell grid (``<prefix>_persistence_<year>.tif``: the data, in
the DEM's CRS, for QGIS), a PNG of the pole (``<prefix>_persistence_<year>.png``: a dark hillshade of the DEM, the PSRs,
a polar graticule and one square per cell coloured by its persistence) and, for all years together, one contact sheet
(``<prefix>_persistence_contact_sheet.png``) with the same colour scale in every panel so the years compare by eye. The
colour is one blue hue, brighter for more months; a cell with no seed is left as the hillshade.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time as clock
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .pixel_coordinates import dem_header

MONTHS = 12

# Dark chart surface and ink (the validated reference instance of the dataviz skill).
SURFACE = "#1a1a19"
PRIMARY_INK = "#f0efec"
SECONDARY_INK = "#c3c2b7"
MUTED_INK = "#898781"
PAGE = "#0d0d0d"

# One hue, light for more months. Steps 450 to 100 of the reference blue ramp (step 450 is clearly visible on the dark
# hillshade), interpolated to the 12 months.
_BLUE_STEPS = ("#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb")


def _rgb(colour: str) -> tuple[int, int, int]:
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def persistence_ramp() -> npt.NDArray[np.uint8]:
    """``(13, 3)`` RGB: row 0 (no seed) is unused, rows 1 to 12 are the months from dimmer to brighter blue."""
    steps = np.array([_rgb(c) for c in _BLUE_STEPS], dtype=float)
    positions = np.linspace(0, len(steps) - 1, MONTHS)
    ramp = np.zeros((MONTHS + 1, 3), dtype=np.uint8)
    for channel in range(3):
        ramp[1:, channel] = np.rint(np.interp(positions, np.arange(len(steps)), steps[:, channel])).astype(np.uint8)
    return ramp


@dataclass(frozen=True)
class Legend:
    """The colour key of a map: ``colours`` is ``(classes + 1, 3)`` RGB (row 0, the class of no value, is unused), one
    label per class 1 to ``classes`` and a caption under the swatches."""

    colours: npt.NDArray[np.uint8]
    labels: tuple[str, ...]
    caption: str


def persistence_legend() -> Legend:
    return Legend(persistence_ramp(), tuple(str(month) for month in range(1, MONTHS + 1)),
                  "months with a seed (cells with none show the terrain)")


@dataclass
class OverviewReport:
    years: dict                      # per year: the files, the seeds, the cells with seeds and how many months they have
    cell_shape: list                 # cell rows and columns of the grid
    cell_pixels: int
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Persistence on the cell grid
# ---------------------------------------------------------------------------


def cell_grid_shape(height: int, width: int, cell_pixels: int) -> tuple[int, int]:
    return -(-height // cell_pixels), -(-width // cell_pixels)


def read_cell_table(path: str | Path, cell_shape: tuple[int, int], value_column: str | None = None):
    """``(month_keys, cell_rows, cell_cols, values)`` of every row of a stage-2 or stage-4 table: ``year * 12 + month - 1``,
    the 5.12 km cell and, when ``value_column`` is named, that column as floats (else None). A table that is not one, is
    empty, has an unreadable month or cell, or has a cell outside the grid is refused."""
    rows_n, cols_n = cell_shape
    month_keys, cell_rows, cell_cols, values = [], [], [], []
    wanted = {"month", "cell_row", "cell_col"} | ({value_column} if value_column else set())
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = wanted - set(reader.fieldnames or ())
        if missing:
            raise InputError(f"{path} is not a seed table (no {', '.join(sorted(missing))} column)",
                             code="overview_table_unreadable", details={"missing": sorted(missing)})
        for record in reader:
            try:
                year, month = (int(part) for part in record["month"].split("-"))
                if not 1 <= month <= MONTHS:
                    raise ValueError(record["month"])
                month_keys.append(year * MONTHS + month - 1)
                cell_rows.append(int(record["cell_row"]))
                cell_cols.append(int(record["cell_col"]))
                if value_column:
                    values.append(float(record[value_column]))
            except ValueError as error:
                raise InputError(f"{path} has a seed with an unreadable month, cell or {value_column or 'value'} "
                                 f"({record['month']!r})", code="overview_table_unreadable",
                                 details={"month": record["month"]}) from error
    if not month_keys:
        raise InputError(f"{path} has no seeds", code="overview_table_empty", details={"path": str(path)})
    cell_rows, cell_cols = np.array(cell_rows, dtype=np.int64), np.array(cell_cols, dtype=np.int64)
    if cell_rows.min() < 0 or cell_cols.min() < 0 or cell_rows.max() >= rows_n or cell_cols.max() >= cols_n:
        raise InputError("a seed lies outside the cell grid; the cell size (cell patches) must be stage 2's",
                         code="overview_cell_grid_mismatch",
                         details={"cell_shape": [rows_n, cols_n], "largest_cell": [int(cell_rows.max()), int(cell_cols.max())]})
    return (np.array(month_keys, dtype=np.int64), cell_rows, cell_cols,
            np.array(values, dtype=np.float64) if value_column else None)


def resolve_years(month_keys, years) -> list[int]:
    """``years`` as a list of distinct ints, or (None) every year from the first to the last in the table."""
    in_table = sorted({int(key) // MONTHS for key in np.unique(month_keys)})
    years = list(range(in_table[0], in_table[-1] + 1)) if years is None else [int(y) for y in years]
    if not years or len(set(years)) != len(years):
        raise InputError("years must be a non-empty list of distinct years", code="overview_setting_invalid",
                         details={"years": years})
    return years


def persistence_grids(seeds_path: str | Path, *, cell_shape: tuple[int, int], years=None):
    """``({year: uint8 grid of cell rows x columns}, {year: seeds})`` from the stage-2 seed table: each cell's number of
    distinct months of the year with a seed. ``years`` None takes every year from the first to the last in the table."""
    rows_n, cols_n = cell_shape
    month_keys, cell_rows, cell_cols, _ = read_cell_table(seeds_path, cell_shape)
    years = resolve_years(month_keys, years)
    grids, seeds = {}, {}
    cells = rows_n * cols_n
    for year in years:
        in_year = (month_keys // MONTHS) == year
        seeds[year] = int(in_year.sum())
        grid = np.zeros((rows_n, cols_n), dtype=np.uint8)
        if in_year.any():
            key = ((month_keys[in_year] % MONTHS) * rows_n + cell_rows[in_year]) * cols_n + cell_cols[in_year]
            unique = np.unique(key)                              # each month of a cell once
            np.add.at(grid.reshape(-1), unique % cells, 1)
        grids[year] = grid
    return grids, seeds


def write_persistence_geotiff(path: Path, grid, *, transform, crs_wkt: str, tags: dict) -> None:
    """The persistence grid as a one-band uint8 GeoTIFF on the cell grid (value 0: no seed), through a staging file."""
    import rasterio
    from rasterio.transform import Affine

    staging = path.with_name(path.name + ".partial")
    try:
        with rasterio.open(staging, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1], count=1,
                           dtype="uint8", transform=Affine.from_gdal(*transform), crs=crs_wkt or None, compress="deflate",
                           tiled=False) as dataset:
            dataset.write(grid[None])
            dataset.set_band_description(1, "months_with_a_seed")
            dataset.update_tags(**tags)
        os.replace(staging, path)
        path.with_name(path.name + ".aux.xml").unlink(missing_ok=True)       # a sidecar of the file just replaced
    finally:
        staging.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The backdrop: a hillshade and the PSRs at overview scale
# ---------------------------------------------------------------------------


def block_reduce(strip: npt.NDArray, factor: int, rows: int, columns: int) -> npt.NDArray[np.float32]:
    """The mean of each ``factor`` x ``factor`` block of ``strip`` (a few rows of a raster), ignoring nan; the strip is
    padded with nan on the bottom and right to ``rows`` x ``columns`` blocks, and a block with no value is nan. On a 0/1
    raster the mean is the share of ones."""
    padded = np.full((rows * factor, columns * factor), np.nan, dtype=np.float32)
    padded[:strip.shape[0], :strip.shape[1]] = strip
    blocks = padded.reshape(rows, factor, columns, factor)
    valid = ~np.isnan(blocks)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def read_backdrop(dem_path, psr_ids_path, *, cell_pixels: int, scale: int, cell_shape):
    """``(elevation, psr_fraction)`` on the overview canvas of ``scale`` pixels per cell: the mean elevation of each block
    of ``cell_pixels // scale`` DEM pixels (nan where there is no DEM) and the share of the block that is PSR (nan where
    there is no PSR raster)."""
    import rasterio

    from ..geotiff import _resolve_dem_path

    factor = cell_pixels // scale
    rows_n, cols_n = cell_shape
    elevation = np.full((rows_n * scale, cols_n * scale), np.nan, dtype=np.float32)
    psr = np.full_like(elevation, np.nan) if psr_ids_path is not None else None
    psr_dataset = rasterio.open(psr_ids_path) if psr_ids_path is not None else None
    try:
        with rasterio.open(_resolve_dem_path(Path(dem_path).expanduser().resolve())) as dem:
            if psr_dataset is not None and (psr_dataset.height, psr_dataset.width) != (dem.height, dem.width):
                raise InputError("the PSR id raster and the DEM differ in size", code="overview_psr_mismatch",
                                 details={"psr": [psr_dataset.height, psr_dataset.width], "dem": [dem.height, dem.width]})
            nodata = dem.nodata
            for cell_row in range(rows_n):
                first, last = cell_row * cell_pixels, min((cell_row + 1) * cell_pixels, dem.height)
                if first >= last:
                    continue
                window = rasterio.windows.Window(0, first, dem.width, last - first)
                strip = dem.read(1, window=window).astype(np.float32)
                strip[~np.isfinite(strip)] = np.nan
                if nodata is not None:
                    strip[strip == np.float32(nodata)] = np.nan
                target = slice(cell_row * scale, (cell_row + 1) * scale)
                elevation[target] = block_reduce(strip, factor, scale, cols_n * scale)
                if psr_dataset is not None:
                    flags = (psr_dataset.read(1, window=window) > 0).astype(np.float32)
                    psr[target] = block_reduce(flags, factor, scale, cols_n * scale)
    finally:
        if psr_dataset is not None:
            psr_dataset.close()
    return elevation, psr


def raw_shade(elevation, pixel_metres: float, *, azimuth: float = 315.0, altitude: float = 35.0,
              z_factor: float = 2.0) -> npt.NDArray[np.float32]:
    """The sun's illumination of the terrain in ``[0, 1]`` (nan elevations are filled with the mean for the slopes)."""
    valid = np.isfinite(elevation)
    z = np.where(valid, elevation, np.nanmean(elevation) if valid.any() else 0.0) * z_factor
    d_row, d_col = np.gradient(z, pixel_metres)                   # rows run south, columns east
    d_east, d_north = d_col, -d_row
    norm = np.sqrt(d_east ** 2 + d_north ** 2 + 1.0)
    az, alt = math.radians(azimuth), math.radians(altitude)
    shade = (-d_east * math.sin(az) * math.cos(alt) - d_north * math.cos(az) * math.cos(alt) + math.sin(alt)) / norm
    return np.clip(shade, 0.0, 1.0)


def hillshade_gray(elevation, pixel_metres: float, *, azimuth: float = 315.0, altitude: float = 35.0,
                   z_factor: float = 2.0, low: float = 0.07, high: float = 0.24) -> npt.NDArray[np.float32]:
    """A dark grey hillshade in ``[low, high]`` (the surface grey where ``elevation`` is nan is applied by the caller)."""
    valid = np.isfinite(elevation)
    shade = raw_shade(elevation, pixel_metres, azimuth=azimuth, altitude=altitude, z_factor=z_factor)
    if valid.any():
        lo, hi = np.percentile(shade[valid], (2, 98))
        shade = np.clip((shade - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.full_like(shade, 0.5)
    return (low + (high - low) * shade).astype(np.float32)


def backdrop_rgb(elevation, psr, pixel_metres: float, *, psr_fraction: float = 0.25) -> npt.NDArray[np.float32]:
    """The canvas without cells: surface colour where there is no DEM, the hillshade elsewhere, the PSRs (blocks at least
    ``psr_fraction`` PSR) darker with a muted outline."""
    surface = np.array(_rgb(SURFACE), dtype=np.float32) / 255.0
    image = np.broadcast_to(surface, (*elevation.shape, 3)).copy()
    valid = np.isfinite(elevation)
    gray = hillshade_gray(elevation, pixel_metres)
    image[valid] = gray[valid][:, None]
    if psr is not None:
        mask = np.nan_to_num(psr, nan=0.0) >= psr_fraction
        inner = mask.copy()
        inner[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        inner[0, :] = inner[-1, :] = False
        inner[:, 0] = inner[:, -1] = False
        image[mask] = np.array(_rgb(PAGE), dtype=np.float32) / 255.0
        image[mask & ~inner] = np.array(_rgb(MUTED_INK), dtype=np.float32) / 255.0 * 0.55
    return image


def compose(backdrop, grid, scale: int, *, inset: int = 1, colours=None) -> npt.NDArray[np.uint8]:
    """The backdrop with one square per cell whose class in ``grid`` is above 0, coloured by ``colours`` (rows indexed by
    class; default: the persistence ramp), ``inset`` pixels in from the cell's edge so the hillshade shows between the
    squares."""
    ramp = persistence_ramp() if colours is None else colours
    values = np.kron(grid, np.ones((scale, scale), dtype=np.uint8))
    local = np.arange(values.shape[0]) % scale
    local_cols = np.arange(values.shape[1]) % scale
    inside_rows = (local >= inset) & (local < scale - inset)
    inside_cols = (local_cols >= inset) & (local_cols < scale - inset)
    draw = (values > 0) & inside_rows[:, None] & inside_cols[None, :]
    image = np.rint(backdrop * 255.0).astype(np.uint8)
    image[draw] = ramp[values[draw]]
    return image


# ---------------------------------------------------------------------------
# The pictures
# ---------------------------------------------------------------------------


def stereographic_radius(latitude_deg: float, radius_m: float) -> float:
    """Distance from the pole of a parallel in the south polar stereographic projection of a sphere (scale 1 at the pole)."""
    return 2.0 * radius_m * math.tan(math.radians(90.0 - abs(latitude_deg)) / 2.0)


def _sphere_radius(crs_wkt: str) -> float:
    try:
        from pyproj import CRS

        return float(CRS.from_wkt(crs_wkt).ellipsoid.semi_major_metre)
    except Exception:                                                          # no CRS: the lunar reference sphere
        return 1737400.0


class Geometry:
    """Where the pole and the graticule fall on the canvas."""

    def __init__(self, transform, scale: int, cell_pixels: int, cell_shape, crs_wkt: str):
        if transform[2] != 0 or transform[4] != 0:
            raise InputError("the DEM is rotated; the overview maps need a north-up grid", code="overview_grid_rotated")
        self.pixel_metres = cell_pixels * abs(transform[1]) / scale
        self.pole = ((0.0 - transform[0]) / self.pixel_metres, (transform[3] - 0.0) / self.pixel_metres)
        self.height, self.width = cell_shape[0] * scale, cell_shape[1] * scale
        self.radius_m = _sphere_radius(crs_wkt)

    def point(self, latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
        """Canvas ``(column, row)`` of a point; longitude is east-positive and 0 is toward the top (the projection's +y)."""
        rho = stereographic_radius(latitude_deg, self.radius_m) / self.pixel_metres
        lon = math.radians(longitude_deg)
        return self.pole[0] + rho * math.sin(lon), self.pole[1] - rho * math.cos(lon)


PARALLELS = (-88, -86, -84, -82, -80)
SPOKES = (0, 45, 90, 135, 180, -135, -90, -45)


def _longitude_label(longitude: int) -> str:
    return "0°" if longitude == 0 else "180°" if abs(longitude) == 180 else f"{abs(longitude)}°{'E' if longitude > 0 else 'W'}"


def draw_graticule(axes, geometry: Geometry, *, labels: bool, line_width: float) -> None:
    """Thin muted parallels every 2 degrees and meridians every 45, labelled when ``labels``."""
    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=2.5, foreground=SURFACE)]          # keeps the labels readable on the cells
    lons = np.linspace(-180, 180, 361)
    for latitude in PARALLELS:
        points = np.array([geometry.point(latitude, lon) for lon in lons])
        axes.plot(points[:, 0], points[:, 1], color=MUTED_INK, linewidth=line_width, alpha=0.55, solid_capstyle="round")
        if labels:
            x, y = geometry.point(latitude, 112.5)
            axes.text(x, y, f"{abs(latitude)}°S", color=SECONDARY_INK, fontsize=7, ha="left", va="center",
                      path_effects=halo)
    outer = PARALLELS[-1]
    for longitude in SPOKES:
        x0, y0 = geometry.point(-89.999, longitude)
        x1, y1 = geometry.point(outer, longitude)
        axes.plot([x0, x1], [y0, y1], color=MUTED_INK, linewidth=line_width, alpha=0.35)
        if labels:
            x2, y2 = geometry.point(outer, longitude)
            dx, dy = x2 - geometry.pole[0], y2 - geometry.pole[1]
            norm = math.hypot(dx, dy)
            axes.text(x2 + dx / norm * 14, y2 + dy / norm * 14, _longitude_label(longitude), color=SECONDARY_INK,
                      fontsize=8, ha="center", va="center", path_effects=halo)


def draw_legend(figure, rectangle, legend: Legend | None = None, *, font: float = 9.0) -> None:
    """The colour scale: one swatch per class, with its label under it."""
    legend = persistence_legend() if legend is None else legend
    classes = len(legend.labels)
    axes = figure.add_axes(rectangle)
    axes.set_xlim(0, classes)
    axes.set_ylim(0, 1)
    axes.axis("off")
    for index, label in enumerate(legend.labels):
        colour = tuple(legend.colours[index + 1] / 255.0)
        axes.add_patch(_rectangle(index + 0.06, 0.40, 0.88, 0.55, colour))
        axes.text(index + 0.5, 0.12, label, color=SECONDARY_INK, fontsize=font, ha="center", va="center")


def _rectangle(x, y, w, h, colour):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, facecolor=colour, edgecolor="none")


def _stats_text(grid, seeds: int) -> str:
    with_seeds = int((grid > 0).sum())
    return (f"{with_seeds:,} of {grid.size:,} cells have a seed; {int((grid == MONTHS).sum()):,} in all 12 months; "
            f"{seeds:,} seeds")


def render_year_png(path: Path, image, geometry: Geometry, *, title: str, subtitle: str, stats: str, legend: Legend,
                    footnote: str = "dark terrain: hillshade of the DEM; dark outlined areas: PSRs; south polar stereographic",
                    decorate: Callable | None = None) -> None:
    """One year's PNG: the pole with its graticule, the legend and the numbers. ``decorate(axes)`` draws anything more on
    top of the image, in canvas pixels, before the graticule."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.4, 9.3), dpi=240, facecolor=SURFACE)
    axes = figure.add_axes([0.05, 0.17, 0.90, 0.73])
    axes.imshow(image, interpolation="nearest", extent=(0, geometry.width, geometry.height, 0))
    if decorate is not None:
        decorate(axes)
    draw_graticule(axes, geometry, labels=True, line_width=0.5)
    pad = 26
    axes.set_xlim(-pad, geometry.width + pad)
    axes.set_ylim(geometry.height + pad, -pad)
    axes.set_aspect("equal")
    axes.axis("off")
    figure.text(0.05, 0.955, title, color=PRIMARY_INK, fontsize=15, fontweight="bold", ha="left")
    figure.text(0.05, 0.928, subtitle, color=SECONDARY_INK, fontsize=9.5, ha="left")
    figure.text(0.05, 0.905, stats, color=SECONDARY_INK, fontsize=9, ha="left")
    draw_legend(figure, [0.15, 0.065, 0.70, 0.06], legend)
    figure.text(0.5, 0.045, legend.caption, color=MUTED_INK, fontsize=8.5, ha="center")
    figure.text(0.95, 0.02, footnote, color=MUTED_INK, fontsize=7, ha="right")
    _save_png(figure, path)


def render_contact_sheet(path: Path, images: dict, notes: dict, geometry: Geometry, *, title: str, subtitle: str,
                         legend: Legend, columns: int = 4, decorate: Callable | None = None) -> None:
    """All years as small multiples with one shared colour scale; ``notes[year]`` is the small text at a panel's top right
    and ``decorate(year, axes)`` draws anything more on a panel."""
    from matplotlib.figure import Figure

    years = list(images)
    rows = -(-len(years) // columns)
    figure = Figure(figsize=(4.0 * columns, 4.35 * rows + 1.3), dpi=200, facecolor=SURFACE)
    top, bottom = 1.0 - 0.9 / figure.get_figheight() * 1.0, 1.0 / figure.get_figheight()
    panel_h = (top - bottom - 0.02) / rows
    for index, year in enumerate(years):
        row, column = divmod(index, columns)
        left = 0.01 + column * (0.98 / columns)
        axes = figure.add_axes([left, top - (row + 1) * panel_h - 0.005, 0.98 / columns - 0.005, panel_h - 0.045])
        axes.imshow(images[year], interpolation="antialiased", extent=(0, geometry.width, geometry.height, 0))
        if decorate is not None:
            decorate(year, axes)
        draw_graticule(axes, geometry, labels=False, line_width=0.3)
        axes.set_xlim(0, geometry.width)
        axes.set_ylim(geometry.height, 0)
        axes.set_aspect("equal")
        axes.axis("off")
        figure.text(left + 0.005, top - row * panel_h - 0.012, str(year), color=PRIMARY_INK, fontsize=13, fontweight="bold",
                    ha="left", va="top")
        figure.text(left + 0.98 / columns - 0.01, top - row * panel_h - 0.016, notes[year],
                    color=SECONDARY_INK, fontsize=9, ha="right", va="top")
    figure.text(0.012, 1 - 0.30 / figure.get_figheight(), title, color=PRIMARY_INK, fontsize=17,
                fontweight="bold", ha="left", va="center")
    figure.text(0.012, 1 - 0.62 / figure.get_figheight(), subtitle, color=SECONDARY_INK, fontsize=10.5, ha="left",
                va="center")
    wide = max(len(label) for label in legend.labels) > 3                       # long labels need wider swatches
    width = min(0.96, len(legend.labels) * (0.0833 if wide else 1.0 / 24.0))
    draw_legend(figure, [(1.0 - width) / 2.0, 0.012, width, 0.55 / figure.get_figheight()], legend, font=10.0)
    figure.text(0.5, 0.012 + 0.62 / figure.get_figheight(), legend.caption, color=MUTED_INK, fontsize=9, ha="center")
    _save_png(figure, path)


def _save_png(figure, path: Path) -> None:
    staging = path.with_name(path.name + ".partial")
    try:
        figure.savefig(staging, format="png", facecolor=figure.get_facecolor())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def check_overview_settings(cell_patches: int, overview_pixels_per_cell: int) -> int:
    """The cell size in DEM pixels; refuses a size the overview canvas does not divide and a missing matplotlib."""
    cell_pixels = int(cell_patches) * 128
    if cell_patches < 1 or overview_pixels_per_cell < 4 or cell_pixels % overview_pixels_per_cell:
        raise InputError("overview_pixels_per_cell must be at least 4 and divide the cell size in pixels (cell_patches * 128)",
                         code="overview_setting_invalid",
                         details={"cell_patches": cell_patches, "overview_pixels_per_cell": overview_pixels_per_cell})
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise InputError("the overview maps need matplotlib (pip install \"lunarscout[plot]\")",
                         code="overview_matplotlib_missing") from error
    return cell_pixels


def refuse_existing(paths, overwrite: bool) -> None:
    for path in paths:
        if Path(path).exists() and not overwrite:
            raise InputError(f"{path} exists; pass overwrite=True to replace it", code="overview_output_exists",
                             details={"path": str(path)})


def overview_paths(directory: str | Path, years, prefix: str = "safe_haven_stage_2") -> dict:
    directory = Path(directory)
    return {"years": {y: {"geotiff": directory / f"{prefix}_persistence_{y}.tif", "png": directory / f"{prefix}_persistence_{y}.png"}
                      for y in years},
            "contact_sheet": directory / f"{prefix}_persistence_contact_sheet.png",
            "report": directory / f"{prefix}_persistence.report.json"}


def run_overview_maps(
    output_directory: str | Path, *, seeds_path: str | Path, dem_path: str | Path, psr_ids_path: str | Path | None = None,
    years=None, cell_patches: int = 2, overview_pixels_per_cell: int = 16, prefix: str = "safe_haven_stage_2",
    overwrite: bool = False,
) -> OverviewReport:
    """Write the persistence GeoTIFF and PNG of each of ``years`` (None: every year in the stage-2 table), the contact
    sheet and ``<prefix>_persistence.report.json`` into ``output_directory``. ``cell_patches`` must be stage 2's (a cell is
    ``cell_patches * 128`` pixels); the overview has ``overview_pixels_per_cell`` canvas pixels per cell, a divisor of
    that."""
    started = clock.monotonic()
    cell_pixels = check_overview_settings(cell_patches, overview_pixels_per_cell)
    transform, width, height, crs_wkt = dem_header(dem_path)
    cell_shape = cell_grid_shape(height, width, cell_pixels)
    grids, seeds = persistence_grids(seeds_path, cell_shape=cell_shape, years=years)
    directory = Path(output_directory)
    paths = overview_paths(directory, list(grids))
    refuse_existing([p for y in paths["years"].values() for p in y.values()] + [paths["contact_sheet"], paths["report"]],
                    overwrite)
    geometry = Geometry(transform, overview_pixels_per_cell, cell_pixels, cell_shape, crs_wkt)
    elevation, psr = read_backdrop(dem_path, psr_ids_path, cell_pixels=cell_pixels, scale=overview_pixels_per_cell,
                                   cell_shape=cell_shape)
    backdrop = backdrop_rgb(elevation, psr, geometry.pixel_metres)
    del elevation, psr
    directory.mkdir(parents=True, exist_ok=True)
    cell_transform = (transform[0], cell_pixels * transform[1], 0.0, transform[3], 0.0, cell_pixels * transform[5])
    images, result = {}, {}
    for year, grid in grids.items():
        image = compose(backdrop, grid, overview_pixels_per_cell)
        images[year] = image
        write_persistence_geotiff(
            paths["years"][year]["geotiff"], grid, transform=cell_transform, crs_wkt=crs_wkt,
            tags={"LUNARSCOUT_PRODUCT": "safe-haven-persistence-v1", "YEAR": str(year),
                  "DESCRIPTION": f"months of {year} in which the 5.12 km cell has at least one valid haven seed (0: none)",
                  "SOURCE_TABLE": Path(seeds_path).name, "CELL_PIXELS": str(cell_pixels)})
        render_year_png(
            paths["years"][year]["png"], image, geometry, title=f"Safe-haven persistence, {year}",
            subtitle="Months of the year in which a 5.12 km cell has at least one valid haven seed",
            stats=_stats_text(grid, seeds[year]), legend=persistence_legend())
        histogram = np.bincount(grid.ravel(), minlength=MONTHS + 1)
        result[year] = {"geotiff": str(paths["years"][year]["geotiff"]), "png": str(paths["years"][year]["png"]),
                        "seeds": seeds[year], "cells_with_seeds": int((grid > 0).sum()),
                        "cells_by_months": [int(v) for v in histogram],
                        "mean_months_of_cells_with_seeds": float(grid[grid > 0].mean()) if (grid > 0).any() else 0.0}
    render_contact_sheet(
        paths["contact_sheet"], images, {year: f"{int((grid > 0).sum()):,} cells" for year, grid in grids.items()},
        geometry, title="Safe-haven persistence by year",
        subtitle="Months of the year in which a 5.12 km cell has at least one valid haven seed; the same scale in every panel",
        legend=persistence_legend())
    report = OverviewReport(years=result, cell_shape=list(cell_shape), cell_pixels=cell_pixels,
                            seconds=clock.monotonic() - started)
    paths["report"].write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return report
