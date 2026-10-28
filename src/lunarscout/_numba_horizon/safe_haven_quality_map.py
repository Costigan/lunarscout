"""Overview maps of how long a rover mission can be from each part of the pole, one per year (private).

The stage-4 table gives every seed a ``mission_hours``: the hours from the first wake-up of the chain that reaches the
seed to the end of the seed's lit endurance (a seed whose lit endurance is cut by the 28-day flood cap gives a lower
bound). The **quality** of a 5.12 km cell in a year is the longest ``mission_hours`` among the seeds of the cell whose
month is in that year: the longest mission a rover can have completed by a haven in the cell. It comes straight from the
table, like the persistence map, and is placed by the seed's own month, so the two maps agree on which seeds belong to a
year.

Per year the run writes a one-band float32 GeoTIFF on the cell grid (``<prefix>_cell_quality_<year>.tif``: hours, nan
where the cell has no seed, in the DEM's CRS, for QGIS), a PNG of the pole (``<prefix>_cell_quality_<year>.png``: the
same hillshade, PSRs and graticule as the persistence map and one square per cell, in one orange hue brighter for longer
missions) and, for all years together, a contact sheet with one colour scale (``<prefix>_cell_quality_contact_sheet.png``).
The colour classes are in days: below 2, 2 to 4, 4 to 7, 7 to 10, 10 to 14, 14 to 21, 21 to 60 and 60 or more.
"""

from __future__ import annotations

import json
import os
import time as clock
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .pixel_coordinates import dem_header
from .safe_haven_overview import (
    MONTHS, Geometry, Legend, backdrop_rgb, cell_grid_shape, check_overview_settings, compose, read_backdrop,
    read_cell_table, refuse_existing, render_contact_sheet, render_year_png, resolve_years, _rgb,
)

# Class edges in hours: a cell is in class k (1-based) when its longest mission is at least edge k-1 and below edge k.
QUALITY_EDGES_HOURS = (48.0, 96.0, 168.0, 240.0, 336.0, 504.0, 1440.0)
QUALITY_LABELS = ("<2", "2-4", "4-7", "7-10", "10-14", "14-21", "21-60", "60+")      # days

# One orange hue, brighter for longer missions: tints of the reference orange (#d95926 on the dark surface), dim enough
# to show against the dark hillshade at the low end.
_ORANGE_STEPS = ("#a8461c", "#c4511f", "#d95926", "#e57444", "#ee8f66", "#f4aa88", "#f8c5ac", "#fbe0d0")


def quality_colours() -> npt.NDArray[np.uint8]:
    """``(9, 3)`` RGB: row 0 (no seed) is unused, rows 1 to 8 are the classes from dimmer to brighter orange."""
    return np.array([(0, 0, 0)] + [_rgb(c) for c in _ORANGE_STEPS], dtype=np.uint8)


def quality_legend() -> Legend:
    return Legend(quality_colours(), QUALITY_LABELS, "longest mission, days (cells with no seed show the terrain)")


def quality_classes(hours: npt.NDArray) -> npt.NDArray[np.uint8]:
    """The colour class 1 to 8 of each cell's longest mission in hours; 0 where it is nan (no seed)."""
    classes = np.searchsorted(np.array(QUALITY_EDGES_HOURS), np.nan_to_num(hours, nan=0.0), side="right") + 1
    return np.where(np.isnan(hours), 0, classes).astype(np.uint8)


@dataclass
class QualityReport:
    years: dict                      # per year: the files, the seeds, the cells with seeds, the cells in each class
    cell_shape: list
    cell_pixels: int
    classes_days: list               # the label of each colour class
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def quality_grids(chains_path: str | Path, *, cell_shape: tuple[int, int], years=None):
    """``({year: float32 grid of cell rows x columns}, {year: seeds})`` from the stage-4 table: each cell's longest
    ``mission_hours`` among the seeds of the year, nan where there is none."""
    rows_n, cols_n = cell_shape
    month_keys, cell_rows, cell_cols, mission = read_cell_table(chains_path, cell_shape, "mission_hours")
    years = resolve_years(month_keys, years)
    grids, seeds = {}, {}
    for year in years:
        in_year = (month_keys // MONTHS) == year
        seeds[year] = int(in_year.sum())
        grid = np.full((rows_n, cols_n), np.nan, dtype=np.float32)
        np.fmax.at(grid, (cell_rows[in_year], cell_cols[in_year]), mission[in_year].astype(np.float32))
        grids[year] = grid
    return grids, seeds


def write_quality_geotiff(path: Path, hours, *, transform, crs_wkt: str, tags: dict) -> None:
    """The longest missions as a one-band float32 GeoTIFF on the cell grid (nodata nan), through a staging file."""
    import rasterio
    from rasterio.transform import Affine

    staging = path.with_name(path.name + ".partial")
    try:
        with rasterio.open(staging, "w", driver="GTiff", height=hours.shape[0], width=hours.shape[1], count=1,
                           dtype="float32", nodata=float("nan"), transform=Affine.from_gdal(*transform),
                           crs=crs_wkt or None, compress="deflate", tiled=False) as dataset:
            dataset.write(hours[None])
            dataset.set_band_description(1, "longest_mission_hours")
            dataset.update_tags(**tags)
        os.replace(staging, path)
        path.with_name(path.name + ".aux.xml").unlink(missing_ok=True)       # a sidecar of the file just replaced
    finally:
        staging.unlink(missing_ok=True)


def quality_paths(directory: str | Path, years, prefix: str = "safe_haven_stage_4") -> dict:
    directory = Path(directory)
    return {"years": {y: {"geotiff": directory / f"{prefix}_cell_quality_{y}.tif", "png": directory / f"{prefix}_cell_quality_{y}.png"}
                      for y in years},
            "contact_sheet": directory / f"{prefix}_cell_quality_contact_sheet.png",
            "report": directory / f"{prefix}_cell_quality.report.json"}


def _stats_text(hours, seeds: int) -> str:
    present = hours[~np.isnan(hours)]
    if not len(present):
        return "no cell has a seed"
    long_cells = int((present >= QUALITY_EDGES_HOURS[-2]).sum())
    return (f"{len(present):,} of {hours.size:,} cells have a seed; median {np.median(present) / 24:.1f} days, longest "
            f"{present.max() / 24:.0f} days; {long_cells:,} cells reach 21 days or more; {seeds:,} seeds")


def run_quality_maps(
    output_directory: str | Path, *, chains_path: str | Path, dem_path: str | Path, psr_ids_path: str | Path | None = None,
    years=None, cell_patches: int = 2, overview_pixels_per_cell: int = 16, prefix: str = "safe_haven_stage_4",
    overwrite: bool = False,
) -> QualityReport:
    """Write the quality GeoTIFF and PNG of each of ``years`` (None: every year in the stage-4 table), the contact sheet
    and ``<prefix>_cell_quality.report.json`` into ``output_directory``. ``cell_patches`` must be stage 2's (a cell is
    ``cell_patches * 128`` pixels); the overview has ``overview_pixels_per_cell`` canvas pixels per cell, a divisor of
    that."""
    started = clock.monotonic()
    cell_pixels = check_overview_settings(cell_patches, overview_pixels_per_cell)
    transform, width, height, crs_wkt = dem_header(dem_path)
    cell_shape = cell_grid_shape(height, width, cell_pixels)
    grids, seeds = quality_grids(chains_path, cell_shape=cell_shape, years=years)
    directory = Path(output_directory)
    paths = quality_paths(directory, list(grids), prefix)
    refuse_existing([p for y in paths["years"].values() for p in y.values()] + [paths["contact_sheet"], paths["report"]],
                    overwrite)
    geometry = Geometry(transform, overview_pixels_per_cell, cell_pixels, cell_shape, crs_wkt)
    elevation, psr = read_backdrop(dem_path, psr_ids_path, cell_pixels=cell_pixels, scale=overview_pixels_per_cell,
                                   cell_shape=cell_shape)
    backdrop = backdrop_rgb(elevation, psr, geometry.pixel_metres)
    del elevation, psr
    directory.mkdir(parents=True, exist_ok=True)
    cell_transform = (transform[0], cell_pixels * transform[1], 0.0, transform[3], 0.0, cell_pixels * transform[5])
    legend = quality_legend()
    images, result = {}, {}
    for year, hours in grids.items():
        classes = quality_classes(hours)
        images[year] = compose(backdrop, classes, overview_pixels_per_cell, colours=legend.colours)
        write_quality_geotiff(
            paths["years"][year]["geotiff"], hours, transform=cell_transform, crs_wkt=crs_wkt,
            tags={"LUNARSCOUT_PRODUCT": "safe-haven-cell-quality-v1", "YEAR": str(year),
                  "DESCRIPTION": f"longest mission_hours of the seeds of {year} in the 5.12 km cell (nan: no seed)",
                  "SOURCE_TABLE": Path(chains_path).name, "CELL_PIXELS": str(cell_pixels)})
        render_year_png(
            paths["years"][year]["png"], images[year], geometry, title=f"Longest mission per cell, {year}",
            subtitle="Longest mission that ends at a haven of the 5.12 km cell, by the month of the haven",
            stats=_stats_text(hours, seeds[year]), legend=legend)
        present = hours[~np.isnan(hours)]
        result[year] = {"geotiff": str(paths["years"][year]["geotiff"]), "png": str(paths["years"][year]["png"]),
                        "seeds": seeds[year], "cells_with_seeds": int(len(present)),
                        "cells_by_class": [int(v) for v in np.bincount(classes.ravel(), minlength=len(QUALITY_LABELS) + 1)],
                        "median_hours": float(np.median(present)) if len(present) else 0.0,
                        "longest_hours": float(present.max()) if len(present) else 0.0}
    render_contact_sheet(
        paths["contact_sheet"], images, {year: f"{int(np.sum(~np.isnan(h))):,} cells" for year, h in grids.items()},
        geometry, title="Longest mission per cell by year",
        subtitle="Longest mission, in days, that ends at a haven of a 5.12 km cell; the same scale in every panel",
        legend=legend)
    report = QualityReport(years=result, cell_shape=list(cell_shape), cell_pixels=cell_pixels,
                           classes_days=list(QUALITY_LABELS), seconds=clock.monotonic() - started)
    paths["report"].write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return report
