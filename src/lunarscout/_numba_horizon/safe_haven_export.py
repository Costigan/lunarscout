"""Full-resolution GeoTIFF exports of the safe-haven viewer's pictures.

``export_background`` writes the hillshade with the permanent shadow on it as one tile-compressed 3-band (RGB) GeoTIFF.
``export_monthly_chains`` writes the pictures of the Chains (monthly) layer, with the viewer's two filters (the PSR class a
chain touches and its longest hop), as one tile-compressed 4-band (RGBA) GeoTIFF per month, without the background: alpha is 255
where something is drawn and 0 elsewhere. Both are on the DEM's grid, so they line up with it in a GIS. The pixels are
those the viewer draws at its finest zoom, from the same data.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import InputError, ProductStorageError
from .pixel_coordinates import dem_header
from .safe_haven_viewer_data import TILE, ViewerData

Progress = Callable[[str], None]

PSR_NAMES = ("any", "small", "medium", "large")        # by PSR class 0-3: the smallest PSR a chain must touch
BACKGROUND_NAME = "hillshade_psr_rgb.tif"
BACKGROUND_COMPRESSIONS = ("deflate", "webp", "jpeg")     # lossless DEFLATE (opens everywhere), lossless WebP, lossy JPEG (YCbCr)
JPEG_QUALITY = 95


def _quiet(_message: str) -> None:
    return None


@dataclass
class ExportReport:
    """What an export wrote: the files, the months that had no chains and (for the chains) the chain counts."""

    background: str | None = None
    chain_files: dict[str, str] = field(default_factory=dict)         # month -> path
    chains: dict[str, int] = field(default_factory=dict)              # month -> chains drawn
    empty_months: list[str] = field(default_factory=list)             # months with no chain passing the filters, no file written
    pixels: dict[str, int] = field(default_factory=dict)              # month -> pixels drawn

    def as_dict(self) -> dict:
        return {"background": self.background, "chain_files": dict(self.chain_files), "chains": dict(self.chains),
                "empty_months": list(self.empty_months), "pixels": dict(self.pixels)}


def chain_image_name(month: str, psr_class: int, hop_m: float) -> str:
    """``chains_monthly_<psr>_hop_<metres>m_<month>.tif``, for example ``chains_monthly_medium_hop_500m_2026-03.tif``."""
    hop = f"{int(hop_m)}" if float(hop_m) == int(hop_m) else f"{hop_m:g}"
    return f"chains_monthly_{PSR_NAMES[psr_class]}_hop_{hop}m_{month}.tif"


def _check_filters(psr_class: int, hop_m: float) -> None:
    if int(psr_class) != psr_class or not 0 <= psr_class <= 3:
        raise InputError("psr_class must be 0 (any), 1 (small), 2 (medium) or 3 (large)", code="export_psr_class_invalid",
                         details={"psr_class": psr_class})
    if not np.isfinite(hop_m) or hop_m < 0:
        raise InputError("hop_m must be a finite number of metres, 0 or more", code="export_hop_invalid",
                         details={"hop_m": hop_m})


def _open_output(path: Path, *, count: int, size: int, transform_gdal, crs_wkt: str, alpha: bool, tags: dict,
                 compression: str = "deflate", jpeg_quality: int = JPEG_QUALITY):
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import Affine

    codec = {"deflate": dict(compress="deflate", predictor=2, photometric="RGB"),
             "webp": dict(compress="webp", webp_lossless=True, photometric="RGB"),
             "jpeg": dict(compress="jpeg", quality=int(jpeg_quality), photometric="YCBCR")}[compression]
    dataset = rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=count, dtype="uint8",
                            crs=crs_wkt or None, transform=Affine.from_gdal(*transform_gdal), tiled=True, blockxsize=TILE,
                            blockysize=TILE, interleave="pixel", BIGTIFF="YES", SPARSE_OK="TRUE", **codec)
    if compression != "jpeg":                  # a YCbCr image keeps the colour interpretation GDAL gives it
        dataset.colorinterp = ((ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha) if alpha
                               else (ColorInterp.red, ColorInterp.green, ColorInterp.blue))
    dataset.update_tags(**tags)
    return dataset


def _write_tile(dataset, tx: int, ty: int, size: int, tile: np.ndarray) -> None:
    from rasterio.windows import Window

    x0, y0 = tx * TILE, ty * TILE
    width, height = min(TILE, size - x0), min(TILE, size - y0)
    dataset.write(np.moveaxis(tile[:height, :width], -1, 0), window=Window(x0, y0, width, height))


def _publish(staging: Path, final: Path) -> None:
    os.replace(staging, final)


def export_background(viewer_directory: str | Path, cache_directory: str | Path, psr_path: str | Path,
                      output_path: str | Path, *, compression: str = "deflate", jpeg_quality: int = JPEG_QUALITY,
                      progress: Progress = _quiet) -> Path:
    """Write the hillshade with the permanent shadow composited on it (the viewer's background at full resolution) as a
    tile-compressed RGB GeoTIFF on the DEM's grid. The pyramid must be in the cache (the viewer or ``prepare_viewer`` makes it).
    ``compression`` is "deflate" (lossless, the default, opens in every tool), "webp" (lossless, about 27% smaller on the real
    background, slower to write, needs a WebP-enabled reader) or "jpeg" (lossy at ``jpeg_quality`` 1-100, YCbCr, about 7 times smaller
    than DEFLATE at 95). It is written to a staging file and renamed, so a failed run leaves an existing file as it was."""
    if compression not in BACKGROUND_COMPRESSIONS:
        raise InputError(f"compression must be one of {BACKGROUND_COMPRESSIONS}", code="export_compression_invalid",
                         details={"compression": compression})
    if not 1 <= int(jpeg_quality) <= 100:
        raise InputError("jpeg_quality must be 1 to 100", code="export_jpeg_quality_invalid",
                         details={"jpeg_quality": jpeg_quality})
    if not Path(psr_path).is_file():
        raise InputError(f"{psr_path} is not a file", code="export_psr_missing", details={"psr": str(psr_path)})
    data = ViewerData(viewer_directory, cache_directory, psr_path)
    if not data.has_background():
        raise ProductStorageError("the background pyramid is not in the cache (run the viewer or prepare_viewer first)",
                                  code="export_background_missing", details={"cache": str(cache_directory)})
    transform, _width, _height, crs = dem_header(data.meta["dem"]["path"])
    final = Path(output_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(final.name + ".partial")
    staging.unlink(missing_ok=True)
    across = data.tiles_across(0)
    try:
        with _open_output(staging, count=3, size=data.size, transform_gdal=transform, crs_wkt=crs, alpha=False,
                          compression=compression, jpeg_quality=jpeg_quality,
                          tags={"DESCRIPTION": "hillshade with the permanent shadow (light blue, 30% opacity)"}) as dataset:
            for ty in range(across):
                progress(f"Background: tile row {ty + 1} of {across}")
                for tx in range(across):
                    _write_tile(dataset, tx, ty, data.size, data.background_tile(0, tx, ty))
        _publish(staging, final)
    finally:
        staging.unlink(missing_ok=True)
    return final


def _tile_buckets(rows: np.ndarray, columns: np.ndarray) -> dict[int, np.ndarray]:
    """``{ty * 100000 + tx: indices of the pixels that lie in tile (tx, ty)}``, each in the order of the arrays."""
    keys = (rows >> 9) * 100_000 + (columns >> 9)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_keys)) + 1))
    return {int(sorted_keys[start]): group for start, group in zip(starts, np.split(order, starts[1:]))}


def export_monthly_chains(viewer_directory: str | Path, cache_directory: str | Path, output_directory: str | Path, *,
                          psr_class: int = 2, hop_m: float = 500.0, write_empty: bool = False,
                          progress: Progress = _quiet) -> ExportReport:
    """Write the Chains (monthly) layer of each month as a tile-compressed RGBA GeoTIFF without the background (alpha 255
    where a pixel is drawn, 0 elsewhere), for the chains that touch a PSR of at least ``psr_class`` (0 any, 1 small, 2 medium,
    3 large) and whose longest hop is at least ``hop_m`` metres. A month with no such chain gets no file (an empty, fully
    transparent one with ``write_empty``) and is listed in the report. Files are written to staging and renamed."""
    _check_filters(psr_class, hop_m)
    data = ViewerData(viewer_directory, cache_directory)
    if data.monthly_chains is None:
        raise ProductStorageError("the viewer has no monthly chain file (run the monthly-chain-maps step)",
                                  code="export_chains_missing", details={"viewer": str(viewer_directory)})
    if psr_class > 0 and not data.psr_classes_known:
        raise InputError("the monthly chain file has no PSR classes (stage 4 ran without the routes), so only psr_class 0 is possible",
                         code="export_psr_unknown", details={"psr_class": psr_class})
    if hop_m > 0 and not data.has_chain_filters():
        raise InputError("the monthly chain file keeps no per-chain table, so the filters are unavailable (run the monthly-chain-maps step again)",
                         code="export_filters_unavailable", details={"hop_m": hop_m})
    transform, _width, _height, crs = dem_header(data.meta["dem"]["path"])
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = ExportReport()
    across = data.tiles_across(0)
    for index, month in enumerate(data.months):
        chains = data.monthly_chain_count(index, psr_class, hop_m)
        pixels = data.monthly_chain_pixels(index, psr_class, hop_m)
        count = 0 if pixels is None else len(pixels[0])
        if chains == 0 or count == 0:
            report.empty_months.append(month)
            if not write_empty:
                progress(f"{month}: no chains pass the filters, no file")
                continue
        final = directory / chain_image_name(month, psr_class, hop_m)
        staging = final.with_name(final.name + ".partial")
        staging.unlink(missing_ok=True)
        tags = {"DESCRIPTION": f"Chains (monthly) {month}: PSR {PSR_NAMES[psr_class]} or larger, longest hop at least {hop_m:g} m",
                "MONTH": month, "PSR_CLASS_AT_LEAST": str(psr_class), "LONGEST_HOP_AT_LEAST_M": f"{hop_m:g}",
                "CHAINS": str(chains)}
        try:
            with _open_output(staging, count=4, size=data.size, transform_gdal=transform, crs_wkt=crs, alpha=True,
                              tags=tags) as dataset:
                if count:
                    rows, columns, codes = pixels
                    buckets = _tile_buckets(rows, columns)
                    for ty in range(across):
                        for tx in range(across):
                            chosen = [buckets[key] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                                      if 0 <= tx + dx < across and 0 <= ty + dy < across
                                      and (key := (ty + dy) * 100_000 + tx + dx) in buckets]
                            if not chosen:
                                continue
                            keep = np.sort(np.concatenate(chosen))          # the file's order: the last pixel drawn is on top
                            tile = data.paint_chain_tile((rows[keep], columns[keep], codes[keep]), 0, tx, ty)
                            if tile[..., 3].any():
                                _write_tile(dataset, tx, ty, data.size, tile)
                progress(f"{month}: {chains:,} chains, {count:,} pixels")
            _publish(staging, final)
        finally:
            staging.unlink(missing_ok=True)
        report.chain_files[month] = str(final)
        report.chains[month] = int(chains)
        report.pixels[month] = int(count)
    return report
