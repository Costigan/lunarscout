#!/usr/bin/env python3
"""Write timestamped single-band safe-haven TIFFs with minimum-value pyramids.

Edit THRESHOLD (raw byte counts, not hours) and OUTPUT_FILENAME_TEMPLATE below,
then run:

    .venv/bin/python scripts/threshold_safe_havens.py
    .venv/bin/python scripts/threshold_safe_havens.py input.tif output_directory --workers 8

Requires rasterio, NumPy, Numba, and Dask (``pip install dask`` if missing).
Dask schedules independent read/compute chunks on CPU threads; Numba compiles
the kernels and releases the GIL. Only the coordinating thread writes the TIFF.
Each month is processed separately, with a bounded number of chunks in memory.

Each input band becomes a DEFLATE-compressed, tiled single-band BigTIFF with
internal overviews. Names use that band's TIMESTAMP_UTC, normalized to UTC:
``safe_havens_2024-01-01T00-00-00.tif``. OUTPUT_FILENAME_TEMPLATE uses Python
datetime formatting; add ``.%f`` before the closing brace to include microseconds.
Missing or invalid timestamps and filename collisions fail
preflight before any output is written. Each file retains its band's metadata.
Every overview pixel is the minimum of its aligned 2x2 footprint
in the preceding level; partial edge footprints are supported. Since 255 is
larger than every valid value, it is selected only for all-NODATA footprints.
GDAL first allocates nearest-neighbor overviews; all their pixels are then
overwritten with the custom minimum reduction (no averages are used).

A unique staging directory inside the output directory protects existing files
on failure. Each file is published only once its band and all pyramid levels
are complete. A later failure preserves earlier completed files. The complete
set is not an atomic transaction; this script does not resume failures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
import os
from pathlib import Path
from string import Formatter
import tempfile
import time

import dask
from dask import delayed
from numba import njit
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


INPUT_PATH = Path("/e/lunar_analyst_scenarios/70south/safe_havens.tif")
OUTPUT_DIRECTORY = INPUT_PATH.with_name("safe_havens_thresholded")
# EDIT: {timestamp:...} formats the band's TIMESTAMP_UTC as a UTC datetime.
# For example: "sun_image_{timestamp:%Y-%m-%dT%H-%M-%S}.tif"
OUTPUT_FILENAME_TEMPLATE = "safe_havens_{timestamp:%Y-%m-%dT%H-%M-%S}.tif"
THRESHOLD = 25  # EDIT: raw byte count; 25 represents 50 hours in the source.
NODATA = 255
TILE_SIZE = 1024
CHUNK_SIZE = 2048  # One band at a time; overview reads use up to 4096 x 4096.
MIN_OVERVIEW_SIZE = 256  # Stop once the largest overview dimension is <= this.
GDAL_CACHE_BYTES = 256 * 1024 * 1024


@njit(cache=True, nogil=True)
def _threshold(values: np.ndarray, threshold: np.uint8) -> np.ndarray:
    result = np.empty_like(values)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            result[y, x] = value if value <= threshold else NODATA
    return result


@njit(cache=True, nogil=True)
def _minimum2(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    result = np.full(((height + 1) // 2, (width + 1) // 2), NODATA, dtype=np.uint8)
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            minimum = NODATA
            for row in range(y * 2, min(y * 2 + 2, height)):
                for column in range(x * 2, min(x * 2 + 2, width)):
                    minimum = min(minimum, int(values[row, column]))
            result[y, x] = minimum
    return result


def _windows(width: int, height: int):
    for y in range(0, height, CHUNK_SIZE):
        for x in range(0, width, CHUNK_SIZE):
            yield Window(x, y, min(CHUNK_SIZE, width - x), min(CHUNK_SIZE, height - y))


def _read_chunk(path: Path, band: int, window: Window, source_level: int | None,
                threshold: int | None, source_width: int, source_height: int):
    options = {} if source_level is None else {"overview_level": source_level}
    with rasterio.open(path, sharing=False, NUM_THREADS="1", **options) as source:
        if threshold is None:
            read_window = Window(
                window.col_off * 2, window.row_off * 2,
                min(window.width * 2, source_width - window.col_off * 2),
                min(window.height * 2, source_height - window.row_off * 2),
            )
        else:
            read_window = window
        values = source.read(band, window=read_window, masked=True).filled(NODATA)
    result = _minimum2(values) if threshold is None else _threshold(values, np.uint8(threshold))
    return window, result


def _process_band(source_path: Path, target, source_band: int, *, workers: int,
                  source_level: int | None, threshold: int | None,
                  source_width: int, source_height: int) -> None:
    windows = iter(_windows(target.width, target.height))
    # Batching prevents Dask from materializing an entire raster or a graph with
    # all months/levels. At most 2*workers chunk results are retained per batch.
    while batch := list(islice(windows, workers * 2)):
        jobs = [delayed(_read_chunk)(
            source_path, source_band, window, source_level, threshold, source_width, source_height
        ) for window in batch]
        for window, values in dask.compute(*jobs):
            target.write(values, 1, window=window)


def _overview_factors(width: int, height: int) -> list[int]:
    factors = []
    factor = 2
    while max(width, height) > MIN_OVERVIEW_SIZE:
        factors.append(factor)
        width, height = (width + 1) // 2, (height + 1) // 2
        factor *= 2
    return factors


def _output_filename(timestamp: str | None, band: int) -> str:
    if not timestamp:
        raise ValueError(f"Band {band} is missing TIMESTAMP_UTC metadata")
    try:
        instant = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        instant = instant.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Band {band} has invalid TIMESTAMP_UTC: {timestamp!r}") from exc
    try:
        fields = [field for _, field, _, _ in Formatter().parse(OUTPUT_FILENAME_TEMPLATE)]
        if "timestamp" not in fields:
            raise ValueError("template must contain a {timestamp:...} field")
        filename = OUTPUT_FILENAME_TEMPLATE.format(timestamp=instant)
    except (ValueError, KeyError, IndexError, AttributeError, TypeError) as exc:
        raise ValueError(f"Invalid OUTPUT_FILENAME_TEMPLATE: {exc}") from exc
    if (
        not filename or any(character in filename for character in '/\\:*?"<>|\x00')
        or Path(filename).suffix.lower() not in (".tif", ".tiff")
    ):
        raise ValueError("OUTPUT_FILENAME_TEMPLATE must produce a plain, filesystem-safe .tif or .tiff filename")
    return filename


def _preflight_output(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if input_path == output_path or (output_path.exists() and os.path.samefile(input_path, output_path)):
        raise ValueError("input and output must be different files")
    if output_path.is_dir():
        raise IsADirectoryError(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite to replace it")
    for suffix in (".ovr", ".msk", ".aux.xml"):
        if Path(str(output_path) + suffix).exists():
            raise FileExistsError(f"Move the existing output sidecar first: {output_path}{suffix}")


def convert(input_path: Path, output_directory: Path, *, threshold: int = THRESHOLD,
            workers: int = 8, overwrite: bool = False) -> tuple[Path, ...]:
    """Convert every band to its own TIFF; publish each completed file atomically."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, np.integer)) or not 0 <= threshold <= 255:
        raise ValueError("threshold must be an integer byte value in [0, 255]")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    input_path = Path(input_path).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if output_directory.exists() and not output_directory.is_dir():
        raise NotADirectoryError(output_directory)
    with rasterio.open(input_path) as source:
        if any(dtype != "uint8" for dtype in source.dtypes):
            raise ValueError("Input must contain only GDT_Byte/uint8 bands")
        if any(value != NODATA for value in source.nodatavals):
            raise ValueError("Input must declare NODATA=255 for every band")
        width, height, count = source.width, source.height, source.count
        profile = dict(
            driver="GTiff", width=width, height=height, count=1,
            dtype="uint8", nodata=NODATA, crs=source.crs, transform=source.transform,
            tiled=True, blockxsize=TILE_SIZE, blockysize=TILE_SIZE,
            compress="DEFLATE", zlevel=6, predictor=2, interleave="band",
            photometric="MINISBLACK", BIGTIFF="YES", NUM_THREADS=str(workers),
        )
        # Source statistics describe the unthresholded values and must not be
        # advertised by the derived raster (GIS viewers may otherwise mis-scale).
        tags = {
            key: value for key, value in source.tags().items()
            if not key.startswith("STATISTICS_") and key not in (
                "LUNARSCOUT_TIMESTAMPS_UTC", "MONTHS", "TIMESTAMP_UTC",
            )
        }
        band_tags = [
            {key: value for key, value in source.tags(band).items() if not key.startswith("STATISTICS_")}
            for band in source.indexes
        ]
        descriptions, scales, offsets, units = source.descriptions, source.scales, source.offsets, source.units
    output_paths = tuple(
        output_directory / _output_filename(metadata.get("TIMESTAMP_UTC"), band)
        for band, metadata in enumerate(band_tags, start=1)
    )
    if len(set(output_paths)) != count:
        raise ValueError("TIMESTAMP_UTC values or OUTPUT_FILENAME_TEMPLATE produce duplicate output filenames")
    # Check every destination before converting the first month, including
    # stale sidecars that could override the new TIFF's metadata or overviews.
    for output_path in output_paths:
        _preflight_output(input_path, output_path, overwrite)
    output_directory.mkdir(parents=True, exist_ok=True)
    factors = _overview_factors(width, height)
    print(f"Input: {width} x {height}, {count} monthly bands; threshold={threshold}; workers={workers}", flush=True)
    # Compile once before launching tasks, avoiding concurrent first compilation.
    sample = np.zeros((2, 2), dtype=np.uint8)
    _threshold(sample, np.uint8(threshold))
    _minimum2(sample)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool, dask.config.set(scheduler="threads", pool=pool), rasterio.Env(
            GDAL_CACHEMAX=GDAL_CACHE_BYTES, GDAL_TIFF_OVR_BLOCKSIZE=str(TILE_SIZE),
            COMPRESS_OVERVIEW="DEFLATE", PREDICTOR_OVERVIEW="2",
            INTERLEAVE_OVERVIEW="BAND", GDAL_NUM_THREADS=str(workers),
    ):
        for band, output_path in enumerate(output_paths, start=1):
            with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.", dir=output_directory) as folder:
                staging = Path(folder) / "product.tif"
                _convert_month(
                    input_path, staging, band, count=count, profile=profile, tags=tags,
                    metadata=band_tags[band - 1], description=descriptions[band - 1],
                    scale=scales[band - 1], offset=offsets[band - 1], unit=units[band - 1],
                    threshold=threshold, workers=workers, factors=factors,
                )
                _preflight_output(input_path, output_path, overwrite)
                os.replace(staging, output_path)
                print(f"Wrote month {band}/{count}: {output_path}", flush=True)
    print(f"Wrote {count} monthly files in {time.perf_counter() - started:.1f}s", flush=True)
    return output_paths


def _convert_month(input_path: Path, staging: Path, band: int, *, count: int,
                   profile: dict, tags: dict, metadata: dict, description: str | None,
                   scale: float, offset: float, unit: str | None,
                   threshold: int, workers: int, factors: list[int]) -> None:
    width, height = profile["width"], profile["height"]
    with rasterio.open(staging, "w", **profile) as target:
        target.update_tags(**tags)
        target.update_tags(
            TIMESTAMP_UTC=metadata["TIMESTAMP_UTC"],
            SAFE_HAVEN_SOURCE_BAND=str(band),
            SAFE_HAVEN_BYTE_THRESHOLD=str(threshold),
            SAFE_HAVEN_OVERVIEW_REDUCTION="minimum-valid-aligned-2x2",
        )
        target.scales, target.offsets = (scale,), (offset,)
        target.update_tags(1, **metadata)
        if description is not None:
            target.set_band_description(1, description)
        if unit is not None:
            target.set_band_unit(1, unit)
        _process_band(input_path, target, band, workers=workers,
                      source_level=None, threshold=threshold,
                      source_width=width, source_height=height)
        print(f"Threshold: band {band}/{count}", flush=True)
        if factors:
            print(f"Allocating internal pyramid: {factors}", flush=True)
            target.build_overviews(factors, Resampling.nearest)
    previous_width, previous_height = width, height
    for level, factor in enumerate(factors):
        with rasterio.open(staging, "r+", overview_level=level, NUM_THREADS=str(workers)) as target:
            expected = ((previous_width + 1) // 2, (previous_height + 1) // 2)
            if (target.width, target.height) != expected:
                raise RuntimeError(f"Unexpected overview dimensions at factor {factor}")
            # The staging TIFF has one band regardless of its source-band index.
            _process_band(staging, target, 1, workers=workers,
                          source_level=level - 1 if level else None,
                          threshold=None, source_width=previous_width,
                          source_height=previous_height)
            print(f"Minimum pyramid: factor {factor}, band {band}/{count}", flush=True)
            previous_width, previous_height = target.width, target.height
    with rasterio.open(staging, "r+") as target:
        target.update_tags(ns="rio_overview", resampling="min")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path, default=INPUT_PATH)
    parser.add_argument("output_directory", nargs="?", type=Path, default=OUTPUT_DIRECTORY)
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    parser.add_argument("--workers", type=int, default=min(8, available))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    convert(args.input, args.output_directory, threshold=THRESHOLD, workers=args.workers, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
