#!/usr/bin/env python3
"""Write one timestamped GeoTIFF per month of the pixels that are safe havens, with minimum-value pyramids.

    .venv/bin/python scripts/threshold_safe_havens.py [--survival-limit-hours 48 --workers 8 --no-overwrite ...]

Every constant in the OPTIONS list below is a command-line option, --constant-name (lower case, hyphens), and the
constant is its default; `--help` lists them. make_haven_month_images(...) takes the same values as keyword arguments, so
another script can call it (scripts/run_safe_haven.py runs the whole safe-haven sequence that way, as its
haven-months step).

The input is the stage-1 hibernation product (scripts/run_safe_haven_stage1.py): one uint8 band per month, the pixel's
hibernation in two-hour counts, NODATA 255. A pixel is a haven in a month when its count is at most the survival limit
(SURVIVAL_LIMIT_HOURS divided by HOURS_PER_COUNT, rounded down). Every pixel of every month is kept, not only the seeds
that stage 2 plans, so the sequence shows all the havens. The values above the limit become NODATA; the values at or
below it are kept, so a viewer can colour a haven by how long it hibernates.

Each input band becomes a DEFLATE-compressed, tiled single-band BigTIFF with internal overviews. Names come from
OUTPUT_FILENAME_TEMPLATE and that band's TIMESTAMP_UTC, normalized to UTC: ``{timestamp:%Y-%m}`` gives
``safe_haven_stage_1_haven_months_2024-01.tif``. Each file records the limit and the number of haven pixels in its tags
(SAFE_HAVEN_BYTE_THRESHOLD, SAFE_HAVEN_SURVIVAL_LIMIT_HOURS, SAFE_HAVEN_HOURS_PER_COUNT, SAFE_HAVEN_PIXELS) and retains its
band's metadata. Missing or invalid timestamps and filename collisions fail preflight before any output is written.
Every overview pixel is the minimum of its aligned 2x2 footprint in the preceding level, so a haven pixel survives at
every zoom; partial edge footprints are supported. Since 255 is larger than every valid value, it is selected only for
all-NODATA footprints. GDAL first allocates nearest-neighbor overviews; all their pixels are then overwritten with the
custom minimum reduction (no averages are used).

Requires rasterio, NumPy, Numba, and Dask (``pip install dask`` if missing). Dask schedules independent read/compute
chunks on CPU threads; Numba compiles the kernels and releases the GIL. Only the coordinating thread writes the TIFF.
Each month is processed separately, with a bounded number of chunks in memory. CPU only; no NVIDIA device is used.

A unique staging directory inside the output directory protects existing files on failure. Each file is published only
once its band and all pyramid levels are complete. A later failure preserves earlier completed files. The complete set is
not an atomic transaction; this script does not resume failures, so run it again (OVERWRITE True replaces the files).

Progress and status messages are logged: to the console, colored with rich on a terminal when
`pip install "lunarscout[console]"` is installed and plain when piped, and to a plain-text log
file in OUTPUT_DIRECTORY (LOG_PATH, COLOR and LOG_LEVEL below).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
import logging
import math
import os
from pathlib import Path
from string import Formatter
import sys
import tempfile
import time

import dask
from dask import delayed
from numba import njit
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._script_cli import (  # noqa: E402
    Option, add_logging_options, add_options, defaults_from, options_from, run_with_logging,
)


def threshold_count(survival_limit_hours: float, hours_per_count: float) -> int:
    """The largest stored count that is within ``survival_limit_hours`` (the decision of
    ``Stage1Rasters.read_haven_window``: counts of ``hours_per_count`` hours, rounded down)."""
    step = float(hours_per_count)
    limit_hours = float(survival_limit_hours)
    if not step > 0.0 or not math.isfinite(limit_hours):
        raise ValueError("hours_per_count must be positive and the survival limit finite")
    count = int(math.floor(limit_hours / step))
    while (count + 1) * step <= limit_hours:
        count += 1
    while count >= 0 and count * step > limit_hours:
        count -= 1
    if count < 0:
        raise ValueError("threshold: no count is within the survival limit")
    return count


# Edit these paths before running. They match run_safe_haven.py.
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
INPUT_PATH = STAGE_DIRECTORY / "safe_haven_stage_1_hibernation_hours.tif"
OUTPUT_DIRECTORY = STAGE_DIRECTORY / "safe_haven_stage_1_haven_months"
# {timestamp:...} formats the band's TIMESTAMP_UTC as a UTC datetime. The viewer
# (scripts/view_safe_haven.py) looks for files named like this default.
OUTPUT_FILENAME_TEMPLATE = "safe_haven_stage_1_haven_months_{timestamp:%Y-%m}.tif"
SURVIVAL_LIMIT_HOURS = 48.0     # a pixel is a haven in a month when its hibernation is at most this
HOURS_PER_COUNT = 2.0           # the stage-1 product stores hibernation in counts of this many hours
THRESHOLD = threshold_count(SURVIVAL_LIMIT_HOURS, HOURS_PER_COUNT)    # the raw byte count of that limit
WORKERS = min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
OVERWRITE = True                # existing files are replaced as the new ones are complete; False refuses to replace them
NODATA = 255
TILE_SIZE = 1024
CHUNK_SIZE = 2048  # One band at a time; overview reads use up to 4096 x 4096.
MIN_OVERVIEW_SIZE = 256  # Stop once the largest overview dimension is <= this.
GDAL_CACHE_BYTES = 256 * 1024 * 1024

# Messages go to the console (colored with rich on a terminal when "lunarscout[console]" is
# installed; plain when piped) and to a plain-text log file, which replaces `| tee`. LOG_PATH
# None writes next to the outputs. COLOR is "auto", "always" (even into a pipe) or "never".
LOG_PATH = None
COLOR = "auto"
LOG_LEVEL = "INFO"
LOG = logging.getLogger("lunarscout.scripts.haven_months")

OPTIONS = (
    Option("INPUT_PATH", "path", "the stage-1 hibernation product (uint8 counts, one band per month, NODATA 255)"),
    Option("OUTPUT_DIRECTORY", "path", "where the monthly files are written"),
    Option("OUTPUT_FILENAME_TEMPLATE", "str", "a file's name, with a {timestamp:...} field for the month"),
    Option("SURVIVAL_LIMIT_HOURS", "float", "a pixel is a haven in a month when its hibernation is at most this (hours)"),
    Option("HOURS_PER_COUNT", "float", "the hours one stored count stands for (stage 1 writes 2)"),
    Option("WORKERS", "int", "threads reading and thresholding chunks"),
    Option("OVERWRITE", "bool", "replace existing files as the new ones are complete"),
)


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
                  source_width: int, source_height: int) -> int:
    """Fill ``target`` from one band (thresholded) or from the previous overview (minimum of 2 x 2); returns the number
    of pixels written that are not NODATA."""
    valid = 0
    windows = iter(_windows(target.width, target.height))
    # Batching prevents Dask from materializing an entire raster or a graph with
    # all months/levels. At most 2*workers chunk results are retained per batch.
    while batch := list(islice(windows, workers * 2)):
        jobs = [delayed(_read_chunk)(
            source_path, source_band, window, source_level, threshold, source_width, source_height
        ) for window in batch]
        for window, values in dask.compute(*jobs):
            target.write(values, 1, window=window)
            valid += int(np.count_nonzero(values != NODATA))
    return valid


def _overview_factors(width: int, height: int) -> list[int]:
    factors = []
    factor = 2
    while max(width, height) > MIN_OVERVIEW_SIZE:
        factors.append(factor)
        width, height = (width + 1) // 2, (height + 1) // 2
        factor *= 2
    return factors


def _output_filename(timestamp: str | None, band: int, template: str | None = None) -> str:
    template = OUTPUT_FILENAME_TEMPLATE if template is None else template
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
        fields = [field for _, field, _, _ in Formatter().parse(template)]
        if "timestamp" not in fields:
            raise ValueError("template must contain a {timestamp:...} field")
        filename = template.format(timestamp=instant)
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
            workers: int = 8, overwrite: bool = False, filename_template: str | None = None,
            extra_tags: dict | None = None) -> tuple[Path, ...]:
    """Convert every band to its own TIFF; publish each completed file atomically. ``filename_template`` (default:
    OUTPUT_FILENAME_TEMPLATE) names the files; ``extra_tags`` are written to every file."""
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
        output_directory / _output_filename(metadata.get("TIMESTAMP_UTC"), band, filename_template)
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
                    threshold=threshold, workers=workers, factors=factors, extra_tags=extra_tags or {},
                )
                _preflight_output(input_path, output_path, overwrite)
                os.replace(staging, output_path)
                print(f"Wrote month {band}/{count}: {output_path}", flush=True)
    print(f"Wrote {count} monthly files in {time.perf_counter() - started:.1f}s", flush=True)
    return output_paths


def _convert_month(input_path: Path, staging: Path, band: int, *, count: int,
                   profile: dict, tags: dict, metadata: dict, description: str | None,
                   scale: float, offset: float, unit: str | None,
                   threshold: int, workers: int, factors: list[int], extra_tags: dict) -> None:
    width, height = profile["width"], profile["height"]
    with rasterio.open(staging, "w", **profile) as target:
        target.update_tags(**tags)
        target.update_tags(
            TIMESTAMP_UTC=metadata["TIMESTAMP_UTC"],
            SAFE_HAVEN_SOURCE_BAND=str(band),
            SAFE_HAVEN_BYTE_THRESHOLD=str(threshold),
            SAFE_HAVEN_OVERVIEW_REDUCTION="minimum-valid-aligned-2x2",
            **{key: str(value) for key, value in extra_tags.items()},
        )
        target.scales, target.offsets = (scale,), (offset,)
        target.update_tags(1, **metadata)
        if description is not None:
            target.set_band_description(1, description)
        if unit is not None:
            target.set_band_unit(1, unit)
        pixels = _process_band(input_path, target, band, workers=workers,
                               source_level=None, threshold=threshold,
                               source_width=width, source_height=height)
        target.update_tags(SAFE_HAVEN_PIXELS=str(pixels))
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


def default_options() -> dict:
    """The options with this file's constants as their values (read now)."""
    return defaults_from(OPTIONS, globals())


def make_haven_month_images(*, input_path, output_directory, output_filename_template, survival_limit_hours,
                            hours_per_count, workers, overwrite) -> int:
    """Write one file per month of the pixels within ``survival_limit_hours``; returns 0."""
    started = time.perf_counter()
    threshold = threshold_count(survival_limit_hours, hours_per_count)
    LOG.info("Havens: hibernation at most %g h (counts up to %d of %g h each)", survival_limit_hours, threshold, hours_per_count)
    paths = convert(Path(input_path), Path(output_directory), threshold=threshold, workers=workers, overwrite=overwrite,
                    filename_template=output_filename_template,
                    extra_tags={"SAFE_HAVEN_SURVIVAL_LIMIT_HOURS": f"{float(survival_limit_hours):g}",
                                "SAFE_HAVEN_HOURS_PER_COUNT": f"{float(hours_per_count):g}"})
    LOG.info("%d monthly files written to %s in %.1f s", len(paths), output_directory, time.perf_counter() - started)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One GeoTIFF per month of the pixels that are safe havens (hibernation at most the survival limit). "
                    "Every option's default is the constant of the same name at the top of this file.")
    add_options(parser, OPTIONS, default_options())
    add_logging_options(parser, globals())
    return parser


def main(argv=()) -> int:
    """Read the options in ``argv`` (none: the defaults), set up logging and write the monthly files."""
    args = build_parser().parse_args(list(argv))
    options = options_from(args, OPTIONS)
    output_directory = Path(options["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    return run_with_logging(args, default_log_path=output_directory / "safe_haven_stage_1_haven_months.log", logger=LOG,
                            label="Haven months", procedure=lambda: make_haven_month_images(**options))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
