#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    PYTHONPATH=src .venv/bin/python scripts/measure_light_store.py

Measures what the stage-0 thresholded-lighting store would cost on real patches
(src/lunarscout/_numba_horizon/safe_haven_lightstore.py). For each patch in PATCHES it reads the
horizon tile, lights the whole time range on the CPU (Sun fraction and Earth margin thresholded into
three bit-planes, 2-hour steps by default), then compresses the planes with every combination of
codec and chunk length (an earlier version also tried storing each step as the exclusive-or of the
one before; it made zstd files about 10% larger and was dropped), and reports the compressed size, the size extrapolated to
every patch of the raster, and the time to read one month back. It uses the CPU only: no GPU, no
CUDA, no network. SPICE supplies the Sun and Earth positions.

Lighting a patch takes a few seconds on all cores. The planes are cached under WORK_DIR (about
215 MB per patch for 8 years at 2 hours) so a rerun only repeats the compression. The compressors
(zstandard, lz4, python-snappy) must be installed; a codec that is missing is skipped.
Importing this file starts nothing.
"""

from __future__ import annotations

import json
import logging
import sys
import time as clock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunarscout._numba_horizon import safe_haven_lightstore as lightstore  # noqa: E402
from lunarscout._numba_horizon.file_format import PATCH_SIZE, HorizonTileStore, read_patch_horizons  # noqa: E402
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession  # noqa: E402
from lunarscout._numba_horizon.safe_haven_windows import FloodGrid, spice_vector_source  # noqa: E402
from lunarscout._script_logging import reset_script_logging, setup_script_logging  # noqa: E402
from lunarscout.errors import LunarscoutError  # noqa: E402
from lunarscout.products import _load_dem  # noqa: E402

LOG = logging.getLogger("lunarscout.scripts.measure_light_store")

# Edit these paths before running. They match run_safe_haven_stage2.py.
DEM_PATH = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
HORIZONS_PATH = Path("/e/lunar_analyst_scenarios/70south/lighting/horizons")
WORK_DIR = Path("/tmp/lunarscout_light_store_measurement")      # planes cache, written files, results
OBSERVER_HEIGHT_M = 0.0

START_UTC = "2024-01-01T00:00:00"
STOP_UTC = "2032-01-01T00:00:00"                                # exclusive
STEP_MINUTES = 120

# (patch_row, patch_column, note). The raster is 237 x 237 patches; the pole is near (118, 118).
PATCHES = [
    (118, 118, "at the pole"),
    (112, 125, "near the pole"),
    (100, 140, "about 56 km out"),
    (118, 170, "about 130 km out"),
    (60, 60, "about 150 km out, diagonal"),
    (24, 202, "cell (12, 101): many haven-months"),
    (6, 234, "cell (3, 117): the highest scores"),
    (200, 40, "outer ring"),
]
CODECS = ["zlib-6", "snappy", "lz4", "lz4-9", "zstd-1", "zstd-3", "zstd-9", "zstd-19"]
CHUNK_DAYS = [1, 4, 8, 16]
THREADS = 12                  # compression runs in threads; the codecs release the GIL
TIMING_CHUNK_DAYS = 8         # chunk length of the timed single-thread read and write
RASTER_PATCHES = 237 * 237


def planes_for(patch, grid, first_step, sun, earth, dem, horizon_store, session):
    row, column, _note = patch
    cache = WORK_DIR / f"planes_{row}_{column}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r"), None
    started = clock.monotonic()
    horizons = read_patch_horizons(horizon_store, row * PATCH_SIZE, column * PATCH_SIZE, OBSERVER_HEIGHT_M)
    decode_s = clock.monotonic() - started
    if horizons is None:
        return None, None
    started = clock.monotonic()
    planes = lightstore.light_patch_planes(session, dem, horizons, row, column, sun, earth,
                                           lightstore.LightStoreSettings(step_minutes=STEP_MINUTES))
    light_s = clock.monotonic() - started
    np.save(cache, planes)
    LOG.info("patch (%d, %d): horizons decoded in %.2f s, %d steps lit in %.1f s", row, column, decode_s,
             planes.shape[1], light_s)
    return planes, {"decode_s": decode_s, "light_s": light_s}


def plane_statistics(planes) -> dict:
    """How much of each plane is set, and how often a pixel changes between steps."""
    steps = planes.shape[1]
    out = {}
    pixels = float(PATCH_SIZE * PATCH_SIZE)
    for index, name in enumerate(lightstore.PLANE_NAMES):
        ones = 0
        changes = 0
        for start in range(0, steps, 2048):
            block = np.asarray(planes[index, start:start + 2048])
            ones += int(np.unpackbits(block.view(np.uint8)).sum())
            flipped = block[1:] ^ block[:-1]
            changes += int(np.unpackbits(flipped.view(np.uint8)).sum())
            if start + 2048 < steps:                      # the join between blocks
                changes += int(np.unpackbits((np.asarray(planes[index, start + 2048]) ^ block[-1]).view(np.uint8)).sum())
        out[name] = {"fraction_set": ones / (steps * pixels), "changes_per_pixel": changes / pixels}
    return out


def compressed_size(planes, steps_per_chunk, codec) -> int:
    settings = lightstore.LightStoreSettings(
        step_minutes=STEP_MINUTES, steps_per_chunk=steps_per_chunk, codec=codec)
    total = 0
    for start in range(0, planes.shape[1], steps_per_chunk):
        total += len(lightstore.encode_chunk(np.asarray(planes[:, start:start + steps_per_chunk]), settings))
    return total


def timed_round_trip(planes, first_step, codec, steps_per_chunk):
    """Single-thread write time, whole-file read time and one-month read time and bytes."""
    settings = lightstore.LightStoreSettings(
        step_minutes=STEP_MINUTES, steps_per_chunk=steps_per_chunk, codec=codec)
    path = WORK_DIR / f"timing_{codec}.lit"
    started = clock.monotonic()
    size = lightstore.write_patch(path, np.asarray(planes), settings, patch_row=0, patch_column=0,
                                  first_step=first_step)
    write_s = clock.monotonic() - started
    steps_per_month = 30 * 24 * 60 // STEP_MINUTES
    with lightstore.LightStore(path, settings) as opened:
        middle = first_step + planes.shape[1] // 2
        started = clock.monotonic()
        month = opened.read_steps(middle, middle + steps_per_month)
        month_s = clock.monotonic() - started
        month_bytes = opened.stored_bytes(middle, middle + steps_per_month)
        assert np.array_equal(month, np.asarray(planes[:, middle - first_step:middle - first_step + steps_per_month]))
        started = clock.monotonic()
        everything = opened.read_steps(first_step, first_step + planes.shape[1])
        all_s = clock.monotonic() - started
        assert np.array_equal(everything, np.asarray(planes))
    path.unlink()
    return {"file_bytes": size, "write_s": write_s, "read_month_s": month_s, "read_month_bytes_stored": month_bytes,
            "read_all_s": all_s}


def _run() -> int:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    grid = FloodGrid(STEP_MINUTES)
    start_us = int(np.datetime64(START_UTC, "us").astype("int64"))
    stop_us = int(np.datetime64(STOP_UTC, "us").astype("int64"))
    first_step = grid.ceil_index(start_us)
    stop_step = grid.ceil_index(stop_us)
    steps = np.arange(first_step, stop_step, dtype=np.int64)
    LOG.info("%d steps of %d minutes, %s to %s", steps.size, STEP_MINUTES, START_UTC, STOP_UTC)

    vectors_file = WORK_DIR / "vectors.npz"
    if vectors_file.exists():
        loaded = np.load(vectors_file)
        sun, earth = loaded["sun"], loaded["earth"]
    else:
        started = clock.monotonic()
        stamps = grid.times(steps)
        sun, earth = spice_vector_source("SUN")(stamps), spice_vector_source("EARTH")(stamps)
        np.savez(vectors_file, sun=sun, earth=earth)
        LOG.info("SPICE vectors for %d steps in %.1f s", steps.size, clock.monotonic() - started)

    dem, _georef = _load_dem(DEM_PATH)
    horizon_store = HorizonTileStore(HORIZONS_PATH)
    session = LightmapCpuSession(time_batch_size=64)

    patches = []
    for patch in PATCHES:
        planes, timing = planes_for(patch, grid, first_step, sun, earth, dem, horizon_store, session)
        if planes is None:
            LOG.warning("patch (%d, %d) has no horizon tile; skipped", patch[0], patch[1])
            continue
        patches.append((patch, planes, timing))
    raw = lightstore.chunk_raw_size(steps.size)
    LOG.info("%d patches; %.1f MB raw each (%.2f TB for %d patches)", len(patches), raw / 1e6,
             raw * RASTER_PATCHES / 1e12, RASTER_PATCHES)

    results = {"steps": int(steps.size), "step_minutes": STEP_MINUTES, "raw_bytes_per_patch": raw,
               "patches": [], "sizes": {}, "timing": {}}
    for patch, planes, timing in patches:
        stats = plane_statistics(planes)
        results["patches"].append({"row": patch[0], "column": patch[1], "note": patch[2], "timing": timing,
                                   "planes": stats})
        LOG.info("patch (%d, %d) %s: %s", patch[0], patch[1], patch[2],
                 "; ".join(f"{name} set {value['fraction_set']:.3f}, {value['changes_per_pixel']:.0f} changes/pixel"
                           for name, value in stats.items()))

    configs = []
    for codec in CODECS:
        try:
            lightstore.codec_functions(codec)
        except LunarscoutError as error:
            LOG.warning("skipping %s: %s", codec, error)
            continue
        for days in CHUNK_DAYS:
            configs.append((codec, days))
    started = clock.monotonic()
    tasks = [(patch_index, config) for patch_index in range(len(patches)) for config in configs]

    def work(task):
        patch_index, (codec, days) = task
        spc = days * 24 * 60 // STEP_MINUTES
        return task, compressed_size(patches[patch_index][1], spc, codec)

    with ThreadPoolExecutor(THREADS) as pool:
        for done, (task, size) in enumerate(pool.map(work, tasks), 1):
            patch_index, (codec, days) = task
            results["sizes"].setdefault(f"{codec}|{days}d", {})[patch_index] = size
            if done % 50 == 0:
                LOG.info("compressed %d of %d (%.0f s)", done, len(tasks), clock.monotonic() - started)

    rows = []
    for key, by_patch in results["sizes"].items():
        sizes = np.asarray([by_patch[i] for i in range(len(patches))], dtype=float)
        rows.append((key, sizes.mean(), sizes.min(), sizes.max()))
    rows.sort(key=lambda row: row[1])
    LOG.info("compressed size per patch for %d steps (raw %.1f MB); all-raster = mean x %d", steps.size,
             raw / 1e6, RASTER_PATCHES)
    LOG.info("%-28s %10s %8s %10s %10s %12s", "codec|chunk", "mean MB", "ratio", "min MB", "max MB",
             "raster TB")
    for key, mean, low, high in rows:
        LOG.info("%-28s %10.2f %8.1f %10.2f %10.2f %12.2f", key, mean / 1e6, raw / mean, low / 1e6, high / 1e6,
                 mean * RASTER_PATCHES / 1e12)
    results["ranked"] = [{"config": key, "mean_bytes": mean, "min_bytes": low, "max_bytes": high,
                          "ratio": raw / mean, "raster_tb": mean * RASTER_PATCHES / 1e12} for key, mean, low, high in rows]

    for patch, planes, _timing in patches[:1]:
        LOG.info("single-thread timing on patch (%d, %d), %d-day chunks:", patch[0], patch[1], TIMING_CHUNK_DAYS)
        spc = TIMING_CHUNK_DAYS * 24 * 60 // STEP_MINUTES
        for codec in CODECS:
            try:
                lightstore.codec_functions(codec)
            except LunarscoutError:
                continue
            t = timed_round_trip(planes, first_step, codec, spc)
            results["timing"][codec] = t
            LOG.info("  %-9s: file %6.2f MB, write %5.2f s, read all %5.2f s, read 30 days %6.3f s (%.2f MB stored)",
                     codec, t["file_bytes"] / 1e6, t["write_s"], t["read_all_s"], t["read_month_s"],
                     t["read_month_bytes_stored"] / 1e6)

    (WORK_DIR / "results.json").write_text(json.dumps(results, indent=1, default=float))
    LOG.info("wrote %s", WORK_DIR / "results.json")
    return 0


def main() -> int:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    setup_script_logging(WORK_DIR / "measure_light_store.log", color="auto", level="INFO")
    try:
        return _run()
    finally:
        reset_script_logging()


if __name__ == "__main__":
    raise SystemExit(main())
