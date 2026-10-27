#!/usr/bin/env python3
"""Edit the constants below, then run this script with the project environment.

    .venv/bin/python scripts/summarize_lander_continuous_sun.py

Summarizes the lander sun-duration product written by scripts/run_lander_continuous_sun.py (one uint16 band
per month: the longest continuous Sun from a landing in that month, in 2-hour steps; 65535 is
nodata). It prints, per month, the share of valid pixels with any Sun and the percentiles of the
longest period in days, then the same for the best month of each pixel and the share of pixels that reach
each of a list of durations, and finally the pixels with the longest periods. A per-month CSV is written
next to the product. It reads the whole product once (about 45 GB), strip by strip, with READ_THREADS
threads, and uses the CPU only. The last months read low by design (a period under way at the end of
the store is credited with the length seen, a lower bound). Importing it starts nothing.
"""

from __future__ import annotations

import csv
import logging
import sys
import time as clock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Edit these before running.
PRODUCT_PATH = Path("/e/lunar_analyst_scenarios/70south/lander_sun_longest_period.tif")
CSV_PATH = None                  # None writes <product>.summary.csv next to the product
HOURS_PER_STEP = 2.0             # the light store's step
STRIP_ROWS = 128                 # rows read at once (one block row); memory is about 0.75 GB per strip
READ_THREADS = 8
PERCENTILES = (50, 90, 99)       # of the valid pixels that have a period, per month
DURATION_DAYS = (1, 3, 7, 14, 28, 60, 120)
TOP_PIXELS = 10
LOG = logging.getLogger("lunarscout.scripts.summarize_lander_continuous_sun")


def _percentile(histogram: np.ndarray, fraction: float) -> int:
    """Smallest value whose cumulative share reaches fraction (histogram over the values 0..N-1)."""
    cumulative = np.cumsum(histogram)
    if cumulative[-1] == 0:
        return 0
    return int(np.searchsorted(cumulative, fraction * cumulative[-1]))


def summarize(path: Path) -> dict:
    with rasterio.open(path) as src:
        bands, height, width = src.count, src.height, src.width
        nodata = int(src.nodata)
        months = [t[:7] for t in __import__("json").loads(src.tags()["LUNARSCOUT_TIMESTAMPS_UTC"])]
        transform = src.transform
    hist = np.zeros((bands, nodata + 1), dtype=np.int64)       # per month, values 0..nodata-1 (last bin unused)
    best_hist = np.zeros(nodata + 1, dtype=np.int64)           # per pixel, the best month
    valid_pixels = 0
    top: list[tuple[int, int, int, int]] = []                  # (steps, row, column, month index)
    started = clock.monotonic()

    def read_band(arguments):
        band, window = arguments
        with rasterio.open(path) as src:
            return src.read(band + 1, window=window)

    with ThreadPoolExecutor(READ_THREADS) as pool:
        for row in range(0, height, STRIP_ROWS):
            window = Window(0, row, width, min(STRIP_ROWS, height - row))
            cube = np.stack(list(pool.map(read_band, ((b, window) for b in range(bands)))))
            valid = cube[0] != nodata
            valid_pixels += int(valid.sum())
            for b in range(bands):
                hist[b] += np.bincount(cube[b][valid[...]].ravel(), minlength=nodata + 1)[: nodata + 1]
            cube = np.where(cube == nodata, 0, cube)
            best = cube.max(axis=0)
            best_hist += np.bincount(best[valid].ravel(), minlength=nodata + 1)[: nodata + 1]
            where = np.argmax(cube, axis=0)
            flat = np.argpartition(best.ravel(), -TOP_PIXELS)[-TOP_PIXELS:]
            for index in flat:
                r, c = divmod(int(index), width)
                top.append((int(best[r, c]), row + r, c, int(where[r, c])))
            top = sorted(top, reverse=True)[:TOP_PIXELS]
            done = row // STRIP_ROWS + 1
            total = -(-height // STRIP_ROWS)
            if done % 20 == 0 or done == total:
                elapsed = clock.monotonic() - started
                LOG.info("%d/%d strips, %.1f min elapsed, about %.1f min left", done, total, elapsed / 60,
                         elapsed / done * (total - done) / 60)
    return {"months": months, "hist": hist, "best_hist": best_hist,
            "valid_pixels": valid_pixels, "top": top, "transform": transform, "width": width,
            "height": height, "nodata": nodata}


def _days(steps: int) -> float:
    return steps * HOURS_PER_STEP / 24.0


def report(result: dict, csv_path: Path) -> None:
    hist, months, valid = result["hist"], result["months"], result["valid_pixels"]
    print(f"\nProduct: {result['width']} x {result['height']} px, {len(months)} monthly bands, "
          f"{valid:,} valid pixels ({valid / (result['width'] * result['height']):.1%} of the raster)")
    head = "  ".join(f"p{p}[d]".rjust(8) for p in PERCENTILES)
    print(f"\n{'month':8s}{'with Sun period':>17s}  {head}{'max[d]':>9s}")
    rows = []
    for b, month in enumerate(months):
        counts = hist[b][: result["nodata"]]
        positive = int(counts[1:].sum())
        pct = [_percentile(np.concatenate(([0], counts[1:])), p / 100) for p in PERCENTILES]
        maximum = int(np.flatnonzero(counts)[-1]) if counts.any() else 0
        rows.append([month, positive, positive / valid, *[_days(p) for p in pct], _days(maximum)])
        print(f"{month:8s}{positive / valid:>16.1%}  " + "  ".join(f"{_days(p):8.1f}" for p in pct)
              + f"{_days(maximum):9.1f}")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "pixels_with_period", "share_of_valid",
                         *[f"p{p}_days" for p in PERCENTILES], "max_days"])
        writer.writerows(rows)

    best = result["best_hist"][: result["nodata"]]
    cumulative_from_top = np.cumsum(best[::-1])[::-1]
    print("\nBest month of each pixel: share of valid pixels whose longest period reaches")
    for days in DURATION_DAYS:
        steps = int(np.ceil(days * 24 / HOURS_PER_STEP))
        count = int(cumulative_from_top[steps]) if steps < len(cumulative_from_top) else 0
        print(f"  at least {days:4d} d: {count:>12,}  ({count / valid:.2%})")

    print(f"\nPixels with the longest period (row, column, x, y in the raster CRS, month):")
    for steps, r, c, month in result["top"]:
        x, y = result["transform"] * (c + 0.5, r + 0.5)
        print(f"  {_days(steps):7.1f} d  row {r:5d}  col {c:5d}  x {x:12.1f}  y {y:12.1f}  {months[month]}")
    print(f"\nPer-month table written to: {csv_path}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    csv_path = CSV_PATH if CSV_PATH is not None else PRODUCT_PATH.with_name(PRODUCT_PATH.name + ".summary.csv")
    report(summarize(PRODUCT_PATH), csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
