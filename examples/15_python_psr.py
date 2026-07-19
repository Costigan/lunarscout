#!/usr/bin/env python3
"""Generate a public Lunarscout PSR GeoTIFF for the Mons Mouton scenario.

The defaults read /e/lunar_analyst_scenarios/mons-mouton and write
examples/mons-mouton-psr.tif.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
import time

import lunarscout as ls
import numpy as np
import rasterio


SCENARIO_PATH = Path("/e/lunar_analyst_scenarios/mons-mouton")
OUTPUT_PATH = Path(__file__).resolve().parent / "mons-mouton-psr.tif"


def make_progress_reporter() -> Callable[[float], None]:
    """Return a throttled console callback accepting completed fraction."""
    started_at = datetime.now().astimezone()
    started_clock = time.monotonic()
    initial_fraction: float | None = None
    last_percent_bucket = -1

    def report(completed_fraction: float) -> None:
        nonlocal initial_fraction, last_percent_bucket
        fraction = min(1.0, max(0.0, float(completed_fraction)))
        if initial_fraction is None:
            initial_fraction = fraction
        percent_bucket = int(fraction * 100.0)
        if fraction < 1.0 and percent_bucket <= last_percent_bucket:
            return
        last_percent_bucket = percent_bucket

        now = datetime.now().astimezone()
        elapsed_seconds = time.monotonic() - started_clock
        completed_this_run = fraction - initial_fraction
        if completed_this_run > 0.0 and elapsed_seconds > 0.0:
            fraction_per_second = completed_this_run / elapsed_seconds
            remaining_seconds = (1.0 - fraction) / fraction_per_second
            completion_at = now + timedelta(seconds=remaining_seconds)
            estimate = (
                f"remaining={remaining_seconds / 60.0:.1f} min, "
                f"completion={completion_at:%Y-%m-%d %H:%M:%S %Z}"
            )
        else:
            estimate = "remaining=estimating, completion=estimating"
        print(
            f"PSR {fraction * 100.0:6.2f}% | "
            f"elapsed={elapsed_seconds / 60.0:.1f} min, {estimate}",
            flush=True,
        )

    print(f"PSR started={started_at:%Y-%m-%d %H:%M:%S %Z}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=SCENARIO_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
        help="Use CUDA for this full scenario unless CPU execution is intentional.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scenario = args.scenario.expanduser().resolve()
    dem_path = scenario / "dem.tif"
    horizon_path = scenario / "horizons"
    missing = [path for path in (dem_path, horizon_path) if not path.exists()]
    if missing:
        raise SystemExit("Missing scenario input: " + ", ".join(map(str, missing)))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    times = ls.times(
        "1970-01-01T00:00:00Z",
        "2044-01-01T00:00:00Z",
        step_hours=6,
    )

    print(f"Writing PSR product: {output}", flush=True)
    started = time.perf_counter()
    progress = make_progress_reporter()
    result = ls.generate_psr(
        dem_path,
        horizon_path,
        output,
        times=times,
        backend=args.backend,
        overwrite=args.overwrite,
        progress_callback=progress,
    )
    elapsed = time.perf_counter() - started

    with rasterio.open(result) as dataset:
        values = dataset.read(1)
        valid = dataset.dataset_mask() != 0
        psr_pixels = int(np.count_nonzero(valid & (values == 255)))
        illuminated_pixels = int(np.count_nonzero(valid & (values == 0)))
        invalid_pixels = int(np.count_nonzero(~valid))
    print(
        f"Complete in {elapsed:.2f} s: {result}\n"
        f"PSR pixels={psr_pixels:,}, illuminated pixels={illuminated_pixels:,}, "
        f"invalid pixels={invalid_pixels:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
