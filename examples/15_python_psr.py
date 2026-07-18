#!/usr/bin/env python3
"""Generate a Python/Numba PSR GeoTIFF for the Mons Mouton scenario.

The defaults read /e/lunar_analyst_scenarios/mons-mouton and write
examples/mons-mouton-psr.tif. This example uses the private prototype API until
the validated product pipeline is promoted to Lunarscout's public facade.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
import time

import numpy as np
import rasterio

from lunarscout._numba_horizon.file_format import HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.product_vectors import generate_moon_me_vectors
from lunarscout._numba_horizon.psr_pipeline import run_psr_product
from lunarscout.georeference import GeoReference
from lunarscout.spice_geometry import iter_times


SCENARIO_PATH = Path("/e/lunar_analyst_scenarios/mons-mouton")
OUTPUT_PATH = Path(__file__).resolve().parent / "mons-mouton-psr.tif"


def load_dem(path: Path) -> tuple[DemGrid, GeoReference]:
    """Load the DEM calculation grid and its public GeoReference metadata."""
    import lunarscout as ls

    elevation, georef = ls.read_geotiff(path)
    if georef is None:
        raise RuntimeError(f"DEM has no georeferencing: {path}")
    with rasterio.open(path) as dataset:
        crs = dataset.crs.to_dict()
    projection = ProjectionParameters(
        radius_m=float(crs["R"]),
        latitude_origin_rad=float(np.deg2rad(crs["lat_0"])),
        longitude_origin_rad=float(np.deg2rad(crs["lon_0"])),
        scale=float(crs.get("k", crs.get("k_0", 1.0))),
        false_easting_m=float(crs.get("x_0", 0.0)),
        false_northing_m=float(crs.get("y_0", 0.0)),
    )
    dem = DemGrid(
        np.ascontiguousarray(elevation, dtype=np.float32),
        np.asarray(georef.affine_transform, dtype=np.float64),
        projection,
    )
    return dem, georef


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
    print(f"Loading DEM: {dem_path}", flush=True)
    dem, georef = load_dem(dem_path)

    times = tuple(
        iter_times(
            "1970-01-01T00:00:00Z",
            "2044-01-01T00:00:00Z",
            timedelta(hours=6),
        )
    )
    print(f"Generating {len(times):,} exact Moon-ME Sun vectors...", flush=True)
    started = time.perf_counter()
    vectors = generate_moon_me_vectors("sun", times)
    print(f"Sun vectors ready in {time.perf_counter() - started:.2f} s", flush=True)

    print(f"Writing PSR product: {output}", flush=True)
    started = time.perf_counter()
    progress = make_progress_reporter()
    result = run_psr_product(
        dem=dem,
        georef=georef,
        horizon_store=HorizonTileStore(horizon_path),
        output_path=output,
        sun_vectors_m=vectors.vectors_m,
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
