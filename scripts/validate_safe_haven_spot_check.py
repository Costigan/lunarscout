#!/usr/bin/env python3
"""Spot-check safe-haven raster values against point-based time series.

Pick a pixel within a scenario, extract its horizon, compute sunlight
fractions and Earth elevation over time, manually derive safe-haven
durations, and compare against the raster product.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio

import lunarscout as ls


def _safe_haven_duration_for_point(
    fractions: np.ndarray,
    earth_elevations: np.ndarray,
    *,
    time_step_hours: float,
    sunlight_threshold: float,
    earth_threshold_deg: float,
) -> float | None:
    """Compute the longest low-sun run overlapping any Earth outage.

    Returns NaN if Earth never crosses the threshold (no outage defined).
    """
    earth_below = earth_elevations < earth_threshold_deg
    sun_low = fractions < sunlight_threshold

    # Detect Earth outages
    changes = np.diff(np.pad(earth_below.astype(np.int8), (1, 1)))
    starts = (changes == 1).nonzero()[0]
    stops = (changes == -1).nonzero()[0]

    if len(starts) == 0:
        return None  # never below → NODATA

    # If always below, NODATA
    if len(starts) == 1 and starts[0] == 0 and stops[0] == len(earth_below):
        return None

    best_run = 0
    for start, stop in zip(starts, stops):
        # Find longest low-sun run overlapping this outage
        max_run = 0
        current = 0
        for t in range(len(sun_low)):
            if sun_low[t]:
                current += 1
            else:
                current = 0
            # Check overlap: run touches [start, stop)
            # A run ending at t starts at t - current + 1
            run_start = t - current + 1
            run_stop = t + 1
            if current > 0 and run_stop > start and run_start < stop:
                max_run = max(max_run, current)
        best_run = max(best_run, max_run)

    return float(best_run) * time_step_hours


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="Path to scenario root")
    parser.add_argument("--pixel", type=int, nargs=2, default=None,
                        help="DEM pixel coordinates (x y), default uses center")
    parser.add_argument("--start", default="2027-01-01T00:00:00Z",
                        help="Evaluation start (UTC ISO-8601)")
    parser.add_argument("--stop", default="2027-04-01T00:00:00Z",
                        help="Evaluation stop (UTC ISO-8601)")
    parser.add_argument("--step-hours", type=float, default=1.0,
                        help="Time step in hours")
    parser.add_argument("--earth-threshold", type=float, default=2.0,
                        help="Earth elevation threshold in degrees")
    parser.add_argument("--sun-threshold", type=float, default=0.2,
                        help="Sunlight fraction threshold")
    args = parser.parse_args()

    scenario = ls.open_scenario(args.scenario)
    dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        print("DEM is not georeferenced", file=sys.stderr)
        return 1

    if args.pixel:
        pixel_x, pixel_y = int(args.pixel[0]), int(args.pixel[1])
    else:
        pixel_x, pixel_y = dem.shape[1] // 2, dem.shape[0] // 2

    horizon = scenario.horizon_for_pixel(pixel_x, pixel_y, 0)
    if horizon is None:
        print(f"No horizon at pixel ({pixel_x}, {pixel_y})", file=sys.stderr)
        return 1

    times = ls.times(args.start, args.stop, step_hours=args.step_hours)
    time_list = list(ls.iter_times(args.start, args.stop,
                                    timedelta(hours=args.step_hours)))
    if len(time_list) < 2:
        print("Need at least 2 time samples", file=sys.stderr)
        return 1

    point = ls.LonLat(*georef.pixel_to_lonlat(pixel_x, pixel_y))

    print(f"Pixel ({pixel_x}, {pixel_y}) → lon/lat ({point.longitude:.4f}, {point.latitude:.4f})")
    print(f"Time range: {time_list[0]} to {time_list[-1]}, {len(time_list)} samples, "
          f"{args.step_hours}h step")

    fractions = ls.sunlight_fraction(point, times, horizon)
    earth_angles = ls.body_azimuth_elevation_over_horizon(point, "earth", times, horizon)
    earth_elevations = earth_angles[:, 1]

    print(f"Sunlight fraction: [{fractions.min():.4f}, {fractions.max():.4f}]")
    print(f"Earth over horizon: [{earth_elevations.min():.2f}, {earth_elevations.max():.2f}] deg")

    # Build calendar months
    months: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for i, t in enumerate(time_list):
        key = (t.year, t.month)
        months.setdefault(key, []).append((i, fractions[i], earth_elevations[i]))

    print(f"\nCalendar months: {len(months)}")
    for (year, month), samples in sorted(months.items()):
        idxs = [s[0] for s in samples]
        fracs = np.array([s[1] for s in samples])
        earths = np.array([s[2] for s in samples])

        duration = _safe_haven_duration_for_point(
            fracs, earths,
            time_step_hours=args.step_hours,
            sunlight_threshold=args.sun_threshold,
            earth_threshold_deg=args.earth_threshold,
        )

        below = np.sum(earths < args.earth_threshold)
        low = np.sum(fracs < args.sun_threshold)

        status = "NODATA" if duration is None else f"{duration:.1f} h"
        print(f"  {year}-{month:02d}: {status}  "
              f"(below {args.earth_threshold}°: {below}/{len(earths)}, "
              f"low-sun: {low}/{len(fracs)})")

    # Generate safe-haven raster for same time range and compare
    print("\n--- Generating safe-haven raster for comparison ---")
    safe_path = scenario.safe_havens(
        "analysis/spot-check-safe-havens.tif",
        times=times,
        earth_elevation_threshold_deg=args.earth_threshold,
        sunlight_fraction_threshold=args.sun_threshold,
        backend="cpu",
        verbose=True,
    )

    with rasterio.open(safe_path) as ds:
        print(f"\nRaster bands: {ds.count}")
        for band_idx in range(1, ds.count + 1):
            value = ds.read(band_idx)[pixel_y, pixel_x]
            tags = ds.tags(band_idx)
            ts = tags.get("TIMESTAMP_UTC", "?")
            if np.isnan(value):
                print(f"  Band {band_idx} ({ts}): NODATA")
            else:
                print(f"  Band {band_idx} ({ts}): {value:.1f} h")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
