#!/usr/bin/env python3
"""Independently spot-check safe-haven raster values at one DEM pixel.

The signal calculation uses the production product frame with explicit
geometric Moon-ME vectors.  The duration calculation materializes the full
point history and does not reuse the production streaming reducer.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio

import lunarscout as ls
from lunarscout._numba_horizon.lightmap import (
    _pixel_horizon_margins_reference,
    _pixel_sunlight_fractions_reference,
)
from lunarscout.products import _load_dem


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _safe_haven_duration_for_month(
    fractions: np.ndarray,
    earth_elevations: np.ndarray,
    times: tuple[datetime, ...],
    *,
    month_start: datetime,
    time_step_hours: float,
    sunlight_threshold: float,
    earth_threshold_deg: float,
) -> float | None:
    """Return the longest complete low-Sun run touching a monthly outage."""
    if month_start.month == 12:
        month_stop = month_start.replace(
            year=month_start.year + 1, month=1, day=1
        )
    else:
        month_stop = month_start.replace(month=month_start.month + 1, day=1)

    in_month = np.fromiter(
        (month_start <= value < month_stop for value in times),
        dtype=np.bool_,
        count=len(times),
    )
    earth_below = earth_elevations < earth_threshold_deg
    monthly_earth_below = earth_below[in_month]
    if monthly_earth_below.size == 0:
        raise ValueError(f"no evaluation samples fall in {month_start:%Y-%m}")
    if not np.any(monthly_earth_below) or np.all(monthly_earth_below):
        return None

    sun_low = fractions < sunlight_threshold
    changes = np.diff(np.pad(sun_low.astype(np.int8), (1, 1)))
    run_starts = np.flatnonzero(changes == 1)
    run_stops = np.flatnonzero(changes == -1)
    best_samples = 0
    for run_start, run_stop in zip(run_starts, run_stops, strict=True):
        overlaps_monthly_outage = np.any(
            earth_below[run_start:run_stop] & in_month[run_start:run_stop]
        )
        if overlaps_monthly_outage:
            best_samples = max(best_samples, int(run_stop - run_start))
    return float(best_samples) * time_step_hours


def _resolve_raster(scenario: ls.Scenario, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else scenario.path(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="Path to scenario root")
    parser.add_argument(
        "--raster",
        default="analysis/safe-havens-test.tif",
        help="Existing safe-haven raster, absolute or scenario-relative",
    )
    parser.add_argument(
        "--pixel",
        type=int,
        nargs=2,
        default=(2000, 2000),
        metavar=("X", "Y"),
        help="DEM pixel coordinates (default: 2000 2000)",
    )
    parser.add_argument("--start", default="2027-09-01T00:00:00Z")
    parser.add_argument("--stop", default="2028-04-01T00:00:00Z")
    parser.add_argument("--step-hours", type=float, default=2.0)
    parser.add_argument("--earth-threshold", type=float, default=2.0)
    parser.add_argument("--sun-threshold", type=float, default=0.2)
    parser.add_argument("--tolerance-hours", type=float, default=0.1)
    args = parser.parse_args()

    scenario = ls.open_scenario(args.scenario)
    dem, georef = _load_dem(scenario.dem_path())
    pixel_x, pixel_y = args.pixel
    if not 0 <= pixel_x < dem.width or not 0 <= pixel_y < dem.height:
        print(
            f"Pixel ({pixel_x}, {pixel_y}) is outside the "
            f"{dem.width}x{dem.height} DEM",
            file=sys.stderr,
        )
        return 2

    horizon = scenario.horizon_for_pixel(pixel_x, pixel_y, 0)
    if horizon is None:
        print(f"No horizon at pixel ({pixel_x}, {pixel_y})", file=sys.stderr)
        return 2

    time_range = ls.times(args.start, args.stop, step_hours=args.step_hours)
    times = tuple(
        ls.iter_times(
            args.start,
            args.stop,
            timedelta(hours=args.step_hours),
        )
    )
    if len(times) < 2:
        print("Need at least two time samples", file=sys.stderr)
        return 2

    print("Generating geometric Moon-ME vectors with SPICE...")
    sun_vectors = ls.body_vectors_moon_me("sun", time_range)
    earth_vectors = ls.body_vectors_moon_me("earth", time_range)
    fractions = _pixel_sunlight_fractions_reference(
        dem,
        horizon,
        sun_vectors,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
    )
    earth_elevations = _pixel_horizon_margins_reference(
        dem,
        horizon,
        earth_vectors,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
    )

    longitude, latitude = georef.pixel_to_lonlat(pixel_x, pixel_y)
    print(
        f"Pixel ({pixel_x}, {pixel_y}); lon/lat "
        f"({longitude:.6f}, {latitude:.6f})"
    )
    print(
        f"Evaluation: {times[0].isoformat()} through {times[-1].isoformat()}, "
        f"{len(times)} samples at {args.step_hours:g} h"
    )
    print(f"Sunlight fraction range: {fractions.min():.6f} to {fractions.max():.6f}")
    print(
        "Earth terrain-relative elevation range: "
        f"{earth_elevations.min():.6f} to {earth_elevations.max():.6f} deg"
    )

    raster_path = _resolve_raster(scenario, args.raster)
    failures = 0
    with rasterio.open(raster_path) as dataset:
        if (dataset.width, dataset.height) != (dem.width, dem.height):
            print("Raster and DEM dimensions differ", file=sys.stderr)
            return 2
        values = dataset.read(window=((pixel_y, pixel_y + 1), (pixel_x, pixel_x + 1)))[
            :, 0, 0
        ]
        valid = bool(
            dataset.dataset_mask(
                window=((pixel_y, pixel_y + 1), (pixel_x, pixel_x + 1))
            )[0, 0]
        )
        if not valid:
            print("The selected raster pixel is invalid", file=sys.stderr)
            return 2

        print(f"Comparing {raster_path}:")
        for band_index, actual in enumerate(values, start=1):
            timestamp = dataset.tags(band_index).get("TIMESTAMP_UTC")
            if timestamp is None:
                print(f"  Band {band_index}: missing TIMESTAMP_UTC", file=sys.stderr)
                failures += 1
                continue
            month_start = _parse_utc(timestamp)
            expected = _safe_haven_duration_for_month(
                fractions,
                earth_elevations,
                times,
                month_start=month_start,
                time_step_hours=args.step_hours,
                sunlight_threshold=args.sun_threshold,
                earth_threshold_deg=args.earth_threshold,
            )
            agrees = (
                (expected is None and np.isnan(actual))
                or (
                    expected is not None
                    and np.isfinite(actual)
                    and abs(float(actual) - expected) <= args.tolerance_hours
                )
            )
            expected_text = "NODATA" if expected is None else f"{expected:.3f} h"
            actual_text = "NODATA" if np.isnan(actual) else f"{actual:.3f} h"
            print(
                f"  {month_start:%Y-%m}: expected {expected_text}, "
                f"raster {actual_text} -- {'PASS' if agrees else 'FAIL'}"
            )
            failures += int(not agrees)

    if failures:
        print(f"Spot check failed for {failures} band(s).", file=sys.stderr)
        return 1
    print("Spot check passed for every raster band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
