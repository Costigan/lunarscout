#!/usr/bin/env python3
"""How much does the terrain interpolation model change horizons? (CPU only, no GPU.)

Traces the same real-terrain rays as s6 (the s2 sample: 12 pixels x 71 azimuths, and the s5
sun-relevant sample: 40 pixels x 36 azimuths, reference horizon in [-2, 5] deg) with the s2 float64
brute force, three ways:
  bilinear, 1/4 pixel steps    the reference used throughout the review
  bilinear, 1/16 pixel steps   checks that reference has converged
  cubic B-spline, 1/4 pixel    interpolating, C2 (curvature-minimizing) surface through the same nodes
and prints the differences overall and by distance to the limiting terrain point.

    .venv/bin/python review_scripts/s8_interpolation_sensitivity.py --dem /tmp/lunarscout-phase1-lola-512.tif
Used for docs/numba-horizon-terrain-interpolation.md (2026-10-09). About 3 minutes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import s2_horizon_vs_bruteforce as s2  # noqa: E402
import s6_cull_step_isolation as s6  # noqa: E402
from lunarscout.products import _load_dem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dem", default="/tmp/lunarscout-phase1-lola-512.tif")
    args = parser.parse_args()
    real = _load_dem(Path(args.dem))[0]
    sets = {label: (rays, limit) for label, dataset, rays, limit in s6.samples(s2, real, real)
            if dataset == "real" and label in ("s2 real", "sun")}
    bilinear = s2.bilinear_vec
    coefficients = ndimage.spline_filter(real.elevation_m.astype(np.float64), order=3, mode="nearest")

    def cubic(elevation, column, row):
        height, width = elevation.shape
        inside = (column >= 0) & (column <= width - 1) & (row >= 0) & (row <= height - 1)
        values = ndimage.map_coordinates(coefficients, [np.where(inside, row, 0), np.where(inside, column, 0)],
                                         order=3, prefilter=False, mode="nearest")
        values[~inside] = np.nan
        return values

    def trace(sampler, step_fraction):
        s2.bilinear_vec = sampler
        try:
            return {label: np.array([s2.brute_horizon_deg([real], c, r, a * s2.AZ_STEP_DEG, 0.0, limit, step_fraction)
                                     for c, r, a in rays])
                    for label, (rays, limit) in sets.items()}
        finally:
            s2.bilinear_vec = bilinear

    base, fine, smooth = trace(bilinear, 0.25), trace(bilinear, 0.0625), trace(cubic, 0.25)

    def stats(d):
        d = d[np.isfinite(d)]
        a = np.abs(d)
        return (f"n={d.size:4d} mean sgn {d.mean():+.4f} mean|d| {a.mean():.4f} p95 {np.percentile(a, 95):.4f} "
                f"max {a.max():.4f} >0.005 {100 * (a > 0.005).mean():5.1f}% >0.05 {100 * (a > 0.05).mean():4.1f}%")

    for label in sets:
        degrees, distance = base[label][:, 0], base[label][:, 1]
        keep = np.isfinite(degrees)
        if label == "sun":
            keep &= (degrees >= -2.0) & (degrees <= 5.0)
        convergence = fine[label][:, 0] - degrees
        model = smooth[label][:, 0] - degrees
        print(f"\n{label}{' (bilinear reference in [-2, 5] deg)' if label == 'sun' else ''}")
        print("  bilinear 1/16 px - 1/4 px :", stats(convergence[keep]))
        print("  cubic - bilinear          :", stats(model[keep]))
        print("  cubic - bilinear by distance to the bilinear limiting point:")
        for low, high in ((0, 20), (20, 100), (100, 500), (500, 2000), (2000, 1e9)):
            m = keep & np.isfinite(model) & (distance >= low) & (distance < high)
            if m.any():
                a = np.abs(model[m])
                print(f"    {low:>5.0f}-{high:<9.0f} m: n={int(m.sum()):4d} mean sgn {model[m].mean():+.4f} "
                      f"mean|d| {a.mean():.4f} p95 {np.percentile(a, 95):.4f} max {a.max():.4f} "
                      f">0.05 {100 * (a > 0.05).mean():5.1f}%")


if __name__ == "__main__":
    main()
