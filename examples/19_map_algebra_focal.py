"""Focal cleanup and distance fields with map algebra.

Inputs: deterministic DEM under --workspace.
Outputs: focal/smoothed_slope.tif, focal/opened_mask.tif, focal/distance_to_steep.tif.
Resources: small eager rasters; demonstrates focal smoothing, morphology
opening, and distance field calculation.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import ensure_synthetic_scenario, example_parser

ma = ls.map_algebra


def main() -> None:
    args = example_parser("Map-algebra focal cleanup and distance fields.").parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("The DEM must be georeferenced.")

    slope_bare, slope_georef = ls.slope(dem, georef, output_nodata=-9999.0)
    slope = ma.from_existing(slope_bare, slope_georef, units="degrees", name="slope")

    smoothed = ma.focal_mean(slope, size=3, edge="nearest")
    out = scenario.output_path("analysis/focal/smoothed_slope.tif")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(ma.write(out, smoothed.expression(), dtype="float32", invalid_value=-9999.0))

    steep = slope >= 8.0
    opened = ma.opening(steep, size=3)
    out = scenario.output_path("analysis/focal/opened_mask.tif")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(ma.write(out, opened.expression(), dtype="uint8", invalid_value=0))

    dist = ma.distance_to(opened, metric="euclidean", units="pixels")
    out = scenario.output_path("analysis/focal/distance_to_steep.tif")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(ma.write(out, dist.expression(), dtype="float32", invalid_value=-9999.0))

    stats = ma.statistics(dist)
    print(f"Distance stats: min={stats.min_val:.1f} max={stats.max_val:.1f} mean={stats.mean:.1f} px")


if __name__ == "__main__":
    main()
