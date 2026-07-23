"""Site suitability with map algebra: combine slope and sunlight with explicit validity.

Inputs: deterministic DEM and file-backed illumination series under --workspace.
Outputs: screening/candidate.tif, screening/candidate_score.tif.
Resources: small eager rasters; demonstrates local operations, comparisons,
Boolean logic, where, and expressive validity handling.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import ensure_synthetic_scenario, ensure_synthetic_series, example_parser

ma = ls.map_algebra


def main() -> None:
    args = example_parser("Map-algebra site screening example.").parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    series = ensure_synthetic_series(args.workspace)
    dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("The DEM must be georeferenced.")

    slope_bare, slope_georef = ls.slope(dem, georef, output_nodata=-9999.0)
    mean_sun_bare, mean_sun_georef = ls.temporal_mean(series)
    ls.require_same_grid(slope_georef, mean_sun_georef)

    slope = ma.from_existing(slope_bare, slope_georef, units="degrees", name="slope")
    sun = ma.from_existing(mean_sun_bare, mean_sun_georef.with_nodata(None), units="fraction", name="mean_sun")

    candidate = (slope <= 8.0) & (sun >= 0.60)

    # Combine slope and sun fraction into a weighted score.
    slope_score = (1.0 - (slope / 8.0)).with_units("score")
    sun_score = sun.with_units("score")
    score = ma.where(candidate, 0.4 * sun_score + 0.6 * slope_score, ma.invalid)

    for name, raster in (
        ("candidate.tif", candidate),
        ("candidate_score.tif", score),
    ):
        out = scenario.output_path(f"analysis/screening/{name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        print(ma.write(out, raster.expression(), dtype="float32", invalid_value=-9999.0))


if __name__ == "__main__":
    main()
