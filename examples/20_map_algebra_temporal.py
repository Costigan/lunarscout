"""Temporal map algebra: reduce a time series and compose with spatial constraints.

Inputs: deterministic DEM and file-backed illumination series under --workspace.
Outputs: temporal/mean_sun.tif, temporal/temporal_candidate.tif.
Resources: file-backed temporal source read via eager compute; demonstrates
temporal_source, temporal_mean, compose with spatial algebra, and write.
"""

from __future__ import annotations

import lunarscout as ls

from _example_support import ensure_synthetic_scenario, ensure_synthetic_series, example_parser

ma = ls.map_algebra


def main() -> None:
    args = example_parser("Map-algebra temporal example.").parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    series = ensure_synthetic_series(args.workspace)
    dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("The DEM must be georeferenced.")

    slope_bare, slope_georef = ls.slope(dem, georef, output_nodata=-9999.0)
    ls.require_same_grid(slope_georef, series.georef)

    slope = ma.from_existing(slope_bare, slope_georef, units="degrees", name="slope")

    sun_expr = ma.temporal_source(series)
    mean_sun = ma.temporal_mean(sun_expr)

    out = scenario.output_path("analysis/temporal/mean_sun.tif")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Temporal reductions are not yet spatial-window planner nodes. Make their
    # whole-raster materialization explicit before using the bounded writer.
    mean_raster = ma.compute(mean_sun)
    print(ma.write(out, mean_raster.expression(), dtype="float32", invalid_value=-9999.0))

    candidate = (mean_sun >= 0.40) & (slope.expression() <= 8.0)
    out = scenario.output_path("analysis/temporal/temporal_candidate.tif")
    out.parent.mkdir(parents=True, exist_ok=True)
    candidate_raster = ma.compute(candidate)
    print(ma.write(out, candidate_raster.expression(), dtype="uint8", invalid_value=0))

    series.close()


if __name__ == "__main__":
    main()
