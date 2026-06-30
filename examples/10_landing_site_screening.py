"""Combine terrain, illumination, and region constraints for site screening.

Inputs: deterministic DEM and file-backed illumination series under --workspace.
Outputs: candidate mask and candidate borders under analysis/screening.
Resources: small eager terrain calculation plus streaming temporal mean.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import ensure_synthetic_scenario, ensure_synthetic_series, example_parser


def main() -> None:
    args = example_parser(__doc__).parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    series = ensure_synthetic_series(args.workspace)
    dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("The DEM must be georeferenced.")
    ls.require_same_grid(georef, series.georef)

    slope, slope_georef = ls.slope(dem, georef, output_nodata=-9999.0)
    mean_sun, mean_sun_georef = ls.temporal_mean(series)
    ls.require_same_grid(slope_georef, mean_sun_georef)
    valid = slope != slope_georef.nodata
    combined = np.ma.array(
        valid & (slope <= 8.0) & (mean_sun >= 0.60),
        mask=~valid,
    )
    candidates, candidate_georef = ls.filter_regions_by_size(
        combined,
        slope_georef,
        threshold=80,
        comparator=">=",
    )
    borders, border_georef = ls.find_borders(candidates, candidate_georef)

    for filename, values, output_georef in (
        ("candidate_sites.tif", candidates, candidate_georef),
        ("candidate_borders.tif", borders, border_georef),
    ):
        print(
            ls.write_geotiff(
                scenario.output_path(f"analysis/screening/{filename}"),
                np.ma.filled(values, False).astype(np.uint8),
                output_georef.with_nodata(0),
                overwrite=True,
            )
        )


if __name__ == "__main__":
    main()
