"""Read/write GeoTIFFs and convert pixel, projected, and lunar coordinates.

Inputs: none; creates a deterministic 64 x 64 synthetic scenario.
Outputs: analysis/dem_copy.tif under --workspace.
Resources: negligible; a GPU is not required.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import ensure_synthetic_scenario, example_parser


def main() -> None:
    args = example_parser(__doc__).parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("The DEM must contain projection and affine metadata.")

    easting, northing = georef.pixel_to_projected(20, 16)
    longitude, latitude = georef.pixel_to_lonlat(20, 16)
    columns = np.asarray([0, 20, 63])
    rows = np.asarray([0, 16, 63])
    array_easting, array_northing = georef.pixel_to_projected(columns, rows)

    output = ls.write_geotiff(
        scenario.output_path("analysis/dem_copy.tif"),
        dem,
        georef,
        overwrite=True,
    )
    print(f"dtype={dem.dtype}, shape={dem.shape}, nodata={georef.nodata}")
    print(f"pixel (20,16) -> ({easting:.3f}, {northing:.3f}) metres")
    print(f"pixel (20,16) -> ({longitude:.8f}, {latitude:.8f}) degrees")
    print(f"array projected coordinates: {array_easting}, {array_northing}")
    print(output)


if __name__ == "__main__":
    main()
