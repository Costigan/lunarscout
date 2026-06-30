"""Generate slope, aspect, and hillshade products from a DEM.

Inputs: deterministic synthetic scenario created under --workspace.
Outputs: three GeoTIFFs under analysis/terrain.
Resources: small GDAL/NumPy calculation; native runtime is not required.
"""

from __future__ import annotations

import lunarscout as ls

from _example_support import ensure_synthetic_scenario, example_parser


def main() -> None:
    args = example_parser(__doc__).parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("The DEM must be georeferenced.")

    products = {
        "slope_deg.tif": ls.slope(dem, georef, output_nodata=-9999.0),
        "aspect_deg.tif": ls.aspect(dem, georef, output_nodata=-9999.0),
        "hillshade.tif": ls.hillshade(dem, georef, output_nodata=0),
    }
    for filename, (values, product_georef) in products.items():
        path = ls.write_geotiff(
            scenario.output_path(f"analysis/terrain/{filename}"),
            values,
            product_georef,
            overwrite=True,
        )
        print(path)


if __name__ == "__main__":
    main()
