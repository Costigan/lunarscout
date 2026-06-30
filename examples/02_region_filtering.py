"""Label, size-filter, clean, and outline disconnected candidate regions.

Inputs: deterministic synthetic scenario created under --workspace.
Outputs: labels, sizes, filtered mask, and borders under analysis/regions.
Resources: small NumPy/SciPy calculation; native runtime is not required.
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
        raise RuntimeError("The DEM must be georeferenced.")
    slope, slope_georef = ls.slope(dem, georef, output_nodata=-9999.0)
    candidate = np.ma.array(slope <= 8.0, mask=slope == slope_georef.nodata)

    labels, _ = ls.label_regions(candidate, slope_georef)
    sizes, _ = ls.region_sizes(candidate, slope_georef)
    filtered, filtered_georef = ls.filter_regions_by_size(
        candidate,
        slope_georef,
        threshold=80,
        comparator=">=",
        cleanup="opening",
        iterations=1,
    )
    borders, borders_georef = ls.find_borders(filtered, filtered_georef)

    outputs = {
        "labels.tif": (np.ma.filled(labels, 0).astype(np.int32), slope_georef.with_nodata(0)),
        "region_sizes.tif": (np.ma.filled(sizes, 0).astype(np.int32), slope_georef.with_nodata(0)),
        "large_regions.tif": (np.ma.filled(filtered, False).astype(np.uint8), filtered_georef.with_nodata(0)),
        "borders.tif": (np.ma.filled(borders, False).astype(np.uint8), borders_georef.with_nodata(0)),
    }
    for filename, (values, output_georef) in outputs.items():
        print(
            ls.write_geotiff(
                scenario.output_path(f"analysis/regions/{filename}"),
                values,
                output_georef,
                overwrite=True,
            )
        )


if __name__ == "__main__":
    main()
