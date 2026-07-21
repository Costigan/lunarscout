"""Compare grids and explicitly align a raster to a reference grid.

Inputs: deterministic synthetic scenario created under --workspace.
Outputs: a deliberately shifted source and its aligned result.
Resources: small GDAL warp; a GPU is not required.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import ensure_synthetic_scenario, example_parser, synthetic_georef


def main() -> None:
    args = example_parser(__doc__).parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    reference, reference_georef = ls.read_geotiff(scenario.dem_path())
    if reference_georef is None:
        raise RuntimeError("The DEM must be georeferenced.")

    source = np.roll(reference, shift=1, axis=1)
    source_georef = synthetic_georef(origin_x=-315.0, origin_y=315.0)
    print(f"same grid before alignment: {ls.same_grid(source_georef, reference_georef)}")
    aligned, aligned_georef = ls.align(
        source,
        source_georef,
        to=reference_georef,
        resampling="bilinear",
        output_nodata=-9999.0,
    )
    ls.require_same_grid(aligned_georef, reference_georef)
    print(f"same grid after alignment: {ls.same_grid(aligned_georef, reference_georef)}")
    print(
        ls.write_geotiff(
            scenario.output_path("analysis/alignment/aligned.tif"),
            aligned,
            aligned_georef,
            overwrite=True,
        )
    )


if __name__ == "__main__":
    main()
