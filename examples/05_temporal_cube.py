"""Create an in-memory temporal cube and reduce its UTC time axis.

Inputs: deterministic synthetic scenario created under --workspace.
Outputs: temporal mean, minimum, maximum, and standard deviation GeoTIFFs.
Resources: six small in-memory float32 layers; a GPU is not required.
"""

from __future__ import annotations

import lunarscout as ls

from _example_support import (
    ensure_synthetic_scenario,
    example_parser,
    synthetic_temporal_cube,
)


def main() -> None:
    args = example_parser(__doc__).parse_args()
    scenario = ensure_synthetic_scenario(args.workspace)
    _dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("The DEM must be georeferenced.")
    cube = synthetic_temporal_cube(georef)

    reducers = {
        "mean.tif": ls.temporal_mean,
        "minimum.tif": ls.temporal_min,
        "maximum.tif": ls.temporal_max,
        "standard_deviation.tif": ls.temporal_std,
    }
    print(f"cube shape={cube.shape}, dtype={cube.dtype}, bytes={cube.nbytes}")
    print(f"UTC samples: {cube.times[0]} through {cube.times[-1]}")
    for filename, reducer in reducers.items():
        values, output_georef = reducer(cube)
        print(
            ls.write_geotiff(
                scenario.output_path(f"analysis/temporal/{filename}"),
                values,
                output_georef,
                overwrite=True,
            )
        )


if __name__ == "__main__":
    main()
