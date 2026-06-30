"""Stream temporal reductions from timestamped TIFFs without loading a cube.

Inputs: deterministic file-backed series created under --workspace.
Outputs: mean, minimum, maximum, and standard deviation GeoTIFFs.
Resources: memory proportional to one layer plus reducer accumulators.
"""

from __future__ import annotations

import lunarscout as ls

from _example_support import ensure_synthetic_series, example_parser


def main() -> None:
    args = example_parser(__doc__).parse_args()
    series = ensure_synthetic_series(args.workspace)
    reducers = {
        "mean.tif": ls.temporal_mean,
        "minimum.tif": ls.temporal_min,
        "maximum.tif": ls.temporal_max,
        "standard_deviation.tif": ls.temporal_std,
    }
    for filename, reducer in reducers.items():
        values, georef = reducer(series)
        output = ls.write_geotiff(
            series.root.parent / "streamed_reductions" / filename,
            values,
            georef,
            overwrite=True,
        )
        print(output)


if __name__ == "__main__":
    main()
