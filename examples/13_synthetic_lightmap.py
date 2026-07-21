"""Generate a synthetic CPU lightmap from explicit Moon-ME Sun vectors.

Requires the synthetic horizon scenario (downloaded on first use from
GitHub Releases; cached after that).  Uses explicit vectors so no SPICE
kernel loading is needed.  Runs entirely on CPU.  Outputs a multi-band
BigTIFF with one uint8 band per time sample.
"""

from __future__ import annotations

import numpy as np
import lunarscout as ls

from _example_support import ensure_synthetic_horizon_scenario, example_parser


def main() -> None:
    args = example_parser(__doc__ or "").parse_args()
    scenario = ensure_synthetic_horizon_scenario(args.workspace)
    if scenario is None:
        print("Synthetic horizon data unavailable.  Skipping.")
        return

    dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("Synthetic DEM unexpectedly lacks georeferencing.")

    times = ls.times("2027-01-01T00:00:00Z", "2027-01-01T06:00:00Z", step_hours=2)
    moon_radius_m = 1_737_400.0
    sun_distance_m = 147_010_225_480.0

    sun_vectors_m = np.array(
        [
            [-17_640_711.75, -145_927_452.22, -2_447_524.08],
            [-20_218_498.23, -145_586_452.13, -2_421_408.36],
            [-22_789_853.74, -145_199_933.92, -2_395_370.49],
            [-25_353_974.44, -144_768_018.14, -2_369_416.88],
        ],
        dtype=np.float64,
    )

    output = scenario.output_path("analysis/synthetic_lightmap.tif")

    print(f"DEM: {dem.shape}  {dem.dtype}")
    print(f"Times: {times.time_count}")
    print(f"Sun vectors: {sun_vectors_m.shape}  {sun_vectors_m.dtype}")
    print(f"Output: {output}")

    result = ls.generate_lightmap(
        str(scenario.dem_path()),
        str(scenario.horizons_path()),
        str(output),
        times=times,
        sun_vectors_m=sun_vectors_m,
        backend="cpu",
        verbose=True,
    )
    print(f"Wrote: {result}")

    lightmap, lightmap_georef = ls.read_geotiff(str(result))

    print(f"\nLightmap: shape={lightmap.shape}  dtype={lightmap.dtype}")
    print(f"  nodata={lightmap_georef.nodata if lightmap_georef else None}")
    for b in range(times.time_count):
        band, _ = ls.read_geotiff(str(result), band=b + 1)
        valid = band > 0
        print(
            f"  band {b}: "
            f"min={int(band.min()):3d}  max={int(band.max()):3d}  "
            f"mean={band.mean():6.1f}  valid_px={valid.sum()}"
        )


if __name__ == "__main__":
    main()
