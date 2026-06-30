"""Generate native solar visibility as an explicitly in-memory cube.

Inputs: --scenario with dem.tif, lighting/horizons, SPICE, and native runtime.
Outputs: analysis/native_mean_sun_fraction.tif in the supplied scenario.
Resources: preflighted float32 cube; default limit is 2 GiB.
"""

from __future__ import annotations

import lunarscout as ls

from _example_support import example_parser, native_times, require_native_scenario


def report(progress: ls.native.NativeTemporalProgress) -> None:
    print(f"{progress.percent:6.2f}% [{progress.stage}] {progress.message}")


def main() -> None:
    args = example_parser(__doc__, native=True).parse_args()
    scenario = require_native_scenario(args.scenario)
    times = native_times(args)
    _dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise SystemExit("Scenario DEM must be georeferenced.")
    estimate = ls.native.estimate_temporal_allocation(
        signal="sun_fraction",
        times=times,
        georef=georef,
        storage="memory",
    )
    print(f"preflight shape={estimate.shape}, bytes={estimate.estimated_bytes}")
    cube = scenario.sun_fraction(
        times=times,
        storage="memory",
        observer_elevation_meters=args.observer_elevation_meters,
        progress_callback=report,
    )
    mean, mean_georef = ls.temporal_mean(cube)
    print(
        ls.write_geotiff(
            scenario.output_path("analysis/native_mean_sun_fraction.tif"),
            mean,
            mean_georef,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
