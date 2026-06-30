"""Generate file-backed native Sun and Earth horizon-margin time series.

Inputs: --scenario with dem.tif, lighting/horizons, SPICE, and native runtime.
Outputs: two timestamped series under analysis/ in the supplied scenario.
Resources: temporary uncompressed disk scratch plus final compressed outputs.
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
    operations = (
        (
            scenario.sun_over_horizon_deg,
            "analysis/sun_over_horizon.temporal",
        ),
        (
            scenario.earth_over_horizon_deg,
            "analysis/earth_over_horizon.temporal",
        ),
    )
    for operation, output in operations:
        series = operation(
            times=times,
            storage="geotiff_series",
            output=output,
            observer_elevation_meters=args.observer_elevation_meters,
            overwrite=args.overwrite,
            progress_callback=report,
        )
        print(f"{series.signal_name}: {series.root}")


if __name__ == "__main__":
    main()
