"""Plan an exact dynamic rover path on a synthetic lunar grid.

Inputs: none; occupancy intervals and terrain are deterministic.
Outputs: printed route, arrival, and wait summaries; no files are written.
Resources: small in-memory CPU arrays; no files, SPICE kernels, or GPU.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import lunarscout as ls
import numpy as np

from _example_support import example_parser, synthetic_georef


def main() -> None:
    example_parser(__doc__).parse_args()
    georef = synthetic_georef(width=7, height=3, pixel_size=10.0, nodata=None)
    start_time = datetime(2035, 1, 1, tzinfo=timezone.utc)
    boundaries = tuple(start_time + timedelta(hours=index) for index in range(5))

    sunlight = np.full((4, georef.height, georef.width), 255, dtype=np.uint8)
    sunlight[:2, :, 4:] = 0
    sunlight[:, 0, 3] = 0
    signal = ls.trajectory.ArraySunlightProvider(boundaries, sunlight, georef)
    sun_required = ls.trajectory.SunlightThresholdProvider(signal, 0.2)
    static_allowed = np.ones((georef.height, georef.width), dtype=np.bool_)
    configuration = ls.trajectory.AllOfConfigurationSpaceProvider(
        (
            ls.trajectory.StaticConfigurationSpaceProvider(
                static_allowed, georef
            ),
            sun_required,
        )
    )

    result = ls.trajectory.dynamic_path(
        static_allowed,
        georef,
        (0, 1),
        (6, 1),
        boundaries,
        configuration,
        start_time,
        model=ls.trajectory.StaticTravelModel(
            speed_m_per_h=20.0,
            include_diagonals=False,
        ),
        algorithm="gridrunner",
        backend="cpu",
    )

    print(f"Route reachable: {result.reachable}")
    print(f"Arrival: {result.arrival_time.isoformat()}")
    print(f"Elapsed time: {result.travel_time_hours:.3f} hours")
    print(f"Path cells [x, y]:\n{result.cells}")
    print(f"Wait intervals: {result.wait_intervals}")


if __name__ == "__main__":
    main()
