"""Compare exact and greedy SOC planning on a synthetic lunar grid.

Inputs: none; terrain, sunlight, and power models are deterministic.
Outputs: printed reachability, route, arrival, and battery summaries; no files.
Resources: small in-memory CPU arrays; no files, SPICE kernels, or GPU.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import lunarscout as ls
import numpy as np

from _example_support import example_parser, synthetic_georef


def main() -> None:
    example_parser(__doc__).parse_args()
    georef = synthetic_georef(width=3, height=2, pixel_size=10.0, nodata=None)
    start_time = datetime(2035, 1, 1, tzinfo=timezone.utc)
    boundaries = (start_time, start_time + timedelta(hours=4))
    available = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.bool_)
    configuration = ls.trajectory.StaticConfigurationSpaceProvider(
        available, georef
    )
    sunlight_values = np.zeros((1, 2, 3), dtype=np.uint8)
    sunlight_values[:, 1, :2] = 255
    sunlight = ls.trajectory.ArraySunlightProvider(
        boundaries, sunlight_values, georef
    )
    solar = ls.trajectory.SolarPowerModel(rated_power_w=500.0)
    battery = ls.trajectory.BatteryModel(
        capacity_wh=300.0,
        initial_energy_wh=300.0,
        minimum_energy_wh=0.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    rover = ls.trajectory.RoverPowerModel(
        drive_power_w=400.0,
        idle_power_w=0.0,
    )
    common = {
        "model": ls.trajectory.StaticTravelModel(
            speed_m_per_h=20.0,
            include_diagonals=False,
        ),
        "solar": solar,
        "battery": battery,
        "rover": rover,
        "backend": "cpu",
    }

    exact = ls.trajectory.soc_path(
        available,
        georef,
        (0, 0),
        (2, 0),
        boundaries,
        configuration,
        sunlight,
        start_time,
        algorithm="exact",
        **common,
    )
    greedy = ls.trajectory.soc_path(
        available,
        georef,
        (0, 0),
        (2, 0),
        boundaries,
        configuration,
        sunlight,
        start_time,
        algorithm="greedy",
        **common,
    )

    print(f"Exact reachable: {exact.reachable}")
    print(f"Exact arrival: {exact.arrival_time.isoformat()}")
    print(f"Exact final battery: {exact.final_energy_wh:.1f} Wh")
    print(f"Exact cells [x, y]:\n{exact.cells}")
    print(f"Greedy reachable: {greedy.reachable}")
    print(f"Greedy complete guarantee: {greedy.complete}")


if __name__ == "__main__":
    main()
