"""Plan a static rover path and travel-time field on synthetic lunar terrain.

Inputs: none; the grid, elevation, validity, and barriers are deterministic.
Outputs: printed route/cost summaries; no files are written.
Resources: small in-memory CPU arrays; no files, SPICE kernels, or GPU.
"""

from __future__ import annotations

import lunarscout as ls
import numpy as np

from _example_support import example_parser, synthetic_georef


def main() -> None:
    example_parser(__doc__).parse_args()
    georef = synthetic_georef(width=12, height=9, pixel_size=10.0, nodata=None)
    rows, columns = np.indices((georef.height, georef.width), dtype=np.float64)
    elevation_m = 100.0 + 0.25 * columns + 0.1 * rows
    traversable = np.ones(elevation_m.shape, dtype=np.bool_)
    traversable[1:8, 5] = False
    traversable[6, 5] = True  # A single pass through the barrier.
    valid = np.ones(elevation_m.shape, dtype=np.bool_)
    valid[0, 10:] = False

    slip = ls.trajectory.SlipFunction(
        signed_slopes=(-0.5, 0.0, 0.5),
        factors=(0.8, 1.0, 2.0),
        extrapolation="infeasible",
    )
    model = ls.trajectory.StaticTravelModel(
        speed_m_per_h=36.0,
        include_diagonals=True,
        slip=slip,
    )
    start = (1, 1)
    goal = (10, 7)
    field = ls.trajectory.static_travel_time(
        traversable,
        georef,
        start,
        valid=valid,
        elevation=elevation_m,
        model=model,
    )
    route = ls.trajectory.static_path(
        traversable,
        georef,
        start,
        goal,
        valid=valid,
        elevation=elevation_m,
        model=model,
    )

    print(f"Reached cells: {int(field.reached.sum())}/{field.reached.size}")
    print(f"Route reachable: {route.reachable}")
    print(f"Travel time: {route.travel_time_hours:.3f} hours")
    print(f"Path cells [x, y]:\n{route.path}")
    print(
        "A* and Dijkstra costs agree:",
        np.isclose(route.travel_time_hours, field.travel_time_hours[goal[1], goal[0]]),
    )

    blocked = traversable.copy()
    blocked[:, 5] = False
    unreachable = ls.trajectory.static_path(
        blocked,
        georef,
        start,
        goal,
        valid=valid,
        elevation=elevation_m,
        model=model,
    )
    print(f"Goal behind complete barrier reachable: {unreachable.reachable}")


if __name__ == "__main__":
    main()
