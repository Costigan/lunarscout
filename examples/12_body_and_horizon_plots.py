"""Plot body elevation over time and overlay paths on a terrain horizon.

Requires the synthetic horizon scenario (downloaded on first use from
GitHub Releases; cached after that).  Uses matplotlib to render plots as
PNG files under analysis/plots/.  No GPU needed.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import lunarscout as ls

from _example_support import ensure_synthetic_horizon_scenario, example_parser


def main() -> None:
    args = example_parser(__doc__ or "").parse_args()
    scenario = ensure_synthetic_horizon_scenario(args.workspace)
    if scenario is None:
        print("Synthetic horizon data unavailable.  Skipping.")
        return

    plots_dir = args.workspace.expanduser().resolve() / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    point = ls.LonLat(longitude=0.0, latitude=-89.99)
    sample_times = list(
        ls.iter_times(
            "2027-01-01T00:00:00Z",
            "2027-01-01T12:00:00Z",
            timedelta(hours=1),
        )
    )

    # ------------------------------------------------------------------
    # 1. Body elevation over time
    # ------------------------------------------------------------------
    print("1. Sun elevation over time")
    fig, ax = ls.plot_body_elevation(point, "sun", sample_times, grid=True)
    fig.savefig(str(plots_dir / "sun_elevation_over_time.png"), dpi=120)
    matplotlib.pyplot.close(fig)

    # ------------------------------------------------------------------
    # 2. Multiple bodies over time
    # ------------------------------------------------------------------
    print("2. Sun and Earth elevation over time")
    fig, ax = ls.plot_body_elevations(
        point, ["sun", "earth"], sample_times, grid=True
    )
    fig.savefig(str(plots_dir / "sun_earth_elevation_over_time.png"), dpi=120)
    matplotlib.pyplot.close(fig)

    # ------------------------------------------------------------------
    # 3. Empty azimuth/elevation polar axes
    # ------------------------------------------------------------------
    print("3. Azimuth/elevation axes")
    fig, ax = scenario.plot_azimuth_elevation_axes(
        center_azimuth=180.0,
        elevation_limits=(-20.0, 60.0),
    )
    fig.savefig(str(plots_dir / "azimuth_elevation_axes.png"), dpi=120)
    matplotlib.pyplot.close(fig)

    # ------------------------------------------------------------------
    # 4. Horizon plot for a surface point
    # ------------------------------------------------------------------
    print("4. Terrain horizon at surface point")
    fig, ax = scenario.plot_horizon(point, center_azimuth=0.0)
    fig.savefig(str(plots_dir / "horizon.png"), dpi=120)
    matplotlib.pyplot.close(fig)

    # ------------------------------------------------------------------
    # 5. Body positions as markers on the horizon
    # ------------------------------------------------------------------
    print("5. Sun and Earth positions overlaid on horizon")
    fig, ax = scenario.plot_horizon(point, center_azimuth=90.0)
    scenario.plot_body_position(ax, point, "sun", "2027-01-01T06:00:00Z", style="center")
    scenario.plot_body_position(
        ax, point, "earth", "2027-01-01T06:00:00Z", style="limb"
    )
    fig.savefig(str(plots_dir / "horizon_with_body_positions.png"), dpi=120)
    matplotlib.pyplot.close(fig)

    # ------------------------------------------------------------------
    # 6. Body path with limb bands
    # ------------------------------------------------------------------
    print("6. Sun and Earth paths with limb bands")
    fig, ax = scenario.plot_azimuth_elevation_axes(center_azimuth=90.0)
    scenario.plot_body_path(
        ax, point, "sun", sample_times, style="center_and_limbs", label="Sun"
    )
    scenario.plot_body_path(
        ax, point, "earth", sample_times, style="limbs", label="Earth"
    )
    ax.legend()
    fig.savefig(str(plots_dir / "body_paths.png"), dpi=120)
    matplotlib.pyplot.close(fig)

    # ------------------------------------------------------------------
    # 7. Zoomed body path (equal-scale horizon view)
    # ------------------------------------------------------------------
    print("7. Zoomed Sun/Earth path against horizon")
    fig, ax = scenario.plot_zoomed_body_path(
        point,
        bodies=["sun", "earth"],
        times=sample_times,
        observer_height_decimeters=0,
    )
    fig.savefig(str(plots_dir / "zoomed_body_path.png"), dpi=120)
    matplotlib.pyplot.close(fig)

    print(f"\nAll plots saved to {plots_dir}")
    for p in sorted(plots_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
