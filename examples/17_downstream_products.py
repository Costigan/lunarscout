#!/usr/bin/env python3
"""Generate public lighting products from an existing scenario.

The scenario must contain ``dem.tif`` and compatible ``horizons/`` tiles.
Generated Sun and Earth vectors use Lunarscout's configured SPICE kernels.
Use ``--backend cpu`` on a machine without an NVIDIA GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lunarscout as ls


PRODUCTS = (
    "lightmap",
    "psr",
    "sun-elevation",
    "earth-elevation",
    "safe-havens",
    "mission-sunlight",
    "mission-sun-elevation",
    "mission-sunlight-earth",
    "mission-sun-earth-elevation",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("scenario", type=Path, help="Scenario containing dem.tif and horizons/.")
    result.add_argument(
        "--product",
        action="append",
        choices=(*PRODUCTS, "all"),
        help="Product to generate; repeat the option or pass 'all'. Default: lightmap.",
    )
    result.add_argument(
        "--output-directory",
        default="analysis/example-products",
        help="Scenario-relative output directory.",
    )
    result.add_argument("--start", default="2029-01-01T00:00:00Z")
    result.add_argument("--stop", default="2029-01-02T00:00:00Z")
    result.add_argument("--step-hours", type=float, default=6.0)
    result.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="CPU works with the base installation; CUDA requires lunarscout[cuda].",
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> int:
    arguments = parser().parse_args()
    scenario = ls.open_scenario(arguments.scenario)
    missing = [
        path
        for path in (scenario.dem_path(), scenario.horizons_path())
        if not path.exists()
    ]
    if missing:
        raise SystemExit("Missing scenario input: " + ", ".join(map(str, missing)))

    selected = arguments.product or ["lightmap"]
    if "all" in selected:
        selected = list(PRODUCTS)
    selected = list(dict.fromkeys(selected))
    times = ls.times(
        arguments.start,
        arguments.stop,
        step_hours=arguments.step_hours,
    )
    if times.time_count < 2:
        raise SystemExit("At least two time samples are required.")

    common = {
        "times": times,
        "backend": arguments.backend,
        "overwrite": arguments.overwrite,
        "verbose": True,
    }
    mission_common = {
        **common,
        "evaluation_start": times.start,
        "evaluation_stop": times.stop,
        "candidate_start_intervals": ((times.start, times.stop),),
        "output_unit": "hours",
    }
    output_directory = Path(arguments.output_directory)
    operations = {
        "lightmap": lambda: scenario.lightmap(
            output_directory / "lightmap.tif", **common
        ),
        "psr": lambda: scenario.psr(output_directory / "psr.tif", **common),
        "sun-elevation": lambda: scenario.sun_elevation(
            output_directory / "sun-elevation.tif", **common
        ),
        "earth-elevation": lambda: scenario.earth_elevation(
            output_directory / "earth-elevation.tif", **common
        ),
        "safe-havens": lambda: scenario.safe_havens(
            output_directory / "safe-havens.tif",
            earth_elevation_threshold_deg=2.0,
            sunlight_fraction_threshold=0.2,
            **common,
        ),
        "mission-sunlight": lambda: scenario.mission_duration_from_sunlight(
            output_directory / "mission-sunlight.tif",
            sunlight_fraction_threshold=0.5,
            **mission_common,
        ),
        "mission-sun-elevation": lambda: scenario.mission_duration_from_sun_elevation(
            output_directory / "mission-sun-elevation.tif",
            sun_elevation_threshold_deg=0.0,
            **mission_common,
        ),
        "mission-sunlight-earth": lambda: scenario.mission_duration_from_sunlight_and_earth(
            output_directory / "mission-sunlight-earth.tif",
            sunlight_fraction_threshold=0.5,
            earth_elevation_threshold_deg=0.0,
            **mission_common,
        ),
        "mission-sun-earth-elevation": lambda: scenario.mission_duration_from_sun_and_earth_elevation(
            output_directory / "mission-sun-earth-elevation.tif",
            sun_elevation_threshold_deg=0.0,
            earth_elevation_threshold_deg=0.0,
            **mission_common,
        ),
    }

    for product in selected:
        print(f"Generating {product} with backend={arguments.backend}", flush=True)
        print(f"Completed: {operations[product]()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
