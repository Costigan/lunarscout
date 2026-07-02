"""Shared public-API-only support for the Lunarscout example scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import lunarscout as ls
import numpy as np


_WKT = (
    'PROJCS["ESRI:103878",'
    'GEOGCS["Moon_2000",DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["Meter",1]]'
)
_PROJ4 = (
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+R=1737400 +units=m +no_defs"
)


def example_parser(description: str, *, native: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("LUNARSCOUT_EXAMPLE_WORKSPACE", "/tmp/lunarscout_examples")),
        help="Directory for deterministic fixtures and generated outputs.",
    )
    if native:
        parser.add_argument(
            "--scenario",
            type=Path,
            default=(
                Path(os.environ["LUNARSCOUT_EXAMPLE_SCENARIO"])
                if "LUNARSCOUT_EXAMPLE_SCENARIO" in os.environ
                else None
            ),
            help="Real scenario containing dem.tif and horizons.",
        )
        parser.add_argument("--start", default="2027-01-01T00:00:00Z")
        parser.add_argument("--stop", default="2027-01-01T02:00:00Z")
        parser.add_argument("--step-hours", type=float, default=1.0)
        parser.add_argument("--observer-elevation-meters", type=float, default=0.0)
        parser.add_argument("--overwrite", action="store_true")
    return parser


def synthetic_georef(
    *,
    width: int = 64,
    height: int = 64,
    origin_x: float = -320.0,
    origin_y: float = 320.0,
    pixel_size: float = 10.0,
    nodata: int | float | None = -9999.0,
) -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4=_PROJ4,
        affine_transform=(origin_x, pixel_size, 0.0, origin_y, 0.0, -pixel_size),
        width=width,
        height=height,
        pixel_size_x=pixel_size,
        pixel_size_y=-pixel_size,
        nodata=nodata,
    )


def synthetic_dem() -> np.ndarray:
    rows, columns = np.indices((64, 64), dtype=np.float32)
    dem = 100.0 + 0.15 * columns + 0.08 * rows
    dem += 15.0 * np.exp(-((columns - 48.0) ** 2 + (rows - 18.0) ** 2) / 45.0)
    dem[12:30, 10:30] = 104.0
    dem[38:55, 35:57] = 112.0
    dem[0, 0] = -9999.0
    return dem.astype(np.float32)


def ensure_synthetic_scenario(workspace: Path) -> ls.Scenario:
    root = workspace.expanduser().resolve() / "synthetic_scenario"
    root.mkdir(parents=True, exist_ok=True)
    scenario = ls.open_scenario(root)
    if not scenario.dem_path().is_file():
        ls.write_geotiff(scenario.dem_path(), synthetic_dem(), synthetic_georef())
    return scenario


def synthetic_times() -> ls.TimeRange:
    return ls.times(
        "2027-01-01T00:00:00Z",
        "2027-01-01T05:00:00Z",
        step_hours=1,
    )


def synthetic_temporal_cube(georef: ls.GeoReference) -> ls.TemporalCube:
    times = synthetic_times()
    rows, columns = np.indices((georef.height, georef.width), dtype=np.float32)
    spatial = np.clip(0.35 + columns / max(1, georef.width - 1) * 0.45, 0.0, 1.0)
    layers = [
        np.clip(spatial + 0.08 * np.sin(index * np.pi / 3) - rows * 0.001, 0.0, 1.0)
        for index in range(times.time_count)
    ]
    return ls.TemporalCube(
        np.asarray(layers, dtype=np.float32),
        times,
        georef.with_nodata(None),
    )


def ensure_synthetic_series(workspace: Path) -> ls.TemporalGeoTiffSeries:
    scenario = ensure_synthetic_scenario(workspace)
    _dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("Synthetic DEM unexpectedly lacks georeferencing.")
    path = scenario.output_path("analysis/synthetic_sun.temporal")
    if path.is_dir():
        return ls.open_temporal_cube(path)
    return ls.write_temporal_cube(
        path,
        synthetic_temporal_cube(georef),
        signal_name="sun_fraction",
        units="fraction",
        provenance={"source": "deterministic Lunarscout example"},
    )


def require_native_scenario(path: Path | None) -> ls.Scenario:
    if path is None:
        raise SystemExit(
            "A real scenario is required. Pass --scenario or set "
            "LUNARSCOUT_EXAMPLE_SCENARIO."
        )
    scenario = ls.open_scenario(path)
    missing = [
        str(required)
        for required in (scenario.dem_path(), scenario.horizons_path())
        if not required.exists()
    ]
    if missing:
        raise SystemExit("Scenario prerequisites are missing: " + ", ".join(missing))
    native_status = ls.native.status()
    if not native_status["available"]:
        unavailable = [
            name
            for name, detail in native_status["components"].items()
            if not detail.get("available", False)
        ]
        raise SystemExit("Native runtime is unavailable: " + ", ".join(unavailable))
    return scenario


def native_times(args: argparse.Namespace) -> ls.TimeRange:
    return ls.times(args.start, args.stop, step_hours=args.step_hours)
