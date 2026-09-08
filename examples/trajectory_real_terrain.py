#!/usr/bin/env python3
"""Compare dynamic planners on a local real DEM and sunlight archive.

Prepare an ``.npz`` archive with ``sunlight`` as a uint8 array shaped
``(interval, y, x)`` and ``boundaries_utc`` as ``interval + 1`` ISO-8601 UTC
strings. Supply a georeferenced DEM on the same grid and edit/select start and
goal cells for that site. This manual CPU example never generates horizons and
does not require a GPU. It prints a reproducibility report and optionally writes
that report atomically as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import lunarscout as ls
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _boundary(value: object) -> datetime:
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("boundaries_utc values must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _fingerprint(result: ls.trajectory.DynamicPathResult) -> str | None:
    if not result.reachable:
        return None
    assert result.cells is not None
    assert result.arrival_times is not None
    assert result.departure_times is not None
    digest = hashlib.sha256(result.cells.tobytes(order="C"))
    for value in (*result.arrival_times, *result.departure_times):
        digest.update(value.isoformat().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _run(
    algorithm: str,
    traversable: np.ndarray,
    elevation: np.ndarray,
    georef: ls.GeoReference,
    start: tuple[int, int],
    goal: tuple[int, int],
    boundaries: tuple[datetime, ...],
    configuration: ls.trajectory.ConfigurationSpaceProvider,
    model: ls.trajectory.StaticTravelModel,
) -> dict[str, object]:
    tracemalloc.start()
    started = time.perf_counter()
    result = ls.trajectory.dynamic_path(
        traversable,
        georef,
        start,
        goal,
        boundaries,
        configuration,
        boundaries[0],
        valid=traversable,
        elevation=elevation,
        model=model,
        algorithm=algorithm,
        backend="cpu",
    )
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "algorithm": algorithm,
        "backend": "cpu",
        "reachable": result.reachable,
        "arrival_time_utc": (
            None if result.arrival_time is None else result.arrival_time.isoformat()
        ),
        "travel_time_hours": result.travel_time_hours,
        "path_cell_count": None if result.cells is None else len(result.cells),
        "wait_intervals": [
            [start.isoformat(), stop.isoformat()]
            for start, stop in result.wait_intervals
        ],
        "trajectory_sha256": _fingerprint(result),
        "runtime_seconds": elapsed,
        "peak_tracemalloc_bytes": peak,
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--sunlight", type=Path, required=True)
    parser.add_argument("--start", type=int, nargs=2, metavar=("X", "Y"), required=True)
    parser.add_argument("--goal", type=int, nargs=2, metavar=("X", "Y"), required=True)
    parser.add_argument("--speed-m-per-h", type=float, default=36.0)
    parser.add_argument("--minimum-sunlight", type=float, default=0.2)
    parser.add_argument(
        "--algorithm",
        choices=("both", "gridrunner", "safe_interval"),
        default="both",
    )
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    dem_path = arguments.dem.expanduser().resolve()
    sunlight_path = arguments.sunlight.expanduser().resolve()
    output = (
        None
        if arguments.output_report is None
        else arguments.output_report.expanduser().resolve()
    )
    if output is not None and output.exists() and not arguments.overwrite:
        parser.error("--output-report already exists; pass --overwrite to replace it")

    elevation, georef = ls.read_geotiff(dem_path)
    if georef is None:
        parser.error("--dem must contain complete georeferencing")
    with np.load(sunlight_path, allow_pickle=False) as archive:
        if set(archive.files) != {"boundaries_utc", "sunlight"}:
            parser.error(
                "--sunlight must contain only boundaries_utc and sunlight arrays"
            )
        boundaries = tuple(_boundary(value) for value in archive["boundaries_utc"])
        sunlight = np.array(archive["sunlight"], copy=True)

    valid = np.isfinite(elevation)
    if georef.nodata is not None:
        valid &= elevation != georef.nodata
    source = ls.trajectory.ArraySunlightProvider(boundaries, sunlight, georef)
    configuration = ls.trajectory.AllOfConfigurationSpaceProvider(
        (
            ls.trajectory.StaticConfigurationSpaceProvider(valid, georef),
            ls.trajectory.SunlightThresholdProvider(
                source, arguments.minimum_sunlight
            ),
        )
    )
    model = ls.trajectory.StaticTravelModel(
        speed_m_per_h=arguments.speed_m_per_h,
        include_diagonals=True,
    )
    algorithms = (
        ("gridrunner", "safe_interval")
        if arguments.algorithm == "both"
        else (arguments.algorithm,)
    )
    results = [
        _run(
            algorithm,
            valid,
            elevation,
            georef,
            tuple(arguments.start),
            tuple(arguments.goal),
            boundaries,
            configuration,
            model,
        )
        for algorithm in algorithms
    ]
    spacings = np.diff(
        np.asarray(
            [value.timestamp() for value in boundaries],
            dtype=np.float64,
        )
    ) / 3600.0
    report = {
        "schema": "lunarscout-trajectory-real-terrain-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dem": {"path": str(dem_path), "sha256": _sha256(dem_path)},
            "sunlight": {
                "path": str(sunlight_path),
                "sha256": _sha256(sunlight_path),
            },
            "grid": {"width": georef.width, "height": georef.height},
            "start": list(arguments.start),
            "goal": list(arguments.goal),
        },
        "environment_sampling": {
            "boundary_count": len(boundaries),
            "start_utc": boundaries[0].isoformat(),
            "stop_utc": boundaries[-1].isoformat(),
            "minimum_step_hours": float(np.min(spacings)),
            "maximum_step_hours": float(np.max(spacings)),
            "sunlight_dtype": str(sunlight.dtype),
            "minimum_sunlight_fraction": arguments.minimum_sunlight,
        },
        "model": {
            "speed_m_per_h": model.speed_m_per_h,
            "include_diagonals": model.include_diagonals,
            "slip": None,
        },
        "software": {
            "lunarscout": ls.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "gpu_required": False,
        },
        "results": results,
        "cost_agreement": (
            len(results) < 2
            or results[0]["reachable"] == results[1]["reachable"]
            and results[0]["travel_time_hours"] == results[1]["travel_time_hours"]
        ),
    }
    if output is not None:
        _write_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["cost_agreement"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
