from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPOSITORY_ROOT / "examples"
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
_DETERMINISTIC_SCRIPTS = [
    "00_geotiff_and_coordinates.py",
    "01_terrain_products.py",
    "02_region_filtering.py",
    "03_alignment.py",
    "04_temporal_cube.py",
    "05_file_backed_series.py",
    "06_streaming_reductions.py",
    "09_qgis_vrt.py",
    "10_landing_site_screening.py",
    "14_timeseries_two_file_prototype.py",
]

_SCRIPT_ARGS = {
    "14_timeseries_two_file_prototype.py": [
        "--width",
        "256",
        "--height",
        "256",
        "--time-count",
        "64",
    ],
}


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_SOURCE_ROOT)
    environment["PYTHONWARNINGS"] = "ignore::FutureWarning"
    return environment


def test_deterministic_example_sequence_runs(tmp_path: Path) -> None:
    for script_name in _DETERMINISTIC_SCRIPTS:
        completed = subprocess.run(
            [
                sys.executable,
                str(_EXAMPLES / script_name),
                    "--workspace",
                    str(tmp_path),
                    *_SCRIPT_ARGS.get(script_name, []),
                ],
            cwd=_REPOSITORY_ROOT,
            env=_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, (
            f"{script_name} failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    scenario = tmp_path / "synthetic_scenario"
    assert (scenario / "analysis" / "dem_copy.tif").is_file()
    assert (scenario / "analysis" / "terrain" / "slope_deg.tif").is_file()
    assert (scenario / "analysis" / "synthetic_sun.temporal" / "COMPLETE").is_file()
    assert (scenario / "analysis" / "screening" / "candidate_sites.tif").is_file()
    prototype = scenario / "analysis" / "timeseries_storage_prototype"
    assert (prototype / "shadow_maps.tif").is_file()
    assert (prototype / "light_curves_chunk16.h5").is_file()
    report = json.loads((prototype / "report.json").read_text(encoding="utf-8"))
    assert report["prototype"] == "two_file_timeseries_storage"
    assert report["shape"] == {"height": 256, "time": 64, "width": 256}


def test_native_examples_explain_missing_scenario() -> None:
    environment = _environment()
    environment.pop("LUNARSCOUT_EXAMPLE_SCENARIO", None)
    for script_name in (
        "07_native_sun_fraction.py",
        "08_native_horizon_margins.py",
        "11_native_end_to_end_validation.py",
        "12_native_performance_benchmark.py",
        "13_native_psr.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(_EXAMPLES / script_name)],
            cwd=_REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode != 0
        assert "Pass --scenario" in completed.stderr
