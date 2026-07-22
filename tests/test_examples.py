from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPOSITORY_ROOT / "examples"
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
_DETERMINISTIC_SCRIPTS = [
    "01_geotiff_and_coordinates.py",
    "02_terrain_products.py",
    "03_region_filtering.py",
    "04_alignment.py",
    "05_temporal_cube.py",
    "06_file_backed_series.py",
    "07_incremental_writer.py",
    "08_streaming_reductions.py",
    "09_qgis_vrt.py",
    "10_landing_site_screening.py",
    "21_map_algebra_terrain_resample.py",
]


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
    terrain_resample = tmp_path / "terrain_resample"
    assert (terrain_resample / "slope.tif").is_file()
    assert (terrain_resample / "resampled_hillshade.tif").is_file()
    assert (terrain_resample / "combined_score.tif").is_file()


def test_historical_hdf5_prototype_when_manual_dependencies_are_present(
    tmp_path: Path,
) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")

    completed = subprocess.run(
        [
            sys.executable,
            str(_REPOSITORY_ROOT / "benchmarks" / "timeseries_two_file_prototype.py"),
            "--workspace",
            str(tmp_path),
            "--width",
            "256",
            "--height",
            "256",
            "--time-count",
            "64",
        ],
        cwd=_REPOSITORY_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"historical HDF5 prototype failed\nstdout:\n{completed.stdout}"
        f"\nstderr:\n{completed.stderr}"
    )

    scenario = tmp_path / "synthetic_scenario"
    prototype = scenario / "analysis" / "timeseries_storage_prototype"
    assert (prototype / "shadow_maps.tif").is_file()
    assert (prototype / "light_curves_chunk16.h5").is_file()
    report = json.loads((prototype / "report.json").read_text(encoding="utf-8"))
    assert report["prototype"] == "two_file_timeseries_storage"
    assert report["shape"] == {"height": 256, "time": 64, "width": 256}


def test_downstream_product_example_is_public_and_has_working_help() -> None:
    source = (_EXAMPLES / "17_downstream_products.py").read_text(encoding="utf-8")
    assert "_numba_horizon" not in source
    assert "lunarscout.native" not in source

    completed = subprocess.run(
        [sys.executable, str(_EXAMPLES / "17_downstream_products.py"), "--help"],
        cwd=_REPOSITORY_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--backend {auto,cpu,cuda}" in completed.stdout
    for product in (
        "lightmap",
        "safe-havens",
        "mission-sun-earth-elevation",
    ):
        assert product in completed.stdout
