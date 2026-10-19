"""scripts/plan_safe_haven_windows.py runs pass 1 end to end on the synthetic scene."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import windows_scene as scene
from lunarscout._numba_horizon.safe_haven_windows import WindowParameters

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "plan_safe_haven_windows.py"


def load_script():
    spec = importlib.util.spec_from_file_location("plan_safe_haven_windows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_importing_the_script_starts_nothing() -> None:
    module = load_script()
    assert module.PLAN_PATH.name == "window_plan.jsonl"
    assert module.PARAMETERS.survival_limit_hours == 72.0
    assert module.PARAMETERS.max_seeds_per_cell_month == 5


def test_the_script_plans_the_scene_and_prints_the_report(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_script()
    dem_path, horizons_path, products, _stamps = scene.stage_one(tmp_path)
    plan = tmp_path / "plan.jsonl"
    vectors = {"sun": scene.sun_vectors, "earth": scene.earth_vectors}
    monkeypatch.setattr(module, "DEM_PATH", dem_path)
    monkeypatch.setattr(module, "HORIZONS_PATH", horizons_path)
    monkeypatch.setattr(module, "STAGE1_DIRECTORY", products.hibernation.parent)
    monkeypatch.setattr(module, "PLAN_PATH", plan)
    monkeypatch.setattr(module, "START_UTC", scene.START)
    monkeypatch.setattr(module, "STOP_UTC", scene.STOP)
    monkeypatch.setattr(module, "STEP_HOURS", scene.STEP_HOURS)
    monkeypatch.setattr(module, "HOURS_PER_COUNT", None)
    monkeypatch.setattr(module, "STAGE1_NODATA", None)
    monkeypatch.setattr(module, "PROGRESS_EVERY_CELLS", 1)
    monkeypatch.setattr(module, "spice_vector_source", lambda body: vectors[body])
    monkeypatch.setattr(module, "PARAMETERS", WindowParameters(
        cell_patches=1, rover_speed_m_s=1.0 / 1800.0, haven_neighbourhood_m=3.0, seed_separation_m=4.0,
    ))

    assert module.main() == 0

    output = capsys.readouterr().out
    assert "Pass 1: 1/1 cells" in output
    assert '"cell_months_kept": 3' in output and f"Plan written to: {plan}" in output
    assert [json.loads(line)["month"] for line in plan.read_text().splitlines()] == [0, 1, 2]
