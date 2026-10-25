"""scripts/run_safe_haven_stage0.py builds the thresholded-lighting store end to end on a synthetic scene."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import blocks_scene as bs
import windows_scene as scene
from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_uses_the_stage_one_thresholds_and_the_stage_three_inputs() -> None:
    stage0, stage1, stage3 = load("run_safe_haven_stage0"), load("run_generate_safe_havens"), load("run_safe_haven_stage3")
    assert stage0.SHADOW_FRACTION == stage1.SUNLIGHT_FRACTION_THRESHOLD
    assert stage0.OPERATING_FRACTION == stage1.OPERATING_SUNLIGHT_FRACTION
    assert stage0.EARTH_ELEVATION_DEG == stage1.EARTH_ELEVATION_THRESHOLD_DEG
    assert (stage0.DEM_PATH, stage0.HORIZONS_PATH) == (stage3.DEM_PATH, stage3.HORIZONS_PATH)
    assert stage0.STEP_MINUTES == 120 and stage0.CODEC == "zstd-3" and stage0.STEPS_PER_CHUNK == 96
    assert stage0.PATCHES is None and stage0.LIGHTING == "auto"


def test_the_script_writes_a_readable_store(tmp_path: Path, monkeypatch) -> None:
    module = load("run_safe_haven_stage0")
    dem_path, horizons_path = bs.write_inputs(tmp_path)
    out = tmp_path / "store"
    vectors = {"sun": scene.sun_vectors, "earth": scene.earth_vectors}
    monkeypatch.setattr(module, "DEM_PATH", dem_path)
    monkeypatch.setattr(module, "HORIZONS_PATH", horizons_path)
    monkeypatch.setattr(module, "OUTPUT_DIRECTORY", out)
    monkeypatch.setattr(module, "START_UTC", "2027-01-10T00:00:00")
    monkeypatch.setattr(module, "STOP_UTC", "2027-01-20T00:00:00")
    monkeypatch.setattr(module, "STEPS_PER_CHUNK", 48)
    monkeypatch.setattr(module, "CODEC", "zlib-1")
    monkeypatch.setattr(module, "LIGHTING", "cpu")
    monkeypatch.setattr(module, "WORKERS", 2)
    monkeypatch.setattr(module, "PATCHES", [(0, 0), (2, 1)])
    monkeypatch.setattr(module, "spice_vector_source", lambda body: vectors[body])

    assert module.main() == 0

    store = LightStoreDirectory(out)
    assert (store.step_count, store.settings.codec, store.settings.step_minutes) == (120, "zlib-1", 120)
    assert store.open_patch(1, 1) is None
    for patch in [(0, 0), (2, 1)]:
        with store.open_patch(*patch) as opened:
            planes = opened.read_steps(store.first_step, store.stop_step)
            assert planes.shape == (3, 120, 128, 2) and planes.any()
    assert (tmp_path / "store.log").read_text().count("stage 0 completed") == 1
    assert module.main() == 0                                           # a rerun finds everything stored
    assert (tmp_path / "store.log").read_text().count("2 already stored") == 1


def test_an_interrupted_run_returns_nonzero(tmp_path: Path, monkeypatch) -> None:
    module = load("run_safe_haven_stage0")
    dem_path, horizons_path = bs.write_inputs(tmp_path)
    out = tmp_path / "store"
    vectors = {"sun": scene.sun_vectors, "earth": scene.earth_vectors}
    for name, value in dict(DEM_PATH=dem_path, HORIZONS_PATH=horizons_path, OUTPUT_DIRECTORY=out,
                            START_UTC="2027-01-10T00:00:00", STOP_UTC="2027-01-12T00:00:00", STEPS_PER_CHUNK=12,
                            CODEC="zlib-1", LIGHTING="cpu", WORKERS=1).items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "spice_vector_source", lambda body: vectors[body])
    real = module.generate_light_store

    def interrupted(*args, **kwargs):
        report = real(*args, **kwargs)
        report.status = "interrupted"
        return report

    monkeypatch.setattr(module, "generate_light_store", interrupted)
    assert module.main() == 1
    assert "run this script again" in (tmp_path / "store.log").read_text()
