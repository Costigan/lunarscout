"""scripts/run_light_store.py builds the thresholded-lighting store end to end on a synthetic scene."""

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


def test_the_script_uses_the_stage_one_thresholds_and_the_stage_two_inputs() -> None:
    stage0, stage1, stage2 = load("run_light_store"), load("run_safe_haven_stage1"), load("run_safe_haven_stage2")
    assert stage0.SHADOW_FRACTION == stage1.SUNLIGHT_FRACTION_THRESHOLD
    assert stage0.OPERATING_FRACTION == stage1.OPERATING_SUNLIGHT_FRACTION
    assert stage0.EARTH_ELEVATION_DEG == stage1.EARTH_ELEVATION_THRESHOLD_DEG
    assert stage0.DEM_PATH == stage2.DEM_PATH and stage2.LIGHT_STORE_PATH == stage0.OUTPUT_DIRECTORY
    assert stage0.STEP_MINUTES == 120 and stage0.CODEC == "zstd-3" and stage0.STEPS_PER_CHUNK == 96
    assert stage0.PATCHES is None and stage0.LIGHTING == "auto"


def test_the_stage_one_script_reads_the_store_the_stage_zero_script_writes() -> None:
    stage0, stage1 = load("run_light_store"), load("run_safe_haven_stage1")
    assert stage1.LIGHT_STORE_PATH == stage0.OUTPUT_DIRECTORY and stage1.DEM_PATH == stage0.DEM_PATH
    assert stage1.STEP_HOURS * 60 == stage0.STEP_MINUTES
    assert stage1.OBSERVER_HEIGHT_M == stage0.OBSERVER_HEIGHT_M
    # The samples are the store's grid: from its first step to the step before its stop.
    first = np.datetime64(stage1.START_UTC.rstrip("Z"))
    last = np.datetime64(stage1.STOP_UTC.rstrip("Z"))
    assert first == np.datetime64(stage0.START_UTC)
    assert last == np.datetime64(stage0.STOP_UTC) - np.timedelta64(stage0.STEP_MINUTES, "m")
    # The windows are in hours, so they scale with the step: 12 samples each at two hours.
    assert stage1.MERGE_GAP_HOURS / stage1.STEP_HOURS == 12 and stage1.DEGENERATE_OUTAGE_WINDOW_HOURS / stage1.STEP_HOURS == 12


def test_the_script_writes_a_readable_store(tmp_path: Path, monkeypatch) -> None:
    module = load("run_light_store")
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
    module = load("run_light_store")
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
