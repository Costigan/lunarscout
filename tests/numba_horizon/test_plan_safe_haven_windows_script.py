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


def test_the_stage_two_scripts_read_the_prefix_stage_one_writes() -> None:
    generate = importlib.util.spec_from_file_location(
        "run_safe_haven_stage1", SCRIPT.parent / "run_safe_haven_stage1.py")
    module = importlib.util.module_from_spec(generate)
    generate.loader.exec_module(module)
    assert module.NAME_PREFIX != "safe_haven"                  # never overwrites the earlier run
    for other in (load_script(), load_run_script(), load_list_script()):
        assert other.STAGE1_PREFIX == module.NAME_PREFIX


def test_importing_the_script_starts_nothing() -> None:
    module = load_script()
    assert module.PLAN_PATH.name == "safe_haven_2h_window_plan.jsonl"
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
    monkeypatch.setattr(module, "STAGE1_PREFIX", "safe_haven")      # the scene's default prefix
    monkeypatch.setattr(module, "PLAN_PATH", plan)
    monkeypatch.setattr(module, "START_UTC", scene.START)
    monkeypatch.setattr(module, "STOP_UTC", scene.STOP)
    monkeypatch.setattr(module, "STEP_HOURS", scene.STEP_HOURS)
    monkeypatch.setattr(module, "HOURS_PER_COUNT", None)
    monkeypatch.setattr(module, "STAGE1_NODATA", None)
    monkeypatch.setattr(module, "PROGRESS_EVERY_CELLS", 1)
    monkeypatch.setattr(module, "spice_vector_source", lambda body: vectors[body])
    monkeypatch.setattr(module, "PARAMETERS", WindowParameters(
        cell_patches=1, flood_step_minutes=30, moves_per_step=1, haven_neighbourhood_m=3.0, seed_separation_m=4.0,
    ))

    assert module.main() == 0

    output = capsys.readouterr().out
    assert "Pass 1: 1/1 cells" in output
    assert '"cell_months_kept": 3' in output and f"Plan written to: {plan}" in output
    assert [json.loads(line)["month"] for line in plan.read_text().splitlines()] == [0, 1, 2]


LIST_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "list_safe_haven_cells.py"


def load_list_script():
    spec = importlib.util.spec_from_file_location("list_safe_haven_cells", LIST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_cell_listing_script_prints_cells_and_a_cells_line(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_list_script()
    assert module.AROUND_NORTHING_EASTING_M == (117030.0, 71534.0)
    dem_path, _horizons_path, products, _stamps = scene.stage_one(tmp_path)
    monkeypatch.setattr(module, "DEM_PATH", dem_path)
    monkeypatch.setattr(module, "STAGE1_DIRECTORY", products.hibernation.parent)
    monkeypatch.setattr(module, "STAGE1_PREFIX", "safe_haven")
    monkeypatch.setattr(module, "HOURS_PER_COUNT", None)
    monkeypatch.setattr(module, "STAGE1_NODATA", None)
    monkeypatch.setattr(module, "CELL_SIZE", 10)
    monkeypatch.setattr(module, "AROUND_NORTHING_EASTING_M", (-1005.0, 1005.0))    # pixel (5, 5) of the scene
    monkeypatch.setattr(module, "AROUND_RADIUS_CELLS", 0)

    assert module.main() == 0

    output = capsys.readouterr().out
    assert "is in cell (0, 0); scanning 1 cells." in output
    assert "1 cells have a haven pixel at or under 72 h." in output
    assert "CELLS = [(0, 0)]" in output
    monkeypatch.setattr(module, "AROUND_NORTHING_EASTING_M", None)
    assert module.main() == 0
    # Scanning every cell: April makes every pixel a haven, so all four 10 x 10 cells count.
    assert "4 cells have a haven pixel at or under 72 h." in capsys.readouterr().out


RUN_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven_stage3.py"


def load_run_script():
    spec = importlib.util.spec_from_file_location("run_safe_haven_stage3", RUN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_stage_three_script_defaults_and_importing_it_starts_nothing() -> None:
    module = load_run_script()
    assert module.OUTPUT_PATH.name == "safe_haven_operating_opportunity_scores.csv" and module.OVERWRITE is False
    assert module.REPORT_PATH is None and module._report_path().name == "safe_haven_operating_opportunity_scores.report.json"
    assert module.PARAMETERS.block_patches % module.PARAMETERS.cell_patches == 0
    assert (module.PLAN_WORKERS, module.LIGHT_WORKERS, module.NUMBA_THREADS_PER_LIGHT_WORKER) == (4, 12, 2)
    assert module.FLOOD_WORKERS == 2 and module.STORE_BUDGET_GB == 8.0 and module.CELLS_IN_FLIGHT == 24
    assert (module.LIGHTING, module.GPU_DEVICE_ID, module.GPU_LIGHT_WORKERS, module.GPU_CHUNKS_PER_BATCH) == ("auto", 0, 6, 4)
    assert module.PARAMETERS.survival_limit_hours == 72.0 and module.PARAMETERS.operational_hours == 48.0
    assert module.CELLS is None and module.CELL_LIST_PATH.name == "safe_haven_2h_haven_cells.json"


def test_the_stage_three_script_writes_the_table_and_resumes(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_run_script()
    dem_path, horizons_path, products, _stamps = scene.stage_one(tmp_path)
    output = tmp_path / "windows.csv"
    vectors = {"sun": scene.sun_vectors, "earth": scene.earth_vectors}
    for name, value in dict(
        DEM_PATH=dem_path, HORIZONS_PATH=horizons_path, STAGE1_DIRECTORY=products.hibernation.parent,
        STAGE1_PREFIX="safe_haven", OUTPUT_PATH=output, START_UTC=scene.START, STOP_UTC=scene.STOP, STEP_HOURS=scene.STEP_HOURS,
        HOURS_PER_COUNT=None, STAGE1_NODATA=None, CELLS=[(0, 0)],
    ).items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "spice_vector_source", lambda body: vectors[body])
    monkeypatch.setattr(module, "PARAMETERS", WindowParameters(
        cell_patches=1, block_patches=1, flood_step_minutes=30, moves_per_step=1, haven_neighbourhood_m=3.0,
        seed_separation_m=4.0, operational_hours=24.0,
    ))

    assert module.main() == 0

    printed = capsys.readouterr().out
    assert "Stage 3: 0/1 blocks already in the checkpoint." in printed and "Stage 3: 1/1 blocks" in printed
    assert f"Table written to: {output}" in printed and '"rows":' in printed
    lines = output.read_text().splitlines()
    assert lines[0].startswith("cell_row,cell_col,") and len(lines) > 1
    # a second run refuses to replace the table unless OVERWRITE is set
    with pytest.raises(ValueError, match="exists"):
        module.main()
    monkeypatch.setattr(module, "OVERWRITE", True)
    assert module.main() == 0
    assert output.read_text().splitlines() == lines


def stage_three_setup(module, monkeypatch, tmp_path, cells):
    dem_path, horizons_path, products, _stamps = scene.stage_one(tmp_path)
    vectors = {"sun": scene.sun_vectors, "earth": scene.earth_vectors}
    for name, value in dict(
        DEM_PATH=dem_path, HORIZONS_PATH=horizons_path, STAGE1_DIRECTORY=products.hibernation.parent,
        STAGE1_PREFIX="safe_haven", OUTPUT_PATH=tmp_path / "windows.csv", CELL_LIST_PATH=tmp_path / "cells.json",
        START_UTC=scene.START, STOP_UTC=scene.STOP, STEP_HOURS=scene.STEP_HOURS, HOURS_PER_COUNT=None,
        STAGE1_NODATA=None, CELLS=cells,
    ).items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "spice_vector_source", lambda body: vectors[body])
    monkeypatch.setattr(module, "PARAMETERS", WindowParameters(
        cell_patches=1, block_patches=1, flood_step_minutes=30, moves_per_step=1, haven_neighbourhood_m=3.0,
        seed_separation_m=4.0, operational_hours=24.0,
    ))
    return products


def test_stage_three_reads_the_stage_two_list_and_refuses_a_missing_or_stale_one(
        tmp_path: Path, monkeypatch, capsys) -> None:
    import os

    from lunarscout._numba_horizon import safe_haven_cells as stage2

    module = load_run_script()
    products = stage_three_setup(module, monkeypatch, tmp_path, cells=None)
    with pytest.raises(ValueError, match="run scripts/run_safe_haven_stage2.py first"):
        module.main()
    stage1 = module.Stage1Rasters(products.hibernation, products.joint_total)
    found = stage2.scan_cells(stage1, survival_limit_hours=72.0, cell_size=128)
    stage2.write_cell_list(tmp_path / "cells.json", found, stage1, survival_limit_hours=72.0, cell_size=128)

    assert module.main() == 0

    assert f"Stage 3: {len(found)} cells from {tmp_path / 'cells.json'}" in capsys.readouterr().out
    assert (tmp_path / "windows.csv").read_text().startswith("cell_row,cell_col,")
    monkeypatch.setattr(module, "OVERWRITE", True)
    stat = products.hibernation.stat()
    os.utime(products.hibernation, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))   # stage 1 rewritten
    with pytest.raises(ValueError, match="run stage 2 again"):
        module.main()
    os.utime(products.hibernation, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    monkeypatch.setattr(module, "PARAMETERS", WindowParameters(
        cell_patches=1, block_patches=1, flood_step_minutes=30, moves_per_step=1, survival_limit_hours=48.0,
        haven_neighbourhood_m=3.0, seed_separation_m=4.0, operational_hours=24.0))
    with pytest.raises(ValueError, match="survival_limit_hours"):
        module.main()


def test_stage_three_needs_no_list_when_cells_are_given(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_run_script()
    stage_three_setup(module, monkeypatch, tmp_path, cells=[(0, 0)])
    assert module.main() == 0
    assert "cells from" not in capsys.readouterr().out and not (tmp_path / "cells.json").exists()


def test_stage_three_passes_the_listed_cells_on_in_order_and_explicit_cells_win(
        tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from lunarscout._numba_horizon import safe_haven_cells as stage2

    module = load_run_script()
    products = stage_three_setup(module, monkeypatch, tmp_path, cells=None)
    stage1 = module.Stage1Rasters(products.hibernation, products.joint_total)
    listed = [dict(cell_row=3, cell_col=5, months_with_haven=2, haven_pixels=9),
              dict(cell_row=1, cell_col=2, months_with_haven=1, haven_pixels=4)]
    stage2.write_cell_list(tmp_path / "cells.json", listed, stage1, survival_limit_hours=72.0, cell_size=128)
    seen = []
    monkeypatch.setattr(module, "run_pass_two", lambda **kwargs: seen.append(kwargs["cells"])
                        or SimpleNamespace(as_dict=lambda: {}))
    assert module.main() == 0
    assert seen == [[(3, 5), (1, 2)]]
    monkeypatch.setattr(module, "CELLS", [(7, 7)])
    assert module.main() == 0 and seen[-1] == [(7, 7)]


# --- the scripts log their events to the console and a plain log file ---------------------------------


def test_the_scripts_log_to_a_file_next_to_their_output_and_clean_up_after_themselves(
        tmp_path: Path, monkeypatch, capsys) -> None:
    import logging

    module = load_run_script()
    assert (module.LOG_PATH, module.COLOR, module.LOG_LEVEL) == (None, "auto", "INFO")
    stage_three_setup(module, monkeypatch, tmp_path, cells=[(0, 0)])
    assert module.main() == 0
    log = (tmp_path / "windows.csv.log").read_text()
    assert "INFO    lunarscout.scripts.stage3: Stage 3: 0/1 blocks already in the checkpoint." in log
    assert "Stage 3: 1/1 blocks" in log and f"Table written to: {tmp_path / 'windows.csv'}" in log
    assert f"Report written to: {tmp_path / 'windows.report.json'}" in log
    assert "\x1b[" not in log
    out = capsys.readouterr().out
    assert "Stage 3: 1/1 blocks" in out and '"rows":' in out                      # the JSON report is still printed
    assert not [h for h in logging.getLogger("lunarscout").handlers if getattr(h, "_lunarscout_script_handler", False)]
    monkeypatch.setattr(module, "LOG_PATH", tmp_path / "elsewhere.log")
    monkeypatch.setattr(module, "OVERWRITE", True)
    assert module.main() == 0
    assert "Table written to" in (tmp_path / "elsewhere.log").read_text()
    assert (tmp_path / "windows.csv.log").read_text() == log                        # the default file was not touched


def test_the_script_passes_its_color_and_level_settings_on(tmp_path: Path, monkeypatch, capsys) -> None:
    pytest.importorskip("rich")
    module = load_run_script()
    stage_three_setup(module, monkeypatch, tmp_path, cells=[(0, 0)])
    monkeypatch.setattr(module, "COLOR", "always")
    monkeypatch.setattr(module, "LOG_LEVEL", "WARNING")
    assert module.main() == 0
    log = (tmp_path / "windows.csv.log").read_text()
    assert "Stage 3: 1/1 blocks" not in log and "\x1b[" not in log                  # INFO is below WARNING
    monkeypatch.setattr(module, "LOG_LEVEL", "INFO")
    monkeypatch.setattr(module, "OVERWRITE", True)
    capsys.readouterr()
    assert module.main() == 0
    assert "\x1b[" in capsys.readouterr().out                                       # rich was forced into the pipe
    assert "\x1b[" not in (tmp_path / "windows.csv.log").read_text()


def test_a_failing_script_logs_the_error_and_still_raises(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_run_script()
    stage_three_setup(module, monkeypatch, tmp_path, cells=None)                    # no stage-2 list exists
    with pytest.raises(ValueError, match="run scripts/run_safe_haven_stage2.py first"):
        module.main()
    log = (tmp_path / "windows.csv.log").read_text()
    assert "ERROR   lunarscout.scripts.stage3: Stage 3 stopped: ValueError: " in log
    assert "ERROR   Stage 3 stopped" in capsys.readouterr().out


def test_the_other_scripts_log_the_same_way(tmp_path: Path, monkeypatch, capsys) -> None:
    for script in (load_script(), load_run_script(), load_stage_two_script()):
        assert (script.LOG_PATH, script.COLOR, script.LOG_LEVEL) == (None, "auto", "INFO")
    # pass 1 on the scene
    module = load_script()
    dem_path, horizons_path, products, _stamps = scene.stage_one(tmp_path)
    vectors = {"sun": scene.sun_vectors, "earth": scene.earth_vectors}
    for name, value in dict(
        DEM_PATH=dem_path, HORIZONS_PATH=horizons_path, STAGE1_DIRECTORY=products.hibernation.parent,
        STAGE1_PREFIX="safe_haven", PLAN_PATH=tmp_path / "plan.jsonl", START_UTC=scene.START,
        STOP_UTC=scene.STOP, STEP_HOURS=scene.STEP_HOURS, HOURS_PER_COUNT=None, STAGE1_NODATA=None,
        PROGRESS_EVERY_CELLS=1,
    ).items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "spice_vector_source", lambda body: vectors[body])
    monkeypatch.setattr(module, "PARAMETERS", WindowParameters(
        cell_patches=1, flood_step_minutes=30, moves_per_step=1, haven_neighbourhood_m=3.0, seed_separation_m=4.0))
    assert module.main() == 0
    log = (tmp_path / "plan.jsonl.log").read_text()
    assert "lunarscout.scripts.plan: Pass 1: 1/1 cells" in log and "Plan written to" in log


def load_stage_two_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven_stage2.py"
    spec = importlib.util.spec_from_file_location("run_safe_haven_stage2_logging", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage_three_passes_its_pipeline_settings_on_and_reports_the_memory_it_expects(
        tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    module = load_run_script()
    stage_three_setup(module, monkeypatch, tmp_path, cells=[(0, 0)])
    seen = {}
    monkeypatch.setattr(module, "run_pass_two", lambda **kwargs: seen.update(kwargs)
                        or SimpleNamespace(as_dict=lambda: {}))
    for name, value in dict(PLAN_WORKERS=3, LIGHT_WORKERS=5, NUMBA_THREADS_PER_LIGHT_WORKER=4, FLOOD_WORKERS=2,
                            DECODE_CACHE_PATCHES=7, CELLS_IN_FLIGHT=9, STORE_BUDGET_GB=1.5,
                            STATUS_SECONDS=11.0, LIGHTING="cuda", GPU_DEVICE_ID=1, GPU_LIGHT_WORKERS=3,
                            GPU_CHUNKS_PER_BATCH=2).items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 10**12)
    assert module.main() == 0
    settings = seen["pipeline"]
    assert (settings.plan_workers, settings.light_workers, settings.numba_threads_per_light_worker,
            settings.flood_workers, settings.decode_cache_patches, settings.cells_in_flight,
            settings.store_budget_bytes, settings.status_seconds) == (3, 5, 4, 2, 7, 9, int(1.5 * 2**30), 11.0)
    assert (settings.lighting, settings.gpu_device_id, settings.gpu_light_workers,
            settings.gpu_chunks_per_batch) == ("cuda", 1, 3, 2)
    assert seen["workers"] if "workers" in seen else True
    log = (tmp_path / "windows.csv.log").read_text()
    assert "7 cached patches, 2 GB of units, 3 planning processes" in log and "WARNING" not in log


def test_stage_three_warns_when_the_pipeline_will_not_fit_in_memory(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    module = load_run_script()
    stage_three_setup(module, monkeypatch, tmp_path, cells=[(0, 0)])
    monkeypatch.setattr(module, "run_pass_two", lambda **kwargs: SimpleNamespace(as_dict=lambda: {}))
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 4 * 10**9)
    assert module.main() == 0
    log = (tmp_path / "windows.csv.log").read_text()
    assert "WARNING" in log and "lower STORE_BUDGET_GB" in log
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: None)             # unreadable: no warning, no crash
    monkeypatch.setattr(module, "OVERWRITE", True)
    assert module.main() == 0


# --- the scripts log their events to the console and a plain log file ---------------------------------


def test_the_scripts_log_to_a_file_next_to_their_output_and_clean_up_after_themselves(
        tmp_path: Path, monkeypatch, capsys) -> None:
    import logging

    module = load_run_script()
    assert (module.LOG_PATH, module.COLOR, module.LOG_LEVEL) == (None, "auto", "INFO")
    stage_three_setup(module, monkeypatch, tmp_path, cells=[(0, 0)])
    assert module.main() == 0
    log = (tmp_path / "windows.csv.log").read_text()
    assert "INFO    lunarscout.scripts.stage3: Stage 3: 0/1 blocks already in the checkpoint." in log
    assert "Stage 3: 1/1 blocks" in log and f"Table written to: {tmp_path / 'windows.csv'}" in log
    assert "\x1b[" not in log
    out = capsys.readouterr().out
    assert "Stage 3: 1/1 blocks" in out and '"rows":' in out                      # the JSON report is still printed
    assert not [h for h in logging.getLogger("lunarscout").handlers if getattr(h, "_lunarscout_script_handler", False)]
    monkeypatch.setattr(module, "LOG_PATH", tmp_path / "elsewhere.log")
    monkeypatch.setattr(module, "OVERWRITE", True)
    assert module.main() == 0
    assert "Table written to" in (tmp_path / "elsewhere.log").read_text()
    assert (tmp_path / "windows.csv.log").read_text() == log                        # the default file was not touched


def test_the_script_passes_its_color_and_level_settings_on(tmp_path: Path, monkeypatch, capsys) -> None:
    pytest.importorskip("rich")
    module = load_run_script()
    stage_three_setup(module, monkeypatch, tmp_path, cells=[(0, 0)])
    monkeypatch.setattr(module, "COLOR", "always")
    monkeypatch.setattr(module, "LOG_LEVEL", "WARNING")
    assert module.main() == 0
    log = (tmp_path / "windows.csv.log").read_text()
    assert "Stage 3: 1/1 blocks" not in log and "\x1b[" not in log                  # INFO is below WARNING
    monkeypatch.setattr(module, "LOG_LEVEL", "INFO")
    monkeypatch.setattr(module, "OVERWRITE", True)
    capsys.readouterr()
    assert module.main() == 0
    assert "\x1b[" in capsys.readouterr().out                                       # rich was forced into the pipe
    assert "\x1b[" not in (tmp_path / "windows.csv.log").read_text()


def test_a_failing_script_logs_the_error_and_still_raises(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_run_script()
    stage_three_setup(module, monkeypatch, tmp_path, cells=None)                    # no stage-2 list exists
    with pytest.raises(ValueError, match="run scripts/run_safe_haven_stage2.py first"):
        module.main()
    log = (tmp_path / "windows.csv.log").read_text()
    assert "ERROR   lunarscout.scripts.stage3: Stage 3 stopped: ValueError: " in log
    assert "ERROR   Stage 3 stopped" in capsys.readouterr().out


def test_the_other_scripts_log_the_same_way(tmp_path: Path, monkeypatch, capsys) -> None:
    for script in (load_script(), load_run_script(), load_stage_two_script()):
        assert (script.LOG_PATH, script.COLOR, script.LOG_LEVEL) == (None, "auto", "INFO")
    # pass 1 on the scene
    module = load_script()
    dem_path, horizons_path, products, _stamps = scene.stage_one(tmp_path)
    vectors = {"sun": scene.sun_vectors, "earth": scene.earth_vectors}
    for name, value in dict(
        DEM_PATH=dem_path, HORIZONS_PATH=horizons_path, STAGE1_DIRECTORY=products.hibernation.parent,
        STAGE1_PREFIX="safe_haven", PLAN_PATH=tmp_path / "plan.jsonl", START_UTC=scene.START,
        STOP_UTC=scene.STOP, STEP_HOURS=scene.STEP_HOURS, HOURS_PER_COUNT=None, STAGE1_NODATA=None,
        PROGRESS_EVERY_CELLS=1,
    ).items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "spice_vector_source", lambda body: vectors[body])
    monkeypatch.setattr(module, "PARAMETERS", WindowParameters(
        cell_patches=1, flood_step_minutes=30, moves_per_step=1, haven_neighbourhood_m=3.0, seed_separation_m=4.0))
    assert module.main() == 0
    log = (tmp_path / "plan.jsonl.log").read_text()
    assert "lunarscout.scripts.plan: Pass 1: 1/1 cells" in log and "Plan written to" in log


def load_stage_two_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven_stage2.py"
    spec = importlib.util.spec_from_file_location("run_safe_haven_stage2_logging", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
