"""scripts/run_safe_haven.py: the sequence on the synthetic scene, and what it skips because it is up to date."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import pytest

import blocks_scene as bs
import lunarscout as ls
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_chains as chains
from lunarscout.products import _load_dem

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven.py"
STEPS = ("psr-regions", "light-store", "stage1", "stage2", "stage3", "stage4", "chain-maps", "yearly-chain-maps",
         "haven-months", "seed-haven-areas", "monthly-chain-maps", "overview-maps", "quality-maps", "cluster-maps")


def load_driver():
    spec = importlib.util.spec_from_file_location("run_safe_haven", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = load_driver()


# --- the decisions, on small fake steps --------------------------------------------------------------------

def make_step(tmp_path, *, inputs=("in.txt",), outputs=("out.txt",), caches=(), **options):
    for name in inputs:
        (tmp_path / name).write_text("x")
    return driver.Step("fake", "run_light_store", "run_light_store", {"a": 1, **options},
                       [tmp_path / n for n in inputs], [tmp_path / n for n in outputs], [tmp_path / n for n in caches])


def set_time(path, seconds):
    os.utime(path, (seconds, seconds))


def record(step):
    return {"fake": {"fingerprint": driver.fingerprint(step), "parameters": driver.parameters_of(step)}}


def test_a_step_with_no_outputs_is_stale_and_one_with_a_missing_input_is_blocked(tmp_path) -> None:
    step = make_step(tmp_path)
    assert driver.status_of(step, {}) == driver.Status("stale", "no outputs yet")
    blocked = driver.Step("fake", "x", "x", {}, [tmp_path / "absent"], [tmp_path / "out.txt"], [])
    assert driver.status_of(blocked, {}).kind == "blocked"


def test_outputs_this_driver_never_recorded_are_unrecorded_not_current(tmp_path) -> None:
    step = make_step(tmp_path)
    (tmp_path / "out.txt").write_text("y")
    set_time(tmp_path / "in.txt", 1000)
    set_time(tmp_path / "out.txt", 2000)
    assert driver.status_of(step, {}).kind == "unrecorded"
    assert driver.status_of(step, record(step)) == driver.Status("current", "up to date")


def test_a_newer_input_a_changed_parameter_or_a_missing_output_makes_a_step_stale(tmp_path) -> None:
    step = make_step(tmp_path, caches=("cache.bin",))
    (tmp_path / "out.txt").write_text("y")
    (tmp_path / "cache.bin").write_text("c")
    set_time(tmp_path / "in.txt", 1000)
    set_time(tmp_path / "out.txt", 2000)
    state = record(step)
    assert driver.status_of(step, state).kind == "current"
    set_time(tmp_path / "in.txt", 3000)
    stale = driver.status_of(step, state)
    assert stale.kind == "stale" and "input newer" in stale.reason and "in.txt" in stale.reason
    set_time(tmp_path / "in.txt", 1000)
    step.options["a"] = 2
    changed = driver.status_of(step, state)
    assert changed.kind == "stale" and changed.reason == "parameters changed: a"
    step.options["a"] = 1
    (tmp_path / "cache.bin").unlink()
    assert driver.status_of(step, state) == driver.Status("stale", "an output is missing")


def test_the_oldest_output_is_compared_with_the_newest_input_and_a_cache_is_not_compared(tmp_path) -> None:
    step = make_step(tmp_path, inputs=("a", "b"), outputs=("o1", "o2"), caches=("cache",))
    for name in ("o1", "o2", "cache"):
        (tmp_path / name).write_text("y")
    set_time(tmp_path / "a", 1000)
    set_time(tmp_path / "b", 1500)
    set_time(tmp_path / "o1", 2000)
    set_time(tmp_path / "o2", 1400)               # older than input b: the step has to run
    set_time(tmp_path / "cache", 10)              # a cache kept for the step is old on purpose
    assert driver.status_of(step, record(step)).kind == "stale"
    set_time(tmp_path / "o2", 1600)
    assert driver.status_of(step, record(step)).kind == "current"


def test_the_time_of_a_directory_is_the_newest_file_in_it(tmp_path) -> None:
    (tmp_path / "d" / "sub").mkdir(parents=True)
    old, new = tmp_path / "d" / "a", tmp_path / "d" / "sub" / "b"
    old.write_text("1")
    new.write_text("2")
    set_time(old, 1000)
    set_time(new, 5000)
    assert driver.stamp(tmp_path / "d") == 5000 and driver.stamp(old) == 1000 and driver.stamp(tmp_path / "none") is None


def test_an_input_another_step_will_rebuild_makes_the_step_stale_in_a_dry_run(tmp_path) -> None:
    step = make_step(tmp_path)
    (tmp_path / "out.txt").write_text("y")
    set_time(tmp_path / "in.txt", 1000)
    set_time(tmp_path / "out.txt", 2000)
    pending = frozenset({str(tmp_path / "in.txt")})
    assert driver.status_of(step, record(step), pending).kind == "stale"


def test_the_fingerprint_covers_what_changes_a_result_and_not_how_it_runs(tmp_path) -> None:
    from lunarscout._numba_horizon.safe_haven_floods import FloodSettings

    base = make_step(tmp_path, settings=FloodSettings(max_slope_deg=15.0), flood_workers=16, overwrite=False)
    same = make_step(tmp_path, settings=FloodSettings(max_slope_deg=15.0, unit_cache_mib=64), flood_workers=2, overwrite=True)
    assert driver.fingerprint(base) == driver.fingerprint(same)
    slope = make_step(tmp_path, settings=FloodSettings(max_slope_deg=20.0), flood_workers=16)
    assert driver.fingerprint(slope) != driver.fingerprint(base)
    assert driver.parameters_of(base)["settings"]["max_slope_deg"] == 15.0 and "unit_cache_mib" not in driver.parameters_of(base)["settings"]


# --- the plan --------------------------------------------------------------------------------------------

def test_the_last_sample_is_the_step_before_the_exclusive_stop() -> None:
    assert driver.last_sample("2032-01-01T00:00:00", 120) == "2031-12-31T22:00:00Z"
    assert driver.last_sample("2027-04-01T00:00:00", 30) == "2027-03-31T23:30:00Z"


def test_paths_default_to_places_inside_the_scenario_directory(tmp_path) -> None:
    p = driver.resolve_paths({**driver.default_options(), "scenario_directory": tmp_path})
    assert p["light_store_path"] == tmp_path / "light_store" and p["output_directory"] == tmp_path / "safe_havens"
    assert p["psr_ids_path"] == tmp_path / "psr_ids.tif" and p["horizons_path"] == tmp_path / "lighting" / "horizons"
    assert p["state_path"] == tmp_path / "safe_havens" / "safe_haven_run_state.json"
    given = driver.resolve_paths({**driver.default_options(), "scenario_directory": tmp_path, "state_path": Path("/s.json")})
    assert given["state_path"] == Path("/s.json")


def test_the_steps_share_one_set_of_parameters() -> None:
    p = driver.resolve_paths(driver.default_options())
    steps = driver.build_steps(p)
    assert tuple(steps) == STEPS
    assert steps["stage1"].options["max_slope_deg"] == 15.0
    for name in ("stage3", "stage4", "chain-maps", "yearly-chain-maps", "monthly-chain-maps"):
        settings = steps[name].options.get("settings") or steps[name].options["flood_settings"]
        assert settings == steps["stage3"].options["settings"] and settings.max_slope_deg == 15.0
    assert steps["stage1"].options["stop_utc"] == "2031-12-31T22:00:00Z" == steps["stage2"].options["stop_utc"]
    assert steps["light-store"].options["stop_utc"] == "2032-01-01T00:00:00"
    assert steps["stage2"].options["parameters"].flood_step_minutes == 120 and steps["stage2"].options["parameters"].forward_only
    assert steps["stage1"].options["name_prefix"] == steps["stage2"].options["stage1_prefix"]
    # each step's outputs are the next ones' inputs
    assert set(map(str, steps["stage2"].outputs)) <= set(map(str, steps["stage3"].inputs))
    assert set(map(str, steps["stage3"].outputs)) <= set(map(str, steps["stage4"].inputs))
    assert set(map(str, steps["stage4"].outputs)) <= set(map(str, steps["chain-maps"].inputs))
    assert set(map(str, steps["stage4"].outputs)) <= set(map(str, steps["yearly-chain-maps"].inputs))
    assert set(map(str, steps["stage2"].outputs)) <= set(map(str, steps["overview-maps"].inputs))
    overview = [p.name for p in steps["overview-maps"].outputs]
    assert overview[:2] == ["safe_haven_stage_2_persistence_2024.tif", "safe_haven_stage_2_persistence_2024.png"]
    assert overview[-1] == "safe_haven_stage_2_persistence_contact_sheet.png" and len(overview) == 8 * 2 + 1
    assert steps["overview-maps"].options["years"] == list(range(2024, 2032)) and steps["overview-maps"].options["cell_patches"] == 2
    assert set(map(str, steps["stage4"].outputs)) <= set(map(str, steps["quality-maps"].inputs))
    quality = [p.name for p in steps["quality-maps"].outputs]
    assert quality[:2] == ["safe_haven_stage_4_cell_quality_2024.tif", "safe_haven_stage_4_cell_quality_2024.png"]
    assert quality[-1] == "safe_haven_stage_4_cell_quality_contact_sheet.png" and len(quality) == 8 * 2 + 1
    assert steps["quality-maps"].options["years"] == list(range(2024, 2032)) and steps["quality-maps"].options["cell_patches"] == 2
    assert set(map(str, steps["stage4"].outputs)) <= set(map(str, steps["cluster-maps"].inputs))
    assert set(map(str, steps["stage3"].outputs)) & set(map(str, steps["cluster-maps"].inputs))      # the edge table
    boxes = [p.name for p in steps["cluster-maps"].outputs]
    assert boxes[:2] == ["safe_haven_stage_4_cluster_boxes_2024.tif", "safe_haven_stage_4_cluster_boxes_2024.png"]
    assert boxes[-2:] == ["safe_haven_stage_4_cluster_boxes_contact_sheet.png", "safe_haven_stage_4_cluster_boxes.csv"]
    assert steps["cluster-maps"].options["years"] == list(range(2024, 2032)) and steps["cluster-maps"].options["connect"] == "chains"
    assert [p.name for p in steps["yearly-chain-maps"].outputs] == [
        f"safe_haven_stage_4_longest_chain_per_root_{year}.tif" for year in range(2024, 2032)]
    assert steps["yearly-chain-maps"].options["years"] == list(range(2024, 2032))
    assert set(map(str, steps["stage4"].outputs)) <= set(map(str, steps["monthly-chain-maps"].inputs))
    assert [p.name for p in steps["monthly-chain-maps"].outputs] == ["safe_haven_stage_4_monthly_chain_pixels.npz"]
    assert set(map(str, steps["stage3"].caches)) <= set(map(str, steps["stage4"].inputs))
    # every haven pixel of every month: from the stage-1 hibernation product alone, at the survival limit of the stage-2 seeds
    months = steps["haven-months"]
    assert [str(p) for p in months.inputs] == [str(steps["stage1"].outputs[0])]
    assert [p.name for p in months.outputs] == ["safe_haven_stage_1_haven_months"]
    assert months.options["survival_limit_hours"] == steps["stage2"].options["parameters"].survival_limit_hours \
        == driver.SURVIVAL_LIMIT_HOURS
    assert months.options["output_filename_template"] == "safe_haven_stage_1_haven_months_{timestamp:%Y-%m}.tif"
    assert fingerprint_differs(driver, "survival_limit_hours", 36.0, "haven-months")
    # the seed haven areas: the chain table, the DEM and the monthly haven images, at the driver's radius
    areas = steps["seed-haven-areas"]
    assert [p.name for p in areas.outputs] == ["safe_haven_stage_4_seed_haven_areas", "safe_haven_stage_4_seed_haven_areas.npz"]
    assert set(map(str, steps["stage4"].outputs)) <= set(map(str, areas.inputs)) and str(months.outputs[0]) in map(str, areas.inputs)
    assert areas.options["area_radius_meters"] == driver.AREA_RADIUS_METERS == 200.0
    assert areas.options["haven_months_directory"] == months.outputs[0]
    assert fingerprint_differs(driver, "area_radius_meters", 300.0, "seed-haven-areas")
    # the monthly chain pixels draw those areas: they run after them, read their file and are stale when it is rebuilt
    monthly = steps["monthly-chain-maps"]
    assert list(steps).index("seed-haven-areas") < list(steps).index("monthly-chain-maps")
    assert monthly.options["seed_areas_path"] == areas.outputs[1] and areas.outputs[1] in monthly.inputs


def fingerprint_differs(module, key, value, step) -> bool:
    """Whether changing a driver option changes the recorded parameters of ``step`` (it would be run again)."""
    base = module.default_options()
    changed = {**base, key: value}
    before = module.fingerprint(module.build_steps(module.resolve_paths(base))[step])
    after = module.fingerprint(module.build_steps(module.resolve_paths(changed))[step])
    return before != after


def test_a_step_given_an_option_its_script_does_not_have_is_refused(monkeypatch) -> None:
    p = driver.resolve_paths(driver.default_options())
    module = driver.load_script("run_light_store")
    monkeypatch.setattr(module, "default_options", lambda: {"dem_path": 1})
    with pytest.raises(KeyError, match="no option"):
        driver.build_steps(p)


def test_unknown_step_names_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        driver.run_safe_haven(run_steps=("stage9",), dry_run=True)


# --- the whole sequence on the scene ---------------------------------------------------------------------

CALLS: list = []


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    directory = tmp_path_factory.mktemp("driver_scene")
    dem_path, horizons_path = bs.write_inputs(directory)
    georef = _load_dem(dem_path)[1]
    psr_map = np.zeros((300, 300), dtype=np.uint8)
    psr_map[20:23, 150:153] = 255
    psr_map[200:210, 20:30] = 255
    psr_map[270, 270] = 255
    ls.write_geotiff(directory / "psr.tif", psr_map, georef.with_nodata(None))
    patch = pytest.MonkeyPatch()
    first, stop = "2027-01-01T00:00:00", "2027-04-01T00:00:00"
    stamps = ls.times(first + "Z", driver.last_sample(stop, 120), step_hours=2.0).values
    vectors = {"sun": scene.sun_vectors, "earth": scene.earth_vectors}
    patch.setattr(driver.load_script("run_light_store"), "spice_vector_source", lambda body: vectors[body])
    patch.setattr(driver.load_script("run_safe_haven_stage2"), "spice_vector_source", lambda body: vectors[body])
    patch.setattr(driver.load_script("run_safe_haven_stage1"), "EARTH_VECTORS_M", scene.earth_vectors(stamps))
    patch.setattr(driver.load_script("run_safe_haven_stage1"), "SUN_VECTORS_M", scene.sun_vectors(stamps))
    original = driver.Step.call

    def recording(self, *, overwrite):
        CALLS.append(self.name)
        return original(self, overwrite=overwrite)

    patch.setattr(driver.Step, "call", recording)
    options = dict(
        scenario_directory=directory, dem_path=dem_path, horizons_path=horizons_path, start_utc=first, stop_utc=stop,
        codec="zlib-1", steps_per_chunk=48, lighting="cpu", backend="cpu", light_workers=2, plan_workers=2,
        flood_workers=2, route_workers=2, cell_patches=1, haven_neighbourhood_m=3.0, seed_separation_m=4.0,
        operational_hours=24.0)
    yield directory, options
    patch.undo()


@pytest.fixture(scope="module")
def first_run(world):
    directory, options = world
    CALLS.clear()
    code = driver.run_safe_haven(**options)
    return code, list(CALLS)


def outputs_of(directory):
    out = directory / "safe_havens"
    return [directory / "psr_ids.tif", directory / "psr_regions.csv", directory / "light_store" / "manifest.json",
            out / "safe_haven_stage_1_hibernation_hours.tif", out / "safe_haven_stage_2_plan_seeds.csv",
            out / "safe_haven_stage_3_floods.csv", out / "safe_haven_stage_3_edges.csv",
            out / "safe_haven_stage_3_passable_mask.npy", out / "safe_haven_stage_4_chains.csv",
            out / "safe_haven_stage_4_predecessor_edges.tif", out / "safe_haven_stage_4_longest_chain_per_root.gpkg",
            out / "safe_haven_stage_4_longest_chain_per_root_2027.tif", out / "safe_haven_stage_2_persistence_2027.tif",
            out / "safe_haven_stage_2_persistence_contact_sheet.png"]


def test_the_first_run_makes_every_product_in_order(world, first_run) -> None:
    directory, _options = world
    code, calls = first_run
    assert code == 0 and calls == list(STEPS)
    for path in outputs_of(directory):
        if path.name != "manifest.json":
            assert path.exists(), path
    assert any((directory / "light_store").iterdir())
    rows = chains.read_chain_table(directory / "safe_havens" / "safe_haven_stage_4_chains.csv")
    assert rows and all(r["chain_psr_small"] is not None for r in rows)          # the routes were reconstructed
    with rasterio.open(directory / "safe_havens" / "safe_haven_stage_4_longest_chain_per_root_2027.tif") as image:
        assert image.count == 4 and (image.read(4) > 0).any()                      # the year's image, with its havens
    with rasterio.open(directory / "safe_havens" / "safe_haven_stage_2_persistence_2027.tif") as grid:
        assert grid.dtypes == ("uint8",) and grid.read(1).max() >= 1                 # the persistence of the seeds' cells
    # every haven pixel of every month, from the stage-1 product, at the driver's survival limit
    months_directory = directory / "safe_havens" / "safe_haven_stage_1_haven_months"
    with rasterio.open(directory / "safe_havens" / "safe_haven_stage_1_hibernation_hours.tif") as stage1:
        files = sorted(months_directory.glob("safe_haven_stage_1_haven_months_*.tif"))
        assert len(files) == stage1.count
        limit = int(driver.SURVIVAL_LIMIT_HOURS // 2)
        for band, path in enumerate(files, start=1):
            with rasterio.open(path) as month:
                raw = stage1.read(band)
                assert np.array_equal(month.read(1), np.where(raw <= limit, raw, 255))
                assert int(month.tags()["SAFE_HAVEN_PIXELS"]) == int((raw <= limit).sum()) and path.name.endswith(
                    stage1.tags(band)["TIMESTAMP_UTC"][:7] + ".tif")
    # the seed haven areas: one image per month and the pixel file, with the chain starts of the table
    areas = ls.read_seed_areas(directory / "safe_havens" / "safe_haven_stage_4_seed_haven_areas.npz")
    assert areas["radius_meters"] == driver.AREA_RADIUS_METERS and areas["with_image"].all()   # every month had its haven image
    from lunarscout._numba_horizon.safe_haven_chain_maps import build_chain_sets, seed_month_spans
    first, last, names = seed_month_spans(rows)
    lines = build_chain_sets(rows)[2].lines
    starts, alone = [], []
    for month in range(len(names)):                  # a brute-force count: the chains with 2 or more seeds not passed, and with one
        left = [[s for s in line if last[s - 1] >= month >= first[line[0] - 1]] for line in lines]
        starts.append(sum(len(seeds) >= 2 for seeds in left))
        alone.append(sum(len(seeds) == 1 and seeds[0] == line[-1] for seeds, line in zip(left, lines)))
    assert areas["months"] == names and areas["starts"].tolist() == starts and areas["alone"].tolist() == alone
    assert sum(starts) > 0
    image_files = sorted((directory / "safe_havens" / "safe_haven_stage_4_seed_haven_areas").glob("*.tif"))
    assert [f.name[-11:-4] for f in image_files] == areas["months"]
    state = json.loads((directory / "safe_havens" / "safe_haven_run_state.json").read_text())
    assert set(state) == set(STEPS) and all(state[s]["fingerprint"] for s in STEPS)
    assert state["stage3"]["parameters"]["settings"]["max_slope_deg"] == 15.0
    assert "unit_cache_mib" not in json.dumps(state)


def test_a_second_run_runs_nothing(world, first_run) -> None:
    _directory, options = world
    CALLS.clear()
    assert driver.run_safe_haven(**options) == 0 and CALLS == []


def test_a_dry_run_reports_and_changes_nothing(world, first_run, caplog) -> None:
    directory, options = world
    before = {p: p.stat().st_mtime for p in outputs_of(directory) if p.exists()}
    CALLS.clear()
    with caplog.at_level("INFO", logger="lunarscout.scripts.safe_haven"):
        assert driver.run_safe_haven(**options, max_slope_deg=10.0, dry_run=True) == 0
    assert CALLS == [] and {p: p.stat().st_mtime for p in before} == before
    text = caplog.text
    assert "psr-regions: up to date" in text and "light-store: up to date" in text
    assert "stage1: would run (parameters changed: max_slope_deg" in text
    assert "stage2: would run (an input will be rebuilt" in text and "chain-maps: would run" in text


def backdate(directory, seconds, *, keep=()):
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = Path(root) / name
            if path not in keep:
                os.utime(path, (path.stat().st_atime - seconds, path.stat().st_mtime - seconds))


def test_a_changed_input_reruns_only_what_depends_on_it(world, first_run) -> None:
    directory, options = world
    backdate(directory, 100, keep=())
    os.utime(directory / "psr.tif", None)                                       # the PSR map changed just now
    CALLS.clear()
    assert driver.run_safe_haven(**options) == 0
    assert CALLS == ["psr-regions", "stage3", "stage4", "chain-maps", "yearly-chain-maps", "seed-haven-areas",
                     "monthly-chain-maps", "overview-maps", "quality-maps", "cluster-maps"]   # not the store, stage 1, 2 or the haven months
    CALLS.clear()
    assert driver.run_safe_haven(**options) == 0 and CALLS == []


def test_a_changed_parameter_reruns_the_steps_that_use_it_and_what_follows(world, first_run) -> None:
    _directory, options = world
    CALLS.clear()
    assert driver.run_safe_haven(**options, max_days=20.0) == 0
    assert CALLS == ["stage3", "stage4", "chain-maps", "yearly-chain-maps", "seed-haven-areas", "monthly-chain-maps", "quality-maps", "cluster-maps"]
    CALLS.clear()
    assert driver.run_safe_haven(**{**options, "flood_workers": 1, "route_workers": 1, "max_days": 20.0}) == 0
    assert CALLS == []                                                          # how it runs is not what it makes
    assert driver.run_safe_haven(**options) == 0 and CALLS == ["stage3", "stage4", "chain-maps", "yearly-chain-maps", "seed-haven-areas", "monthly-chain-maps", "quality-maps", "cluster-maps"]    # and back again


def test_force_runs_a_step_and_what_depends_on_it(world, first_run) -> None:
    _directory, options = world
    CALLS.clear()
    assert driver.run_safe_haven(**options, force=("stage4",)) == 0
    assert CALLS == ["stage4", "chain-maps", "yearly-chain-maps", "seed-haven-areas", "monthly-chain-maps", "quality-maps", "cluster-maps"]


def test_run_steps_selects_a_subset_in_the_sequences_order(world, first_run) -> None:
    _directory, options = world
    CALLS.clear()
    assert driver.run_safe_haven(**options, run_steps=("cluster-maps", "quality-maps", "monthly-chain-maps", "yearly-chain-maps", "chain-maps", "stage4"),
                                 force=("stage4", "chain-maps")) == 0
    assert CALLS == ["stage4", "chain-maps", "yearly-chain-maps", "monthly-chain-maps", "quality-maps", "cluster-maps"]   # the rebuilt table makes them stale
    CALLS.clear()
    assert driver.run_safe_haven(**options, run_steps=("chain-maps", "stage4")) == 0 and CALLS == []
    assert driver.run_safe_haven(**options) == 0 and CALLS == ["seed-haven-areas", "monthly-chain-maps"]   # the areas the subset left behind the table, then the pixels that draw them


def test_outputs_the_driver_never_ran_are_not_replaced_unless_adopted_or_forced(world, first_run, tmp_path, caplog) -> None:
    directory, options = world
    state = tmp_path / "state.json"
    CALLS.clear()
    with caplog.at_level("INFO", logger="lunarscout.scripts.safe_haven"):
        assert driver.run_safe_haven(**options, state_path=state) == 1 and CALLS == []
    assert "never ran the step" in caplog.text and "--adopt-existing psr-regions" in caplog.text
    assert driver.run_safe_haven(**options, state_path=state, adopt_existing=STEPS) == 0 and CALLS == []
    recorded = json.loads(state.read_text())
    assert set(recorded) == set(STEPS) and all(recorded[s].get("adopted") for s in STEPS)
    assert driver.run_safe_haven(**options, state_path=state) == 0 and CALLS == []          # now they are trusted
    assert driver.run_safe_haven(**options, state_path=state, force=("psr-regions",)) == 0
    assert CALLS[0] == "psr-regions"                                                       # forced replaces what is there


def test_a_failing_step_stops_the_sequence_and_is_not_recorded(world, first_run, monkeypatch) -> None:
    directory, options = world
    def failing(self, *, overwrite):
        CALLS.append(self.name)
        return 1 if self.name == "stage3" else 0

    monkeypatch.setattr(driver.Step, "call", failing)
    state_before = json.loads((directory / "safe_havens" / "safe_haven_run_state.json").read_text())
    CALLS.clear()
    assert driver.run_safe_haven(**options, force=("stage3",)) == 1
    assert CALLS == ["stage3"]                                          # stage 4 and the maps did not run
    assert json.loads((directory / "safe_havens" / "safe_haven_run_state.json").read_text()) == state_before


def test_the_main_function_reads_options_and_lists_the_steps(world, first_run, capsys) -> None:
    directory, options = world
    assert driver.main(["--list-steps"]) == 0 and capsys.readouterr().out.split() == list(STEPS)
    CALLS.clear()
    code = driver.main(["--scenario-directory", str(directory), "--dem-path", str(options["dem_path"]),
                        "--horizons-path", str(options["horizons_path"]), "--start-utc", options["start_utc"],
                        "--stop-utc", options["stop_utc"], "--codec", "zlib-1", "--steps-per-chunk", "48",
                        "--lighting", "cpu", "--backend", "cpu", "--cell-patches", "1", "--haven-neighbourhood-m", "3.0",
                        "--seed-separation-m", "4.0", "--operational-hours", "24.0", "--dry-run", "--color", "never"])
    assert code == 0 and CALLS == []
    assert (directory / "safe_havens" / "safe_haven_run.log").exists()
