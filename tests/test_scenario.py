from __future__ import annotations

from pathlib import Path

import pytest

import lunarscout as ls


def test_open_scenario_resolves_existing_directory(tmp_path: Path):
    root = tmp_path / "scenario"
    root.mkdir()

    scenario = ls.open_scenario(root / ".")

    assert scenario.root == root.resolve()


def test_open_scenario_rejects_missing_root(tmp_path: Path):
    missing = tmp_path / "missing"

    with pytest.raises(ls.ScenarioError) as raised:
        ls.open_scenario(missing)

    assert raised.value.code == "scenario_root_not_found"


def test_open_scenario_rejects_file_root(tmp_path: Path):
    path = tmp_path / "not-a-directory"
    path.write_text("data", encoding="utf-8")

    with pytest.raises(ls.ScenarioError) as raised:
        ls.open_scenario(path)

    assert raised.value.code == "scenario_root_not_directory"


def test_standard_scenario_paths_do_not_need_to_exist(tmp_path: Path):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    assert scenario.dem_path() == root / "dem.tif"
    assert scenario.horizons_path() == root / "lighting" / "horizons"
    assert scenario.output_path("analysis/result.tif") == root / "analysis" / "result.tif"
    assert not (root / "lighting").exists()
    assert not (root / "analysis").exists()


def test_path_normalizes_relative_components(tmp_path: Path):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    assert scenario.path("analysis/../dem.tif") == root / "dem.tif"
    assert scenario.path(".") == root


@pytest.mark.parametrize("method_name", ["path", "output_path"])
def test_relative_methods_reject_absolute_paths(tmp_path: Path, method_name: str):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.ScenarioPathError) as raised:
        getattr(scenario, method_name)(tmp_path / "outside.tif")

    assert raised.value.code == "scenario_absolute_path_rejected"


@pytest.mark.parametrize("relative_path", ["../outside.tif", "a/../../outside.tif"])
def test_relative_methods_reject_parent_escape(tmp_path: Path, relative_path: str):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.ScenarioPathError) as raised:
        scenario.path(relative_path)

    assert raised.value.code == "scenario_path_escape"


def test_relative_methods_reject_symlink_escape(tmp_path: Path):
    root = tmp_path / "scenario"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.ScenarioPathError) as raised:
        scenario.output_path("linked/result.tif")

    assert raised.value.code == "scenario_path_escape"


@pytest.mark.parametrize("relative_path", ["", "."])
def test_output_path_rejects_scenario_root(tmp_path: Path, relative_path: str):
    root = tmp_path / "scenario"
    root.mkdir()
    scenario = ls.open_scenario(root)

    with pytest.raises(ls.ScenarioPathError) as raised:
        scenario.output_path(relative_path)

    assert raised.value.code == "scenario_output_path_empty"


def test_open_scenario_rejects_attached_state_until_implemented(tmp_path: Path):
    root = tmp_path / "scenario"
    root.mkdir()

    with pytest.raises(ls.ScenarioStateError) as raised:
        ls.open_scenario(root, state=object())

    assert raised.value.code == "scenario_state_unavailable"
