"""The command-line options of the scripts: the shared helper, and every script's options against its procedure."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from lunarscout import _script_cli as cli
from lunarscout._numba_horizon.safe_haven_floods import FloodSettings
from lunarscout._numba_horizon.safe_haven_windows import WindowParameters

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
# script file, the procedure it defines, and the options that are not constants (structured settings)
SCRIPT_PROCEDURES = {
    "generate_psr_regions": ("run_psr_regions", set()),
    "run_light_store": ("run_light_store", set()),
    "run_safe_haven_stage1": ("run_stage1", set()),
    "run_safe_haven_stage2": ("run_stage2", {"parameters"}),
    "run_safe_haven_stage3": ("run_stage3", {"settings"}),
    "run_safe_haven_stage4": ("run_stage4", {"flood_settings"}),
    "run_safe_haven_chain_maps": ("make_chain_maps", {"flood_settings"}),
    "run_safe_haven_yearly_chain_maps": ("make_yearly_chain_maps", {"flood_settings"}),
    "run_safe_haven_overview_maps": ("make_overview_maps", set()),
    "run_safe_haven_quality_maps": ("make_quality_maps", set()),
    "run_safe_haven_cluster_maps": ("make_cluster_maps", set()),
}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- the helper -------------------------------------------------------------------------------------------

OPTIONS = (
    cli.Option("A_PATH", "path", "a path"), cli.Option("B_PATH", "optional_path", "an optional path"),
    cli.Option("NAME", "str", "a name"), cli.Option("COUNT", "int", "an integer"), cli.Option("LIMIT", "optional_float", "a limit"),
    cli.Option("RATIO", "float", "a ratio"), cli.Option("FLAG", "bool", "a flag"), cli.Option("MODE", "str", "a mode", choices=("x", "y")),
    cli.Option("CELLS", "pairs", "cells"), cli.Option("IDS", "ids", "ids"), cli.Option("WORDS", "strs", "words", choices=("a", "b", "c")),
)
DEFAULTS = {"a_path": Path("/x"), "b_path": None, "name": "n", "count": 3, "limit": 20.0, "ratio": 0.5, "flag": False,
            "mode": "x", "cells": None, "ids": None, "words": ("a", "b")}


def parse(argv):
    parser = argparse.ArgumentParser()
    cli.add_options(parser, OPTIONS, DEFAULTS)
    return cli.options_from(parser.parse_args(argv), OPTIONS)


def test_an_option_is_the_constant_name_in_lower_case_with_hyphens() -> None:
    assert [o.flag for o in OPTIONS][:3] == ["--a-path", "--b-path", "--name"] and OPTIONS[0].key == "a_path"


def test_no_arguments_give_the_defaults_and_each_kind_is_read() -> None:
    assert parse([]) == DEFAULTS
    got = parse(["--a-path", "~/y", "--b-path", "/z", "--name", "m", "--count", "7", "--limit", "none", "--ratio", "1e-3",
                 "--flag", "--mode", "y", "--cells", "0,1", "2,3", "--ids", "4", "5", "--words", "c"])
    assert got["a_path"] == Path("~/y").expanduser() and got["b_path"] == Path("/z") and got["name"] == "m"
    assert got["count"] == 7 and got["limit"] is None and got["ratio"] == 1e-3 and got["flag"] is True and got["mode"] == "y"
    assert got["cells"] == [(0, 1), (2, 3)] and got["ids"] == [4, 5] and got["words"] == ("c",)
    assert parse(["--no-flag"])["flag"] is False and parse(["--b-path", "none"])["b_path"] is None
    assert parse(["--cells"])["cells"] is None and parse(["--words"])["words"] == ()          # no value: none, nothing


def test_a_bad_value_is_refused() -> None:
    for argv in (["--count", "x"], ["--mode", "z"], ["--cells", "1"], ["--words", "d"], ["--unknown", "1"]):
        with pytest.raises(SystemExit):
            parse(argv)


def test_the_help_shows_each_default() -> None:
    parser = argparse.ArgumentParser()
    cli.add_options(parser, OPTIONS, DEFAULTS)
    text = parser.format_help()
    assert "(default: /x)" in text and "(default: 20.0)" in text and "(default: a b)" in text and "(default: none)" in text


def test_the_flood_settings_flags_are_one_per_field_and_round_trip() -> None:
    parser = argparse.ArgumentParser()
    defaults = FloodSettings(max_slope_deg=15.0, max_days=10.0)
    cli.add_flood_settings(parser, defaults)
    assert cli.flood_settings_from(parser.parse_args([]), defaults) == defaults
    changed = cli.flood_settings_from(parser.parse_args(["--max-slope-deg", "12", "--edge-rule", "any_later",
                                                         "--psr-small-max-m2", "800"]), defaults)
    assert (changed.max_slope_deg, changed.edge_rule, changed.psr_small_max_m2, changed.max_days) == (12.0, "any_later", 800.0, 10.0)
    assert {name for name, _k, _t in cli.FLOOD_FIELDS} == {f.name for f in dataclasses.fields(FloodSettings)}


def test_the_window_flags_cover_the_exposed_fields_and_keep_forward_only() -> None:
    parser = argparse.ArgumentParser()
    defaults = WindowParameters(forward_only=True)
    cli.add_window_parameters(parser, defaults)
    assert cli.window_parameters_from(parser.parse_args([]), defaults) == defaults
    changed = cli.window_parameters_from(parser.parse_args(["--survival-limit-hours", "60", "--cell-patches", "1"]), defaults)
    assert (changed.survival_limit_hours, changed.cell_patches, changed.forward_only) == (60.0, 1, True)
    assert {name for name, _k, _t in cli.WINDOW_FIELDS} <= {f.name for f in dataclasses.fields(WindowParameters)}


# --- every script ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SCRIPT_PROCEDURES))
def test_a_scripts_options_are_exactly_its_procedures_parameters(name) -> None:
    module = load(name)
    procedure, structured = SCRIPT_PROCEDURES[name]
    parameters = set(inspect.signature(getattr(module, procedure)).parameters)
    defaults = module.default_options()
    assert set(defaults) == parameters and structured <= set(defaults)
    assert {option.key for option in module.OPTIONS} == set(defaults) - structured
    assert all(inspect.signature(getattr(module, procedure)).parameters[k].kind is inspect.Parameter.KEYWORD_ONLY
               for k in parameters)                                           # called by keyword, as the driver does
    for option in module.OPTIONS:
        assert hasattr(module, option.name), option.name                      # the default is a constant of that name


@pytest.mark.parametrize("name", sorted(SCRIPT_PROCEDURES))
def test_the_defaults_follow_the_constants_when_they_are_read(name, monkeypatch) -> None:
    module = load(name)
    option = next(o for o in module.OPTIONS if o.kind == "path")
    monkeypatch.setattr(module, option.name, Path("/patched") / option.key)
    assert module.default_options()[option.key] == Path("/patched") / option.key
    parser = module.build_parser()
    assert parser.parse_args([]).__dict__[option.key] == Path("/patched") / option.key


@pytest.mark.parametrize("name", sorted(SCRIPT_PROCEDURES))
def test_a_command_line_value_reaches_the_procedure(name, monkeypatch, tmp_path) -> None:
    module = load(name)
    procedure, structured = SCRIPT_PROCEDURES[name]
    seen = {}
    monkeypatch.setattr(module, procedure, lambda **kwargs: seen.update(kwargs) or 0)
    option = next(o for o in module.OPTIONS if o.kind == "path" and o.name != "LOG_PATH")
    # the first path option names where the log goes (or its parent): give it a directory that exists
    argv = [option.flag, str(tmp_path / "given"), "--log-path", str(tmp_path / "run.log"), "--color", "never"]
    for other in module.OPTIONS:
        if other.kind == "path" and other is not option and other.name.endswith("DIRECTORY"):
            (tmp_path / other.key).mkdir(exist_ok=True)
            argv += [other.flag, str(tmp_path / other.key)]
    (tmp_path / "given").mkdir(exist_ok=True)
    assert module.main(argv) == 0
    assert seen[option.key] == tmp_path / "given" and set(seen) == set(module.default_options())
    assert (tmp_path / "run.log").exists()


def test_the_slope_limit_is_15_in_every_script_that_has_one() -> None:
    assert load("run_safe_haven_stage3").SETTINGS.max_slope_deg == 15.0
    assert load("run_safe_haven_stage4").FLOOD_SETTINGS.max_slope_deg == 15.0
    assert load("run_safe_haven_chain_maps").FLOOD_SETTINGS.max_slope_deg == 15.0
    assert load("run_safe_haven_stage1").MAX_SLOPE_DEG == 15.0 and load("run_safe_haven").MAX_SLOPE_DEG == 15.0


@pytest.mark.parametrize("name", sorted(SCRIPT_PROCEDURES) + ["run_safe_haven"])
def test_help_lists_the_options_and_importing_starts_nothing(name, capsys) -> None:
    module = load(name)
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "--log-path" in out and "default" in out
