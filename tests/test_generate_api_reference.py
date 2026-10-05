"""Behavior checks for the documentation generator, including fresh-process use."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_api_reference.py"
spec = importlib.util.spec_from_file_location("generate_api_reference", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_parameter_entries_and_grouped_callbacks():
    doc = """Summary.

Parameters
----------
array : ndarray
    Input array.
progress_callback / cancellation_requested:
    Shared callbacks.
Returns
-------
result : ndarray
    Output.
"""
    assert generator.documented_parameters(doc) == {
        "array", "progress_callback", "cancellation_requested"
    }
    assert generator.documented_parameters("Args:\n    value (int): Input.\nReturns:\n    result: Output.") == {"value"}
    assert generator.documented_parameters(":param int value: Input.") == {"value"}


def test_fallback_exports_exclude_imported_helpers_and_private_functions():
    module = types.ModuleType("lunarscout.sample")
    def public(value):
        pass
    public.__module__ = "lunarscout.sample"
    module.public = public
    module._private = public
    module.Path = Path
    assert generator.public_members(module) == [("public", public)]
    module.__all__ = []
    assert generator.public_members(module) == []


def test_class_property_is_never_evaluated():
    class Example:
        @property
        def danger(self):
            """Property documentation."""
            raise AssertionError("must not execute")
    Example.__module__ = "lunarscout.sample"
    text = generator.render_class("lunarscout.sample.Example", Example)
    assert "Property documentation." in text
    assert "This member is a property." in text


def test_fresh_process_generation_and_check(tmp_path):
    output = tmp_path / "reference.md"
    command = [sys.executable, str(SCRIPT), "--output", str(output)]
    subprocess.run(command, check=True, cwd=tmp_path, capture_output=True)
    text = output.read_text()
    for name in ("lunarscout.generate_horizons", "lunarscout.slope",
                 "lunarscout.cuda.status", "lunarscout.spice.furnish",
                 "lunarscout.trajectory.static_path", "lunarscout.map_algebra.add",
                 "lunarscout.Raster.copy"):
        assert f"`{name}`" in text
    assert "Parameters without explicit entries" in text
    assert "Observer height above the DEM surface" in text
    subprocess.run(command + ["--check"], check=True, capture_output=True)
    output.write_text("stale\n")
    result = subprocess.run(command + ["--check"], capture_output=True)
    assert result.returncode == 1
    assert output.read_text() == "stale\n"
    output.unlink()
    result = subprocess.run(command + ["--check"], capture_output=True)
    assert result.returncode == 1
    assert not output.exists()


def test_generation_does_not_import_cuda_or_spice_runtimes():
    code = f"""
import runpy
import sys
import importlib.abc
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'numba', 'spiceypy'}}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
generator = runpy.run_path({str(SCRIPT)!r})
generator['build_reference']()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
