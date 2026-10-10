"""Optional dependencies fail with a structured install hint (review finding H9).

pandas and matplotlib were imported lazily but declared nowhere, so the
root-exported DataFrame and plotting helpers raised a bare
ModuleNotFoundError after a plain install.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import tomllib

import pytest

import lunarscout as ls

_TIMES = [datetime(2027, 1, 1, tzinfo=timezone.utc)]
_POINT = ls.LonLat(0.0, -89.5)


@pytest.mark.parametrize(
    ("hidden", "call", "extra"),
    [
        ("pandas", lambda: ls.body_vectors_ned_dataframe(_POINT, "sun", _TIMES), "analysis"),
        ("pandas", lambda: ls.body_azimuth_elevation_dataframe(_POINT, "sun", _TIMES), "analysis"),
        ("matplotlib.pyplot", lambda: ls.plot_body_elevation(_POINT, "sun", _TIMES), "plot"),
        ("matplotlib.pyplot", lambda: ls.plot_body_elevations(_POINT, ["sun"], _TIMES), "plot"),
    ],
    ids=["ned_dataframe", "azimuth_elevation_dataframe", "plot_body_elevation", "plot_body_elevations"],
)
def test_missing_optional_dependency_raises_structured_error(
    monkeypatch: pytest.MonkeyPatch, hidden: str, call, extra: str
) -> None:
    monkeypatch.setitem(sys.modules, hidden, None)
    with pytest.raises(ls.OptionalDependencyError) as error:
        call()
    assert error.value.code == "optional_dependency_missing"
    assert error.value.details["extra"] == extra
    assert error.value.details["install"] == f"pip install 'lunarscout[{extra}]'"
    assert isinstance(error.value, ImportError)  # existing except ImportError still works


def test_scenario_plotting_reports_the_plot_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    scenario = ls.open_scenario(tmp_path)
    with pytest.raises(ls.OptionalDependencyError) as error:
        scenario.plot_azimuth_elevation_axes()
    assert error.value.details["extra"] == "plot"


def test_optional_dependencies_are_declared_as_extras() -> None:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    extras = project["optional-dependencies"]
    assert any(requirement.startswith("pandas") for requirement in extras["analysis"])
    assert any(requirement.startswith("matplotlib") for requirement in extras["plot"])
    assert not any(
        requirement.startswith(("pandas", "matplotlib")) for requirement in project["dependencies"]
    )
