from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import lunarscout as ls
import lunarscout.native as native
from lunarscout._native_runtime import bootstrap as native_bootstrap


def _fake_bootstrap(tmp_path: Path, *, loaded: bool = False):
    moonlib_path = tmp_path / "moonlib.dll"
    moonlib_path.write_bytes(b"")
    cspice_path = tmp_path / "libcspice.so"
    cspice_path.write_bytes(b"")
    bootstrap_state = {
        "loaded": loaded,
        "smoke_check": {
            "spice_output": 2,
            "gdal_config_probe": True,
        }
        if loaded
        else None,
    }
    calls: list[dict[str, bool]] = []

    def bootstrap_pythonnet(*, force: bool, verify_bridge_smoke: bool):
        calls.append({"force": force, "verify": verify_bridge_smoke})
        bootstrap_state["loaded"] = True
        bootstrap_state["smoke_check"] = {
            "spice_output": 2,
            "gdal_config_probe": True,
        }

    class MoonlibBridge:
        pass

    return SimpleNamespace(
        calls=calls,
        load_native_bootstrap_config=lambda: SimpleNamespace(
            dll_resolver_search_dirs=(tmp_path,),
            dll_resolver_imports=(),
        ),
        resolve_moonlib_dll=lambda _config: moonlib_path,
        bootstrap_status=lambda: dict(bootstrap_state),
        bootstrap_pythonnet=bootstrap_pythonnet,
        import_moonlib=lambda **_kwargs: SimpleNamespace(MoonlibBridge=MoonlibBridge),
        bridge_type=MoonlibBridge,
    )


def test_import_hides_native_and_does_not_load_pythonnet():
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source_root
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import lunarscout as ls; "
                "print(json.dumps({'native': hasattr(ls, 'native'), "
                "'pythonnet': 'pythonnet' in sys.modules, "
                "'bootstrap': 'lunarscout._native_runtime.bootstrap' in sys.modules}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(completed.stdout) == {
        "native": False,
        "pythonnet": False,
        "bootstrap": False,
    }


def test_status_reports_components_without_initializing(monkeypatch, tmp_path):
    fake = _fake_bootstrap(tmp_path)
    monkeypatch.setattr(native, "_bootstrap_module", lambda: fake)
    monkeypatch.setattr(
        native,
        "_module_available",
        lambda name: name in {"pythonnet", "osgeo"},
    )
    monkeypatch.setattr(
        native,
        "_dotnet_probe",
        lambda: {
            "available": True,
            "executable": "/usr/bin/dotnet",
            "runtimes": ["10.0"],
        },
    )

    result = native.status()

    assert result["available"] is True
    assert result["loaded"] is False
    assert set(result["components"]) == {
        "pythonnet",
        "dotnet",
        "moonlib",
        "cspice",
        "gdal",
    }
    assert fake.calls == []


def test_initialize_is_explicit_and_returns_loaded_status(monkeypatch, tmp_path):
    fake = _fake_bootstrap(tmp_path)
    monkeypatch.setattr(native, "_bootstrap_module", lambda: fake)
    monkeypatch.setattr(native, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        native,
        "_dotnet_probe",
        lambda: {"available": True, "executable": "dotnet", "runtimes": ["10.0"]},
    )

    result = native.initialize(force=True, verify=False)

    assert fake.calls == [{"force": True, "verify": False}]
    assert result["loaded"] is True
    assert result["components"]["cspice"]["verified"] is True
    assert result["components"]["gdal"]["verified_by_native_smoke"] is True


def test_initialize_translates_component_failure(monkeypatch):
    fake = SimpleNamespace(
        bootstrap_pythonnet=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("pythonnet is unavailable")
        )
    )
    monkeypatch.setattr(native, "_bootstrap_module", lambda: fake)

    with pytest.raises(ls.NativeUnavailableError) as raised:
        native.initialize()

    assert raised.value.code == "native_pythonnet_unavailable"
    assert raised.value.details["component"] == "pythonnet"


def test_create_bridge_uses_moonlib_bridge_only(monkeypatch, tmp_path):
    fake = _fake_bootstrap(tmp_path)
    monkeypatch.setattr(native, "_bootstrap_module", lambda: fake)

    bridge = native._create_moonlib_bridge(force=True, verify=False)

    assert isinstance(bridge, fake.bridge_type)


def test_standalone_bootstrap_discovers_local_moonlib_build(monkeypatch, tmp_path):
    repo = tmp_path / "lunarscout"
    payload_dir = repo / "native" / "moonlib" / "bin" / "Debug" / "net10.0" / "linux-x64"
    payload_dir.mkdir(parents=True)
    moonlib = payload_dir / "moonlib.dll"
    moonlib.write_bytes(b"")

    monkeypatch.delenv(native_bootstrap.MOONLIB_DLL_ENV, raising=False)
    monkeypatch.setattr(native_bootstrap, "_repo_root", lambda: repo)

    assert native_bootstrap.resolve_moonlib_dll() == moonlib.resolve()
