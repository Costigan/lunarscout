"""The Phase 1 fixture acquisition script verifies before publishing (review LOW)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/numba_horizon/acquire_phase1_real_fixture.py"

pytestmark = pytest.mark.skipif(
    shutil.which("gdal_translate") is None, reason="needs the gdal_translate tool"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("acquire_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expected_hash: str) -> Path:
    module = _load_script()
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    source = tmp_path / "source.tif"
    with rasterio.open(
        source, "w", driver="GTiff", width=4, height=4, count=1, dtype="float32"
    ) as dataset:
        dataset.write(values, 1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"small_automatically_acquired_fixture": {
        "source_window": [0, 0, 4, 4],
        "source_url": "https://example.invalid/source.tif",
        "output_raster_float32_sha256": expected_hash,
    }}))
    monkeypatch.setattr(module, "MANIFEST", manifest)
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if command[-2].startswith("/vsicurl/"):
            # Stand in for the HTTP read: copy the local source window.
            assert kwargs["timeout"] > 0
            assert kwargs["env"]["GDAL_HTTP_TIMEOUT"]
            command = [*command[:-2], str(source), command[-1]]
        return real_run(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output = tmp_path / "out" / "fixture.tif"
    monkeypatch.setattr(sys, "argv", ["acquire", "--output", str(output)])
    module.main()
    return output


def test_hash_mismatch_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _run(tmp_path, monkeypatch, "0" * 64)

    assert list((tmp_path / "out").iterdir()) == []


def test_verified_download_is_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = hashlib.sha256(np.arange(16, dtype="<f4").tobytes()).hexdigest()

    output = _run(tmp_path, monkeypatch, expected)

    assert [path.name for path in output.parent.iterdir()] == ["fixture.tif"]
    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read(1), np.arange(16).reshape(4, 4))
