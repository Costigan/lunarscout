from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "docs/numba-horizon-phase-1-real-terrain-fixtures.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_real_terrain_manifest_has_reproducible_provenance() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    small = manifest["small_automatically_acquired_fixture"]
    local = manifest["larger_local_scenario"]

    assert manifest["schema_version"] == 1
    assert small["source_url"].startswith("https://pgda.gsfc.nasa.gov/")
    assert small["data_doi"] == "10.60903/gsfcpgda-lola-spole"
    assert small["source_window"] == [14944, 14944, 512, 512]
    assert len(small["output_raster_float32_sha256"]) == 64
    assert {"smooth", "boundary", "high-latitude"} <= set(
        small["observer_classes"]
    )
    assert len(local["dems"]) == 4
    assert {"smooth", "rugged", "boundary", "high-latitude"} <= set(
        local["observer_classes"]
    )
    assert all(len(dem["sha256"]) == 64 for dem in local["dems"])


def test_external_real_terrain_files_match_recorded_hashes() -> None:
    if os.environ.get("LUNARSCOUT_RUN_PHASE1_EXTERNAL_TERRAIN") != "1":
        pytest.skip(
            "set LUNARSCOUT_RUN_PHASE1_EXTERNAL_TERRAIN=1 to validate external DEMs"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    small = manifest["small_automatically_acquired_fixture"]
    small_path = Path(
        os.environ.get(
            "LUNARSCOUT_PHASE1_LOLA_SUBSET",
            "/tmp/lunarscout-phase1-lola-512.tif",
        )
    )
    assert small_path.is_file(), (
        "missing automatically acquired fixture; run "
        f"{small['acquisition_script']}"
    )
    with tempfile.TemporaryDirectory(prefix="lunarscout-phase1-test-") as temp_dir:
        raw_path = Path(temp_dir) / "elevation.bin"
        subprocess.run(
            ["gdal_translate", "-q", "-of", "ENVI", str(small_path), str(raw_path)],
            check=True,
        )
        assert _sha256(raw_path) == small["output_raster_float32_sha256"]
    for dem in manifest["larger_local_scenario"]["dems"]:
        path = Path(dem["path"])
        assert path.is_file(), f"missing external fixture: {path}"
        assert path.stat().st_size == dem["size_bytes"]
        assert _sha256(path) == dem["sha256"]
