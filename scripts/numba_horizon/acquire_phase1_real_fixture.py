#!/usr/bin/env python3
"""Acquire and verify the bounded NASA LOLA Phase 1 real-terrain fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs/numba-horizon-phase-1-real-terrain-fixtures.json"
DEFAULT_OUTPUT = Path("/tmp/lunarscout-phase1-lola-512.tif")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = json.loads(MANIFEST.read_text(encoding="utf-8"))[
        "small_automatically_acquired_fixture"
    ]
    x, y, width, height = fixture["source_window"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gdal_translate",
            "-q",
            "-srcwin",
            str(x),
            str(y),
            str(width),
            str(height),
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "PREDICTOR=3",
            f"/vsicurl/{fixture['source_url']}",
            str(args.output),
        ],
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="lunarscout-phase1-lola-") as temp_dir:
        raw_path = Path(temp_dir) / "elevation.bin"
        subprocess.run(
            ["gdal_translate", "-q", "-of", "ENVI", str(args.output), str(raw_path)],
            check=True,
        )
        actual = sha256(raw_path)
    expected = fixture["output_raster_float32_sha256"]
    if actual != expected:
        raise RuntimeError(f"LOLA raster hash mismatch: expected {expected}, got {actual}")
    print(args.output)
    print(f"raster_float32_sha256={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
