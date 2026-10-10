#!/usr/bin/env python3
"""Acquire and verify the bounded NASA LOLA Phase 1 real-terrain fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs/numba-horizon-phase-1-real-terrain-fixtures.json"
DEFAULT_OUTPUT = Path("/tmp/lunarscout-phase1-lola-512.tif")
# GDAL reads the source window over HTTP; bound each request and each step.
GDAL_HTTP_ENVIRONMENT = {
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "5",
}
DOWNLOAD_TIMEOUT_SECONDS = 900
CONVERT_TIMEOUT_SECONDS = 120


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
    if not fixture["source_url"].startswith("https://"):
        raise RuntimeError(f"refusing a non-HTTPS source: {fixture['source_url']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Download into an exclusively created file beside the output, verify it,
    # then rename it into place: a mismatch or failure never leaves a file at
    # the output path, and a pre-existing symlink there is replaced, not
    # followed.
    handle, staging_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp.tif", dir=args.output.parent
    )
    os.close(handle)
    staging = Path(staging_name)
    try:
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
                str(staging),
            ],
            check=True,
            env=os.environ | GDAL_HTTP_ENVIRONMENT,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
        # The hash covers the decoded float32 pixels (an ENVI raw copy), not
        # the GeoTIFF bytes, which differ between GDAL versions' encoders.
        with tempfile.TemporaryDirectory(prefix="lunarscout-phase1-lola-") as temp_dir:
            raw_path = Path(temp_dir) / "elevation.bin"
            subprocess.run(
                ["gdal_translate", "-q", "-of", "ENVI", str(staging), str(raw_path)],
                check=True,
                timeout=CONVERT_TIMEOUT_SECONDS,
            )
            actual = sha256(raw_path)
        expected = fixture["output_raster_float32_sha256"]
        if actual != expected:
            raise RuntimeError(
                f"LOLA raster hash mismatch: expected {expected}, got {actual}"
            )
        os.replace(staging, args.output)
    finally:
        staging.unlink(missing_ok=True)
    print(args.output)
    print(f"raster_float32_sha256={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
