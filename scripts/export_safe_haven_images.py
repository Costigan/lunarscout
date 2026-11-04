#!/usr/bin/env python3
"""Export the safe-haven viewer's pictures as full-resolution, tile-compressed GeoTIFFs.

    .venv/bin/python scripts/export_safe_haven_images.py [--psr-class 2 --hop-m 500 --no-background ...]

Writes two kinds of file into OUTPUT_DIRECTORY, on the DEM's grid (so they line up with it in a GIS):

* hillshade_psr_rgb.tif: the hillshade with the permanent shadow (psr.tif) composited on it, light blue at 30% opacity, as one
  3-band (RGB) image: the viewer's background at full resolution;
* chains_monthly_<psr>_hop_<metres>m_<YYYY-MM>.tif, one per month: the "Chains (monthly)" layer of the viewer, only the chains
  that touch a PSR of at least PSR_CLASS (0 any, 1 small, 2 medium, 3 large) and whose longest hop (the greatest straight-line
  distance between successive seeds) is at least HOP_M metres, as a 4-band (RGBA) image without the background: alpha is 255
  where something is drawn (routes, PSR marks, havens, the magenta chain starts, lone seeds) and 0 elsewhere, so it lays over the
  RGB image. A month with no chain passing the filters gets no file (WRITE_EMPTY writes a transparent one) and is reported.

Both are tiled (512 pixels), BigTIFF and compressed: the chain images with DEFLATE and a horizontal predictor (lossless), the RGB
background with BACKGROUND_COMPRESSION: "deflate" (lossless, the default, about 20% of the raw size, opens in every tool), "webp"
(lossless, about 27% smaller than deflate, about 5 times slower to write, needs a WebP-enabled reader such as GDAL or QGIS) or "jpeg"
(lossy YCbCr at BACKGROUND_QUALITY, 95 by default, about 7 times smaller than deflate; the edges of the PSR blue blur slightly). Each is written through a staging file that is
renamed on success, so a failed run leaves an existing file as it was. The RGB file is about 2.8 GB before compression.

It reads the viewer's files, so run scripts/view_safe_haven.py (or its preparation) first: the lists (viewer.json, havens.npz) in
VIEWER_DIRECTORY, the background pyramid in CACHE_DIRECTORY and the monthly chain file made by the monthly-chain-maps step. The
exported pixels are those the viewer draws at its finest zoom. Edit the paths below before running, or give them as options
(--constant-name). CPU only; no NVIDIA device is used. Importing this file starts nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunarscout._numba_horizon.safe_haven_export import (  # noqa: E402
    BACKGROUND_COMPRESSIONS, BACKGROUND_NAME, JPEG_QUALITY, PSR_NAMES, export_background, export_monthly_chains)

# Edit these paths before running.
STAGE_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/safe_havens")
PSR_PATH = Path("/e/lunar_analyst_scenarios/70south/psr.tif")              # the permanent shadow; needed for the RGB background
VIEWER_DIRECTORY = STAGE_DIRECTORY / "viewer"
CACHE_DIRECTORY = Path("/tmp/lunarscout_viewer")
OUTPUT_DIRECTORY = STAGE_DIRECTORY.parent / "safe_haven_exports"
PSR_CLASS = 2                  # chains touching a PSR of at least this size: 0 any, 1 small, 2 medium or larger, 3 large
HOP_M = 500.0                  # chains whose longest hop between successive seeds is at least this many metres
BACKGROUND_COMPRESSION = "deflate"   # the RGB background: "deflate" (lossless), "webp" (lossless, smaller, slower, needs a WebP reader) or "jpeg" (lossy)
BACKGROUND_QUALITY = JPEG_QUALITY    # JPEG quality 1-100 when BACKGROUND_COMPRESSION is "jpeg"
WRITE_EMPTY = False            # also write a transparent image for a month with no chain passing the filters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the viewer's background and monthly chain pictures as GeoTIFFs.")
    for flag, default, text in (
        ("--psr-path", PSR_PATH, "the permanent shadow raster (255 is PSR) on the DEM's grid"),
        ("--viewer-directory", VIEWER_DIRECTORY, "where the viewer's viewer.json and havens.npz are"),
        ("--cache-directory", CACHE_DIRECTORY, "where the viewer's image pyramids are cached"),
        ("--output-directory", OUTPUT_DIRECTORY, "where the GeoTIFFs are written"),
    ):
        parser.add_argument(flag, type=lambda value: Path(value).expanduser(), default=default, help=f"{text} (default: {default})")
    parser.add_argument("--psr-class", type=int, choices=range(4), default=PSR_CLASS,
                        help="smallest PSR a chain must touch: 0 any, 1 small, 2 medium, 3 large (default: %(default)s)")
    parser.add_argument("--hop-m", type=float, default=HOP_M, help="smallest longest hop in metres (default: %(default)s)")
    parser.add_argument("--background-compression", choices=BACKGROUND_COMPRESSIONS, default=BACKGROUND_COMPRESSION,
                        help="compression of the RGB background: deflate and webp are lossless, jpeg is lossy and about 7 times "
                             "smaller than deflate (default: %(default)s); the chain images are always lossless deflate")
    parser.add_argument("--background-quality", type=int, default=BACKGROUND_QUALITY,
                        help="JPEG quality 1-100 for --background-compression jpeg (default: %(default)s)")
    parser.add_argument("--write-empty", action="store_true", default=WRITE_EMPTY,
                        help="write a transparent image for a month without chains")
    parser.add_argument("--no-background", action="store_true", help="do not write the RGB background")
    parser.add_argument("--no-chains", action="store_true", help="do not write the monthly chain images")
    return parser


def export_safe_haven_images(*, psr_path=PSR_PATH, viewer_directory=VIEWER_DIRECTORY, cache_directory=CACHE_DIRECTORY,
                             output_directory=OUTPUT_DIRECTORY, psr_class=PSR_CLASS, hop_m=HOP_M, write_empty=WRITE_EMPTY,
                             background_compression=BACKGROUND_COMPRESSION, background_quality=BACKGROUND_QUALITY, background=True, chains=True, progress=print) -> dict:
    """Write the exports and return what was written (a dictionary, as the module's report)."""
    output_directory = Path(output_directory)
    result: dict = {}
    if background:
        path = export_background(viewer_directory, cache_directory, psr_path, output_directory / BACKGROUND_NAME,
                                 compression=background_compression, jpeg_quality=background_quality, progress=progress)
        result["background"] = str(path)
    if chains:
        report = export_monthly_chains(viewer_directory, cache_directory, output_directory, psr_class=psr_class,
                                       hop_m=hop_m, write_empty=write_empty, progress=progress).as_dict()
        report.pop("background")                  # the chain report has no background; keep the one written above
        result.update(report)
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Chains: PSR {PSR_NAMES[args.psr_class]} or larger, longest hop at least {args.hop_m:g} m")
    result = export_safe_haven_images(
        psr_path=args.psr_path, viewer_directory=args.viewer_directory, cache_directory=args.cache_directory,
        output_directory=args.output_directory, psr_class=args.psr_class, hop_m=args.hop_m, write_empty=args.write_empty,
        background_compression=args.background_compression, background_quality=args.background_quality,
        background=not args.no_background, chains=not args.no_chains)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
