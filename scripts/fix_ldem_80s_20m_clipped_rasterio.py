#!/usr/bin/env python3
"""Rescale the Int16 LOLA LDEM to Float32 metres above the reference sphere.

``ldem_80s_20m.clipped.tif`` stores elevation as signed 16-bit integers with a
declared scale of 0.5 m/bit (and an offset equal to the lunar reference-sphere
radius).  Code that ignores the scale therefore reads values that are twice the
true elevation.  This script applies the scale and writes a Float32 GeoTIFF of
metres relative to the 1737.4 km reference sphere, tiled and DEFLATE-compressed
in 128 x 128 pixel blocks, so plain GeoTIFF readers get the correct values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import rasterio


DEFAULT_INPUT = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
DEFAULT_SCALE = 0.5
TILE_SIZE = 128


def _derived_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"fixed_{input_path.name}")


def _write_profile(source: rasterio.DatasetReader, tile_size: int) -> dict:
    return {
        "driver": "GTiff",
        "width": int(source.width),
        "height": int(source.height),
        "count": 1,
        "dtype": "float32",
        "crs": source.crs,
        "transform": source.transform,
        "nodata": float("nan"),
        "tiled": True,
        "blockxsize": tile_size,
        "blockysize": tile_size,
        "compress": "deflate",
        "predictor": 3,
        "bigtiff": "IF_SAFER",
    }


def fix_dem(
    input_path: Path,
    output_path: Path,
    *,
    scale: float,
    overwrite: bool,
) -> None:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path} (use --overwrite)"
        )

    print(f"Input: {input_path}", flush=True)
    with rasterio.open(str(input_path)) as source:
        print(f"  dtype: {source.dtypes[0]}  shape: {source.width} x {source.height}")
        print(
            f"  declared scale/offset: {source.scales} / {source.offsets}",
            flush=True,
        )
        print(f"  applying scale: {scale}", flush=True)

        raw = source.read(1)
        values = raw.astype(np.float32) * float(scale)
        if source.nodata is not None:
            nodata = source.nodata
            if np.issubdtype(raw.dtype, np.integer):
                nodata = int(nodata)
            values[raw == nodata] = np.nan
        tags = dict(source.tags())
        tags["LUNARSCOUT_RESCALED_FROM"] = str(input_path)

    print(f"Output: {output_path}", flush=True)
    with rasterio.open(str(input_path)) as source:
        profile = _write_profile(source, TILE_SIZE)
        with rasterio.open(str(output_path), "w", **profile) as target:
            target.write(values, 1)
            target.update_tags(**tags)
            target.set_band_description(1, "elevation_m")

    print(f"Wrote Float32 metres-above-sphere DEM: {output_path}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"scale applied to raw pixel values (default: {DEFAULT_SCALE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path (default: fixed_<input name> beside the input)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output if it already exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.output or _derived_output_path(args.input)
    try:
        fix_dem(args.input, output, scale=args.scale, overwrite=args.overwrite)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
