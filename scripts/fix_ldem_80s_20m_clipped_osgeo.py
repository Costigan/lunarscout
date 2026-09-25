#!/usr/bin/env python3
"""Rescale the Int16 LOLA LDEM to Float32 metres above the reference sphere.

``ldem_80s_20m.clipped.tif`` stores elevation as signed 16-bit integers with a
declared scale of 0.5 m/bit (and an offset equal to the lunar reference-sphere
radius).  Code that ignores the scale therefore reads values that are twice the
true elevation.  This script applies the scale and writes an uncompressed,
striped Float32 GeoTIFF of metres relative to the 1737.4 km reference sphere,
so plain GeoTIFF readers get the correct values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from osgeo import gdal


gdal.UseExceptions()


DEFAULT_INPUT = Path("/d/viper/maps/lola/ldem_80s_20m.clipped.tif")
DEFAULT_SCALE = 0.5


def _derived_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"fixed_{input_path.name}")


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
    source = gdal.Open(str(input_path), gdal.GA_ReadOnly)
    if source is None:
        raise RuntimeError(f"unable to open: {input_path}")
    band = source.GetRasterBand(1)
    print(
        f"  dtype: {gdal.GetDataTypeName(band.DataType)}  "
        f"shape: {source.RasterXSize} x {source.RasterYSize}"
    )
    print(
        f"  declared scale/offset: {band.GetScale()} / {band.GetOffset()}",
        flush=True,
    )
    print(f"  applying scale: {scale}", flush=True)

    raw = band.ReadAsArray()
    values = raw.astype(np.float32) * float(scale)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        values[raw == int(nodata)] = np.nan

    metadata = source.GetMetadata()
    metadata["LUNARSCOUT_RESCALED_FROM"] = str(input_path)
    geotransform = source.GetGeoTransform()
    projection = source.GetProjection()

    print(f"Output: {output_path}", flush=True)
    driver = gdal.GetDriverByName("GTiff")
    target = driver.Create(
        str(output_path),
        source.RasterXSize,
        source.RasterYSize,
        1,
        gdal.GDT_Float32,
    )
    if target is None:
        raise RuntimeError(f"unable to create: {output_path}")
    target.SetGeoTransform(geotransform)
    target.SetProjection(projection)
    target.SetMetadata(metadata)
    out_band = target.GetRasterBand(1)
    out_band.SetNoDataValue(float("nan"))
    out_band.SetDescription("elevation_m")
    out_band.WriteArray(values)

    out_band.FlushCache()
    target.FlushCache()
    out_band = None
    target = None
    source = None

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
