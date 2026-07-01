#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import rasterio
from rasterio.windows import Window


DEFAULT_TILE_SIZE = 128


def _backup_path(path: Path) -> Path:
    for index in range(1000):
        suffix = ".original.tif" if index == 0 else f".original.{index}.tif"
        candidate = path.with_name(f"{path.name}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available temporary name for {path}")


def _temporary_output_path(path: Path) -> Path:
    for index in range(1000):
        suffix = ".clipped-tiled.tmp.tif"
        if index:
            suffix = f".clipped-tiled.{index}.tmp.tif"
        candidate = path.with_name(f"{path.name}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available temporary name for {path}")


def _predictor(dtype: str) -> int:
    return 3 if dtype.startswith("float") else 2


def _copy_tags(source: rasterio.DatasetReader, target: rasterio.DatasetWriter) -> None:
    target.update_tags(**source.tags())
    for band_index in range(1, source.count + 1):
        target.update_tags(band_index, **source.tags(band_index))


def _copy_color_interpretation(
    source: rasterio.DatasetReader,
    target: rasterio.DatasetWriter,
) -> None:
    try:
        target.colorinterp = source.colorinterp
    except Exception:
        pass


def _clip_profile(
    source: rasterio.DatasetReader,
    *,
    width: int,
    height: int,
    window: Window,
    tile_size: int,
) -> dict:
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        width=width,
        height=height,
        transform=source.window_transform(window),
        tiled=True,
        blockxsize=tile_size,
        blockysize=tile_size,
        bigtiff="IF_SAFER",
    )
    if not profile.get("compress"):
        profile["compress"] = "DEFLATE"
    if not profile.get("predictor"):
        profile["predictor"] = _predictor(str(profile["dtype"]))
    return profile


def clip_geotiff(
    path: Path,
    *,
    tile_size: int,
    keep_original: bool,
    force: bool,
) -> None:
    path = path.expanduser().resolve()
    if tile_size <= 0:
        raise ValueError("Tile size must be positive.")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)

    print(f"Input: {path}", flush=True)
    with rasterio.open(str(path)) as source:
        width = int(source.width)
        height = int(source.height)
        clipped_width = width - (width % tile_size)
        clipped_height = height - (height % tile_size)
        print(f"Original size: {width} x {height}", flush=True)
        print(f"Clipped size:  {clipped_width} x {clipped_height}", flush=True)
        if clipped_width <= 0 or clipped_height <= 0:
            raise ValueError(
                "Raster is smaller than one tile after clipping "
                f"to {tile_size} x {tile_size}."
            )
        if clipped_width == width and clipped_height == height:
            print(
                "Width and height are already multiples of "
                f"{tile_size}; no clipping is needed.",
                flush=True,
            )
            if not force:
                print(
                    "Leaving file unchanged. Use --force to rewrite it anyway.",
                    flush=True,
                )
                return
            print("--force specified; rewriting the file anyway.", flush=True)

    temporary_output = _temporary_output_path(path)
    backup = _backup_path(path)
    completed = False
    try:
        with rasterio.open(str(path)) as source:
            window = Window(0, 0, clipped_width, clipped_height)
            profile = _clip_profile(
                source,
                width=clipped_width,
                height=clipped_height,
                window=window,
                tile_size=tile_size,
            )
            print(f"Writing clipped, tiled GeoTIFF to: {temporary_output}", flush=True)
            with rasterio.open(str(temporary_output), "w", **profile) as target:
                for band_index in range(1, source.count + 1):
                    print(f"  Copying band {band_index}/{source.count}", flush=True)
                    data = source.read(band_index, window=window)
                    target.write(data, band_index)
                try:
                    mask = source.dataset_mask(window=window)
                    target.write_mask(mask)
                except Exception:
                    pass
                _copy_tags(source, target)
                _copy_color_interpretation(source, target)
        if not temporary_output.exists():
            raise RuntimeError(
                f"Expected temporary output was not created: {temporary_output}"
            )
        print(f"Renaming original to: {backup}", flush=True)
        path.rename(backup)
        print(f"Renaming temporary output to: {path}", flush=True)
        temporary_output.rename(path)
        completed = True
    finally:
        if not completed:
            print("Operation failed; leaving original file unchanged.", flush=True)
            temporary_output.unlink(missing_ok=True)

    if not path.exists():
        raise RuntimeError(f"Expected output filename was not created: {path}")
    if keep_original:
        print(f"Keeping renamed original: {backup}", flush=True)
    else:
        print(f"Deleting renamed original: {backup}", flush=True)
        backup.unlink()
    print(f"Done: {path}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clip a GeoTIFF on the bottom and right so width and height are "
            "multiples of the tile size, then rewrite it as a tiled GeoTIFF."
        )
    )
    parser.add_argument("filename", type=Path)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"tile size and dimension multiple (default: {DEFAULT_TILE_SIZE})",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="keep the renamed original instead of deleting it after success",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite even when width and height are already tile-size multiples",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        clip_geotiff(
            args.filename,
            tile_size=args.tile_size,
            keep_original=args.keep_original,
            force=args.force,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
