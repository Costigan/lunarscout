"""Prototype two-file storage for large lighting time series.

Inputs: synthetic 128 x 128 patch-aligned temporal data generated under --workspace.
Outputs: shadow_maps.tif, light_curves_*.h5, and a JSON benchmark report.
Resources: small fixture by default; scale --width, --height, and --time-count for CephFS tests.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window

from _example_support import example_parser


@dataclass(frozen=True)
class Patch:
    row: int
    col: int
    height: int
    width: int


def _dtype(value: str) -> np.dtype[Any]:
    allowed = {
        "uint8": np.uint8,
        "int8": np.int8,
        "uint16": np.uint16,
        "int16": np.int16,
        "uint32": np.uint32,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }
    try:
        return np.dtype(allowed[value])
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unsupported dtype {value!r}; choose one of {', '.join(sorted(allowed))}"
        ) from exc


def _iter_patches(width: int, height: int, patch_size: int) -> list[Patch]:
    patches: list[Patch] = []
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            patches.append(
                Patch(
                    row=row,
                    col=col,
                    height=min(patch_size, height - row),
                    width=min(patch_size, width - col),
                )
            )
    return patches


def _generate_patch(patch: Patch, time_count: int, dtype: np.dtype[Any]) -> np.ndarray:
    yy = np.arange(patch.row, patch.row + patch.height, dtype=np.float32)[:, None, None]
    xx = np.arange(patch.col, patch.col + patch.width, dtype=np.float32)[None, :, None]
    tt = np.arange(time_count, dtype=np.float32)[None, None, :]

    if dtype.kind in {"u", "i"}:
        info = np.iinfo(dtype)
        span = int(info.max) - int(info.min) + 1
        values = (
            (xx.astype(np.int64) * 3)
            + (yy.astype(np.int64) * 5)
            + (tt.astype(np.int64) * 7)
            + (((xx.astype(np.int64) // 32) + (yy.astype(np.int64) // 32)) * 11)
        )
        values = (values % span) + int(info.min)
        return values.astype(dtype, copy=False)

    values = (
        np.sin((xx + tt) / 17.0)
        + np.cos((yy - tt) / 23.0)
        + ((xx + yy) / 10_000.0)
    )
    return values.astype(dtype, copy=False)


def _tiff_predictor(dtype: np.dtype[Any]) -> int | None:
    if dtype.kind == "f":
        return 3
    if dtype.itemsize > 1:
        return 2
    return None


def _can_create_tiff_with_compression(compression: str, dtype: np.dtype[Any]) -> bool:
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": 16,
        "height": 16,
        "count": 1,
        "dtype": dtype.name,
        "transform": from_origin(0.0, 16.0, 1.0, 1.0),
        "crs": "ESRI:103878",
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
        "compress": compression,
    }
    predictor = _tiff_predictor(dtype)
    if predictor is not None:
        profile["predictor"] = predictor
    try:
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(np.zeros((16, 16), dtype=dtype), 1)
        return True
    except Exception:
        return False


def _resolve_tiff_compression(requested: str, dtype: np.dtype[Any]) -> str:
    if requested != "auto":
        return requested
    if _can_create_tiff_with_compression("zstd", dtype):
        return "zstd"
    return "deflate"


def _hdf5_blosc_kwargs(dtype: np.dtype[Any]) -> dict[str, Any]:
    if dtype.itemsize == 1:
        return hdf5plugin.Blosc(
            cname="lz4",
            clevel=5,
            shuffle=hdf5plugin.Blosc.NOSHUFFLE,
        )
    return hdf5plugin.Blosc(
        cname="zstd",
        clevel=3,
        shuffle=hdf5plugin.Blosc.SHUFFLE,
    )


def _write_shadow_maps_tiff(
    *,
    path: Path,
    width: int,
    height: int,
    time_count: int,
    dtype: np.dtype[Any],
    patches: list[Patch],
    patch_size: int,
    compression: str,
    transform: Any,
    crs: str,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": time_count,
        "dtype": dtype.name,
        "transform": transform,
        "crs": crs,
        "tiled": True,
        "blockxsize": patch_size,
        "blockysize": patch_size,
        "compress": compression,
        "BIGTIFF": "YES",
    }
    predictor = _tiff_predictor(dtype)
    if predictor is not None:
        profile["predictor"] = predictor

    started = time.perf_counter()
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.update_tags(
            storage_role="shadow_maps",
            axis_order="band=time,y,x",
            time_count=str(time_count),
            patch_size=str(patch_size),
            compression_policy="zstd-if-available-else-deflate",
        )
        for patch in patches:
            values = _generate_patch(patch, time_count, dtype)
            window = Window(patch.col, patch.row, patch.width, patch.height)
            for band_index in range(time_count):
                dataset.write(values[:, :, band_index], band_index + 1, window=window)
    return time.perf_counter() - started


def _write_light_curves_h5(
    *,
    path: Path,
    width: int,
    height: int,
    time_count: int,
    dtype: np.dtype[Any],
    patches: list[Patch],
    chunk_xy: int,
    transform: Any,
    crs: str,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "light_curves",
            shape=(height, width, time_count),
            dtype=dtype,
            chunks=(min(chunk_xy, height), min(chunk_xy, width), time_count),
            **_hdf5_blosc_kwargs(dtype),
        )
        handle.create_dataset("time_index", data=np.arange(time_count, dtype=np.int64))
        dataset.attrs["axis_order"] = "y,x,time"
        dataset.attrs["signal_name"] = "synthetic_shadow_value"
        dataset.attrs["units"] = "1" if dtype.kind == "f" else "digital_number"
        dataset.attrs["crs"] = crs
        dataset.attrs["transform"] = tuple(float(value) for value in transform)
        dataset.attrs["chunk_xy"] = int(chunk_xy)
        dataset.attrs["compression_policy"] = (
            "blosc-lz4-clevel5-noshuffle"
            if dtype.itemsize == 1
            else "blosc-zstd-clevel3-shuffle"
        )
        for patch in patches:
            values = _generate_patch(patch, time_count, dtype)
            dataset[patch.row : patch.row + patch.height, patch.col : patch.col + patch.width, :] = values
    return time.perf_counter() - started


def _timed(operation) -> tuple[Any, float]:  # noqa: ANN001
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def _benchmark_reads(
    *,
    shadow_maps_path: Path,
    light_curves_path: Path,
    width: int,
    height: int,
    time_count: int,
) -> dict[str, Any]:
    frame_index = min(time_count - 1, max(0, time_count // 2))
    point_y = height // 2
    point_x = width // 2
    neighborhood_half = 5
    y0 = max(0, point_y - neighborhood_half)
    y1 = min(height, y0 + 10)
    x0 = max(0, point_x - neighborhood_half)
    x1 = min(width, x0 + 10)

    with rasterio.open(shadow_maps_path) as src:
        frame, frame_seconds = _timed(lambda: src.read(frame_index + 1))
        point_from_tiff, point_from_tiff_seconds = _timed(
            lambda: src.read(window=Window(point_x, point_y, 1, 1)).reshape(time_count)
        )

    with h5py.File(light_curves_path, "r") as handle:
        dataset = handle["light_curves"]
        point_curve, point_seconds = _timed(lambda: dataset[point_y, point_x, :])
        neighborhood, neighborhood_seconds = _timed(lambda: dataset[y0:y1, x0:x1, :])
        threshold_mask, threshold_seconds = _timed(lambda: dataset[point_y, point_x, :] > 200)

    if not np.array_equal(point_curve, point_from_tiff):
        raise RuntimeError("BigTIFF and HDF5 point light curves differ.")

    return {
        "frame_index": frame_index,
        "frame_read_seconds": frame_seconds,
        "frame_checksum": float(np.asarray(frame, dtype=np.float64).sum()),
        "point": {"y": point_y, "x": point_x},
        "point_curve_hdf5_seconds": point_seconds,
        "point_curve_bigtiff_seconds": point_from_tiff_seconds,
        "point_curve_checksum": float(np.asarray(point_curve, dtype=np.float64).sum()),
        "neighborhood": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
        "neighborhood_hdf5_seconds": neighborhood_seconds,
        "neighborhood_checksum": float(np.asarray(neighborhood, dtype=np.float64).sum()),
        "threshold_seconds": threshold_seconds,
        "threshold_true_count": int(np.count_nonzero(threshold_mask)),
    }


def _file_report(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size}


def main() -> None:
    parser = example_parser(__doc__)
    parser.add_argument("--width", type=int, default=4992)
    parser.add_argument("--height", type=int, default=5248)
    parser.add_argument("--time-count", type=int, default=5000)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--dtype", type=_dtype, default=np.dtype(np.uint8))
    parser.add_argument(
        "--hdf5-chunk-xy",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="One or more HDF5 spatial chunk sizes to benchmark.",
    )
    parser.add_argument(
        "--tiff-compression",
        choices=("auto", "zstd", "deflate", "lzw"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        default="analysis/timeseries_storage_prototype",
        help="Scenario-relative output directory.",
    )
    args = parser.parse_args()

    width = int(args.width)
    height = int(args.height)
    time_count = int(args.time_count)
    patch_size = int(args.patch_size)
    dtype = np.dtype(args.dtype)
    if width < 1 or height < 1 or time_count < 1:
        raise SystemExit("--width, --height, and --time-count must be positive.")
    if patch_size < 16 or patch_size % 16 != 0:
        raise SystemExit("--patch-size must be a multiple of 16 and at least 16.")
    for chunk_xy in args.hdf5_chunk_xy:
        if chunk_xy < 1:
            raise SystemExit("--hdf5-chunk-xy values must be positive.")

    scenario_root = Path(args.workspace) / "synthetic_scenario"
    output_root = scenario_root / args.output
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    transform = from_origin(0.0, float(height), 1.0, 1.0)
    crs = "ESRI:103878"
    patches = _iter_patches(width, height, patch_size)
    compression = _resolve_tiff_compression(args.tiff_compression, dtype)
    shadow_maps_path = output_root / "shadow_maps.tif"
    report_path = output_root / "report.json"

    tiff_seconds = _write_shadow_maps_tiff(
        path=shadow_maps_path,
        width=width,
        height=height,
        time_count=time_count,
        dtype=dtype,
        patches=patches,
        patch_size=patch_size,
        compression=compression,
        transform=transform,
        crs=crs,
    )

    hdf5_reports: list[dict[str, Any]] = []
    for chunk_xy in args.hdf5_chunk_xy:
        h5_path = output_root / f"light_curves_chunk{chunk_xy}.h5"
        h5_seconds = _write_light_curves_h5(
            path=h5_path,
            width=width,
            height=height,
            time_count=time_count,
            dtype=dtype,
            patches=patches,
            chunk_xy=int(chunk_xy),
            transform=transform,
            crs=crs,
        )
        read_report = _benchmark_reads(
            shadow_maps_path=shadow_maps_path,
            light_curves_path=h5_path,
            width=width,
            height=height,
            time_count=time_count,
        )
        hdf5_reports.append(
            {
                "chunk_xy": int(chunk_xy),
                "write_seconds": h5_seconds,
                "file": _file_report(h5_path),
                "reads": read_report,
            }
        )

    report = {
        "prototype": "two_file_timeseries_storage",
        "shape": {"time": time_count, "height": height, "width": width},
        "dtype": dtype.name,
        "patch_size": patch_size,
        "patch_count": len(patches),
        "uncompressed_bytes_per_representation": int(width * height * time_count * dtype.itemsize),
        "shadow_maps": {
            "write_seconds": tiff_seconds,
            "compression": compression,
            "tile_shape": [patch_size, patch_size],
            "file": _file_report(shadow_maps_path),
        },
        "light_curves": hdf5_reports,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    best_point = min(
        hdf5_reports,
        key=lambda item: item["reads"]["point_curve_hdf5_seconds"],
    )
    best_neighborhood = min(
        hdf5_reports,
        key=lambda item: item["reads"]["neighborhood_hdf5_seconds"],
    )
    print(f"output={output_root}")
    print(f"shape=(time={time_count}, y={height}, x={width}), dtype={dtype.name}")
    print(f"patches={len(patches)}, uncompressed={report['uncompressed_bytes_per_representation']} bytes")
    print(
        "shadow_maps.tif "
        f"compression={compression}, size={shadow_maps_path.stat().st_size} bytes, "
        f"write={tiff_seconds:.4f}s"
    )
    for item in hdf5_reports:
        reads = item["reads"]
        print(
            f"light_curves chunk={item['chunk_xy']} size={item['file']['size_bytes']} bytes "
            f"write={item['write_seconds']:.4f}s "
            f"point={reads['point_curve_hdf5_seconds']:.6f}s "
            f"neighborhood={reads['neighborhood_hdf5_seconds']:.6f}s "
            f"bigtiff_point={reads['point_curve_bigtiff_seconds']:.6f}s"
        )
    print(f"best_point_chunk={best_point['chunk_xy']}")
    print(f"best_neighborhood_chunk={best_neighborhood['chunk_xy']}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
