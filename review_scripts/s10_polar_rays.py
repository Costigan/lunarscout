#!/usr/bin/env python3
"""Near-pole horizon error from interpolated ray segments, and the subpatch-size-1 fix (M-C).

Production builds exact ray segments only at subpatch centres (every 8 pixels) and gives every other
pixel a translated, bilinearly blended copy of its four neighbours' rays. Near the projection pole of a
polar stereographic DEM, grid north rotates quickly from pixel to pixel, so the copies point the wrong
way. With subpatch size 1 every pixel is its own centre and gets its own exact rays.

The script, on a DEM that contains the projection pole (the 512 px LOLA fixture does, at pixel 256,256):
  1. generates production horizons for the whole DEM (subpatch 8) on the GPU;
  2. rebuilds the four patches that meet nearest the pole with subpatch size 1, timing the CPU
     segment build and reporting segment memory;
  3. compares both with the s2 float64 brute force for pixels along four directions from the pole,
     binned by distance.

    .venv/bin/python review_scripts/s10_polar_rays.py --dem /tmp/lunarscout-phase1-lola-512.tif
Needs a CUDA GPU. Used for review finding M-C (2026-10-10).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import s2_horizon_vs_bruteforce as s2  # noqa: E402

import lunarscout as ls  # noqa: E402
from lunarscout._numba_horizon.contract import ContractConfiguration, SegmentTensor  # noqa: E402
from lunarscout._numba_horizon.cuda_backend import CudaSession  # noqa: E402
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, PATCH_SIZE  # noqa: E402
from lunarscout._numba_horizon.generator import generate_patch_horizons  # noqa: E402
from lunarscout._numba_horizon.geometry import (  # noqa: E402
    GridConvergenceInput,
    build_subpatch_segments_numba,
)
from lunarscout.products import _load_dem  # noqa: E402

DISTANCES_PX = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 120)
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
AZIMUTHS = np.arange(0, AZIMUTH_COUNT, 40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dem", default="/tmp/lunarscout-phase1-lola-512.tif")
    args = parser.parse_args()
    dem = _load_dem(Path(args.dem))[0]
    projection = dem.projection
    pole_c, pole_r = s2.crs_to_pixel_vec(
        dem, np.array([projection.false_easting_m]), np.array([projection.false_northing_m])
    )
    pole = (int(round(pole_c[0])), int(round(pole_r[0])))
    print(f"projection pole at pixel {pole}; resolution {dem.map_resolution_m:.1f} m")

    # 1. production horizons (subpatch 8) for the whole DEM
    out = Path(tempfile.mkdtemp(prefix="ls_s10_"))
    started = time.perf_counter()
    ls.generate_horizons(out, [args.dem], observer_height_m=0.0, compress=False, overwrite=True)
    print(f"production horizons: {time.perf_counter() - started:.1f} s")
    tiles: dict[tuple[int, int], np.ndarray] = {}

    def production(col: int, row: int, az: int) -> float:
        key = (row // PATCH_SIZE * PATCH_SIZE, col // PATCH_SIZE * PATCH_SIZE)
        if key not in tiles:
            tiles[key] = s2.read_production_tile(out, key[0], key[1], 0.0)
        return float(tiles[key][row - key[0], col - key[1], az])

    # 2. subpatch size 1 for the patches meeting nearest the pole
    session = CudaSession(device_id=0)
    pyramid = session.build_max_pyramid(dem)
    exact: dict[tuple[int, int], np.ndarray] = {}
    origins = {
        (y, x)
        for y in (pole[1] - 1, pole[1]) for x in (pole[0] - 1, pole[0])
        if 0 <= y < dem.height and 0 <= x < dem.width
    }
    for y, x in sorted({(y // PATCH_SIZE * PATCH_SIZE, x // PATCH_SIZE * PATCH_SIZE) for y, x in origins}):
        started = time.perf_counter()
        values, _, _ = build_subpatch_segments_numba(
            [dem], tile_column=x, tile_row=y, tile_width=PATCH_SIZE,
            azimuth_count=AZIMUTH_COUNT, maximum_distance_m=1_000_000.0,
            observer_elevation_m=0.0, subpatch_size=1,
            grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0), parallel=True,
        )
        build_seconds = time.perf_counter() - started
        configuration = ContractConfiguration(
            PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT, 1, 1, dem.width, dem.height,
        )
        tensor = SegmentTensor(values, np.zeros(values.shape[:-1], dtype=np.int32), configuration)
        started = time.perf_counter()
        slopes = generate_patch_horizons(session, tensor, (pyramid,), tile_column=x, tile_row=y).degrees()
        print(f"patch (y={y}, x={x}) subpatch 1: segments {values.nbytes / 2**30:.2f} GiB, "
              f"build {build_seconds:.1f} s, GPU {time.perf_counter() - started:.1f} s")
        exact[(y, x)] = slopes.reshape(PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT)

    # 3. compare with the brute force by distance from the pole
    print(f"\n{'dist px':>7} {'dist m':>7} {'n':>4} {'prod mean|d|':>12} {'prod max|d|':>11} "
          f"{'sub1 mean|d|':>12} {'sub1 max|d|':>11}")
    for distance in DISTANCES_PX:
        prod_d, sub1_d = [], []
        pixels = {(pole[0] + dx * distance, pole[1] + dy * distance) for dx, dy in DIRECTIONS}
        for col, row in sorted(pixels):
            if not (0 <= col < dem.width - 1 and 0 <= row < dem.height - 1):
                continue
            key = (row // PATCH_SIZE * PATCH_SIZE, col // PATCH_SIZE * PATCH_SIZE)
            for az in AZIMUTHS:
                ref, _ = s2.brute_horizon_deg([dem], col, row, az * s2.AZ_STEP_DEG, 0.0, 12288.0)
                if not np.isfinite(ref):
                    continue
                prod_d.append(production(col, row, int(az)) - ref)
                if key in exact:
                    sub1_d.append(float(exact[key][row - key[0], col - key[1], az]) - ref)
        prod_a, sub1_a = np.abs(prod_d), np.abs(sub1_d)
        sub1 = (f"{sub1_a.mean():12.4f} {sub1_a.max():11.4f}" if sub1_a.size else f"{'-':>12} {'-':>11}")
        print(f"{distance:7d} {distance * dem.map_resolution_m:7.0f} {prod_a.size:4d} "
              f"{prod_a.mean():12.4f} {prod_a.max():11.4f} {sub1}")


if __name__ == "__main__":
    main()
