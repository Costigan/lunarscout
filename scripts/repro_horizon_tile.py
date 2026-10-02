"""Reproduce a single horizon tile for diagnosis under compute-sanitizer.

Runs exactly the production kernel path for one 128-pixel tile so a CUDA fault
can be replayed deterministically (and inspected with
``/usr/local/cuda/bin/compute-sanitizer --tool memcheck``) without generating an
entire DEM's worth of patches.

Usage::

    /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
        .venv/bin/python scripts/repro_horizon_tile.py \
        --primary-dem /workspace/70south/fixed_ldem_80s_20m.clipped.tif \
        --surrounding-dems /workspace/70south/ldem_70s_118m.tif \
        --tile-x 29056 --tile-y 18944

The default tile coordinates are the first tile that faulted on NRP (pod 3,
pass 0 of 2). Edit ``--primary-dem`` and ``--surrounding-dems`` to point at
your local copies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunarscout._numba_horizon.contract import ContractConfiguration, SegmentTensor
from lunarscout._numba_horizon.cuda_backend import CudaSession
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, PATCH_SIZE
from lunarscout._numba_horizon.generator import generate_patch_horizons
from lunarscout._numba_horizon.geometry import (
    GridConvergenceInput,
    build_subpatch_segments_numba,
)
from lunarscout._numba_horizon.pyramid import (
    load_max_pyramid_cache,
    pyramid_cache_path,
    write_max_pyramid_cache,
)
from lunarscout.products import _load_dem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dem", required=True, help="Primary DEM path.")
    parser.add_argument(
        "--surrounding-dems",
        default="",
        help="Comma-separated surrounding DEM paths (optional).",
    )
    parser.add_argument("--tile-x", type=int, default=29056)
    parser.add_argument("--tile-y", type=int, default=18944)
    parser.add_argument("--observer-height-m", type=float, default=0.0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the same tile this many times in one process (sustained-load test).",
    )
    args = parser.parse_args(argv)

    dem_paths: list[Path] = [Path(args.primary_dem).expanduser().resolve()]
    dem_paths += [
        Path(p).expanduser().resolve()
        for p in args.surrounding_dems.split(",")
        if p.strip()
    ]
    dems = tuple(_load_dem(path)[0] for path in dem_paths)
    primary = dems[0]

    session = CudaSession(device_id=0, production_concurrency=1)
    pyramids = []
    for dem, dem_path in zip(dems, dem_paths, strict=True):
        cache_path = pyramid_cache_path(dem_path)
        try:
            pyramid = load_max_pyramid_cache(dem, cache_path)
        except (OSError, ValueError):
            pyramid = session.build_max_pyramid(dem)
            write_max_pyramid_cache(pyramid, cache_path)
        pyramids.append(pyramid)

    configuration = ContractConfiguration(
        PATCH_SIZE,
        PATCH_SIZE,
        AZIMUTH_COUNT,
        8,
        len(dems),
        primary.width,
        primary.height,
    )

    values, _centers, _convergence = build_subpatch_segments_numba(
        dems,
        tile_column=args.tile_x,
        tile_row=args.tile_y,
        tile_width=PATCH_SIZE,
        azimuth_count=AZIMUTH_COUNT,
        maximum_distance_m=1_000_000.0,
        observer_elevation_m=args.observer_height_m,
        subpatch_size=8,
        grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
        parallel=True,
    )
    dem_ids = np.broadcast_to(
        np.arange(len(dems), dtype=np.int32), values.shape[:-1]
    ).copy()
    tensor = SegmentTensor(values, dem_ids, configuration)

    horizons = None
    for iteration in range(1, args.repeat + 1):
        horizons = generate_patch_horizons(
            session,
            tensor,
            tuple(pyramids),
            tile_column=args.tile_x,
            tile_row=args.tile_y,
            observer_elevation_m=args.observer_height_m,
        )
        print(
            f"iteration {iteration}/{args.repeat}: tile "
            f"({args.tile_x}, {args.tile_y}) OK: "
            f"{horizons.slopes.shape} slopes, "
            f"finite={np.isfinite(horizons.slopes).sum()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
