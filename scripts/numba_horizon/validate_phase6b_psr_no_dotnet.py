#!/usr/bin/env python3
"""Run one complete private PSR product in a fresh process without .NET."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import rasterio

from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.psr_cuda import PsrCudaSession
from lunarscout._numba_horizon.psr_pipeline import run_psr_product
from lunarscout._numba_horizon.psr import (
    compute_psr_patch_reference,
    reduce_sun_vectors_for_psr,
)
from lunarscout.georeference import GeoReference


def _inputs(repository: Path):
    fixture_path = repository / "tests/data/numba_horizon/phase6b_psr_csharp.json"
    artifact = json.loads(fixture_path.read_text(encoding="utf-8"))
    case = next(
        value
        for value in artifact["cases"]
        if value["name"] == "compressed_quantized_mixed"
    )
    uncompressed_case = next(
        value for value in artifact["cases"] if value["name"] == "interpolated_mixed"
    )
    projection = artifact["projection"]
    x = np.arange(128, dtype=np.float64)[None, :]
    y = np.arange(128, dtype=np.float64)[:, None]
    elevation = np.ascontiguousarray((x - y) * 0.1, dtype=np.float32)
    dem = DemGrid(
        elevation,
        np.asarray(artifact["geotransform"], dtype=np.float64),
        ProjectionParameters(
            radius_m=projection["radius_m"],
            latitude_origin_rad=projection["latitude_origin_rad"],
            longitude_origin_rad=projection["longitude_origin_rad"],
            scale=projection["scale"],
            false_easting_m=projection["false_easting_m"],
            false_northing_m=projection["false_northing_m"],
        ),
    )
    pixel = np.arange(128 * 128, dtype=np.int32)[:, None]
    azimuth = np.arange(AZIMUTH_COUNT, dtype=np.int32)[None, :]
    horizons = (
        np.float32(0.45)
        + np.float32(0.01) * (pixel % 23)
        + np.float32(0.0001) * (azimuth % 17)
    ).astype(np.float32).reshape(128, 128, AZIMUTH_COUNT)
    georef = GeoReference(
        projection_wkt='PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=tuple(float(value) for value in artifact["geotransform"]),
        width=128,
        height=128,
        pixel_size_x=float(artifact["geotransform"][1]),
        pixel_size_y=float(artifact["geotransform"][5]),
        nodata=None,
    )
    vectors = np.asarray(case["sun_vectors_m"], dtype=np.float64).reshape(-1, 3)
    expected = np.frombuffer(
        base64.b64decode(case["output_base64"]), dtype=np.uint8
    ).reshape(128, 128)
    return dem, georef, horizons, vectors, expected, case, uncompressed_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    dem, georef, horizons, vectors, expected, case, uncompressed_case = _inputs(repository)

    with tempfile.TemporaryDirectory(prefix="lunarscout-phase6b-psr-") as temporary:
        root = Path(temporary)
        horizon_store = HorizonTileStore(root / "horizons")
        started = time.perf_counter()
        horizon_path = horizon_store.write(
            0, 0, 0.0, horizons.reshape(-1, AZIMUTH_COUNT), compress=True
        )
        horizon_write_seconds = time.perf_counter() - started
        decoded_horizons = horizon_store.read(0, 0, 0.0)
        if decoded_horizons is None:
            raise RuntimeError("compressed horizon round trip was not readable")
        reduced_vectors, _indices = reduce_sun_vectors_for_psr(dem, vectors)
        decoded_reference = compute_psr_patch_reference(
            dem,
            decoded_horizons,
            reduced_vectors,
            tile_y=0,
            tile_x=0,
        )

        started = time.perf_counter()
        session = PsrCudaSession()
        cuda_session_seconds = time.perf_counter() - started
        output_path = root / "psr.tif"
        started = time.perf_counter()
        run_psr_product(
            dem=dem,
            georef=georef,
            horizon_store=horizon_store,
            output_path=output_path,
            sun_vectors_m=vectors,
            patch_calculator=session.compute_patch,
        )
        pipeline_seconds = time.perf_counter() - started
        with rasterio.open(output_path) as dataset:
            actual = dataset.read(1)
            mask = dataset.dataset_mask()
            profile = dataset.profile
        if not np.array_equal(actual, decoded_reference):
            raise RuntimeError(
                "fresh-process CUDA output differs from the decoded-horizon reference"
            )
        csharp_mismatch_count = int(np.count_nonzero(actual != expected))
        uncompressed_expected = np.frombuffer(
            base64.b64decode(uncompressed_case["output_base64"]), dtype=np.uint8
        ).reshape(128, 128)
        uncompressed_mismatch_count = int(
            np.count_nonzero(actual != uncompressed_expected)
        )

        forbidden = sorted(
            name
            for name in sys.modules
            if name == "clr"
            or name.startswith("clr.")
            or name == "pythonnet"
            or name.startswith("pythonnet.")
            or name == "moonlib"
            or name.startswith("moonlib.")
        )
        report = {
            "schema": "lunarscout-numba-phase6b-psr-no-dotnet-v1",
            "compressed_csharp_oracle_sha256": case["output_sha256"],
            "uncompressed_csharp_oracle_sha256": uncompressed_case["output_sha256"],
            "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "decoded_reference_sha256": hashlib.sha256(
                decoded_reference.tobytes()
            ).hexdigest(),
            "exact_decoded_reference_match": True,
            "compressed_csharp_oracle_mismatch_count": csharp_mismatch_count,
            "uncompressed_csharp_oracle_mismatch_count": uncompressed_mismatch_count,
            "horizon_file_bytes": horizon_path.stat().st_size,
            "output_file_bytes": output_path.stat().st_size,
            "horizon_write_seconds": horizon_write_seconds,
            "cuda_session_seconds": cuda_session_seconds,
            "pipeline_seconds": pipeline_seconds,
            "mask_valid_pixels": int(np.count_nonzero(mask == 255)),
            "mask_invalid_pixels": int(np.count_nonzero(mask == 0)),
            "block_shape": [profile["blockysize"], profile["blockxsize"]],
            "compression": profile["compress"],
            "dtype": profile["dtype"],
            "forbidden_modules_loaded": forbidden,
        }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
