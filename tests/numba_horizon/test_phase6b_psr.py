from __future__ import annotations

from datetime import timedelta
import base64
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import numpy as np
import pytest
import rasterio

from lunarscout._numba_horizon.cuda_backend import CudaBackendError
from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT
from lunarscout._numba_horizon.file_format import HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.product_vectors import (
    generate_moon_me_vectors,
    resolve_moon_me_vectors,
)
from lunarscout._numba_horizon.psr import (
    _pixel_frame,
    compute_psr_patch_reference,
    reduce_sun_vectors_for_psr,
)
from lunarscout._numba_horizon.psr_pipeline import (
    PsrPipelineCancelled,
    run_psr_product,
)
from lunarscout._numba_horizon.psr_cuda import PsrCudaSession
from lunarscout.georeference import GeoReference


def _dem(width: int = 1, height: int = 1) -> DemGrid:
    return DemGrid(
        np.zeros((height, width), dtype=np.float32),
        np.asarray((1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0), dtype=np.float64),
        ProjectionParameters(
            radius_m=1_737_400.0,
            latitude_origin_rad=-np.pi / 2.0,
            longitude_origin_rad=0.0,
            scale=1.0,
            false_easting_m=0.0,
            false_northing_m=0.0,
        ),
    )


def _position_for_local_angle(
    dem: DemGrid, azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    rotation, translation = _pixel_frame(dem, 0, 0)
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    local = np.asarray(
        (
            np.sin(azimuth) * np.cos(elevation),
            np.cos(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ),
        dtype=np.float64,
    )
    return (local * 150_000_000_000.0 - translation) @ rotation.T


def test_psr_reference_uses_strict_upper_solar_limb_threshold() -> None:
    dem = _dem()
    horizons = np.full((1, 1, AZIMUTH_COUNT), 0.3, dtype=np.float32)

    shadow = compute_psr_patch_reference(
        dem,
        horizons,
        _position_for_local_angle(dem, 0.0, 0.0)[None, :],
        tile_y=0,
        tile_x=0,
        valid_height=1,
        valid_width=1,
    )
    illuminated = compute_psr_patch_reference(
        dem,
        horizons,
        _position_for_local_angle(dem, 0.0, 0.1)[None, :],
        tile_y=0,
        tile_x=0,
        valid_height=1,
        valid_width=1,
    )

    assert shadow.item() == 255
    assert illuminated.item() == 0


def test_psr_vector_reduction_keeps_highest_elevation_per_viewpoint_bin() -> None:
    dem = _dem()
    vectors = np.stack(
        (
            _position_for_local_angle(dem, 10.01, 1.0),
            _position_for_local_angle(dem, 10.02, 2.0),
            _position_for_local_angle(dem, 20.0, -1.0),
        )
    )

    reduced, indices = reduce_sun_vectors_for_psr(dem, vectors)

    np.testing.assert_array_equal(indices, (1, 2))
    np.testing.assert_array_equal(reduced, vectors[indices])


def test_psr_reference_matches_actual_csharp_ilgpu_kernel_fixture() -> None:
    fixture_path = (
        Path(__file__).parents[1]
        / "data"
        / "numba_horizon"
        / "phase6b_psr_csharp.json"
    )
    artifact = json.loads(fixture_path.read_text(encoding="utf-8"))
    projection = artifact["projection"]
    x = np.arange(128, dtype=np.float64)[None, :]
    y = np.arange(128, dtype=np.float64)[:, None]
    dem = DemGrid(
        np.ascontiguousarray((x - y) * 0.1, dtype=np.float32),
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
    for case in artifact["cases"]:
        if case["name"] == "constant_shadow":
            horizons = np.full((128 * 128, AZIMUTH_COUNT), 0.3, dtype=np.float32)
        else:
            horizons = (
                np.float32(0.45)
                + np.float32(0.01) * (pixel % 23)
                + np.float32(0.0001) * (azimuth % 17)
            ).astype(np.float32)
            if case["name"] == "compressed_quantized_mixed":
                scaled = horizons * np.float32(32767.0 / 50.0)
                horizons = (
                    np.floor(scaled + np.float32(0.5)).astype(np.int16)
                    * np.float32(50.0 / 32767.0)
                ).astype(np.float32)
        expected = np.frombuffer(
            base64.b64decode(case["output_base64"]), dtype=np.uint8
        ).reshape(128, 128)

        actual = compute_psr_patch_reference(
            dem,
            horizons.reshape(128, 128, AZIMUTH_COUNT),
            np.asarray(case["sun_vectors_m"], dtype=np.float64).reshape(-1, 3),
            tile_y=0,
            tile_x=0,
        )

        np.testing.assert_array_equal(actual, expected, err_msg=case["name"])


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_numba_cuda_psr_matches_actual_csharp_ilgpu_kernel_fixture() -> None:
    fixture_path = (
        Path(__file__).parents[1]
        / "data"
        / "numba_horizon"
        / "phase6b_psr_csharp.json"
    )
    artifact = json.loads(fixture_path.read_text(encoding="utf-8"))
    projection = artifact["projection"]
    x = np.arange(128, dtype=np.float64)[None, :]
    y = np.arange(128, dtype=np.float64)[:, None]
    dem = DemGrid(
        np.ascontiguousarray((x - y) * 0.1, dtype=np.float32),
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
    case = next(case for case in artifact["cases"] if case["name"] == "interpolated_mixed")
    expected = np.frombuffer(
        base64.b64decode(case["output_base64"]), dtype=np.uint8
    ).reshape(128, 128)

    actual = PsrCudaSession().compute_patch(
        dem,
        horizons,
        np.asarray(case["sun_vectors_m"], dtype=np.float64).reshape(-1, 3),
        tile_y=0,
        tile_x=0,
    )

    np.testing.assert_array_equal(actual, expected)


def test_explicit_vectors_override_generation_arguments_without_spice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "spiceypy", None)
    vectors = np.asarray(((1.0, 2.0, 3.0),), dtype=np.float64)

    result = resolve_moon_me_vectors(
        "sun",
        explicit_vectors_m=vectors,
        explicit_times=("2027-01-01T00:00:00Z",),
        start="not used",
        stop="not used",
        step=timedelta(0),
    )

    np.testing.assert_array_equal(result.vectors_m, vectors)
    assert result.times_utc[0].isoformat() == "2027-01-01T00:00:00+00:00"


def test_generated_vectors_match_csharp_geometric_moon_me_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def spkpos(target, et, frame, correction, observer):
        calls.append((target, np.asarray(et).tolist(), frame, correction, observer))
        return np.tile((1.0, 2.0, 3.0), (len(et), 1)), np.zeros(len(et))

    fake_spice = SimpleNamespace(utc2et=lambda _value: 42.0, spkpos=spkpos)
    monkeypatch.setitem(sys.modules, "spiceypy", fake_spice)

    result = generate_moon_me_vectors(
        "earth", ("2027-01-01T00:00:00Z",), ensure_kernels=False
    )

    assert calls == [("EARTH", [42.0], "MOON_ME", "NONE", "MOON")]
    np.testing.assert_array_equal(result.vectors_m, ((1000.0, 2000.0, 3000.0),))


def test_linear_vector_time_conversion_uses_one_anchor_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utc_calls = []
    ephemeris_calls = []

    def utc2et(value):
        utc_calls.append(value)
        return 1000.0

    def spkpos(_target, et, _frame, _correction, _observer):
        ephemeris_calls.append(np.asarray(et).copy())
        return np.zeros((len(et), 3)), np.zeros(len(et))

    monkeypatch.setitem(
        sys.modules,
        "spiceypy",
        SimpleNamespace(utc2et=utc2et, spkpos=spkpos),
    )

    generate_moon_me_vectors(
        "sun",
        ("2023-12-01T00:00:00Z", "2023-12-01T06:00:00Z"),
        ensure_kernels=False,
        time_conversion="linear_from_anchor",
    )

    assert utc_calls == ["2023-12-01T00:00:00.000000"]
    np.testing.assert_array_equal(ephemeris_calls[0], (1000.0, 22600.0))


def _georef(width: int, height: int) -> GeoReference:
    return GeoReference(
        projection_wkt='PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
        width=width,
        height=height,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=None,
    )


def test_psr_pipeline_resumes_by_horizon_patch_and_marks_missing_tiles_invalid(
    tmp_path: Path,
) -> None:
    dem = _dem(width=129, height=1)
    horizons = HorizonTileStore(tmp_path / "horizons")
    horizons.write(
        0,
        0,
        0.0,
        np.full((128, AZIMUTH_COUNT), 5.0, dtype=np.float32),
        compress=True,
        valid_width=128,
        valid_height=1,
    )
    output = tmp_path / "psr.tif"
    cancelled = threading.Event()
    progress = []
    fractions = []

    def on_progress(event):
        progress.append(event)
        if event.state == "valid":
            cancelled.set()

    with pytest.raises(PsrPipelineCancelled):
        run_psr_product(
            dem=dem,
            georef=_georef(129, 1),
            horizon_store=horizons,
            output_path=output,
            sun_vectors_m=_position_for_local_angle(dem, 0.0, 0.0)[None, :],
            invalid_value=9,
            cancellation_requested=cancelled.is_set,
            progress_callback=fractions.append,
            progress_event_callback=on_progress,
        )

    assert not output.exists()
    assert fractions == [0.0, 0.5]
    journal = json.loads(
        (tmp_path / ".psr.tif.lunarscout-partial.journal.json").read_text()
    )
    assert journal["completed_patches"] == {"0,0": "valid"}

    progress_stream = StringIO()
    resumed_fractions = []
    result = run_psr_product(
        dem=dem,
        georef=_georef(129, 1),
        horizon_store=horizons,
        output_path=output,
        sun_vectors_m=_position_for_local_angle(dem, 0.0, 0.0)[None, :],
        invalid_value=9,
        progress_callback=resumed_fractions.append,
        progress_stream=progress_stream,
    )

    assert result == output
    assert resumed_fractions == [0.5, 1.0]
    assert progress_stream.getvalue().endswith("PSR complete: 2/2 patches\n")
    with rasterio.open(output) as dataset:
        assert np.all(dataset.read(1)[:, :128] == 255)
        assert dataset.read(1)[0, 128] == 9
        assert np.all(dataset.dataset_mask()[:, :128] == 255)
        assert dataset.dataset_mask()[0, 128] == 0


def test_psr_auto_backend_falls_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import psr_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(psr_cuda, "PsrCudaSession", UnavailableCudaSession)
    dem = _dem()
    horizons = HorizonTileStore(tmp_path / "horizons")
    horizons.write(
        0,
        0,
        0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )

    result = run_psr_product(
        dem=dem,
        georef=_georef(1, 1),
        horizon_store=horizons,
        output_path=tmp_path / "auto-psr.tif",
        sun_vectors_m=_position_for_local_angle(dem, 0.0, 1.0)[None, :],
        backend="auto",
    )

    with rasterio.open(result) as dataset:
        assert dataset.read(1).item() == 0


def test_psr_explicit_cuda_does_not_silently_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lunarscout._numba_horizon import psr_cuda

    class UnavailableCudaSession:
        def __init__(self, **_kwargs) -> None:
            raise CudaBackendError("deliberately unavailable")

    monkeypatch.setattr(psr_cuda, "PsrCudaSession", UnavailableCudaSession)

    with pytest.raises(CudaBackendError, match="deliberately unavailable"):
        run_psr_product(
            dem=_dem(),
            georef=_georef(1, 1),
            horizon_store=HorizonTileStore(tmp_path / "horizons"),
            output_path=tmp_path / "cuda-psr.tif",
            sun_vectors_m=np.asarray(((1.0, 2.0, 3.0),)),
            backend="cuda",
        )
