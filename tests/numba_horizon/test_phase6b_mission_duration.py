from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio

from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.lightmap_cuda import LightmapCudaSession
from lunarscout._numba_horizon.mission_duration import (
    CandidateStartInterval,
    monthly_candidate_intervals,
    reduce_longest_candidate_duration_stream,
    weekly_candidate_intervals,
)
from lunarscout._numba_horizon.mission_duration_pipeline import (
    MissionDurationPipelineCancelled,
    run_sun_elevation_duration_product,
    run_sun_elevation_earth_elevation_duration_product,
    run_sunlight_duration_product,
    run_sunlight_earth_elevation_duration_product,
)
from lunarscout._numba_horizon.psr import _pixel_frame
from lunarscout.georeference import GeoReference


UTC = timezone.utc


def _dem() -> DemGrid:
    return DemGrid(
        np.zeros((1, 1), dtype=np.float32),
        np.asarray((1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0), dtype=np.float64),
        ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0),
    )


def _georef() -> GeoReference:
    return GeoReference(
        projection_wkt='PROJCS["Moon_South_Pole_Stereographic",GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m",
        affine_transform=(1000.0, 1.0, 0.0, -1000.0, 0.0, -1.0),
        width=1,
        height=1,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=None,
    )


def _position(dem: DemGrid, elevation_deg: float) -> np.ndarray:
    rotation, translation = _pixel_frame(dem, 0, 0)
    elevation = np.deg2rad(elevation_deg)
    local = np.asarray((0.0, np.cos(elevation), np.sin(elevation)))
    return (local * 150_000_000_000.0 - translation) @ rotation.T


def test_interval_helpers_clip_months_and_anchor_weeks() -> None:
    months = monthly_candidate_intervals(
        "2027-01-15T00:00:00Z", "2027-03-10T00:00:00Z"
    )
    assert [(item.start_utc.day, item.stop_utc.day) for item in months] == [
        (15, 1),
        (1, 1),
        (1, 10),
    ]
    weeks = weekly_candidate_intervals(
        "2027-01-03T12:00:00Z", "2027-01-20T12:00:00Z"
    )
    assert [item.stop_utc - item.start_utc for item in weeks] == [
        timedelta(days=7),
        timedelta(days=7),
        timedelta(days=3),
    ]


def test_duration_starts_inside_band_continues_beyond_it_and_is_censored() -> None:
    times = tuple(
        datetime(2027, 1, day, tzinfo=UTC) for day in range(1, 7)
    )
    intervals = (
        CandidateStartInterval(times[1], times[3]),
        CandidateStartInterval(times[3], times[5]),
    )
    conditions = (np.asarray([[True]]) for _ in times)

    durations = reduce_longest_candidate_duration_stream(
        conditions,
        times_utc=times,
        evaluation_start_utc=times[0],
        evaluation_stop_utc=times[-1],
        start_intervals=intervals,
        output_unit="days",
    )

    assert [item.item() for item in durations] == [4.0, 2.0]


def test_duration_uses_following_irregular_sample_intervals_and_inclusive_threshold() -> None:
    times = (
        datetime(2027, 1, 1, tzinfo=UTC),
        datetime(2027, 1, 1, 6, tzinfo=UTC),
        datetime(2027, 1, 1, 15, tzinfo=UTC),
        datetime(2027, 1, 2, tzinfo=UTC),
    )
    values = (0.5, 0.5, 0.49, 1.0)
    conditions = (np.asarray([[value >= 0.5]]) for value in values)

    result = reduce_longest_candidate_duration_stream(
        conditions,
        times_utc=times,
        evaluation_start_utc=times[0],
        evaluation_stop_utc=times[-1],
        start_intervals=((times[0], times[-1]),),
        output_unit="hours",
    )

    assert result[0].item() == 15.0


def test_last_sample_is_credited_to_evaluation_stop_when_stop_is_not_sampled() -> None:
    start = datetime(2027, 1, 1, tzinfo=UTC)
    stop = start + timedelta(hours=10)
    times = (start, start + timedelta(hours=6))

    result = reduce_longest_candidate_duration_stream(
        (np.asarray([[True]]) for _ in times),
        times_utc=times,
        evaluation_start_utc=start,
        evaluation_stop_utc=stop,
        start_intervals=((start, stop),),
        output_unit="hours",
    )

    assert result[0].item() == 10.0


def _write_horizon(tmp_path: Path) -> HorizonTileStore:
    store = HorizonTileStore(tmp_path / "horizons")
    store.write(
        0,
        0,
        0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )
    return store


def _signal(values):
    def calculate(*_args, **_kwargs):
        for value in values:
            yield np.asarray([[value]], dtype=np.float32)

    return calculate


def test_four_product_functions_write_float32_days_and_interval_metadata(
    tmp_path: Path,
) -> None:
    dem = _dem()
    store = _write_horizon(tmp_path)
    times = tuple(
        datetime(2027, 1, 1, tzinfo=UTC) + timedelta(days=index)
        for index in range(4)
    )
    intervals = ((times[0], times[2]), (times[2], times[3]))
    vectors = np.stack(tuple(_position(dem, 1.0) for _ in times))
    common = {
        "dem": dem,
        "georef": _georef(),
        "horizon_store": store,
        "times_utc": times,
        "evaluation_start_utc": times[0],
        "evaluation_stop_utc": times[-1],
        "start_intervals": intervals,
        "sun_vectors_m": vectors,
        "output_unit": "days",
        "backend": "cpu",
    }
    outputs = (
        run_sunlight_duration_product(
            **common,
            output_path=tmp_path / "sunlight.tif",
            sunlight_fraction_threshold=0.5,
            _sun_calculator=_signal((0.5, 0.5, 0.0, 0.0)),
        ),
        run_sun_elevation_duration_product(
            **common,
            output_path=tmp_path / "sun-elevation.tif",
            sun_elevation_threshold_deg=1.0,
            _sun_calculator=_signal((1.0, 1.0, 0.0, 0.0)),
        ),
        run_sunlight_earth_elevation_duration_product(
            **common,
            output_path=tmp_path / "sunlight-earth.tif",
            earth_vectors_m=vectors,
            sunlight_fraction_threshold=0.5,
            earth_elevation_threshold_deg=2.0,
            _sun_calculator=_signal((0.5, 0.5, 0.5, 0.0)),
            _earth_calculator=_signal((2.0, 1.0, 2.0, 2.0)),
        ),
        run_sun_elevation_earth_elevation_duration_product(
            **common,
            output_path=tmp_path / "sun-earth-elevation.tif",
            earth_vectors_m=vectors,
            sun_elevation_threshold_deg=1.0,
            earth_elevation_threshold_deg=2.0,
            _sun_calculator=_signal((1.0, 1.0, 1.0, 0.0)),
            _earth_calculator=_signal((2.0, 1.0, 2.0, 2.0)),
        ),
    )

    expected = ((2.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, 1.0))
    for output, values in zip(outputs, expected, strict=True):
        with rasterio.open(output) as dataset:
            assert dataset.dtypes == ("float32", "float32")
            np.testing.assert_array_equal(dataset.read()[:, 0, 0], values)
            assert dataset.tags(1)["DURATION_UNIT"] == "days"
            assert dataset.tags(1)["CANDIDATE_START_UTC"].startswith("2027-01-01")
            assert dataset.tags(1)["CANDIDATE_STOP_UTC"].startswith("2027-01-03")


def test_cpu_margin_is_body_center_elevation_relative_to_local_horizon() -> None:
    dem = _dem()
    horizons = np.full((128, 128, AZIMUTH_COUNT), 0.25, dtype=np.float32)
    vectors = np.stack((_position(dem, 1.25), _position(dem, -0.75)))

    margins = tuple(
        LightmapCpuSession(time_batch_size=1).iter_patch_margin_tiles(
            dem,
            horizons,
            vectors,
            tile_y=0,
            tile_x=0,
            valid_height=1,
            valid_width=1,
        )
    )

    np.testing.assert_allclose(
        np.asarray([tile.item() for tile in margins]), (1.0, -1.0), atol=2e-5
    )


def test_missing_horizon_uses_configured_invalid_payload_and_mask(tmp_path: Path) -> None:
    dem = _dem()
    times = (
        datetime(2027, 1, 1, tzinfo=UTC),
        datetime(2027, 1, 2, tzinfo=UTC),
    )
    vectors = np.stack(tuple(_position(dem, 1.0) for _ in times))

    output = run_sunlight_duration_product(
        dem=dem,
        georef=_georef(),
        horizon_store=HorizonTileStore(tmp_path / "missing-horizons"),
        output_path=tmp_path / "invalid.tif",
        times_utc=times,
        evaluation_start_utc=times[0],
        evaluation_stop_utc=times[-1],
        start_intervals=((times[0], times[1]),),
        sun_vectors_m=vectors,
        sunlight_fraction_threshold=0.5,
        output_unit="days",
        invalid_value=-7.0,
        backend="cpu",
        _sun_calculator=_signal((1.0, 1.0)),
    )

    with rasterio.open(output) as dataset:
        assert dataset.read(1)[0, 0] == -7.0
        assert dataset.dataset_mask()[0, 0] == 0


def test_cancelled_patch_resumes_as_one_durable_work_unit(tmp_path: Path) -> None:
    dem = _dem()
    store = _write_horizon(tmp_path)
    times = tuple(
        datetime(2027, 1, 1, tzinfo=UTC) + timedelta(days=index)
        for index in range(4)
    )
    vectors = np.stack(tuple(_position(dem, 1.0) for _ in times))
    output_path = tmp_path / "resumed.tif"
    checks = 0

    def cancel_during_reduction() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 5

    kwargs = {
        "dem": dem,
        "georef": _georef(),
        "horizon_store": store,
        "output_path": output_path,
        "times_utc": times,
        "evaluation_start_utc": times[0],
        "evaluation_stop_utc": times[-1],
        "start_intervals": ((times[0], times[-1]),),
        "sun_vectors_m": vectors,
        "sunlight_fraction_threshold": 0.5,
        "output_unit": "days",
        "backend": "cpu",
        "_sun_calculator": _signal((1.0, 1.0, 1.0, 1.0)),
    }
    with pytest.raises(MissionDurationPipelineCancelled):
        run_sunlight_duration_product(
            **kwargs, cancellation_requested=cancel_during_reduction
        )
    assert not output_path.exists()

    output = run_sunlight_duration_product(**kwargs)

    with rasterio.open(output) as dataset:
        assert dataset.read(1)[0, 0] == 3.0
        assert dataset.dataset_mask()[0, 0] == 255


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_cpu_and_cuda_body_center_margins_agree() -> None:
    dem = _dem()
    horizons = np.full((128, 128, AZIMUTH_COUNT), 0.25, dtype=np.float32)
    vectors = np.stack((_position(dem, 1.25), _position(dem, -0.75)))
    kwargs = {
        "tile_y": 0,
        "tile_x": 0,
        "valid_height": 1,
        "valid_width": 1,
    }
    cpu = tuple(
        LightmapCpuSession(time_batch_size=1).iter_patch_margin_tiles(
            dem, horizons, vectors, **kwargs
        )
    )
    cuda = tuple(
        LightmapCudaSession(time_batch_size=1).iter_patch_margin_tiles(
            dem, horizons, vectors, **kwargs
        )
    )

    np.testing.assert_allclose(np.stack(cuda), np.stack(cpu), atol=2e-5)


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_combined_mission_product_cpu_and_cuda_agree(tmp_path: Path) -> None:
    dem = _dem()
    store = _write_horizon(tmp_path)
    times = tuple(
        datetime(2027, 1, 1, tzinfo=UTC) + timedelta(days=index)
        for index in range(4)
    )
    sun_vectors = np.stack(
        tuple(_position(dem, value) for value in (1.0, 1.0, -1.0, 1.0))
    )
    earth_vectors = np.stack(
        tuple(_position(dem, value) for value in (1.0, -1.0, 1.0, 1.0))
    )
    common = {
        "dem": dem,
        "georef": _georef(),
        "horizon_store": store,
        "times_utc": times,
        "evaluation_start_utc": times[0],
        "evaluation_stop_utc": times[-1],
        "start_intervals": ((times[0], times[-1]),),
        "sun_vectors_m": sun_vectors,
        "earth_vectors_m": earth_vectors,
        "sun_elevation_threshold_deg": 0.0,
        "earth_elevation_threshold_deg": 0.0,
        "output_unit": "days",
        "time_batch_size": 2,
    }
    cpu = run_sun_elevation_earth_elevation_duration_product(
        **common, output_path=tmp_path / "combined-cpu.tif", backend="cpu"
    )
    cuda = run_sun_elevation_earth_elevation_duration_product(
        **common, output_path=tmp_path / "combined-cuda.tif", backend="cuda"
    )

    with rasterio.open(cpu) as cpu_dataset, rasterio.open(cuda) as cuda_dataset:
        np.testing.assert_array_equal(cuda_dataset.read(), cpu_dataset.read())
        assert cpu_dataset.read(1)[0, 0] == 1.0
