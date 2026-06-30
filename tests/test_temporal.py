from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import numpy as np
import pytest

import lunarscout as ls
from lunarscout import utc_datetime


def _georef(*, nodata=None) -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt='PROJCS["test",GEOGCS["g",DATUM["d",SPHEROID["s",1,0]],PRIMEM["p",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=eqc +R=1 +units=m +no_defs",
        affine_transform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
        width=2,
        height=2,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=nodata,
    )


def test_utc_datetime_defaults_to_aware_utc_midnight() -> None:
    value = utc_datetime(2027, 1, 2)

    assert value.isoformat() == "2027-01-02T00:00:00+00:00"
    assert value.tzinfo is timezone.utc
    assert value.utcoffset().total_seconds() == 0


def test_utc_datetime_preserves_components_and_fold() -> None:
    value = utc_datetime(2027, 2, 3, 4, 5, 6, 789, fold=1)

    assert (value.year, value.month, value.day) == (2027, 2, 3)
    assert (value.hour, value.minute, value.second) == (4, 5, 6)
    assert value.microsecond == 789
    assert value.fold == 1


def test_utc_datetime_uses_standard_datetime_validation() -> None:
    with pytest.raises(ValueError):
        utc_datetime(2027, 2, 30)


def test_times_defaults_naive_inputs_to_utc_and_includes_aligned_stop() -> None:
    time_range = ls.times(
        "2027-01-01T00:00:00",
        datetime(2027, 1, 1, 3),
        step_hours=1,
    )

    assert time_range.start.tzinfo is timezone.utc
    assert time_range.stop.tzinfo is timezone.utc
    assert time_range.time_count == 4
    assert time_range.values.dtype == np.dtype("datetime64[us]")
    np.testing.assert_array_equal(
        time_range.values,
        np.asarray(
            [
                "2027-01-01T00:00:00",
                "2027-01-01T01:00:00",
                "2027-01-01T02:00:00",
                "2027-01-01T03:00:00",
            ],
            dtype="datetime64[us]",
        ),
    )


def test_times_does_not_exceed_unaligned_stop() -> None:
    time_range = ls.times(
        "2027-01-01T00:00:00Z",
        "2027-01-01T02:30:00Z",
        step_hours=1,
    )

    assert time_range.time_count == 3
    assert time_range.values[-1] == np.datetime64("2027-01-01T02:00:00", "us")


def test_times_converts_explicit_source_timezone_to_utc() -> None:
    time_range = ls.times(
        "2027-01-01T00:00:00",
        "2027-01-01T02:00:00",
        step_hours=2,
        source_timezone="America/Los_Angeles",
    )

    assert time_range.start.isoformat() == "2027-01-01T08:00:00+00:00"
    assert time_range.stop.isoformat() == "2027-01-01T10:00:00+00:00"


@pytest.mark.parametrize("step_hours", [0, -1, np.inf, np.nan])
def test_times_rejects_invalid_step(step_hours) -> None:
    with pytest.raises(ls.TimeRangeError) as raised:
        ls.times("2027-01-01", "2027-01-02", step_hours=step_hours)

    assert raised.value.code == "time_range_invalid_step"


def test_times_rejects_reverse_range() -> None:
    with pytest.raises(ls.TimeRangeError) as raised:
        ls.times("2027-01-02", "2027-01-01", step_hours=1)

    assert raised.value.code == "time_range_invalid_order"


def test_time_range_is_frozen_and_time_values_are_read_only() -> None:
    time_range = ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1)

    with pytest.raises(FrozenInstanceError):
        time_range.step_hours = 2  # type: ignore[misc]
    with pytest.raises(ValueError):
        time_range.values[0] = np.datetime64("2028-01-01")


def test_temporal_cube_validates_and_exposes_metadata() -> None:
    time_range = ls.times("2027-01-01", "2027-01-01T02:00:00", step_hours=1)
    values = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

    cube = ls.TemporalCube(values=values, times=time_range, georef=_georef())

    assert cube.values is values
    assert cube.shape == (3, 2, 2)
    assert cube.dtype == np.dtype(np.float32)
    assert cube.time_count == 3
    assert cube.height == 2
    assert cube.width == 2
    assert cube.dimensions == ("time", "y", "x")
    assert cube.times.dtype == np.dtype("datetime64[us]")
    assert cube.nbytes == values.nbytes + cube.times.nbytes
    with pytest.raises(FrozenInstanceError):
        cube.georef = _georef()  # type: ignore[misc]


def test_temporal_cube_rejects_spatial_and_time_mismatches() -> None:
    time_range = ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1)

    with pytest.raises(ls.TemporalCubeError) as spatial:
        ls.TemporalCube(
            values=np.zeros((2, 3, 2), dtype=np.float32),
            times=time_range,
            georef=_georef(),
        )
    assert spatial.value.code == "temporal_cube_spatial_shape_mismatch"

    with pytest.raises(ls.TemporalCubeError) as count:
        ls.TemporalCube(
            values=np.zeros((3, 2, 2), dtype=np.float32),
            times=time_range,
            georef=_georef(),
        )
    assert count.value.code == "temporal_cube_time_count_mismatch"


def test_temporal_cube_rejects_non_increasing_times() -> None:
    time_values = np.asarray(
        ["2027-01-01T01:00:00", "2027-01-01T00:00:00"],
        dtype="datetime64[us]",
    )

    with pytest.raises(ls.TemporalCubeError) as raised:
        ls.TemporalCube(
            values=np.zeros((2, 2, 2), dtype=np.float32),
            times=time_values,
            georef=_georef(),
        )

    assert raised.value.code == "temporal_cube_times_not_increasing"


def test_temporal_cube_rejects_non_numeric_and_complex_values() -> None:
    time_range = ls.times("2027-01-01", "2027-01-01", step_hours=1)

    for values in (
        np.full((1, 2, 2), "text", dtype=object),
        np.ones((1, 2, 2), dtype=np.complex64),
    ):
        with pytest.raises(ls.TemporalCubeError) as raised:
            ls.TemporalCube(values, time_range, _georef())
        assert raised.value.code == "temporal_cube_unsupported_dtype"


def test_temporal_reducers_use_time_axis_and_preserve_georeferencing() -> None:
    time_range = ls.times("2027-01-01", "2027-01-01T02:00:00", step_hours=1)
    values = np.asarray(
        [
            [[1, 9], [3, 8]],
            [[2, 7], [4, 6]],
            [[3, 5], [5, 4]],
        ],
        dtype=np.float32,
    )
    cube = ls.TemporalCube(values, time_range, _georef())

    mean, mean_georef = ls.temporal_mean(cube)
    minimum, _ = ls.temporal_min(cube)
    maximum, _ = ls.temporal_max(cube)
    standard_deviation, _ = ls.temporal_std(cube)

    np.testing.assert_allclose(mean, np.mean(values, axis=0))
    np.testing.assert_array_equal(minimum, np.min(values, axis=0))
    np.testing.assert_array_equal(maximum, np.max(values, axis=0))
    np.testing.assert_allclose(standard_deviation, np.std(values, axis=0))
    assert mean_georef == cube.georef


def test_temporal_reducers_exclude_georef_nodata_by_default() -> None:
    time_range = ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1)
    values = np.asarray(
        [
            [[1, -99], [-99, -99]],
            [[3, 5], [7, -99]],
        ],
        dtype=np.int16,
    )
    cube = ls.TemporalCube(values, time_range, _georef(nodata=-99))

    mean, georef = ls.temporal_mean(cube)

    np.testing.assert_array_equal(mean, np.asarray([[2, 5], [7, -99]], dtype=np.float64))
    assert georef.nodata == -99


def test_temporal_reducer_can_disable_nodata_processing() -> None:
    time_range = ls.times("2027-01-01", "2027-01-01T01:00:00", step_hours=1)
    values = np.asarray(
        [[[1, -99], [1, 1]], [[3, 5], [1, 1]]],
        dtype=np.int16,
    )
    cube = ls.TemporalCube(values, time_range, _georef(nodata=-99))

    mean, georef = ls.temporal_mean(cube, nodata=None)

    assert mean[0, 1] == -47
    assert georef.nodata is None
