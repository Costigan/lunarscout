from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

import lunarscout as ls
import lunarscout.temporal_store as temporal_store


def _georef(*, nodata=-99) -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt='PROJCS["test",GEOGCS["g",DATUM["d",SPHEROID["s",1,0]],PRIMEM["p",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=eqc +R=1 +units=m +no_defs",
        affine_transform=(100.0, 2.0, 0.0, 200.0, 0.0, -2.0),
        width=3,
        height=2,
        pixel_size_x=2.0,
        pixel_size_y=-2.0,
        nodata=nodata,
    )


def _cube(*, nodata=-99) -> ls.TemporalCube:
    values = np.asarray(
        [
            [[1, 2, -99], [4, 5, 6]],
            [[3, 4, -99], [6, 7, 8]],
            [[5, 6, -99], [8, 9, 10]],
        ],
        dtype=np.int16,
    )
    return ls.TemporalCube(
        values,
        ls.times("2027-01-01", "2027-01-01T02:00:00", step_hours=1),
        _georef(nodata=nodata),
    )


def _rewrite_manifest(root: Path, mutate) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    completion = {
        "format_version": 1,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    (root / "COMPLETE").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_write_and_open_temporal_cube_round_trip(tmp_path: Path) -> None:
    cube = _cube()
    root = tmp_path / "sun-series"

    series = ls.write_temporal_cube(
        root,
        cube,
        signal_name="sun_fraction",
        units="fraction",
        provenance={"source": "unit-test"},
    )

    assert series.shape == cube.shape
    assert series.dtype == cube.dtype
    assert series.dimensions == ("time", "y", "x")
    assert series.signal_name == "sun_fraction"
    assert series.units == "fraction"
    assert series.vrt_path == root / "series.vrt"
    assert not series.times.flags.writeable
    assert [path.name for path in series.layer_paths] == [
        "20270101T000000.000000Z.tif",
        "20270101T010000.000000Z.tif",
        "20270101T020000.000000Z.tif",
    ]
    values, georef = series.read_layer(1)
    np.testing.assert_array_equal(values, cube.values[1])
    assert georef == cube.georef
    assert not values.flags.writeable

    manifest_bytes = (root / "manifest.json").read_bytes()
    completion = json.loads((root / "COMPLETE").read_text())
    assert completion["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert b"NaN" not in manifest_bytes


def test_vrt_is_relative_and_readable_by_gdal(tmp_path: Path) -> None:
    import rasterio

    cube = _cube()
    series = ls.write_temporal_cube(tmp_path / "series", cube)
    vrt_text = series.vrt_path.read_text(encoding="utf-8")  # type: ignore[union-attr]

    assert 'relativeToVRT="1"' in vrt_text
    assert str(tmp_path) not in vrt_text
    with rasterio.open(series.vrt_path) as dataset:
        assert dataset.count == 3
        assert dataset.descriptions[1] == "2027-01-01T01:00:00.000000Z"
        np.testing.assert_array_equal(dataset.read(3), cube.values[2])


def test_time_lookup_is_explicit_and_nearest_ties_choose_earlier(tmp_path: Path) -> None:
    series = ls.write_temporal_cube(tmp_path / "series", _cube())

    assert series.layer_for_time("2027-01-01T01:00:00Z") == 1
    assert series.layer_for_time("2027-01-01T01:30:00Z", method="nearest") == 1
    assert series.layer_for_time("2027-01-01T01:30:00Z", method="before") == 1
    assert series.layer_for_time("2027-01-01T01:30:00Z", method="after") == 2
    for value in ("2026-12-31T23:00:00Z", "2027-01-01T03:00:00Z"):
        with pytest.raises(ls.TemporalLookupError) as raised:
            series.layer_for_time(value, method="nearest")
        assert raised.value.code == "temporal_time_not_found"
    with pytest.raises(ls.TemporalLookupError):
        series.read_layer(-1)
    with pytest.raises(ls.TemporalLookupError):
        series.read_layer(1.0)  # type: ignore[arg-type]


def test_layer_and_dataset_caches_are_bounded(tmp_path: Path) -> None:
    layer_bytes = _cube().values[0].nbytes
    series = ls.open_temporal_cube(
        ls.write_temporal_cube(tmp_path / "series", _cube()).root,
        layer_cache_bytes=layer_bytes,
        max_open_datasets=1,
    )

    first, _ = series.read_layer(0)
    assert series.read_layer(0)[0] is first
    series.read_layer(1)
    assert list(series._layer_cache) == [1]
    assert len(series._dataset_cache) == 1
    series.close()
    with pytest.raises(ls.TemporalSeriesOpenError) as raised:
        series.read_layer(0)
    assert raised.value.code == "temporal_series_closed"


def test_open_rejects_incomplete_tampered_and_escaping_series(tmp_path: Path) -> None:
    root = ls.write_temporal_cube(tmp_path / "series", _cube()).root
    (root / "COMPLETE").unlink()
    with pytest.raises(ls.TemporalSeriesOpenError) as incomplete:
        ls.open_temporal_cube(root)
    assert incomplete.value.code == "temporal_series_incomplete"

    ls.write_temporal_cube(root, _cube(), overwrite=True).close()
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(ls.TemporalSeriesOpenError) as tampered:
        ls.open_temporal_cube(root)
    assert tampered.value.code == "temporal_series_completion_mismatch"

    ls.write_temporal_cube(root, _cube(), overwrite=True).close()
    _rewrite_manifest(root, lambda manifest: manifest["layers"][0].update(relative_path="../outside.tif"))
    with pytest.raises(ls.TemporalSeriesOpenError) as escaping:
        ls.open_temporal_cube(root)
    assert escaping.value.code == "temporal_series_path_escape"


def test_failed_overwrite_preserves_completed_existing_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = ls.write_temporal_cube(tmp_path / "series", _cube()).root
    original_write = temporal_store.write_geotiff
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(temporal_store, "write_geotiff", fail_second)
    with pytest.raises(ls.TemporalSeriesWriteError):
        ls.write_temporal_cube(root, _cube(), overwrite=True)

    existing = ls.open_temporal_cube(root)
    np.testing.assert_array_equal(existing.read_layer(0)[0], _cube().values[0])
    assert not list(tmp_path.glob(".series.staging-*"))


def test_streaming_reducers_match_in_memory_reducers(tmp_path: Path) -> None:
    cube = _cube()
    series = ls.write_temporal_cube(tmp_path / "series", cube)

    for reducer in (ls.temporal_mean, ls.temporal_min, ls.temporal_max, ls.temporal_std):
        expected, _ = reducer(cube)
        actual, georef = reducer(series)
        np.testing.assert_allclose(actual, expected)
        assert georef.nodata == -99
    assert not series._layer_cache


def test_series_can_omit_vrt_and_rejects_invalid_cache_config(tmp_path: Path) -> None:
    root = ls.write_temporal_cube(tmp_path / "series", _cube(), create_vrt=False).root
    series = ls.open_temporal_cube(root)
    assert series.vrt_path is None
    assert not (root / "series.vrt").exists()
    with pytest.raises(ls.TemporalSeriesOpenError) as raised:
        ls.open_temporal_cube(root, max_open_datasets=0)
    assert raised.value.code == "temporal_series_invalid_cache_config"


def test_incremental_writer_finalizes_context_and_reports_progress(
    tmp_path: Path,
) -> None:
    cube = _cube()
    progress: list[ls.TemporalWriteProgress] = []
    writer = ls.TemporalGeoTiffSeriesWriter(
        tmp_path / "series",
        georef=cube.georef,
        dtype=cube.dtype,
        signal_name="sun_fraction",
        progress_callback=progress.append,
    )

    with writer:
        for index, time_value in enumerate(cube.times):
            returned_path = writer.write_layer(time_value, cube.values[index])
            assert returned_path.name.endswith("Z.tif")

    assert writer.result is not None
    assert writer.result.time_count == 3
    assert writer.layers_written == 3
    assert [item.layers_written for item in progress] == [1, 2, 3]
    assert [item.last_index for item in progress] == [0, 1, 2]
    assert progress[-1].last_time == cube.times[-1]
    assert progress[-1].layer_path == writer.result.layer_paths[-1]


@pytest.mark.parametrize(
    ("time_value", "values", "code"),
    [
        ("2027-01-01T01:00:00Z", np.zeros((3, 3), dtype=np.int16), "temporal_series_layer_shape_mismatch"),
        ("2027-01-01T01:00:00Z", np.zeros((2, 3), dtype=np.float32), "temporal_series_layer_dtype_mismatch"),
    ],
)
def test_incremental_writer_validation_failure_removes_staging(
    tmp_path: Path, time_value, values, code: str
) -> None:
    writer = ls.TemporalGeoTiffSeriesWriter(
        tmp_path / "series", georef=_georef(), dtype=np.int16
    )

    with pytest.raises(ls.TemporalSeriesWriteError) as raised:
        writer.write_layer(time_value, values)

    assert raised.value.code == code
    assert not (tmp_path / "series").exists()
    assert not list(tmp_path.glob(".series.staging-*"))
    with pytest.raises(ls.TemporalSeriesWriteError) as inactive:
        writer.finalize()
    assert inactive.value.code == "temporal_series_writer_inactive"


def test_incremental_writer_requires_increasing_times(tmp_path: Path) -> None:
    writer = ls.TemporalGeoTiffSeriesWriter(
        tmp_path / "series", georef=_georef(), dtype=np.int16
    )
    writer.write_layer("2027-01-01T01:00:00Z", np.ones((2, 3), dtype=np.int16))

    with pytest.raises(ls.TemporalSeriesWriteError) as raised:
        writer.write_layer(
            "2027-01-01T01:00:00Z", np.ones((2, 3), dtype=np.int16)
        )

    assert raised.value.code == "temporal_series_times_not_increasing"
    assert not list(tmp_path.glob(".series.staging-*"))


def test_incremental_writer_cancellation_aborts_and_preserves_existing(
    tmp_path: Path,
) -> None:
    cube = _cube()
    root = ls.write_temporal_cube(tmp_path / "series", cube).root
    cancel = False

    def cancellation_requested() -> bool:
        return cancel

    writer = ls.TemporalGeoTiffSeriesWriter(
        root,
        georef=cube.georef,
        dtype=cube.dtype,
        overwrite=True,
        cancellation_requested=cancellation_requested,
    )
    writer.write_layer(cube.times[0], cube.values[0])
    cancel = True

    with pytest.raises(ls.TemporalSeriesWriteError) as raised:
        writer.write_layer(cube.times[1], cube.values[1])

    assert raised.value.code == "temporal_series_write_cancelled"
    existing = ls.open_temporal_cube(root)
    np.testing.assert_array_equal(existing.read_layer(2)[0], cube.values[2])
    assert not list(tmp_path.glob(".series.staging-*"))


def test_incremental_writer_rejects_empty_finalize(tmp_path: Path) -> None:
    writer = ls.TemporalGeoTiffSeriesWriter(
        tmp_path / "series", georef=_georef(), dtype=np.int16
    )

    with pytest.raises(ls.TemporalSeriesWriteError) as raised:
        writer.finalize()

    assert raised.value.code == "temporal_series_empty"
    assert not list(tmp_path.glob(".series.staging-*"))


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_RUN_TEMPORAL_BENCHMARK") != "1",
    reason="set LUNARSCOUT_RUN_TEMPORAL_BENCHMARK=1 to run the 3,800-layer benchmark",
)
def test_benchmark_incremental_writer_3800_layers(tmp_path: Path) -> None:
    georef = ls.GeoReference(
        projection_wkt=_georef().projection_wkt,
        projection_proj4=_georef().projection_proj4,
        affine_transform=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0),
        width=1,
        height=1,
        pixel_size_x=1.0,
        pixel_size_y=-1.0,
        nodata=None,
    )
    value = np.ones((1, 1), dtype=np.float32)
    start = np.datetime64("2027-01-01T00:00:00", "us")
    started = time.perf_counter()
    writer = ls.TemporalGeoTiffSeriesWriter(
        tmp_path / "series", georef=georef, dtype=value.dtype
    )
    for index in range(3_800):
        writer.write_layer(start + np.timedelta64(index, "m"), value)
    series = writer.finalize()
    elapsed_seconds = time.perf_counter() - started

    assert series.time_count == 3_800
    assert series.vrt_path is not None
    assert series.vrt_path.stat().st_size > 0
    print(f"3,800-layer temporal series: {elapsed_seconds:.3f} seconds")
