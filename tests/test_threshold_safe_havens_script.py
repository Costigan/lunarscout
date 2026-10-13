from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine


pytest.importorskip("dask")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "threshold_safe_havens.py"
spec = importlib.util.spec_from_file_location("threshold_safe_havens", SCRIPT)
script = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = script
spec.loader.exec_module(script)


def _write_input(path, values, *, dtype="uint8", nodata=255):
    count, height, width = values.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=count,
        dtype=dtype, nodata=nodata, crs="EPSG:4326",
        transform=Affine(0.2, 0.01, 30, 0.03, -0.2, 60),
        tiled=True, blockxsize=128, blockysize=128, interleave="band",
    ) as target:
        target.write(values.astype(dtype))
        target.update_tags(MONTHS="January,February", STATISTICS_MEAN="99")
        target.scales = (2.0,) * count
        target.offsets = (1.0,) * count
        for band in target.indexes:
            target.set_band_description(band, f"Month {band}")
            target.set_band_unit(band, "two-hour-counts")
            target.update_tags(band, TIMESTAMP_UTC=f"2024-{band:02d}-01T00:00:00Z",
                               STATISTICS_MINIMUM="99")


def _minimum_reference(values):
    count, height, width = values.shape
    output = np.full((count, (height + 1) // 2, (width + 1) // 2), 255, dtype=np.uint8)
    for y in range(output.shape[1]):
        for x in range(output.shape[2]):
            output[:, y, x] = values[:, y * 2:y * 2 + 2, x * 2:x * 2 + 2].min(axis=(1, 2))
    return output


def test_multiband_threshold_and_every_minimum_overview(tmp_path, monkeypatch):
    # Odd edges, several chunk boundaries, different months, all-NODATA and
    # mixed footprints, zero, exact threshold, threshold+1, and 254/255.
    values = np.random.default_rng(123).integers(0, 256, (2, 35, 53), dtype=np.uint8)
    values[:, :4, :4] = 255
    values[0, :2, 4:6] = [[255, script.THRESHOLD], [script.THRESHOLD + 1, 254]]
    values[1, :2, 4:6] = [[0, 255], [script.THRESHOLD, script.THRESHOLD + 1]]
    source, output_directory = tmp_path / "input.tif", tmp_path / "months"
    _write_input(source, values)
    monkeypatch.setattr(script, "CHUNK_SIZE", 16)
    monkeypatch.setattr(script, "MIN_OVERVIEW_SIZE", 4)
    outputs = script.convert(source, output_directory, workers=3)
    assert [path.name for path in outputs] == [
        "safe_havens_2024-01-01T00-00-00.tif",
        "safe_havens_2024-02-01T00-00-00.tif",
    ]
    assert sorted(output_directory.iterdir()) == list(outputs)
    expected = np.where(values <= script.THRESHOLD, values, 255).astype(np.uint8)
    for band, output in enumerate(outputs, start=1):
        expected_month = expected[band - 1:band]
        with rasterio.open(source) as original, rasterio.open(output) as actual:
            np.testing.assert_array_equal(actual.read(), expected_month)
            assert actual.count == 1
            assert actual.dtypes == ("uint8",)
            assert actual.nodatavals == (255,)
            assert actual.transform == original.transform
            assert actual.crs == original.crs
            assert actual.descriptions == (original.descriptions[band - 1],)
            assert actual.units == (original.units[band - 1],)
            assert actual.scales == (original.scales[band - 1],)
            assert actual.offsets == (original.offsets[band - 1],)
            assert actual.block_shapes == [(1024, 1024)]
            assert actual.profile["compress"] == "deflate"
            assert actual.profile["interleave"] == "band"
            assert "STATISTICS_MEAN" not in actual.tags()
            assert "STATISTICS_MINIMUM" not in actual.tags(1)
            assert actual.tags(1)["TIMESTAMP_UTC"] == original.tags(band)["TIMESTAMP_UTC"]
            assert actual.tags()["TIMESTAMP_UTC"] == original.tags(band)["TIMESTAMP_UTC"]
            assert actual.tags()["SAFE_HAVEN_SOURCE_BAND"] == str(band)
            assert actual.tags(ns="rio_overview")["resampling"] == "min"
            assert len(actual.overviews(1)) == 4
        for level in range(4):
            expected_month = _minimum_reference(expected_month)
            with rasterio.open(output, overview_level=level) as overview:
                np.testing.assert_array_equal(overview.read(), expected_month)
                assert overview.profile["compress"] == "deflate"
        with output.open("rb") as stream:
            header = stream.read(4)
        assert header in (b"II+\x00", b"MM\x00+")
        assert not Path(str(output) + ".ovr").exists()


@pytest.mark.parametrize("threshold", [0, 254, 255])
def test_byte_threshold_limits_and_all_nodata(tmp_path, threshold):
    values = np.array([[[0, 1, 24, 25, 254, 255]], [[255] * 6]], dtype=np.uint8)
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    _write_input(source, values)
    outputs = script.convert(source, output_directory, workers=2, threshold=threshold)
    for band, output in enumerate(outputs):
        with rasterio.open(output) as actual:
            np.testing.assert_array_equal(actual.read(), np.where(values[band:band + 1] <= threshold, values[band:band + 1], 255))
            assert actual.overviews(1) == []


@pytest.mark.parametrize("threshold", [-1, 256, 1.5, True])
def test_invalid_threshold_rejected_before_output(tmp_path, threshold):
    with pytest.raises(ValueError, match="threshold"):
        script.convert(tmp_path / "missing.tif", tmp_path / "output.tif", threshold=threshold)
    assert not (tmp_path / "output.tif").exists()


@pytest.mark.parametrize("dtype,nodata", [("uint16", 255), ("uint8", 0), ("uint8", None)])
def test_wrong_input_contract_rejected(tmp_path, dtype, nodata):
    source = tmp_path / "source.tif"
    _write_input(source, np.zeros((1, 2, 2)), dtype=dtype, nodata=nodata)
    with pytest.raises(ValueError):
        script.convert(source, tmp_path / "output.tif")


def test_output_preflight_and_failed_overwrite_preserve_product(tmp_path, monkeypatch):
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    _write_input(source, np.zeros((1, 300, 300), dtype=np.uint8))
    with pytest.raises(NotADirectoryError):
        script.convert(source, source, overwrite=True)
    output, = script.convert(source, output_directory)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        script.convert(source, output_directory)

    original_kernel = script._minimum2

    def fail(values):
        if values.shape == (2, 2):
            return original_kernel(values)
        raise RuntimeError("simulated pyramid failure")

    monkeypatch.setattr(script, "_minimum2", fail)
    with pytest.raises(RuntimeError, match="pyramid failure"):
        script.convert(source, output_directory, overwrite=True)
    assert output.read_bytes() == original
    assert list(output_directory.iterdir()) == [output]


def test_cli_fresh_process(tmp_path):
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    values = np.full((2, 513, 517), 255, dtype=np.uint8)
    values[0, 0, 0] = 24
    values[0, -1, -1] = 0  # Leave the second month entirely NODATA.
    _write_input(source, values)
    subprocess.run([sys.executable, str(SCRIPT), str(source), str(output_directory), "--workers", "2"],
                   check=True, capture_output=True, text=True)
    for band, stamp in enumerate(("2024-01-01T00-00-00", "2024-02-01T00-00-00")):
        output = output_directory / f"safe_havens_{stamp}.tif"
        with rasterio.open(output, overview_level=1) as actual:
            expected = _minimum_reference(_minimum_reference(values[band:band + 1]))
            np.testing.assert_array_equal(actual.read(), expected)


def test_dask_tasks_run_on_multiple_threads(tmp_path, monkeypatch):
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    _write_input(source, np.zeros((1, 32, 32), dtype=np.uint8))
    monkeypatch.setattr(script, "CHUNK_SIZE", 16)
    barrier = threading.Barrier(2)
    threads = set()
    original = script._read_chunk

    def read(*args):
        threads.add(threading.get_ident())
        barrier.wait(timeout=10)
        return original(*args)

    monkeypatch.setattr(script, "_read_chunk", read)
    script.convert(source, output_directory, workers=2)
    assert len(threads) == 2


@pytest.mark.parametrize("timestamp", ["", "not-a-date", "2024-01-01T00:00:00", "2024-02-30T00:00:00Z"])
def test_invalid_later_timestamp_fails_before_any_output(tmp_path, timestamp):
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    _write_input(source, np.zeros((2, 2, 2), dtype=np.uint8))
    with rasterio.open(source, "r+") as target:
        target.update_tags(2, TIMESTAMP_UTC=timestamp)
    with pytest.raises(ValueError, match="Band 2.*TIMESTAMP_UTC"):
        script.convert(source, output_directory)
    assert not output_directory.exists()


def test_missing_band_timestamp_rejected(tmp_path):
    source = tmp_path / "source.tif"
    with rasterio.open(source, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint8", nodata=255, transform=Affine.translation(1, 1)) as target:
        target.write(np.zeros((1, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="missing TIMESTAMP_UTC"):
        script.convert(source, tmp_path / "months")


def test_utc_filename_preserves_fractional_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "OUTPUT_FILENAME_TEMPLATE", "sun_image_{timestamp:%Y-%m-%dT%H-%M-%S.%f}.tif")
    source = tmp_path / "source.tif"
    _write_input(source, np.zeros((1, 2, 2), dtype=np.uint8))
    timestamp = "2024-01-01T01:02:03.123456+01:00"
    with rasterio.open(source, "r+") as target:
        target.update_tags(1, TIMESTAMP_UTC=timestamp)
    output, = script.convert(source, tmp_path / "months")
    assert output.name == "sun_image_2024-01-01T00-02-03.123456.tif"
    with rasterio.open(output) as target:
        assert target.tags(1)["TIMESTAMP_UTC"] == timestamp


def test_duplicate_timestamp_after_utc_normalization_rejected(tmp_path):
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    _write_input(source, np.zeros((2, 2, 2), dtype=np.uint8))
    with rasterio.open(source, "r+") as target:
        target.update_tags(2, TIMESTAMP_UTC="2024-01-01T01:00:00.000000+01:00")
    with pytest.raises(ValueError, match="duplicate output filenames"):
        script.convert(source, output_directory)
    assert not output_directory.exists()


@pytest.mark.parametrize("suffix", ["", ".ovr", ".msk", ".aux.xml"])
def test_later_destination_conflict_is_preflighted(tmp_path, suffix):
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    _write_input(source, np.zeros((2, 2, 2), dtype=np.uint8))
    output_directory.mkdir()
    conflict = output_directory / f"safe_havens_2024-02-01T00-00-00.tif{suffix}"
    conflict.write_bytes(b"existing product or sidecar")
    with pytest.raises(FileExistsError):
        script.convert(source, output_directory)
    assert list(output_directory.iterdir()) == [conflict]
    assert conflict.read_bytes() == b"existing product or sidecar"


def test_later_month_failure_preserves_earlier_completed_month(tmp_path, monkeypatch):
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    _write_input(source, np.zeros((2, 300, 300), dtype=np.uint8))
    original = script._convert_month

    def convert_month(input_path, staging, band, **kwargs):
        if band == 2:
            raise RuntimeError("simulated second month failure")
        return original(input_path, staging, band, **kwargs)

    monkeypatch.setattr(script, "_convert_month", convert_month)
    with pytest.raises(RuntimeError, match="second month failure"):
        script.convert(source, output_directory)
    outputs = list(output_directory.iterdir())
    assert len(outputs) == 1
    assert outputs[0].name == "safe_havens_2024-01-01T00-00-00.tif"
    with rasterio.open(outputs[0], overview_level=0) as month:
        assert month.count == 1
        assert np.all(month.read(1) == 0)


def test_editable_template_matches_requested_example(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "OUTPUT_FILENAME_TEMPLATE", "sun_image_{timestamp:%Y-%m-%dT%H-%M-%S}.tif")
    source = tmp_path / "source.tif"
    _write_input(source, np.zeros((1, 2, 2), dtype=np.uint8))
    with rasterio.open(source, "r+") as target:
        target.update_tags(1, TIMESTAMP_UTC="2027-09-01T00:00:00.000000Z")
    output, = script.convert(source, tmp_path / "months")
    assert output.name == "sun_image_2027-09-01T00-00-00.tif"
    with rasterio.open(output) as month:
        assert month.count == 1
        assert month.tags(1)["TIMESTAMP_UTC"] == "2027-09-01T00:00:00.000000Z"


@pytest.mark.parametrize("template", [
    "constant.tif", "../{timestamp:%Y}.tif", "{unknown}.tif",
    "image_{timestamp", "{timestamp:%Y-%m-%dT%H:%M:%S}.tif", "{timestamp:%Y}.png",
])
def test_bad_filename_template_rejected_before_output(tmp_path, monkeypatch, template):
    monkeypatch.setattr(script, "OUTPUT_FILENAME_TEMPLATE", template)
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    _write_input(source, np.zeros((1, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="OUTPUT_FILENAME_TEMPLATE"):
        script.convert(source, output_directory)
    assert not output_directory.exists()


def test_template_with_insufficient_timestamp_precision_rejects_collisions(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "OUTPUT_FILENAME_TEMPLATE", "image_{timestamp:%Y}.tif")
    source, output_directory = tmp_path / "source.tif", tmp_path / "months"
    _write_input(source, np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="duplicate output filenames"):
        script.convert(source, output_directory)
    assert not output_directory.exists()
