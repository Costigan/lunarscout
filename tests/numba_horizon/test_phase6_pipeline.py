from __future__ import annotations

from io import StringIO
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from lunarscout._numba_horizon import file_format
from lunarscout._numba_horizon.file_format import (
    AZIMUTH_COUNT,
    HorizonTileStore,
    PIXEL_COUNT,
    RAW_FILE_BYTES,
    _encode_horizons,
    _encode_horizons_python,
    read_horizon_tile,
)
from lunarscout._numba_horizon.pipeline import (
    HorizonPipelineCancelled,
    enumerate_patches,
    run_bounded_pipeline,
)
from lunarscout.scenario import Scenario


class _FakeStore:
    def __init__(self, existing: set[tuple[int, int]] | None = None) -> None:
        self.existing = existing or set()
        self.writes: list[tuple[int, int, int, int]] = []

    def find_existing_path(self, tile_y, tile_x, _observer):
        if (tile_y, tile_x) in self.existing:
            return Path(f"horizon_{tile_y}_{tile_x}.bin")
        return None

    def write(
        self,
        tile_y,
        tile_x,
        _observer,
        _degrees,
        *,
        compress,
        valid_width,
        valid_height,
    ):
        self.writes.append((tile_y, tile_x, valid_width, valid_height))
        suffix = ".cbin" if compress else ".bin"
        return Path(f"horizon_{tile_y:05d}_{tile_x:05d}_000{suffix}")


def test_patch_enumeration_matches_csharp_for_aligned_dem() -> None:
    patches = enumerate_patches(256, 128)

    assert [(item.index, item.tile_x, item.tile_y) for item in patches] == [
        (0, 0, 0),
        (1, 128, 0),
    ]
    assert all((item.width, item.height) == (128, 128) for item in patches)


def test_patch_enumeration_retains_partial_right_and_bottom_edges() -> None:
    patches = enumerate_patches(300, 257)

    assert len(patches) == 9
    assert [(item.tile_x, item.width) for item in patches[:3]] == [
        (0, 128),
        (128, 128),
        (256, 44),
    ]
    assert [(item.tile_y, item.height) for item in patches[::3]] == [
        (0, 128),
        (128, 128),
        (256, 1),
    ]
    with pytest.raises(ValueError, match="even multiples"):
        enumerate_patches(300, 257, include_partial_edges=False)


def test_store_uses_csharp_naming_precedence_and_structural_completion(tmp_path: Path) -> None:
    store = HorizonTileStore(tmp_path)
    partitioned_raw = store.build_path(128, 256, 0.59, compress=False)
    legacy_compressed = tmp_path / store.build_file_name(128, 256, 0.59, compress=True)
    partitioned_raw.parent.mkdir(parents=True)
    partitioned_raw.write_bytes(b"incomplete")
    legacy_compressed.write_bytes(b"incomplete")

    assert partitioned_raw.name == "horizon_00128_00256_005.bin"
    assert store.find_existing_path(128, 256, 0.59) is None
    assert store.find_existing_path(128, 256, 0.59, require_complete=False) == partitioned_raw

    partitioned_raw.write_bytes(b"\0" * RAW_FILE_BYTES)
    assert store.find_existing_path(128, 256, 0.59) == partitioned_raw


def test_numba_compressor_matches_reference_payload() -> None:
    values = np.linspace(-60.0, 60.0, AZIMUTH_COUNT, dtype=np.float32)[None, :]
    compiled, compiled_lengths = _encode_horizons(values)
    reference, reference_lengths = _encode_horizons_python(values)
    length = int(reference_lengths[0])

    np.testing.assert_array_equal(compiled_lengths, reference_lengths)
    np.testing.assert_array_equal(compiled[0, :length], reference[0, :length])


def test_staged_write_preserves_existing_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = HorizonTileStore(tmp_path)
    final_path = store.build_path(0, 0, 0.0, compress=False)
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"completed product")

    def fail_after_partial_write(handle, _degrees):
        handle.write(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(file_format, "_write_uncompressed", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        store.write(
            0,
            0,
            0.0,
            np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
            compress=False,
            valid_width=1,
            valid_height=1,
        )

    assert final_path.read_bytes() == b"completed product"
    assert list(final_path.parent.glob("*.tmp.bin")) == []


def test_uncompressed_partial_tile_is_readable_and_padded(tmp_path: Path) -> None:
    store = HorizonTileStore(tmp_path)
    degrees = np.full((PIXEL_COUNT, AZIMUTH_COUNT), 99.0, dtype=np.float32)
    degrees[0] = 1.25
    degrees[1] = 2.5
    path = store.write(
        0, 0, 0.0, degrees, compress=False, valid_width=2, valid_height=1
    )

    assert path.stat().st_size == RAW_FILE_BYTES
    assert store.is_complete(path)
    with path.open("rb") as handle:
        np.testing.assert_array_equal(
            Scenario.horizon_from_open_file(handle, 1, 0), degrees[1]
        )
        np.testing.assert_array_equal(
            Scenario.horizon_from_open_file(handle, 2, 0),
            np.full(AZIMUTH_COUNT, -50.0, dtype=np.float32),
        )


def test_compressed_tile_is_readable_by_existing_scenario_reader(tmp_path: Path) -> None:
    store = HorizonTileStore(tmp_path)
    degrees = np.full((1, AZIMUTH_COUNT), 5.9, dtype=np.float32)
    path = store.write(
        0, 0, 0.0, degrees, compress=True, valid_width=1, valid_height=1
    )

    assert store.is_complete(path)
    assert path.stat().st_size < RAW_FILE_BYTES
    with path.open("rb") as handle:
        actual = Scenario.horizon_from_open_file(handle, 0, 0)
    np.testing.assert_allclose(actual, degrees[0], atol=0.002)


@pytest.mark.parametrize("compress", [False, True])
def test_full_tile_reader_returns_fixed_pixel_major_cube(
    tmp_path: Path, compress: bool
) -> None:
    store = HorizonTileStore(tmp_path)
    degrees = np.stack(
        (
            np.linspace(-10.0, 10.0, AZIMUTH_COUNT, dtype=np.float32),
            np.linspace(5.0, -5.0, AZIMUTH_COUNT, dtype=np.float32),
        )
    )
    path = store.write(
        0,
        0,
        0.0,
        degrees,
        compress=compress,
        valid_width=2,
        valid_height=1,
    )

    actual = read_horizon_tile(path)

    assert actual.shape == (128, 128, AZIMUTH_COUNT)
    assert actual.dtype == np.float32
    tolerance = 0.002 if compress else 0.0
    np.testing.assert_allclose(actual[0, :2], degrees, atol=tolerance)
    np.testing.assert_allclose(actual[0, 2], -50.0, atol=tolerance)
    np.testing.assert_allclose(actual[1, 0], -50.0, atol=tolerance)
    np.testing.assert_allclose(store.read(0, 0, 0.0), actual)


def test_full_tile_reader_rejects_trailing_compressed_data(tmp_path: Path) -> None:
    store = HorizonTileStore(tmp_path)
    path = store.write(
        0,
        0,
        0.0,
        np.zeros((1, AZIMUTH_COUNT), dtype=np.float32),
        compress=True,
        valid_width=1,
        valid_height=1,
    )
    with path.open("ab") as handle:
        handle.write(b"trailing")

    with pytest.raises(ValueError, match="trailing bytes"):
        read_horizon_tile(path)


def test_bounded_pipeline_skips_complete_tiles_streams_and_flushes_progress() -> None:
    patches = enumerate_patches(384, 128)
    store = _FakeStore(existing={(0, 128)})
    prepared: list[int] = []
    progress = []
    stream = StringIO()

    def prepare(patch):
        prepared.append(patch.index)
        return patch.index

    def factory(worker_id):
        def process(_patch, payload):
            time.sleep(0.01)
            return np.asarray([[payload + worker_id]], dtype=np.float32)

        return process

    result = run_bounded_pipeline(
        patches,
        store=store,
        prepare_patch=prepare,
        processor_factory=factory,
        skip_existing=True,
        prepared_queue_capacity=1,
        worker_count=1,
        progress_callback=progress.append,
        progress_stream=stream,
    )

    assert prepared == [0, 2]
    assert result.skipped_patches == 1
    assert result.maximum_prepared_queue_depth <= 1
    assert len(result.producer_enqueue_wait_seconds) == 2
    assert len(result.consumer_dequeue_wait_seconds) == 2
    assert result.writer_enqueue_wait_seconds == ()
    assert store.writes == [(0, 0, 128, 128), (0, 256, 128, 128)]
    assert [item.stage for item in progress] == [
        "prepare_patches",
        "process_patches",
        "process_patches",
        "process_patches",
        "complete",
    ]
    assert progress[-2].file_name == "horizon_00000_00256_000.bin"
    assert stream.getvalue().endswith("Horizons generation complete.\n")


def test_pipeline_cancellation_after_compute_writes_no_incomplete_product() -> None:
    patch = enumerate_patches(128, 128)
    store = _FakeStore()
    cancelled = threading.Event()

    def factory(_worker_id):
        def process(_patch, _payload):
            cancelled.set()
            return np.zeros((1, 1), dtype=np.float32)

        return process

    with pytest.raises(HorizonPipelineCancelled):
        run_bounded_pipeline(
            patch,
            store=store,
            prepare_patch=lambda item: item.index,
            processor_factory=factory,
            cancellation_requested=cancelled.is_set,
        )

    assert store.writes == []


def test_bounded_writer_queue_finalizes_and_writes_off_compute_thread() -> None:
    patches = enumerate_patches(256, 128)
    store = _FakeStore()
    compute_threads = []
    finalize_threads = []

    def factory(_worker_id):
        def process(_patch, payload):
            compute_threads.append(threading.current_thread().name)
            return payload

        return process

    def finalize(_patch, payload):
        finalize_threads.append(threading.current_thread().name)
        return np.asarray([[payload]], dtype=np.float32)

    result = run_bounded_pipeline(
        patches,
        store=store,
        prepare_patch=lambda item: item.index,
        processor_factory=factory,
        finalize_patch=finalize,
        prepared_queue_capacity=1,
        writer_queue_capacity=1,
    )

    assert compute_threads == ["horizon-gpu-worker-0"] * 2
    assert finalize_threads == ["horizon-output-writer"] * 2
    assert result.maximum_writer_queue_depth <= 1
    assert len(result.writer_enqueue_wait_seconds) == 2
    assert len(result.writer_dequeue_wait_seconds) == 2
    assert len(result.output_paths) == 2


def test_pipeline_worker_failure_drains_bounded_queue_without_writing() -> None:
    patches = enumerate_patches(384, 128)
    store = _FakeStore()

    def factory(_worker_id):
        def process(_patch, _payload):
            raise RuntimeError("simulated CUDA failure")

        return process

    with pytest.raises(RuntimeError, match="simulated CUDA failure"):
        run_bounded_pipeline(
            patches,
            store=store,
            prepare_patch=lambda item: item.index,
            processor_factory=factory,
            prepared_queue_capacity=1,
        )

    assert store.writes == []
