from __future__ import annotations

import pytest

from lunarscout._numba_horizon import worker as worker_module


def test_node_label_combines_hostname_and_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOSTNAME", "pod-abc123")
    monkeypatch.setenv("NODE_NAME", "gpu-node-7")
    monkeypatch.delenv("POD_NAME", raising=False)
    assert worker_module._node_label() == "pod-abc123 node=gpu-node-7"


def test_node_label_falls_back_to_pod_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.setenv("POD_NAME", "pod-fallback")
    monkeypatch.delenv("NODE_NAME", raising=False)
    assert worker_module._node_label() == "pod-fallback"


def test_node_label_unknown_when_no_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.delenv("POD_NAME", raising=False)
    monkeypatch.delenv("NODE_NAME", raising=False)
    assert worker_module._node_label() == "unknown node"


def test_describe_gpu_formats_device_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from lunarscout import cuda as cuda_module

    fake = cuda_module.CudaStatus(
        True,
        "0.60.0",
        "NVIDIA GeForce RTX 3090",
        (8, 6),
        None,
        cuda_driver_version="525.60",
        total_memory_bytes=25 * 2**30,
    )
    monkeypatch.setattr(cuda_module, "status", lambda: fake)
    description = worker_module._describe_gpu()
    assert "RTX 3090" in description
    assert "sm_86" in description
    assert "driver 525.60" in description


def test_describe_gpu_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from lunarscout import cuda as cuda_module

    fake = cuda_module.CudaStatus(False, None, None, None, "no usable device")
    monkeypatch.setattr(cuda_module, "status", lambda: fake)
    assert worker_module._describe_gpu() == "gpu: unavailable (no usable device)"


def test_run_horizon_partition_is_verbose_and_partitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_generate_horizons(output, dem_paths, **kwargs):
        captured["output"] = output
        captured["dem_paths"] = dem_paths
        captured["kwargs"] = kwargs
        return output

    monkeypatch.setattr(worker_module, "generate_horizons", fake_generate_horizons)

    result = worker_module.run_horizon_partition(
        "/data/dem.tif",
        "/data/horizons",
        surrounding_dems=["/data/outer.tif"],
        observer_height_m=1.5,
        compress=True,
        pod_count=4,
        pod_nth=1,
    )

    assert result == "/data/horizons"
    assert captured["dem_paths"] == ["/data/dem.tif", "/data/outer.tif"]
    # Progress output with the completion-time estimate is a hard requirement.
    assert captured["kwargs"]["verbose"] is True
    assert captured["kwargs"]["patch_offset"] == 1
    assert captured["kwargs"]["patch_stride"] == 4
    assert captured["kwargs"]["observer_height_m"] == 1.5
    assert captured["kwargs"]["compress"] is True


def test_main_rejects_an_index_outside_pod_count(tmp_path) -> None:
    with pytest.raises(SystemExit) as raised:
        worker_module.main([
            "--primary-dem", str(tmp_path / "dem.tif"),
            "--output", str(tmp_path / "out"),
            "--pod-count", "2",
            "--pod-nth", "2",
        ])
    assert raised.value.code == 2


def test_main_stops_between_patches_on_sigterm(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import os
    import signal

    from lunarscout.errors import OperationCancelledError

    seen = {}

    def fake_generate_horizons(output, dem_paths, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        seen["cancelled"] = kwargs["cancellation_requested"]()
        seen["skip"] = kwargs["skip_failed_patches"]
        raise OperationCancelledError("cancelled", code="horizon_generation_cancelled")

    monkeypatch.setattr(worker_module, "generate_horizons", fake_generate_horizons)
    original = signal.getsignal(signal.SIGTERM)
    try:
        status = worker_module.main([
            "--primary-dem", str(tmp_path / "dem.tif"),
            "--output", str(tmp_path / "out"),
            "--pod-count", "1",
            "--pod-nth", "0",
        ])
    finally:
        signal.signal(signal.SIGTERM, original)

    assert status == worker_module.EXIT_TERMINATED == 143
    assert seen == {"cancelled": True, "skip": True}
    assert "TERMINATED: pod 0/1" in capsys.readouterr().out


def test_main_removes_orphaned_staging_files_and_reports_coverage(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import os
    import signal

    import numpy as np
    import rasterio

    from lunarscout._numba_horizon.file_format import AZIMUTH_COUNT, HorizonTileStore

    dem = tmp_path / "dem.tif"
    with rasterio.open(
        dem, "w", driver="GTiff", width=200, height=100, count=1, dtype="float32"
    ) as dataset:
        dataset.write(np.zeros((1, 100, 200), dtype=np.float32))
    output = tmp_path / "out"
    store = HorizonTileStore(output)
    store.write(0, 0, 0.0, np.zeros((1, AZIMUTH_COUNT), np.float32),
                compress=True, valid_width=1, valid_height=1)
    row = output / "00000"
    old = row / f"horizon_00000_00128_000.{'a' * 32}.tmp.cbin"
    fresh = row / f"horizon_00000_00128_000.{'b' * 32}.tmp.bin"
    unrelated = row / "notes.tmp.cbin"
    for path in (old, fresh, unrelated):
        path.write_bytes(b"partial")
    os.utime(old, (0, 0))
    os.utime(unrelated, (0, 0))
    monkeypatch.setattr(worker_module, "generate_horizons", lambda output, *_a, **_k: output)
    original = signal.getsignal(signal.SIGTERM)
    try:
        status = worker_module.main([
            "--primary-dem", str(dem), "--output", str(output),
            "--pod-count", "2", "--pod-nth", "0",
        ])
    finally:
        signal.signal(signal.SIGTERM, original)

    out = capsys.readouterr().out
    assert status == 0
    assert not old.exists() and fresh.exists() and unrelated.exists()
    assert "removed 1 orphaned staging files" in out
    assert "coverage: 1 of 2 tiles present" in out
    assert "check POD_COUNT" in out
