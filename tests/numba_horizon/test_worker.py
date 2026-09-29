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
