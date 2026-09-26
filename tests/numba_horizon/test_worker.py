from __future__ import annotations

import pytest

from lunarscout._numba_horizon import worker as worker_module


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
