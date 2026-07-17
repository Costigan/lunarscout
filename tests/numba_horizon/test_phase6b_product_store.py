from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from lunarscout.georeference import GeoReference
from lunarscout.geotiff import read_geotiff
import lunarscout._numba_horizon.product_store as product_store
from lunarscout._numba_horizon.product_store import (
    IncompatibleProductJobError,
    ProductJob,
    ResumableTiledProduct,
    TIMESTAMPS_TAG,
    TIMESTAMP_TAG,
)


def _georef(width: int = 257, height: int = 130) -> GeoReference:
    return GeoReference(
        projection_wkt='PROJCS["test",GEOGCS["g",DATUM["d",SPHEROID["s",1737400,0]],PRIMEM["p",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        projection_proj4="+proj=eqc +R=1737400 +units=m +no_defs",
        affine_transform=(0.0, 20.0, 0.0, 0.0, 0.0, -20.0),
        width=width,
        height=height,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )


def _job(*, algorithm: str = "lightmap-test") -> ProductJob:
    return ProductJob(
        georef=_georef(),
        dtype=np.uint8,
        band_count=2,
        timestamps_utc=("2027-01-01T00:00:00Z", "2027-01-01T06:00:00+00:00"),
        invalid_value=7,
        algorithm=algorithm,
        configuration={"solar_disk": "uniform"},
        horizon_inventory_identity="sha256:test-inventory",
    )


def test_product_store_resumes_by_patch_and_publishes_partial_edges(
    tmp_path: Path,
) -> None:
    output = tmp_path / "lightmap.tif"
    store = ResumableTiledProduct(output, _job())
    store.write_patch(
        0,
        0,
        (
            np.full((128, 128), 11, dtype=np.uint8),
            np.full((128, 128), 22, dtype=np.uint8),
        ),
    )

    resumed = ResumableTiledProduct(output, _job())
    assert resumed.is_complete(0, 0)
    assert resumed.completed_patches == {"0,0": "valid"}
    resumed.write_invalid_patch(0, 128)
    resumed.write_invalid_patch(0, 256)
    resumed.write_invalid_patch(128, 0)
    resumed.write_invalid_patch(128, 128)
    resumed.write_patch(
        128,
        256,
        (
            np.full((2, 1), 33, dtype=np.uint8),
            np.full((2, 1), 44, dtype=np.uint8),
        ),
    )
    assert resumed.finalize() == output

    assert not resumed.staging_path.exists()
    assert not resumed.manifest_path.exists()
    assert not resumed.journal_path.exists()
    with rasterio.open(output) as dataset:
        assert dataset.count == 2
        assert dataset.block_shapes == [(128, 128), (128, 128)]
        assert dataset.profile["interleave"] == "band"
        assert dataset.tags()[TIMESTAMPS_TAG] == (
            '["2027-01-01T00:00:00.000000Z","2027-01-01T06:00:00.000000Z"]'
        )
        assert dataset.tags(1)[TIMESTAMP_TAG] == "2027-01-01T00:00:00.000000Z"
        assert dataset.tags(2)[TIMESTAMP_TAG] == "2027-01-01T06:00:00.000000Z"
        assert np.all(dataset.read(1, window=((0, 128), (0, 128))) == 11)
        assert np.all(dataset.read(2, window=((0, 128), (0, 128))) == 22)
        assert np.all(dataset.read(1, window=((0, 128), (128, 256))) == 7)
        assert np.all(dataset.dataset_mask(window=((0, 128), (128, 256))) == 0)
        assert np.all(dataset.read(1, window=((128, 130), (256, 257))) == 33)
        assert np.all(dataset.read(2, window=((128, 130), (256, 257))) == 44)
        assert np.all(dataset.dataset_mask(window=((128, 130), (256, 257))) == 255)
    second_band, second_georef = read_geotiff(output, band=2)
    assert second_georef is not None
    assert (second_georef.width, second_georef.height) == (257, 130)
    assert second_band[0, 0] == 22
    assert second_band[129, 256] == 44


def test_product_store_does_not_journal_a_partially_written_patch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "partial.tif"
    store = ResumableTiledProduct(output, _job())

    def one_band_only():
        yield np.ones((128, 128), dtype=np.uint8)

    with pytest.raises(ValueError, match="fewer entries"):
        store.write_patch(0, 0, one_band_only())

    resumed = ResumableTiledProduct(output, _job())
    assert not resumed.is_complete(0, 0)
    resumed.write_patch(
        0,
        0,
        (
            np.full((128, 128), 3, dtype=np.uint8),
            np.full((128, 128), 4, dtype=np.uint8),
        ),
    )
    assert resumed.is_complete(0, 0)


def test_product_store_rejects_an_incompatible_resume_job(tmp_path: Path) -> None:
    output = tmp_path / "incompatible.tif"
    original = ResumableTiledProduct(output, _job(algorithm="first"))
    original.write_invalid_patch(0, 0)

    with pytest.raises(IncompatibleProductJobError, match="does not match"):
        ResumableTiledProduct(output, _job(algorithm="second"))

    assert original.staging_path.exists()
    fresh = ResumableTiledProduct(
        output,
        _job(algorithm="second"),
        start_fresh=True,
    )
    assert fresh.completed_patches == {}


def test_journal_failure_does_not_advance_patch_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "journal-failure.tif"
    store = ResumableTiledProduct(output, _job())
    real_atomic_json = product_store._atomic_json

    def fail_journal(path, value):
        if path == store.journal_path:
            raise OSError("simulated journal failure")
        real_atomic_json(path, value)

    monkeypatch.setattr(product_store, "_atomic_json", fail_journal)
    with pytest.raises(OSError, match="simulated journal failure"):
        store.write_invalid_patch(0, 0)

    assert store.completed_patches == {}
