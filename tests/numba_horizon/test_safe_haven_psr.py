"""PSR ids and sizes: only PSR pixels are labelled, regions are connected as asked, and the products read back."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import lunarscout as ls
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_psr as psr
from lunarscout.errors import InputError


def blobs() -> np.ndarray:
    """0 = not PSR, 255 = PSR: a 2x3 block, a lone pixel, a diagonal pair, and a ring around a hole."""
    values = np.zeros((20, 20), dtype=np.uint8)
    values[1:3, 1:4] = 255                    # 6 pixels
    values[10, 10] = 255                      # 1 pixel
    values[5, 15] = values[6, 16] = 255       # touches diagonally: 2 pixels with 8 neighbours, 2 regions with 4
    values[12:17, 3:8] = 255
    values[13:16, 4:7] = 0                    # a ring of 16 pixels around a 3 x 3 hole of not-PSR
    return values


def test_only_psr_pixels_are_labelled() -> None:
    values = blobs()
    mask = psr.psr_mask(values)
    assert np.array_equal(mask, values == 255) and mask.sum() == 6 + 1 + 2 + 16
    ids, count = psr.label_psr(mask)
    assert count == 4 and ids.dtype == np.int32
    assert (ids[~mask] == 0).all() and (ids[mask] > 0).all()                       # background is 0, PSR pixels never 0
    assert set(np.unique(ids[mask])) == {1, 2, 3, 4}
    assert (ids[13:16, 4:7] == 0).all()                                            # the hole inside the ring is not a PSR
    assert ids[1, 1] == 1 and ids[5, 15] == ids[6, 16] == 2 and ids[10, 10] == 3 and ids[12, 3] == 4   # raster-scan order


def test_four_neighbour_connectivity_splits_the_diagonal_pair() -> None:
    ids, count = psr.label_psr(psr.psr_mask(blobs()), connectivity=4)
    assert count == 5 and ids[5, 15] != ids[6, 16]
    with pytest.raises(Exception):
        psr.label_psr(psr.psr_mask(blobs()), connectivity=6)


def test_a_map_that_is_inverted_or_foreign_is_refused() -> None:
    inverted = np.where(blobs() == 255, 0, 255).astype(np.uint8)
    with pytest.raises(InputError) as caught:
        psr.psr_mask(inverted)
    assert caught.value.code == "psr_map_inverted"
    other = blobs()
    other[0, 0] = 7
    with pytest.raises(InputError) as caught:
        psr.psr_mask(other)
    assert caught.value.code == "psr_map_values"
    with pytest.raises(InputError):
        psr.psr_mask(np.zeros(5, dtype=np.uint8))
    with pytest.raises(InputError):
        psr.psr_mask(blobs(), psr_value=0)
    assert psr.psr_mask(np.zeros((4, 4), dtype=np.uint8)).sum() == 0              # no PSR at all is not an error
    assert psr.psr_mask((blobs() > 0).astype(np.uint8), psr_value=1).sum() == 25  # another PSR value


def test_the_region_table_matches_a_brute_force_count() -> None:
    from scipy import ndimage

    rng = np.random.default_rng(4)
    values = np.where(rng.random((60, 70)) < 0.3, 255, 0).astype(np.uint8)
    mask = psr.psr_mask(values)
    ids, count = psr.label_psr(mask)
    transform = (1000.0, 20.0, 0.0, -500.0, 0.0, -20.0)
    table = psr.region_table(ids, count, transform)
    assert len(table) == count and (table.psr_id == np.arange(1, count + 1)).all()
    assert table.pixels.sum() == mask.sum() and (table.area_m2 == table.pixels * 400.0).all()
    for i in rng.integers(0, count, 25):
        rows, columns = np.nonzero(ids == i + 1)
        assert table.pixels[i] == len(rows)
        assert (table.row_min[i], table.row_max[i], table.col_min[i], table.col_max[i]) == (
            rows.min(), rows.max(), columns.min(), columns.max())
        assert table.x_m[i] == pytest.approx(1000.0 + (columns.mean() + 0.5) * 20.0)
        assert table.y_m[i] == pytest.approx(-500.0 + (rows.mean() + 0.5) * -20.0)
    assert np.allclose(
        [table.x_m[0], table.y_m[0]],
        [1000.0 + (ndimage.center_of_mass(ids == 1)[1] + 0.5) * 20.0, -500.0 - (ndimage.center_of_mass(ids == 1)[0] + 0.5) * 20.0])
    by_id = table.pixels_by_id()
    assert by_id[0] == 0 and (by_id[ids] == np.where(ids > 0, table.pixels[np.maximum(ids, 1) - 1], 0)).all()


def test_a_rotated_grid_has_no_pixel_area() -> None:
    ids, count = psr.label_psr(psr.psr_mask(blobs()))
    with pytest.raises(InputError):
        psr.region_table(ids, count, (0.0, 20.0, 1.0, 0.0, 0.0, -20.0))


def test_the_table_reads_back_and_refuses_to_overwrite(tmp_path: Path) -> None:
    ids, count = psr.label_psr(psr.psr_mask(blobs()))
    table = psr.region_table(ids, count, (0.0, 20.0, 0.0, 0.0, 0.0, -20.0))
    path = psr.write_region_table(tmp_path / "regions.csv", table)
    back = psr.read_region_table(path)
    assert list(back.pixels) == [6, 2, 1, 16] and list(back.area_m2) == [2400.0, 800.0, 400.0, 6400.0]
    for name in psr.REGION_COLUMNS:
        assert np.allclose(getattr(back, name), getattr(table, name), atol=0.01)
    with pytest.raises(InputError):
        psr.write_region_table(path, table)
    psr.write_region_table(path, table, overwrite=True)
    assert not list(tmp_path.glob("*.partial"))
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(InputError):
        psr.read_region_table(tmp_path / "bad.csv")


def test_the_products_are_written_on_the_maps_grid(tmp_path: Path) -> None:
    georef = scene.georef().with_nodata(None)
    values = np.zeros((20, 20), dtype=np.uint8)
    values[2:5, 2:5] = 255
    values[10, 10] = 255
    values[15:18, 0:2] = 255
    source = ls.write_geotiff(tmp_path / "psr.tif", values, georef)
    count, pixels = psr.make_psr_products(source, tmp_path / "ids.tif", tmp_path / "regions.csv")
    assert (count, pixels) == (3, 16)
    ids, ids_georef = ls.read_geotiff(tmp_path / "ids.tif")
    assert ids.dtype == np.int32 and ids_georef.affine_transform == georef.affine_transform and ids.shape == values.shape
    assert (ids[values == 0] == 0).all() and (ids[values == 255] > 0).all() and ids.max() == 3
    table = psr.read_region_table(tmp_path / "regions.csv")
    assert list(table.pixels) == [9, 1, 6] and (table.pixels_by_id()[ids] == np.where(ids > 0, np.array([0, 9, 1, 6])[ids], 0)).all()
    with pytest.raises(InputError):                                   # nothing is overwritten
        psr.make_psr_products(source, tmp_path / "ids.tif", tmp_path / "regions.csv")
    assert psr.make_psr_products(source, tmp_path / "ids.tif", tmp_path / "regions.csv", overwrite=True) == (3, 16)
    inverted = ls.write_geotiff(tmp_path / "inverted.tif", np.where(values == 255, 0, 255).astype(np.uint8), georef)
    with pytest.raises(InputError):
        psr.make_psr_products(inverted, tmp_path / "x.tif", tmp_path / "x.csv")
    assert not (tmp_path / "x.tif").exists() and not (tmp_path / "x.csv").exists()


def test_the_script_writes_the_products_and_importing_it_starts_nothing(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "generate_psr_regions", Path(__file__).resolve().parents[2] / "scripts" / "generate_psr_regions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.PSR_VALUE_IN_MAP == 255 and module.CONNECTIVITY == 8 and module.OVERWRITE is False
    values = np.zeros((20, 20), dtype=np.uint8)
    values[3:6, 3:6] = 255
    source = ls.write_geotiff(tmp_path / "psr.tif", values, scene.georef().with_nodata(None))
    for name, value in dict(PSR_PATH=source, IDS_PATH=tmp_path / "ids.tif", TABLE_PATH=tmp_path / "regions.csv").items():
        monkeypatch.setattr(module, name, value)
    assert module.main() == 0
    assert "1 PSR regions from 9 PSR pixels" in (tmp_path / "regions.csv.log").read_text()
    with pytest.raises(InputError):
        module.main()                                                  # the outputs now exist
