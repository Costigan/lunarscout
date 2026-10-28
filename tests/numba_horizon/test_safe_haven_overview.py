"""The persistence overview maps: the grid from the seed table, the backdrop, the composition and the files."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from lunarscout._numba_horizon import safe_haven_overview as ov
from lunarscout.errors import InputError

CRS = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
GT = (-1500.0, 20.0, 0.0, 1500.0, 0.0, -20.0)          # 300 x 300 pixels of 20 m, the pole 75 pixels from the top-left


def write_seeds(path: Path, seeds) -> Path:
    """``seeds``: (month, cell_row, cell_col) per seed."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed_id", "month", "cell_row", "cell_col", "row", "column"])
        for number, (month, cell_row, cell_col) in enumerate(seeds, start=1):
            writer.writerow([number, month, cell_row, cell_col, 0, 0])
    return path


def write_dem(path: Path, height=300, width=300) -> None:
    rows, columns = np.mgrid[0:height, 0:width]
    elevation = (0.5 * rows + 4.0 * np.sin(columns / 17.0)).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32", crs=CRS,
                       transform=Affine.from_gdal(*GT)) as dataset:
        dataset.write(elevation[None])


def write_psr(path: Path, height=300, width=300) -> None:
    ids = np.zeros((height, width), dtype=np.int32)
    ids[140:200, 20:100] = 1
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype="int32", crs=CRS,
                       transform=Affine.from_gdal(*GT)) as dataset:
        dataset.write(ids[None])


# --- the colours -------------------------------------------------------------------------------------------

def test_the_ramp_is_one_hue_brighter_for_more_months() -> None:
    ramp = ov.persistence_ramp()
    assert ramp.shape == (13, 3) and tuple(ramp[1]) == (0x2a, 0x78, 0xd6) and tuple(ramp[12]) == (0xcd, 0xe2, 0xfb)
    luminance = ramp[1:] @ np.array([0.2126, 0.7152, 0.0722])
    assert (np.diff(luminance) > 0).all()                                    # every month count is brighter than the last
    assert len({tuple(row) for row in ramp[1:]}) == 12 and (ramp[1:, 2] > ramp[1:, 0]).all()   # twelve colours, all blue


# --- persistence from the table -----------------------------------------------------------------------------

def test_persistence_counts_the_distinct_months_of_each_year_per_cell(tmp_path) -> None:
    seeds = [("2026-01", 0, 1)] * 3 + [("2026-02", 0, 1), ("2026-05", 0, 1), ("2026-12", 1, 0), ("2027-03", 0, 1),
                                        ("2025-07", 1, 1)]
    grids, counts = ov.persistence_grids(write_seeds(tmp_path / "s.csv", seeds), cell_shape=(2, 2))
    assert sorted(grids) == [2025, 2026, 2027] and counts == {2025: 1, 2026: 6, 2027: 1}
    assert grids[2026].tolist() == [[0, 3], [1, 0]]                              # three months, not six seeds
    assert grids[2027].tolist() == [[0, 1], [0, 0]] and grids[2025].tolist() == [[0, 0], [0, 1]]
    assert grids[2026].dtype == np.uint8


def test_a_year_between_the_first_and_last_is_an_empty_grid_and_years_can_be_chosen(tmp_path) -> None:
    path = write_seeds(tmp_path / "s.csv", [("2026-01", 0, 0), ("2028-01", 1, 1)])
    grids, _ = ov.persistence_grids(path, cell_shape=(2, 2))
    assert sorted(grids) == [2026, 2027, 2028] and not grids[2027].any()
    chosen, counts = ov.persistence_grids(path, cell_shape=(2, 2), years=[2028, 2030])
    assert sorted(chosen) == [2028, 2030] and counts == {2028: 1, 2030: 0} and not chosen[2030].any()


def test_twelve_months_in_one_cell_is_twelve(tmp_path) -> None:
    path = write_seeds(tmp_path / "s.csv", [(f"2026-{m:02d}", 0, 0) for m in range(1, 13)])
    grids, _ = ov.persistence_grids(path, cell_shape=(1, 1))
    assert grids[2026][0, 0] == 12


def test_bad_tables_and_settings_are_refused(tmp_path) -> None:
    with pytest.raises(InputError) as error:
        ov.persistence_grids(write_seeds(tmp_path / "a.csv", [("2026-01", 2, 0)]), cell_shape=(2, 2))
    assert error.value.code == "overview_cell_grid_mismatch"
    with pytest.raises(InputError) as error:
        ov.persistence_grids(write_seeds(tmp_path / "b.csv", [("2026-13", 0, 0)]), cell_shape=(2, 2))
    assert error.value.code == "overview_table_unreadable"
    (tmp_path / "c.csv").write_text("seed_id,month\n1,2026-01\n")
    with pytest.raises(InputError) as error:
        ov.persistence_grids(tmp_path / "c.csv", cell_shape=(2, 2))
    assert error.value.code == "overview_table_unreadable" and "cell_row" in str(error.value)
    (tmp_path / "d.csv").write_text("seed_id,month,cell_row,cell_col\n")
    with pytest.raises(InputError) as error:
        ov.persistence_grids(tmp_path / "d.csv", cell_shape=(2, 2))
    assert error.value.code == "overview_table_empty"
    with pytest.raises(InputError) as error:
        ov.persistence_grids(write_seeds(tmp_path / "e.csv", [("2026-01", 0, 0)]), cell_shape=(2, 2), years=[2026, 2026])
    assert error.value.code == "overview_setting_invalid"


def test_the_geotiff_is_on_the_cell_grid_and_replaces_a_stale_sidecar(tmp_path) -> None:
    path = tmp_path / "p.tif"
    (tmp_path / "p.tif.aux.xml").write_text("<PAMDataset/>")
    grid = np.array([[0, 3], [12, 1]], dtype=np.uint8)
    ov.write_persistence_geotiff(path, grid, transform=(-1500.0, 2560.0, 0.0, 1500.0, 0.0, -2560.0), crs_wkt=rasterio.crs.CRS.from_string(CRS).to_wkt(),
                                 tags={"YEAR": "2026"})
    assert not (tmp_path / "p.tif.aux.xml").exists()
    with rasterio.open(path) as dataset:
        assert dataset.read(1).tolist() == [[0, 3], [12, 1]] and dataset.dtypes == ("uint8",)
        assert dataset.transform.a == 2560.0 and dataset.transform.e == -2560.0 and dataset.tags()["YEAR"] == "2026"
        assert dataset.descriptions == ("months_with_a_seed",)


# --- the backdrop and the composition -------------------------------------------------------------------------

def test_block_reduce_means_the_blocks_ignoring_nan_and_pads_the_edge() -> None:
    strip = np.arange(25, dtype=np.float32).reshape(5, 5)
    strip[0, 0] = np.nan
    out = ov.block_reduce(strip, 2, 3, 3)
    assert out.shape == (3, 3)
    assert out[0, 0] == pytest.approx((1 + 5 + 6) / 3)                          # the nan is left out
    assert out[2, 2] == pytest.approx(24.0)                                     # a corner block of one pixel
    assert out[2, 0] == pytest.approx((20 + 21) / 2)
    assert np.isnan(ov.block_reduce(np.zeros((0, 0), dtype=np.float32), 2, 1, 1)[0, 0])
    ones = np.zeros((4, 4), dtype=np.float32)
    ones[:2, :2] = 1.0
    assert ov.block_reduce(ones, 2, 2, 2).tolist() == [[1.0, 0.0], [0.0, 0.0]]  # on a 0/1 raster, the share of ones


def test_the_hillshade_is_dark_and_the_surface_shows_where_there_is_no_dem() -> None:
    elevation = np.full((24, 24), np.nan, dtype=np.float32)
    rows, columns = np.mgrid[0:16, 0:16]
    elevation[:16, :16] = 40.0 * np.sin(columns / 3.0) + 3.0 * rows
    image = ov.backdrop_rgb(elevation, None, 320.0)
    assert image.shape == (24, 24, 3) and (image[:16, :16] <= 0.2401).all() and (image[:16, :16] >= 0.0699).all()
    assert image[:16, :16, 0].std() > 0.01                                      # there is relief
    assert np.allclose(image[20, 20], np.array([0x1a, 0x1a, 0x19]) / 255.0)    # surface colour outside the DEM


def test_psr_blocks_are_darker_with_a_muted_outline() -> None:
    elevation = np.full((12, 12), 100.0, dtype=np.float32)
    psr = np.zeros((12, 12), dtype=np.float32)
    psr[3:9, 3:9] = 1.0
    psr[0, 0] = 0.1                                                             # under the fraction: not a PSR
    image = ov.backdrop_rgb(elevation, psr, 320.0)
    assert np.allclose(image[5, 5], np.array([0x0d, 0x0d, 0x0d]) / 255.0)       # inside
    assert image[3, 5, 0] > image[5, 5, 0] and image[3, 5, 0] > 0.2             # the outline is lighter than the inside
    assert np.allclose(image[0, 0], image[11, 11])                              # a faint block is the plain hillshade


def test_the_cells_are_squares_inset_from_their_edges_in_the_ramp_colour() -> None:
    backdrop = np.full((16, 24, 3), 0.1, dtype=np.float32)
    grid = np.array([[0, 3, 12]], dtype=np.uint8)[:, :]
    grid = np.vstack([grid, np.zeros((1, 3), dtype=np.uint8)])
    image = ov.compose(backdrop, grid, 8)
    ramp = ov.persistence_ramp()
    assert image.shape == (16, 24, 3)
    assert tuple(image[4, 12]) == tuple(ramp[3]) and tuple(image[4, 20]) == tuple(ramp[12])
    assert tuple(image[0, 12]) == (26, 26, 26) and tuple(image[4, 8]) == (26, 26, 26)      # the one-pixel inset shows the terrain
    assert tuple(image[4, 9]) == tuple(ramp[3]) and tuple(image[6, 14]) == tuple(ramp[3])
    assert (image[8:] == 26).all() and (image[:, :8] == 26).all()                # no seed, no square


# --- the polar graticule ---------------------------------------------------------------------------------------

def test_the_parallels_and_meridians_fall_where_the_projection_puts_them() -> None:
    radius = 1737400.0
    assert ov.stereographic_radius(-90.0, radius) == 0.0
    # the DEM's corner (-304000, 304000) is at 75 deg 53' 37.7" S, 45 deg W (gdalinfo)
    assert ov.stereographic_radius(-(75 + 53 / 60 + 37.7 / 3600), radius) == pytest.approx(math.hypot(304000, 304000), rel=1e-4)
    geometry = ov.Geometry((-304000.0, 20.0, 0.0, 304000.0, 0.0, -20.0), 16, 256, (119, 119), CRS)
    assert geometry.pixel_metres == 320.0 and geometry.pole == pytest.approx((950.0, 950.0))
    column, row = geometry.point(-75 - 53 / 60 - 37.7 / 3600, -45.0)
    assert (column, row) == pytest.approx((0.0, 0.0), abs=1.0)                  # the upper left corner of the DEM
    column, row = geometry.point(-85.0, 90.0)                                    # 90 E is to the right of the pole
    assert column > geometry.pole[0] and row == pytest.approx(geometry.pole[1], abs=1e-6)
    column, row = geometry.point(-85.0, 0.0)                                     # 0 is toward the top
    assert row < geometry.pole[1] and column == pytest.approx(geometry.pole[0], abs=1e-6)
    with pytest.raises(InputError):
        ov.Geometry((0.0, 20.0, 5.0, 0.0, 0.0, -20.0), 16, 256, (119, 119), CRS)


# --- the run ------------------------------------------------------------------------------------------------

@pytest.fixture()
def scene(tmp_path):
    pytest.importorskip("matplotlib")
    write_dem(tmp_path / "dem.tif")
    write_psr(tmp_path / "psr.tif")
    seeds = [("2026-01", 0, 0), ("2026-02", 0, 0), ("2026-02", 0, 0), ("2026-07", 1, 2), ("2027-05", 2, 2),
             ("2027-06", 2, 2), ("2027-06", 2, 1)]
    write_seeds(tmp_path / "seeds.csv", seeds)
    return tmp_path


def run(scene, **options):
    options.setdefault("cell_patches", 1)
    options.setdefault("overwrite", True)
    return ov.run_overview_maps(scene / "out", seeds_path=scene / "seeds.csv", dem_path=scene / "dem.tif",
                                psr_ids_path=scene / "psr.tif", **options)


def test_a_run_writes_a_geotiff_and_a_png_for_each_year_and_a_contact_sheet(scene) -> None:
    from PIL import Image

    report = run(scene)
    assert sorted(report.years) == [2026, 2027] and report.cell_shape == [3, 3] and report.cell_pixels == 128
    out = scene / "out"
    for year in (2026, 2027):
        with rasterio.open(out / f"safe_haven_stage_2_persistence_{year}.tif") as dataset:
            assert dataset.shape == (3, 3) and dataset.transform.a == 2560.0 and dataset.crs is not None
            grid = dataset.read(1)
        assert grid.tolist() == {2026: [[2, 0, 0], [0, 0, 1], [0, 0, 0]], 2027: [[0, 0, 0], [0, 0, 0], [0, 1, 2]]}[year]
        with Image.open(out / f"safe_haven_stage_2_persistence_{year}.png") as image:
            assert image.size[0] > 1500 and image.size[1] > 1500 and image.mode in ("RGB", "RGBA")
    with Image.open(out / "safe_haven_stage_2_persistence_contact_sheet.png") as sheet:
        assert sheet.size[0] > sheet.size[1] > 500
    saved = json.loads((out / "safe_haven_stage_2_persistence.report.json").read_text())
    assert saved["years"]["2026"]["seeds"] == 4 and saved["years"]["2026"]["cells_with_seeds"] == 2
    assert saved["years"]["2027"]["cells_by_months"][2] == 1 and saved["years"]["2027"]["cells_by_months"][1] == 1
    assert saved["years"]["2026"]["mean_months_of_cells_with_seeds"] == pytest.approx(1.5)
    assert not list(out.glob("*.partial"))


def test_the_png_shows_the_ramp_colours_of_the_cells(scene) -> None:
    from PIL import Image

    run(scene, years=[2026])
    with Image.open(scene / "out" / "safe_haven_stage_2_persistence_2026.png") as image:
        pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)
    ramp = ov.persistence_ramp()
    colours = {tuple(c) for c in np.unique(pixels, axis=0)}
    assert tuple(ramp[2]) in colours and tuple(ramp[1]) in colours              # the cells with 2 and 1 months (and the legend)


def test_years_can_be_chosen_and_an_empty_year_is_an_empty_map(scene) -> None:
    report = run(scene, years=[2027, 2030])
    assert sorted(report.years) == [2027, 2030] and report.years[2030]["cells_with_seeds"] == 0
    with rasterio.open(scene / "out" / "safe_haven_stage_2_persistence_2030.tif") as dataset:
        assert not dataset.read(1).any()


def test_a_run_refuses_to_replace_files_unless_told_and_refuses_bad_settings(scene) -> None:
    run(scene, years=[2026])
    with pytest.raises(InputError) as error:
        run(scene, years=[2026], overwrite=False)
    assert error.value.code == "overview_output_exists"
    with pytest.raises(InputError) as error:
        run(scene, overview_pixels_per_cell=24)                                 # does not divide 128
    assert error.value.code == "overview_setting_invalid"
    with pytest.raises(InputError) as error:
        run(scene, cell_patches=2)                                              # stage 2's cell was 128 px: cell 2 is outside
    assert error.value.code == "overview_cell_grid_mismatch"
    write_psr_small = scene / "psr_small.tif"
    write_psr(write_psr_small, 200, 200)
    with pytest.raises(InputError) as error:
        ov.run_overview_maps(scene / "out2", seeds_path=scene / "seeds.csv", dem_path=scene / "dem.tif",
                             psr_ids_path=write_psr_small, cell_patches=1)
    assert error.value.code == "overview_psr_mismatch"


def test_without_a_psr_raster_the_maps_are_still_made(scene) -> None:
    report = ov.run_overview_maps(scene / "out3", seeds_path=scene / "seeds.csv", dem_path=scene / "dem.tif",
                                  psr_ids_path=None, cell_patches=1, years=[2026])
    assert report.years[2026]["cells_with_seeds"] == 2 and (scene / "out3" / "safe_haven_stage_2_persistence_2026.png").exists()
