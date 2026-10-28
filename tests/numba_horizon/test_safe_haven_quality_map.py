"""The cell-quality overview maps: the longest mission per cell and year, the classes, the colours and the files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from lunarscout._numba_horizon import safe_haven_quality_map as qm
from lunarscout.errors import InputError

from test_safe_haven_overview import write_dem, write_psr


def write_chains(path: Path, seeds) -> Path:
    """``seeds``: (month, cell_row, cell_col, mission_hours) per seed."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed_id", "month", "cell_row", "cell_col", "mission_hours"])
        for number, (month, cell_row, cell_col, mission) in enumerate(seeds, start=1):
            writer.writerow([number, month, cell_row, cell_col, mission])
    return path


SEEDS = [("2026-01", 0, 0, 30.0), ("2026-02", 0, 0, 200.0), ("2026-02", 0, 0, 100.0), ("2026-07", 1, 2, 2000.0),
         ("2027-05", 2, 2, 50.0), ("2027-06", 2, 1, 600.0)]


def test_the_classes_follow_the_edges_in_days_and_no_seed_is_class_zero() -> None:
    hours = np.array([[np.nan, 0.0, 47.9, 48.0], [95.9, 96.0, 167.9, 168.0], [239.9, 240.0, 335.9, 336.0],
                      [503.9, 504.0, 1439.9, 1440.0]], dtype=np.float32)
    assert qm.quality_classes(hours).tolist() == [[0, 1, 1, 2], [2, 3, 3, 4], [4, 5, 5, 6], [6, 7, 7, 8]]
    assert len(qm.QUALITY_LABELS) == len(qm.QUALITY_EDGES_HOURS) + 1 == 8


def test_the_colours_are_one_hue_brighter_for_longer_missions() -> None:
    colours = qm.quality_colours()
    assert colours.shape == (9, 3)
    luma = colours[1:].astype(float) @ [0.2126, 0.7152, 0.0722]
    assert (np.diff(luma) > 0).all()
    assert (colours[1:, 0] >= colours[1:, 1]).all() and (colours[1:, 1] >= colours[1:, 2]).all()      # orange: R > G > B
    assert qm.quality_legend().labels == qm.QUALITY_LABELS and len(qm.quality_legend().colours) == 9


def test_quality_is_the_longest_mission_of_the_year_per_cell(tmp_path) -> None:
    path = write_chains(tmp_path / "chains.csv", SEEDS)
    grids, seeds = qm.quality_grids(path, cell_shape=(3, 3))
    assert sorted(grids) == [2026, 2027] and seeds == {2026: 4, 2027: 2}
    first = grids[2026]
    assert first.dtype == np.float32 and first[0, 0] == 200.0 and first[1, 2] == 2000.0
    assert np.isnan(first).sum() == 7
    assert grids[2027][2, 2] == 50.0 and grids[2027][2, 1] == 600.0 and np.isnan(grids[2027][0, 0])


def test_years_between_are_empty_and_bad_tables_are_refused(tmp_path) -> None:
    path = write_chains(tmp_path / "chains.csv", SEEDS)
    grids, seeds = qm.quality_grids(path, cell_shape=(3, 3), years=[2027, 2028])
    assert list(grids) == [2027, 2028] and seeds[2028] == 0 and np.isnan(grids[2028]).all()
    with pytest.raises(InputError) as error:
        qm.quality_grids(path, cell_shape=(2, 2))
    assert error.value.code == "overview_cell_grid_mismatch"
    with pytest.raises(InputError) as error:
        qm.quality_grids(path, cell_shape=(3, 3), years=[2026, 2026])
    assert error.value.code == "overview_setting_invalid"
    (tmp_path / "no_mission.csv").write_text("seed_id,month,cell_row,cell_col\n1,2026-01,0,0\n")
    with pytest.raises(InputError) as error:
        qm.quality_grids(tmp_path / "no_mission.csv", cell_shape=(3, 3))
    assert error.value.code == "overview_table_unreadable"
    write_chains(tmp_path / "bad.csv", [("2026-01", 0, 0, "x")])
    with pytest.raises(InputError):
        qm.quality_grids(tmp_path / "bad.csv", cell_shape=(3, 3))
    write_chains(tmp_path / "empty.csv", [])
    with pytest.raises(InputError) as error:
        qm.quality_grids(tmp_path / "empty.csv", cell_shape=(3, 3))
    assert error.value.code == "overview_table_empty"


@pytest.fixture()
def scene(tmp_path):
    pytest.importorskip("matplotlib")
    write_dem(tmp_path / "dem.tif")
    write_psr(tmp_path / "psr.tif")
    write_chains(tmp_path / "chains.csv", SEEDS)
    return tmp_path


def run(scene, **options):
    options.setdefault("cell_patches", 1)
    options.setdefault("overwrite", True)
    return qm.run_quality_maps(scene / "out", chains_path=scene / "chains.csv", dem_path=scene / "dem.tif",
                               psr_ids_path=scene / "psr.tif", **options)


def test_a_run_writes_a_geotiff_and_a_png_per_year_a_contact_sheet_and_a_report(scene) -> None:
    from PIL import Image

    report = run(scene)
    out = scene / "out"
    assert sorted(report.years) == [2026, 2027] and report.classes_days == list(qm.QUALITY_LABELS)
    for year in (2026, 2027):
        with rasterio.open(out / f"safe_haven_stage_4_cell_quality_{year}.tif") as dataset:
            assert dataset.shape == (3, 3) and dataset.transform.a == 2560.0 and dataset.crs is not None
            assert dataset.dtypes == ("float32",) and dataset.descriptions == ("longest_mission_hours",)
            hours = dataset.read(1)
        assert np.isnan(hours).sum() == 7
        with Image.open(out / f"safe_haven_stage_4_cell_quality_{year}.png") as image:
            assert image.size[0] > 1500 and image.mode in ("RGB", "RGBA")
    with Image.open(out / "safe_haven_stage_4_cell_quality_contact_sheet.png") as sheet:
        assert sheet.size[0] > sheet.size[1] > 500
    saved = json.loads((out / "safe_haven_stage_4_cell_quality.report.json").read_text())
    entry = saved["years"]["2026"]
    assert entry["seeds"] == 4 and entry["cells_with_seeds"] == 2 and entry["longest_hours"] == 2000.0
    assert entry["cells_by_class"][4] == 1 and entry["cells_by_class"][8] == 1        # 200 h: class 4 (7-10 d); 2000 h: 60+ d
    assert not list(out.glob("*.partial"))


def test_the_png_shows_the_class_colours_of_the_cells(scene) -> None:
    from PIL import Image

    run(scene, years=[2026])
    with Image.open(scene / "out" / "safe_haven_stage_4_cell_quality_2026.png") as image:
        colours = {tuple(c) for c in np.unique(np.asarray(image.convert("RGB")).reshape(-1, 3), axis=0)}
    ramp = qm.quality_colours()
    assert tuple(ramp[4]) in colours and tuple(ramp[8]) in colours


def test_a_run_replaces_a_stale_sidecar_refuses_to_overwrite_and_refuses_bad_settings(scene) -> None:
    run(scene, years=[2026])
    tif = scene / "out" / "safe_haven_stage_4_cell_quality_2026.tif"
    sidecar = tif.with_name(tif.name + ".aux.xml")
    sidecar.write_text("<PAMDataset/>")
    run(scene, years=[2026])
    assert not sidecar.exists()
    with pytest.raises(InputError) as error:
        run(scene, years=[2026], overwrite=False)
    assert error.value.code == "overview_output_exists"
    with pytest.raises(InputError) as error:
        run(scene, overview_pixels_per_cell=7)
    assert error.value.code == "overview_setting_invalid"


def test_without_a_psr_raster_the_maps_are_still_made(scene) -> None:
    qm.run_quality_maps(scene / "out2", chains_path=scene / "chains.csv", dem_path=scene / "dem.tif", years=[2027],
                        cell_patches=1, overwrite=True)
    assert (scene / "out2" / "safe_haven_stage_4_cell_quality_2027.png").exists()
