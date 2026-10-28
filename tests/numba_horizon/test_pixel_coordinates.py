"""x_m and y_m: pixel-centre coordinates in the DEM's CRS for the stage-2 and stage-4 tables, and the script that
adds them to an existing table."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import Affine

from lunarscout._numba_horizon import pixel_coordinates as pc
from lunarscout._numba_horizon import safe_haven_plan as plan
from lunarscout.errors import InputError
from lunarscout.products import _load_dem
from test_safe_haven_floods import seeds  # noqa: F401  (a stage-2 table of the scene)
from test_safe_haven_stage3_store import prepared  # noqa: F401

GT = (1000.0, 20.0, 0.0, -500.0, 0.0, -20.0)
SKEWED = (-5000.0, 17.0, 3.0, 2000.0, -2.0, -19.0)         # rotated and skewed, to catch swapped terms


def write_table(path: Path, rows, header=("id", "row", "column", "other")):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read(path: Path):
    with path.open(newline="") as handle:
        return list(csv.reader(handle))


def test_the_centre_of_a_pixel_is_half_a_pixel_in_from_its_corner() -> None:
    x, y = pc.pixel_centres(GT, [0, 4], [0, 12])
    assert list(x) == [1010.0, 1250.0] and list(y) == [-510.0, -590.0]
    for transform in (GT, SKEWED):
        rows, columns = np.array([0, 7, 300]), np.array([5, 0, 123])
        x, y = pc.pixel_centres(transform, rows, columns)
        expected = [Affine.from_gdal(*transform) * (c + 0.5, r + 0.5) for r, c in zip(rows, columns)]
        assert np.allclose(x, [e[0] for e in expected]) and np.allclose(y, [e[1] for e in expected])
    with pytest.raises(InputError) as caught:
        pc.pixel_centres((1, 2, 3), 0, 0)
    assert caught.value.code == "coordinates_transform_invalid"


def test_the_transform_read_from_the_dem_equals_the_one_the_products_use(prepared) -> None:  # noqa: F811
    path = prepared.directory / "dem.tif"
    dem = _load_dem(path)[0]
    assert tuple(dem.geo_transform) == pc.dem_geo_transform(path)
    for row, column in ((0, 0), (17, 250), (299, 299)):
        assert pc.pixel_centres(dem.geo_transform, row, column) == pytest.approx(dem.pixel_to_crs(column, row))


def test_the_columns_are_added_after_column_and_the_other_columns_are_untouched(tmp_path) -> None:
    table = tmp_path / "t.csv"
    write_table(table, [(1, 0, 0, "a b"), (2, 4, 12, "x,y"), (3, "", "", "")])
    assert pc.add_coordinate_columns(table, GT) == 3
    assert read(table) == [["id", "row", "column", "x_m", "y_m", "other"], ["1", "0", "0", "1010.000", "-510.000", "a b"],
                           ["2", "4", "12", "1250.000", "-590.000", "x,y"], ["3", "", "", "", "", ""]]


def test_existing_columns_are_overwritten_in_place_and_a_rerun_changes_nothing(tmp_path) -> None:
    table = tmp_path / "t.csv"
    write_table(table, [(1, 0, 0, "-1", "-2", "k"), (2, 4, 12, "-1", "-2", "k")],
                header=("id", "row", "column", "x_m", "y_m", "other"))
    pc.add_coordinate_columns(table, GT)
    assert read(table)[1:] == [["1", "0", "0", "1010.000", "-510.000", "k"], ["2", "4", "12", "1250.000", "-590.000", "k"]]
    before = table.read_bytes()
    pc.add_coordinate_columns(table, GT)
    assert table.read_bytes() == before
    pc.add_coordinate_columns(table, SKEWED)                     # a different transform overwrites them
    assert table.read_bytes() != before and read(table)[0] == ["id", "row", "column", "x_m", "y_m", "other"]


def test_columns_in_other_positions_stay_where_they_are(tmp_path) -> None:
    table = tmp_path / "t.csv"
    write_table(table, [(1, 0, "-1", 4, "k", "-2")], header=("id", "row", "x_m", "column", "other", "y_m"))
    pc.add_coordinate_columns(table, GT)
    assert read(table) == [["id", "row", "x_m", "column", "other", "y_m"], ["1", "0", "1090.000", "4", "k", "-510.000"]]


def test_a_table_with_only_one_of_the_two_gets_both_after_the_later_of_row_and_column(tmp_path) -> None:
    table = tmp_path / "t.csv"
    write_table(table, [(1, 3, "-7", 2, "k")], header=("id", "column", "x_m", "row", "other"))
    pc.add_coordinate_columns(table, GT)
    assert read(table) == [["id", "column", "row", "x_m", "y_m", "other"], ["1", "3", "2", "1070.000", "-550.000", "k"]]


def test_an_output_path_leaves_the_original_and_a_failure_leaves_it_too(tmp_path) -> None:
    table, out = tmp_path / "t.csv", tmp_path / "o.csv"
    write_table(table, [(1, 0, 0, "a")])
    original = table.read_bytes()
    pc.add_coordinate_columns(table, GT, output_path=out)
    assert table.read_bytes() == original and read(out)[1][3:5] == ["1010.000", "-510.000"]
    write_table(table, [(1, 0, 0, "a"), (2, "q", 1, "b")])           # a row that is not a number
    broken = table.read_bytes()
    with pytest.raises(ValueError):
        pc.add_coordinate_columns(table, GT)
    assert table.read_bytes() == broken and not list(tmp_path.glob("*.partial"))
    no_columns = tmp_path / "n.csv"
    no_columns.write_text("a,b\n1,2\n")
    with pytest.raises(InputError) as caught:
        pc.add_coordinate_columns(no_columns, GT)
    assert caught.value.code == "coordinates_columns_missing"


# --- the stage-2 table ---------------------------------------------------------------------------------------

def test_the_stage_2_table_holds_the_pixel_centres(prepared, seeds) -> None:  # noqa: F811
    path, rows = seeds
    assert plan.SEED_COLUMNS[4:8] == ("row", "column", "x_m", "y_m") and rows
    for seed in rows:
        assert (seed["x_m"], seed["y_m"]) == pytest.approx(prepared.dem.pixel_to_crs(seed["column"], seed["row"]), abs=1e-3)
    assert read(path)[0] == list(plan.SEED_COLUMNS)


def test_a_table_made_before_the_coordinates_is_read_and_the_script_restores_it_byte_for_byte(prepared, seeds, tmp_path) -> None:  # noqa: F811
    path, rows = seeds
    old = tmp_path / "old.csv"
    lines = read(path)
    drop = {lines[0].index("x_m"), lines[0].index("y_m")}
    with old.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows([[v for i, v in enumerate(line) if i not in drop] for line in lines])
    assert [s["seed_id"] for s in plan.read_seed_table(old)] == [s["seed_id"] for s in rows]
    assert "x_m" not in plan.read_seed_table(old)[0]
    pc.add_coordinate_columns(old, tuple(_load_dem(prepared.directory / "dem.tif")[0].geo_transform))
    assert old.read_bytes() == path.read_bytes()
    bad = tmp_path / "bad.csv"
    bad.write_text("seed_id,x_m\n1,2\n")
    with pytest.raises(InputError) as caught:
        plan.read_seed_table(bad)
    assert caught.value.code == "plan_table_unreadable"


# --- the script ----------------------------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "add_pixel_coordinates.py"


def load_script():
    spec = importlib.util.spec_from_file_location("add_pixel_coordinates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_defaults_name_the_stage_2_table_and_importing_it_starts_nothing() -> None:
    module = load_script()
    two = importlib.util.spec_from_file_location("run_safe_haven_stage2", SCRIPT.parent / "run_safe_haven_stage2.py")
    stage2 = importlib.util.module_from_spec(two)
    two.loader.exec_module(stage2)
    assert module.TABLE_PATH == stage2.OUTPUT_PATH and module.DEM_PATH == stage2.DEM_PATH and module.OUTPUT_PATH is None


def test_the_script_adds_the_columns_in_place_or_to_a_new_table(tmp_path, monkeypatch) -> None:
    import rasterio

    module = load_script()
    dem = tmp_path / "dem.tif"
    with rasterio.open(dem, "w", driver="GTiff", height=8, width=8, count=1, dtype="float32",
                       transform=Affine.from_gdal(*GT)) as dataset:
        dataset.write(np.zeros((1, 8, 8), dtype="float32"))
    table = tmp_path / "t.csv"
    write_table(table, [(1, 0, 0, "a"), (2, 4, 12, "b")])
    monkeypatch.setattr(module, "DEM_PATH", dem)
    monkeypatch.setattr(module, "TABLE_PATH", table)
    monkeypatch.setattr(module, "OUTPUT_PATH", tmp_path / "new.csv")
    assert module.main() == 0
    assert read(table)[0] == ["id", "row", "column", "other"] and read(tmp_path / "new.csv")[2][3:5] == ["1250.000", "-590.000"]
    monkeypatch.setattr(module, "OUTPUT_PATH", None)
    assert module.main() == 0                                                  # in place
    assert read(table) == read(tmp_path / "new.csv")
    assert "x_m and y_m written for 2 rows" in (tmp_path / "t.csv.coordinates.log").read_text()
    write_table(table, [(1, 0, 0, "a")], header=("id", "a", "b", "c"))
    with pytest.raises(InputError):
        module.main()
