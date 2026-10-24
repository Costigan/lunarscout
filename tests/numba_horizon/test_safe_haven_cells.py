"""Stage 2: the strip-wise scan for cells with havens, and the cell-list file."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio

from lunarscout._numba_horizon import safe_haven_cells as cells
from lunarscout._numba_horizon.safe_haven_windows import Stage1Rasters, haven_cells

HEIGHT, WIDTH, BANDS, CELL = 70, 100, 5, 32          # 3 x 4 cells, the last row and column partial


def write_raster(path, data, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=data.dtype, nodata=nodata) as dataset:
        dataset.write(data)


def make_products(directory, kind):
    rng = np.random.default_rng(11)
    if kind == "byte":
        hibernation = rng.integers(0, 200, size=(BANDS, HEIGHT, WIDTH)).astype(np.uint8)
        hibernation[rng.random(hibernation.shape) < 0.3] = 255            # NODATA
        hibernation[:, :, 70:] = 255                                      # a block with no havens at all
        hibernation[2, 5, 5] = 36                                         # exactly 72 h
        hibernation[2, 5, 6] = 37                                         # 74 h
        joint = np.zeros_like(hibernation)
        options = dict(hours_per_count=2.0, nodata=255)
    else:
        hibernation = (rng.random((BANDS, HEIGHT, WIDTH)) * 400).astype(np.float32)
        hibernation[rng.random(hibernation.shape) < 0.3] = np.nan
        hibernation[:, :, 70:] = np.nan
        hibernation[2, 5, 5] = 72.0
        joint = np.zeros_like(hibernation)
        options = {}
    write_raster(directory / f"{kind}_hibernation.tif", hibernation)
    write_raster(directory / f"{kind}_joint.tif", joint)
    return Stage1Rasters(directory / f"{kind}_hibernation.tif", directory / f"{kind}_joint.tif", **options)


@pytest.fixture(scope="module")
def products(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cells")
    return SimpleNamespace(directory=directory, byte=make_products(directory, "byte"),
                           float=make_products(directory, "float"))


@pytest.mark.parametrize("kind", ["byte", "float"])
@pytest.mark.parametrize("limit", [0.0, 1.0, 71.9, 72.0, 73.9, 150.0, 399.0])
def test_the_strip_scan_equals_the_per_cell_scan(products, kind, limit) -> None:
    stage1 = getattr(products, kind)
    assert cells.scan_cells(stage1, survival_limit_hours=limit, cell_size=CELL) == haven_cells(
        stage1, survival_limit_hours=limit, cell_size=CELL)


@pytest.mark.parametrize("kind", ["byte", "float"])
def test_the_haven_mask_equals_the_decoded_comparison(products, kind) -> None:
    stage1 = getattr(products, kind)
    for window in [(0, HEIGHT, 0, WIDTH), (3, 40, 10, 77), (60, 70, 90, 100)]:
        decoded = stage1.read_hibernation_window(*window)
        with np.errstate(invalid="ignore"):
            expected = decoded <= 72.0
        assert np.array_equal(stage1.read_haven_window(*window, 72.0), expected)


def test_exactly_the_limit_is_a_haven_and_just_over_is_not_and_nodata_never_is(products) -> None:
    mask = products.byte.read_haven_window(0, HEIGHT, 0, WIDTH, 72.0)
    assert mask[2, 5, 5] and not mask[2, 5, 6]
    assert not mask[:, :, 70:].any()                    # the NODATA block
    # a limit between two counts rounds down to the lower one (73.9 h -> 72 h -> count 36)
    assert products.byte.read_haven_window(0, HEIGHT, 0, WIDTH, 73.9)[2, 5, 5]
    assert not products.byte.read_haven_window(0, HEIGHT, 0, WIDTH, 73.9)[2, 5, 6]
    assert not products.byte.read_haven_window(0, HEIGHT, 0, WIDTH, 71.9)[2, 5, 5]


def test_a_nodata_value_inside_the_threshold_is_still_not_a_haven(tmp_path) -> None:
    data = np.full((2, 4, 4), 5, dtype=np.uint8)
    data[0, 1, 1] = 0                                    # NODATA is 0 here, inside every limit
    write_raster(tmp_path / "h.tif", data)
    write_raster(tmp_path / "j.tif", data)
    stage1 = Stage1Rasters(tmp_path / "h.tif", tmp_path / "j.tif", hours_per_count=2.0, nodata=0)
    mask = stage1.read_haven_window(0, 4, 0, 4, 100.0)
    assert not mask[0, 1, 1] and mask[0, 0, 0] and mask.sum() == 31
    assert cells.scan_cells(stage1, survival_limit_hours=100.0, cell_size=4) == haven_cells(
        stage1, survival_limit_hours=100.0, cell_size=4)


def test_no_haven_anywhere_gives_an_empty_list_and_progress_counts_strips(products) -> None:
    calls = []
    assert cells.scan_cells(products.byte, survival_limit_hours=-1.0 + 0.0, cell_size=CELL,
                            progress=lambda d, n: calls.append((d, n))) == haven_cells(
        products.byte, survival_limit_hours=-1.0, cell_size=CELL) == []
    assert calls == [(1, 3), (2, 3), (3, 3)]
    with pytest.raises(ValueError):
        cells.scan_cells(products.byte, survival_limit_hours=72.0, cell_size=0)


def test_the_result_is_ordered_by_months_then_pixels_then_cell() -> None:
    found = [dict(cell_row=1, cell_col=0, months_with_haven=3, haven_pixels=5),
             dict(cell_row=0, cell_col=1, months_with_haven=3, haven_pixels=9),
             dict(cell_row=0, cell_col=0, months_with_haven=3, haven_pixels=5),
             dict(cell_row=2, cell_col=2, months_with_haven=7, haven_pixels=1)]
    assert [(e["cell_row"], e["cell_col"]) for e in cells.sort_cells(found)] == [(2, 2), (0, 1), (0, 0), (1, 0)]


# --- the cell-list file ------------------------------------------------------------------------


def test_a_cell_list_round_trips_and_is_tied_to_its_inputs(products, tmp_path) -> None:
    stage1 = products.byte
    found = cells.scan_cells(stage1, survival_limit_hours=72.0, cell_size=CELL)
    path = tmp_path / "cells.json"
    cells.write_cell_list(path, found, stage1, survival_limit_hours=72.0, cell_size=CELL)
    assert not path.with_name("cells.json.partial").exists()
    assert cells.read_cell_list(path, stage1, survival_limit_hours=72.0, cell_size=CELL) == found
    document = json.loads(path.read_text())
    assert document["raster"] == {"height": HEIGHT, "width": WIDTH, "bands": BANDS} and document["cell_size"] == CELL
    with pytest.raises(ValueError, match="survival_limit_hours"):
        cells.read_cell_list(path, stage1, survival_limit_hours=48.0, cell_size=CELL)
    with pytest.raises(ValueError, match="cell_size"):
        cells.read_cell_list(path, stage1, survival_limit_hours=72.0, cell_size=64)
    with pytest.raises(ValueError, match="stage1"):
        cells.read_cell_list(path, products.float, survival_limit_hours=72.0, cell_size=CELL)
    stat = stage1.hibernation_path.stat()
    os.utime(stage1.hibernation_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    try:
        with pytest.raises(ValueError, match="run stage 2 again"):
            cells.read_cell_list(path, stage1, survival_limit_hours=72.0, cell_size=CELL)
    finally:
        os.utime(stage1.hibernation_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cells.read_cell_list(path, stage1, survival_limit_hours=72.0, cell_size=CELL) == found


def test_the_cell_list_file_is_not_replaced_without_asking_and_bad_files_are_refused(products, tmp_path) -> None:
    stage1 = products.byte
    path = tmp_path / "cells.json"
    cells.write_cell_list(path, [], stage1, survival_limit_hours=72.0, cell_size=CELL)
    with pytest.raises(ValueError, match="exists"):
        cells.write_cell_list(path, [], stage1, survival_limit_hours=72.0, cell_size=CELL)
    cells.write_cell_list(path, [dict(cell_row=0, cell_col=0, months_with_haven=1, haven_pixels=1)], stage1,
                          survival_limit_hours=72.0, cell_size=CELL, overwrite=True)
    assert len(cells.read_cell_list(path, stage1, survival_limit_hours=72.0, cell_size=CELL)) == 1
    with pytest.raises(ValueError, match="directory"):
        cells.write_cell_list(tmp_path / "nowhere" / "c.json", [], stage1, survival_limit_hours=72.0, cell_size=CELL)
    bad = tmp_path / "bad.json"
    for text, message in [("not json", "cannot read"), (json.dumps({"format": "other"}), "not a stage-2 cell list")]:
        bad.write_text(text)
        with pytest.raises(ValueError, match=message):
            cells.read_cell_list(bad, stage1, survival_limit_hours=72.0, cell_size=CELL)
    with pytest.raises(ValueError, match="cannot read"):
        cells.read_cell_list(tmp_path / "missing.json", stage1, survival_limit_hours=72.0, cell_size=CELL)


# --- the script -------------------------------------------------------------------------------


def load_script():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven_stage2.py"
    spec = importlib.util.spec_from_file_location("run_safe_haven_stage2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_stage_2_script_defaults_agree_with_stage_1_and_import_starts_nothing() -> None:
    import importlib.util
    from pathlib import Path

    module = load_script()
    generate = importlib.util.spec_from_file_location(
        "run_generate_safe_havens", Path(module.__file__).with_name("run_generate_safe_havens.py"))
    stage1 = importlib.util.module_from_spec(generate)
    generate.loader.exec_module(stage1)
    assert module.STAGE1_PREFIX == stage1.NAME_PREFIX
    assert module.CELL_LIST_PATH.name == "safe_haven_degenerate_haven_cells.json"
    assert module.SURVIVAL_LIMIT_HOURS == 72.0 and module.CELL_SIZE == 256 and module.OVERWRITE is False


def test_the_stage_2_script_writes_the_cell_list_and_prints_it(products, tmp_path, monkeypatch, capsys) -> None:
    module = load_script()
    for name, value in dict(
        STAGE1_DIRECTORY=products.directory, STAGE1_PREFIX="byte", CELL_LIST_PATH=tmp_path / "cells.json",
        CELL_SIZE=CELL,
    ).items():
        monkeypatch.setattr(module, name, value)
    # the fixture names its files byte_hibernation.tif / byte_joint.tif; point the script at them
    monkeypatch.setattr(module, "Stage1Rasters", lambda *a, **k: products.byte)
    assert module.main() == 0
    printed = capsys.readouterr().out
    found = haven_cells(products.byte, survival_limit_hours=72.0, cell_size=CELL)
    assert f"{len(found)} of 12 cells have a haven pixel at or under 72 h" in printed
    assert "Stage 2: 3/3 cell rows" in printed and f"Cell list written to: {tmp_path / 'cells.json'}" in printed
    assert cells.read_cell_list(tmp_path / "cells.json", products.byte, survival_limit_hours=72.0,
                                cell_size=CELL) == found
    log = (tmp_path / "cells.json.log").read_text()
    assert "lunarscout.scripts.stage2: Stage 2: 3/3 cell rows" in log and "Cell list written to" in log
    with pytest.raises(ValueError, match="exists"):
        module.main()
    assert "Stage 2 stopped: ValueError" in (tmp_path / "cells.json.log").read_text()
    monkeypatch.setattr(module, "OVERWRITE", True)
    assert module.main() == 0
