"""The cluster-box maps: which havens are connected, the clusters and their boxes, the raster, the table and the files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from lunarscout._numba_horizon import safe_haven_cluster_maps as cm
from lunarscout._numba_horizon.safe_haven_chains import CHAIN_COLUMNS
from lunarscout.errors import InputError

from test_safe_haven_overview import GT, write_dem, write_psr


def seed(number, row, column, root, predecessor, root_start="2026-03-01T00:00:00.000000Z", mission=10.0):
    return dict(seed_id=number, row=row, column=column, root_id=root, predecessor_id=predecessor,
                root_start_utc=root_start, mission_hours=mission)


# Year 2026: seeds 1-3 are one tree (root 1); seed 4 is alone; seeds 5-6 are a tree (root 5), and an edge 3 -> 6 that stage 4
# did not keep joins the two trees. Year 2027: seeds 7-8 are a tree and an edge 8 -> 9 reaches seed 9, a root of its own
# in 2027 that stage 4 left alone.
ROWS = [
    seed(1, 10, 10, 1, 0), seed(2, 20, 12, 1, 1), seed(3, 14, 30, 1, 2),
    seed(4, 200, 200, 4, 0),
    seed(5, 100, 60, 5, 0), seed(6, 110, 80, 5, 5),
    seed(7, 50, 150, 7, 0, "2027-06-01T00:00:00.000000Z", 99.0), seed(8, 60, 150, 7, 7, "2027-06-01T00:00:00.000000Z", 120.0),
    seed(9, 250, 5, 9, 0, "2027-08-01T00:00:00.000000Z"),
]
EDGES = [(1, 2, 5.0), (2, 3, 5.0), (5, 6, 5.0), (3, 6, 9.0), (7, 8, 5.0), (8, 9, 5.0)]    


def test_in_the_chains_a_cluster_is_one_roots_tree_and_lone_seeds_are_dropped() -> None:
    clusters = cm.find_clusters(ROWS)
    assert clusters.year.tolist() == [2026, 2026, 2027] and clusters.havens.tolist() == [3, 2, 2]
    assert clusters.first_seed.tolist() == [1, 5, 7] and clusters.edges_not_used == 0
    assert (clusters.row_min.tolist(), clusters.row_max.tolist()) == ([10, 100, 50], [20, 110, 60])
    assert (clusters.column_min.tolist(), clusters.column_max.tolist()) == ([10, 60, 150], [30, 80, 150])
    assert clusters.longest_mission.tolist() == [10.0, 10.0, 120.0]


def test_over_the_edges_an_edge_stage_4_did_not_keep_joins_trees() -> None:
    clusters = cm.find_clusters(ROWS, connect="edges", edges=EDGES)
    assert clusters.year.tolist() == [2026, 2027] and clusters.havens.tolist() == [5, 3]      # seeds 1, 2, 3, 5, 6 and 7, 8, 9
    assert clusters.first_seed.tolist() == [1, 7] and clusters.edges_not_used == 0
    assert (clusters.row_min[0], clusters.row_max[0], clusters.column_min[0], clusters.column_max[0]) == (10, 110, 10, 80)


def test_an_edge_between_roots_of_different_years_is_not_used() -> None:
    rows = [dict(r) for r in ROWS]
    rows[8]["root_start_utc"] = "2028-01-01T00:00:00.000000Z"             # seed 9 now belongs to 2028
    clusters = cm.find_clusters(rows, connect="edges", edges=EDGES)
    assert clusters.edges_not_used == 1 and clusters.year.tolist() == [2026, 2027] and clusters.havens.tolist() == [5, 2]


def test_bad_inputs_are_refused() -> None:
    for call in (lambda: cm.find_clusters(ROWS, connect="loops"), lambda: cm.find_clusters(ROWS, connect="edges"),
                 lambda: cm.find_clusters([]), lambda: cm.find_clusters(ROWS[1:]),
                 lambda: cm.find_clusters(ROWS, connect="edges", edges=[(1, 99, 1.0)])):
        with pytest.raises(InputError) as error:
            call()
        assert error.value.code in ("cluster_setting_invalid", "cluster_table_invalid")


def test_the_classes_follow_the_counts_and_the_colours_are_one_hue_brighter_for_more_havens() -> None:
    counts = np.array([2, 3, 4, 5, 6, 7, 9, 10, 19, 20, 49, 50, 500])
    assert cm.cluster_classes(counts).tolist() == [1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    colours = cm.cluster_colours()
    luma = colours[1:].astype(float) @ [0.2126, 0.7152, 0.0722]
    assert colours.shape == (9, 3) and (np.diff(luma) > 0).all()
    assert (colours[1:, 1] > colours[1:, 0]).all() and (colours[1:, 1] >= colours[1:, 2]).all()      # green-leaning aqua
    assert len(cm.COUNT_LABELS) == len(cm.COUNT_EDGES) + 1 == 8


def test_outlines_lie_just_outside_the_box_and_the_later_box_is_on_top() -> None:
    strip = np.zeros((40, 40), dtype=np.uint8)
    cm.draw_outlines(strip, 0, [(10, 12, 10, 14, 3)], 2)
    assert strip[8, 8:17].tolist() == [3] * 9 and strip[14, 8:17].tolist() == [3] * 9      # the rows above and below
    assert strip[10:13, 8:10].min() == 3 and strip[10:13, 15:17].min() == 3                # the sides
    assert strip[10:13, 10:15].max() == 0                                                  # the inside is untouched
    assert strip.sum() == 3 * (2 * 9 * 2 + 2 * 3 * 2)
    cm.draw_outlines(strip, 0, [(9, 13, 9, 15, 5)], 1)                                     # a later box, one pixel out
    assert strip[8, 12] == 5 and strip[9, 12] == 3


def test_outlines_are_clipped_to_a_strip_and_to_the_grid() -> None:
    strip = np.zeros((10, 10), dtype=np.uint8)
    cm.draw_outlines(strip, 100, [(90, 92, 0, 3, 2), (200, 205, 0, 3, 2)], 3)               # above and below the strip
    assert strip.sum() == 0
    cm.draw_outlines(strip, 100, [(98, 101, 8, 9, 4)], 3)                                    # crosses the top and the right edge
    assert strip[2, 5:10].tolist() == [4] * 5 and strip[1, 5] == 4 and strip[1, 9] == 0 and strip.max() == 4
    corner = np.zeros((10, 10), dtype=np.uint8)
    cm.draw_outlines(corner, 0, [(0, 0, 0, 0, 7)], 1)                                        # at the grid's corner
    assert corner[0, 0] == 0 and corner[0, 1] == 7 and corner[1, 0] == 7 and corner[1, 1] == 7 and corner[2, 0] == 0


@pytest.fixture()
def scene(tmp_path):
    pytest.importorskip("matplotlib")
    write_dem(tmp_path / "dem.tif")
    write_psr(tmp_path / "psr.tif")
    with open(tmp_path / "chains.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHAIN_COLUMNS)
        writer.writeheader()
        for row in ROWS:
            writer.writerow({**{c: 0 for c in CHAIN_COLUMNS}, **row, "month": "2026-03", "outage_kind": "outage_sun_lit",
                             "outage_start_utc": "x", "outage_end_utc": "x"})
    with open(tmp_path / "edges.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed_id", "target_id", "arrival_hours"])
        writer.writerows(EDGES)
    return tmp_path


def run(scene, **options):
    options.setdefault("cell_patches", 1)
    options.setdefault("overwrite", True)
    options.setdefault("outline_pixels", 2)
    return cm.run_cluster_maps(scene / "out", chains_path=scene / "chains.csv", dem_path=scene / "dem.tif",
                               edges_path=scene / "edges.csv", psr_ids_path=scene / "psr.tif", **options)


def test_a_run_writes_the_rasters_pngs_contact_sheet_table_and_report(scene) -> None:
    from PIL import Image

    report = run(scene)
    out = scene / "out"
    assert report.clusters == 3 and report.connected_havens == 7 and report.connect == "chains"
    assert sorted(report.years) == [2026, 2027] and report.years[2026]["clusters"] == 2 and report.years[2026]["largest_cluster"] == 3
    with rasterio.open(out / "safe_haven_stage_4_cluster_boxes_2026.tif") as dataset:
        assert dataset.shape == (300, 300) and dataset.count == 4 and dataset.transform.to_gdal() == GT
        assert dataset.colorinterp[3].name == "alpha"
        image = dataset.read()
    colours = cm.cluster_colours()
    alpha = image[3] > 0
    assert tuple(image[:3, 9, 10]) == tuple(colours[2])                       # the 3-haven box (class 2): just above its top row
    assert tuple(image[:3, 99, 70]) == tuple(colours[1])                      # a 2-haven box (class 1): just above its top row
    assert image[3, 15, 15] == 0                                              # inside the box: transparent, the havens show
    assert not alpha[200, 200] and not alpha[250, 5]                          # lone seeds are not drawn
    with rasterio.open(out / "safe_haven_stage_4_cluster_boxes_2027.tif") as dataset:
        assert (dataset.read(4) > 0).sum() > 0
    with Image.open(out / "safe_haven_stage_4_cluster_boxes_2026.png") as picture:
        assert picture.size[0] > 1500 and picture.mode in ("RGB", "RGBA")
        pixels = {tuple(c) for c in np.unique(np.asarray(picture.convert("RGB")).reshape(-1, 3), axis=0)}
    assert tuple(colours[2]) in pixels and tuple(colours[1]) in pixels
    with Image.open(out / "safe_haven_stage_4_cluster_boxes_contact_sheet.png") as sheet:
        assert sheet.size[0] > sheet.size[1] > 500
    with open(out / "safe_haven_stage_4_cluster_boxes.csv", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert [r["havens"] for r in table] == ["3", "2", "2"] and [r["year"] for r in table] == ["2026", "2026", "2027"]
    first = table[0]
    assert (int(first["row_min"]), int(first["column_max"])) == (10, 30)
    assert float(first["x_min_m"]) == GT[0] + 20.0 * 10 and float(first["x_max_m"]) == GT[0] + 20.0 * 31     # outer pixel edges
    assert float(first["y_max_m"]) == GT[3] - 20.0 * 10 and float(first["y_min_m"]) == GT[3] - 20.0 * 21
    assert json.loads((out / "safe_haven_stage_4_cluster_boxes.report.json").read_text())["clusters"] == 3
    assert not list(out.glob("*.partial")) and not list(out.glob("*.aux.xml"))


def test_over_the_edges_the_run_joins_trees_and_reports_the_edges_it_did_not_use(scene) -> None:
    report = run(scene, connect="edges")
    assert report.connect == "edges" and report.clusters == 2 and report.years[2026]["largest_cluster"] == 5
    assert report.edges_not_used == 0


def test_years_can_be_chosen_an_empty_year_is_an_empty_map_and_settings_are_checked(scene) -> None:
    report = run(scene, years=[2027, 2030])
    assert report.years[2030]["clusters"] == 0
    with rasterio.open(scene / "out" / "safe_haven_stage_4_cluster_boxes_2030.tif") as dataset:
        assert dataset.shape == (300, 300) and (dataset.read(4) == 0).all()
    for options, code in (({"connect": "loops"}, "cluster_setting_invalid"), ({"outline_pixels": 0}, "cluster_setting_invalid"),
                          ({"years": [2026, 2026]}, "cluster_setting_invalid"),
                          ({"overview_pixels_per_cell": 7}, "overview_setting_invalid"),
                          ({"overwrite": False}, "overview_output_exists")):
        with pytest.raises(InputError) as error:
            run(scene, **options)
        assert error.value.code == code
    with pytest.raises(InputError) as error:
        cm.run_cluster_maps(scene / "o", chains_path=scene / "chains.csv", dem_path=scene / "dem.tif", connect="edges",
                            cell_patches=1)
    assert error.value.code == "cluster_setting_invalid"


def test_without_a_psr_raster_the_maps_are_still_made(scene) -> None:
    cm.run_cluster_maps(scene / "out2", chains_path=scene / "chains.csv", dem_path=scene / "dem.tif", years=[2026],
                        cell_patches=1, overwrite=True)
    assert (scene / "out2" / "safe_haven_stage_4_cluster_boxes_2026.png").exists()
