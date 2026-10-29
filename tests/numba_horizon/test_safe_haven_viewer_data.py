"""The data behind the pole viewer: the pregenerated lists, the pyramids and the tiles of every layer."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from lunarscout._numba_horizon import safe_haven_viewer_data as vd
from lunarscout._numba_horizon.safe_haven_chains import CHAIN_COLUMNS
from lunarscout.errors import InputError, ProductStorageError

from test_safe_haven_overview import CRS, GT, write_dem

SIZE = 1100            # three 512-pixel tiles across at the full size, two at level 1, one at level 2


def seed(number, month, row, column, root, predecessor, root_start):
    return dict(seed_id=number, month=month, row=row, column=column, root_id=root, predecessor_id=predecessor,
                root_start_utc=root_start, mission_hours=10.0)


# Seeds 1-3 are one tree whose chain starts in 2026-03 (seeds 2 and 3 are of 2026-04); seed 4 is alone; seeds 5-6 are a tree
# that starts in 2026-05. The months run 2026-03 to 2026-05.
SEEDS = [
    seed(1, "2026-03", 10, 10, 1, 0, "2026-03-05T00:00:00.000000Z"), seed(2, "2026-04", 20, 12, 1, 1, "2026-03-05T00:00:00.000000Z"),
    seed(3, "2026-04", 14, 600, 1, 2, "2026-03-05T00:00:00.000000Z"), seed(4, "2026-04", 900, 900, 4, 0, "2026-04-02T00:00:00.000000Z"),
    seed(5, "2026-05", 700, 700, 5, 0, "2026-05-09T00:00:00.000000Z"), seed(6, "2026-05", 702, 703, 5, 5, "2026-05-09T00:00:00.000000Z"),
]


def write_chains(path: Path, seeds=SEEDS) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHAIN_COLUMNS)
        writer.writeheader()
        for row in seeds:
            writer.writerow({**{c: 0 for c in CHAIN_COLUMNS}, "outage_kind": "outage_sun_lit",
                             "outage_start_utc": "2026-03-01T00:00:00.000000Z", "outage_end_utc": "2026-03-01T01:00:00.000000Z", **row})
    return path


def write_rgba(path: Path, pixels) -> None:
    """A SIZE x SIZE RGBA image, transparent except the ``(row, column, colour)`` pixels."""
    image = np.zeros((4, SIZE, SIZE), dtype=np.uint8)
    for row, column, colour in pixels:
        image[:, row, column] = colour
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=4, dtype="uint8", crs=CRS,
                       transform=Affine.from_gdal(*GT), tiled=True, blockxsize=128, blockysize=128) as dataset:
        dataset.write(image)


@pytest.fixture
def products(tmp_path):
    dem = tmp_path / "dem.tif"
    write_dem(dem, SIZE, SIZE)
    chains = write_chains(tmp_path / "safe_haven_stage_4_chains.csv")
    return tmp_path, dem, chains


def test_a_pyramid_level_halves_the_grid_until_it_fits_a_tile() -> None:
    assert vd.level_sizes(30336) == [30336, 15168, 7584, 3792, 1896, 948, 474]
    assert vd.level_sizes(512) == [512] and vd.level_sizes(1100) == [1100, 550, 275]


def test_a_reduction_keeps_thin_lines_with_max_and_averages_with_mean_and_pads_an_odd_edge() -> None:
    block = np.zeros((5, 3), dtype=np.uint8)
    block[1, 1] = 200
    assert vd.reduce2(block, "max").tolist() == [[200, 0], [0, 0], [0, 0]]
    assert vd.reduce2(block, "mean").tolist() == [[50, 0], [0, 0], [0, 0]]
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[3, 3] = (1, 2, 3, 255)
    assert vd.reduce2(rgba, "max")[1, 1].tolist() == [1, 2, 3, 255]


def test_months_run_from_the_first_to_the_last_inclusive() -> None:
    assert vd.month_range("2026-11", "2027-02") == ["2026-11", "2026-12", "2027-01", "2027-02"]
    assert vd.month_range("2026-03", "2026-03") == ["2026-03"]


def test_the_lists_hold_the_boxes_of_each_month_and_every_havens_pixel(products) -> None:
    tmp_path, dem, chains = products
    write_rgba(tmp_path / "safe_haven_stage_4_longest_chain_per_root_2026.tif", [])
    report = vd.build_viewer_lists(tmp_path / "viewer", chains_path=chains, dem_path=dem)
    assert (report.months, report.seeds, report.clusters, report.connected_havens) == (3, 6, 2, 5)
    assert report.chain_years == ["2026"]
    meta = json.loads((tmp_path / "viewer" / "viewer.json").read_text())
    assert meta["months"] == ["2026-03", "2026-04", "2026-05"] and meta["connect"] == "chains"
    assert meta["boxes"] == {"2026-03": [[10, 20, 10, 600, 3]], "2026-04": [], "2026-05": [[700, 702, 700, 703, 2]]}
    assert meta["chain_images"] == {"2026": "safe_haven_stage_4_longest_chain_per_root_2026.tif"}
    with np.load(tmp_path / "viewer" / "havens.npz") as arrays:
        assert arrays["offsets"].tolist() == [0, 1, 4, 6]
        assert list(zip(arrays["row"].tolist(), arrays["column"].tolist()))[:1] == [(10, 10)]       # seeds of 2026-03 first
        assert sorted(zip(arrays["row"][1:4].tolist(), arrays["column"][1:4].tolist())) == [(14, 600), (20, 12), (900, 900)]
    assert not list((tmp_path / "viewer").glob("*.partial"))


def test_the_lists_are_current_until_the_chain_table_dem_or_rule_changes(products) -> None:
    tmp_path, dem, chains = products
    viewer = tmp_path / "viewer"
    assert not vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="chains")
    vd.build_viewer_lists(viewer, chains_path=chains, dem_path=dem)
    assert vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="chains")
    assert not vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="edges")
    assert not vd.lists_current(viewer, chains_path=chains, dem_path=tmp_path / "other.tif", connect="chains")
    write_chains(chains, SEEDS[:5])
    assert not vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="chains")


def test_a_seed_outside_the_dem_and_a_non_square_dem_are_refused(products, tmp_path) -> None:
    folder, dem, _chains = products
    outside = write_chains(folder / "outside.csv", [seed(1, "2026-03", 5000, 10, 1, 0, "2026-03-05T00:00:00.000000Z"),
                                                    seed(2, "2026-03", 6, 10, 1, 1, "2026-03-05T00:00:00.000000Z")])
    with pytest.raises(InputError) as error:
        vd.build_viewer_lists(folder / "viewer", chains_path=outside, dem_path=dem)
    assert error.value.code == "viewer_table_invalid"
    wide = folder / "wide.tif"
    write_dem(wide, 100, 200)
    with pytest.raises(InputError) as error:
        vd.build_hillshade(folder / "cache", wide)
    assert error.value.code == "viewer_dem_invalid" and not list((folder / "cache").glob("*.partial"))


def test_the_hillshade_pyramid_is_built_once_and_its_tiles_are_dark_grey(products) -> None:
    tmp_path, dem, _chains = products
    cache = tmp_path / "cache"
    first = vd.build_hillshade(cache, dem)
    marker = (first / "done.json").stat().st_mtime_ns
    assert vd.build_hillshade(cache, dem) == first and (first / "done.json").stat().st_mtime_ns == marker        # found, not rebuilt
    levels = sorted(first.glob("level*.npy"))
    assert [np.load(p, mmap_mode="r").shape for p in levels] == [(1100, 1100), (550, 550), (275, 275)]
    low, high = round(255 * 0.07), round(255 * 0.24)
    full = np.load(first / "level0.npy", mmap_mode="r")
    assert full.min() >= low and full.max() <= high and full.max() > full.min()       # the overview maps' grey range


def test_tiles_are_512_pixels_and_past_the_edge_they_are_zero(products) -> None:
    tmp_path, dem, chains = products
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert data.size == SIZE and data.levels == 3 and data.tiles_across(0) == 3 and data.tiles_across(2) == 1
    tile = data.hillshade_tile(0, 2, 2)                                   # rows and columns 1024..1099 are the image
    assert tile.shape == (512, 512) and tile[:76, :76].min() > 0 and (tile[76:] == 0).all() and (tile[:, 76:] == 0).all()
    assert data.hillshade_tile(2, 0, 0)[:275, :275].min() > 0


def test_prepare_makes_only_what_is_missing(products) -> None:
    tmp_path, dem, chains = products
    options = dict(chains_path=chains, dem_path=dem)
    messages = []
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", progress=messages.append, **options)
    assert messages[0].startswith("Reading") and messages[-1] == "Ready"
    stamps = {p: p.stat().st_mtime_ns for p in (tmp_path / "viewer").iterdir()}
    messages.clear()
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", progress=messages.append, **options)
    assert messages == ["Ready"] and {p: p.stat().st_mtime_ns for p in (tmp_path / "viewer").iterdir()} == stamps
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", rebuild=True, progress=messages.append, **options)
    assert messages[1].startswith("Reading")                              # the lists again


def test_a_viewer_without_its_pyramids_says_so(products) -> None:
    tmp_path, dem, chains = products
    vd.build_viewer_lists(tmp_path / "viewer", chains_path=chains, dem_path=dem)
    with pytest.raises(ProductStorageError) as error:
        vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert error.value.code == "viewer_cache_missing"


def test_a_havens_tile_has_a_pixel_for_each_haven_of_the_month_and_coarser_levels_draw_2_pixel_dots(products) -> None:
    tmp_path, dem, chains = products
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert [data.havens_count(m) for m in range(3)] == [1, 3, 2] and [data.boxes_count(m) for m in range(3)] == [1, 0, 1]
    tile = data.havens_tile(1, 0, 0, 0)                                   # 2026-04: seeds (20, 12), (14, 600), (900, 900)
    assert np.argwhere(tile[..., 3] > 0).tolist() == [[20, 12]] and tile[20, 12].tolist() == list(vd.HAVEN_COLOUR)
    assert np.argwhere(data.havens_tile(1, 0, 1, 0)[..., 3] > 0).tolist() == [[14, 600 - 512]]
    coarse = data.havens_tile(1, 1, 0, 0)                                 # level 1: (10, 6) and (7, 300) are dots of 2 x 2
    assert (coarse[..., 3] > 0).sum() == 4 + 4 + 4 and (coarse[10:12, 6:8, 3] == 255).all()
    assert not data.havens_tile(0, 0, 2, 2)[..., 3].any()                 # nothing of 2026-03 down there


def test_a_boxes_tile_outlines_each_box_just_outside_it_in_its_class_colour(products) -> None:
    tmp_path, dem, chains = products
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    tile = data.boxes_tile(2, 0, 1, 1)                                    # 2026-05: rows 700-702, columns 700-703, 2 havens
    colour = [int(v) for v in vd.cluster_colours()[vd.cluster_classes(np.array([2]))[0]]] + [255]
    top, left = 700 - 512 - 1, 700 - 512 - 1                              # the ring is one pixel outside the box
    ring = tile[top:top + 5, left:left + 6]
    assert (ring[0] == colour).all() and (ring[-1] == colour).all() and (ring[:, 0] == colour).all() and (ring[:, -1] == colour).all()
    assert not ring[1:-1, 1:-1, 3].any()                                  # hollow inside
    wide = data.boxes_tile(0, 0, 0, 0)                                    # 2026-03: rows 10-20, columns 10-600: spans two tiles
    three = [int(v) for v in vd.cluster_colours()[vd.cluster_classes(np.array([3]))[0]]] + [255]      # this cluster has 3 havens
    assert three != colour and (wide[9, 9:200] == three).all()
    assert (wide[10:21, 9, 3] == 255).all() and not wide[12:19, 12:500, 3].any()       # hollow
    assert (data.boxes_tile(0, 0, 1, 0)[9, :90, 3] == 255).all()          # the top edge goes on in the next tile
    assert not data.boxes_tile(1, 0, 0, 0)[..., 3].any()                  # no clusters start in 2026-04
    coarse = data.boxes_tile(2, 2, 0, 0)                                  # level 2: 175-175 pixel, solid 3 x 3 at least
    assert (coarse[..., 3] > 0).sum() >= 9


def test_chain_tiles_come_from_the_tif_at_full_size_and_from_the_pyramid_above(products) -> None:
    tmp_path, dem, chains = products
    write_rgba(tmp_path / "safe_haven_stage_4_longest_chain_per_root_2026.tif", [(600, 700, (255, 122, 26, 255)), (3, 4, (1, 2, 3, 255))])
    # prepare_viewer builds the chain pyramids in worker processes; one year is built here directly
    vd.build_viewer_lists(tmp_path / "viewer", chains_path=chains, dem_path=dem)
    vd.build_hillshade(tmp_path / "cache", dem)
    tif = tmp_path / "safe_haven_stage_4_longest_chain_per_root_2026.tif"
    directory = Path(vd._build_chain_year((str(tmp_path / "cache"), str(tif))))
    assert (directory / "done.json").is_file() and not list((tmp_path / "cache").glob("*.partial"))
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    full = data.chain_tile(2026, 0, 1, 1)
    assert full[600 - 512, 700 - 512].tolist() == [255, 122, 26, 255] and (full[..., 3] > 0).sum() == 1
    level_one = data.chain_tile(2026, 1, 1, 1)                            # level 1 pixel (300, 350) is tile (1, 1)'s (300-512 < 0)
    assert not level_one[..., 3].any()
    level_one = data.chain_tile(2026, 1, 0, 0)
    assert level_one[300, 350].tolist() == [255, 122, 26, 255] and level_one[1, 2].tolist() == [1, 2, 3, 255]
    assert data.chain_tile(2026, 2, 0, 0)[150, 175].tolist() == [255, 122, 26, 255]             # the line survives 4 : 1
    assert data.chain_tile(2031, 0, 0, 0) is None and data.chain_tile(2031, 2, 0, 0) is None    # no image for that year
