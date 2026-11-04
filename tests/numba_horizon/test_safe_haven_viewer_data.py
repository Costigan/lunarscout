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

import lunarscout as ls
from lunarscout._numba_horizon import safe_haven_viewer_data as vd
from lunarscout._numba_horizon.safe_haven_chains import CHAIN_COLUMNS
from lunarscout.errors import InputError, ProductStorageError

from test_safe_haven_overview import CRS, GT, write_dem

SIZE = 1100            # three 512-pixel tiles across at the full size, two at level 1, one at level 2


def seed(number, month, row, column, root, predecessor, root_start, hours=0.0, length=1):
    return dict(seed_id=number, month=month, row=row, column=column, root_id=root, predecessor_id=predecessor,
                root_start_utc=root_start, mission_hours=10.0, hibernation_hours=hours, chain_length=length)


# Seeds 1-3 are one tree whose chain starts in 2026-03 (all three are of that month, so the chain is over after it); seed 4 is
# alone; seeds 5-6 are a tree that starts in 2026-05. The months run 2026-03 to 2026-05. The seed haven area of a month is the
# current start of each chain drawn in it, and the havens of the month's image near it: seed 1 in 2026-03 and seed 5 in
# 2026-05, nothing in 2026-04 (seed 4 is alone, so it starts no chain). Continuations and lone seeds are in
# test_safe_haven_areas.py.
SEEDS = [
    seed(1, "2026-03", 10, 10, 1, 0, "2026-03-05T00:00:00.000000Z", 0.0), seed(2, "2026-03", 20, 12, 1, 1, "2026-03-05T00:00:00.000000Z", 6.0, 2),
    seed(3, "2026-03", 14, 600, 1, 2, "2026-03-05T00:00:00.000000Z", 20.0, 3), seed(4, "2026-04", 900, 900, 4, 0, "2026-04-02T00:00:00.000000Z", 40.0),
    seed(5, "2026-05", 700, 700, 5, 0, "2026-05-09T00:00:00.000000Z", 0.0), seed(6, "2026-05", 702, 703, 5, 5, "2026-05-09T00:00:00.000000Z", 12.0, 2),
]


def write_chains(path: Path, seeds=SEEDS) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHAIN_COLUMNS)
        writer.writeheader()
        for row in seeds:
            writer.writerow({**{c: 0 for c in CHAIN_COLUMNS}, "outage_kind": "outage_sun_lit",
                             "outage_start_utc": "2026-03-01T00:00:00.000000Z", "outage_end_utc": "2026-03-01T01:00:00.000000Z", **row})
    return path


def write_rgba(path: Path, pixels, tags=None) -> None:
    """A SIZE x SIZE RGBA image, transparent except the ``(row, column, colour)`` pixels, with the tags ``tags``."""
    image = np.zeros((4, SIZE, SIZE), dtype=np.uint8)
    for row, column, colour in pixels:
        image[:, row, column] = colour
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=4, dtype="uint8", crs=CRS,
                       transform=Affine.from_gdal(*GT), tiled=True, blockxsize=128, blockysize=128) as dataset:
        dataset.write(image)
        dataset.update_tags(**(tags or {}))


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


def test_the_lists_hold_the_boxes_of_each_month_and_the_chain_starts_when_there_are_no_haven_images(products) -> None:
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
        assert arrays["offsets"].tolist() == [0, 1, 1, 2]
        assert list(zip(arrays["row"].tolist(), arrays["column"].tolist())) == [(10, 10), (700, 700)]   # seeds 1 and 5 start chains
    assert meta["chain_starts"] == {"2026-03": 1, "2026-04": 0, "2026-05": 1} and meta["area_radius_meters"] is None
    assert meta["seed_areas"] is None                                    # no areas file: the chain starts alone
    assert not list((tmp_path / "viewer").glob("*.partial"))


def test_the_lists_are_current_until_the_chain_table_dem_or_rule_changes(products) -> None:
    tmp_path, dem, chains = products
    viewer = tmp_path / "viewer"
    assert not vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="chains")
    vd.build_viewer_lists(viewer, chains_path=chains, dem_path=dem)
    assert vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="chains")
    assert not vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="edges")
    assert not vd.lists_current(viewer, chains_path=chains, dem_path=tmp_path / "other.tif", connect="chains")
    ls.generate_seed_haven_areas(chains, dem, tmp_path)                                         # the areas file is new: lists are made again
    assert not vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="chains")
    vd.build_viewer_lists(viewer, chains_path=chains, dem_path=dem)
    assert vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="chains")
    ls.generate_seed_haven_areas(chains, dem, tmp_path, radius_meters=300.0, overwrite=True)    # another radius: the file changed
    assert not vd.lists_current(viewer, chains_path=chains, dem_path=dem, connect="chains")
    vd.build_viewer_lists(viewer, chains_path=chains, dem_path=dem)
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


# The haven images of the seed areas: 10 pixels (200 m) around seed 1 of 2026-03, which starts a chain, and around seed 5 of 2026-05.
# (20, 12) is seed 2, 10.2 pixels from seed 1 and too far; (14, 600) is seed 3 and far; (900, 900) is seed 4, which starts nothing.
AREA_IMAGES = {
    "2026-03": {(10, 10): 5, (10, 18): 3, (10, 20): 24, (10, 21): 0, (10, 12): 25, (18, 16): 0, (19, 16): 0, (20, 12): 1, (14, 600): 10,
                (500, 500): 0},
    "2026-04": {(900, 900): 0, (20, 12): 0},
    "2026-05": {(700, 700): 0, (702, 703): 6, (700, 709): 12, (700, 711): 1},
}


def area_data(products, images=None, **options):
    tmp_path, dem, chains = products
    write_haven_months(chains, AREA_IMAGES if images is None else images)
    ls.generate_seed_haven_areas(chains, dem, tmp_path, **options)          # the step that makes the areas the viewer shows
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    return vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")


def pixels_of(data, month: int):
    low, high = int(data._offsets[month]), int(data._offsets[month + 1])
    return list(zip(data._rows[low:high].tolist(), data._columns[low:high].tolist(), data._hours[low:high].tolist()))


def test_a_months_seed_haven_area_is_its_chain_starts_and_the_havens_within_200_m_of_them(products) -> None:
    data = area_data(products)
    assert pixels_of(data, 0) == [(10, 10, 0.0), (10, 18, 6.0), (10, 20, 48.0), (18, 16, 0.0)]    # 10 pixels reach: not (10, 21), (19, 16)
    assert pixels_of(data, 1) == []                                      # seed 4 is a haven but starts no chain: nothing in 2026-04
    assert pixels_of(data, 2) == [(700, 700, 0.0), (700, 709, 24.0), (702, 703, 12.0)]
    assert [data.havens_count(m) for m in range(3)] == [4, 0, 3] and [data.chain_starts_count(m) for m in range(3)] == [1, 0, 1]
    assert data.hours_max == 48.0


def test_the_start_keeps_its_table_hours_and_a_pixel_over_the_limit_or_without_data_is_not_a_haven(products) -> None:
    data = area_data(products)
    assert (10, 10, 0.0) in pixels_of(data, 0)                           # the image says 5 counts (10 h), the chain table 0 h
    assert (10, 12) not in [p[:2] for p in pixels_of(data, 0)]           # 25 counts is over the threshold of 24
    assert (500, 500) not in [p[:2] for p in pixels_of(data, 0)]         # far away
    assert (20, 12) not in [p[:2] for p in pixels_of(data, 0)]           # seed 2 is not a start and 10.2 pixels away


def test_a_havens_tile_has_a_pixel_for_each_haven_of_the_area_and_coarser_levels_draw_2_pixel_dots(products) -> None:
    data = area_data(products)
    assert [data.havens_count(m) for m in range(3)] == [4, 0, 3] and [data.boxes_count(m) for m in range(3)] == [1, 0, 1]
    tile = data.havens_tile(0, 0, 0, 0)
    assert np.argwhere(tile[..., 3] > 0).tolist() == [[10, 10], [10, 18], [10, 20], [18, 16]]
    assert tile[10, 10].tolist() == list(WHITE) and tile[10, 18].tolist() == list(YELLOW)           # 0 h, 6 h
    assert tile[10, 20].tolist() == list(VIOLET)                                                      # 48 h
    assert not data.havens_tile(0, 0, 1, 0)[..., 3].any()                 # seed 3 (14, 600) is not near a start
    assert not data.havens_tile(1, 0, 0, 0)[..., 3].any()                 # 2026-04 has no chain start
    late = data.havens_tile(2, 0, 1, 1)                                   # 2026-05: (188, 188), (190, 191), (188, 197)
    assert np.argwhere(late[..., 3] > 0).tolist() == [[188, 188], [188, 197], [190, 191]]
    assert late[190, 191].tolist() == list(YELLOW) and late[188, 197].tolist() == list(CYAN)         # 12 h, 24 h
    coarse = data.havens_tile(0, 1, 0, 0)                                 # level 1: 2 x 2 dots at (5, 5), (5, 9), (5, 10) and (9, 8)
    assert (coarse[..., 3] > 0).sum() == 4 * 4 - 2 and (coarse[5:7, 5:7, 3] == 255).all()      # the dots at columns 9 and 10 share 2 pixels


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


def write_monthly_chains(directory: Path, entries, chain_table=None) -> Path:
    """A monthly chain pixel file with ``entries``: ``month -> (chains, [(row, column, code[, chain]), ...][, started])``; the
    table has 3 classes. ``chain_table`` makes the file keep its chains apart, as ``run_monthly_chain_maps`` does:
    ``(psr_class, hop_m, start_month, drawn)`` with a PSR class, a longest hop in metres and a start month (index) per chain
    and ``month -> [chain, ...]`` of the chains drawn in each (a pixel's chain is its fourth entry, 0 when there is none)."""
    months = sorted(entries)
    rows, columns, codes, owners, offsets, chains, started = [], [], [], [], [0], [], []
    for month in months:
        count, pixels, *rest = entries[month]
        pixels = sorted(pixels, key=lambda p: p[2])
        rows += [p[0] for p in pixels]
        columns += [p[1] for p in pixels]
        codes += [p[2] for p in pixels]
        owners += [p[3] if len(p) > 3 else 0 for p in pixels]
        offsets.append(len(rows))
        chains.append(count)
        started.append(rest[0] if rest else 0)
    table = np.array([[0, 0, 0, 0], [255, 0, 0, 255], [0, 255, 0, 255], [255, 255, 255, 255]], dtype=np.uint8)
    path = directory / "safe_haven_stage_4_monthly_chain_pixels.npz"
    extra = {}
    if chain_table is not None:
        psr_class, hop_m, start_month, drawn = chain_table
        extra = dict(chain=np.array(owners, dtype=np.int32), chain_psr_class=np.array(psr_class, dtype=np.uint8),
                     chain_hop_m=np.array(hop_m, dtype=np.float32), chain_start_month=np.array(start_month, dtype=np.int32),
                     chain_root=np.arange(1, len(psr_class) + 1), psr_classes_known=np.array(any(psr_class)),
                     drawn_offsets=np.cumsum([0] + [len(drawn.get(m, [])) for m in months]).astype(np.int64),
                     drawn_chain=np.array([c for m in months for c in drawn.get(m, [])], dtype=np.int32))
    np.savez_compressed(path, months=np.array(months), offsets=np.array(offsets, dtype=np.int64),
                        row=np.array(rows, dtype=np.int32), column=np.array(columns, dtype=np.int32),
                        code=np.array(codes, dtype=np.uint8), table=table, chains=np.array(chains, dtype=np.int64),
                        started=np.array(started, dtype=np.int64),
                        labels=np.array(["a", "b", "haven"]), **extra)
    return path


def test_the_monthly_chain_layer_appears_when_its_file_is_beside_the_chain_table(products) -> None:
    tmp_path, dem, chains = products
    options = dict(chains_path=chains, dem_path=dem)
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", **options)
    assert vd.ViewerData(tmp_path / "viewer", tmp_path / "cache").monthly_chains is None
    assert vd.ViewerData(tmp_path / "viewer", tmp_path / "cache").monthly_chain_tile(0, 0, 0, 0) is None
    write_monthly_chains(tmp_path, {"2026-03": (1, [(5, 6, 1)])})
    assert not vd.lists_current(tmp_path / "viewer", connect="chains", **options)           # the file is new: lists are made again
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", **options)
    assert vd.lists_current(tmp_path / "viewer", connect="chains", **options)
    assert vd.ViewerData(tmp_path / "viewer", tmp_path / "cache").monthly_chains is not None


def test_a_chains_start_is_a_3_by_3_marker_over_the_chain_and_the_haven(products) -> None:
    tmp_path, dem, chains = products
    path = write_monthly_chains(tmp_path, {"2026-03": (1, [(5, 6, 1), (5, 6, 3), (5, 6, 4), (5, 9, 3), (20, 12, 3)])})
    with np.load(path) as data:                                            # a file whose last class is the start marker
        arrays = {k: data[k] for k in data.files}
    arrays["table"] = np.concatenate([arrays["table"], [[255, 0, 255, 255]]]).astype(np.uint8)
    arrays["labels"] = np.array(["a", "b", "haven", "start of a chain"])
    np.savez_compressed(path, **arrays)
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    tile = data.monthly_chain_tile(0, 0, 0, 0)
    assert (tile[4:7, 5:8] == [255, 0, 255, 255]).all()                    # the start, 3 x 3 around (5, 6)
    assert tile[5, 9].tolist() == [255, 255, 255, 255] and tile[20, 12].tolist() == [255, 255, 255, 255]   # the file's own colours
    assert (tile[..., 3] > 0).sum() == 9 + 2                               # a haven stays one pixel
    assert (data.monthly_chain_tile(0, 2, 0, 0)[0:3, 0:3, 3] > 0).all()   # and is 3 level pixels wide at every level


def test_a_monthly_chain_tile_draws_the_months_pixels_in_their_colours_with_the_last_class_on_top(products) -> None:
    tmp_path, dem, chains = products
    write_monthly_chains(tmp_path, {"2026-03": (2, [(5, 6, 1), (5, 7, 2), (14, 600, 3), (14, 600, 1)]),
                                    "2026-05": (1, [(700, 701, 2)]), "2020-01": (9, [(1, 1, 1)])})   # 2020-01 is not a viewer month
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert [data.monthly_chain_count(m) for m in range(3)] == [2, 0, 1]
    assert [data.monthly_chain_started(m) for m in range(3)] == [0, 0, 0]       # this file has no "started" counts of its own
    tile = data.monthly_chain_tile(0, 0, 0, 0)
    assert tile.shape == (512, 512, 4) and (tile[..., 3] > 0).sum() == 2                    # (5, 6) and (5, 7); (14, 600) is next door
    assert tile[5, 6].tolist() == [255, 0, 0, 255] and tile[5, 7].tolist() == [0, 255, 0, 255]
    assert data.monthly_chain_tile(0, 0, 1, 0)[14, 600 - 512].tolist() == [255, 255, 255, 255]   # white over red at one pixel
    assert not data.monthly_chain_tile(1, 0, 0, 0)[..., 3].any()                            # 2026-04 has none
    assert data.monthly_chain_tile(2, 0, 1, 1)[700 - 512, 701 - 512].tolist() == [0, 255, 0, 255]
    coarse = data.monthly_chain_tile(0, 2, 0, 0)                                            # level 2: dots of 2 x 2
    assert (coarse[..., 3] > 0).sum() == 8 and coarse[1, 1].tolist() == [0, 255, 0, 255]    # (5, 6) and (5, 7) share a pixel: the higher class wins
    assert (data.monthly_chain_tile(0, 1, 0, 0)[..., 3] > 0).sum() == 2                     # level 1, single pixels: (2, 3) shared, and (10, 300)


def test_the_combined_tile_is_the_boxes_with_the_havens_over_them(products) -> None:
    tmp_path, dem, chains = products
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    both = data.boxes_havens_tile(0, 0, 0, 0)                             # 2026-03: the box rows 10-20, columns 10-600
    boxes, dots = data.boxes_tile(0, 0, 0, 0), data.havens_tile(0, 0, 0, 0)
    assert (both[dots[..., 3] > 0] == dots[dots[..., 3] > 0]).all()       # a haven is drawn over the box line
    assert both[10, 10].tolist() == list(WHITE) and boxes[10, 10].tolist() != list(WHITE)         # seed 1: no shadow
    rest = (dots[..., 3] == 0)
    assert (both[rest] == boxes[rest]).all()                              # elsewhere the box is unchanged
    assert (data.boxes_havens_tile(1, 0, 0, 0) == data.havens_tile(1, 0, 0, 0)).all()      # 2026-04 has no box: only the dots


def write_haven_months(chains: Path, months: dict, *, threshold: int = 24, limit_hours: float = 48.0, overviews: int = 2) -> Path:
    """The monthly haven images of ``scripts/threshold_safe_havens.py`` beside the chain table: per ``"YYYY-MM"`` a SIZE x SIZE
    uint8 image (255 NODATA) holding the ``(row, column): count`` pixels, with minimum overviews and the tags the script writes."""
    from rasterio.enums import Resampling

    directory = chains.parent / vd.HAVEN_MONTHS_DIRECTORY
    directory.mkdir(exist_ok=True)
    for month, pixels in months.items():
        image = np.full((SIZE, SIZE), 255, dtype=np.uint8)
        for (row, column), count in pixels.items():
            image[row, column] = count
        path = directory / f"{vd.HAVEN_MONTHS_DIRECTORY}_{month}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1, dtype="uint8", nodata=255, crs=CRS,
                           transform=Affine.from_gdal(*GT), tiled=True, blockxsize=128, blockysize=128) as dataset:
            dataset.write(image, 1)
            dataset.update_tags(SAFE_HAVEN_BYTE_THRESHOLD=str(threshold), SAFE_HAVEN_SURVIVAL_LIMIT_HOURS=f"{limit_hours:g}",
                                SAFE_HAVEN_PIXELS=str(len(pixels)))
            if overviews:
                dataset.build_overviews([2 ** k for k in range(1, overviews + 1)], Resampling.nearest)
        previous = image
        for level in range(overviews):
            padded = np.pad(previous, ((0, previous.shape[0] % 2), (0, previous.shape[1] % 2)), constant_values=255)
            previous = padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2).min(axis=(1, 3))
            with rasterio.open(path, "r+", overview_level=level) as overview:
                overview.write(previous, 1)
    return directory


WHITE, YELLOW, CYAN, VIOLET = (255, 255, 255, 255), (242, 201, 76, 255), (56, 182, 200, 255), (122, 92, 214, 255)


def test_the_haven_months_are_listed_in_the_viewer_file_and_make_the_lists_stale_when_they_change(products) -> None:
    tmp_path, dem, chains = products
    options = dict(chains_path=chains, dem_path=dem)
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", **options)
    assert vd.ViewerData(tmp_path / "viewer", tmp_path / "cache").haven_months == {}
    assert vd.ViewerData(tmp_path / "viewer", tmp_path / "cache").haven_months_tile(0, 0, 0, 0) is None
    write_haven_months(chains, {"2026-03": {(5, 6): 0}, "2026-05": {(700, 701): 24}, "2031-01": {(1, 1): 3}})
    assert not vd.lists_current(tmp_path / "viewer", connect="chains", **options)            # new files: the lists are made again
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", **options)
    assert vd.lists_current(tmp_path / "viewer", connect="chains", **options)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert sorted(data.haven_months) == [0, 2]                                                # 2031-01 is not a month of the table
    write_haven_months(chains, {"2026-04": {(9, 9): 1}})
    assert not vd.lists_current(tmp_path / "viewer", connect="chains", **options)


def test_a_haven_months_tile_colours_each_pixel_by_its_shadow_period_and_leaves_the_rest_clear(products) -> None:
    tmp_path, dem, chains = products
    # counts of two hours: 0 h, 2 h, 12 h | 14 h, 24 h | 26 h, 48 h | 50 h (over the limit) and NODATA
    write_haven_months(chains, {"2026-03": {(5, 6): 0, (5, 7): 1, (5, 8): 6, (5, 9): 7, (5, 10): 12, (5, 11): 13, (5, 12): 24,
                                            (5, 13): 25, (600, 700): 0}})
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    tile = data.haven_months_tile(0, 0, 0, 0)
    assert tile.shape == (512, 512, 4) and tile.dtype == np.uint8
    colours = [tile[5, column].tolist() for column in range(6, 13)]
    assert colours == [list(WHITE), list(YELLOW), list(YELLOW), list(CYAN), list(CYAN), list(VIOLET), list(VIOLET)]
    assert tile[5, 13, 3] == 0 and tile[0, 0, 3] == 0                  # above the limit and NODATA: transparent
    assert (tile[..., 3] > 0).sum() == 7                               # the pixel at (600, 700) is in another tile
    assert data.haven_months_tile(0, 0, 1, 1)[600 - 512, 700 - 512].tolist() == list(WHITE)
    assert data.haven_months_tile(0, 0, 2, 2)[..., 3].sum() == 0 and data.haven_months_tile(0, 0, 2, 2).shape == (512, 512, 4)
    assert data.haven_months_tile(1, 0, 0, 0) is None and data.haven_month_info(1) == {}      # no image for 2026-04
    assert data.haven_month_info(0) == {"pixels": 9, "limit_hours": 48.0, "threshold": 24, "hours_per_count": 2.0}
    assert data.haven_month_legend(1) == [] and [label for _c, label in data.haven_month_legend(0)] == [
        "0 h", "up to 12 h", "12 to 24 h", "24 to 48 h"]


def test_the_classes_break_at_12_24_and_the_limit_and_the_last_one_runs_to_the_limit() -> None:
    for hours, index in ((0, 0), (-1, 0), (0.1, 1), (12, 1), (12.1, 2), (24, 2), (24.1, 3), (48, 3), (500, 3)):
        assert vd.haven_shadow_class(hours) == index
    colours = [c for _b, c in vd.HAVEN_SHADOW_CLASSES]
    assert len(set(colours)) == 4 and colours[0] == (255, 255, 255)
    brightness = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in colours]
    assert brightness == sorted(brightness, reverse=True)               # decreasing brightness
    labels = lambda limit: [label for _c, label in vd.haven_shadow_legend(limit)]      # noqa: E731
    assert labels(48) == ["0 h", "up to 12 h", "12 to 24 h", "24 to 48 h"] == labels(None)
    assert labels(72) == ["0 h", "up to 12 h", "12 to 24 h", "24 to 72 h"]
    assert labels(20) == ["0 h", "up to 12 h", "12 to 20 h"] and labels(10) == ["0 h", "up to 10 h"] and labels(0) == ["0 h"]
    assert [c for c, _l in vd.haven_shadow_legend(48)] == colours


def test_the_colours_follow_the_hours_per_count_recorded_in_the_file(products) -> None:
    tmp_path, dem, chains = products
    directory = write_haven_months(chains, {"2026-03": {(5, 6): 13}}, threshold=48, limit_hours=48.0)
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    assert vd.ViewerData(tmp_path / "viewer", tmp_path / "cache").haven_months_tile(0, 0, 0, 0)[5, 6].tolist() == list(VIOLET)   # 13 x 2 h
    with rasterio.open(directory / "safe_haven_stage_1_haven_months_2026-03.tif", "r+") as month:
        month.update_tags(SAFE_HAVEN_HOURS_PER_COUNT="1")              # counts of one hour: 13 h
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert data.haven_month_info(0)["hours_per_count"] == 1.0
    assert data.haven_months_tile(0, 0, 0, 0)[5, 6].tolist() == list(CYAN)


def test_a_haven_survives_at_every_coarser_level_because_the_overviews_keep_the_minimum(products) -> None:
    tmp_path, dem, chains = products
    write_haven_months(chains, {"2026-03": {(101, 203): 6, (100, 202): 20}})        # one block at levels 1 and 2: the better wins
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    lut = data._haven_lut(24, 2.0)
    for level in (1, 2):
        tile = data.haven_months_tile(0, level, 0, 0)
        assert (tile[..., 3] > 0).sum() == 1
        assert tile[101 >> level, 203 >> level].tolist() == lut[6].tolist()
    assert data.haven_months_tile(0, 3, 0, 0) is None                              # the file has only two overviews


def test_a_months_image_with_no_havens_is_all_clear_and_a_zero_threshold_still_colours(products) -> None:
    tmp_path, dem, chains = products
    write_haven_months(chains, {"2026-03": {}, "2026-04": {(3, 3): 0}}, threshold=0)
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert not data.haven_months_tile(0, 0, 0, 0)[..., 3].any() and data.haven_month_info(0)["pixels"] == 0
    assert data.haven_months_tile(1, 0, 0, 0)[3, 3].tolist() == list(WHITE)


# --- the havens in every layer take the colour of their shadow period -----------------------------------------------------

def test_the_lists_keep_the_shadow_period_of_each_haven_of_each_month(products) -> None:
    data = area_data(products)
    with np.load(products[0] / "viewer" / "havens.npz") as arrays:
        assert arrays["hours"].tolist() == [0.0, 6.0, 48.0, 0.0, 0.0, 24.0, 12.0]       # the areas of 2026-03 and 2026-05, by pixel
    assert json.loads((products[0] / "viewer" / "viewer.json").read_text())["hours_max"] == 48.0 and data.hours_max == 48.0


def test_a_havens_tile_draws_each_haven_in_the_colour_of_its_shadow_period(products) -> None:
    data = area_data(products)
    assert data.havens_tile(0, 0, 0, 0)[10, 18].tolist() == list(YELLOW)               # 6 h
    assert data.havens_tile(0, 0, 0, 0)[10, 20].tolist() == list(VIOLET)               # 48 h
    assert data.havens_tile(0, 0, 0, 0)[10, 10].tolist() == list(WHITE)                # none
    assert data.havens_tile(2, 0, 1, 1)[702 - 512, 703 - 512].tolist() == list(YELLOW)  # 12 h is still the first class
    coarse = data.havens_tile(0, 1, 0, 0)                                              # level 1: dots of 2 x 2 in the same colour
    assert (coarse[5:7, 9:11] == list(YELLOW)).all()
    boxed = data.boxes_havens_tile(0, 0, 0, 0)
    assert boxed[10, 18].tolist() == list(YELLOW)                                      # over the boxes too


def test_where_havens_overlap_the_brightest_class_is_on_top(products) -> None:
    tmp_path, dem, _chains = products
    chains = write_chains(tmp_path / "safe_haven_stage_4_chains.csv", [
        seed(1, "2026-03", 10, 10, 1, 0, "2026-03-05T00:00:00.000000Z", 40.0), seed(2, "2026-03", 30, 30, 1, 1, "2026-03-05T00:00:00.000000Z", 0.0, 2)])
    write_haven_months(chains, {"2026-03": {(10, 10): 20, (11, 11): 0, (12, 12): 30}}, threshold=40)
    ls.generate_seed_haven_areas(chains, dem, tmp_path)
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    tile = data.havens_tile(0, 2, 0, 0)                                                # level 2: 2 x 2 dots at (2, 2), (2, 2) and (3, 3)
    assert tile[2, 2].tolist() == list(WHITE) and tile[3, 3].tolist() == list(WHITE)    # the 0 h haven covers the 40 h one and a corner of the third
    assert tile[4, 4].tolist() == list(VIOLET)                                          # 60 h has no class of its own: the last
    assert data.hours_max == 60.0 and [label for _c, label in data.shadow_legend()][-1] == "24 to 60 h"


def test_the_legend_of_the_seeds_runs_to_48_hours_unless_a_seed_hibernates_longer(products) -> None:
    tmp_path, dem, chains = products
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert data.shadow_legend() == vd.haven_shadow_legend(48.0) and data.chain_start_colour() is None


def test_a_yearly_image_is_shown_as_it_is_with_the_havens_colours_baked_into_it(products) -> None:
    tmp_path, dem, chains = products
    tif = tmp_path / "safe_haven_stage_4_longest_chain_per_root_2026.tif"
    pixels = [(10, 10, WHITE), (20, 12, YELLOW), (14, 600, CYAN), (900, 900, VIOLET), (5, 5, (255, 0, 0, 255))]
    write_rgba(tif, pixels)
    vd.build_viewer_lists(tmp_path / "viewer", chains_path=chains, dem_path=dem)
    vd.build_hillshade(tmp_path / "cache", dem)
    vd._build_chain_year((str(tmp_path / "cache"), str(tif)))
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    full = data.chain_tile(2026, 0, 0, 0)
    assert [full[r, c].tolist() for r, c, _k in pixels[:2] + pixels[4:]] == [list(WHITE), list(YELLOW), [255, 0, 0, 255]]
    assert data.chain_tile(2026, 0, 1, 0)[14, 600 - 512].tolist() == list(CYAN)
    assert data.chain_tile(2026, 0, 1, 1)[900 - 512, 900 - 512].tolist() == list(VIOLET)
    assert data.chain_tile(2026, 1, 0, 0)[10, 6].tolist() == list(YELLOW)                  # from the pyramid: a haven keeps its colour
    assert data.chain_tile(2026, 2, 0, 0)[5, 3].tolist() == list(YELLOW) and data.chain_tile(2026, 2, 0, 0)[1, 1].tolist() == [255, 0, 0, 255]


def test_the_chain_start_colour_is_the_monthly_files_start_class(products) -> None:
    tmp_path, dem, chains = products
    path = write_monthly_chains(tmp_path, {"2026-03": (1, [(5, 6, 1)])})
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["table"] = np.concatenate([arrays["table"], [[255, 0, 255, 255]]]).astype(np.uint8)
    arrays["labels"] = np.array(["a", "b", "haven", "start of a chain"])
    np.savez_compressed(path, **arrays)
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    assert vd.ViewerData(tmp_path / "viewer", tmp_path / "cache").chain_start_colour() == (255, 0, 255)


def write_psr(path: Path, pixels, size: int = SIZE) -> Path:
    """A permanent-shadow raster (255 PSR, 0 not) on the grid of the viewer tests' DEM."""
    image = np.zeros((size, size), dtype=np.uint8)
    for row, column in pixels:
        image[row, column] = 255
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype="uint8", crs=CRS,
                       transform=Affine.from_gdal(*GT), tiled=True, blockxsize=128, blockysize=128) as dataset:
        dataset.write(image, 1)
    return path


def test_the_filters_choose_the_chains_of_the_monthly_file_by_psr_class_and_longest_hop(products) -> None:
    tmp_path, dem, chains = products
    options = dict(chains_path=chains, dem_path=dem)
    # three chains: 0 (class 2, hop 60 m, starts in month 0), 1 (class 1, hop 300 m, month 0) and 2 (class 0, hop 900 m, month 1);
    # chains 0 and 1 are drawn in 2026-03, chain 2 in 2026-04, and pixel (9, 9) is owned by both 0 and 1
    write_monthly_chains(tmp_path, {
        "2026-03": (2, [(5, 6, 1, 0), (5, 7, 2, 1), (9, 9, 3, 0), (9, 9, 3, 1)], 2),
        "2026-04": (1, [(20, 20, 1, 2)], 1)},
        chain_table=([2, 1, 0], [60.0, 300.0, 900.0], [0, 0, 1], {"2026-03": [0, 1], "2026-04": [2]}))
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", **options)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert data.has_chain_filters() and data.psr_classes_known
    assert [data.monthly_chain_count(m) for m in range(3)] == [2, 1, 0]                       # no filter: the file's own counts
    assert [data.monthly_chain_started(m) for m in range(3)] == [2, 1, 0]

    def drawn(month, psr_class=0, hop_m=0.0):
        tile = data.monthly_chain_tile(month, 0, 0, 0, psr_class, hop_m)
        return {(int(r), int(c)) for r, c in np.argwhere(tile[..., 3] > 0)}
    assert drawn(0) == {(5, 6), (5, 7), (9, 9)} and drawn(1) == {(20, 20)}
    assert drawn(0, psr_class=2) == {(5, 6), (9, 9)}                                          # chain 0 only; (9, 9) is its too
    assert drawn(0, psr_class=1) == {(5, 6), (5, 7), (9, 9)} and drawn(1, psr_class=1) == set()   # chain 2 touches no PSR
    assert drawn(0, hop_m=100.0) == {(5, 7), (9, 9)} and drawn(1, hop_m=100.0) == {(20, 20)}  # chain 1 (and the shared pixel)
    assert drawn(0, psr_class=2, hop_m=100.0) == set() and drawn(0, psr_class=3) == set()     # both must pass; none is class 3
    assert drawn(0, hop_m=1000.0) == set()
    assert (data.monthly_chain_count(0, 2), data.monthly_chain_count(0, 0, 100.0), data.monthly_chain_count(0, 2, 100.0)) == (1, 1, 0)
    assert (data.monthly_chain_count(1, 0, 500.0), data.monthly_chain_started(1, 0, 500.0), data.monthly_chain_started(0, 1)) == (1, 1, 2)
    assert data.monthly_chain_started(1, 2) == 0 and data.monthly_chain_started(0, 2) == 1


def test_a_monthly_file_that_does_not_keep_its_chains_apart_still_draws_without_filters(products) -> None:
    tmp_path, dem, chains = products
    write_monthly_chains(tmp_path, {"2026-03": (2, [(5, 6, 1), (5, 7, 2)], 1)})                  # an older file
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem)
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert not data.has_chain_filters() and not data.psr_classes_known
    assert data.monthly_chain_count(0) == 2 and data.monthly_chain_count(0, 3, 1000.0) == 2   # filters are ignored
    assert (data.monthly_chain_tile(0, 0, 0, 0, 3, 1000.0)[..., 3] > 0).sum() == 2


def test_the_background_is_the_hillshade_with_the_permanent_shadow_composited_on_it_then_averaged(products) -> None:
    tmp_path, dem, chains = products
    psr = write_psr(tmp_path / "psr.tif", [(5, 6), (5, 7), (600, 700)])
    vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem, psr_path=psr)
    assert (vd.background_directory(tmp_path / "cache", dem, psr) / "done.json").is_file()
    data = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache", psr)
    assert data.has_background() and data.psr_colour() == vd.PSR_RGB == (142, 203, 255) and vd.PSR_ALPHA == 0.30
    grey = data.hillshade_tile(0, 0, 0).astype(float)
    tile = data.background_tile(0, 0, 0)
    assert tile.shape == (512, 512, 3) and tile.dtype == np.uint8
    expected = np.rint(grey[5, 6] * 0.7 + np.array(vd.PSR_RGB) * 0.3)                    # a PSR pixel: 30 % of the way to light blue
    assert tile[5, 6].tolist() == expected.tolist() and tile[5, 7].tolist() == expected.tolist()
    assert tile[5, 8].tolist() == [int(grey[5, 8])] * 3 and (tile[0, 0] == int(grey[0, 0])).all()     # elsewhere the hillshade
    assert data.background_tile(0, 1, 1)[600 - 512, 700 - 512].tolist() == np.rint(
        data.hillshade_tile(0, 1, 1)[600 - 512, 700 - 512] * 0.7 + np.array(vd.PSR_RGB) * 0.3).tolist()
    # level 1 averages 2 x 2 blocks of the composite: (2, 3) holds two PSR pixels of four, so half of the 30 % blend
    shade1 = data.hillshade_tile(1, 0, 0).astype(float)[2, 3]
    half = np.rint(shade1 * 0.85 + np.array(vd.PSR_RGB) * 0.15)
    assert np.abs(data.background_tile(1, 0, 0)[2, 3].astype(float) - half).max() <= 1.0
    quarter = data.background_tile(2, 0, 0)[1, 1].astype(float)                           # level 2: 2 of 16 pixels
    assert np.abs(quarter - np.rint(data.hillshade_tile(2, 0, 0).astype(float)[1, 1] * (1 - 0.3 * 2 / 16)
                                    + np.array(vd.PSR_RGB) * 0.3 * 2 / 16)).max() <= 1.0
    far = data.background_tile(2, 0, 0)[100, 100]                                          # no PSR near: the hillshade itself
    assert far.tolist() == [int(data.hillshade_tile(2, 0, 0)[100, 100])] * 3
    nothing = vd.ViewerData(tmp_path / "viewer", tmp_path / "cache")
    assert not nothing.has_background() and nothing.background_tile(0, 0, 0) is None and nothing.psr_colour() is None
    assert not vd.ViewerData(tmp_path / "viewer", tmp_path / "cache", tmp_path / "missing.tif").has_background()


def test_the_background_is_made_again_when_the_psr_raster_changes_and_kept_otherwise(products) -> None:
    tmp_path, dem, chains = products
    psr = write_psr(tmp_path / "psr.tif", [(5, 6)])
    first = vd.build_background(tmp_path / "cache", dem, psr)
    stamp = (first / "done.json").stat().st_mtime_ns
    assert vd.build_background(tmp_path / "cache", dem, psr) == first and (first / "done.json").stat().st_mtime_ns == stamp
    write_psr(psr, [(5, 6), (9, 9)])
    second = vd.build_background(tmp_path / "cache", dem, psr)
    assert second != first and (second / "done.json").is_file()


def test_a_permanent_shadow_raster_off_the_dem_grid_is_refused(products) -> None:
    tmp_path, dem, chains = products
    psr = write_psr(tmp_path / "psr.tif", [(5, 6)], size=600)
    with pytest.raises(InputError) as error:
        vd.prepare_viewer(tmp_path / "viewer", tmp_path / "cache", chains_path=chains, dem_path=dem, psr_path=psr)
    assert error.value.code == "viewer_psr_invalid" and not list((tmp_path / "cache").glob("background_*"))
