"""Rover chain maps: the two chain sets, the colour images (thin and dilated) and the GeoPackages."""

from __future__ import annotations

import colorsys
import importlib.util
import random
import shutil
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine

from lunarscout._numba_horizon import chain_map_styles as styles
from lunarscout._numba_horizon import safe_haven_chain_maps as maps
from lunarscout._numba_horizon import safe_haven_chains as chains
from lunarscout.errors import InputError
from test_safe_haven_chains import GT, write_tables

CRS = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
HEIGHT, WIDTH = 12, 30

# seeds 1..8 at pixels (i, 3i) for i = 0..7; edges 1>2, 2>3, 3>4, 2>5 (a branch), 6>7; seed 8 is alone
AFTER = [10, 20, 30, 40, 50, 60, 70, 80]
EDGES = [(1, 2), (2, 3), (3, 4), (2, 5), (6, 7)]

# the lengths of the real run's longest chains: 11,754 of 2 seeds and 130 of 3 or more
REAL_LENGTHS = [2] * 11754 + [3] * 39 + [4] * 24 + [5] * 21 + [6] * 22 + [7] * 14 + [8] * 8 + [9] + [10]
REAL = styles.build_intervals(REAL_LENGTHS)
RGB = lambda colour: tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))  # noqa: E731


def write_dem(path: Path, height=HEIGHT, width=WIDTH) -> None:
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32", crs=CRS,
                       transform=Affine.from_gdal(*GT)) as dataset:
        dataset.write(np.zeros((1, height, width), dtype="float32"))


def write_psr(path: Path, pixels, height=HEIGHT, width=WIDTH) -> None:
    """A PSR id raster on the DEM's grid with one region of one pixel per entry of ``pixels``."""
    ids = np.zeros((height, width), dtype=np.int32)
    for number, (row, column) in enumerate(pixels, start=1):
        ids[row, column] = number
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype="int32", crs=CRS,
                       transform=Affine.from_gdal(*GT)) as dataset:
        dataset.write(ids[None])


@pytest.fixture()
def tables(tmp_path):
    seeds, floods, edges = write_tables(tmp_path, AFTER, EDGES)
    chains.run_chains(tmp_path / "chains.csv", seeds_path=seeds, floods_path=floods, edges_path=edges, geo_transform=GT)
    write_dem(tmp_path / "dem.tif")
    return tmp_path


def run(tables, **options):
    options.setdefault("dilate_pixels", 2)
    return maps.run_chain_maps(tables / "out", chains_path=tables / "chains.csv", dem_path=tables / "dem.tif", **options)


def read_image(path):
    """``(rgba, info)``: the four bands and what the header says, read while the file is open."""
    with rasterio.open(path) as dataset:
        info = SimpleNamespace(nodata=dataset.nodata, transform=dataset.transform, crs=dataset.crs, dtypes=dataset.dtypes,
                               colorinterp=dataset.colorinterp, tags=dataset.tags(), profile=dataset.profile)
        return dataset.read(), info


def colour_at(image, row, column):
    return tuple(int(v) for v in image[:, row, column])


# --- the chain sets --------------------------------------------------------------------------------------

def test_the_predecessor_edges_take_the_longest_chain_through_them(tables) -> None:
    rows = chains.read_chain_table(tables / "chains.csv")
    _, edge_set, _ = maps.build_chain_sets(rows)
    by_head = {int(h): k for k, h in enumerate(edge_set.segment_attributes["seed_id"])}
    assert sorted(by_head) == [2, 3, 4, 5, 7]
    value = lambda seed, name: int(edge_set.segment_attributes[name][by_head[seed]])  # noqa: E731
    assert [value(s, "chain_length") for s in (2, 3, 4, 5, 7)] == [4, 4, 4, 3, 2]      # 2>5 is a branch: chains of 3 pass
    assert [value(s, "seed_chain_length") for s in (2, 3, 4, 5, 7)] == [2, 3, 4, 3, 2]
    assert [value(s, "predecessor_id") for s in (2, 3, 4, 5, 7)] == [1, 2, 3, 2, 6]
    # the mission of the longest at or below: seed 5 wakes at step 50, 40 steps (80 h) after root 1, plus 10 h
    assert [value(s, "mission_hours") for s in (2, 3, 4, 5, 7)] == [90, 70, 70, 90, 30]
    assert edge_set.segments.length.tolist() == [int(edge_set.segment_attributes["chain_length"][k])
                                                 for k in range(5)]


def test_the_longest_chain_of_each_root_and_the_isolated_seeds(tables) -> None:
    rows = chains.read_chain_table(tables / "chains.csv")
    _, _, chain_set = maps.build_chain_sets(rows)
    assert [list(line) for line in chain_set.lines] == [[1, 2, 3, 4], [6, 7]]
    assert chain_set.line_attributes["chain_length"].tolist() == [4, 2]
    assert chain_set.line_attributes["end_seed_id"].tolist() == [4, 7]
    assert chain_set.line_attributes["seed_ids"] == ["1,2,3,4", "6,7"]
    assert chain_set.dots.tolist() == [8]
    # every segment of a chain has the chain's own length, not the edge's
    assert chain_set.segments.length.tolist() == [4, 4, 4, 2]
    assert chain_set.line_attributes["chain_psr_small"] == [None, None]        # no routes were given: blank counts


def forest(count: int, generator: random.Random):
    after = [generator.randrange(0, 30) for _ in range(count)]
    edges = sorted((a, b) for a in range(1, count + 1) for b in range(1, count + 1)
                   if after[b - 1] > after[a - 1] and generator.random() < 0.1)
    root, predecessor, length = chains.chain_labels(
        np.array(after, dtype=np.int64), np.array([e[0] for e in edges], dtype=np.int64),
        np.array([e[1] for e in edges], dtype=np.int64))
    rows = [dict(seed_id=i + 1, row=i % 7, column=i % 11, predecessor_id=int(predecessor[i]), root_id=int(root[i]),
                 chain_length=int(length[i]), mission_hours=float(generator.randrange(1, 500)), month="2024-01",
                 root_start_utc="2024-03-01T00:00:00.000000Z", chain_psr_small=None, chain_psr_medium=None,
                 chain_psr_large=None) for i in range(count)]
    return rows


def test_the_sets_equal_an_exhaustive_search_on_random_forests() -> None:
    for seed in range(6):
        generator = random.Random(seed)
        rows = forest(generator.randrange(8, 45), generator)
        _, edge_set, chain_set = maps.build_chain_sets(rows)
        children = {}
        for r in rows:
            if r["predecessor_id"]:
                children.setdefault(r["predecessor_id"], []).append(r["seed_id"])

        def below(node):
            yield node
            for child in children.get(node, ()):
                yield from below(child)

        by_id = {r["seed_id"]: r for r in rows}
        for k, head in enumerate(edge_set.segment_attributes["seed_id"]):
            descendants = list(below(int(head)))
            assert edge_set.segment_attributes["chain_length"][k] == max(by_id[d]["chain_length"] for d in descendants)
            assert edge_set.segment_attributes["mission_hours"][k] == max(
                min(round(by_id[d]["mission_hours"]), 65535) for d in descendants)
        assert len(edge_set.segments.length) == sum(r["predecessor_id"] > 0 for r in rows)
        roots = {}
        for r in rows:
            roots.setdefault(r["root_id"], []).append(r)
        for line, end in zip(chain_set.lines, chain_set.line_attributes["end_seed_id"]):
            members = roots[by_id[int(end)]["root_id"]]
            best = max(members, key=lambda r: (r["chain_length"], round(r["mission_hours"]), -r["seed_id"]))
            assert int(end) == best["seed_id"] and len(line) == best["chain_length"] >= 2
            assert all(by_id[int(b)]["predecessor_id"] == int(a) for a, b in zip(line[:-1], line[1:]))
            assert by_id[int(line[0])]["predecessor_id"] == 0
        drawn_roots = {int(l[0]) for l in chain_set.lines}
        assert drawn_roots == {root for root, members in roots.items() if max(m["chain_length"] for m in members) >= 2}
        assert set(chain_set.dots.tolist()) == {root for root, members in roots.items()
                                                if max(m["chain_length"] for m in members) == 1}


def test_a_tie_between_two_ends_of_one_root_goes_to_the_longer_mission_then_the_lower_seed_id() -> None:
    def table(missions):
        # root 1 with children 2 and 3 (length 2 each) and a grandchild 4 of 3 only in the last case
        return [dict(seed_id=i + 1, row=i, column=2 * i, predecessor_id=p, root_id=1, chain_length=n, mission_hours=m,
                     month="2024-01", root_start_utc="2024-01-01T00:00:00.000000Z", chain_psr_small=None,
                     chain_psr_medium=None, chain_psr_large=None)
                for i, (p, n, m) in enumerate(missions)]

    for missions, expected_end in (([(0, 1, 5), (1, 2, 50), (1, 2, 50)], 2),       # equal: the lower id
                                   ([(0, 1, 5), (1, 2, 50), (1, 2, 60)], 3),       # the longer mission
                                   ([(0, 1, 5), (1, 2, 70), (1, 2, 60)], 2)):
        _, _, chain_set = maps.build_chain_sets(table(missions))
        assert chain_set.line_attributes["end_seed_id"].tolist() == [expected_end]
        assert [list(line) for line in chain_set.lines] == [[1, expected_end]]



# --- the intervals and their colours ------------------------------------------------------------------------

def test_the_intervals_are_two_seeds_then_the_quartiles_of_the_longer_chains() -> None:
    assert [(i.first, i.last, i.label) for i in REAL] == [
        (2, 2, "2 seeds"), (3, 3, "3 seeds"), (4, 5, "4-5 seeds"), (6, 6, "6 seeds"), (7, None, "7-10 seeds")]
    assert [i.colour for i in REAL] == list(styles.PALETTE) and [i.width_mm for i in REAL] == list(styles.LINE_WIDTHS_MM)
    counts = [int(((np.array(REAL_LENGTHS) >= i.first) & (np.array(REAL_LENGTHS) <= (i.last or 99))).sum()) for i in REAL]
    assert counts == [11754, 39, 45, 22, 24]                       # about a quarter of the 130 longer chains each
    only_pairs = styles.build_intervals([2, 2, 2])
    assert [(i.first, i.last) for i in only_pairs] == [(2, 2)] and len(only_pairs) == 1
    same = styles.build_intervals([2, 2, 3, 3, 3])                 # all quartiles equal: the cut points merge
    assert [(i.first, i.last) for i in same] == [(2, 2), (3, None)] and same[0].colour == styles.PALETTE[0]
    assert same[1].colour == styles.PALETTE[-1]                    # fewer intervals spread over the whole palette
    three = styles.build_intervals([2, 3, 3, 3, 4, 9])
    assert [i.first for i in three] == sorted({i.first for i in three}) and three[-1].last is None
    assert styles.build_intervals([]) [0].first == 2


def test_class_of_puts_every_length_in_its_interval() -> None:
    assert styles.class_of([1, 2, 3, 4, 5, 6, 7, 10, 500], REAL).tolist() == [0, 1, 2, 3, 3, 4, 5, 5, 5]


def relative_luminance(colour):
    channels = [c / 255 for c in RGB(colour)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def test_the_colours_are_distinct_saturated_not_dark_and_brighter_for_longer_chains() -> None:
    hsv = [colorsys.rgb_to_hsv(*(c / 255 for c in RGB(h))) for h in styles.PALETTE]
    luminance = [relative_luminance(h) for h in styles.PALETTE]
    assert luminance == sorted(luminance) and all(b - a > 0.05 for a, b in zip(luminance, luminance[1:]))
    assert luminance[0] > 0.15                                         # nothing is dark on a black or dark-grey basemap
    assert all(s >= 0.6 and v == 1.0 for _h, s, v in hsv)              # strongly saturated, full brightness
    hues = sorted(h * 360 for h, _s, _v in hsv)
    assert min(b - a for a, b in zip(hues, hues[1:])) >= 15            # distinct hues
    everything = [np.array(RGB(c)) for c in (*styles.PALETTE, styles.PSR_COLOUR, styles.HAVEN_COLOUR)]
    for i in range(len(everything)):
        for j in range(i + 1, len(everything)):
            assert np.linalg.norm(everything[i] - everything[j]) > 80, (i, j)       # also from aqua and white
    assert styles.PSR_COLOUR == "#00ffff" and styles.HAVEN_COLOUR == "#ffffff"


def test_the_lookup_table_has_transparent_nothing_then_the_intervals_aqua_and_white() -> None:
    table = styles.lookup_table(REAL)
    assert table.shape == (8, 4) and table[0].tolist() == [0, 0, 0, 0] and (table[1:, 3] == 255).all()
    assert [tuple(table[k, :3]) for k in range(1, 6)] == [RGB(c) for c in styles.PALETTE]
    assert tuple(table[6, :3]) == (0, 255, 255) and tuple(table[7, :3]) == (255, 255, 255)


# --- drawing ---------------------------------------------------------------------------------------------

def segments(*items):
    columns = list(zip(*items))
    return maps.Segments(*(np.array(c, dtype=np.int64) for c in columns))


def planes_of(*shape):
    return np.zeros((3, *(shape or (HEIGHT, WIDTH))), dtype=np.uint8)


@pytest.mark.parametrize("compiled", [True, False])
def test_a_segment_is_a_bresenham_line_in_its_interval_class(compiled) -> None:
    planes = planes_of()
    maps.draw_segments(planes, segments((0, 0, 3, 9, 4)), REAL, compiled=compiled)
    drawn = np.argwhere(planes[0] > 0)
    assert len(drawn) == 10 and tuple(drawn[0]) == (0, 0) and tuple(drawn[-1]) == (3, 9)
    assert set(np.unique(planes[0][planes[0] > 0])) == {3} and not planes[1].any() and not planes[2].any()
    assert np.all(np.abs(np.diff(drawn[:, 1])) <= 1) and np.all(np.abs(np.diff(drawn[:, 0])) <= 1)      # no gaps
    single = planes_of()
    maps.draw_segments(single, segments((5, 5, 5, 5, 2)), REAL, compiled=compiled)
    assert np.argwhere(single[0] > 0).tolist() == [[5, 5]] and single[0, 5, 5] == 1


def test_the_longer_chain_is_on_top_where_lines_cross_whatever_the_order() -> None:
    items = [(0, 0, 10, 10, 2), (0, 10, 10, 0, 9), (5, 0, 5, 10, 4)]
    reference = None
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
        planes = planes_of()
        maps.draw_segments(planes, segments(*[items[i] for i in order]), REAL)
        reference = planes if reference is None else reference
        assert np.array_equal(planes, reference)
    assert planes[0, 5, 5] == 5 and planes[0, 5, 2] == 3 and planes[0, 2, 2] == 1      # 9 seeds, 4 seeds, 2 seeds


def test_a_stamp_dilates_the_line_and_is_clipped_at_the_border() -> None:
    planes = planes_of()
    maps.draw_segments(planes, segments((6, 15, 6, 15, 3)), REAL, radius=2)
    assert (planes[0] > 0).sum() == 25
    corner = planes_of()
    maps.draw_segments(corner, segments((0, 0, 0, 0, 3), (11, 29, 11, 29, 3)), REAL, radius=2)
    assert (corner[0] > 0).sum() == 18
    long = planes_of()
    maps.draw_segments(long, segments((6, 3, 6, 20, 3)), REAL, radius=1)
    assert (long[0, 5:8, 3:21] > 0).all() and not long[0, :5].any() and not long[0, 8:].any()
    maps.draw_segments(long, segments((-3, -3, 2, 4, 3)), REAL)                         # ends outside: clipped, no error


def neighbours_of(mask):
    from scipy import ndimage

    return ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool))


@pytest.mark.parametrize("compiled", [True, False])
def test_route_pixels_next_to_a_psr_are_marked(compiled) -> None:
    psr = np.zeros((HEIGHT, WIDTH), bool)
    psr[3, 5] = True
    planes = planes_of()
    maps.draw_segments(planes, segments((0, 0, 3, 9, 4)), REAL, psr=psr, compiled=compiled)
    assert np.array_equal(planes[1] > 0, (planes[0] > 0) & neighbours_of(psr)) and planes[1].any()
    plain = planes_of()
    maps.draw_segments(plain, segments((0, 0, 3, 9, 4)), REAL, compiled=compiled)          # no mask: nothing is marked
    assert not plain[1].any() and np.array_equal(plain[0], planes[0])


def test_a_mark_stays_when_a_longer_chain_is_drawn_over_the_pixel() -> None:
    psr = np.zeros((HEIGHT, WIDTH), bool)
    psr[6, 5] = True
    planes = planes_of()
    maps.draw_segments(planes, segments((5, 0, 5, 10, 2), (0, 5, 11, 5, 9)), REAL, psr=psr)
    assert planes[0, 5, 5] == 5 and planes[1, 5, 5] == 1 and planes[1, 5, 6] == 1 and planes[1, 5, 3] == 0


def test_a_stamp_marks_the_whole_square_around_a_marked_pixel() -> None:
    psr = np.zeros((HEIGHT, WIDTH), bool)
    psr[8, 15] = True
    planes = planes_of()
    maps.draw_segments(planes, segments((6, 15, 6, 15, 3)), REAL, radius=1, psr=psr)       # two rows away: not next to it
    assert not planes[1].any()
    maps.draw_segments(planes, segments((7, 15, 7, 15, 3)), REAL, radius=1, psr=psr)
    assert (planes[1, 6:9, 14:17] == 1).all() and planes[1].sum() == 9


def test_the_images_show_every_haven_over_the_aqua_over_the_routes() -> None:
    planes = planes_of()
    planes[0, 2, 2] = 3
    planes[0, 2, 3] = 3
    planes[1, 2, 3] = 1
    planes[1, 2, 4] = 1
    planes[2, 2, 4] = 1
    image = maps.colour_image(planes, REAL)
    assert colour_at(image, 2, 2) == (*RGB(REAL[2].colour), 255)
    assert colour_at(image, 2, 3) == (0, 255, 255, 255)                  # aqua over the route
    assert colour_at(image, 2, 4) == (255, 255, 255, 255)                # white over aqua
    assert colour_at(image, 0, 0) == (0, 0, 0, 0)


def test_replacing_an_image_removes_the_sidecar_of_the_old_file(tmp_path) -> None:
    path = tmp_path / "image.tif"
    sidecar = tmp_path / "image.tif.aux.xml"
    sidecar.write_text("<PAMDataset/>")                     # statistics and band names of the file being replaced
    maps.write_image(path, planes_of(), REAL, transform=(0.0, 20.0, 0.0, 100.0, 0.0, -20.0), crs_wkt="",
                     tags={"SET": "x"})
    assert path.exists() and not sidecar.exists()


# --- the yearly images ---------------------------------------------------------------------------------

def start_in_years(tables, years_of_roots) -> Path:
    """Rewrite ``root_start_utc`` of the fixture's table so that each root's chain starts in the given year."""
    import csv

    path = tables / "chains.csv"
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        names, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["root_start_utc"] = f"{years_of_roots[int(row['root_id'])]}-03-04T00:00:00.000000Z"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_yearly(tables, **options):
    return maps.run_yearly_chain_maps(tables / "out", chains_path=tables / "chains.csv", dem_path=tables / "dem.tif",
                                      **options)


def test_each_year_has_the_chains_that_start_in_it_and_the_years_make_up_the_whole_image(tables) -> None:
    start_in_years(tables, {1: 2024, 6: 2025, 8: 2026})          # chains: roots 1 (4 seeds), 6 (2 seeds), 8 (alone)
    report = run_yearly(tables, overwrite=True)
    assert sorted(report.years) == [2024, 2025, 2026]
    assert [report.years[y]["chains"] for y in (2024, 2025, 2026)] == [1, 1, 0]            # root 8 is isolated
    assert [report.years[y]["havens"] for y in (2024, 2025, 2026)] == [5, 2, 1]
    for year in (2024, 2025, 2026):
        assert maps.yearly_path(tables / "out", year) == tables / "out" / f"safe_haven_stage_4_longest_chain_per_root_{year}.tif"
        assert (tables / "out" / f"safe_haven_stage_4_longest_chain_per_root_{year}.tif").exists()
    assert (tables / "out" / "safe_haven_stage_4_yearly_chain_maps.report.json").exists()
    run(tables, overwrite=True)
    whole, _ = read_image(tables / "out" / "safe_haven_stage_4_longest_chain_per_root.tif")
    images = [read_image(maps.yearly_path(tables / "out", y))[0] for y in (2024, 2025, 2026)]
    opaque = [image[3] > 0 for image in images]
    assert not (opaque[0] & opaque[1]).any() and not (opaque[0] & opaque[2]).any() and not (opaque[1] & opaque[2]).any()
    stacked = np.where(opaque[0] | opaque[1] | opaque[2], 1, 0)
    assert (stacked.astype(bool) == (whole[3] > 0)).all()                    # the same pixels are drawn
    for image, mask in zip(images, opaque):
        assert (image[:, mask] == whole[:, mask]).all()                       # in the same colours
    assert colour_at(images[0], 3, 9) == (255, 255, 255, 255)                 # a 2024 haven
    assert colour_at(images[1], 3, 9) == (0, 0, 0, 0) and colour_at(images[0], 5, 15) == (0, 0, 0, 0)
    assert colour_at(images[2], 7, 21) == (255, 255, 255, 255) and opaque[2].sum() == 1


def test_a_year_without_chains_is_a_transparent_image_and_years_can_be_chosen(tables) -> None:
    start_in_years(tables, {1: 2024, 6: 2025, 8: 2026})
    report = run_yearly(tables, years=[2025, 2030], overwrite=True)
    assert sorted(report.years) == [2025, 2030] and report.years[2030] == {
        "file": str(maps.yearly_path(tables / "out", 2030)), "chains": 0, "segments": 0, "havens": 0}
    image, info = read_image(maps.yearly_path(tables / "out", 2030))
    assert image.shape[0] == 4 and not image.any()
    assert info.colorinterp == (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    assert info.tags["YEAR"] == "2030" and info.tags["DILATE_PIXELS"] == "0" and "COLOURS" in info.tags
    assert not maps.yearly_path(tables / "out", 2024).exists()


def test_a_yearly_image_equals_the_whole_image_when_every_chain_starts_in_that_year(tables) -> None:
    start_in_years(tables, {1: 2024, 6: 2024, 8: 2024})
    run_yearly(tables, years=[2024], overwrite=True)
    run(tables, overwrite=True)
    year, _ = read_image(maps.yearly_path(tables / "out", 2024))
    whole, _ = read_image(tables / "out" / "safe_haven_stage_4_longest_chain_per_root.tif")
    assert (year == whole).all()


def test_yearly_images_refuse_to_replace_files_and_bad_years(tables) -> None:
    start_in_years(tables, {1: 2024, 6: 2025, 8: 2026})
    run_yearly(tables, years=[2024], overwrite=True)
    with pytest.raises(InputError) as error:
        run_yearly(tables, years=[2024])
    assert error.value.code == "chain_maps_output_exists"
    for bad in ([], [2024, 2024]):
        with pytest.raises(InputError) as error:
            run_yearly(tables, years=bad, overwrite=True)
        assert error.value.code == "chain_maps_setting_invalid"


def test_the_predecessor_edge_geopackage_holds_one_segment_per_edge(tables) -> None:
    run(tables)
    path = tables / "out" / "safe_haven_stage_4_predecessor_edges.gpkg"
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA application_id").fetchone()[0] == 0x47504B47
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 10300
    assert connection.execute("SELECT table_name, data_type, srs_id FROM gpkg_contents").fetchall() == [
        ("predecessor_edges", "features", 100000)]
    assert connection.execute("SELECT table_name, column_name, geometry_type_name FROM gpkg_geometry_columns").fetchall() == [
        ("predecessor_edges", "geom", "LINESTRING")]
    rows = connection.execute("SELECT geom, seed_id, predecessor_id, root_id, chain_length, seed_chain_length, "
                              "mission_hours, root_start_utc FROM predecessor_edges ORDER BY seed_id").fetchall()
    assert [r[1:5] for r in rows] == [(2, 1, 1, 4), (3, 2, 1, 4), (4, 3, 1, 4), (5, 2, 1, 3), (7, 6, 6, 2)]
    assert rows[3][6] == 90.0 and rows[3][7] == "1970-01-01T20:00:00.000000Z"
    blob = rows[3][0]                                           # seed 5: from seed 2 at (1, 3) to (4, 12), centres
    assert blob[:2] == b"GP" and blob[2] == 0 and blob[3] == 1 and struct.unpack("<i", blob[4:8])[0] == 100000
    kind, count = struct.unpack("<II", blob[9:17])
    coordinates = struct.unpack("<4d", blob[17:49])
    assert blob[8] == 1 and (kind, count) == (2, 2)
    assert coordinates == (1000.0 + 20.0 * 3.5, -500.0 - 20.0 * 1.5, 1000.0 + 20.0 * 12.5, -500.0 - 20.0 * 4.5)
    assert connection.execute("SELECT min_x, min_y, max_x, max_y FROM gpkg_contents").fetchone() == (
        1000.0 + 20.0 * 0.5, -500.0 - 20.0 * 6.5, 1000.0 + 20.0 * 18.5, -500.0 - 20.0 * 0.5)
    definition = connection.execute("SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id = 100000").fetchone()[0]
    assert "PROJCS" in definition or "PROJCRS" in definition


def test_the_longest_chain_geopackage_has_a_line_per_chain_and_the_dots_as_points(tables) -> None:
    run(tables)
    connection = sqlite3.connect(tables / "out" / "safe_haven_stage_4_longest_chain_per_root.gpkg")
    assert {r[0] for r in connection.execute("SELECT table_name FROM gpkg_contents")} == {"chains", "isolated_seeds"}
    lines = connection.execute("SELECT geom, root_id, end_seed_id, chain_length, mission_hours, seed_ids FROM chains "
                               "ORDER BY root_id").fetchall()
    assert [r[1:] for r in lines] == [(1, 4, 4, 70.0, "1,2,3,4"), (6, 7, 2, 30.0, "6,7")]
    assert struct.unpack("<II", lines[0][0][9:17]) == (2, 4)                    # one line with four vertices
    points = connection.execute("SELECT geom, seed_id, mission_hours FROM isolated_seeds").fetchall()
    assert len(points) == 1 and points[0][1:] == (8, 10.0)
    assert struct.unpack("<I", points[0][0][9:13])[0] == 1                       # WKB point
    assert struct.unpack("<2d", points[0][0][13:29]) == (1000.0 + 20.0 * 21.5, -500.0 - 20.0 * 7.5)


@pytest.mark.skipif(shutil.which("ogrinfo") is None, reason="GDAL's ogrinfo is not installed")
def test_gdal_reads_the_geopackages(tables) -> None:
    run(tables)
    out = tables / "out"
    first = subprocess.run(["ogrinfo", "-so", "-al", str(out / "safe_haven_stage_4_predecessor_edges.gpkg")],
                           capture_output=True, text=True, check=True).stdout
    assert "Feature Count: 5" in first and "Geometry: Line String" in first and "predecessor_edges" in first
    assert "chain_length: Integer" in first and "mission_hours: Real" in first and "Stereographic" in first
    second = subprocess.run(["ogrinfo", "-so", "-al", str(out / "safe_haven_stage_4_longest_chain_per_root.gpkg")],
                            capture_output=True, text=True, check=True).stdout
    assert "Layer name: chains" in second and "Feature Count: 2" in second
    assert "Layer name: isolated_seeds" in second and "Feature Count: 1" in second and "Geometry: Point" in second
    extent = subprocess.run(["ogrinfo", "-al", "-geom=WKT", str(out / "safe_haven_stage_4_longest_chain_per_root.gpkg"),
                             "isolated_seeds"], capture_output=True, text=True, check=True).stdout
    assert "POINT (1430" in extent and "-650" in extent


def qml_of(connection, table):
    from xml.etree import ElementTree

    rows = connection.execute("SELECT styleQML, useAsDefault, f_geometry_column FROM layer_styles WHERE f_table_name = ?",
                              (table,)).fetchall()
    assert len(rows) == 1 and rows[0][1] == 1 and rows[0][2] == "geom"
    return ElementTree.fromstring(rows[0][0].split("\n", 1)[1])            # the text after the DOCTYPE line



def qml_of(connection, table):
    from xml.etree import ElementTree

    rows = connection.execute("SELECT styleQML, useAsDefault, f_geometry_column FROM layer_styles WHERE f_table_name = ?",
                              (table,)).fetchall()
    assert len(rows) == 1 and rows[0][1] == 1 and rows[0][2] == "geom"
    return ElementTree.fromstring(rows[0][0].split("\n", 1)[1])            # the text after the DOCTYPE line


def rules_of(renderer):
    symbols = {s.get("name"): s for s in renderer.findall("symbols/symbol")}
    result = []
    for rule in renderer.findall("rules/rule"):
        options = {o.get("name"): o.get("value") for o in symbols[rule.get("symbol")].find("layer/Option")}
        colour = "#" + "".join(f"{int(v):02x}" for v in options["line_color"].split(",")[:3])
        result.append((rule.get("label"), rule.get("filter"), colour, float(options["line_width"])))
    return result


def test_the_line_style_has_a_rule_per_interval_shortest_first() -> None:
    from xml.etree import ElementTree

    renderer = ElementTree.fromstring(styles.line_style_qml(REAL).split("\n", 1)[1]).find("renderer-v2")
    assert renderer.get("type") == "RuleRenderer"
    assert rules_of(renderer) == [
        ("2 seeds", '"chain_length" = 2', "#6a5cff", 0.3), ("3 seeds", '"chain_length" = 3', "#ff4f6e", 0.45),
        ("4-5 seeds", '"chain_length" >= 4 AND "chain_length" <= 5', "#ff7a1a", 0.65),
        ("6 seeds", '"chain_length" = 6', "#ffcc00", 0.9), ("7-10 seeds", '"chain_length" >= 7', "#7dff3a", 1.2)]


def test_the_geopackage_layers_carry_the_styles_of_the_runs_own_intervals(tables) -> None:
    run(tables)                                         # chains of 2 and 4 seeds: two intervals, "2 seeds" and "3-4 seeds"
    expected = [("2 seeds", '"chain_length" = 2', "#6a5cff", 0.3), ("3-4 seeds", '"chain_length" >= 3', "#7dff3a", 1.2)]
    for name, table in (("predecessor_edges", "predecessor_edges"), ("longest_chain_per_root", "chains")):
        connection = sqlite3.connect(tables / "out" / f"safe_haven_stage_4_{name}.gpkg")
        assert rules_of(qml_of(connection, table).find("renderer-v2")) == expected
    connection = sqlite3.connect(tables / "out" / "safe_haven_stage_4_longest_chain_per_root.gpkg")
    assert [r[0] for r in connection.execute("SELECT table_name FROM gpkg_contents ORDER BY rowid")] == ["isolated_seeds", "chains"]
    renderer = qml_of(connection, "isolated_seeds").find("renderer-v2")
    options = {o.get("name"): o.get("value") for o in renderer.find("symbols/symbol/layer/Option")}
    assert renderer.get("type") == "singleSymbol" and options["color"] == "255,255,255,255"        # havens are white
    assert float(options["size"]) < 1.5 and options["outline_style"] == "no" and options["name"] == "circle"
    assert connection.execute("SELECT count(*) FROM layer_styles").fetchone()[0] == 2


# --- the images ------------------------------------------------------------------------------------------

def test_the_images_are_transparent_rgba_geotiffs_with_routes_aqua_and_white_havens(tables) -> None:
    write_psr(tables / "psr.tif", [(3, 6)])
    report = run(tables, psr_ids_path=tables / "psr.tif")
    out = tables / "out"
    lime, violet = (*RGB(styles.PALETTE[4]), 255), (*RGB(styles.PALETTE[0]), 255)
    assert [i["label"] for i in report.intervals] == ["2 seeds", "3-4 seeds"]
    assert [i["chains"] for i in report.intervals] == [1, 1] and report.raster_lines == "straight"
    expected = {p.name for g in maps.output_paths(out).values() for p in g.values()} | {"safe_haven_stage_4_chain_maps.report.json"}
    assert {p.name for p in out.iterdir()} == expected                    # no .qml files, no data layers
    edges, info = read_image(out / "safe_haven_stage_4_predecessor_edges.tif")
    assert edges.shape == (4, HEIGHT, WIDTH) and edges.dtype == np.uint8 and info.nodata is None
    assert info.colorinterp == (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    assert info.transform.to_gdal() == GT and info.crs is not None and info.profile["tiled"]
    assert info.tags["SEGMENTS"].startswith("straight lines") and "2 seeds #6a5cff" in info.tags["COLOURS"]
    for i in range(8):                                                    # every haven is white, isolated ones too
        assert colour_at(edges, i, 3 * i) == (255, 255, 255, 255)
    assert colour_at(edges, 0, 1) == lime and colour_at(edges, 1, 2) == lime        # the chain of 4 seeds: edge 1 > 2
    assert colour_at(edges, 1, 4) == lime                                 # edge 2 > 3 (through-length 4)
    assert colour_at(edges, 5, 16) == violet and colour_at(edges, 6, 17) == violet      # edge 6 > 7 (a chain of 2)
    assert colour_at(edges, 2, 5) == (0, 255, 255, 255) and colour_at(edges, 2, 7) == (0, 255, 255, 255)   # beside the PSR
    assert colour_at(edges, 3, 8) != (0, 255, 255, 255) and colour_at(edges, 3, 6) == (0, 0, 0, 0)          # not beside, the PSR itself
    assert colour_at(edges, 11, 0) == (0, 0, 0, 0) and colour_at(edges, 11, 29) == (0, 0, 0, 0)             # transparent background
    allowed = {violet, lime, (0, 255, 255, 255), (255, 255, 255, 255), (0, 0, 0, 0)}
    assert {colour_at(edges, r, c) for r in range(HEIGHT) for c in range(WIDTH)} <= allowed
    assert ((edges[3] == 0) == (edges[:3].sum(axis=0) == 0)).all()       # transparent exactly where nothing is painted
    longest, _ = read_image(out / "safe_haven_stage_4_longest_chain_per_root.tif")
    assert colour_at(longest, 7, 21) == (255, 255, 255, 255)              # the isolated haven is drawn here as well
    assert (edges[3] > 0).sum() > (longest[3] > 0).sum()                  # the branch 2 > 5 is not part of a longest chain
    assert colour_at(longest, 2, 5) == (0, 255, 255, 255)                 # the aqua is on this image too
    shown = int((edges[:3] == [[[0]], [[255]], [[255]]]).all(axis=0).sum())
    assert shown == 2 and report.psr_marked_pixels["predecessor_edges"] == 3      # (2, 6) is marked but a haven shows over it


def test_the_dilated_images_are_wider_and_their_havens_one_pixel_larger(tables) -> None:
    write_psr(tables / "psr.tif", [(3, 6)])
    run(tables, psr_ids_path=tables / "psr.tif", dilate_pixels=2)
    out = tables / "out"
    for name in maps.CHAIN_SETS:
        thin, _ = read_image(out / f"safe_haven_stage_4_{name}.tif")
        wide, info = read_image(out / f"safe_haven_stage_4_{name}_dilated.tif")
        assert info.tags["DILATE_PIXELS"] == "2" and wide.shape == thin.shape
        assert (wide[3] > 0).sum() > 3 * (thin[3] > 0).sum() and ((thin[3] > 0) <= (wide[3] > 0)).all()
        assert (wide[0] == 0).sum() > 0 and not ((wide[3] == 0) & (thin[3] > 0)).any()
    wide, _ = read_image(out / "safe_haven_stage_4_longest_chain_per_root_dilated.tif")
    assert (wide[:, 4:11, 18:25] == 255).all()                           # the haven at (7, 21): 7 x 7, white, over the routes
    assert colour_at(wide, 11, 25) == (0, 0, 0, 0) and colour_at(wide, 3, 24) == (0, 0, 0, 0)
    # (the havens' larger stamps cover the aqua in a raster this small; the stamps of marked pixels are tested above)


def test_refusals_and_a_failure_leaves_earlier_files_alone(tables) -> None:
    run(tables)
    with pytest.raises(InputError) as caught:
        run(tables)
    assert caught.value.code == "chain_maps_output_exists"
    before = (tables / "out" / "safe_haven_stage_4_predecessor_edges.tif").read_bytes()
    with pytest.raises(InputError) as caught:
        run(tables, overwrite=True, dilate_pixels=0)
    assert caught.value.code == "chain_maps_setting_invalid"
    write_dem(tables / "small.tif", height=5, width=5)
    with pytest.raises(InputError) as caught:
        maps.run_chain_maps(tables / "out", chains_path=tables / "chains.csv", dem_path=tables / "small.tif", overwrite=True)
    assert caught.value.code == "chain_maps_grid_mismatch"
    assert (tables / "out" / "safe_haven_stage_4_predecessor_edges.tif").read_bytes() == before
    assert not list((tables / "out").glob("*.partial"))
    (tables / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(InputError) as caught:
        maps.run_chain_maps(tables / "out", chains_path=tables / "bad.csv", dem_path=tables / "dem.tif", overwrite=True)
    assert caught.value.code == "chains_table_unreadable"


def test_overwrite_replaces_the_products(tables) -> None:
    run(tables)
    run(tables, overwrite=True, dilate_pixels=3)
    _, info = read_image(tables / "out" / "safe_haven_stage_4_predecessor_edges_dilated.tif")
    assert info.tags["DILATE_PIXELS"] == "3"


# --- the script ----------------------------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven_chain_maps.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_safe_haven_chain_maps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_defaults_follow_stage_4_and_importing_it_starts_nothing() -> None:
    module = load_script()
    four = importlib.util.spec_from_file_location("run_safe_haven_stage4", SCRIPT.parent / "run_safe_haven_stage4.py")
    stage4 = importlib.util.module_from_spec(four)
    four.loader.exec_module(stage4)
    assert module.CHAINS_PATH == stage4.OUTPUT_PATH and module.DEM_PATH == stage4.DEM_PATH
    assert module.DILATE_PIXELS == 2 and module.OVERWRITE is True and module.RASTER_ROUTES is True
    assert module.PSR_IDS_PATH == stage4.PSR_IDS_PATH
    three = importlib.util.spec_from_file_location("run_safe_haven_stage3", SCRIPT.parent / "run_safe_haven_stage3.py")
    stage3 = importlib.util.module_from_spec(three)
    three.loader.exec_module(stage3)
    assert module.LIGHT_STORE_PATH == stage3.LIGHT_STORE_PATH and module.PASSABLE_PATH == stage3.PASSABLE_PATH
    for name in ("max_days", "moves_per_step", "connectivity", "max_slope_deg"):          # what the flood followed
        assert getattr(module.FLOOD_SETTINGS, name) == getattr(stage3.SETTINGS, name)


def test_the_script_writes_the_six_files_and_prints_the_report(tables, monkeypatch, capsys) -> None:
    module = load_script()
    write_psr(tables / "psr.tif", [(3, 6)])
    for name, value in dict(DEM_PATH=tables / "dem.tif", CHAINS_PATH=tables / "chains.csv", PSR_IDS_PATH=tables / "psr.tif",
                            OUTPUT_DIRECTORY=tables / "out", RASTER_ROUTES=False).items():
        monkeypatch.setattr(module, name, value)
    assert module.main() == 0
    printed = capsys.readouterr().out
    assert '"predecessor_edges": 5' in printed and '"isolated_seeds": 1' in printed
    assert len(list((tables / "out").glob("*.tif"))) == 4 and len(list((tables / "out").glob("*.gpkg"))) == 2
    assert not list((tables / "out").glob("*.qml"))
    assert "six files written" in (tables / "out" / "safe_haven_stage_4_chain_maps.log").read_text()
    assert module.main() == 0                                          # the default is to replace the tables
    monkeypatch.setattr(module, "OVERWRITE", False)
    with pytest.raises(InputError, match="exists"):
        module.main()



def test_a_psr_raster_of_another_size_is_refused_and_nothing_is_written(tables) -> None:
    write_psr(tables / "small.tif", [(1, 1)], height=5, width=5)
    with pytest.raises(InputError) as caught:
        run(tables, psr_ids_path=tables / "small.tif")
    assert caught.value.code == "chain_maps_psr_mismatch" and not (tables / "out").exists()


# --- the monthly chain pixels ---------------------------------------------------------------------------------
def start_in_months(tables, months_of_roots, seed_months=None) -> Path:
    """Rewrite ``root_start_utc`` so that each root's chain starts in the given month (``"2024-03"``), and every seed's
    ``month`` to its chain's start month unless ``seed_months`` (seed id -> month) says otherwise."""
    import csv

    path = tables / "chains.csv"
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        names, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["root_start_utc"] = f"{months_of_roots[int(row['root_id'])]}-04T00:00:00.000000Z"
        row["month"] = (seed_months or {}).get(int(row["seed_id"]), months_of_roots[int(row["root_id"])])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_monthly(tables, **options):
    return maps.run_monthly_chain_maps(tables / "out", chains_path=tables / "chains.csv", dem_path=tables / "dem.tif",
                                       **options)


def month_image(path, month):
    """The month's pixels of the monthly file as an RGBA image ``(4, HEIGHT, WIDTH)``, and the codes in file order."""
    with np.load(path) as data:
        index = [str(m) for m in data["months"]].index(month)
        lo, hi = int(data["offsets"][index]), int(data["offsets"][index + 1])
        image = np.zeros((4, HEIGHT, WIDTH), dtype=np.uint8)
        rows, columns, codes = data["row"][lo:hi], data["column"][lo:hi], data["code"][lo:hi]
        image[:, rows, columns] = data["table"][codes].T
        return image, codes


def test_each_month_has_the_chains_that_start_in_it_and_the_months_make_up_the_yearly_image(tables) -> None:
    start_in_months(tables, {1: "2024-03", 6: "2024-05", 8: "2025-01"})
    write_psr(tables / "psr.tif", [(2, 7), (6, 20)])
    options = dict(overwrite=True, psr_ids_path=tables / "psr.tif")
    report = run_monthly(tables, **options)
    assert list(report.months) == [f"2024-{m:02d}" for m in range(3, 13)] + ["2025-01"]
    assert report.months["2024-03"]["chains"] == 1 and report.months["2024-05"]["chains"] == 1
    assert sum(v["chains"] for v in report.months.values()) == 2                       # root 8 is alone
    assert [report.months[m]["havens"] for m in ("2024-03", "2024-04", "2024-05", "2025-01")] == [5, 0, 2, 1]
    assert report.months["2024-04"]["pixels"] == 0 and report.pixels == sum(v["pixels"] for v in report.months.values())
    assert (tables / "out" / "safe_haven_stage_4_monthly_chain_maps.report.json").exists()
    path = maps.monthly_path(tables / "out")
    assert path == tables / "out" / "safe_haven_stage_4_monthly_chain_pixels.npz"
    run_yearly(tables, years=[2024, 2025], psr_ids_path=tables / "psr.tif", overwrite=True)
    for year, year_months in ((2024, [f"2024-{m:02d}" for m in range(3, 13)]), (2025, ["2025-01"])):
        yearly, _ = read_image(maps.yearly_path(tables / "out", year))
        opaque = np.zeros((HEIGHT, WIDTH), dtype=bool)
        for month in year_months:
            image, _codes = month_image(path, month)
            mask = image[3] > 0
            assert not (mask & opaque).any()                                       # no pixel in two months here
            opaque |= mask
            assert (image[:, mask] == yearly[:, mask]).all()                       # the same colour
        assert (opaque == (yearly[3] > 0)).all()                                   # the same pixels
    aqua = styles.lookup_table(maps.build_intervals(np.array([2, 2]))).shape[0] - 2
    with np.load(path) as data:
        assert (data["code"] == aqua).any()                                        # a route passed next to a PSR
        assert data["table"].shape[1] == 4 and list(data["labels"])[-2:] == ["next to a PSR", "haven"]
        assert data["chains"].tolist() == [report.months[str(m)]["chains"] for m in data["months"]]


def test_a_months_pixels_are_ordered_by_class_so_the_havens_are_drawn_last(tables) -> None:
    start_in_months(tables, {1: "2024-03", 6: "2024-03", 8: "2024-03"})
    run_monthly(tables, overwrite=True)
    _image, codes = month_image(maps.monthly_path(tables / "out"), "2024-03")
    assert (np.diff(codes.astype(int)) >= 0).all() and codes[-1] == codes.max() and codes[0] >= 1
    with np.load(maps.monthly_path(tables / "out")) as data:
        assert int(codes.max()) == data["table"].shape[0] - 1                       # the white haven class is the last


def test_monthly_pixels_equal_the_yearly_image_when_every_chain_starts_in_one_month(tables) -> None:
    start_in_months(tables, {1: "2024-03", 6: "2024-03", 8: "2024-03"})
    run_monthly(tables, overwrite=True)
    run_yearly(tables, years=[2024], overwrite=True)
    yearly, _ = read_image(maps.yearly_path(tables / "out", 2024))
    image, _codes = month_image(maps.monthly_path(tables / "out"), "2024-03")
    assert (image == yearly).all()


def test_monthly_pixels_refuse_to_replace_a_file_and_leave_no_staging_file(tables) -> None:
    start_in_months(tables, {1: "2024-03", 6: "2024-05", 8: "2025-01"})
    run_monthly(tables, overwrite=True)
    with pytest.raises(InputError) as error:
        run_monthly(tables)
    assert error.value.code == "chain_maps_output_exists"
    run_monthly(tables, overwrite=True)
    assert not list((tables / "out").glob("*.partial"))


def test_month_range_runs_from_the_first_to_the_last_month() -> None:
    assert maps.month_range("2026-11", "2027-02") == ["2026-11", "2026-12", "2027-01", "2027-02"]
    assert maps.month_range("2026-03", "2026-03") == ["2026-03"]


def test_a_chain_that_goes_on_into_later_months_is_drawn_shorter_in_each_with_its_remaining_colour(tables) -> None:
    # root 1's chain 1-2-3-4 (branch seed 5) starts in 2024-03; seeds 3 and 5 are of 2024-04 and seed 4 of 2024-05; seed 1 is
    # of the month before the start, which does not make it earlier. Root 6's chain 6-7 and seed 8 are all of 2024-03.
    start_in_months(tables, {1: "2024-03", 6: "2024-03", 8: "2024-03"},
                    seed_months={1: "2024-02", 3: "2024-04", 4: "2024-05", 5: "2024-04"})
    report = run_monthly(tables, overwrite=True)
    assert list(report.months) == ["2024-03", "2024-04", "2024-05"]
    assert [report.months[m]["chains"] for m in report.months] == [2, 1, 0]               # drawn: 1-2-3-4 and 6-7; 3-4; none
    assert [report.months[m]["started"] for m in report.months] == [2, 0, 0]
    assert [report.months[m]["havens"] for m in report.months] == [8, 3, 1]                # seeds 3, 4, 5 and then seed 4
    path = maps.monthly_path(tables / "out")
    run_yearly(tables, years=[2024], overwrite=True)
    yearly, _ = read_image(maps.yearly_path(tables / "out", 2024))
    first, first_codes = month_image(path, "2024-03")
    assert (first == yearly).all()                                                        # the start month is the whole image
    intervals = maps.build_intervals(np.array([4, 2]))
    long_class, short_class = (int(c) for c in styles.class_of(np.array([4, 2]), intervals))
    haven = len(intervals) + 2
    assert long_class != short_class and long_class in first_codes and short_class in first_codes
    second, second_codes = month_image(path, "2024-04")
    seeds = {(2, 6), (3, 9), (4, 12)}                                                      # seeds 3, 4 and 5 at (k - 1, 3 (k - 1))
    with np.load(path) as data:
        lo, hi = (int(data["offsets"][1]), int(data["offsets"][2]))
        pixels = list(zip(data["row"][lo:hi].tolist(), data["column"][lo:hi].tolist(), data["code"][lo:hi].tolist()))
    assert {(r, c) for r, c, k in pixels if k == haven} == seeds
    segment = [(r, c, k) for r, c, k in pixels if k != haven]
    assert segment and {k for _r, _c, k in segment} == {short_class}                      # 3 -> 4 is 2 seeds long now
    assert all(2 <= r <= 3 and 6 <= c <= 9 for r, c, _k in segment)
    third, third_codes = month_image(path, "2024-05")
    assert (third[3] > 0).sum() == 1 and third_codes.tolist() == [haven] and third[3, 3, 9] == 255   # seed 4 alone
    with np.load(path) as data:
        assert data["started"].tolist() == [2, 0, 0] and data["chains"].tolist() == [2, 1, 0]


def test_a_chain_with_one_seed_left_is_a_lone_haven_and_the_months_run_to_the_last_seed(tables) -> None:
    # seed 4 alone reaches 2024-09, seed 5 is of 2024-04, and root 6's chain 6-7 starts only in 2024-06
    start_in_months(tables, {1: "2024-03", 6: "2024-06", 8: "2024-03"}, seed_months={4: "2024-09", 5: "2024-04"})
    report = run_monthly(tables, overwrite=True)
    assert list(report.months) == [f"2024-{m:02d}" for m in range(3, 10)]                 # to seed 4's month
    assert [report.months[m]["chains"] for m in report.months] == [1, 0, 0, 1, 0, 0, 0]   # only 6-7 in 2024-06 has 2 seeds left
    assert [report.months[m]["started"] for m in report.months] == [1, 0, 0, 1, 0, 0, 0]
    assert [report.months[m]["havens"] for m in report.months] == [6, 2, 1, 3, 1, 1, 1]   # seed 4 stays to 2024-09
    assert [report.months[m]["pixels"] > 0 for m in report.months] == [True] * 7
