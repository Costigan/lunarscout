"""Rover chain maps: the two chain sets, the rasters (full-resolution and dilated) and the GeoPackages."""

from __future__ import annotations

import importlib.util
import random
import shutil
import sqlite3
import struct
from types import SimpleNamespace
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from lunarscout._numba_horizon import safe_haven_chain_maps as maps
from lunarscout._numba_horizon import safe_haven_chains as chains
from lunarscout.errors import InputError
from test_safe_haven_chains import GT, write_tables

CRS = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
HEIGHT, WIDTH = 12, 30

# seeds 1..8 at pixels (i, 3i) for i = 0..7; edges 1>2, 2>3, 3>4, 2>5 (a branch), 6>7; seed 8 is alone
AFTER = [10, 20, 30, 40, 50, 60, 70, 80]
EDGES = [(1, 2), (2, 3), (3, 4), (2, 5), (6, 7)]


def write_dem(path: Path, height=HEIGHT, width=WIDTH) -> None:
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32", crs=CRS,
                       transform=Affine.from_gdal(*GT)) as dataset:
        dataset.write(np.zeros((1, height, width), dtype="float32"))


@pytest.fixture()
def tables(tmp_path):
    seeds, floods, edges = write_tables(tmp_path, AFTER, EDGES)
    chains.run_chains(tmp_path / "chains.csv", seeds_path=seeds, floods_path=floods, edges_path=edges, geo_transform=GT)
    write_dem(tmp_path / "dem.tif")
    return tmp_path


def run(tables, **options):
    options.setdefault("dilate_pixels", 2)
    return maps.run_chain_maps(tables / "out", chains_path=tables / "chains.csv", dem_path=tables / "dem.tif", **options)


def read_raster(path):
    """``(bands, info)``: the bands and what the header says, read while the file is open."""
    with rasterio.open(path) as dataset:
        info = SimpleNamespace(nodata=dataset.nodata, transform=dataset.transform, crs=dataset.crs,
                               descriptions=dataset.descriptions, tags=dataset.tags(), profile=dataset.profile)
        return dataset.read(), info


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
    # every segment of a chain has the chain's own length and mission (the end seed's), not the edge's
    assert chain_set.segments.length.tolist() == [4, 4, 4, 2] and chain_set.segments.mission.tolist() == [70, 70, 70, 30]
    assert chain_set.dot_segments.length.tolist() == [1] and chain_set.dot_segments.mission.tolist() == [10]


def forest(count: int, generator: random.Random):
    after = [generator.randrange(0, 30) for _ in range(count)]
    edges = sorted((a, b) for a in range(1, count + 1) for b in range(1, count + 1)
                   if after[b - 1] > after[a - 1] and generator.random() < 0.1)
    root, predecessor, length = chains.chain_labels(
        np.array(after, dtype=np.int64), np.array([e[0] for e in edges], dtype=np.int64),
        np.array([e[1] for e in edges], dtype=np.int64))
    rows = [dict(seed_id=i + 1, row=i % 7, column=i % 11, predecessor_id=int(predecessor[i]), root_id=int(root[i]),
                 chain_length=int(length[i]), mission_hours=float(generator.randrange(1, 500)), month="2024-01",
                 root_start_utc="2024-03-01T00:00:00.000000Z") for i in range(count)]
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
                     month="2024-01", root_start_utc="2024-01-01T00:00:00.000000Z")
                for i, (p, n, m) in enumerate(missions)]

    for missions, expected_end in (([(0, 1, 5), (1, 2, 50), (1, 2, 50)], 2),       # equal: the lower id
                                   ([(0, 1, 5), (1, 2, 50), (1, 2, 60)], 3),       # the longer mission
                                   ([(0, 1, 5), (1, 2, 70), (1, 2, 60)], 2)):
        _, _, chain_set = maps.build_chain_sets(table(missions))
        assert chain_set.line_attributes["end_seed_id"].tolist() == [expected_end]
        assert [list(line) for line in chain_set.lines] == [[1, expected_end]]


def test_months_are_counted_from_the_first_month_of_the_table() -> None:
    rows = [dict(month="2024-03", root_start_utc="2024-03-05T00:00:00.000000Z"),
            dict(month="2024-02", root_start_utc="2024-04-01T00:00:00.000000Z")]
    assert maps.month_origin(rows) == (2024, 2)
    assert maps.month_index(["2024-02-01T00:00:00Z", "2025-01-31T00:00:00Z"], (2024, 2)).tolist() == [1, 12]


# --- drawing ---------------------------------------------------------------------------------------------

def segments(*items):
    columns = list(zip(*items))
    return maps.Segments(*(np.array(c, dtype=np.int64) for c in columns))


@pytest.mark.parametrize("compiled", [True, False])
def test_a_segment_is_a_bresenham_line_with_its_ends_and_the_three_bands(compiled) -> None:
    bands = np.zeros((3, 12, 30), dtype=np.uint16)
    maps.draw_segments(bands, segments((0, 0, 3, 9, 4, 7, 70)), compiled=compiled)
    drawn = np.argwhere(bands[0] > 0)
    assert len(drawn) == 10 and tuple(drawn[0]) == (0, 0) and tuple(drawn[-1]) == (3, 9)
    assert set(np.unique(bands[0][bands[0] > 0])) == {4} and set(np.unique(bands[1][bands[0] > 0])) == {7}
    assert set(np.unique(bands[2][bands[0] > 0])) == {70}
    assert (bands[:, bands[0] == 0] == 0).all()
    assert np.all(np.abs(np.diff(drawn[:, 1])) <= 1) and np.all(np.abs(np.diff(drawn[:, 0])) <= 1)      # no gaps
    single = np.zeros((3, 12, 30), dtype=np.uint16)
    maps.draw_segments(single, segments((5, 5, 5, 5, 1, 1, 9)), compiled=compiled)
    assert np.argwhere(single[0] > 0).tolist() == [[5, 5]]


def test_the_longer_chain_is_on_top_where_lines_cross_whatever_the_order() -> None:
    items = [(0, 0, 10, 10, 2, 1, 20), (0, 10, 10, 0, 5, 2, 50), (5, 0, 5, 10, 3, 3, 30)]
    reference = None
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
        bands = np.zeros((3, 12, 30), dtype=np.uint16)
        maps.draw_segments(bands, segments(*[items[i] for i in order]))
        reference = bands if reference is None else reference
        assert np.array_equal(bands, reference)
    assert bands[0, 5, 5] == 5 and bands[1, 5, 5] == 2 and bands[2, 5, 5] == 50       # the crossing of all three
    assert bands[0, 5, 2] == 3 and bands[0, 2, 2] == 2                                  # where only one passes
    assert ((bands[0] == 5) == (bands[2] == 50)).all()                                   # the bands come from one segment


def test_equal_lengths_are_decided_by_the_longer_mission() -> None:
    bands = np.zeros((3, 12, 30), dtype=np.uint16)
    maps.draw_segments(bands, segments((0, 0, 10, 10, 3, 1, 40), (0, 10, 10, 0, 3, 2, 90)))
    assert bands[0, 5, 5] == 3 and bands[2, 5, 5] == 90 and bands[1, 5, 5] == 2


def test_a_stamp_dilates_the_line_and_is_clipped_at_the_border() -> None:
    bands = np.zeros((3, 12, 30), dtype=np.uint16)
    maps.draw_segments(bands, segments((6, 15, 6, 15, 1, 1, 1)), radius=2)
    assert (bands[0] > 0).sum() == 25
    corner = np.zeros((3, 12, 30), dtype=np.uint16)
    maps.draw_segments(corner, segments((0, 0, 0, 0, 1, 1, 1), (11, 29, 11, 29, 1, 1, 1)), radius=2)
    assert (corner[0] > 0).sum() == 18
    long = np.zeros((3, 12, 30), dtype=np.uint16)
    maps.draw_segments(long, segments((6, 3, 6, 20, 1, 1, 1)), radius=1)
    assert (long[0, 5:8, 3:21] > 0).all() and not long[0, :5].any() and not long[0, 8:].any()


def test_the_values_saturate_and_ends_outside_the_raster_are_clipped() -> None:
    bands = np.zeros((3, 12, 30), dtype=np.uint16)
    maps.draw_segments(bands, segments((0, 0, 0, 5, 70000, 70000, 70000)))
    assert bands[0, 0, 3] == 65535 and bands[2, 0, 3] == 65535
    maps.draw_segments(bands, segments((-3, -3, 2, 4, 1, 1, 1)), radius=0)           # does not raise or wrap


# --- the products ----------------------------------------------------------------------------------------

def test_the_rasters_hold_the_chains_on_the_dem_grid(tables) -> None:
    report = run(tables)
    out = tables / "out"
    assert report.seeds == 8 and report.predecessor_edges == 5 and report.chains == 2 and report.isolated_seeds == 1
    expected = {p.name for g in maps.output_paths(out).values() for p in g.values()} | {"safe_haven_stage_4_chain_maps.report.json"}
    assert {p.name for p in out.iterdir()} == expected
    edges, dataset = read_raster(out / "safe_haven_stage_4_predecessor_edges.tif")
    assert edges.shape == (3, HEIGHT, WIDTH) and edges.dtype == np.uint16 and dataset.nodata is None
    assert dataset.transform.to_gdal() == GT and dataset.crs is not None
    assert list(dataset.descriptions) == list(maps.BANDS) and dataset.profile["tiled"]
    assert dataset.tags["SEGMENTS"].startswith("straight lines") and "1 = 1970-01" in dataset.tags["BANDS"]
    # seeds: (0,0) (1,3) (2,6) (3,9) (4,12) (5,15) (6,18) (7,21)
    assert [int(edges[0, i, 3 * i]) for i in range(8)] == [4, 4, 4, 4, 3, 2, 2, 0]      # an edge's pixels carry its value
    assert edges[0, 7, 21] == 0 and edges[2, 4, 12] == 90 and edges[2, 3, 9] == 70
    longest, _ = read_raster(out / "safe_haven_stage_4_longest_chain_per_root.tif")
    assert [int(longest[0, i, 3 * i]) for i in range(8)] == [4, 4, 4, 4, 0, 2, 2, 1]     # seed 5 (the branch) is not drawn
    assert longest[0, 7, 21] == 1 and (longest[0] == 1).sum() == 1                       # the dot
    assert (longest[2][longest[0] == 4] == 70).all() and (longest[2][longest[0] == 2] == 30).all()
    assert (longest[1][longest[0] > 0] == 1).all()
    assert not np.isin(longest[0], [3]).any() and 3 in edges[0]                           # the branch is only in the edge map


def test_the_dilated_copies_are_wider_and_agree_with_the_thin_ones(tables) -> None:
    run(tables, dilate_pixels=2)
    out = tables / "out"
    for name in maps.CHAIN_SETS:
        thin, _ = read_raster(out / f"safe_haven_stage_4_{name}.tif")
        wide, dataset = read_raster(out / f"safe_haven_stage_4_{name}_dilated.tif")
        assert dataset.tags["DILATE_PIXELS"] == "2" and wide.shape == thin.shape
        assert (wide[0] > 0).sum() > 3 * (thin[0] > 0).sum()
        assert (wide[0][thin[0] > 0] > 0).all()                                      # the thin lines are inside the wide ones
        assert ((wide[0] == 0) <= (thin[0] == 0)).all() and wide[0].max() == thin[0].max()
        assert ((wide[0] > 0) == (wide[2] > 0)).all() and ((wide[0] > 0) == (wide[1] > 0)).all()
    dots, _ = read_raster(out / "safe_haven_stage_4_longest_chain_per_root_dilated.tif")
    # the dot (7 x 7, one larger than the chain stamps) shows at its corners; the chain 6>7 passes over its left edge
    assert dots[0, 7, 21] == dots[0, 4, 24] == dots[0, 10, 24] == dots[0, 10, 21] == 1
    assert dots[0, 11, 25] == 0 and dots[0, 3, 24] == 0 and dots[0, 7, 25] == 0


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


def test_the_line_layers_carry_a_default_style_by_chain_length(tables) -> None:
    run(tables)
    for name, table in (("predecessor_edges", "predecessor_edges"), ("longest_chain_per_root", "chains")):
        connection = sqlite3.connect(tables / "out" / f"safe_haven_stage_4_{name}.gpkg")
        renderer = qml_of(connection, table).find("renderer-v2")
        assert renderer.get("type") == "categorizedSymbol" and renderer.get("attr") == "chain_length"
        assert renderer.get("enableorderby") == "1"
        assert renderer.find("orderby/orderByClause").get("asc") == "1" and renderer.find("orderby/orderByClause").text == "chain_length"
        categories = renderer.findall("categories/category")
        assert [c.get("value") for c in categories] == [str(n) for n in range(2, 11)] + [""]       # 2 to 10, then the rest
        assert [c.get("label") for c in categories][:2] == ["2 seeds", "3 seeds"]
        symbols = {s.get("name"): s for s in renderer.findall("symbols/symbol")}
        assert {c.get("symbol") for c in categories} == set(symbols)
        widths, colours = [], []
        for category in categories[:-1]:
            options = {o.get("name"): o.get("value") for o in symbols[category.get("symbol")].find("layer/Option")}
            widths.append(float(options["line_width"]))
            colours.append(tuple(int(v) for v in options["line_color"].split(",")[:3]))
        assert widths == sorted(widths) and len(set(widths)) == 9                    # longer chains are wider
        luminance = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in colours]
        assert luminance == sorted(luminance) and len(set(colours)) == 9             # and brighter, every length distinct
        assert luminance[-1] - luminance[0] > 150


def test_the_dots_are_small_translucent_grey_and_come_first_in_the_file(tables) -> None:
    run(tables)
    connection = sqlite3.connect(tables / "out" / "safe_haven_stage_4_longest_chain_per_root.gpkg")
    assert [r[0] for r in connection.execute("SELECT table_name FROM gpkg_contents ORDER BY rowid")] == ["isolated_seeds", "chains"]
    renderer = qml_of(connection, "isolated_seeds").find("renderer-v2")
    options = {o.get("name"): o.get("value") for o in renderer.find("symbols/symbol/layer/Option")}
    red, green, blue, alpha = (int(v) for v in options["color"].split(","))
    assert renderer.get("type") == "singleSymbol" and red == green == blue and alpha < 255
    assert float(options["size"]) < 1.5 and options["outline_style"] == "no" and options["name"] == "circle"
    assert connection.execute("SELECT count(*) FROM layer_styles").fetchone()[0] == 2


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
    _, dataset = read_raster(tables / "out" / "safe_haven_stage_4_predecessor_edges_dilated.tif")
    assert dataset.tags["DILATE_PIXELS"] == "3"


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
    assert module.DILATE_PIXELS == 2 and module.OVERWRITE is False and module.RASTER_ROUTES is True
    three = importlib.util.spec_from_file_location("run_safe_haven_stage3", SCRIPT.parent / "run_safe_haven_stage3.py")
    stage3 = importlib.util.module_from_spec(three)
    three.loader.exec_module(stage3)
    assert module.LIGHT_STORE_PATH == stage3.LIGHT_STORE_PATH and module.PASSABLE_PATH == stage3.PASSABLE_PATH
    for name in ("max_days", "moves_per_step", "connectivity", "max_slope_deg"):          # what the flood followed
        assert getattr(module.FLOOD_SETTINGS, name) == getattr(stage3.SETTINGS, name)


def test_the_script_writes_the_six_files_and_prints_the_report(tables, monkeypatch, capsys) -> None:
    module = load_script()
    for name, value in dict(DEM_PATH=tables / "dem.tif", CHAINS_PATH=tables / "chains.csv",
                            OUTPUT_DIRECTORY=tables / "out", RASTER_ROUTES=False).items():
        monkeypatch.setattr(module, name, value)
    assert module.main() == 0
    printed = capsys.readouterr().out
    assert '"predecessor_edges": 5' in printed and '"isolated_seeds": 1' in printed
    assert len(list((tables / "out").glob("*.tif"))) == 4 and len(list((tables / "out").glob("*.gpkg"))) == 2
    assert "six files written" in (tables / "out" / "safe_haven_stage_4_chain_maps.log").read_text()
    with pytest.raises(InputError, match="exists"):
        module.main()
