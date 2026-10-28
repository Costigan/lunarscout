"""Chain maps whose rasters follow reconstructed rover routes, on the scene's real stage-2 and stage-3 tables."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
import rasterio

from lunarscout._numba_horizon import safe_haven_chain_maps as maps
from lunarscout._numba_horizon import safe_haven_chains as chains
from lunarscout._numba_horizon import safe_haven_paths as paths
from lunarscout.errors import InputError
from test_safe_haven_chains import GT
from test_safe_haven_floods import SIZE, full_run, regions, seeds, settings  # noqa: F401
from test_safe_haven_stage3_store import prepared  # noqa: F401


@pytest.fixture(scope="module")
def chain_table(prepared, seeds, full_run, tmp_path_factory):  # noqa: F811
    directory, _ = full_run
    out = tmp_path_factory.mktemp("routes") / "final.csv"
    chains.run_chains(out, seeds_path=seeds[0], floods_path=directory / "a.csv", edges_path=directory / "a_edges.csv",
                      geo_transform=GT)
    return out


def run(prepared, full_run, chain_table, out, **options):  # noqa: F811
    options.setdefault("dilate_pixels", 1)
    options.setdefault("flood_settings", settings(max_days=28.0))
    return maps.run_chain_maps(
        out, chains_path=chain_table, dem_path=prepared.directory / "dem.tif",
        light_store_path=prepared.store_directory, passable_path=full_run[0] / "passable.npy", **options)


def paint(chain_set, route_set, radius, dots_extra=1):
    """The three bands by a plain loop: dots first, then the routes shortest chain first."""
    bands = np.zeros((3, SIZE, SIZE), dtype=np.uint16)

    def stamp(row, column, radius, values):
        for r in range(max(row - radius, 0), min(row + radius, SIZE - 1) + 1):
            for c in range(max(column - radius, 0), min(column + radius, SIZE - 1) + 1):
                bands[:, r, c] = values

    dots = chain_set.dot_segments
    if dots is not None:
        for i in range(len(dots.length)):
            stamp(int(dots.row0[i]), int(dots.column0[i]), radius + (dots_extra if radius else 0),
                  (dots.length[i], dots.month[i], dots.mission[i]))
    segments = chain_set.segments
    for i in segments.ascending():
        rows, columns = route_set.path(int(chain_set.segment_seeds[i, 1]))
        for row, column in zip(rows, columns):
            stamp(int(row), int(column), radius, (segments.length[i], segments.month[i], segments.mission[i]))
    return bands


def test_the_rasters_follow_the_routes_and_equal_a_plain_painter(prepared, full_run, chain_table, tmp_path) -> None:  # noqa: F811
    report = run(prepared, full_run, chain_table, tmp_path / "out")
    rows = chains.read_chain_table(chain_table)
    _, edge_set, chain_set = maps.build_chain_sets(rows)
    route_set = paths.reconstruct_paths(rows, directory=prepared.light_store, passable_path=full_run[0] / "passable.npy",
                                        dem_shape=(SIZE, SIZE), settings=settings(max_days=28.0))
    assert report.raster_lines == "routes" and report.route_pixels == len(route_set.rows) > 0
    assert report.predecessor_edges == len(edge_set.segments.length) > 0 and report.chains > 0
    for name, chain in zip(maps.CHAIN_SETS, (edge_set, chain_set)):
        for kind, radius in (("raster", 0), ("dilated", 1)):
            with rasterio.open(tmp_path / "out" / f"safe_haven_stage_4_{name}{'_dilated' if kind == 'dilated' else ''}.tif") as dataset:
                bands, tags = dataset.read(), dataset.tags()
            assert tags["SEGMENTS"].startswith("rover routes")
            assert np.array_equal(bands, paint(chain, route_set, radius)), (name, kind)
    thin = rasterio.open(tmp_path / "out" / "safe_haven_stage_4_predecessor_edges.tif").read()
    by_id = {r["seed_id"]: r for r in rows}
    for row in rows:
        if row["predecessor_id"] > 0:
            source = by_id[row["predecessor_id"]]
            route_rows, route_columns = route_set.path(row["seed_id"])
            chebyshev = max(abs(row["row"] - source["row"]), abs(row["column"] - source["column"]))
            assert len(route_rows) >= chebyshev + 1                  # a route is never shorter than the straight line
            assert thin[0, row["row"], row["column"]] > 0            # every chain seed's pixel is on its edge's route


def test_the_geopackages_stay_straight_whatever_the_rasters_do(prepared, full_run, chain_table, tmp_path) -> None:  # noqa: F811
    run(prepared, full_run, chain_table, tmp_path / "routes")
    maps.run_chain_maps(tmp_path / "straight", chains_path=chain_table, dem_path=prepared.directory / "dem.tif",
                        dilate_pixels=1)
    for name, tables in (("predecessor_edges", ["predecessor_edges"]), ("longest_chain_per_root", ["chains", "isolated_seeds"])):
        for table in tables:
            contents = []
            for folder in ("routes", "straight"):
                connection = sqlite3.connect(tmp_path / folder / f"safe_haven_stage_4_{name}.gpkg")
                contents.append(connection.execute(f'SELECT * FROM "{table}" ORDER BY fid').fetchall())
                connection.close()
            assert contents[0] == contents[1] and contents[0]


def test_routes_need_the_cached_passable_mask_and_a_failure_writes_nothing(prepared, full_run, chain_table, tmp_path) -> None:  # noqa: F811
    with pytest.raises(InputError) as caught:
        maps.run_chain_maps(tmp_path / "a", chains_path=chain_table, dem_path=prepared.directory / "dem.tif",
                            light_store_path=prepared.store_directory, passable_path=tmp_path / "none.npy")
    assert caught.value.code == "paths_passable_missing"
    with pytest.raises(InputError) as caught:
        maps.run_chain_maps(tmp_path / "a", chains_path=chain_table, dem_path=prepared.directory / "dem.tif",
                            light_store_path=prepared.store_directory)
    assert caught.value.code == "chain_maps_setting_invalid"
    assert not (tmp_path / "a").exists() or not list((tmp_path / "a").iterdir())


def test_the_workers_do_not_change_the_rasters(prepared, full_run, chain_table, tmp_path) -> None:  # noqa: F811
    run(prepared, full_run, chain_table, tmp_path / "one")
    run(prepared, full_run, chain_table, tmp_path / "two", workers=2)
    for name in maps.CHAIN_SETS:
        for suffix in ("", "_dilated"):
            first = rasterio.open(tmp_path / "one" / f"safe_haven_stage_4_{name}{suffix}.tif").read()
            second = rasterio.open(tmp_path / "two" / f"safe_haven_stage_4_{name}{suffix}.tif").read()
            assert np.array_equal(first, second)
