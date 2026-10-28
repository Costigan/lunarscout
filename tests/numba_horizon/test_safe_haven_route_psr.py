"""The PSRs a chain's route passes next to: the counts in the stage-4 table."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

import lunarscout as ls
from lunarscout._numba_horizon import safe_haven_chains as chains
from lunarscout._numba_horizon import safe_haven_floods as floods
from lunarscout._numba_horizon import safe_haven_paths as paths
from lunarscout._numba_horizon import safe_haven_psr as psr
from lunarscout._numba_horizon import safe_haven_route_psr as route_psr
from lunarscout.errors import InputError
from lunarscout.products import _load_dem
from test_safe_haven_chains import GT
from test_safe_haven_floods import SIZE, full_run, regions, seeds, settings  # noqa: F401
from test_safe_haven_stage3_store import prepared  # noqa: F401

COUNT_COLUMNS = ("chain_psr_small", "chain_psr_medium", "chain_psr_large")


def hand_paths(*routes):
    """A PathSet of ``{head id: [(row, column), ...]}`` routes."""
    heads = np.array([h for h, _ in routes], dtype=np.int64)
    lengths = [len(pixels) for _, pixels in routes]
    flat = [p for _, pixels in routes for p in pixels]
    return paths.PathSet(heads, np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
                         np.array([p[0] for p in flat], dtype=np.int32), np.array([p[1] for p in flat], dtype=np.int32))


# --- the pieces -------------------------------------------------------------------------------------------------

def test_a_route_passes_next_to_the_psr_pixels_among_its_eight_neighbours() -> None:
    ids = np.zeros((10, 12), dtype=np.int32)
    ids[2, 5] = 1              # diagonal to (3, 4)
    ids[3, 8] = 2              # two columns from (3, 6), none from (3, 7)? (3, 7) is adjacent
    ids[6, 2] = 3              # far from the route
    ids[3, 6] = 4              # on the route itself
    route = [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
    found = route_psr.passed_psrs(hand_paths((7, route), (9, [(8, 0), (8, 1)])), ids)
    assert found[7].tolist() == [1, 2, 4] and 9 not in found
    assert route_psr.passed_psrs(hand_paths((7, [(0, 0)])), np.zeros((4, 4), np.int32)) == {}
    edge = route_psr.passed_psrs(hand_paths((7, [(0, 0), (0, 1)])), np.pad(np.ones((1, 1), np.int32), ((0, 3), (2, 1))))
    assert edge[7].tolist() == [1]                                       # a PSR in a corner of the raster, route on the border
    with pytest.raises(InputError) as caught:
        route_psr.passed_psrs(hand_paths((7, [(20, 20)])), ids)
    assert caught.value.code == "route_psr_grid_mismatch"


def test_the_chain_counts_the_distinct_psrs_of_each_class_over_its_whole_route() -> None:
    # seeds 1..6: 1 > 2 > 3 (a chain), 2 > 4 (a branch), 5 alone, 6 a second root with 6 > 7
    predecessor = np.array([0, 1, 2, 2, 0, 0, 6])
    length = np.array([1, 2, 3, 3, 1, 1, 2])
    category = np.array([0, 1, 2, 3, 3, 2], dtype=np.int8)               # PSR ids 1..5: small, medium, large, large, medium
    edge_psrs = {2: np.array([1, 2]), 3: np.array([2, 3, 4]), 4: np.array([5]), 7: np.array([1])}
    counts = route_psr.chain_psr_counts(predecessor, length, edge_psrs, category)
    assert counts.tolist() == [[0, 0, 0],      # a root: no route
                               [1, 1, 0],      # PSRs 1 (small), 2 (medium)
                               [1, 1, 2],      # + 3 and 4 (large); PSR 2 is seen again on the edge 2 > 3 and counted once
                               [1, 2, 0],      # the branch: 1, 2 and 5 (medium)
                               [0, 0, 0], [0, 0, 0],
                               [1, 0, 0]]
    assert route_psr.chain_psr_counts(np.zeros(3, np.int64), np.ones(3, np.int64), {}, category).sum() == 0


# --- on the scene's real stage-2 and stage-3 tables --------------------------------------------------------------

@pytest.fixture(scope="module")
def scene(prepared, seeds, full_run, tmp_path_factory):  # noqa: F811
    """Stage-4 rows without PSR counts, their routes, and PSR blocks of three sizes placed beside some route pixels."""
    directory, _ = full_run
    work = tmp_path_factory.mktemp("psr")
    chains.run_chains(work / "plain.csv", seeds_path=seeds[0], floods_path=directory / "a.csv",
                      edges_path=directory / "a_edges.csv", geo_transform=GT)
    rows = chains.read_chain_table(work / "plain.csv")
    flood_settings = settings(max_days=28.0, psr_small_max_m2=1.5, psr_medium_max_m2=10.0)
    route_set = paths.reconstruct_paths(rows, directory=prepared.light_store, passable_path=directory / "passable.npy",
                                        dem_shape=(SIZE, SIZE), settings=flood_settings)
    ids = np.zeros((SIZE, SIZE), dtype=np.int32)
    generator = np.random.default_rng(5)
    number = 0
    for head in generator.permutation(route_set.heads)[:12]:
        route_rows, route_columns = route_set.path(int(head))
        k = int(generator.integers(len(route_rows)))
        size = (1, 2, 4)[number % 3]                                     # 1, 4 and 16 pixels of 1 m2
        top, left = min(int(route_rows[k]) + 1, SIZE - size), min(int(route_columns[k]) + 1, SIZE - size)
        if not ids[top:top + size, left:left + size].any():
            number += 1
            ids[top:top + size, left:left + size] = number
    georef = _load_dem(prepared.directory / "dem.tif")[1]
    labelled, count = psr.label_psr(ids > 0)
    assert count >= number > 3
    ids = labelled                                                       # the same blocks, ids as the region table counts them
    ls.write_geotiff(work / "ids.tif", ids, georef.with_nodata(None))
    psr.write_region_table(work / "table.csv", psr.region_table(ids, count, georef.affine_transform))
    return rows, route_set, ids, work, flood_settings


def routes_for(prepared, full_run, scene, **options):  # noqa: F811
    rows, _route_set, _ids, work, flood_settings = scene
    return chains.RouteInputs(
        light_store_path=prepared.store_directory, passable_path=full_run[0] / "passable.npy",
        psr_ids_path=work / "ids.tif", psr_table_path=work / "table.csv", dem_shape=(SIZE, SIZE),
        flood_settings=flood_settings, **options)


def brute_force_counts(rows, route_set, ids, category):
    by_id = {r["seed_id"]: r for r in rows}
    counts = np.zeros((len(rows), 3), dtype=int)
    for row in rows:
        chain, seed = set(), row
        while seed["predecessor_id"] > 0:
            route_rows, route_columns = route_set.path(seed["seed_id"])
            for r, c in zip(route_rows, route_columns):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE and ids[r + dr, c + dc] > 0:
                            chain.add(int(ids[r + dr, c + dc]))
            seed = by_id[seed["predecessor_id"]]
        for psr_id in chain:
            counts[row["seed_id"] - 1, category[psr_id] - 1] += 1
    return counts


def test_the_table_holds_the_counts_of_the_psrs_each_chain_passes(prepared, full_run, seeds, scene, tmp_path) -> None:  # noqa: F811
    rows, route_set, ids, work, flood_settings = scene
    directory, _ = full_run
    report = chains.run_chains(tmp_path / "final.csv", seeds_path=seeds[0], floods_path=directory / "a.csv",
                               edges_path=directory / "a_edges.csv", geo_transform=GT,
                               routes=routes_for(prepared, full_run, scene))
    final = chains.read_chain_table(tmp_path / "final.csv")
    assert tuple(final[0]) == chains.CHAIN_COLUMNS and chains.CHAIN_COLUMNS[-3:] == COUNT_COLUMNS
    category = floods.psr_categories(work / "table.csv", flood_settings)
    expected = brute_force_counts(rows, route_set, ids, category)
    got = np.array([[r[name] for name in COUNT_COLUMNS] for r in final])
    assert np.array_equal(got, expected) and got.sum() > 0
    assert (got.sum(axis=1) > 0).sum() > 3 and (got[:, 0] > 0).any() and (got[:, 1] > 0).any() and (got[:, 2] > 0).any()
    assert all(not any(r[name] for name in COUNT_COLUMNS) for r in final if r["predecessor_id"] == 0)      # a root has no route
    assert report.psr_counts == {"small": int((got[:, 0] > 0).sum()), "medium": int((got[:, 1] > 0).sum()),
                                 "large": int((got[:, 2] > 0).sum())}
    plain = chains.read_chain_table(work / "plain.csv")                # the other columns are those of the run without routes
    assert [{k: v for k, v in r.items() if k not in COUNT_COLUMNS} for r in final] == \
           [{k: v for k, v in r.items() if k not in COUNT_COLUMNS} for r in plain]


def test_the_workers_do_not_change_the_counts(prepared, full_run, seeds, scene, tmp_path) -> None:  # noqa: F811
    directory, _ = full_run
    for name, workers in (("one", 1), ("two", 2)):
        chains.run_chains(tmp_path / f"{name}.csv", seeds_path=seeds[0], floods_path=directory / "a.csv",
                          edges_path=directory / "a_edges.csv", geo_transform=GT,
                          routes=routes_for(prepared, full_run, scene, workers=workers))
    assert (tmp_path / "one.csv").read_bytes() == (tmp_path / "two.csv").read_bytes()


def test_a_psr_raster_of_another_size_or_with_unknown_ids_is_refused(prepared, full_run, seeds, scene, tmp_path) -> None:  # noqa: F811
    directory, _ = full_run
    georef = _load_dem(prepared.directory / "dem.tif")[1]
    work = scene[3]
    small = tmp_path / "small.tif"
    with rasterio.open(small, "w", driver="GTiff", height=5, width=5, count=1, dtype="int32",
                       transform=Affine.from_gdal(*GT)) as dataset:
        dataset.write(np.zeros((1, 5, 5), dtype="int32"))
    bad = routes_for(prepared, full_run, scene)
    for ids_path, code in ((small, "chains_psr_mismatch"),):
        with pytest.raises(InputError) as caught:
            chains.run_chains(tmp_path / "a.csv", seeds_path=seeds[0], floods_path=directory / "a.csv",
                              edges_path=directory / "a_edges.csv", geo_transform=GT,
                              routes=chains.RouteInputs(**{**bad.__dict__, "psr_ids_path": ids_path}))
        assert caught.value.code == code and not (tmp_path / "a.csv").exists()
    unknown = np.full((SIZE, SIZE), 0, dtype=np.int32)
    unknown[0, 0] = 10**6
    ls.write_geotiff(tmp_path / "unknown.tif", unknown, georef.with_nodata(None))
    with pytest.raises(InputError) as caught:
        chains.run_chains(tmp_path / "b.csv", seeds_path=seeds[0], floods_path=directory / "a.csv",
                          edges_path=directory / "a_edges.csv", geo_transform=GT,
                          routes=chains.RouteInputs(**{**bad.__dict__, "psr_ids_path": tmp_path / "unknown.tif"}))
    assert caught.value.code == "chains_psr_mismatch"
    assert work.exists()


# --- the script ---------------------------------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven_stage4.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_safe_haven_stage4", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_writes_the_table_with_the_psr_counts(prepared, full_run, seeds, scene, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    module = load_script()
    directory, _ = full_run
    work, flood_settings = scene[3], scene[4]
    for name, value in dict(
        DEM_PATH=prepared.directory / "dem.tif", SEEDS_PATH=seeds[0], FLOODS_PATH=directory / "a.csv",
        EDGES_PATH=directory / "a_edges.csv", OUTPUT_PATH=tmp_path / "final.csv", LIGHT_STORE_PATH=prepared.store_directory,
        PASSABLE_PATH=directory / "passable.npy", PSR_IDS_PATH=work / "ids.tif", PSR_TABLE_PATH=work / "table.csv",
        FLOOD_SETTINGS=flood_settings, ROUTE_WORKERS=1, PROGRESS_EVERY_SEEDS=1,
    ).items():
        monkeypatch.setattr(module, name, value)
    assert module.main() == 0
    printed = capsys.readouterr().out
    assert '"psr_counts"' in printed and '"longest_chain"' in printed
    final = chains.read_chain_table(tmp_path / "final.csv")
    assert any(r["chain_psr_medium"] for r in final) and "Routes:" in (tmp_path / "final.csv.log").read_text()
    with pytest.raises(InputError, match="exists"):
        module.main()
