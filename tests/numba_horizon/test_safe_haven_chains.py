"""Rover stage 4: the chain labels over the stage-3 edges, the final table, the report and the script."""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from lunarscout._numba_horizon import safe_haven_chains as chains
from lunarscout._numba_horizon import safe_haven_floods as floods
from lunarscout._numba_horizon import safe_haven_plan as plan
from lunarscout.errors import InputError
from test_safe_haven_floods import full_run, regions, seeds  # noqa: F401  (a real run of stages 2 and 3 on the scene)
from test_safe_haven_stage3_store import prepared  # noqa: F401


GT = (1000.0, 20.0, 0.0, -500.0, 0.0, -20.0)          # GDAL's transform of the pixel corner


def write_dem_header(path):
    """A tiny GeoTIFF whose transform is GT (the script reads only the header)."""
    import rasterio
    from rasterio.transform import Affine

    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1, dtype="float32",
                       transform=Affine.from_gdal(*GT)) as dataset:
        dataset.write(np.zeros((1, 8, 8), dtype="float32"))


# --- hand-built tables ---------------------------------------------------------------------------------

def write_tables(directory: Path, after, edges, *, endurance=None, before=None, name="t"):
    """A seed table, a flood table and an edge table for seeds 1..len(after) with the given after steps."""
    count = len(after)
    endurance = endurance or [10.0] * count
    seeds_path, floods_path, edges_path = (directory / f"{name}_seeds.csv", directory / f"{name}_floods.csv",
                                           directory / f"{name}_edges.csv")
    with seeds_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(plan.SEED_COLUMNS)
        for index, step in enumerate(after):
            writer.writerow([index + 1, "2024-01", 0, 0, index, 3 * index, -1.0, -1.0, 0.0, "outage_sun_lit", 5,
                             "2024-01-01T00:00:00.000000Z", "2024-01-01T04:00:00.000000Z",
                             "" if before is None else before[index], step])
    with floods_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(floods.FLOOD_COLUMNS)
        for index in range(count):
            writer.writerow([index + 1, 1, endurance[index], 100 + index, "", "", 14.0, 7, "", ""])
    with edges_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(floods.EDGE_COLUMNS)
        for source, target in edges:
            writer.writerow([source, target, 6.0])
    return seeds_path, floods_path, edges_path


def labels(after, edges):
    sources = np.array([e[0] for e in edges], dtype=np.int64)
    targets = np.array([e[1] for e in edges], dtype=np.int64)
    root, predecessor, length = chains.chain_labels(np.array(after, dtype=np.int64), sources, targets)
    return list(root), list(predecessor), list(length)


def test_a_seed_with_no_incoming_edge_is_its_own_root() -> None:
    assert labels([5, 6, 7], []) == ([1, 2, 3], [0, 0, 0], [1, 1, 1])
    root, predecessor, length = labels([5, 6, 7], [(1, 2)])
    assert (root, predecessor, length) == ([1, 1, 3], [0, 1, 0], [1, 2, 1])


def test_a_seed_takes_the_root_with_the_earliest_operations_start() -> None:
    # 3 is reached from 1 (root 1, after 10) and from 2 (root 2, after 20); 5 from 3 and from 4 (root 4, after 40)
    after = [10, 20, 30, 40, 50]
    root, predecessor, length = labels(after, [(2, 3), (1, 3), (4, 5), (3, 5)])
    assert root == [1, 2, 1, 4, 1] and predecessor == [0, 0, 1, 0, 3] and length == [1, 1, 2, 1, 3]


def test_equal_operations_starts_are_broken_by_the_lower_root_id() -> None:
    root, predecessor, _ = labels([10, 10, 30], [(2, 3), (1, 3)])
    assert root[2] == 1 and predecessor[2] == 1


def test_the_longer_chain_of_the_same_root_is_kept_then_the_lower_predecessor_id() -> None:
    # root 1 reaches 4 directly (length 2) and through 2 (length 3): the longer one, via 2
    root, predecessor, length = labels([10, 20, 25, 30], [(1, 4), (2, 4), (1, 2)])
    assert (root[3], predecessor[3], length[3]) == (1, 2, 3)
    # two predecessors of equal length 2: the lower id
    root, predecessor, length = labels([10, 20, 21, 30], [(1, 3), (1, 2), (3, 4), (2, 4)])
    assert (root[3], predecessor[3], length[3]) == (1, 2, 3)


def test_the_labels_do_not_depend_on_the_order_of_the_edges_or_of_equal_after_steps() -> None:
    generator = random.Random(3)
    after = [generator.randrange(0, 40) for _ in range(60)]
    edges = {(a, b) for a in range(1, 61) for b in range(1, 61) if after[b - 1] > after[a - 1] and generator.random() < 0.08}
    base = labels(after, sorted(edges))
    for _ in range(5):
        shuffled = sorted(edges, key=lambda _e: generator.random())
        assert labels(after, shuffled) == base


def brute_force(after, edges):
    """For every seed, the best root over all its ancestors and the longest chain from that root."""
    parents = {}
    for a, b in edges:
        parents.setdefault(b, set()).add(a)

    def ancestors(node, seen=None):
        seen = set() if seen is None else seen
        for p in parents.get(node, ()):
            if p not in seen:
                seen.add(p)
                ancestors(p, seen)
        return seen

    longest = {}

    def longest_path(node, root):
        """The seeds of the longest chain from ``root`` to ``node`` (0 if root does not reach it)."""
        if node == root:
            return 1
        key = (node, root)
        if key not in longest:
            best = 0
            for p in parents.get(node, ()):
                inner = longest_path(p, root)
                best = max(best, inner + 1 if inner else 0)
            longest[key] = best
        return longest[key]

    result = {}
    for node in range(1, len(after) + 1):
        roots = [a for a in ancestors(node) if not parents.get(a)]
        if not roots:
            result[node] = (node, 1)
            continue
        root = min(roots, key=lambda r: (after[r - 1], r))
        result[node] = (root, longest_path(node, root))
    return result


def test_the_labels_equal_an_exhaustive_search_on_random_graphs() -> None:
    for seed in range(8):
        generator = random.Random(seed)
        count = generator.randrange(5, 40)
        after = [generator.randrange(0, 25) for _ in range(count)]
        edges = sorted((a, b) for a in range(1, count + 1) for b in range(1, count + 1)
                       if after[b - 1] > after[a - 1] and generator.random() < 0.12)
        root, predecessor, length = labels(after, edges)
        expected = brute_force(after, edges)
        # a seed with an incoming edge has the best root of any chain reaching it, even one that starts at a
        # seed that itself has an incoming edge: those are not roots, so the best root is over the real roots
        for node in range(1, count + 1):
            assert (root[node - 1], length[node - 1]) == expected[node], (seed, node)
            if predecessor[node - 1]:
                assert (predecessor[node - 1], node) in edges
                assert length[node - 1] == length[predecessor[node - 1] - 1] + 1
                assert root[node - 1] == root[predecessor[node - 1] - 1]


# --- the run ---------------------------------------------------------------------------------------------

def test_the_final_table_carries_the_earlier_columns_and_the_chain_columns(tmp_path) -> None:
    after = [10, 20, 30, 40, 50]
    paths = write_tables(tmp_path, after, [(1, 3), (2, 3), (3, 5)], endurance=[10.0, 11.0, 12.0, 13.0, 14.0])
    report = chains.run_chains(tmp_path / "final.csv", seeds_path=paths[0], floods_path=paths[1], edges_path=paths[2], geo_transform=GT)
    rows = chains.read_chain_table(tmp_path / "final.csv")
    assert tuple(rows[0]) == chains.CHAIN_COLUMNS and [r["seed_id"] for r in rows] == [1, 2, 3, 4, 5]
    third, fifth, fourth = rows[2], rows[4], rows[3]
    assert (third["root_id"], third["predecessor_id"], third["chain_length"]) == (1, 1, 2)
    assert (fifth["root_id"], fifth["predecessor_id"], fifth["chain_length"]) == (1, 3, 3)
    assert (fourth["root_id"], fourth["predecessor_id"], fourth["chain_length"]) == (4, 0, 1)
    # mission: root 1 wakes at step 10; seed 5 wakes at step 50, 40 steps (80 h) later, and floods 14 h
    assert fifth["mission_hours"] == 94.0 and third["mission_hours"] == (30 - 10) * 2 + 12.0
    assert fourth["mission_hours"] == 13.0
    assert (third["previous_seeds"], third["next_seeds"], rows[0]["next_seeds"], fifth["next_seeds"]) == (2, 1, 1, 0)
    assert fifth["root_start_utc"] == "1970-01-01T20:00:00.000000Z"            # step 10 of 120 minutes
    assert fifth["touch_medium_hours"] == 14.0 and fifth["touch_medium_psr"] == 7 and fifth["touch_large_hours"] is None
    assert fifth["reach48_pixels"] == 104 and fifth["outage_kind"] == "outage_sun_lit" and fifth["before_step"] is None
    # seed 5 is pixel (row 4, column 12): the centre of GT's pixel, overwriting the stale -1 in the seed table
    assert (fifth["x_m"], fifth["y_m"]) == (1000.0 + 20.0 * 12.5, -500.0 - 20.0 * 4.5)
    assert [r["x_m"] for r in rows] == [1000.0 + 20.0 * (3 * i + 0.5) for i in range(5)]
    assert all(r[name] is None for r in rows for name in ("chain_psr_small", "chain_psr_medium", "chain_psr_large"))
    assert report.psr_counts is None                         # no routes were given: the PSR counts are blank
    assert report.seeds == 5 and report.edges == 3 and report.roots == 3 and report.seeds_in_chains == 4
    assert report.longest_chain == 3 and report.chain_length_counts == {1: 3, 2: 1, 3: 1}
    assert report.longest_mission_hours == 94.0
    assert report.top_chains[0] == {"seed_id": 5, "chain_length": 3, "mission_hours": 94.0, "chain": [1, 3, 5]}
    on_disk = json.loads((tmp_path / "final.csv.report.json").read_text())
    assert on_disk["longest_chain"] == 3 and on_disk["top_chains"][0]["chain"] == [1, 3, 5]


def test_the_step_length_scales_the_mission(tmp_path) -> None:
    paths = write_tables(tmp_path, [10, 40], [(1, 2)], endurance=[1.0, 5.0])
    chains.run_chains(tmp_path / "a.csv", seeds_path=paths[0], floods_path=paths[1], edges_path=paths[2], geo_transform=GT, step_minutes=30)
    assert chains.read_chain_table(tmp_path / "a.csv")[1]["mission_hours"] == 30 * 0.5 + 5.0


def test_the_table_does_not_depend_on_the_order_of_the_edge_file(tmp_path) -> None:
    after = [10, 20, 30, 40, 50, 60]
    edges = [(1, 3), (2, 3), (3, 5), (4, 5), (5, 6), (1, 6)]
    first = write_tables(tmp_path, after, edges, name="x")
    second = write_tables(tmp_path, after, list(reversed(edges)), name="y")
    chains.run_chains(tmp_path / "x.csv", seeds_path=first[0], floods_path=first[1], edges_path=first[2], geo_transform=GT)
    chains.run_chains(tmp_path / "y.csv", seeds_path=second[0], floods_path=second[1], edges_path=second[2], geo_transform=GT)
    assert (tmp_path / "x.csv").read_bytes() == (tmp_path / "y.csv").read_bytes()


def test_refusals(tmp_path) -> None:
    good = write_tables(tmp_path, [10, 20], [(1, 2)])
    run = lambda out, paths, **kw: chains.run_chains(tmp_path / out, seeds_path=paths[0], floods_path=paths[1], edges_path=paths[2], geo_transform=GT, **kw)  # noqa: E731
    run("o.csv", good)
    with pytest.raises(InputError) as caught:
        run("o.csv", good)
    assert caught.value.code == "chains_output_exists"
    run("o.csv", good, overwrite=True)
    backwards = write_tables(tmp_path, [10, 20], [(2, 1)], name="b")
    with pytest.raises(InputError) as caught:
        run("b.csv", backwards)
    assert caught.value.code == "chains_edge_order" and caught.value.details == {"source": 2, "target": 1}
    equal = write_tables(tmp_path, [10, 10], [(1, 2)], name="e")
    with pytest.raises(InputError) as caught:
        run("e.csv", equal)
    assert caught.value.code == "chains_edge_order"
    outside = write_tables(tmp_path, [10, 20], [(1, 3)], name="s")
    with pytest.raises(InputError) as caught:
        run("s.csv", outside)
    assert caught.value.code == "chains_inputs_mismatch"
    other = write_tables(tmp_path, [10, 20, 30], [], name="m")
    with pytest.raises(InputError) as caught:
        run("m.csv", (good[0], other[1], good[2]))
    assert caught.value.code == "chains_inputs_mismatch"
    for path in (good[0], good[1], good[2]):
        with pytest.raises(InputError):
            bad = tmp_path / "bad.csv"
            bad.write_text("a,b\n1,2\n")
            run("z.csv", tuple(bad if p == path else p for p in good))
    assert not (tmp_path / "z.csv").exists()


def test_no_edges_make_every_seed_a_one_seed_chain(tmp_path) -> None:
    paths = write_tables(tmp_path, [10, 20, 30], [])
    report = chains.run_chains(tmp_path / "n.csv", seeds_path=paths[0], floods_path=paths[1], edges_path=paths[2], geo_transform=GT)
    rows = chains.read_chain_table(tmp_path / "n.csv")
    assert [r["chain_length"] for r in rows] == [1, 1, 1] and [r["mission_hours"] for r in rows] == [10.0] * 3
    assert report.roots == 3 and report.seeds_in_chains == 0 and report.longest_chain == 1


# --- on the scene, from real stage-2 and stage-3 tables ----------------------------------------------------

def test_the_scene_chains_are_consistent_with_the_edges_and_an_exhaustive_search(seeds, full_run, tmp_path) -> None:  # noqa: F811
    directory, _ = full_run
    seeds_path, seed_rows = seeds
    out = tmp_path / "scene.csv"
    report = chains.run_chains(out, seeds_path=seeds_path, floods_path=directory / "a.csv", edges_path=directory / "a_edges.csv", geo_transform=GT)
    rows = chains.read_chain_table(out)
    edges = [(a, b) for a, b, _ in floods.read_edge_table(directory / "a_edges.csv")]
    assert len(rows) == len(seed_rows) == report.seeds and report.edges == len(edges) > 0
    after = [s["after_step"] for s in seed_rows]
    expected = brute_force(after, edges)
    for row in rows:
        assert (row["root_id"], row["chain_length"]) == expected[row["seed_id"]]
        assert row["mission_hours"] == (after[row["seed_id"] - 1] - after[row["root_id"] - 1]) * 2.0 + row["lit_endurance_hours"]
    assert max(r["chain_length"] for r in rows) == report.longest_chain >= 2
    flood_rows = floods.read_flood_table(directory / "a.csv")
    for row, flood, seed in zip(rows, flood_rows, seed_rows):
        assert all(row[name] == flood[name] for name in floods.FLOOD_COLUMNS) and all(row[n] == seed[n] for n in ("month", "row", "column"))


# --- the script ------------------------------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven_stage4.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_safe_haven_stage4", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_defaults_follow_stage_3_and_importing_it_starts_nothing() -> None:
    module = load_script()
    three = importlib.util.spec_from_file_location("run_safe_haven_stage3", SCRIPT.parent / "run_safe_haven_stage3.py")
    stage3 = importlib.util.module_from_spec(three)
    three.loader.exec_module(stage3)
    assert module.SEEDS_PATH == stage3.SEEDS_PATH and module.FLOODS_PATH == stage3.OUTPUT_PATH
    assert module.EDGES_PATH == stage3.EDGES_PATH and module.OUTPUT_PATH.name == "safe_haven_stage_4_chains.csv"
    assert module.STEP_MINUTES == 120 and module.OVERWRITE is False
    assert module.LIGHT_STORE_PATH == stage3.LIGHT_STORE_PATH and module.PASSABLE_PATH == stage3.PASSABLE_PATH
    assert module.PSR_IDS_PATH == stage3.PSR_IDS_PATH and module.PSR_TABLE_PATH == stage3.PSR_TABLE_PATH
    for name in ("max_days", "moves_per_step", "connectivity", "max_slope_deg", "psr_small_max_m2", "psr_medium_max_m2"):
        assert getattr(module.FLOOD_SETTINGS, name) == getattr(stage3.SETTINGS, name)       # what the routes must follow
