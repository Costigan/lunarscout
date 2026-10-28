"""Rover paths along the predecessor edges: a flood with kept frontiers and a trace back from each target."""

from __future__ import annotations

import numpy as np
import pytest

from lunarscout._numba_horizon import safe_haven_chains as chains
from lunarscout._numba_horizon import safe_haven_floods as floods
from lunarscout._numba_horizon import safe_haven_paths as paths
from lunarscout._numba_horizon.safe_haven_flood_packed import unpack_bits
from lunarscout.errors import InputError
from test_safe_haven_chains import GT
from test_safe_haven_floods import SIZE, full_run, lit_cube, regions, seeds, settings  # noqa: F401
from test_safe_haven_stage3_store import prepared  # noqa: F401


@pytest.fixture(scope="module")
def table(prepared, seeds, full_run, tmp_path_factory):  # noqa: F811
    """The stage-4 rows of the scene, from the stage-2 and stage-3 tables of ``full_run``."""
    directory, _ = full_run
    out = tmp_path_factory.mktemp("chains") / "final.csv"
    chains.run_chains(out, seeds_path=seeds[0], floods_path=directory / "a.csv", edges_path=directory / "a_edges.csv",
                      geo_transform=GT)
    return chains.read_chain_table(out)


def reconstruct(prepared, full_run, table, **options):  # noqa: F811
    directory, _ = full_run
    options.setdefault("settings", settings(max_days=28.0))
    return paths.reconstruct_paths(table, directory=prepared.light_store, passable_path=directory / "passable.npy",
                                   dem_shape=(SIZE, SIZE), **options)


def passable_of(full_run):  # noqa: F811
    return unpack_bits(np.load(full_run[0] / "passable.npy"), SIZE)


def replay(rows, columns, lit_after, passable, moves=4):
    """Is there a schedule of the path? Walking it with at most ``moves`` one-pixel moves per step, entering only lit
    passable pixels and staying only on lit ones, from its first pixel at step 0 to its last at the final step.
    ``lit_after[j]`` is the lit mask of step ``j`` (index 0 is the start, not used)."""
    last = len(rows) - 1
    reachable = {0}
    for j in range(1, len(lit_after)):
        lit = lit_after[j]
        nxt = set()
        for i in reachable:
            if lit[rows[i], columns[i]]:
                nxt.add(i)                                      # stay
            for k in range(i + 1, min(i + moves, last) + 1):
                if not (lit[rows[k], columns[k]] and passable[rows[k], columns[k]]):
                    break
                nxt.add(k)
        reachable = nxt
        if not reachable:
            return False
    return last in reachable


def test_every_predecessor_edge_gets_a_feasible_path(prepared, full_run, table) -> None:  # noqa: F811
    result = reconstruct(prepared, full_run, table)
    by_id = {r["seed_id"]: r for r in table}
    heads = [r["seed_id"] for r in table if r["predecessor_id"] > 0]
    assert heads and result.heads.tolist() == sorted(heads)
    passable = passable_of(full_run)
    checked = 0
    for head in heads:
        row = by_id[head]
        source = by_id[row["predecessor_id"]]
        rows, columns = result.path(head)
        steps = row["before_step"] - source["after_step"]
        assert (rows[0], columns[0]) == (source["row"], source["column"])
        assert (rows[-1], columns[-1]) == (row["row"], row["column"])
        assert (np.maximum(abs(np.diff(rows)), abs(np.diff(columns))) == 1).all()           # neighbours, no repeats
        assert len(rows) <= 4 * steps + 1 and passable[rows, columns].all()
        lit = lit_cube(prepared, source["after_step"], source["after_step"] + steps + 1)
        assert replay(rows, columns, lit, passable), (source["seed_id"], head)
        checked += 1
    assert checked == len(heads) > 0


def test_the_paths_do_not_depend_on_the_workers_or_the_kernel_build(prepared, full_run, table) -> None:  # noqa: F811
    one = reconstruct(prepared, full_run, table)
    two = reconstruct(prepared, full_run, table, workers=2)
    for name in ("heads", "offsets", "rows", "columns"):
        assert np.array_equal(getattr(one, name), getattr(two, name))
    # the plain-Python kernels give the same path on the shortest edge
    by_id = {r["seed_id"]: r for r in table}
    head = min((r for r in table if r["predecessor_id"] > 0),
               key=lambda r: r["before_step"] - by_id[r["predecessor_id"]]["after_step"])
    subset = [head, {**by_id[head["predecessor_id"]], "predecessor_id": 0}]
    slow = reconstruct(prepared, full_run, subset, compiled=False)
    assert np.array_equal(slow.path(head["seed_id"])[0], one.path(head["seed_id"])[0])
    assert np.array_equal(slow.path(head["seed_id"])[1], one.path(head["seed_id"])[1])


def test_a_path_prefers_staying_and_is_never_longer_than_the_moves_allow(prepared, full_run, table) -> None:  # noqa: F811
    result = reconstruct(prepared, full_run, table)
    by_id = {r["seed_id"]: r for r in table}
    for head, start, stop in zip(result.heads, result.offsets[:-1], result.offsets[1:]):
        row = by_id[int(head)]
        steps = row["before_step"] - by_id[row["predecessor_id"]]["after_step"]
        assert stop - start <= 4 * steps + 1
    longest = int(np.diff(result.offsets).max())
    assert longest > 4                                          # some path really travels


def test_the_set_finds_paths_by_head_and_refuses_the_missing(prepared, full_run, table) -> None:  # noqa: F811
    result = reconstruct(prepared, full_run, table)
    some = int(result.heads[0])
    assert result.index([some, int(result.heads[-1])]).tolist() == [0, len(result.heads) - 1]
    for bad in (0, int(result.heads[-1]) + 1, 10**9):
        with pytest.raises(InputError) as caught:
            result.index([bad])
        assert caught.value.code == "paths_missing"
    empty = paths.PathSet(np.zeros(0, np.int64), np.zeros(1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32))
    with pytest.raises(InputError):
        empty.index([1])


def test_refusals(prepared, full_run, table, tmp_path) -> None:  # noqa: F811
    with pytest.raises(InputError) as caught:
        paths.reconstruct_paths(table, directory=prepared.light_store, passable_path=tmp_path / "none.npy",
                                dem_shape=(SIZE, SIZE), settings=settings(max_days=28.0))
    assert caught.value.code == "paths_passable_missing"
    with pytest.raises(InputError) as caught:                       # another slope limit: not the cached mask
        reconstruct(prepared, full_run, table, settings=settings(max_days=28.0, max_slope_deg=10.0))
    assert caught.value.code == "paths_passable_missing"
    edited = [dict(r) for r in table]
    head = next(r for r in edited if r["predecessor_id"] > 0)
    head["before_step"] = None
    with pytest.raises(InputError) as caught:
        reconstruct(prepared, full_run, edited)
    assert caught.value.code == "paths_step"
    head["before_step"] = 10**6                                     # due far beyond 28 days
    with pytest.raises(InputError) as caught:
        reconstruct(prepared, full_run, edited)
    assert caught.value.code == "paths_step"


# --- the trace kernel on random lit worlds with walls -------------------------------------------------------------

def dilate(frontier, eight):
    grown = frontier.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)) + (((1, 1), (1, -1), (-1, 1), (-1, -1)) if eight else ()):
        shifted = np.zeros_like(frontier)
        ys = slice(max(dy, 0), frontier.shape[0] + min(dy, 0))
        xs = slice(max(dx, 0), frontier.shape[1] + min(dx, 0))
        yt = slice(max(-dy, 0), frontier.shape[0] + min(-dy, 0))
        xt = slice(max(-dx, 0), frontier.shape[1] + min(-dx, 0))
        shifted[ys, xs] = frontier[yt, xt]
        grown |= shifted
    return grown


def build_world(generator, steps, moves, eight, height=40, width=100, dark=0.3):
    from lunarscout._numba_horizon.safe_haven_flood_packed import pack_bits

    lit = generator.random((steps + 1, height, width)) > dark
    passable = generator.random((height, width)) > 0.05
    seed = (int(generator.integers(10, height - 10)), int(generator.integers(30, width - 30)))
    frontiers = [np.zeros((height, width), bool)]
    frontiers[0][seed] = True
    for j in range(1, steps + 1):
        current = frontiers[-1]
        for _ in range(moves):
            current = dilate(current, eight) & lit[j] & passable
        frontiers.append(current)
    words = -(-width // 64)
    meta, total = paths._geometry(steps, moves, seed[0], seed[1], height, words)
    front, mask = np.zeros(total, np.uint64), np.zeros(total, np.uint64)
    for j in range(steps + 1):
        offset, n_rows, n_words, row0, word0 = (int(v) for v in meta[j])
        for buffer, plane in ((front, frontiers[j]), (mask, lit[j] & passable)):
            packed = pack_bits(plane)
            buffer[offset:offset + n_rows * n_words] = packed[row0:row0 + n_rows, word0:word0 + n_words].ravel()
    return lit, passable, seed, frontiers, front, mask, meta


@pytest.mark.parametrize("moves, eight", [(4, True), (3, True), (2, False), (4, False)])
@pytest.mark.parametrize("compiled", [True, False])
def test_the_trace_finds_a_schedule_through_walls_for_every_pixel_the_flood_reached(moves, eight, compiled) -> None:
    generator = np.random.default_rng(7 + moves + 10 * eight)
    kernel = paths.trace_kernel(compiled)
    traced = 0
    for _ in range(3 if not compiled else 8):
        steps = int(generator.integers(2, 7))
        lit, passable, seed, frontiers, front, mask, meta = build_world(generator, steps, moves, eight)
        reached = np.argwhere(frontiers[steps])
        if not len(reached):
            continue
        for row, column in reached[generator.permutation(len(reached))[:6]]:
            out_row, out_column = np.empty(moves * steps + 1, np.int64), np.empty(moves * steps + 1, np.int64)
            n = kernel(front, meta, mask, meta, int(row), int(column), steps, eight, moves, out_row, out_column)
            assert n >= 1
            rows, columns = out_row[:n][::-1], out_column[:n][::-1]
            assert (rows[0], columns[0]) == seed and (rows[-1], columns[-1]) == (row, column)
            step = np.maximum(abs(np.diff(rows)), abs(np.diff(columns)))
            assert (step == 1).all() and (eight or (abs(np.diff(rows)) + abs(np.diff(columns)) == 1).all())
            assert replay_world(rows, columns, lit, passable, moves)
            traced += 1
        outside = np.argwhere(~frontiers[steps])[0]
        n = kernel(front, meta, mask, meta, int(outside[0]), int(outside[1]), steps, eight, moves,
                   np.empty(moves * steps + 1, np.int64), np.empty(moves * steps + 1, np.int64))
        assert n == -1                                              # a pixel the flood did not reach has no route
    assert traced > 0


def replay_world(rows, columns, lit, passable, moves):
    last = len(rows) - 1
    reachable = {0}
    for j in range(1, len(lit)):
        nxt = set()
        for i in reachable:
            if lit[j][rows[i], columns[i]] and passable[rows[i], columns[i]]:
                nxt.add(i)
            for k in range(i + 1, min(i + moves, last) + 1):
                if not (lit[j][rows[k], columns[k]] and passable[rows[k], columns[k]]):
                    break
                nxt.add(k)
        reachable = nxt
    return last in reachable
