"""Stage 3 of the rover chain: the forward floods of the plan seeds, from the light store."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

import lunarscout as ls
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_floods as floods
from lunarscout._numba_horizon import safe_haven_plan as plan
from lunarscout._numba_horizon import safe_haven_psr as psr
from lunarscout._numba_horizon.file_format import PATCH_SIZE
from lunarscout._numba_horizon.safe_haven_flood_packed import pack_bits, unpack_bits
from lunarscout._numba_horizon.safe_haven_windows import WindowParameters
from lunarscout.errors import InputError
from lunarscout.products import _load_dem
from test_safe_haven_stage3_store import prepared  # noqa: F401  (the block scene and a 2-hour store of it)

SIZE = 300
STEP_MIN = 120
EIGHT, FOUR = np.ones((3, 3), bool), ndimage.generate_binary_structure(2, 1)


def parameters(**overrides):
    settings = dict(cell_patches=1, haven_neighbourhood_m=3.0, seed_separation_m=4.0, operational_hours=24.0,
                    forward_only=True)
    settings.update(overrides)
    return WindowParameters(**settings)


@pytest.fixture(scope="module")
def seeds(prepared, tmp_path_factory):  # noqa: F811
    path = tmp_path_factory.mktemp("floods") / "seeds.csv"
    plan.plan_seeds(path, dem=prepared.dem, stage1=prepared.stage1, light_store=prepared.light_store,
                    times=prepared.stamps, earth_vectors_at=scene.earth_vectors, parameters=parameters())
    return path, plan.read_seed_table(path)


@pytest.fixture(scope="module")
def regions(prepared, tmp_path_factory):  # noqa: F811
    """A PSR map of the scene: a 3 x 3 region (small), a 10 x 10 region (medium at 1 m pixels) and a
    one-pixel region, written as the PSR products are."""
    directory = tmp_path_factory.mktemp("psr")
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[20:23, 150:153] = True          # 9 m2
    mask[200:210, 20:30] = True          # 100 m2
    mask[270, 270] = True                # 1 m2
    ids, count = psr.label_psr(mask)
    georef = _load_dem(prepared.directory / "dem.tif")[1]
    ls.write_geotiff(directory / "ids.tif", ids, georef.with_nodata(None))
    psr.write_region_table(directory / "table.csv", psr.region_table(ids, count, georef.affine_transform))
    return directory / "ids.tif", directory / "table.csv", ids


def settings(**overrides):
    base = dict(max_days=10.0, unit_cache_mib=64, psr_small_max_m2=40.0, psr_medium_max_m2=1_000_000.0)
    base.update(overrides)
    return floods.FloodSettings(**base)


def make_context(prepared, seeds_list, regions, *, passable=None, flood_settings=None):  # noqa: F811
    flood_settings = flood_settings or settings()
    ids, masks = floods.psr_class_masks(regions[0], regions[1], flood_settings)
    packed = pack_bits(np.ones((SIZE, SIZE), bool) if passable is None else passable)
    by_cycle, later = floods.build_targets(seeds_list, flood_settings)
    return floods.FloodContext(
        settings=flood_settings, dem_shape=(SIZE, SIZE), step_minutes=STEP_MIN, passable=packed, psr_masks=masks,
        psr_ids=ids, directory=prepared.light_store, seeds=seeds_list, by_cycle=by_cycle, later=later,
        unit_budget=flood_settings.unit_cache_mib * 2**20)


# --- a brute-force flood on unpacked bits -----------------------------------------------------------


def lit_cube(prepared, first_step, stop_step):  # noqa: F811
    """Operating-plane bits ``(steps, 300, 300)``; steps outside the store are dark."""
    store = prepared.light_store
    cube = np.zeros((stop_step - first_step, SIZE, SIZE), dtype=bool)
    low, high = max(first_step, store.first_step), min(stop_step, store.stop_step)
    if low >= high:
        return cube
    for pr in range(3):
        for pc in range(3):
            opened = store.open_patch(pr, pc)
            if opened is None:
                continue
            with opened:
                bits = unpack_bits(opened.read_steps(low, high)[1], PATCH_SIZE)
            rows, columns = min(PATCH_SIZE, SIZE - pr * PATCH_SIZE), min(PATCH_SIZE, SIZE - pc * PATCH_SIZE)
            cube[low - first_step: high - first_step, pr * PATCH_SIZE: pr * PATCH_SIZE + rows,
                 pc * PATCH_SIZE: pc * PATCH_SIZE + columns] = bits[:, :rows, :columns]
    return cube


def reference(prepared, seed, regions, flood_settings, targets, passable=None):  # noqa: F811
    """``(endurance_hours, reach, touches, arrived)`` by plain dilation."""
    ids = regions[2]
    table = psr.read_region_table(regions[1])
    category = np.zeros(len(table) + 1, dtype=np.int8)
    category[1:] = np.where(table.area_m2 < flood_settings.psr_small_max_m2, 1,
                            np.where(table.area_m2 <= flood_settings.psr_medium_max_m2, 2, 3))
    classes = category[ids]
    ok = np.ones((SIZE, SIZE), bool) if passable is None else passable
    max_steps = int(flood_settings.max_days * 24 * 60 / STEP_MIN)
    structure = EIGHT if flood_settings.connectivity == 8 else FOUR
    after = seed["after_step"]
    cube = lit_cube(prepared, after + 1, after + max_steps)
    cur = np.zeros((SIZE, SIZE), bool)
    cur[seed["row"], seed["column"]] = True
    touches, arrived, count, done, died = {}, set(), 0, 0, False
    for j in range(1, max_steps):
        for _ in range(flood_settings.moves_per_step):
            grown = ndimage.binary_dilation(cur, structure=structure) & ok
            for number, name in enumerate(floods.PSR_CLASSES, start=1):
                hits = grown & (classes == number)
                if name not in touches and hits.any():
                    row, column = np.argwhere(hits)[0]
                    touches[name] = (j, int(ids[row, column]))
            cur = grown & cube[j - 1]
            if not cur.any():
                died = True
                break
        if died:
            break
        done = j
        if j == int(flood_settings.reach_hours * 60 / STEP_MIN):
            count = int(cur.sum())
        for target_id, row, column, before in targets:
            if before - after == j and cur[row, column]:
                arrived.add(target_id)
    return (done + 1) * STEP_MIN / 60, count, touches, arrived


def targets_of(seeds_list, seed, flood_settings):
    if flood_settings.edge_rule == "next_cycle":
        pool = [s for s in seeds_list if s["earth_cycle"] == seed["earth_cycle"] + 1]
    else:
        pool = seeds_list
    return [(s["seed_id"], s["row"], s["column"], s["before_step"]) for s in pool
            if s["before_step"] is not None and s["seed_id"] != seed["seed_id"]
            and 1 <= s["before_step"] - seed["after_step"] <= int(flood_settings.max_days * 24 * 60 / STEP_MIN)]


def check_against_reference(prepared, seeds_list, regions, flood_settings, sample, passable=None):  # noqa: F811
    context = make_context(prepared, seeds_list, regions, passable=passable, flood_settings=flood_settings)
    seen_edges = seen_touches = 0
    for seed in sample:
        row, edges, _ = floods.flood_seed(context, seed)
        hours, count, touches, arrived = reference(
            prepared, seed, regions, flood_settings, targets_of(seeds_list, seed, flood_settings), passable)
        assert row["lit_endurance_hours"] == hours, seed["seed_id"]
        assert row["reach48_pixels"] == count, seed["seed_id"]
        assert {t for t, _ in edges} == arrived, seed["seed_id"]
        for name in floods.PSR_CLASSES:
            if name in touches:
                assert (row[f"touch_{name}_hours"], row[f"touch_{name}_psr"]) == (touches[name][0] * STEP_MIN / 60,
                                                                                 touches[name][1])
            else:
                assert row[f"touch_{name}_hours"] is None and row[f"touch_{name}_psr"] is None
        seen_edges += len(edges)
        seen_touches += len(touches)
    return seen_edges, seen_touches


def spread_sample(seeds_list, count=8):
    picks = np.linspace(0, len(seeds_list) - 1, count).astype(int)
    return [seeds_list[i] for i in picks]


# --- the flood ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("connectivity, max_days", [(8, 10.0), (4, 10.0), (8, 28.0)])
def test_the_flood_equals_a_plain_dilation_reference(prepared, seeds, regions, connectivity, max_days) -> None:  # noqa: F811
    flood_settings = settings(connectivity=connectivity, max_days=max_days)
    edges, touches = check_against_reference(prepared, seeds[1], regions, flood_settings, spread_sample(seeds[1]))
    assert touches > 0
    if max_days == 28.0:
        assert edges > 0, "the scene's January havens reach February's"


def test_walls_in_the_passable_mask_stop_the_frontier_and_the_reference_agrees(prepared, seeds, regions) -> None:  # noqa: F811
    passable = np.ones((SIZE, SIZE), bool)
    passable[:, 100] = False                      # a wall the 8-neighbour frontier cannot cross
    passable[150, 140:] = False
    open_context = make_context(prepared, seeds[1], regions)
    walled = make_context(prepared, seeds[1], regions, passable=passable)
    cluster_seed = next(s for s in seeds[1] if s["column"] < 100 and s["row"] < 150)
    assert (floods.flood_seed(walled, cluster_seed)[0]["reach48_pixels"]
            < floods.flood_seed(open_context, cluster_seed)[0]["reach48_pixels"])
    check_against_reference(prepared, seeds[1], regions, settings(), spread_sample(seeds[1], 6), passable)


def test_a_flood_that_reaches_the_cap_reports_the_cap(prepared, seeds, regions) -> None:  # noqa: F811
    capped = settings(max_days=2.0)
    context = make_context(prepared, seeds[1], regions, flood_settings=capped)
    rows = [floods.flood_seed(context, s)[0] for s in seeds[1][:6]]
    assert all(r["lit_endurance_hours"] == 48.0 for r in rows)          # lit for the 2 days: lower bound at the cap
    longer = make_context(prepared, seeds[1], regions, flood_settings=settings(max_days=28.0))
    assert floods.flood_seed(longer, seeds[1][0])[0]["lit_endurance_hours"] > 48.0


def test_a_seed_on_impassable_terrain_is_flagged_and_the_rover_moves_off_it(prepared, seeds, regions) -> None:  # noqa: F811
    seed = seeds[1][0]
    passable = np.ones((SIZE, SIZE), bool)
    passable[seed["row"], seed["column"]] = False
    row, _, _ = floods.flood_seed(make_context(prepared, seeds[1], regions, passable=passable), seed)
    assert row["seed_passable"] == 0 and row["lit_endurance_hours"] > 2.0
    assert floods.flood_seed(make_context(prepared, seeds[1], regions), seed)[0]["seed_passable"] == 1
    check_against_reference(prepared, seeds[1], regions, settings(), [seed], passable)
    surrounded = np.ones((SIZE, SIZE), bool)                     # a seed walled in on every side stops at once
    surrounded[seed["row"] - 1: seed["row"] + 2, seed["column"] - 1: seed["column"] + 2] = False
    row, edges, _ = floods.flood_seed(make_context(prepared, seeds[1], regions, passable=surrounded), seed)
    assert row["lit_endurance_hours"] == 2.0 and row["reach48_pixels"] == 0 and edges == []


def test_psr_touches_name_the_class_and_the_region_and_need_passable_pixels(prepared, seeds, regions) -> None:  # noqa: F811
    context = make_context(prepared, seeds[1], regions)
    seed = next(s for s in seeds[1] if s["row"] < 140)
    row = floods.flood_seed(context, seed)[0]
    ids = regions[2]
    assert row["touch_small_psr"] == ids[20, 150] or row["touch_small_psr"] in (ids[270, 270],)
    assert row["touch_medium_psr"] == ids[200, 20] and row["touch_large_psr"] is None
    blocked = np.ones((SIZE, SIZE), bool)
    blocked[ids > 0] = False
    row = floods.flood_seed(make_context(prepared, seeds[1], regions, passable=blocked), seed)[0]
    assert all(row[f"touch_{name}_hours"] is None for name in floods.PSR_CLASSES)


def test_the_psr_classes_follow_the_area_limits(regions) -> None:
    ids, masks = floods.psr_class_masks(regions[0], regions[1], settings())
    small, medium, large = (unpack_bits(m, SIZE) for m in masks)
    assert small.sum() == 9 + 1 and medium.sum() == 100 and large.sum() == 0
    _, masks = floods.psr_class_masks(regions[0], regions[1], settings(psr_small_max_m2=5.0, psr_medium_max_m2=50.0))
    small, medium, large = (unpack_bits(m, SIZE) for m in masks)
    assert (small.sum(), medium.sum(), large.sum()) == (1, 9, 100)
    assert not (small & medium).any() and not (medium & large).any()


def test_the_next_cycle_rule_and_the_any_later_rule(prepared, seeds, regions) -> None:  # noqa: F811
    january = [s for s in seeds[1] if s["month"] == "2027-01"][:3]
    # a seed of the same cycle that is due 50 steps into the flood: only "any_later" can arrive at it
    extra = dict(seed_id=len(seeds[1]) + 1, month="2027-01", cell_row=0, cell_col=0, row=70, column=70,
                 hibernation_hours=0.0, outage_kind="outage_sun_lit", earth_cycle=0, outage_start_utc="", outage_end_utc="",
                 before_step=january[0]["after_step"] + 50, after_step=january[0]["after_step"] + 60)
    everyone = seeds[1] + [extra]
    nxt, later = settings(max_days=28.0), settings(max_days=28.0, edge_rule="any_later")
    cycle_of = {s["seed_id"]: s["earth_cycle"] for s in everyone}
    for seed in january:
        n = floods.flood_seed(make_context(prepared, everyone, regions, flood_settings=nxt), seed)[1]
        w = floods.flood_seed(make_context(prepared, everyone, regions, flood_settings=later), seed)[1]
        assert {cycle_of[t] for t, _ in n} == {1}
        assert {t for t, _ in n} <= {t for t, _ in w}
        assert extra["seed_id"] in {t for t, _ in w} and extra["seed_id"] not in {t for t, _ in n}
    check_against_reference(prepared, everyone, regions, later, january)
    check_against_reference(prepared, everyone, regions, nxt, january)


def test_arrival_is_the_frontier_holding_the_pixel_at_the_targets_before_step(prepared, seeds, regions) -> None:  # noqa: F811
    seed = next(s for s in seeds[1] if s["month"] == "2027-01" and (s["row"], s["column"]) == (60, 60))
    flood_settings = settings(max_days=28.0)
    hours = floods.flood_seed(make_context(prepared, seeds[1], regions, flood_settings=flood_settings), seed)[0][
        "lit_endurance_hours"]
    last = seed["after_step"] + int(hours / 2) - 1               # the last step at which the frontier is not empty
    assert hours < 28 * 24                                       # the Sun sets on it inside the 28 days

    def target(offset, number):
        return dict(seed_id=1000 + number, month="2027-02", cell_row=0, cell_col=0, row=61 + number, column=61 + number,        # cluster pixels: the last to be lit
                    hibernation_hours=0.0, outage_kind="outage_sun_lit", earth_cycle=seed["earth_cycle"] + 1,
                    outage_start_utc="", outage_end_utc="", before_step=last + offset, after_step=last + offset + 5)

    extra = [target(-3, 0), target(0, 1), target(1, 2), target(2, 3)]
    context = make_context(prepared, seeds[1] + extra, regions, flood_settings=flood_settings)
    arrived = {t for t, _ in floods.flood_seed(context, seed)[1]}
    assert {1000, 1001} <= arrived and not {1002, 1003} & arrived
    check_against_reference(prepared, seeds[1] + extra, regions, flood_settings, [seed])


# --- lit units ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("per_day", [12, 10])               # aligned with the store's chunks, and not
def test_units_are_the_stores_operating_plane_and_dark_outside_it(prepared, per_day) -> None:  # noqa: F811
    store = prepared.light_store
    cache = floods.UnitCache(store, (SIZE, SIZE), per_day, 2**30)
    opened = store.open_patch(1, 2)
    for day in (store.first_step // per_day - 1, store.first_step // per_day, 5 + store.first_step // per_day + 1,
                store.stop_step // per_day - 1, store.stop_step // per_day + 3):
        unit = cache.get(1, 2, day)
        want = np.zeros((per_day, PATCH_SIZE, 2), dtype=np.uint64)
        a, b = max(day * per_day, store.first_step), min((day + 1) * per_day, store.stop_step)
        if a < b:
            want[a - day * per_day: b - day * per_day] = opened.read_steps(a, b)[1]
        if a >= b:
            assert unit is None
        else:
            np.testing.assert_array_equal(unit, want)
    assert cache.get(5, 5, store.first_step // per_day + 2) is None            # outside the DEM
    again = cache.get(1, 2, store.first_step // per_day + 4)
    assert again is cache.get(1, 2, store.first_step // per_day + 4) and cache.hits > 0
    opened.close()


def test_a_tiny_cache_still_gives_the_same_answers_and_a_missing_patch_is_dark(prepared, seeds, regions, tmp_path) -> None:  # noqa: F811
    import shutil

    from lunarscout._numba_horizon import safe_haven_stage0 as stage0
    from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory

    context = make_context(prepared, seeds[1], regions)
    big = [floods.flood_seed(context, s)[0] for s in spread_sample(seeds[1], 4)]
    tiny = make_context(prepared, seeds[1], regions, flood_settings=settings(unit_cache_mib=0))
    assert [floods.flood_seed(tiny, s)[0] for s in spread_sample(seeds[1], 4)] == big
    copy = tmp_path / "store"
    shutil.copytree(prepared.store_directory, copy)
    stage0.patch_file(copy, 1, 1).unlink()
    gap = make_context(prepared, seeds[1], regions)
    gap.directory = LightStoreDirectory(copy, prepared.dem)
    assert floods.flood_seed(gap, seeds[1][0])[0]["lit_endurance_hours"] <= big[0]["lit_endurance_hours"]


# --- the passable mask -------------------------------------------------------------------------------


def test_the_passable_mask_is_slope_validity_and_patch_presence(prepared, tmp_path, monkeypatch) -> None:  # noqa: F811
    import shutil

    from lunarscout._numba_horizon import safe_haven_stage0 as stage0
    from lunarscout._numba_horizon.safe_haven_stage0 import LightStoreDirectory

    monkeypatch.setattr(floods, "_STRIP_ROWS", 70)                      # several strips, so their seams are checked
    georef = _load_dem(prepared.directory / "dem.tif")[1]
    dem = dataclasses.replace(prepared.dem, elevation_m=np.array(prepared.dem.elevation_m, copy=True))
    dem.elevation_m[:, 200:] += np.arange(100)[None, :] * 0.5          # a 26.6 degree slope on the right
    dem.elevation_m[10:13, 10:13] = np.nan                              # invalid heights
    copy = tmp_path / "store"
    shutil.copytree(prepared.store_directory, copy)
    stage0.patch_file(copy, 2, 0).unlink()
    mask = unpack_bits(floods.passable_mask(dem, georef, LightStoreDirectory(copy, prepared.dem), 20.0), SIZE)
    reference_slope, _ = ls.slope(np.where(np.isfinite(dem.elevation_m), dem.elevation_m, -9999.0), georef.with_nodata(-9999.0),
                                  output_nodata=-1.0)
    want = (reference_slope >= 0) & (reference_slope <= 20.0)
    want[256:, :PATCH_SIZE] = False
    np.testing.assert_array_equal(mask, want)
    assert mask[100, 50] and not mask[100, 250] and not mask[11, 11] and not mask[0, 5] and not mask[270, 5]
    assert floods.passable_mask(dem, georef, LightStoreDirectory(copy, prepared.dem), 30.0)[100, 250 // 64] != 0


# --- the run -----------------------------------------------------------------------------------------


def run(prepared, seeds, regions, tmp_path, name="a", **overrides):  # noqa: F811
    options = dict(seeds_path=seeds[0], dem=prepared.dem, georef=_load_dem(prepared.directory / "dem.tif")[1],
                   directory=prepared.light_store, psr_ids_path=regions[0], psr_table_path=regions[1],
                   passable_path=tmp_path / "passable.npy", settings=settings())
    options.update(overrides)
    return floods.run_floods(tmp_path / f"{name}.csv", tmp_path / f"{name}_edges.csv", **options)


@pytest.fixture(scope="module")
def full_run(prepared, seeds, regions, tmp_path_factory):  # noqa: F811
    directory = tmp_path_factory.mktemp("run")
    report = run(prepared, seeds, regions, directory, settings=settings(max_days=28.0))
    return directory, report


def test_the_run_floods_every_seed_and_writes_the_two_tables(seeds, full_run) -> None:
    directory, report = full_run
    rows = floods.read_flood_table(directory / "a.csv")
    edges = floods.read_edge_table(directory / "a_edges.csv")
    assert [r["seed_id"] for r in rows] == [s["seed_id"] for s in seeds[1]]
    assert report.seeds == len(rows) == 54 and report.edges == len(edges) > 0 and report.seeds_resumed == 0
    assert all(r["seed_passable"] == 1 for r in rows) and report.seeds_not_on_passable_terrain == 0
    assert edges == sorted(edges, key=lambda e: (e[0], e[1]))
    by_id = {s["seed_id"]: s for s in seeds[1]}
    for seed_id, target, hours in edges:
        assert by_id[target]["earth_cycle"] == by_id[seed_id]["earth_cycle"] + 1
        assert hours == (by_id[target]["before_step"] - by_id[seed_id]["after_step"]) * 2.0
    written = json.loads((directory / "a.csv.report.json").read_text())
    assert written["edges"] == len(edges) and written["settings"]["max_days"] == 28.0
    assert not (directory / "a.csv.checkpoint.jsonl").exists()


def test_the_run_matches_the_reference_on_a_sample(prepared, seeds, regions, full_run) -> None:  # noqa: F811
    directory, _ = full_run
    rows = {r["seed_id"]: r for r in floods.read_flood_table(directory / "a.csv")}
    passable = unpack_bits(np.load(directory / "passable.npy"), SIZE)       # the mask the run made from the DEM
    assert not passable[0].any() and passable[1:-1, 1:-1].all()            # valid heights, flat, the edge has no slope
    context = make_context(prepared, seeds[1], regions, passable=passable, flood_settings=settings(max_days=28.0))
    for seed in spread_sample(seeds[1], 5):
        direct = floods.flood_seed(context, seed)[0]
        assert {k: v for k, v in rows[seed["seed_id"]].items()} == direct


def test_workers_and_a_resumed_run_give_the_same_tables(prepared, seeds, regions, full_run, tmp_path) -> None:  # noqa: F811
    directory, _ = full_run
    run(prepared, seeds, regions, tmp_path, "w", workers=2, settings=settings(max_days=28.0))
    assert (tmp_path / "w.csv").read_bytes() == (directory / "a.csv").read_bytes()
    assert (tmp_path / "w_edges.csv").read_bytes() == (directory / "a_edges.csv").read_bytes()


def test_an_interrupted_run_resumes_from_its_checkpoint(prepared, seeds, regions, full_run, tmp_path, monkeypatch) -> None:  # noqa: F811
    directory, _ = full_run
    original = floods.flood_group
    calls = []

    def stop_after_two(context, ids):
        if len(calls) == 2:
            raise KeyboardInterrupt
        calls.append(ids)
        return original(context, ids)

    monkeypatch.setattr(floods, "flood_group", stop_after_two)
    with pytest.raises(KeyboardInterrupt):
        run(prepared, seeds, regions, tmp_path, "r", settings=settings(max_days=28.0))
    assert (tmp_path / "r.csv.checkpoint.jsonl").exists() and not (tmp_path / "r.csv").exists()
    monkeypatch.setattr(floods, "flood_group", original)
    report = run(prepared, seeds, regions, tmp_path, "r", settings=settings(max_days=28.0), overwrite=True)
    assert report.seeds_resumed == sum(len(c) for c in calls) > 0
    assert (tmp_path / "r.csv").read_bytes() == (directory / "a.csv").read_bytes()
    assert (tmp_path / "r_edges.csv").read_bytes() == (directory / "a_edges.csv").read_bytes()
    assert not (tmp_path / "r.csv.checkpoint.jsonl").exists()


def test_a_checkpoint_of_other_settings_is_refused(prepared, seeds, regions, tmp_path, monkeypatch) -> None:  # noqa: F811
    original = floods.flood_group
    monkeypatch.setattr(floods, "flood_group", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        run(prepared, seeds, regions, tmp_path, "c")
    monkeypatch.setattr(floods, "flood_group", original)
    with pytest.raises(InputError) as caught:
        run(prepared, seeds, regions, tmp_path, "c", settings=settings(max_days=5.0), overwrite=True)
    assert caught.value.code == "floods_checkpoint_mismatch"


def test_a_subset_of_seeds_gives_their_rows(prepared, seeds, regions, full_run, tmp_path) -> None:  # noqa: F811
    directory, _ = full_run
    run(prepared, seeds, regions, tmp_path, "s", seed_ids=[3, 4, 30], settings=settings(max_days=28.0))
    got = floods.read_flood_table(tmp_path / "s.csv")
    every = {r["seed_id"]: r for r in floods.read_flood_table(directory / "a.csv")}
    assert [r["seed_id"] for r in got] == [3, 4, 30] and all(r == every[r["seed_id"]] for r in got)


def test_existing_outputs_are_not_replaced_and_the_passable_mask_is_cached(prepared, seeds, regions, full_run) -> None:  # noqa: F811
    directory, _ = full_run
    before = (directory / "a.csv").read_bytes()
    with pytest.raises(InputError) as caught:
        run(prepared, seeds, regions, directory, settings=settings(max_days=28.0))
    assert caught.value.code == "floods_output_exists" and (directory / "a.csv").read_bytes() == before
    stamp = (directory / "passable.npy").stat().st_mtime_ns
    run(prepared, seeds, regions, directory, "again", settings=settings(max_days=28.0))
    assert (directory / "passable.npy").stat().st_mtime_ns == stamp               # reused
    run(prepared, seeds, regions, directory, "slope", settings=settings(max_days=28.0, max_slope_deg=10.0))
    assert (directory / "passable.npy").stat().st_mtime_ns != stamp               # another slope limit remade it


def test_a_psr_map_of_another_size_is_refused(prepared, seeds, regions, tmp_path) -> None:  # noqa: F811
    small = np.zeros((20, 20), dtype=np.int32)
    ls.write_geotiff(tmp_path / "small.tif", small, scene.georef())
    with pytest.raises(InputError) as caught:
        run(prepared, seeds, regions, tmp_path, "x", psr_ids_path=tmp_path / "small.tif")
    assert caught.value.code == "floods_grid_mismatch"


def test_settings_are_validated() -> None:
    for bad in (dict(max_days=0), dict(moves_per_step=0), dict(connectivity=6), dict(max_slope_deg=90),
                dict(psr_small_max_m2=50, psr_medium_max_m2=10), dict(edge_rule="x")):
        with pytest.raises(ValueError):
            floods.FloodSettings(**bad)


# --- the script --------------------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_safe_haven_stage3.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_safe_haven_stage3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_importing_the_script_starts_nothing_and_its_defaults_match_the_earlier_stages() -> None:
    module = load_script()
    stage2 = importlib.util.spec_from_file_location("run_safe_haven_stage2", SCRIPT.parent / "run_safe_haven_stage2.py")
    two = importlib.util.module_from_spec(stage2)
    stage2.loader.exec_module(two)
    assert module.SEEDS_PATH == two.OUTPUT_PATH and module.LIGHT_STORE_PATH == two.LIGHT_STORE_PATH
    assert module.OUTPUT_PATH.name == "safe_haven_2h_floods.csv" and module.EDGES_PATH.name == "safe_haven_2h_edges.csv"
    assert module.SETTINGS.max_days == 28.0 and module.SETTINGS.psr_small_max_m2 == 40.0
    assert module.SETTINGS.psr_medium_max_m2 == 1_000_000.0 and module.SETTINGS.max_slope_deg == 20.0
    assert module.SEED_IDS is None and module.OVERWRITE is False


def test_the_script_writes_the_tables_and_prints_the_report(prepared, seeds, regions, full_run, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    module = load_script()
    for name, value in dict(
        DEM_PATH=prepared.directory / "dem.tif", LIGHT_STORE_PATH=prepared.store_directory, SEEDS_PATH=seeds[0],
        PSR_IDS_PATH=regions[0], PSR_TABLE_PATH=regions[1], PASSABLE_PATH=tmp_path / "passable.npy",
        OUTPUT_PATH=tmp_path / "floods.csv", EDGES_PATH=tmp_path / "edges.csv", SETTINGS=settings(max_days=28.0),
        FLOOD_WORKERS=1, PROGRESS_EVERY_GROUPS=1,
    ).items():
        monkeypatch.setattr(module, name, value)

    assert module.main() == 0

    printed = capsys.readouterr().out
    assert '"seeds": 54' in printed and "Stage 3:" in printed
    assert (tmp_path / "floods.csv").read_bytes() == (full_run[0] / "a.csv").read_bytes()
    with pytest.raises(InputError, match="exists"):
        module.main()
