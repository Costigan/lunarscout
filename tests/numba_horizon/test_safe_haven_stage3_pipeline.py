"""The stage-3 pipeline: same table as the block loop for any workers and bounds, bounded memory,
resume, failure and interrupt handling."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import threading
import time

import pytest

import test_safe_haven_blocks as base            # the scene, the reference tables and helpers
from lunarscout._numba_horizon import safe_haven_blocks as blocks
from lunarscout._numba_horizon import safe_haven_stage3 as stage3
from lunarscout._numba_horizon.safe_haven_stage3 import HorizonCache, PipelineSettings

prepared = base.prepared
tables = base.tables
FAST = dict(plan_workers=2, light_workers=3, numba_threads_per_light_worker=1, flood_workers=2)


def run_pipeline(prepared, name, block_patches=1, settings=None, **options):
    settings = settings or PipelineSettings(**FAST)
    return blocks.run_pass_two(**base.parallel_kwargs(prepared, name, block_patches, pipeline=settings, **options))


@pytest.mark.parametrize("label, settings", [
    ("minimal", dict(plan_workers=1, light_workers=1, numba_threads_per_light_worker=1, flood_workers=1)),
    ("one_cell_at_a_time", dict(FAST, cells_in_flight=1, store_budget_bytes=1)),
    ("many_light_workers", dict(FAST, light_workers=8, flood_workers=3)),
    ("tiny_decode_cache", dict(FAST, decode_cache_patches=1)),
])
def test_the_table_is_identical_to_the_block_loop_for_any_settings(prepared, tables, label, settings) -> None:
    report = run_pipeline(prepared, f"pipe_{label}", settings=PipelineSettings(**settings))
    path = prepared.directory / f"pipe_{label}.csv"
    assert path.read_bytes() == tables[1][0].read_bytes()
    sequential = tables[1][1]
    assert report.rows == sequential.rows and report.plan == sequential.plan
    assert report.blocks_total == sequential.blocks_total and report.blocks_resumed == 0
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("block_patches", [2, 3])
def test_the_checkpoint_blocks_may_be_any_size(prepared, tables, block_patches) -> None:
    report = run_pipeline(prepared, f"pipe_block{block_patches}", block_patches)
    path = prepared.directory / f"pipe_block{block_patches}.csv"
    assert path.read_bytes() == tables[1][0].read_bytes()
    assert report.blocks_total == tables[block_patches][1].blocks_total


def test_every_requested_unit_is_lit_once_or_reused(prepared, tables) -> None:
    sequential = tables[1][1].flood
    report = run_pipeline(prepared, "pipe_units")
    flood = report.flood
    assert flood.lit_units_computed + flood.lit_units_reused == sequential.lit_units_computed + sequential.lit_units_reused
    assert 0 < flood.lit_units_computed <= sequential.lit_units_computed
    assert flood.cell_months_flooded == sequential.cell_months_flooded and flood.rows == sequential.rows


def test_a_run_resumes_between_the_pipeline_and_the_block_loop(prepared, tables) -> None:
    path = prepared.directory / "pipe_mixed.csv"
    kwargs = base.parallel_kwargs(prepared, "pipe_mixed")
    assert blocks.run_pass_two(stop_after_blocks=2, pipeline=PipelineSettings(**FAST), **kwargs) is None
    checkpoint = path.with_name(path.name + ".checkpoint.jsonl")
    assert len(checkpoint.read_text().splitlines()) == 3
    report = blocks.run_pass_two(**kwargs)                                  # the block loop finishes it
    assert report.blocks_resumed == 2 and path.read_bytes() == tables[1][0].read_bytes()
    assert blocks.run_pass_two(stop_after_blocks=1, **kwargs) is None
    report = blocks.run_pass_two(pipeline=PipelineSettings(**FAST), **kwargs)      # and the other way
    assert report.blocks_resumed == 1 and path.read_bytes() == tables[1][0].read_bytes()


def test_the_pipeline_needs_one_worker_and_valid_settings(prepared) -> None:
    with pytest.raises(ValueError, match="workers must be 1"):
        blocks.run_pass_two(**base.parallel_kwargs(prepared, "bad_pipe"), workers=2, pipeline=PipelineSettings())
    for name in ("plan_workers", "light_workers", "flood_workers", "decode_cache_patches", "cells_in_flight",
                 "store_budget_bytes", "numba_threads_per_light_worker"):
        for bad in (0, -1, 1.5):
            with pytest.raises(ValueError, match=name):
                PipelineSettings(**{name: bad})


def test_progress_reports_each_block_once(prepared) -> None:
    calls = []
    run_pipeline(prepared, "pipe_progress", progress=lambda done, total: calls.append((done, total)))
    assert calls == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]


def test_a_failure_in_a_stage_is_raised_and_the_finished_blocks_are_kept(prepared, monkeypatch) -> None:
    path = prepared.directory / "pipe_failing.csv"
    kwargs = base.parallel_kwargs(prepared, "pipe_failing", pipeline=PipelineSettings(**FAST))
    real = blocks._flood_record

    def sometimes_fails(store, record, window, patch_window, passable, counts, *, cell, **rest):
        if cell == (1, 1):
            raise RuntimeError("cell (1, 1) is broken")
        return real(store, record, window, patch_window, passable, counts, cell=cell, **rest)

    monkeypatch.setattr(blocks, "_flood_record", sometimes_fails)
    with pytest.raises(RuntimeError, match="cell .1, 1. is broken"):
        blocks.run_pass_two(**kwargs)
    assert multiprocessing.active_children() == [] and not path.exists()
    monkeypatch.setattr(blocks, "_flood_record", real)
    checkpoint = path.with_name(path.name + ".checkpoint.jsonl")
    recorded = [json.loads(line)["block"] for line in checkpoint.read_text().splitlines()[1:]]
    assert [1, 1] not in recorded
    assert blocks.run_pass_two(**kwargs).blocks_resumed == len(recorded)


def test_the_first_interrupt_finishes_the_block_being_fed_and_the_run_resumes(prepared, tables) -> None:
    path = prepared.directory / "pipe_interrupted.csv"
    kwargs = base.parallel_kwargs(prepared, "pipe_interrupted", pipeline=PipelineSettings(**dict(FAST, cells_in_flight=1)))
    sent = []

    def progress(done, total):
        if done == 1 and not sent:
            sent.append(1)
            os.kill(os.getpid(), signal.SIGINT)

    previous = signal.getsignal(signal.SIGINT)
    assert blocks.run_pass_two(progress=progress, **kwargs) is None
    assert signal.getsignal(signal.SIGINT) == previous and multiprocessing.active_children() == []
    checkpoint = path.with_name(path.name + ".checkpoint.jsonl")
    finished = len(checkpoint.read_text().splitlines()) - 1
    assert 1 <= finished < 4 and not path.exists()
    assert blocks.run_pass_two(**kwargs).blocks_resumed == finished
    assert path.read_bytes() == tables[1][0].read_bytes()


# --- the decode cache --------------------------------------------------------------------------------------


def test_the_horizon_cache_evicts_the_least_recently_used_patch_and_counts_loads() -> None:
    loaded = []
    cache = HorizonCache(lambda patch: loaded.append(patch) or object(), capacity=2)
    first = cache.get((0, 0))
    cache.get((0, 1))
    assert cache.get((0, 0)) is first                     # a hit: no load, and (0, 0) is now the newest
    cache.get((0, 2))                                     # evicts (0, 1)
    cache.get((0, 1))                                     # loaded again
    assert loaded == [(0, 0), (0, 1), (0, 2), (0, 1)] and cache.loads == 4 and cache.peak == 2


def test_concurrent_requests_for_one_patch_share_one_decode() -> None:
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow_loader(patch):
        calls.append(patch)
        started.set()
        release.wait(5)
        return object()

    cache = HorizonCache(slow_loader, capacity=4)
    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get((3, 4)))) for _ in range(6)]
    for thread in threads:
        thread.start()
    started.wait(5)
    time.sleep(0.1)
    release.set()
    for thread in threads:
        thread.join(5)
    assert len(calls) == 1 and len({id(r) for r in results}) == 1 and len(results) == 6


def test_a_patch_without_a_tile_is_cached_as_none() -> None:
    cache = HorizonCache(lambda patch: None, capacity=2)
    assert cache.get((0, 0)) is None and cache.get((0, 0)) is None and cache.loads == 1


def test_default_numba_threads_divide_the_cpu(monkeypatch) -> None:
    monkeypatch.setattr(stage3.os, "cpu_count", lambda: 24)
    assert [stage3.default_numba_threads(n) for n in (1, 6, 12, 24, 48)] == [24, 4, 2, 1, 1]
    assert stage3.default_numba_threads(6, 3) == 3


# --- memory: units are freed when the last cell-month that needs them has been flooded -------------------------


def test_units_are_freed_and_the_bound_limits_how_many_are_held(prepared, tables, caplog) -> None:
    import logging
    import re

    total = tables[1][1].flood
    requested = total.lit_units_computed + total.lit_units_reused          # units if nothing were shared or freed

    def run(name, **settings):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=stage3.logger.name):
            run_pipeline(prepared, name, settings=PipelineSettings(**dict(FAST, **settings)))
        line = next(r.getMessage() for r in caplog.records if "Pipeline finished" in r.getMessage())
        return int(re.search(r"(\d+) units lit", line).group(1)), int(re.search(r"peak (\d+) units", line).group(1)), \
            (int(re.search(r"(\d+) units still held", line).group(1)), int(re.search(r"\((\d+) in the store\)", line).group(1)))

    lit_one, peak_one, held_one = run("pipe_memory_one", cells_in_flight=1, store_budget_bytes=1)
    lit_many, peak_many, held_many = run("pipe_memory_many", cells_in_flight=24)
    assert held_one == held_many == (0, 0)                               # the table and the store are empty at the end
    assert peak_one < peak_many <= requested                      # one cell at a time holds the least
    assert peak_one < requested / 2
    assert lit_one >= lit_many > 0                                       # less sharing when fewer cells overlap


def test_the_status_line_reports_where_work_waits(prepared, caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger=stage3.logger.name):
        run_pipeline(prepared, "pipe_status", settings=PipelineSettings(**dict(FAST, status_seconds=0.05)))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("pipeline:")]
    assert lines and "cells in flight" in lines[0] and "waiting to be lit" in lines[0] and "jobs ready to flood" in lines[0]
    with pytest.raises(ValueError, match="status_seconds"):
        PipelineSettings(status_seconds=-1.0)


# --- the back pressure, on the coordinator itself -----------------------------------------------------------


def make_coordinator(*, cells_in_flight, budget_bytes, cells=((0, 0), (0, 1), (0, 2))):
    import numpy as np

    from lunarscout._numba_horizon.safe_haven_litstore import LitStore
    from lunarscout._numba_horizon.safe_haven_windows import FloodGrid

    finished = []
    store = LitStore(FloodGrid(30), (128, 128 * 3))
    settings = PipelineSettings(cells_in_flight=cells_in_flight, store_budget_bytes=budget_bytes)
    coordinator = stage3._Coordinator(
        store=store, settings=settings, by_block={(0, 0): list(cells)},
        finish=lambda *args: finished.append(args))
    return coordinator, finished


def plan_result(cell, chunks=(0, 1), patch_column=0):
    import numpy as np

    record = {"month": 0, "chunks": list(chunks), "seeds": []}
    return cell, (0, 128 * patch_column, 128, 128 * (patch_column + 1)), (0, patch_column, 1, 1), \
        [(record, np.zeros((128, 2), dtype=np.uint64))], {}


def acquire_in_thread(coordinator):
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(coordinator.acquire_cell_slot()), daemon=True)
    thread.start()
    return thread, outcome


def test_the_unit_budget_holds_the_next_cell_back_until_units_are_freed() -> None:
    import numpy as np

    from lunarscout._numba_horizon.safe_haven_blocks import PassTwoCounts

    coordinator, finished = make_coordinator(cells_in_flight=3, budget_bytes=1)
    assert coordinator.acquire_cell_slot()              # nothing in flight yet: it proceeds, whatever the budget
    coordinator.add_cell(plan_result((0, 0)))
    assert coordinator.requested_bytes == 2 * coordinator.unit_bytes >= 1
    thread, outcome = acquire_in_thread(coordinator)
    thread.join(0.6)
    assert thread.is_alive() and outcome == []          # blocked by the budget, although a slot is free
    coordinator.publish((0, 0), {0: np.zeros(coordinator.store.chunk_shape, np.uint64),
                                 1: np.zeros(coordinator.store.chunk_shape, np.uint64)})
    job = coordinator.ready_jobs.get(timeout=2)
    coordinator.job_done(job, [], PassTwoCounts())
    thread.join(5)
    assert outcome == [True] and coordinator.requested_bytes == 0 and len(coordinator.units) == 0
    assert len(coordinator.store) == 0


def test_the_cell_bound_holds_the_next_cell_back_until_a_cell_is_done() -> None:
    import numpy as np

    from lunarscout._numba_horizon.safe_haven_blocks import PassTwoCounts

    coordinator, finished = make_coordinator(cells_in_flight=1, budget_bytes=10**12)
    assert coordinator.acquire_cell_slot()
    thread, outcome = acquire_in_thread(coordinator)
    thread.join(0.6)
    assert thread.is_alive()                            # the one slot is taken
    coordinator.add_cell(plan_result((0, 0), chunks=(3,)))
    coordinator.publish((0, 0), {3: np.zeros(coordinator.store.chunk_shape, np.uint64)})
    coordinator.job_done(coordinator.ready_jobs.get(timeout=2), [], PassTwoCounts())
    thread.join(5)
    assert outcome == [True]


def test_stopping_releases_a_feeder_waiting_for_a_slot() -> None:
    coordinator, _finished = make_coordinator(cells_in_flight=1, budget_bytes=10**12)
    assert coordinator.acquire_cell_slot()
    thread, outcome = acquire_in_thread(coordinator)
    thread.join(0.4)
    coordinator.fail(RuntimeError("stop"))
    thread.join(5)
    assert outcome == [False] and not coordinator.acquire_cell_slot()
    assert isinstance(coordinator.error, RuntimeError) and coordinator.all_done.is_set()


def test_a_shared_unit_is_lit_once_and_freed_after_its_last_user() -> None:
    import numpy as np

    from lunarscout._numba_horizon.safe_haven_blocks import PassTwoCounts

    coordinator, _finished = make_coordinator(cells_in_flight=3, budget_bytes=10**12)
    assert coordinator.acquire_cell_slot() and coordinator.acquire_cell_slot()
    coordinator.add_cell(plan_result((0, 0), chunks=(5,)))
    coordinator.add_cell(plan_result((0, 1), chunks=(5, 6)))        # shares unit (0, 0, 5)
    assert coordinator.units_computed == 0 and sorted(coordinator.pending[(0, 0)]) == [5, 6]
    task = coordinator.take_lighting_task()
    assert task == ((0, 0), [5, 6])
    bits = {chunk: np.zeros(coordinator.store.chunk_shape, np.uint64) for chunk in (5, 6)}
    coordinator.publish((0, 0), bits)
    jobs = [coordinator.ready_jobs.get(timeout=2) for _ in range(2)]
    assert coordinator.units[(0, 0, 5)].refs == 2 and coordinator.units[(0, 0, 6)].refs == 1
    coordinator.job_done(next(j for j in jobs if j.cell == (0, 1)), [], PassTwoCounts())
    assert (0, 0, 5) in coordinator.units and (0, 0, 6) not in coordinator.units       # 5 is still needed
    coordinator.job_done(next(j for j in jobs if j.cell == (0, 0)), [], PassTwoCounts())
    assert coordinator.units == {} and len(coordinator.store) == 0 and coordinator.requested_bytes == 0
