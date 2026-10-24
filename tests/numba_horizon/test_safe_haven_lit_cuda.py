"""CUDA lighting of stage-3 units, bit for bit against the CPU route.

These tests use a real GPU: set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1. Every kernel they launch is the
production lightmap kernel or the fixed-trip-count packing kernel of ``safe_haven_lit_cuda``."""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

import test_safe_haven_blocks as base
import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_flood_packed as packed
from lunarscout._numba_horizon import safe_haven_litstore as litstore
from lunarscout._numba_horizon import safe_haven_stage3 as stage3
from lunarscout._numba_horizon.file_format import PATCH_SIZE, read_patch_horizons
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.safe_haven_stage3 import PipelineSettings
from lunarscout._numba_horizon.safe_haven_windows import FloodGrid

pytestmark = pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the real-GPU lighting tests",
)

prepared = base.prepared
tables = base.tables
GRID = FloodGrid(30)
SPC = GRID.steps_per_chunk


@pytest.fixture(scope="module")
def session():
    from lunarscout._numba_horizon.safe_haven_lit_cuda import LitCudaSession

    return LitCudaSession(steps_per_chunk=SPC, chunks_per_batch=4)


def cpu_units(prepared, patch, chunks, threshold):
    horizons = read_patch_horizons(prepared.store, patch[0] * PATCH_SIZE, patch[1] * PATCH_SIZE, 0.0)
    return litstore.light_patch_chunks_fused(
        LightmapCpuSession(), prepared.dem, horizons, patch[0], patch[1], GRID, chunks, scene.sun_vectors, threshold)


def gpu_units(session, prepared, patch, chunks, threshold):
    horizons = read_patch_horizons(prepared.store, patch[0] * PATCH_SIZE, patch[1] * PATCH_SIZE, 0.0)
    tile_y, tile_x = patch[0] * PATCH_SIZE, patch[1] * PATCH_SIZE
    vectors = [scene.sun_vectors(GRID.times(np.arange(c * SPC, (c + 1) * SPC))) for c in chunks]
    units = session.light_patch(
        prepared.dem, horizons, vectors, tile_y=tile_y, tile_x=tile_x,
        valid_height=min(PATCH_SIZE, prepared.dem.elevation_m.shape[0] - tile_y),
        valid_width=min(PATCH_SIZE, prepared.dem.elevation_m.shape[1] - tile_x),
        threshold=threshold - 1e-6)
    return dict(zip(chunks, units))


def first_seed_chunk(prepared):
    """The day containing the scene's first outage (mid January), where the Sun comes and goes."""
    return int(np.datetime64("2027-01-15T12:00", "us").astype(np.int64)) // GRID.step_us // SPC


@pytest.mark.parametrize("threshold", [1.0, 0.99, 0.5, 0.0])
@pytest.mark.parametrize("chunk_count", [1, 4, 5, 9])         # a multiple of the batch size, and remainders
def test_gpu_units_are_bit_identical_to_the_cpu_route(prepared, session, threshold, chunk_count) -> None:
    first = first_seed_chunk(prepared)
    chunks = list(range(first, first + chunk_count))
    for patch in [(0, 0), (1, 1), (2, 2), (2, 1)]:                # includes the partial last row and column
        cpu = cpu_units(prepared, patch, chunks, threshold)
        gpu = gpu_units(session, prepared, patch, chunks, threshold)
        assert sorted(cpu) == sorted(gpu)
        for chunk in chunks:
            assert np.array_equal(cpu[chunk], gpu[chunk]), (patch, threshold, chunk)
    assert any(unit.any() for unit in gpu.values())


def test_the_packing_kernel_matches_numpy_on_ties_nan_and_partial_patches(session) -> None:
    rng = np.random.default_rng(7)
    steps = 7
    fractions = rng.random((steps, PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
    fractions[3, 0] = 0.0                                      # exact ties with the thresholds 0.5 and 0.0 below
    fractions[3, 1] = 0.5
    fractions[3, 64:70] = 0.5
    fractions[0, 5] = np.float32(1.0 - 5e-7)                   # inside the fuzz of 1.0 (lit)
    fractions[0, 6] = np.float32(1.0 - 2e-6)                   # outside it (dark)
    fractions[1, :64] = np.nan
    fractions[2, 130:200] = 1.0
    cuda = session._cuda
    for valid_height, valid_width in [(128, 128), (44, 128), (128, 44), (3, 70), (1, 1)]:
        for threshold in (1.0 - 1e-6, 0.5, 0.0, -1e-6):        # 0.5 and 0.0 are exact ties; the last lights every unmasked pixel
            device = cuda.to_device(fractions)
            words = cuda.device_array((steps, PATCH_SIZE, 2), dtype=np.uint64)
            total = steps * PATCH_SIZE * 2
            session._pack[(total + 127) // 128, 128](device, steps, valid_width, valid_height, np.float64(threshold), words)
            cuda.synchronize()
            expected = np.zeros((steps, PATCH_SIZE, 2), dtype=np.uint64)
            litstore.threshold_and_pack_python(
                fractions.reshape(steps, PATCH_SIZE, PATCH_SIZE)[:, :valid_height, :valid_width],
                valid_height, valid_width, threshold, expected)
            assert np.array_equal(words.copy_to_host(), expected), (valid_height, valid_width, threshold)


def test_the_session_refuses_bad_arguments(prepared, session) -> None:
    from lunarscout._numba_horizon.safe_haven_lit_cuda import LitCudaSession

    horizons = read_patch_horizons(prepared.store, 0, 0, 0.0)
    good = [scene.sun_vectors(GRID.times(np.arange(SPC)))]
    common = dict(tile_y=0, tile_x=0, valid_height=128, valid_width=128, threshold=0.5)
    assert session.light_patch(prepared.dem, horizons, [], **common) == []
    with pytest.raises(ValueError, match="valid patch dimensions"):
        session.light_patch(prepared.dem, horizons, good, **dict(common, valid_width=129))
    with pytest.raises(ValueError, match="Sun vectors"):
        session.light_patch(prepared.dem, horizons, [good[0][:-1]], **common)
    with pytest.raises(ValueError, match="horizons"):
        session.light_patch(prepared.dem, horizons[:, :, :100], good, **common)
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            LitCudaSession(steps_per_chunk=bad)
        with pytest.raises(ValueError):
            LitCudaSession(chunks_per_batch=bad)


def test_threads_sharing_one_session_get_each_their_own_correct_units(prepared, session) -> None:
    first = first_seed_chunk(prepared)
    chunks = list(range(first, first + 6))
    patches = [(0, 0), (1, 1), (2, 2), (2, 1), (1, 0), (0, 1), (1, 2), (2, 0)]
    results = {}

    def work(patch):
        results[patch] = gpu_units(session, prepared, patch, chunks, 1.0)

    for _ in range(3):
        threads = [threading.Thread(target=work, args=(patch,)) for patch in patches]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for patch in patches:
            cpu = cpu_units(prepared, patch, chunks, 1.0)
            for chunk in chunks:
                assert np.array_equal(results[patch][chunk], cpu[chunk]), (patch, chunk)


@pytest.mark.parametrize("workers, batch", [(1, 1), (4, 4), (3, 7)])
def test_the_pipeline_on_the_gpu_gives_the_same_table_as_the_cpu(prepared, tables, workers, batch) -> None:
    settings = PipelineSettings(plan_workers=2, flood_workers=2, lighting="cuda", gpu_light_workers=workers,
                                gpu_chunks_per_batch=batch)
    report = base.blocks.run_pass_two(**base.parallel_kwargs(prepared, f"pipe_cuda_{workers}_{batch}", 1, pipeline=settings))
    assert (prepared.directory / f"pipe_cuda_{workers}_{batch}.csv").read_bytes() == tables[1][0].read_bytes()
    assert report.rows == tables[1][1].rows


def test_auto_picks_the_gpu_when_there_is_one(prepared, tables) -> None:
    settings = PipelineSettings(plan_workers=2, flood_workers=2, lighting="auto", gpu_light_workers=2)
    assert stage3.select_lighting(settings, SPC) is not None
    base.blocks.run_pass_two(**base.parallel_kwargs(prepared, "pipe_cuda_auto", 1, pipeline=settings))
    assert (prepared.directory / "pipe_cuda_auto.csv").read_bytes() == tables[1][0].read_bytes()
