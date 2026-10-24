"""The per-month lit-bit store, CPU reference route (spec section 8)."""

from __future__ import annotations

import numpy as np
import pytest

import windows_scene as scene
from lunarscout._numba_horizon import safe_haven_flood as oracle
from lunarscout._numba_horizon import safe_haven_flood_packed as packed
from lunarscout._numba_horizon import safe_haven_litstore as litstore
from lunarscout._numba_horizon import safe_haven_windows as windows
from lunarscout._numba_horizon.file_format import PATCH_SIZE, HorizonTileStore
from lunarscout._numba_horizon.safe_haven_litstore import LitStore, LitStoreError
from lunarscout._numba_horizon.safe_haven_windows import FloodGrid, Stage1Rasters, WindowParameters
from lunarscout.products import _load_dem

GRID = FloodGrid(30)
SPC = GRID.steps_per_chunk
PARAMETERS = WindowParameters(
    cell_patches=1, rover_speed_m_s=1.0 / 1800.0, haven_neighbourhood_m=3.0, seed_separation_m=4.0,
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("litstore_scene")
    dem_path, horizons_path, products, stamps = scene.stage_one(directory)
    dem, _georef = _load_dem(dem_path)
    store = HorizonTileStore(horizons_path)
    stage1 = Stage1Rasters(products.hibernation, products.joint_total)
    records = list(windows.iter_cell_months(
        dem=dem, horizon_store=store, stage1=stage1, times=stamps, sun_vectors_at=scene.sun_vectors,
        earth_vectors_at=scene.earth_vectors, parameters=PARAMETERS,
    ))
    return dict(dem=dem, store=store, stamps=stamps, stage1=stage1, records=records,
                signals=windows.PixelSignals(dem, store), directory=directory)


def filled(prepared, chunks, **options):
    store = LitStore(GRID, prepared["dem"].elevation_m.shape)
    counts = litstore.fill_lit_store(
        store, dem=prepared["dem"], horizon_store=prepared["store"], patches=[(0, 0)],
        chunks=chunks, sun_vectors_at=scene.sun_vectors, **options,
    )
    return store, counts


def a_seed(prepared, month=1):
    record = next(r for r in prepared["records"] if r["month"] == month)
    return record["seeds"][0]


# --- the bits equal the per-pixel reference --------------------------------------------------------


def test_assembled_bits_equal_the_per_pixel_lit_state_across_chunk_edges(prepared) -> None:
    seed = a_seed(prepared)
    chunk = seed["boundary_after_step"] // SPC
    store, counts = filled(prepared, [chunk, chunk + 1, chunk + 2])
    assert (counts.computed, counts.reused, counts.patches_read) == (3, 0, 1)
    first, last = chunk * SPC + SPC - 7, (chunk + 2) * SPC + 11        # spans two chunk edges
    bits = store.assemble(0, 0, 1, 1, first, last)
    assert bits.shape == (last - first + 1, PATCH_SIZE, 2)
    lit = packed.unpack_bits(bits, PATCH_SIZE)
    cube = oracle.lit_reference_cube(prepared["signals"], scene.sun_vectors, GRID, (0, 0), (scene.SIZE, scene.SIZE), first, last)
    assert np.array_equal(lit[:, :scene.SIZE, :scene.SIZE], cube)
    assert not lit[:, scene.SIZE:, :].any() and not lit[:, :, scene.SIZE:].any()
    assert cube.any() and not cube.all()                       # the span really has a transition


def test_pixel_lit_reads_the_same_bits(prepared) -> None:
    seed = a_seed(prepared)
    chunk = seed["boundary_after_step"] // SPC
    store, _ = filled(prepared, [chunk])
    bits = packed.unpack_bits(store.assemble(0, 0, 1, 1, chunk * SPC, chunk * SPC + SPC - 1), PATCH_SIZE)
    for step in range(0, SPC, 7):
        for row, column in [(0, 0), (4, 4), (11, 11), (19, 19), (20, 20), (127, 127)]:
            assert store.pixel_lit(row, column, chunk * SPC + step) == bits[step, row, column]
    assert store.pixel_lit(-1, 0, chunk * SPC) is False


def test_the_operating_fraction_sets_the_threshold_and_only_the_valid_area_is_lit(prepared) -> None:
    chunk = a_seed(prepared)["boundary_after_step"] // SPC       # a day with a sunrise in it
    everything, _ = filled(prepared, [chunk], operating_sunlight_fraction=0.0)
    lit = packed.unpack_bits(everything.assemble(0, 0, 1, 1, chunk * SPC, (chunk + 1) * SPC - 1), PATCH_SIZE)
    assert lit[:, :scene.SIZE, :scene.SIZE].all() and lit.sum() == SPC * scene.SIZE * scene.SIZE
    normal, _ = filled(prepared, [chunk])
    assert 0 < packed.unpack_bits(
        normal.assemble(0, 0, 1, 1, chunk * SPC, (chunk + 1) * SPC - 1), PATCH_SIZE
    ).sum() < lit.sum()


# --- the store itself ------------------------------------------------------------------------------


def test_a_unit_is_a_day_of_packed_bits() -> None:
    store = LitStore(FloodGrid(32), (20, 20))
    assert store.chunk_shape == (45, 128, 2)
    store.put(0, 0, 0, np.zeros(store.chunk_shape, dtype=np.uint64))
    assert store.nbytes == 45 * 128 * 2 * 8 == 92160           # 90 KiB, spec section 8
    assert len(store) == 1
    with pytest.raises(ValueError):
        store.put(0, 0, 1, np.zeros((45, 128, 1), dtype=np.uint64))
    store.clear()
    assert len(store) == 0 and store.nbytes == 0


def test_units_already_stored_are_reused_without_reading_horizons(prepared) -> None:
    store, first = filled(prepared, [5, 6])
    again = litstore.fill_lit_store(
        store, dem=prepared["dem"], horizon_store=prepared["store"], patches=[(0, 0)], chunks=[5, 6, 7],
        sun_vectors_at=scene.sun_vectors,
    )
    assert (again.computed, again.reused, again.patches_read) == (1, 2, 1)
    nothing = litstore.fill_lit_store(
        store, dem=prepared["dem"], horizon_store=prepared["store"], patches=[(0, 0)], chunks=[5, 6, 7],
        sun_vectors_at=scene.sun_vectors,
    )
    assert (nothing.computed, nothing.reused, nothing.patches_read) == (0, 3, 0)


def test_a_missing_unit_is_an_error_and_a_patch_outside_the_dem_is_dark(prepared) -> None:
    store, _ = filled(prepared, [5])
    with pytest.raises(LitStoreError):
        store.assemble(0, 0, 1, 1, 5 * SPC, 6 * SPC)           # chunk 6 was never lit
    with pytest.raises(LitStoreError):
        store.pixel_lit(0, 0, 6 * SPC)
    window = store.assemble(-1, -1, 2, 2, 5 * SPC, 5 * SPC + 3)
    assert window.shape == (4, 2 * PATCH_SIZE, 4)
    assert not window[:, :PATCH_SIZE, :].any() and not window[:, :, :2].any()
    assert store.has(-3, 7, 99) and not store.has(0, 0, 6)
    # the DEM is 20 x 20, so patch (0, 1) is outside it too
    assert store.outside(0, 1) and not store.outside(0, 0)


def test_a_patch_without_horizons_is_stored_unlit(prepared, tmp_path) -> None:
    store = LitStore(GRID, prepared["dem"].elevation_m.shape)
    counts = litstore.fill_lit_store(
        store, dem=prepared["dem"], horizon_store=HorizonTileStore(tmp_path / "none"), patches=[(0, 0)],
        chunks=[2, 3], sun_vectors_at=scene.sun_vectors,
    )
    assert (counts.computed, counts.patches_without_horizons) == (2, 1)
    assert not store.assemble(0, 0, 1, 1, 2 * SPC, 4 * SPC - 1).any()


@pytest.mark.parametrize("call", [
    lambda: LitStore(GRID, (20, 20)).assemble(0, 0, 1, 1, 10, 5),
    lambda: LitStore(GRID, (20, 20)).side_lit((0, 0, 1, 1), 10, 0, 3),
])
def test_invalid_arguments_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_side_lit_is_ordered_along_the_flood_in_each_direction() -> None:
    store = LitStore(GRID, (PATCH_SIZE, PATCH_SIZE))
    for chunk in (0, 1):
        bits = np.zeros(store.chunk_shape, dtype=np.uint64)
        for step in range(SPC):
            bits[step, 0, 0] = np.uint64(chunk * SPC + step + 1)     # the absolute step, plus one
        store.put(0, 0, chunk, bits)
    boundary = SPC                                                   # the first step of chunk 1
    after = store.side_lit((0, 0, 1, 1), boundary, oracle.AFTER, 4)
    before = store.side_lit((0, 0, 1, 1), boundary, oracle.BEFORE, 4)
    assert [int(after[j, 0, 0]) for j in range(4)] == [boundary + j + 1 + 1 for j in range(4)]
    assert [int(before[j, 0, 0]) for j in range(4)] == [boundary - j - 1 + 1 for j in range(4)]
    assert store.side_lit((0, 0, 1, 1), boundary, oracle.AFTER, 0).shape[0] == 0


# --- the flood from the store ----------------------------------------------------------------------


def window_inputs(prepared, month):
    """The passable mask of a (3 x 3)-patch window whose patch (1, 1) is the DEM's patch (0, 0)."""
    hibernation = prepared["stage1"].read_hibernation_window(0, scene.SIZE, 0, scene.SIZE)[month]
    window = np.zeros((3 * PATCH_SIZE, 3 * PATCH_SIZE), dtype=bool)
    window[PATCH_SIZE:PATCH_SIZE + scene.SIZE, PATCH_SIZE:PATCH_SIZE + scene.SIZE] = np.isfinite(hibernation)
    return np.isfinite(hibernation), packed.pack_bits(window)


@pytest.mark.parametrize("steps", [5, 12, 96])
def test_the_flood_from_the_store_equals_the_oracle_on_the_scene(prepared, steps) -> None:
    window = (-1, -1, 3, 3)           # a halo of patches around the one patch, partly outside the DEM
    checked = 0
    for record in prepared["records"]:
        passable_small, passable_window = window_inputs(prepared, record["month"])
        for seed in record["seeds"]:
            first = seed["boundary_before_step"] - steps
            last = seed["boundary_after_step"] + steps
            store, _ = filled(prepared, range(first // SPC, last // SPC + 1))
            cube = oracle.lit_reference_cube(
                prepared["signals"], scene.sun_vectors, GRID, (0, 0), (scene.SIZE, scene.SIZE), first, last,
            )
            lit_at = lambda step, cube=cube, first=first: cube[step - first]
            position = (seed["row"], seed["column"])
            expected = oracle.flood_seed_reference(
                lit_at, passable_small, position, seed["boundary_before_step"], seed["boundary_after_step"],
                steps=steps,
            )
            got = litstore.flood_seed_from_store(
                store, passable_window, window, (PATCH_SIZE + position[0], PATCH_SIZE + position[1]),
                seed["boundary_before_step"], seed["boundary_after_step"], steps=steps,
            )
            assert got == expected
            checked += 1
    assert checked >= 10


def test_a_seed_unlit_at_its_boundary_moves_one_step_or_is_a_mismatch(prepared) -> None:
    seed = a_seed(prepared)
    steps = 6
    window = (0, 0, 1, 1)
    row, column = seed["row"], seed["column"]
    before, after = seed["boundary_before_step"], seed["boundary_after_step"]
    first, last = before - steps - 1, after + steps + 1
    store, _ = filled(prepared, range(first // SPC, last // SPC + 1))
    passable_small, _ = window_inputs(prepared, 1)
    passable = packed.pack_bits(np.pad(passable_small, ((0, PATCH_SIZE - scene.SIZE),) * 2))
    args = dict(steps=steps)
    base = litstore.flood_seed_from_store(store, passable, window, (row, column), before, after, **args)
    assert base is not None

    def clear(step):
        chunk = store._chunks[(0, 0, step // SPC)]
        chunk[step % SPC, row, column // 64] &= ~np.uint64(1 << (column % 64))

    clear(before)                                   # unlit at the before boundary: use the step before it
    shifted = litstore.flood_seed_from_store(store, passable, window, (row, column), before, after, **args)
    expected = litstore.flood_seed_from_store(store, passable, window, (row, column), before - 1, after, **args)
    assert shifted is not None and shifted.count_before == expected.count_before
    assert shifted.count_after == base.count_after
    clear(before - 1)                               # both unlit: a mismatch
    assert litstore.flood_seed_from_store(store, passable, window, (row, column), before, after, **args) is None
    clear(after)
    clear(after + 1)
    assert litstore.flood_seed_from_store(store, passable, window, (row, column), before, after, **args) is None


# --- the fused threshold-and-pack lighting, the unit discard and the valid-pixel mask ------------------


def test_fused_lighting_is_bit_identical_to_threshold_then_pack(prepared) -> None:
    from lunarscout._numba_horizon.file_format import read_patch_horizons
    from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession

    chunk = a_seed(prepared)["boundary_after_step"] // SPC
    horizons = read_patch_horizons(prepared["store"], 0, 0, 0.0)
    session = LightmapCpuSession()
    for fraction in (1.0, 0.99, 0.5, 0.0):
        expected = litstore.light_patch_chunks(
            session, prepared["dem"], horizons, 0, 0, GRID, [chunk, chunk + 1], scene.sun_vectors, fraction)
        for compiled in (True, False):
            got = litstore.light_patch_chunks_fused(
                session, prepared["dem"], horizons, 0, 0, GRID, [chunk + 1, chunk], scene.sun_vectors, fraction,
                compiled=compiled)
            assert sorted(got) == sorted(expected) == [chunk, chunk + 1]
            for key in expected:
                assert np.array_equal(got[key], expected[key]), (fraction, compiled, key)
    assert any(got[key].any() for key in got) and not all(got[key].all() for key in got)


def test_the_threshold_kernel_handles_the_equality_fuzz_and_partial_patches() -> None:
    fractions = np.zeros((2, 3, 70), dtype=np.float32)
    fractions[0, 0, 0] = 1.0
    fractions[0, 0, 63] = np.float32(1.0 - 5e-7)      # within the 1e-6 fuzz of 1.0: lit
    fractions[0, 0, 64] = np.float32(1.0 - 2e-6)      # outside it: dark
    fractions[1, 2, 69] = 1.0
    for compiled in (True, False):
        out = np.zeros((2, 128, 2), dtype=np.uint64)
        litstore._threshold_and_pack(compiled)(fractions, 3, 70, 1.0 - 1e-6, out)
        assert int(out[0, 0, 0]) == (1 | (1 << 63)) and int(out[0, 0, 1]) == 0
        assert int(out[1, 2, 1]) == 1 << 5 and out[:, 3:].sum() == 0 and out[0, 1:].sum() == 0


def test_a_unit_can_be_discarded_and_discarding_twice_is_harmless() -> None:
    store = LitStore(GRID, (300, 300))
    store.put(0, 0, 4, np.zeros(store.chunk_shape, dtype=np.uint64))
    assert store.has(0, 0, 4) and len(store) == 1
    store.discard(0, 0, 4)
    store.discard(0, 0, 4)
    assert not store.has(0, 0, 4) and len(store) == 0


def test_the_valid_pixel_mask_equals_isfinite_of_the_decoded_product(prepared) -> None:
    stage1 = prepared["stage1"]
    window = (0, scene.SIZE, 0, scene.SIZE)
    assert np.array_equal(stage1.read_valid_window(*window), np.isfinite(stage1.read_hibernation_window(*window)))
