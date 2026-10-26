"""The packed-bit flood against the plain-Python oracle (spec section 6)."""

from __future__ import annotations

import numpy as np
import pytest

from lunarscout._numba_horizon import safe_haven_flood as oracle
from lunarscout._numba_horizon import safe_haven_flood_packed as packed


def bits_to_set(bits, columns):
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(packed.unpack_bits(bits, columns)))}


def packed_side(lit_at, passable, seed, boundary, direction, steps, connectivity, compiled=True, moves=1):
    count_shape = passable.shape
    lit = np.stack([np.asarray(lit_at(boundary + direction * j), dtype=bool) for j in range(1, steps + 1)]
                   or [np.zeros(count_shape, dtype=bool)])
    count, bits = packed.flood_side_packed(
        packed.pack_bits(lit), packed.pack_bits(passable), seed,
        steps=steps, connectivity=connectivity, moves_per_step=moves, compiled=compiled,
    )
    return count, bits_to_set(bits, passable.shape[1])


# --- packing ----------------------------------------------------------------------------------


@pytest.mark.parametrize("columns", [1, 63, 64, 65, 130])
def test_packing_round_trips_and_numbers_columns_from_the_low_bit(columns) -> None:
    rng = np.random.default_rng(columns)
    mask = rng.random((3, 4, columns)) < 0.5
    words = packed.pack_bits(mask)
    assert words.shape == (3, 4, -(-columns // 64)) and words.dtype == np.uint64
    assert np.array_equal(packed.unpack_bits(words, columns), mask)
    single = np.zeros((1, columns), dtype=bool)
    single[0, columns - 1] = True
    word = packed.pack_bits(single)
    assert int(word[0, (columns - 1) // 64]) == 1 << ((columns - 1) % 64)


# --- parity with the oracle -------------------------------------------------------------------


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("compiled", [True, False])
def test_the_packed_flood_equals_the_oracle_on_random_inputs(connectivity, compiled) -> None:
    rng = np.random.default_rng(7000 + connectivity)
    cases = 25 if compiled else 4
    for _ in range(cases):
        # widths straddle the 64-bit word boundaries; seeds are often at edges and word seams
        shape = (int(rng.integers(1, 12)), int(rng.choice([5, 63, 64, 65, 70, 129])))
        passable = rng.random(shape) < 0.9
        seed = (int(rng.integers(shape[0])), int(rng.choice([0, 1, 31, 62, 63, 64, 65, shape[1] - 1, 2])) % shape[1])
        passable[seed] = True
        steps = int(rng.integers(0, 8))
        cube = rng.random((60, *shape)) < float(rng.uniform(0.4, 1.0))
        lit_at = lambda step: cube[step]
        for direction in (oracle.BEFORE, oracle.AFTER):
            expected = oracle.flood_side_reference(
                lit_at, passable, seed, 30, direction=direction, steps=steps, connectivity=connectivity,
            )
            count, reached = packed_side(lit_at, passable, seed, 30, direction, steps, connectivity, compiled)
            assert reached == expected and count == len(expected)


def test_a_flood_that_crosses_word_seams_and_the_window_edge_matches() -> None:
    shape = (40, 200)
    passable = np.ones(shape, dtype=bool)
    passable[:, 100] = False
    passable[20, 100] = True       # one gap in a wall across a word
    lit_at = lambda step: np.ones(shape, dtype=bool)
    for seed in [(20, 63), (20, 64), (0, 0), (39, 199), (20, 96)]:
        for connectivity in (4, 8):
            expected = oracle.flood_side_reference(
                lit_at, passable, seed, 0, direction=oracle.AFTER, steps=12, connectivity=connectivity,
            )
            count, reached = packed_side(lit_at, passable, seed, 0, oracle.AFTER, 12, connectivity)
            assert reached == expected and count == len(expected)


def test_the_seed_pair_wrapper_returns_the_oracle_result() -> None:
    rng = np.random.default_rng(5)
    shape = (9, 70)
    passable = rng.random(shape) < 0.95
    passable[4, 33] = True
    cube = rng.random((80, *shape)) < 0.8
    lit_at = lambda step: cube[step]
    args = (lit_at, passable, (4, 33), 30, 50)
    assert packed.flood_seed_packed(*args, steps=6) == oracle.flood_seed_reference(*args, steps=6)
    assert packed.flood_seed_packed(*args, steps=0) == oracle.flood_seed_reference(*args, steps=0)


def test_an_emptied_frontier_gives_zero_bits_and_no_revival() -> None:
    passable = np.ones((3, 70), dtype=bool)

    def lit_at(step):
        return np.full(passable.shape, step != 99)

    count, reached = packed_side(lit_at, passable, (1, 5), 100, oracle.BEFORE, 4, 8)
    assert count == 0 and reached == set()


def test_unused_bits_beyond_the_width_never_flood() -> None:
    passable = np.ones((1, 5), dtype=bool)      # 59 spare bits in the word
    count, reached = packed_side(lambda s: np.ones((1, 5), bool), passable, (0, 4), 0, oracle.AFTER, 20, 8)
    assert count == 5 and reached == {(0, c) for c in range(5)}


@pytest.mark.parametrize("call", [
    lambda: packed.flood_side_packed(np.zeros((1, 2, 1), np.uint64), np.zeros((2, 1), np.uint64), (0, 0), steps=1, connectivity=6),
    lambda: packed.flood_side_packed(np.zeros((1, 2, 1), np.uint64), np.zeros((2, 1), np.uint64), (0, 0), steps=-1),
    lambda: packed.flood_side_packed(np.zeros((1, 2, 1), np.uint64), np.zeros((2, 1), np.uint64), (5, 0), steps=1),
    lambda: packed.flood_side_packed(np.zeros((1, 3, 1), np.uint64), np.zeros((2, 1), np.uint64), (0, 0), steps=1),
    lambda: packed.flood_side_packed(np.zeros((1, 2, 1), np.uint64), np.zeros((2, 1), np.uint64), (0, 0), steps=2),
])
def test_invalid_arguments_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


# --- several moves in a step ------------------------------------------------------------------


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("compiled", [True, False])
def test_the_packed_flood_equals_the_oracle_with_several_moves_per_step(connectivity, compiled) -> None:
    rng = np.random.default_rng(9100 + connectivity)
    for _ in range(25 if compiled else 4):
        shape = (int(rng.integers(1, 14)), int(rng.choice([5, 63, 64, 65, 70, 129])))
        passable = rng.random(shape) < 0.92
        seed = (int(rng.integers(shape[0])), int(rng.choice([0, 1, 31, 62, 63, 64, 65, shape[1] - 1, 2])) % shape[1])
        passable[seed] = True
        steps, moves = int(rng.integers(0, 5)), int(rng.integers(1, 6))
        cube = rng.random((60, *shape)) < float(rng.uniform(0.5, 1.0))
        lit_at = lambda step: cube[step]
        for direction in (oracle.BEFORE, oracle.AFTER):
            expected = oracle.flood_side_reference(
                lit_at, passable, seed, 30, direction=direction, steps=steps, connectivity=connectivity,
                moves_per_step=moves)
            count, reached = packed_side(lit_at, passable, seed, 30, direction, steps, connectivity, compiled, moves)
            assert reached == expected and count == len(expected)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_moves_in_a_step_equal_single_moves_in_steps_that_repeat_the_lighting(connectivity) -> None:
    """Lighting is constant over a step, so m moves in step j are m single-move steps with the same mask."""
    rng = np.random.default_rng(31)
    shape, steps, moves = (30, 70), 4, 3
    passable = rng.random(shape) < 0.93
    passable[15, 30] = True
    cube = rng.random((20, *shape)) < 0.85

    def lit_stepwise(step):                      # flood step j = |step - 10|
        return cube[abs(step - 10)]

    def lit_repeated(step):                      # single-move step k = |step - 10| belongs to flood step (k - 1) // moves + 1
        return cube[(abs(step - 10) - 1) // moves + 1]

    with_moves = oracle.flood_side_reference(lit_stepwise, passable, (15, 30), 10, direction=oracle.AFTER,
                                             steps=steps, connectivity=connectivity, moves_per_step=moves)
    single = oracle.flood_side_reference(lit_repeated, passable, (15, 30), 10, direction=oracle.AFTER,
                                         steps=steps * moves, connectivity=connectivity)
    assert with_moves == single and len(single) > 1
    assert packed_side(lit_stepwise, passable, (15, 30), 10, oracle.AFTER, steps, connectivity, True, moves)[1] == single


def test_a_lit_open_window_is_reached_to_steps_times_moves_pixels() -> None:
    shape = (60, 60)
    passable = np.ones(shape, dtype=bool)
    lit_at = lambda step: np.ones(shape, dtype=bool)
    for steps, moves in ((1, 4), (3, 4), (2, 1), (2, 3)):
        reach = steps * moves
        square, diamond = (2 * reach + 1) ** 2, 2 * reach * reach + 2 * reach + 1
        assert packed_side(lit_at, passable, (30, 30), 0, oracle.AFTER, steps, 8, True, moves)[0] == square
        assert packed_side(lit_at, passable, (30, 30), 0, oracle.AFTER, steps, 4, True, moves)[0] == diamond
    # the per-side ceiling of the default flood: 24 steps of 4 moves, a 193 x 193 square
    big = np.ones((200, 200), dtype=bool)
    assert packed_side(lambda s: big, big, (100, 100), 0, oracle.AFTER, 24, 8, True, 4)[0] == 193 * 193 == 37_249


def test_every_pixel_entered_in_a_step_must_be_lit_in_that_step() -> None:
    """Step 1 lights only row 5, columns 5..8: from (5, 5) the rover walks along the lit run in step 1
    (3 moves, so (5, 8) is reached), but cannot enter an unlit pixel even to pass through it."""
    shape = (11, 14)
    passable = np.ones(shape, dtype=bool)
    step_one = np.zeros(shape, dtype=bool)
    step_one[5, 5:9] = True

    def lit_at(step):
        return step_one if step == 1 else np.ones(shape, dtype=bool)

    count, reached = packed_side(lit_at, passable, (5, 5), 0, oracle.AFTER, 1, 8, True, 4)
    assert reached == {(5, c) for c in range(5, 9)} and count == 4
    # one move per step could not get past (5, 6) in that step
    assert packed_side(lit_at, passable, (5, 5), 0, oracle.AFTER, 1, 8, True, 1)[1] == {(5, 5), (5, 6)}


@pytest.mark.parametrize("moves", [0, -1, 2.5])
def test_moves_per_step_must_be_a_positive_integer(moves) -> None:
    with pytest.raises(ValueError):
        packed.flood_side_packed(np.zeros((1, 2, 1), np.uint64), np.zeros((2, 1), np.uint64), (0, 0), steps=1,
                                 moves_per_step=moves)
    with pytest.raises(ValueError):
        oracle.flood_side_reference(lambda s: np.ones((2, 2), bool), np.ones((2, 2), bool), (0, 0), 0,
                                    direction=oracle.AFTER, steps=1, moves_per_step=moves)
