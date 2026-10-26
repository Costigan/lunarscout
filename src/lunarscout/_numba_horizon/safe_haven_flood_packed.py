"""Stage-2 pass 2: the flood on packed bits (private).

The same rule as the plain-Python oracle in ``safe_haven_flood.py`` (spec section 6),
on rows of 64-bit words: bit ``k`` of word ``w`` in a row is column ``64 * w + k``.
One flood step is a three-row, three-bit dilation of the frontier followed by an AND
with the passable mask and the lit mask of that step, so a whole word of 64 pixels is
one set of integer operations.  The work at step ``j`` is limited to the box of
``j`` pixels around the seed, which is as far as the frontier can have moved.

The kernel is a plain function, compiled with Numba on first use (importing this
module compiles nothing); ``compiled=False`` runs the same function in Python.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .safe_haven_flood import BEFORE, AFTER, FloodResult, LitAt

WORD_BITS = 64


def pack_bits(mask: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    """Pack ``(..., rows, columns)`` booleans into ``(..., rows, ceil(columns / 64))`` words."""
    mask = np.asarray(mask, dtype=bool)
    columns = mask.shape[-1]
    words = -(-columns // WORD_BITS)
    padded = np.zeros(mask.shape[:-1] + (words * WORD_BITS,), dtype=bool)
    padded[..., :columns] = mask
    packed = np.packbits(padded, axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8").astype(np.uint64, copy=False)


def unpack_bits(words: npt.ArrayLike, columns: int) -> npt.NDArray[np.bool_]:
    """Inverse of ``pack_bits`` for a known number of columns."""
    words = np.ascontiguousarray(np.asarray(words, dtype="<u8"))
    bits = np.unpackbits(words.view(np.uint8), axis=-1, bitorder="little")
    return bits[..., :columns].astype(bool)


def build_flood(jit):
    """The flood kernel with ``jit`` applied to it and its helpers (identity for plain Python)."""

    @jit
    def popcount(value):
        one, two, four = np.uint64(1), np.uint64(2), np.uint64(4)
        value = value - ((value >> one) & np.uint64(0x5555555555555555))
        value = (value & np.uint64(0x3333333333333333)) + ((value >> two) & np.uint64(0x3333333333333333))
        value = (value + (value >> four)) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return (value * np.uint64(0x0101010101010101)) >> np.uint64(56)

    @jit
    def spread(cur, row, word, words):
        """Row ``row`` word ``word`` with every set bit also set in both neighbouring columns."""
        one, top = np.uint64(1), np.uint64(63)
        x = cur[row, word]
        below = cur[row, word - 1] if word > 0 else np.uint64(0)
        above = cur[row, word + 1] if word + 1 < words else np.uint64(0)
        return x | (x << one) | (below >> top) | (x >> one) | (above << top)

    @jit
    def flood_packed(lit, passable, seed_row, seed_column, steps, eight, moves):
        """One side of a flood on packed bits; returns ``(count, bits)``.

        ``lit[j - 1]`` is the lit mask at step ``j`` along the flood (the first step away from
        the boundary is ``lit[0]``), shape ``(steps, rows, words)``; ``passable`` is
        ``(rows, words)``.  ``eight`` selects 8 or 4 neighbours.  A step is ``moves`` moves,
        each into a passable pixel lit in that step, so the frontier after ``k`` moves lies
        within ``k`` pixels of the seed.  ``bits`` is the frontier at the last move reached,
        all zero when it emptied.
        """
        rows, words = passable.shape
        zero = np.uint64(0)
        cur = np.zeros((rows, words), dtype=np.uint64)
        nxt = np.zeros((rows, words), dtype=np.uint64)
        cur[seed_row, seed_column >> 6] = np.uint64(1) << np.uint64(seed_column & 63)
        for move in range(1, steps * moves + 1):
            j = (move - 1) // moves + 1
            row_first = max(0, seed_row - move)
            row_last = min(rows - 1, seed_row + move)
            word_first = max(0, seed_column - move) >> 6
            word_last = min(words * WORD_BITS - 1, seed_column + move) >> 6
            any_bits = zero
            for row in range(row_first, row_last + 1):
                for word in range(word_first, word_last + 1):
                    grown = spread(cur, row, word, words)
                    if eight:
                        if row > 0:
                            grown |= spread(cur, row - 1, word, words)
                        if row + 1 < rows:
                            grown |= spread(cur, row + 1, word, words)
                    else:
                        if row > 0:
                            grown |= cur[row - 1, word]
                        if row + 1 < rows:
                            grown |= cur[row + 1, word]
                    value = grown & passable[row, word] & lit[j - 1, row, word]
                    nxt[row, word] = value
                    any_bits |= value
            cur, nxt = nxt, cur
            if any_bits == zero:
                return 0, np.zeros((rows, words), dtype=np.uint64)
        count = 0
        for row in range(rows):
            for word in range(words):
                count += int(popcount(cur[row, word]))
        return count, cur

    return flood_packed


flood_packed = build_flood(lambda function: function)
_COMPILED: dict[str, object] = {}


def _kernel(compiled: bool):
    if not compiled:
        return flood_packed
    if "flood" not in _COMPILED:
        from numba import njit

        _COMPILED["flood"] = build_flood(njit(cache=False, nogil=True))
    return _COMPILED["flood"]


def flood_side_packed(
    lit_packed: npt.NDArray[np.uint64],
    passable_packed: npt.NDArray[np.uint64],
    seed: tuple[int, int],
    *,
    steps: int,
    connectivity: int = 8,
    moves_per_step: int = 1,
    compiled: bool = True,
) -> tuple[int, npt.NDArray[np.uint64]]:
    """The count and the final frontier of one side (see ``flood_packed``)."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if steps < 0:
        raise ValueError("steps must not be negative")
    if int(moves_per_step) != moves_per_step or moves_per_step < 1:
        raise ValueError("moves_per_step must be a positive integer")
    lit_packed = np.ascontiguousarray(lit_packed, dtype=np.uint64)
    passable_packed = np.ascontiguousarray(passable_packed, dtype=np.uint64)
    if lit_packed.ndim != 3 or lit_packed.shape[0] < steps or lit_packed.shape[1:] != passable_packed.shape:
        raise ValueError("lit must be (steps, rows, words) and match the passable mask")
    rows, words = passable_packed.shape
    row, column = int(seed[0]), int(seed[1])
    if not (0 <= row < rows and 0 <= column < words * WORD_BITS):
        raise ValueError("the seed lies outside the window")
    with np.errstate(over="ignore"):
        count, bits = _kernel(compiled)(
            lit_packed, passable_packed, row, column, int(steps), connectivity == 8, int(moves_per_step))
    return int(count), bits


def flood_seed_packed(
    lit_at: LitAt,
    passable: npt.NDArray[np.bool_],
    seed: tuple[int, int],
    boundary_before_step: int,
    boundary_after_step: int,
    *,
    steps: int,
    connectivity: int = 8,
    moves_per_step: int = 1,
    compiled: bool = True,
) -> FloodResult:
    """Both sides, from a ``lit_at(step)`` callable like the oracle's (packs what it needs)."""
    passable_packed = pack_bits(passable)
    counts = []
    for boundary, direction in ((boundary_before_step, BEFORE), (boundary_after_step, AFTER)):
        lit = pack_bits(np.stack([np.asarray(lit_at(boundary + direction * j), dtype=bool)
                                  for j in range(1, steps + 1)]
                                 or [np.zeros(passable.shape, dtype=bool)]))
        counts.append(flood_side_packed(
            lit, passable_packed, seed, steps=steps, connectivity=connectivity,
            moves_per_step=moves_per_step, compiled=compiled,
        )[0])
    return FloodResult(*counts)
