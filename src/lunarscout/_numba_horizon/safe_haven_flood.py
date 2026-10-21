"""Stage-2 pass 2: the flood, as a plain-Python reference (private).

``docs/safe-haven-windows-spec.md`` section 6 is the specification.  This module is
the oracle the packed-bit and GPU implementations are tested against.  It favours
being obviously right over speed: sets of pixels, explicit loops, no NumPy in the
flood itself, one rule per function.  It is used on small windows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from .safe_haven import OPERATING_FUZZ

Pixel = tuple[int, int]
#: Lit state of every window pixel at a flood-grid step index: a 2-D boolean array.
LitAt = Callable[[int], npt.NDArray[np.bool_]]

BEFORE = -1
AFTER = +1


def neighbour_offsets(connectivity: int) -> list[Pixel]:
    """The moves of one flood step: waiting, then the 4 or 8 adjacent pixels."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return offsets


def lit_threshold_test(fraction: npt.ArrayLike, operating_sunlight_fraction: float) -> npt.NDArray[np.bool_]:
    """The stage-1 operating rule: ``fraction >= threshold`` with the stage-1 fuzz."""
    return np.asarray(fraction, dtype=np.float64) >= operating_sunlight_fraction - OPERATING_FUZZ


def flood_side_reference(
    lit_at: LitAt,
    passable: npt.NDArray[np.bool_],
    seed: Pixel,
    boundary_step: int,
    *,
    direction: int,
    steps: int,
    connectivity: int = 8,
) -> set[Pixel]:
    """Pixels reachable at the end of one side of the flood (spec section 6).

    ``R_0 = {seed}`` at ``boundary_step``; for ``j = 1 .. steps``::

        R_j = neighbours(R_{j-1}) AND passable AND lit(boundary_step + direction * j)

    ``direction`` is ``BEFORE`` (-1, time runs backward) or ``AFTER`` (+1).  The seed
    is taken as lit at the boundary and is not tested there.  Pixels outside
    ``passable`` are impassable, so the flood is clipped to the window.  An emptied
    frontier stays empty.
    """
    if direction not in (BEFORE, AFTER):
        raise ValueError("direction must be BEFORE (-1) or AFTER (+1)")
    if steps < 0:
        raise ValueError("steps must not be negative")
    passable = np.asarray(passable, dtype=bool)
    rows, columns = passable.shape
    offsets = neighbour_offsets(connectivity)
    reached: set[Pixel] = {(int(seed[0]), int(seed[1]))}
    for j in range(1, steps + 1):
        lit = np.asarray(lit_at(boundary_step + direction * j), dtype=bool)
        if lit.shape != passable.shape:
            raise ValueError("a lit raster must have the shape of the passable mask")
        following: set[Pixel] = set()
        for row, column in reached:
            for d_row, d_column in offsets:
                r, c = row + d_row, column + d_column
                if 0 <= r < rows and 0 <= c < columns and passable[r, c] and lit[r, c]:
                    following.add((r, c))
        reached = following
        if not reached:
            break
    return reached


@dataclass(frozen=True)
class FloodResult:
    """Both sides of one seed's flood."""

    count_before: int
    count_after: int

    @property
    def score(self) -> int:
        return self.count_before + self.count_after

    @property
    def valid(self) -> bool:
        """A seed is valid only if both sides reach at least one pixel."""
        return self.count_before > 0 and self.count_after > 0


def flood_seed_reference(
    lit_at: LitAt,
    passable: npt.NDArray[np.bool_],
    seed: Pixel,
    boundary_before_step: int,
    boundary_after_step: int,
    *,
    steps: int,
    connectivity: int = 8,
) -> FloodResult:
    """The counts of both sides (spec section 6, "Count" and "Seed score")."""
    before = flood_side_reference(
        lit_at, passable, seed, boundary_before_step,
        direction=BEFORE, steps=steps, connectivity=connectivity,
    )
    after = flood_side_reference(
        lit_at, passable, seed, boundary_after_step,
        direction=AFTER, steps=steps, connectivity=connectivity,
    )
    return FloodResult(len(before), len(after))


def lit_reference_cube(
    signals,
    sun_vectors_at: Callable[[npt.ArrayLike], npt.ArrayLike],
    grid,
    window_origin: Pixel,
    shape: Pixel,
    first_step: int,
    last_step: int,
    operating_sunlight_fraction: float = 1.0,
) -> npt.NDArray[np.bool_]:
    """Lit state of every window pixel at flood steps ``first_step .. last_step``.

    Shape ``(last_step - first_step + 1, rows, columns)``.  ``signals`` is a
    ``PixelSignals`` (the single-pixel kernel with the patch kernel's arithmetic).
    A pixel without a horizon tile, or outside the DEM, is never lit.  This is the slow
    per-pixel route, for tests and small windows.
    """
    if last_step < first_step:
        raise ValueError("last_step must not be before first_step")
    steps = np.arange(first_step, last_step + 1, dtype=np.int64)
    vectors = np.asarray(sun_vectors_at(grid.times(steps)), dtype=np.float64)
    rows, columns = shape
    cube = np.zeros((steps.size, rows, columns), dtype=bool)
    for row in range(rows):
        for column in range(columns):
            fraction = signals.series(
                window_origin[0] + row, window_origin[1] + column, vectors, margin=False
            )
            if fraction is not None:
                cube[:, row, column] = lit_threshold_test(fraction, operating_sunlight_fraction)
    return cube
