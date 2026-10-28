"""The permanently shadowed regions (PSRs) a chain's route passes next to (private).

A route passes next to a PSR where a PSR pixel is among the eight neighbours of one of its pixels (or is the pixel).
For every predecessor edge this gives the distinct PSR ids along its route; a seed's chain is the routes from its
root to it, so its count of each size class is the number of distinct PSRs of that class over those routes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..errors import InputError


def passed_psrs(paths, ids: npt.NDArray[np.int32]) -> dict[int, np.ndarray]:
    """``{head seed id: sorted distinct PSR ids}`` next to the route of each head (``paths`` is a ``PathSet``;
    ``ids`` the PSR id raster, 0 where there is no PSR). Heads whose route passes no PSR are left out."""
    height, width = ids.shape
    if len(paths.rows) and (paths.rows.min() < 0 or paths.rows.max() >= height or paths.columns.min() < 0
                            or paths.columns.max() >= width):
        raise InputError("a route leaves the PSR id raster", code="route_psr_grid_mismatch")
    modulus = int(ids.max(initial=0)) + 1
    rows, columns = paths.rows.astype(np.int64), paths.columns.astype(np.int64)
    which = np.repeat(np.arange(len(paths.heads), dtype=np.int64), np.diff(paths.offsets))
    keys = np.zeros(0, dtype=np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            r, c = rows + dy, columns + dx
            inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
            value = np.zeros(len(rows), dtype=np.int64)
            value[inside] = ids[r[inside], c[inside]]
            found = value > 0
            keys = np.union1d(keys, which[found] * modulus + value[found])
    result: dict[int, np.ndarray] = {}
    if len(keys):
        path_index, psr_id = keys // modulus, keys % modulus
        starts = np.flatnonzero(np.r_[True, path_index[1:] != path_index[:-1]])
        for begin, end in zip(starts, np.r_[starts[1:], len(keys)]):
            result[int(paths.heads[path_index[begin]])] = psr_id[begin:end]
    return result


def chain_psr_counts(predecessor: npt.NDArray[np.int64], length: npt.NDArray[np.int64], edge_psrs: dict,
                     category: npt.NDArray[np.int8]) -> npt.NDArray[np.int64]:
    """``(seeds, 3)``: for every seed the number of distinct small, medium and large PSRs next to the routes of its
    chain (the edges from its root to it). ``predecessor[i]`` is the 1-based predecessor id of seed ``i + 1`` (0 for
    a root), ``length`` its chain length and ``category[psr_id]`` the size class 1 to 3."""
    count = len(predecessor)
    chain: list[frozenset | None] = [None] * count
    counts = np.zeros((count, 3), dtype=np.int64)
    for level in range(2, int(length.max(initial=1)) + 1):
        for index in np.flatnonzero(length == level):
            before = chain[predecessor[index] - 1] or frozenset()
            chain[index] = before | frozenset(edge_psrs.get(index + 1, ()).tolist() if index + 1 in edge_psrs else ())
            for psr in chain[index]:
                counts[index, category[psr] - 1] += 1
    return counts
