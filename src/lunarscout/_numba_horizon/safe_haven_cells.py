"""Stage 2 of the safe-haven analysis: which cells have a haven (private).

A cell is a ``cell_size`` x ``cell_size`` block of the stage-1 raster. Stage 2 scans the
stage-1 ``hibernation`` product once and records, for every cell with at least one haven
pixel (hibernation at most ``survival_limit_hours``) in any month, the number of months that
have one and the number of haven pixels summed over the months. Stage 3
(``safe_haven_blocks.run_pass_two``) reads that list instead of scanning again.

The scan reads one strip of ``cell_size`` rows across the whole raster at a time, so each file
block is read once (the per-cell scan in ``haven_cells`` reads every cell's window separately).
The list is written as JSON together with the identity of the stage-1 files it came from, and a
reader refuses it if those files, the survival limit or the cell size have changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .safe_haven_windows import Stage1Rasters

FORMAT = "lunarscout-safe-haven-cells-v1"


def file_identity(path) -> list:
    """A file's path, size and modification time."""
    stat = Path(path).stat()
    return [str(path), int(stat.st_size), int(stat.st_mtime_ns)]


def sort_cells(found: list[dict]) -> list[dict]:
    """Most months with a haven first, then most haven pixels, then by cell."""
    return sorted(found, key=lambda e: (-e["months_with_haven"], -e["haven_pixels"], e["cell_row"], e["cell_col"]))


def scan_cells(
    stage1: Stage1Rasters,
    *,
    survival_limit_hours: float,
    cell_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Cells with a haven pixel, from one strip-wise pass over the stage-1 hibernation product.

    Each entry has ``cell_row``, ``cell_col``, ``months_with_haven`` and ``haven_pixels``,
    ordered as ``sort_cells`` does; the result equals ``safe_haven_windows.haven_cells`` for all
    cells. ``progress(strips_done, strips_total)`` is called after each strip of cell rows.
    """
    if cell_size < 1:
        raise ValueError("cell_size must be positive")
    cell_rows, cell_columns = -(-stage1.height // cell_size), -(-stage1.width // cell_size)
    found: list[dict] = []
    for cell_row in range(cell_rows):
        row0, row1 = cell_row * cell_size, min(stage1.height, (cell_row + 1) * cell_size)
        haven = stage1.read_haven_window(row0, row1, 0, stage1.width, survival_limit_hours)
        for cell_column in range(cell_columns):
            col0, col1 = cell_column * cell_size, min(stage1.width, (cell_column + 1) * cell_size)
            pixels = haven[:, :, col0:col1].sum(axis=(1, 2))
            if pixels.any():
                found.append({
                    "cell_row": cell_row, "cell_col": cell_column,
                    "months_with_haven": int((pixels > 0).sum()), "haven_pixels": int(pixels.sum()),
                })
        if progress is not None:
            progress(cell_row + 1, cell_rows)
    return sort_cells(found)


def write_cell_list(path, cells: list[dict], stage1: Stage1Rasters, *, survival_limit_hours: float,
                    cell_size: int, overwrite: bool = False) -> None:
    """Write the cell list as JSON, through a staging file, with its provenance."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise ValueError(f"{path} exists; pass overwrite=True to replace it")
    if not path.parent.is_dir():
        raise ValueError(f"the directory of {path} does not exist")
    document = {
        "format": FORMAT,
        "survival_limit_hours": survival_limit_hours,
        "cell_size": cell_size,
        "raster": {"height": stage1.height, "width": stage1.width, "bands": stage1.band_count},
        "stage1": [file_identity(stage1.hibernation_path), file_identity(stage1.joint_total_path)],
        "cells": cells,
    }
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(document), encoding="utf-8")
    os.replace(staging, path)


def read_cell_list(path, stage1: Stage1Rasters, *, survival_limit_hours: float, cell_size: int) -> list[dict]:
    """The cells in a stage-2 file, or ``ValueError`` if it does not belong to these inputs."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the stage-2 cell list {path}: {error}") from error
    if document.get("format") != FORMAT:
        raise ValueError(f"{path} is not a stage-2 cell list (format {document.get('format')!r})")
    expected = {
        "survival_limit_hours": survival_limit_hours,
        "cell_size": cell_size,
        "raster": {"height": stage1.height, "width": stage1.width, "bands": stage1.band_count},
        "stage1": [file_identity(stage1.hibernation_path), file_identity(stage1.joint_total_path)],
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(
                f"the stage-2 cell list {path} was made with a different {key} "
                f"({document.get(key)!r}, now {value!r}); run stage 2 again"
            )
    return document["cells"]
