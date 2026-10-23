"""Stage-2 pass 2: the block schedule and the CPU driver that runs it (private).

``docs/safe-haven-windows-spec.md`` section 8.  The pole is processed in blocks of
``block_patches`` x ``block_patches`` patches (a whole number of cells), scanned in a
snake so that consecutive blocks share a strip of horizons.  For each block:

1. make the horizons of the block's windows resident (keeping those the previous block
   already held), once for all months;
2. for each month, light the missing lit chunks of the block's cell-months into a
   per-month store, then flood the seeds from it (shared windows reuse the store);
3. append the block's rows to a checkpoint, so a run resumes at the next block.

This is the CPU reference route: horizons are resident in host memory, lighting is the
CPU lightmap and the flood is the packed-bit CPU flood.  The same schedule, store keys
and rows apply when the lighting moves to the GPU.  The table is the same for any
``block_patches`` and after a resume.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt

from .file_format import PATCH_SIZE, HorizonTileStore, read_patch_horizons
from .safe_haven_flood_packed import pack_bits
from .safe_haven_litstore import LitStore, fill_lit_store, flood_seed_from_store
from .safe_haven_windows import (
    FloodGrid,
    PlanCounts,
    Stage1Rasters,
    TimeContext,
    VectorSource,
    WindowParameters,
    cell_window,
    flood_step_minutes,
    halo_pixels,
    haven_cells,
    iter_cell_months,
)

Patch = tuple[int, int]
Cell = tuple[int, int]

#: The columns of the output table, in order (specification, section 7).
TABLE_COLUMNS = (
    "cell_row", "cell_col", "x_min_m", "y_min_m", "x_max_m", "y_max_m", "month", "rank",
    "seed_row", "seed_col", "seed_x_m", "seed_y_m", "hibernation_hours",
    "outage_start_utc", "outage_end_utc", "boundary_before_utc", "boundary_after_utc",
    "count_before", "count_after", "score",
    "nearby_haven_count", "nearby_pixel_count", "nearby_haven_fraction",
    "candidate_seeds", "valid_seeds", "earth_visible_before_h", "earth_visible_after_h",
)


# ---------------------------------------------------------------------------
# Geometry and order
# ---------------------------------------------------------------------------


def block_order(cell_rows: int, cell_columns: int, block_cells: int) -> list[tuple[int, int]]:
    """Block indices in snake order: even block rows run left to right, odd ones right to left."""
    rows, columns = -(-cell_rows // block_cells), -(-cell_columns // block_cells)
    order = []
    for row in range(rows):
        for column in (range(columns) if row % 2 == 0 else range(columns - 1, -1, -1)):
            order.append((row, column))
    return order


def block_of_cell(cell: Cell, block_cells: int) -> tuple[int, int]:
    return cell[0] // block_cells, cell[1] // block_cells


def cell_patches_with_halo(
    cell: Cell, *, cell_patches: int, halo_patches: int, patch_rows: int, patch_columns: int
) -> set[Patch]:
    """The patches of a cell's window: the cell plus the halo, clipped to the raster."""
    row0 = max(0, cell[0] * cell_patches - halo_patches)
    row1 = min(patch_rows, (cell[0] + 1) * cell_patches + halo_patches)
    column0 = max(0, cell[1] * cell_patches - halo_patches)
    column1 = min(patch_columns, (cell[1] + 1) * cell_patches + halo_patches)
    return {(r, c) for r in range(row0, row1) for c in range(column0, column1)}


@dataclass(frozen=True)
class LoadPlan:
    """What a sequence of blocks asks of the horizon store."""

    loads: int               # patch reads, counting carried-over patches once
    distinct_patches: int
    peak_resident: int       # most patches resident at once

    @property
    def loads_per_patch(self) -> float:
        return self.loads / self.distinct_patches if self.distinct_patches else 0.0


def plan_horizon_loads(patch_sets: Iterable[set[Patch]], *, carry_over: bool = True) -> LoadPlan:
    """Patch reads for blocks run in the given order (the previous block's patches stay)."""
    held: set[Patch] = set()
    seen: set[Patch] = set()
    loads = peak = 0
    for needed in patch_sets:
        loads += len(needed - held) if carry_over else len(needed)
        held = set(needed) if carry_over else set()
        seen |= needed
        peak = max(peak, len(needed))
    return LoadPlan(loads, len(seen), peak)


class HorizonResidency:
    """Horizons held for the current block.  ``set_resident`` drops what the new block no
    longer needs before loading what it lacks, so memory never exceeds one block."""

    def __init__(self, loader: Callable[[Patch], npt.NDArray[np.float32] | None]) -> None:
        self._loader = loader
        self._held: dict[Patch, npt.NDArray[np.float32] | None] = {}
        self.loads = 0
        self.peak_resident = 0

    def set_resident(self, patches: Iterable[Patch]) -> None:
        wanted = set(patches)
        for patch in [p for p in self._held if p not in wanted]:
            del self._held[patch]
        for patch in sorted(wanted - self._held.keys()):
            self._held[patch] = self._loader(patch)
            self.loads += 1
        self.peak_resident = max(self.peak_resident, len(self._held))

    def get(self, patch: Patch) -> npt.NDArray[np.float32] | None:
        return self._held[patch]

    def __len__(self) -> int:
        return len(self._held)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class PassTwoCounts:
    """Counters of the flood stage; every field adds across blocks."""

    blocks_run: int = 0
    cell_months_flooded: int = 0
    seeds_flooded: int = 0
    seeds_boundary_mismatch: int = 0
    seeds_without_both_sides: int = 0
    cell_months_without_valid_seed: int = 0
    rows: int = 0
    lit_units_computed: int = 0
    lit_units_reused: int = 0


@dataclass
class PassTwoReport:
    """What a pass-2 run did (blocks resumed from a checkpoint are counted, not re-run)."""

    blocks_total: int
    blocks_resumed: int
    horizon_loads: int
    peak_resident_patches: int
    flood_step_minutes: int
    flood_steps: int
    plan: PlanCounts
    flood: PassTwoCounts
    rows: int
    output_path: str

    def as_dict(self) -> dict:
        return {**{k: v for k, v in asdict(self).items() if k not in ("plan", "flood")},
                "pass_one_counts": asdict(self.plan), "flood_counts": asdict(self.flood)}


def _add_counts(total, part: dict) -> None:
    for name, value in part.items():
        setattr(total, name, getattr(total, name) + value)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


class CachedVectors:
    """A vector source that remembers its answers by the exact times asked, so the
    same chunk's Sun vectors are computed once however many patches or cells use it."""

    def __init__(self, source: VectorSource, *, limit: int = 20000) -> None:
        self._source = source
        self._limit = limit
        self._answers: dict[bytes, npt.NDArray[np.float64]] = {}
        self.calls = 0
        self.hits = 0

    def __call__(self, stamps: npt.ArrayLike) -> npt.NDArray[np.float64]:
        stamps = np.ascontiguousarray(np.asarray(stamps).astype("datetime64[us]"))
        key = stamps.tobytes()
        answer = self._answers.get(key)
        if answer is None:
            self.calls += 1
            answer = np.asarray(self._source(stamps), dtype=np.float64)
            if len(self._answers) >= self._limit:
                self._answers.pop(next(iter(self._answers)))
            self._answers[key] = answer
        else:
            self.hits += 1
        return answer


def _month_labels(time: TimeContext) -> list[str]:
    return [str(np.datetime64(int(time.times_us[int(first)]), "us"))[:7] for first in time.month_first]


def _utc(grid: FloodGrid, step: int) -> str:
    return str(grid.times([step])[0]) + "Z"


def _window_patches(window: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """``(patch_row0, patch_column0, patch_rows, patch_columns)`` of a patch-aligned window."""
    rows = -(-(window[1] - window[0]) // PATCH_SIZE)
    columns = -(-(window[3] - window[2]) // PATCH_SIZE)
    return window[0] // PATCH_SIZE, window[2] // PATCH_SIZE, rows, columns


def _file_identity(path) -> list:
    """A file's path, size and modification time, so a rewritten product is not mistaken for
    the one a checkpoint was made from."""
    stat = Path(path).stat()
    return [str(path), int(stat.st_size), int(stat.st_mtime_ns)]


def _checkpoint_cells(path: Path) -> list[Cell] | None:
    """The cell list in a checkpoint's header, or ``None`` if there is no usable header."""
    try:
        with path.open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
        cells = header["config"]["cells"]
        return [(int(r), int(c)) for r, c in cells] if header.get("type") == "header" else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _fingerprint(*, parameters, block_patches, flood_steps, time: TimeContext, cells, stage1,
                 earth_elevation_threshold_deg, sunlight_fraction_threshold, merge_gap_hours) -> dict:
    return {
        "parameters": asdict(parameters), "block_patches": block_patches, "flood_steps": flood_steps,
        "times": [int(time.times_us[0]), int(time.times_us[-1]), int(len(time.times_us))],
        "cells": [list(c) for c in cells],
        "stage1": [_file_identity(stage1.hibernation_path), _file_identity(stage1.joint_total_path)],
        "earth_elevation_threshold_deg": earth_elevation_threshold_deg,
        "sunlight_fraction_threshold": sunlight_fraction_threshold,
        "merge_gap_hours": merge_gap_hours,
    }


def _read_checkpoint(path: Path, fingerprint: dict) -> dict[tuple[int, int], dict]:
    done: dict[tuple[int, int], dict] = {}
    if not path.exists():
        return done
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return done
    header = json.loads(lines[0])
    if header.get("type") != "header" or header.get("config") != fingerprint:
        raise ValueError(
            f"the checkpoint {path} was made with a different configuration; "
            "remove it to start again"
        )
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            break                       # a torn last line: that block is run again
        if entry.get("type") == "block":
            done[tuple(entry["block"])] = entry
    return done


def run_pass_two(
    *,
    dem,
    horizon_store: HorizonTileStore,
    stage1: Stage1Rasters,
    times: npt.ArrayLike,
    sun_vectors_at: VectorSource,
    earth_vectors_at: VectorSource,
    output_path: str | Path,
    parameters: WindowParameters = WindowParameters(),
    checkpoint_path: str | Path | None = None,
    cells: list[Cell] | None = None,
    overwrite: bool = False,
    observer_height_m: float = 0.0,
    earth_elevation_threshold_deg: float = 2.0,
    sunlight_fraction_threshold: float = 0.2,
    merge_gap_hours: float = 24.0,
    compiled: bool = True,
    progress: Callable[[int, int], None] | None = None,
    stop_after_blocks: int | None = None,
) -> PassTwoReport | None:
    """Run the pass-2 schedule and write the table to ``output_path`` (CSV).

    ``cells`` limits the run to those ``(cell_row, cell_column)`` indices; the default is
    every cell with a haven pixel.  A checkpoint (``<output>.checkpoint.jsonl`` by
    default) is appended after each block, and an existing one with the same
    configuration is resumed.  The CSV is written to a staging file and moved into place
    at the end, so a failure leaves any earlier output alone; the checkpoint is removed
    once the CSV is in place.  ``progress(blocks_done, blocks_total)`` is called once at the start (with the
    blocks a checkpoint already holds) and after each block, so a caller can time
    this run's blocks alone.  ``stop_after_blocks`` ends the run early for tests and
    returns ``None``.
    """
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"{output} exists; pass overwrite=True to replace it")
    if not output.parent.is_dir():
        raise ValueError(f"the directory of {output} does not exist")
    checkpoint = Path(checkpoint_path) if checkpoint_path else output.with_name(output.name + ".checkpoint.jsonl")
    if (stage1.height, stage1.width) != (dem.height, dem.width):
        raise ValueError("the stage-1 products and the DEM differ in size")

    pixel_size = abs(float(dem.geo_transform[1]))
    sun_vectors_at = CachedVectors(sun_vectors_at)
    earth_vectors_at = CachedVectors(earth_vectors_at)
    time = TimeContext.build(times, sun_vectors_at, earth_vectors_at)
    grid = FloodGrid(flood_step_minutes(pixel_size, parameters.rover_speed_m_s))
    flood_steps = grid.step_count(parameters.operational_hours)
    halo = halo_pixels(flood_steps)
    cell_size = parameters.cell_patches * PATCH_SIZE
    block_cells = parameters.block_patches // parameters.cell_patches
    patch_rows, patch_columns = -(-dem.height // PATCH_SIZE), -(-dem.width // PATCH_SIZE)
    cell_rows, cell_columns = -(-dem.height // cell_size), -(-dem.width // cell_size)
    if cells is None:
        # A resume takes the cell list from the checkpoint instead of scanning the whole
        # stage-1 raster again (about 24 minutes at the pole). The fingerprint check below
        # still refuses the checkpoint if the settings or the stage-1 files have changed.
        cells = _checkpoint_cells(checkpoint)
    if cells is None:
        cells = [(e["cell_row"], e["cell_col"]) for e in haven_cells(
            stage1, survival_limit_hours=parameters.survival_limit_hours, cell_size=cell_size)]
        cells.sort()
    cells = sorted(set((int(r), int(c)) for r, c in cells))
    by_block: dict[tuple[int, int], list[Cell]] = {}
    for cell in cells:
        by_block.setdefault(block_of_cell(cell, block_cells), []).append(cell)
    order = [b for b in block_order(cell_rows, cell_columns, block_cells) if b in by_block]

    fingerprint = _fingerprint(
        parameters=parameters, block_patches=parameters.block_patches, flood_steps=flood_steps, time=time,
        cells=cells, stage1=stage1, earth_elevation_threshold_deg=earth_elevation_threshold_deg,
        sunlight_fraction_threshold=sunlight_fraction_threshold, merge_gap_hours=merge_gap_hours,
    )
    done = _read_checkpoint(checkpoint, fingerprint)
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        checkpoint.write_text(json.dumps({"type": "header", "config": fingerprint}) + "\n", encoding="utf-8")

    months = _month_labels(time)
    transform = np.asarray(dem.geo_transform, dtype=np.float64)
    residency = HorizonResidency(
        lambda patch: read_patch_horizons(horizon_store, patch[0] * PATCH_SIZE, patch[1] * PATCH_SIZE, observer_height_m)
    )
    plan_counts, flood_counts = PlanCounts(), PassTwoCounts()
    for entry in done.values():
        _add_counts(plan_counts, entry["plan_counts"])
        _add_counts(flood_counts, entry["flood_counts"])
    ran = 0
    if progress is not None:
        progress(len(done), len(order))
    for block in order:
        if block in done:
            continue
        if stop_after_blocks is not None and ran >= stop_after_blocks:
            return None
        block_plan, block_flood, rows = _run_block(
            by_block[block], dem=dem, horizon_store=horizon_store, stage1=stage1, time=time, grid=grid,
            parameters=parameters, flood_steps=flood_steps, halo=halo, cell_size=cell_size,
            residency=residency, months=months, transform=transform, compiled=compiled,
            observer_height_m=observer_height_m, earth_elevation_threshold_deg=earth_elevation_threshold_deg,
            sunlight_fraction_threshold=sunlight_fraction_threshold, merge_gap_hours=merge_gap_hours,
            sun_vectors_at=sun_vectors_at, earth_vectors_at=earth_vectors_at,
        )
        entry = {"type": "block", "block": list(block), "rows": rows,
                 "plan_counts": asdict(block_plan), "flood_counts": asdict(block_flood)}
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        done[block] = entry
        _add_counts(plan_counts, entry["plan_counts"])
        _add_counts(flood_counts, entry["flood_counts"])
        ran += 1
        if progress is not None:
            progress(len(done), len(order))

    all_rows = [row for entry in done.values() for row in entry["rows"]]
    all_rows.sort(key=lambda r: (r["cell_row"], r["cell_col"], r["month"], r["rank"]))
    staging = output.with_name(output.name + ".partial")
    with staging.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(staging, output)
    checkpoint.unlink()
    return PassTwoReport(
        blocks_total=len(order), blocks_resumed=len(order) - ran, horizon_loads=residency.loads,
        peak_resident_patches=residency.peak_resident, flood_step_minutes=grid.step_minutes,
        flood_steps=flood_steps, plan=plan_counts, flood=flood_counts, rows=len(all_rows),
        output_path=str(output),
    )


def _run_block(
    block_cells_list: list[Cell], *, dem, horizon_store, stage1, time, grid, parameters, flood_steps,
    halo, cell_size, residency, months, transform, compiled, observer_height_m,
    earth_elevation_threshold_deg, sunlight_fraction_threshold, merge_gap_hours,
    sun_vectors_at, earth_vectors_at,
):
    """One block: pass-1 seeds for its cells, then lit store and floods month by month."""
    report: list = []
    records = list(iter_cell_months(
        dem=dem, horizon_store=horizon_store, stage1=stage1, times=time.times_us.astype("datetime64[us]"),
        sun_vectors_at=sun_vectors_at, earth_vectors_at=earth_vectors_at, parameters=parameters,
        observer_height_m=observer_height_m, earth_elevation_threshold_deg=earth_elevation_threshold_deg,
        sunlight_fraction_threshold=sunlight_fraction_threshold, merge_gap_hours=merge_gap_hours,
        cells=block_cells_list, report=report, time_context=time,
    ))
    plan_counts = report[0].counts
    flood_counts = PassTwoCounts(blocks_run=1)
    windows: dict[Cell, tuple] = {}
    for record in records:
        cell = (record["cell_row"], record["cell_col"])
        if cell not in windows:
            window, _core = cell_window(cell[0], cell[1], cell_size=cell_size, halo=halo,
                                        height=dem.height, width=dem.width)
            windows[cell] = (window, _window_patches(window), stage1.read_hibernation_window(*window))
    residency.set_resident(
        (pr, pc)
        for _window, (pr0, pc0, prows, pcols), _hibernation in windows.values()
        for pr in range(pr0, pr0 + prows) for pc in range(pc0, pc0 + pcols)
    )
    rows: list[dict] = []
    for month in sorted({record["month"] for record in records}):
        store = LitStore(grid, (dem.height, dem.width))
        for record in (r for r in records if r["month"] == month):
            cell = (record["cell_row"], record["cell_col"])
            window, patch_window, hibernation = windows[cell]
            patches = [
                (pr, pc)
                for pr in range(patch_window[0], patch_window[0] + patch_window[2])
                for pc in range(patch_window[1], patch_window[1] + patch_window[3])
            ]
            counts = fill_lit_store(
                store, dem=dem, horizon_store=horizon_store, patches=patches, chunks=record["chunks"],
                sun_vectors_at=sun_vectors_at,
                operating_sunlight_fraction=parameters.operating_sunlight_fraction,
                observer_height_m=observer_height_m, horizons_for=residency.get,
            )
            flood_counts.lit_units_computed += counts.computed
            flood_counts.lit_units_reused += counts.reused
            passable = np.zeros((patch_window[2] * PATCH_SIZE, patch_window[3] * PATCH_SIZE), dtype=bool)
            usable = np.isfinite(hibernation[month])
            passable[:usable.shape[0], :usable.shape[1]] = usable
            passable_packed = pack_bits(passable)
            flood_counts.cell_months_flooded += 1
            results = []
            for seed in record["seeds"]:
                flood_counts.seeds_flooded += 1
                result = flood_seed_from_store(
                    store, passable_packed, patch_window,
                    (seed["row"] - window[0], seed["column"] - window[2]),
                    seed["boundary_before_step"], seed["boundary_after_step"],
                    steps=flood_steps, connectivity=parameters.connectivity, compiled=compiled,
                )
                if result is None:
                    flood_counts.seeds_boundary_mismatch += 1
                elif not result.valid:
                    flood_counts.seeds_without_both_sides += 1
                else:
                    results.append((seed, result))
            if not results:
                flood_counts.cell_months_without_valid_seed += 1
                continue
            results.sort(key=lambda item: (-item[1].score, item[0]["row"], item[0]["column"]))
            row0, col0 = cell[0] * cell_size, cell[1] * cell_size
            row1, col1 = min(dem.height, row0 + cell_size), min(dem.width, col0 + cell_size)
            for rank, (seed, result) in enumerate(results[: parameters.max_seeds_per_cell_month], start=1):
                rows.append({
                    "cell_row": cell[0], "cell_col": cell[1],
                    "x_min_m": float(transform[0] + col0 * transform[1]),
                    "y_min_m": float(transform[3] + row1 * transform[5]),
                    "x_max_m": float(transform[0] + col1 * transform[1]),
                    "y_max_m": float(transform[3] + row0 * transform[5]),
                    "month": months[month], "rank": rank,
                    "seed_row": seed["row"], "seed_col": seed["column"],
                    "seed_x_m": float(transform[0] + (seed["column"] + 0.5) * transform[1]),
                    "seed_y_m": float(transform[3] + (seed["row"] + 0.5) * transform[5]),
                    "hibernation_hours": seed["hibernation_hours"],
                    "outage_start_utc": seed["outage_start_utc"], "outage_end_utc": seed["outage_end_utc"],
                    "boundary_before_utc": _utc(grid, seed["boundary_before_step"]),
                    "boundary_after_utc": _utc(grid, seed["boundary_after_step"]),
                    "count_before": result.count_before, "count_after": result.count_after,
                    "score": result.score,
                    "nearby_haven_count": seed["nearby_haven_count"],
                    "nearby_pixel_count": seed["nearby_pixel_count"],
                    "nearby_haven_fraction": seed["nearby_haven_fraction"],
                    "candidate_seeds": len(record["seeds"]), "valid_seeds": len(results),
                    "earth_visible_before_h": seed["earth_visible_before_hours"],
                    "earth_visible_after_h": seed["earth_visible_after_hours"],
                })
        store.clear()
    flood_counts.rows = len(rows)
    return plan_counts, flood_counts, rows
