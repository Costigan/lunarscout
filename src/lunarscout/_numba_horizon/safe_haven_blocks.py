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

import contextlib
import csv
import json
import logging
import math
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time as wall_clock
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt

from .file_format import PATCH_SIZE, HorizonTileStore, read_patch_horizons
from .safe_haven_flood_packed import pack_bits
from .safe_haven_cells import scan_cells
from .safe_haven_litstore import LitStore, fill_lit_store, flood_seed_from_store
from .safe_haven_stage3_store import check_run, fill_lit_store_from_light_store
from .safe_haven_stage1_store import store_identity
from .safe_haven_windows import (
    FloodGrid,
    PlanCounts,
    Stage1Rasters,
    TimeContext,
    VectorSource,
    WindowParameters,
    cell_window,
    halo_pixels,
    iter_cell_months,
)

logger = logging.getLogger(__name__)

Patch = tuple[int, int]
Cell = tuple[int, int]

#: The columns of the output table, in order (specification, section 7). Each row is a safe haven scored by
#: its opportunity to operate before and after the non-operations period (the hibernation).
TABLE_COLUMNS = (
    "cell_row", "cell_col", "x_min_m", "y_min_m", "x_max_m", "y_max_m", "month", "rank",
    "seed_row", "seed_col", "seed_x_m", "seed_y_m", "hibernation_hours",
    "non_operations_start_utc", "non_operations_end_utc", "before_window_end_utc", "after_window_start_utc",
    "lit_reachable_pixels_before", "lit_reachable_pixels_after", "operating_opportunity_score",
    "nearby_haven_count", "nearby_pixel_count", "nearby_haven_fraction",
    "candidate_seeds", "valid_seeds", "earth_visible_before_h", "earth_visible_after_h",
)

#: The names the first stage-3 table used, and the columns they became.
COLUMN_RENAMES = {
    "outage_start_utc": "non_operations_start_utc",
    "outage_end_utc": "non_operations_end_utc",
    "boundary_before_utc": "before_window_end_utc",
    "boundary_after_utc": "after_window_start_utc",
    "count_before": "lit_reachable_pixels_before",
    "count_after": "lit_reachable_pixels_after",
    "score": "operating_opportunity_score",
}
OLD_TABLE_COLUMNS = tuple({v: k for k, v in COLUMN_RENAMES.items()}.get(c, c) for c in TABLE_COLUMNS)

REPORT_FORMAT = "lunarscout-safe-haven-scored-table-report-v1"


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
    workers: int = 1
    lighting: str = "cpu"

    def as_dict(self) -> dict:
        return {**{k: v for k, v in asdict(self).items() if k not in ("plan", "flood")},
                "pass_one_counts": asdict(self.plan), "flood_counts": asdict(self.flood)}


def _report_document(report: PassTwoReport, *, fingerprint: dict, pipeline, elapsed_seconds: float,
                     reconstructed_note: str | None = None) -> dict:
    """The sidecar report of a finished table: what it holds, how it was made, and the run's counts."""
    from datetime import datetime, timezone

    settings = {k: v for k, v in fingerprint.items() if k != "cells"}
    document = {
        "format": REPORT_FORMAT,
        "table": Path(report.output_path).name,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "what_a_row_is": (
            "a safe haven (a seed pixel) scored by its opportunity to operate before and after the "
            "non-operations period: operating_opportunity_score = lit_reachable_pixels_before + "
            "lit_reachable_pixels_after, the lit pixels that can reach the seed over the before window "
            "(ending at before_window_end_utc) and that it can reach over the after window (starting at "
            "after_window_start_utc), each operational_hours long"),
        "rows": report.rows,
        "columns": list(TABLE_COLUMNS),
        "cells": len(fingerprint["cells"]),
        "blocks_total": report.blocks_total,
        "blocks_resumed_from_a_checkpoint": report.blocks_resumed,
        "elapsed_seconds_of_the_last_run": round(elapsed_seconds, 1),
        "lighting_of_the_last_run": report.lighting,         # a resumed table can mix runs
        "pipeline_of_the_last_run": asdict(pipeline) if pipeline is not None else None,
        "workers_of_the_last_run": report.workers,
        "horizon_patches_decoded_in_the_last_run": report.horizon_loads,
        "peak_decoded_patches_held_in_the_last_run": report.peak_resident_patches,
        "flood_step_minutes": report.flood_step_minutes,
        "flood_steps": report.flood_steps,
        "settings": settings,
        "pass_one_counts": asdict(report.plan),
        "flood_counts": asdict(report.flood),
    }
    if reconstructed_note:
        document["reconstructed"] = reconstructed_note
    return document


def _write_json(path: Path, document: dict) -> None:
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, path)


def rename_table_columns(source, destination, *, overwrite: bool = False) -> int:
    """Copy a stage-3 table made under the first column names to ``destination`` under the current ones
    (``COLUMN_RENAMES``), leaving every value alone. Returns the number of data rows. The source must have
    exactly the old header; the copy goes through a staging file."""
    source, destination = Path(source), Path(destination)
    if destination.exists() and not overwrite:
        raise ValueError(f"{destination} exists; pass overwrite=True to replace it")
    staging = destination.with_name(destination.name + ".partial")
    count = 0
    with source.open(newline="", encoding="utf-8") as handle, staging.open("w", newline="", encoding="utf-8") as out:
        reader = csv.reader(handle)
        header = next(reader, None)
        if tuple(header or ()) != OLD_TABLE_COLUMNS:
            staging.unlink(missing_ok=True)
            raise ValueError(f"{source} does not have the first stage-3 table header")
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(TABLE_COLUMNS)
        for row in reader:
            writer.writerow(row)
            count += 1
    os.replace(staging, destination)
    return count


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
                 earth_elevation_threshold_deg, sunlight_fraction_threshold, merge_gap_hours,
                 light_store=None) -> dict:
    return {
        "lighting_source": "horizons" if light_store is None else store_identity(light_store),
        "parameters": asdict(parameters), "block_patches": block_patches, "flood_steps": flood_steps,
        "times": [int(time.times_us[0]), int(time.times_us[-1]), int(len(time.times_us))],
        "cells": [list(c) for c in cells],
        "stage1": [_file_identity(stage1.hibernation_path), _file_identity(stage1.joint_total_path)],
        "earth_elevation_threshold_deg": earth_elevation_threshold_deg,
        "sunlight_fraction_threshold": sunlight_fraction_threshold,
        "merge_gap_hours": merge_gap_hours,
        "table_columns": list(TABLE_COLUMNS),
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
    horizon_store: HorizonTileStore | None = None,
    light_store=None,
    stage1: Stage1Rasters,
    times: npt.ArrayLike,
    sun_vectors_at: VectorSource | None = None,
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
    workers: int = 1,
    numba_threads_per_worker: int | None = None,
    pipeline=None,
    report_path: str | Path | None = None,
) -> PassTwoReport | None:
    """Run the pass-2 schedule and write the table to ``output_path`` (CSV).

    The lighting comes from ``horizon_store`` (the CPU lightmap of each patch's horizons, with
    ``sun_vectors_at``; the reference route) or from ``light_store`` (a stage-0
    ``LightStoreDirectory``: stored bits for pass 1 and for the lit units, no horizons and no Sun
    vectors). Give exactly one. A store needs the flood step to be its step, its thresholds to be the
    ones given here, and ``times`` on its grid; ``safe_haven_stage3_store`` says what steps outside
    its range mean.

    ``cells`` limits the run to those ``(cell_row, cell_column)`` indices; the default is
    every cell with a haven pixel.  A checkpoint (``<output>.checkpoint.jsonl`` by
    default) is appended after each block, and an existing one with the same
    configuration is resumed.  The CSV is written to a staging file and moved into place
    at the end, so a failure leaves any earlier output alone; the checkpoint is removed
    once the CSV is in place.  ``progress(blocks_done, blocks_total)`` is called once at the start (with the
    blocks a checkpoint already holds) and after each block, so a caller can time
    this run's blocks alone.  ``stop_after_blocks`` ends the run early for tests and
    returns ``None``.

    ``workers`` above 1 runs that many blocks at once in forked worker processes, each with its
    own resident horizons (memory scales with ``workers``; see ``estimate_block_memory_bytes``)
    and ``numba_threads_per_worker`` numba threads (default: the CPU count divided by
    ``workers``).  Only this process writes the checkpoint, as blocks finish in any order, and
    the table is the same for any ``workers`` and after a resume with a different one.  The first
    Ctrl-C stops handing out blocks and waits for the running ones; a second stops at once.

    ``pipeline`` (a ``safe_haven_stage3.PipelineSettings``) runs the blocks through the stage-3
    pipeline instead: planning in processes, lighting, decoding and flooding in threads, joined by
    bounded queues. It replaces ``workers`` (which must then be 1) and gives the same table.

    ``report_path`` (usually the table's name with ``.report.json``) gets a JSON report of the finished
    table: what a row is, the settings and stage-1 files it was made from, the lighting used and the
    counts of the whole run, including blocks resumed from a checkpoint.
    """
    started = wall_clock.monotonic()
    if (horizon_store is None) == (light_store is None):
        raise ValueError("give exactly one of horizon_store and light_store")
    if horizon_store is not None and sun_vectors_at is None:
        raise ValueError("lighting from horizons needs sun_vectors_at")
    output = Path(output_path)
    report_file = None if report_path is None else Path(report_path)
    if report_file is not None:
        if report_file.exists() and not overwrite:
            raise ValueError(f"{report_file} exists; pass overwrite=True to replace it")
        if not report_file.parent.is_dir():
            raise ValueError(f"the directory of {report_file} does not exist")
    if output.exists() and not overwrite:
        raise ValueError(f"{output} exists; pass overwrite=True to replace it")
    if not output.parent.is_dir():
        raise ValueError(f"the directory of {output} does not exist")
    checkpoint = Path(checkpoint_path) if checkpoint_path else output.with_name(output.name + ".checkpoint.jsonl")
    if (stage1.height, stage1.width) != (dem.height, dem.width):
        raise ValueError("the stage-1 products and the DEM differ in size")

    sun_vectors_at = None if light_store is not None else CachedVectors(sun_vectors_at)
    earth_vectors_at = CachedVectors(earth_vectors_at)
    time = TimeContext.build(times, sun_vectors_at, earth_vectors_at)
    if light_store is not None:
        check_run(
            light_store, flood_step_minutes=parameters.flood_step_minutes, times=time.times_us.astype("datetime64[us]"),
            stage1_step_hours=time.step_hours, sunlight_threshold=sunlight_fraction_threshold,
            operating_sunlight_fraction=parameters.operating_sunlight_fraction,
            earth_threshold_deg=earth_elevation_threshold_deg)
    grid = FloodGrid(parameters.flood_step_minutes)
    flood_steps = grid.step_count(parameters.operational_hours)
    halo = halo_pixels(flood_steps * parameters.moves_per_step)
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
        cells = [(e["cell_row"], e["cell_col"]) for e in scan_cells(
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
        light_store=light_store,
    )
    done = _read_checkpoint(checkpoint, fingerprint)
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        checkpoint.write_text(json.dumps({"type": "header", "config": fingerprint}) + "\n", encoding="utf-8")

    months = _month_labels(time)
    transform = np.asarray(dem.geo_transform, dtype=np.float64)
    if int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)

    def load_patch(patch):
        return read_patch_horizons(horizon_store, patch[0] * PATCH_SIZE, patch[1] * PATCH_SIZE, observer_height_m)

    plan_counts, flood_counts = PlanCounts(), PassTwoCounts()
    for entry in done.values():
        _add_counts(plan_counts, entry["plan_counts"])
        _add_counts(flood_counts, entry["flood_counts"])
    pending = [block for block in order if block not in done]
    run_arguments = dict(
        dem=dem, horizon_store=horizon_store, light_store=light_store, stage1=stage1, time=time, grid=grid, parameters=parameters,
        flood_steps=flood_steps, halo=halo, cell_size=cell_size, months=months, transform=transform,
        compiled=compiled, observer_height_m=observer_height_m,
        earth_elevation_threshold_deg=earth_elevation_threshold_deg,
        sunlight_fraction_threshold=sunlight_fraction_threshold, merge_gap_hours=merge_gap_hours,
        sun_vectors_at=sun_vectors_at, earth_vectors_at=earth_vectors_at,
    )
    ran = 0
    if progress is not None:
        progress(len(done), len(order))

    def finish(block, block_plan: dict, block_flood: dict, rows: list) -> None:
        """Record a finished block: checkpoint (flushed to disk), counts, progress."""
        nonlocal ran
        entry = {"type": "block", "block": list(block), "rows": rows,
                 "plan_counts": block_plan, "flood_counts": block_flood}
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

    horizon_loads = peak_resident = 0
    details: dict = {}
    if light_store is not None:
        details["lighting"] = "light store"
    if pipeline is not None:
        if workers != 1:
            raise ValueError("workers must be 1 when a pipeline is given")
        from .safe_haven_stage3 import _run_blocks_in_pipeline

        stopped_early, horizon_loads, peak_resident = _run_blocks_in_pipeline(
            pending, by_block, run_arguments, finish, settings=pipeline, stop_after_blocks=stop_after_blocks,
            details=details,
        )
        if stopped_early:
            return None
    elif workers == 1:
        residency = HorizonResidency(load_patch) if light_store is None else None
        with _graceful_interrupt() as interrupts:
            for block in pending:
                if interrupts or (stop_after_blocks is not None and ran >= stop_after_blocks):
                    return None
                block_plan, block_flood, rows = _run_block(
                    by_block[block], residency=residency, **run_arguments)
                finish(block, asdict(block_plan), asdict(block_flood), rows)
        if residency is not None:
            horizon_loads, peak_resident = residency.loads, residency.peak_resident
    else:
        stopped_early, horizon_loads, peak_resident = _run_blocks_in_workers(
            pending, by_block, run_arguments, finish, workers=workers,
            numba_threads=numba_threads_per_worker, stop_after_blocks=stop_after_blocks,
        )
        if stopped_early:
            return None

    all_rows = [row for entry in done.values() for row in entry["rows"]]
    all_rows.sort(key=lambda r: (r["cell_row"], r["cell_col"], r["month"], r["rank"]))
    staging = output.with_name(output.name + ".partial")
    with staging.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(staging, output)
    result = PassTwoReport(
        blocks_total=len(order), blocks_resumed=len(order) - ran, horizon_loads=horizon_loads,
        peak_resident_patches=peak_resident, flood_step_minutes=grid.step_minutes,
        flood_steps=flood_steps, plan=plan_counts, flood=flood_counts, rows=len(all_rows),
        output_path=str(output), workers=workers, lighting=details.get("lighting", "cpu"),
    )
    if report_file is not None:
        _write_json(report_file, _report_document(
            result, fingerprint=fingerprint, pipeline=pipeline, elapsed_seconds=wall_clock.monotonic() - started))
    checkpoint.unlink()          # last: if the report could not be written, a rerun finishes both
    return result


@contextlib.contextmanager
def _graceful_interrupt():
    """Yield a list that fills with each SIGINT received. The first interrupt only records
    itself (and logs), so the run can finish what is in progress; the second raises
    ``KeyboardInterrupt``. Python's default handler would raise inside whatever compiled numba
    call is running, which surfaces as a ``SystemError``. Outside the main thread nothing is
    installed and the list stays empty."""
    interrupts: list[int] = []
    previous = None
    if threading.current_thread() is threading.main_thread():
        def on_interrupt(signum, frame):
            interrupts.append(signum)
            if len(interrupts) > 1:
                raise KeyboardInterrupt
            logger.warning("Interrupt received: no new blocks will be started; waiting for the running "
                           "blocks to finish (interrupt again to stop at once).")

        previous = signal.signal(signal.SIGINT, on_interrupt)
    try:
        yield interrupts
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


# ---------------------------------------------------------------------------
# Running blocks in worker processes
# ---------------------------------------------------------------------------

#: What a forked worker needs, set by the parent before the workers are created.
_WORKER: dict = {}


def estimate_block_memory_bytes(block_patches: int, halo_patches: int = 1) -> int:
    """Float32 horizons held by a worker for one block: ``(block + 2 * halo)`` patches squared,
    94 MB each.  Windows, the lit store and the DEM come on top (the DEM is shared by the workers)."""
    side = block_patches + 2 * halo_patches
    return side * side * PATCH_SIZE * PATCH_SIZE * 1440 * 4


def _loaded_spice_kernels() -> list[str]:
    """The SPICE kernel files loaded in this process, in load order, without the meta-kernels
    that merely name them (their contents are listed separately). Empty if SPICE was not used."""
    spiceypy = sys.modules.get("spiceypy")
    if spiceypy is None:
        return []
    files = []
    for index in range(spiceypy.ktotal("ALL")):
        name, kind, _source, _handle = spiceypy.kdata(index, "ALL")
        if kind != "META":
            files.append(name)
    return files


def _refurnish_spice(files: list[str]) -> None:
    """Give a forked worker its own open kernel files. A forked process shares the parent's
    file descriptors, and with them the file offsets CSPICE reads through, so workers reading
    the same kernel concurrently get wrong values or ``SPICE(INVALIDRADIUS)``. Clearing the pool
    closes this process's descriptors and loading the files again opens its own."""
    if not files:
        return
    import spiceypy

    spiceypy.kclear()
    for name in files:
        spiceypy.furnsh(name)


def threads_per_worker(workers: int, requested: int | None = None) -> int:
    """Numba threads for each worker: the request, or the CPU count divided among the workers."""
    if requested is not None:
        if int(requested) != requested or requested < 1:
            raise ValueError("numba_threads_per_worker must be a positive integer")
        return int(requested)
    return max(1, (os.cpu_count() or workers) // workers)


def _worker_init(numba_threads: int | None, spice_kernels: list[str]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)        # the parent decides what an interrupt means
    _refurnish_spice(spice_kernels)
    if numba_threads:
        import numba

        numba.set_num_threads(numba_threads)
    state = _WORKER
    state["residency"] = None if state.get("arguments", {}).get("light_store") is not None else HorizonResidency(
        lambda patch: read_patch_horizons(
            state["arguments"]["horizon_store"], patch[0] * PATCH_SIZE, patch[1] * PATCH_SIZE,
            state["arguments"]["observer_height_m"])
    )


def _worker_task(block: tuple[int, int]):
    state = _WORKER
    residency = state["residency"]
    loads_before = 0 if residency is None else residency.loads
    plan_counts, flood_counts, rows = _run_block(
        state["by_block"][block], residency=residency, **state["arguments"])
    if residency is None:
        return block, asdict(plan_counts), asdict(flood_counts), rows, 0, 0
    return block, asdict(plan_counts), asdict(flood_counts), rows, residency.loads - loads_before, residency.peak_resident


def _run_blocks_in_workers(pending, by_block, run_arguments, finish, *, workers, numba_threads,
                           stop_after_blocks):
    """Run ``pending`` blocks in ``workers`` forked processes, calling ``finish`` here for each
    result as it arrives. Returns ``(stopped_early, horizon_loads, peak_resident)``.

    At most ``workers`` blocks are handed out at a time. The first interrupt stops handing out
    blocks and waits for the running ones (their results are kept); a second one terminates the
    workers. A failure in a worker is raised after the blocks that did finish are recorded.
    """
    numba_threads = threads_per_worker(workers, numba_threads)
    _WORKER.clear()
    _WORKER.update(by_block=by_block, arguments=run_arguments)
    context = multiprocessing.get_context("fork")
    results: queue.Queue = queue.Queue()
    pool = context.Pool(workers, initializer=_worker_init,
                        initargs=(numba_threads, _loaded_spice_kernels()))
    horizon_loads = peak_resident = 0
    first_error: BaseException | None = None
    running = 0
    handed_out = 0
    remaining = list(pending)
    try:
        with _graceful_interrupt() as interrupts:
            while True:
                while (running < workers and remaining and not interrupts and first_error is None
                       and (stop_after_blocks is None or handed_out < stop_after_blocks)):
                    block = remaining.pop(0)
                    pool.apply_async(
                        _worker_task, (block,),
                        callback=lambda value: results.put(("ok", value)),
                        error_callback=lambda error, block=block: results.put(("error", (block, error))),
                    )
                    running += 1
                    handed_out += 1
                if running == 0:
                    break
                try:
                    kind, value = results.get(timeout=0.5)
                except queue.Empty:
                    continue
                running -= 1
                if kind == "ok":
                    block, block_plan, block_flood, rows, loads, peak = value
                    horizon_loads += loads
                    peak_resident = max(peak_resident, peak)
                    finish(block, block_plan, block_flood, rows)
                elif first_error is None:
                    first_error = value[1]
                    logger.error("Block %s failed in a worker: %s: %s; waiting for the running blocks.",
                                 value[0], type(value[1]).__name__, value[1])
            if first_error is not None:
                raise first_error
            stopped_early = bool(remaining)
        return stopped_early, horizon_loads, peak_resident
    finally:
        pool.terminate()
        pool.join()
        _WORKER.clear()


def _passable_packed(usable: npt.NDArray[np.bool_], patch_window: tuple[int, int, int, int]):
    """A cell window's usable pixels, padded to whole patches, as packed bits."""
    passable = np.zeros((patch_window[2] * PATCH_SIZE, patch_window[3] * PATCH_SIZE), dtype=bool)
    passable[:usable.shape[0], :usable.shape[1]] = usable
    return pack_bits(passable)


def _flood_record(
    store, record: dict, window, patch_window, passable_packed, flood_counts: PassTwoCounts, *, cell: Cell,
    month_label: str, grid, parameters, flood_steps: int, compiled: bool, cell_size: int, transform,
    dem_height: int, dem_width: int,
) -> list[dict]:
    """Flood every seed of one cell-month from a lit store and return its table rows (the best
    ``max_seeds_per_cell_month``, ranked). Counts go into ``flood_counts``."""
    flood_counts.cell_months_flooded += 1
    results = []
    for seed in record["seeds"]:
        flood_counts.seeds_flooded += 1
        result = flood_seed_from_store(
            store, passable_packed, patch_window,
            (seed["row"] - window[0], seed["column"] - window[2]),
            seed["boundary_before_step"], seed["boundary_after_step"],
            steps=flood_steps, connectivity=parameters.connectivity,
            moves_per_step=parameters.moves_per_step, compiled=compiled,
        )
        if result is None:
            flood_counts.seeds_boundary_mismatch += 1
        elif not result.valid:
            flood_counts.seeds_without_both_sides += 1
        else:
            results.append((seed, result))
    if not results:
        flood_counts.cell_months_without_valid_seed += 1
        return []
    results.sort(key=lambda item: (-item[1].score, item[0]["row"], item[0]["column"]))   # best operating opportunity first
    row0, col0 = cell[0] * cell_size, cell[1] * cell_size
    row1, col1 = min(dem_height, row0 + cell_size), min(dem_width, col0 + cell_size)
    rows = []
    for rank, (seed, result) in enumerate(results[: parameters.max_seeds_per_cell_month], start=1):
        rows.append({
            "cell_row": cell[0], "cell_col": cell[1],
            "x_min_m": float(transform[0] + col0 * transform[1]),
            "y_min_m": float(transform[3] + row1 * transform[5]),
            "x_max_m": float(transform[0] + col1 * transform[1]),
            "y_max_m": float(transform[3] + row0 * transform[5]),
            "month": month_label, "rank": rank,
            "seed_row": seed["row"], "seed_col": seed["column"],
            "seed_x_m": float(transform[0] + (seed["column"] + 0.5) * transform[1]),
            "seed_y_m": float(transform[3] + (seed["row"] + 0.5) * transform[5]),
            "hibernation_hours": seed["hibernation_hours"],
            "non_operations_start_utc": seed["outage_start_utc"],
            "non_operations_end_utc": seed["outage_end_utc"],
            "before_window_end_utc": _utc(grid, seed["boundary_before_step"]),
            "after_window_start_utc": _utc(grid, seed["boundary_after_step"]),
            "lit_reachable_pixels_before": result.count_before,
            "lit_reachable_pixels_after": result.count_after,
            "operating_opportunity_score": result.score,
            "nearby_haven_count": seed["nearby_haven_count"],
            "nearby_pixel_count": seed["nearby_pixel_count"],
            "nearby_haven_fraction": seed["nearby_haven_fraction"],
            "candidate_seeds": len(record["seeds"]), "valid_seeds": len(results),
            "earth_visible_before_h": seed["earth_visible_before_hours"],
            "earth_visible_after_h": seed["earth_visible_after_hours"],
        })
    return rows


def _run_block(
    block_cells_list: list[Cell], *, dem, horizon_store, light_store, stage1, time, grid, parameters, flood_steps,
    halo, cell_size, residency, months, transform, compiled, observer_height_m,
    earth_elevation_threshold_deg, sunlight_fraction_threshold, merge_gap_hours,
    sun_vectors_at, earth_vectors_at,
):
    """One block: pass-1 seeds for its cells, then lit store and floods month by month."""
    report: list = []
    records = list(iter_cell_months(
        dem=dem, horizon_store=horizon_store, light_store=light_store, stage1=stage1,
        times=time.times_us.astype("datetime64[us]"),
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
    if residency is not None:
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
            if light_store is not None:
                counts = fill_lit_store_from_light_store(
                    store, directory=light_store, patches=patches, chunks=record["chunks"])
            else:
                counts = fill_lit_store(
                    store, dem=dem, horizon_store=horizon_store, patches=patches, chunks=record["chunks"],
                    sun_vectors_at=sun_vectors_at,
                    operating_sunlight_fraction=parameters.operating_sunlight_fraction,
                    observer_height_m=observer_height_m, horizons_for=residency.get,
                )
            flood_counts.lit_units_computed += counts.computed
            flood_counts.lit_units_reused += counts.reused
            passable_packed = _passable_packed(np.isfinite(hibernation[month]), patch_window)
            rows.extend(_flood_record(
                store, record, window, patch_window, passable_packed, flood_counts, cell=cell,
                month_label=months[month], grid=grid, parameters=parameters, flood_steps=flood_steps,
                compiled=compiled, cell_size=cell_size, transform=transform, dem_height=dem.height,
                dem_width=dem.width,
            ))
        store.clear()
    flood_counts.rows = len(rows)
    return plan_counts, flood_counts, rows
