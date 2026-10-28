"""Stage 2 of the rover safe-haven chain: the plan seeds (private).

A seed is one haven pixel in one month, picked from the stage-1 hibernation product, with the timing of
the hibernation that makes it a haven. Stage 3 floods forward from each seed's after boundary; stage 4
chains seeds. This stage only chooses and describes them.

Per 5.12 km cell and month with a haven pixel (hibernation at most the survival limit), up to
``max_seeds_per_cell_month`` seeds are chosen as in the earlier pass 1 (the neighbourhood and spacing
rules are the same), but forward only (``WindowParameters.forward_only``):

* there is no Earth gate: Earth need not be available for 48 hours on either side;
* the after boundary (the first step lit at or after the outage ends, within the search window) is
  required and is where the flood starts;
* the before boundary (the last lit step at or before the outage starts) is kept when there is one: it
  is the step at which a rover coming from an earlier seed must arrive. A seed that is not lit before
  its outage has no before boundary (blank) and cannot be arrived at.

Each seed is labelled with the kind of its outage (``earth_never_sets``: the one-sample outage the
degenerate rule defines where Earth never sets; ``outage_sun_lit``: a real outage no shadow touches;
``outage_in_shadow``: a real outage a shadow run touches) and with its Earth cycle. The cycles are
region-wide: the minima of Earth's elevation above a flat horizon at the raster's centre, one per cycle
(about 27 days apart). A seed belongs to the cycle whose minimum is nearest its outage, so "the next
seed" of a seed is any seed of the next cycle. A seed whose outage is more than ``MAX_CYCLE_OFFSET_DAYS``
from every minimum has cycle -1 and is not linked.

Lighting comes from the stage-0 light store only. The seeds are ordered by month, row and column and
numbered from 1 in that order, so a run's table does not depend on the number of workers.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import multiprocessing
import os
import signal
import time as clock
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .safe_haven_windows import (
    CandidateEvaluator,
    FloodGrid,
    PlanCounts,
    Stage1Rasters,
    TimeContext,
    WindowParameters,
    cell_window,
    halo_pixels,
    select_seeds,
)
from .pixel_coordinates import COORDINATE_COLUMNS, format_coordinate, pixel_centres
from .safe_haven_stage1_store import _INVALID_NODE_M

logger = logging.getLogger(__name__)

#: The table's columns. ``month`` is the UTC month of the outage start ("2027-01"); the boundaries are
#: flood-grid steps (``flood_step_minutes`` since 1970-01-01T00:00:00Z); ``before_step`` is blank for a
#: seed that is not lit before its outage. ``x_m`` and ``y_m`` are the pixel centre in the DEM's CRS.
SEED_COLUMNS = (
    "seed_id", "month", "cell_row", "cell_col", "row", "column", "x_m", "y_m", "hibernation_hours", "outage_kind",
    "earth_cycle", "outage_start_utc", "outage_end_utc", "before_step", "after_step",
)
#: Cycles are told apart by the prominence of Earth's elevation minima, so the small wiggles of the
#: elevation do not count, and are at least this far apart.
MIN_CYCLE_DAYS = 20.0
CYCLE_PROMINENCE_DEG = 1.0
#: Consecutive minima this far apart are one cycle apiece; anything else is refused.
CYCLE_SPACING_DAYS = (20.0, 35.0)
MAX_CYCLE_OFFSET_DAYS = 15.0
_REFERENCE_WINDOW = 512


# ---------------------------------------------------------------------------
# Earth cycles
# ---------------------------------------------------------------------------


def reference_earth_elevation(dem, earth_vectors_m: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Earth's elevation above a flat horizon, per vector, at the valid pixel nearest the raster's centre."""
    from .lightmap_cpu import LightmapCpuSession

    height, width = dem.height, dem.width
    centre_row, centre_column = (height - 1) // 2, (width - 1) // 2
    half = _REFERENCE_WINDOW
    while True:
        row0, row1 = max(0, centre_row - half), min(height, centre_row + half + 1)
        col0, col1 = max(0, centre_column - half), min(width, centre_column + half + 1)
        window = np.asarray(dem.elevation_m[row0:row1, col0:col1])
        with np.errstate(invalid="ignore"):
            valid = window > _INVALID_NODE_M
        if valid.any() or (row0, row1, col0, col1) == (0, height, 0, width):
            break
        half *= 2
    if not valid.any():
        raise InputError("the DEM has no valid pixel", code="plan_dem_empty")
    rows, columns = np.nonzero(valid)
    nearest = int(np.argmin((rows + row0 - centre_row) ** 2 + (columns + col0 - centre_column) ** 2))
    session = LightmapCpuSession(time_batch_size=1)
    return np.asarray(session.pixel_series(
        dem, np.zeros(AZIMUTH_COUNT, dtype=np.float32), np.asarray(earth_vectors_m),
        row=row0 + int(rows[nearest]), column=col0 + int(columns[nearest]), margin=True,
        flat_elevation=True), dtype=np.float32)


def earth_minima(elevation_deg: npt.ArrayLike, times_us: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Times (microseconds since the epoch) of the elevation's cycle minima, one per Earth cycle.

    The series must be regular and long enough to hold at least two minima, and consecutive minima must
    be one cycle apart (``CYCLE_SPACING_DAYS``), or the cycles are not unique and the run is refused.
    """
    from scipy.signal import find_peaks

    elevation = np.asarray(elevation_deg, dtype=np.float64)
    times = np.asarray(times_us, dtype=np.int64)
    if elevation.shape != times.shape or elevation.ndim != 1 or elevation.size < 3:
        raise InputError("the Earth elevation and its times must be one-dimensional and equal in length",
                         code="plan_earth_cycles")
    step_days = float(np.median(np.diff(times))) / 86_400_000_000.0
    found, _ = find_peaks(-elevation, distance=max(1, int(MIN_CYCLE_DAYS / step_days)),
                          prominence=CYCLE_PROMINENCE_DEG)
    minima = times[found]
    if minima.size < 2:
        raise InputError("fewer than two Earth cycles in the time range", code="plan_earth_cycles",
                         details={"minima": int(minima.size)})
    spacing = np.diff(minima) / 86_400_000_000.0
    if spacing.min() < CYCLE_SPACING_DAYS[0] or spacing.max() > CYCLE_SPACING_DAYS[1]:
        raise InputError(
            "Earth's elevation minima are not one per cycle", code="plan_earth_cycles",
            details={"spacing_days_min": float(spacing.min()), "spacing_days_max": float(spacing.max())})
    return minima


def earth_cycle_of(minima_us: npt.NDArray[np.int64], time_us: int) -> int:
    """Index of the minimum nearest ``time_us``, or -1 when none is within ``MAX_CYCLE_OFFSET_DAYS``."""
    distance = np.abs(minima_us - int(time_us))
    nearest = int(np.argmin(distance))
    return nearest if distance[nearest] <= MAX_CYCLE_OFFSET_DAYS * 86_400_000_000 else -1


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


def month_labels(time: TimeContext) -> list[str]:
    """``"YYYY-MM"`` of every month band."""
    first = time.times_us[time.month_first]
    return [str(value.astype("datetime64[M]")) for value in first.astype("datetime64[us]")]


@dataclass
class PlanContext:
    """Everything one cell's planning reads; built once per process."""

    dem: object
    stage1: Stage1Rasters
    time: TimeContext
    make_evaluator: Callable[[], CandidateEvaluator]
    parameters: WindowParameters
    grid: FloodGrid
    minima_us: npt.NDArray[np.int64]
    months: list[str]
    pixel_size_m: float
    cell_size: int
    halo: int
    evaluator: CandidateEvaluator | None = None

    def get_evaluator(self) -> CandidateEvaluator:
        """The evaluator, made on first use so that a forked worker opens its own light-store files."""
        if self.evaluator is None:
            self.evaluator = self.make_evaluator()
        return self.evaluator


def seed_row(seed, cell: tuple[int, int], month: int, context: PlanContext) -> dict:
    """One seed as a table row (without its id)."""
    return {
        "month": context.months[month], "cell_row": cell[0], "cell_col": cell[1], "row": seed.row,
        "column": seed.column, "hibernation_hours": seed.hibernation_hours, "outage_kind": seed.outage_kind,
        "earth_cycle": earth_cycle_of(
            context.minima_us, (seed.outage_start_us + seed.outage_end_us) // 2),
        "outage_start_utc": str(np.datetime64(seed.outage_start_us, "us")) + "Z",
        "outage_end_utc": str(np.datetime64(seed.outage_end_us, "us")) + "Z",
        "before_step": seed.boundary_before_step, "after_step": seed.boundary_after_step,
    }


def plan_cell(context: PlanContext, cell: tuple[int, int]) -> tuple[list[dict], PlanCounts]:
    """The seed rows of one cell, over every month, and the counters.

    A cell with no haven pixel in any month is skipped after reading only its own pixels' hibernation.
    """
    parameters, stage1, time = context.parameters, context.stage1, context.time
    counts = PlanCounts()
    counts.cell_months_examined += time.month_count
    window, core = cell_window(cell[0], cell[1], cell_size=context.cell_size, halo=context.halo,
                               height=context.dem.height, width=context.dem.width)
    row0, col0 = cell[0] * context.cell_size, cell[1] * context.cell_size
    has_haven = stage1.read_haven_window(
        row0, min(stage1.height, row0 + context.cell_size), col0, min(stage1.width, col0 + context.cell_size),
        parameters.survival_limit_hours).any(axis=(1, 2))
    if not has_haven.any():
        return [], counts
    hibernation, joint_total = stage1.read_window(*window)
    rows: list[dict] = []
    for month in np.flatnonzero(has_haven):
        seeds = select_seeds(
            hibernation_hours=hibernation[month], joint_total_hours=joint_total[month],
            window_origin=(window[0], window[2]), core=core, month=int(month), evaluator=context.get_evaluator(),
            grid=context.grid, pixel_size_m=context.pixel_size_m, expected=stage1.expected, counts=counts)
        rows.extend(seed_row(seed, cell, int(month), context) for seed in seeds)
    return rows, counts


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class PlanReport:
    """What the run found."""

    seeds: int
    cells: int
    cells_with_seeds: int
    months: int
    cell_months_with_seeds: int
    by_kind: dict
    seeds_without_before_boundary: int
    seeds_without_cycle: int
    earth_cycles: int
    counts: PlanCounts
    parameters: WindowParameters
    seconds: float = 0.0
    output_path: str = ""

    def as_dict(self) -> dict:
        record = asdict(self)
        record["counts"] = asdict(self.counts)
        record["parameters"] = asdict(self.parameters)
        return record


_PLAN: dict = {}


def _plan_init() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _plan_task(cell: tuple[int, int]):
    rows, counts = plan_cell(_PLAN["context"], cell)
    return cell, rows, asdict(counts)


def _write_table(path: Path, rows: list[dict], geo_transform) -> None:
    staging = path.with_name(path.name + ".partial")
    try:
        with staging.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(SEED_COLUMNS)
            for number, row in enumerate(rows, start=1):
                x, y = pixel_centres(geo_transform, row["row"], row["column"])
                writer.writerow([
                    number, row["month"], row["cell_row"], row["cell_col"], row["row"], row["column"],
                    format_coordinate(float(x)), format_coordinate(float(y)), f"{row['hibernation_hours']:g}", row["outage_kind"], row["earth_cycle"],
                    row["outage_start_utc"], row["outage_end_utc"],
                    "" if row["before_step"] is None else row["before_step"], row["after_step"]])
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def plan_seeds(
    output_path: str | Path,
    *,
    dem,
    stage1: Stage1Rasters,
    light_store,
    times: npt.ArrayLike,
    earth_vectors_at,
    parameters: WindowParameters,
    earth_elevation_threshold_deg: float = 2.0,
    sunlight_fraction_threshold: float = 0.2,
    merge_gap_hours: float = 24.0,
    cells: list[tuple[int, int]] | None = None,
    workers: int = 1,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> PlanReport:
    """Write the seed table of the stage-1 products and return the report.

    ``light_store`` is a stage-0 ``LightStoreDirectory``: the pixel lighting is its stored bits and its
    step must be the flood step and its thresholds the ones given here. ``cells`` lists
    ``(cell_row, cell_col)`` to plan (a trial); the default is every cell. Nothing is written when
    ``output_path`` exists and ``overwrite`` is false. ``workers`` processes plan cells in parallel;
    the table is the same for any number.
    """
    from .safe_haven_stage3_store import StoreSignals, check_run

    started = clock.monotonic()
    output_path = Path(output_path)
    report_path = output_path.with_name(output_path.name + ".report.json")
    for path in (output_path, report_path):
        if path.exists() and not overwrite:
            raise InputError(f"{path} exists; pass overwrite=True to replace it", code="plan_output_exists",
                             details={"path": str(path)})
    if not parameters.forward_only:
        raise InputError("stage 2 plans forward-only seeds: set WindowParameters(forward_only=True)",
                         code="plan_not_forward_only")
    if (stage1.height, stage1.width) != (dem.height, dem.width):
        raise InputError("the stage-1 products and the DEM differ in size", code="plan_grid_mismatch")
    pixel_size = abs(float(dem.geo_transform[1]))
    if not math.isclose(pixel_size, abs(float(dem.geo_transform[5])), rel_tol=1e-6):
        raise InputError("planning needs square pixels", code="plan_grid_mismatch")
    time = TimeContext.build(times, None, earth_vectors_at)
    if time.month_count != stage1.band_count:
        raise InputError(f"the times span {time.month_count} months but the products have "
                         f"{stage1.band_count} bands", code="plan_grid_mismatch")
    check_run(light_store, flood_step_minutes=parameters.flood_step_minutes,
              times=time.times_us.astype("datetime64[us]"), stage1_step_hours=time.step_hours,
              sunlight_threshold=sunlight_fraction_threshold,
              operating_sunlight_fraction=parameters.operating_sunlight_fraction,
              earth_threshold_deg=earth_elevation_threshold_deg)
    minima = earth_minima(reference_earth_elevation(dem, time.earth_vectors), time.times_us)
    grid = FloodGrid(parameters.flood_step_minutes)
    flood_steps = grid.step_count(parameters.operational_hours)
    cell_size = parameters.cell_patches * PATCH_SIZE
    def make_evaluator() -> CandidateEvaluator:
        return CandidateEvaluator(
            StoreSignals(light_store, dem), time, parameters,
            earth_elevation_threshold_deg=earth_elevation_threshold_deg,
            sunlight_fraction_threshold=sunlight_fraction_threshold, merge_gap_hours=merge_gap_hours)

    context = PlanContext(
        dem=dem, stage1=stage1, time=time, make_evaluator=make_evaluator, parameters=parameters, grid=grid,
        minima_us=minima, months=month_labels(time), pixel_size_m=pixel_size, cell_size=cell_size,
        halo=halo_pixels(flood_steps * parameters.moves_per_step))
    if cells is None:
        cells = [(r, c) for r in range(-(-dem.height // cell_size)) for c in range(-(-dem.width // cell_size))]
    cells = [(int(r), int(c)) for r, c in cells]

    rows: list[dict] = []
    counts = PlanCounts()
    cells_with_seeds = 0
    results = None
    pool = None
    try:
        if workers > 1:
            _PLAN["context"] = context
            pool = multiprocessing.get_context("fork").Pool(workers, initializer=_plan_init)
            results = pool.imap_unordered(_plan_task, cells, chunksize=1)
        else:
            def serial():
                for cell in cells:
                    cell_rows, cell_counts = plan_cell(context, cell)
                    yield cell, cell_rows, asdict(cell_counts)
            results = serial()
        for done, (cell, cell_rows, cell_counts) in enumerate(results, start=1):
            rows.extend(cell_rows)
            cells_with_seeds += bool(cell_rows)
            counts.add(PlanCounts(**cell_counts))
            if progress is not None:
                progress(done, len(cells))
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        _PLAN.clear()

    rows.sort(key=lambda r: (r["month"], r["row"], r["column"]))
    _write_table(output_path, rows, dem.geo_transform)
    report = PlanReport(
        seeds=len(rows), cells=len(cells), cells_with_seeds=cells_with_seeds, months=time.month_count,
        cell_months_with_seeds=len({(r["cell_row"], r["cell_col"], r["month"]) for r in rows}),
        by_kind=dict(sorted(Counter(r["outage_kind"] for r in rows).items())),
        seeds_without_before_boundary=sum(r["before_step"] is None for r in rows),
        seeds_without_cycle=sum(r["earth_cycle"] < 0 for r in rows), earth_cycles=int(minima.size),
        counts=counts, parameters=parameters, seconds=clock.monotonic() - started,
        output_path=str(output_path))
    report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return report


def read_seed_table(path: str | Path) -> list[dict]:
    """The table written by :func:`plan_seeds`, one dict per seed with numbers as ints (``before_step``
    is ``None`` where blank) and ``hibernation_hours`` as a float."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header not in (SEED_COLUMNS, tuple(c for c in SEED_COLUMNS if c not in COORDINATE_COLUMNS)):
            raise InputError(f"{path} is not a seed table", code="plan_table_unreadable",
                             details={"header": reader.fieldnames})   # a table made before x_m, y_m is read too
        rows = []
        for record in reader:
            row = dict(record)
            for name in ("seed_id", "cell_row", "cell_col", "row", "column", "earth_cycle", "after_step"):
                row[name] = int(row[name])
            row["before_step"] = int(row["before_step"]) if row["before_step"] else None
            row["hibernation_hours"] = float(row["hibernation_hours"])
            for name in COORDINATE_COLUMNS:
                if name in row:
                    row[name] = float(row[name])
            rows.append(row)
    return rows
