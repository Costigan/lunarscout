"""Lander sites, stage 1: the longest continuous Sun that starts in each month (private).

A lander that carries no rover stays where it lands, so what matters at a pixel is how long the Sun can
stay up without a break. For every pixel and calendar month this stage gives the duration of the longest
period of continuous Sun that *starts* in that month. It reads only the stage-0 light store.

* Sun is the store's plane for the threshold: Sun fraction at least the threshold. The store holds two
  thresholds, the shadow fraction (0.2) and the operating fraction (1.0, with the stage-1 fuzz of 1e-6);
  the threshold parameter must be one of them, because any other would need the store lit again.
* A period is consecutive steps all in Sun. Gaps are not bridged: one step out of Sun ends a period.
* A period belongs to the UTC calendar month of its first step, and it counts in full even when it runs
  into later months. A month in which no period starts is 0.
* A period already under way at the first step has an unknown start and is credited to no month, as in the
  hibernation product of stage 1. A period still under way at the last step is credited with the length
  seen, which is a lower bound: the last months of the range read low.
* The product is a GeoTIFF with one uint16 band per month holding the duration as a count of steps
  (hours are the count times ``step_minutes / 60``, ``65534`` saturates), and 65535 where there is no
  valid terrain (an invalid DEM node, or a patch without a store file).

Each patch is reduced independently, so the run is resumable (``ResumableTiledProduct``): a rerun does
only the patches the journal does not hold. The reduction walks the packed bits one step at a time with
per-pixel state, so a patch's 35,064 steps are never unpacked at once.
"""

from __future__ import annotations

import collections
import concurrent.futures
import logging
import time as wall_clock
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .file_format import PATCH_SIZE
from .pipeline import enumerate_patches
from .product_store import ProductJob, ResumableTiledProduct
from .safe_haven_lightstore import LightStore, LightStoreSettings, patch_geometry
from .safe_haven_stage0 import LightStoreDirectory, _graceful_interrupt
from .safe_haven_stage1_store import store_identity
from .safe_haven_windows import FloodGrid

logger = logging.getLogger(__name__)

NODATA = 65535
MAX_COUNT = 65534
#: Steps decoded and reduced per read (four 8-day chunks at the default 2-hour step).
STEPS_PER_READ = 384
_PIXELS = PATCH_SIZE * PATCH_SIZE
_PLANE_PARTIAL, _PLANE_OPERATING = 0, 1


@dataclass(frozen=True)
class SunDurationSettings:
    """What to compute (``threshold``, the range) and how a run is carried out."""

    threshold: float = 1.0
    start_utc: str | None = None            # None: the store's first step
    stop_utc: str | None = None             # exclusive; None: the end of the store
    workers: int = 16                       # patches reduced at once, one thread each
    checkpoint_patches: int = 32            # patches written between journal checkpoints
    status_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in ("workers", "checkpoint_patches"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise InputError(f"{name} must be a positive integer", code="lander_sun_setting_invalid",
                                 details={name: value})
        if not 0.0 <= self.threshold <= 1.0:
            raise InputError("the Sun threshold must be between 0 and 1", code="lander_sun_threshold_invalid",
                             details={"threshold": self.threshold})


@dataclass
class SunDurationReport:
    status: str = "completed"
    patches_total: int = 0
    patches_written: int = 0
    patches_already_done: int = 0
    patches_without_store_file: int = 0
    bands: int = 0
    first_step: int = 0
    step_count: int = 0
    elapsed_seconds: float = 0.0
    output_path: str = ""
    notes: list[str] = field(default_factory=list)


def plane_for_threshold(settings: LightStoreSettings, threshold: float) -> int:
    """The store plane that answers "Sun fraction at least ``threshold``", or an error naming the thresholds
    the store holds."""
    if float(threshold) == float(settings.operating_fraction):
        return _PLANE_OPERATING
    if float(threshold) == float(settings.shadow_fraction):
        return _PLANE_PARTIAL
    raise InputError(
        f"the light store holds Sun thresholds {settings.shadow_fraction} and {settings.operating_fraction}, "
        f"not {threshold}; a new threshold needs the store lit again",
        code="lander_sun_threshold_unavailable",
        details={"threshold": threshold, "stored": [settings.shadow_fraction, settings.operating_fraction]})


def resolve_range(directory: LightStoreDirectory, settings: SunDurationSettings) -> tuple[int, int]:
    """``(first_step, stop_step)`` of the product: the store's range, or the part of it asked for."""
    grid = FloodGrid(directory.settings.step_minutes)
    first, stop = directory.first_step, directory.stop_step
    for name, text, pick in (("start_utc", settings.start_utc, "first"), ("stop_utc", settings.stop_utc, "stop")):
        if text is None:
            continue
        microseconds = int(np.datetime64(text.rstrip("Z"), "us").astype(np.int64))
        step = grid.floor_index(microseconds)
        if step * grid.step_us != microseconds:
            raise InputError(f"{name} {text} is not on the store's {directory.settings.step_minutes}-minute grid",
                             code="lander_sun_time_misaligned", details={name: text})
        if pick == "first":
            first = step
        else:
            stop = step
    if not directory.first_step <= first < stop <= directory.stop_step:
        raise InputError("the time range is not inside the light store's range", code="lander_sun_time_range",
                         details={"range": [first, stop], "store": [directory.first_step, directory.stop_step]})
    return first, stop


def month_bands(first_step: int, stop_step: int, step_minutes: int):
    """``(month_of, month_starts)``: the band of every step in the range (``int16``) and the UTC start of
    each band's calendar month as ``datetime64[M]``."""
    grid = FloodGrid(step_minutes)
    stamps = grid.times(np.arange(first_step, stop_step, dtype=np.int64))
    months = stamps.astype("datetime64[M]")
    starts = np.arange(months[0], months[-1] + 1)
    return (months - months[0]).astype(np.int16), starts


# ---------------------------------------------------------------------------
# The reduction
# ---------------------------------------------------------------------------

_KERNELS: dict = {}


def advance_python(planes, first_index, month_of, cur_len, cur_month, best, max_count):
    """Walk ``planes`` ((steps, 128, 2) packed words of one plane) one step at a time.

    ``cur_len[p]`` is the length so far of the period in Sun at pixel ``p`` (0: not in Sun) and
    ``cur_month[p]`` the month it was credited to (-1: no month, its start is unknown). ``best[month, p]``
    is the longest period credited to the month, saturated at ``max_count``. Pixel ``p`` is
    ``row * 128 + column``; bit ``column & 63`` of word ``column >> 6`` of the row holds it, and
    ``p & 63`` is ``column & 63``. Steps are the outer loop so that the words and the state are read in
    order.
    """
    one = np.uint64(1)
    steps = planes.shape[0]
    for k in range(steps):
        index = first_index + k
        month = month_of[index]
        for p in range(planes.shape[1] * 128):
            word = planes[k, p >> 7, (p & 127) >> 6]
            if (word >> np.uint64(p & 63)) & one:
                if cur_len[p] > 0:
                    cur_len[p] += 1
                else:
                    cur_len[p] = 1
                    cur_month[p] = month if index > 0 else -1       # nothing is known before the first step
            elif cur_len[p] > 0:
                if cur_month[p] >= 0:
                    length = cur_len[p] if cur_len[p] < max_count else max_count
                    if length > best[cur_month[p], p]:
                        best[cur_month[p], p] = length
                cur_len[p] = 0


def finish_python(cur_len, cur_month, best, max_count):
    """Credit the periods still under way after the last step (their length is a lower bound)."""
    for p in range(cur_len.shape[0]):
        if cur_len[p] > 0 and cur_month[p] >= 0:
            length = cur_len[p] if cur_len[p] < max_count else max_count
            if length > best[cur_month[p], p]:
                best[cur_month[p], p] = length
        cur_len[p] = 0


def _kernels(compiled: bool = True):
    if not compiled:
        return advance_python, finish_python
    if not _KERNELS:
        from numba import njit

        _KERNELS["advance"] = njit(cache=False, nogil=True)(advance_python)
        _KERNELS["finish"] = njit(cache=False, nogil=True)(finish_python)
    return _KERNELS["advance"], _KERNELS["finish"]


def reduce_patch(
    opened: LightStore, plane: int, first_step: int, stop_step: int, month_of: npt.NDArray[np.int16],
    month_count: int, valid_height: int, valid_width: int, *, compiled: bool = True,
) -> npt.NDArray[np.uint16]:
    """``(month_count, valid_height, valid_width)`` uint16: per month, the longest run of steps (in Sun
    in ``plane``) that starts in it."""
    advance, finish = _kernels(compiled)
    pixels = PATCH_SIZE * PATCH_SIZE
    cur_len = np.zeros(pixels, dtype=np.int32)
    cur_month = np.full(pixels, -1, dtype=np.int16)
    best = np.zeros((month_count, pixels), dtype=np.uint16)
    month_of = np.ascontiguousarray(month_of, dtype=np.int16)
    for start in range(first_step, stop_step, STEPS_PER_READ):
        stop = min(start + STEPS_PER_READ, stop_step)
        planes = np.ascontiguousarray(opened.read_steps(start, stop)[plane])
        advance(planes, start - first_step, month_of, cur_len, cur_month, best, MAX_COUNT)
    finish(cur_len, cur_month, best, MAX_COUNT)
    return np.ascontiguousarray(best.reshape(month_count, PATCH_SIZE, PATCH_SIZE)[:, :valid_height, :valid_width])


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------


def generate_sun_duration(
    output_path: str | Path, *, directory: LightStoreDirectory, dem, georef,
    settings: SunDurationSettings = SunDurationSettings(), overwrite: bool = False, start_fresh: bool = False, patches=None, compiled: bool = True,
) -> SunDurationReport:
    """Write the longest-Sun-period product for every patch of the DEM (``georef`` is its ``GeoReference``;
    or just ``patches``, a list of
    ``(patch_row, patch_column)``; the product is then not finalized).

    Returns a report; its status is ``"completed"``, or ``"interrupted"`` after a first Ctrl-C (the patches
    in flight finish and are kept, and a rerun continues). A patch without a store file is written as
    nodata, as is every pixel on an invalid DEM node.
    """
    started = wall_clock.monotonic()
    plane = plane_for_threshold(directory.settings, settings.threshold)
    first_step, stop_step = resolve_range(directory, settings)
    month_of, month_starts = month_bands(first_step, stop_step, directory.settings.step_minutes)
    report = SunDurationReport(bands=len(month_starts), first_step=first_step, step_count=stop_step - first_step,
                               output_path=str(output_path))
    configuration = {
        "product": "lander-sun-longest-period-v1", "threshold": settings.threshold, "plane": plane,
        "step_minutes": directory.settings.step_minutes, "first_step": first_step, "step_count": stop_step - first_step,
        "units": "steps: hours = value * step_minutes / 60", "nodata": NODATA, "saturates_at": MAX_COUNT,
        "unknown_start": "not credited", "unfinished_at_end": "credited, a lower bound",
    }
    job = ProductJob(
        georef=georef, dtype=np.uint16, band_count=len(month_starts),
        timestamps_utc=tuple(f"{month}-01T00:00:00+00:00" for month in month_starts.astype(str)),
        invalid_value=NODATA, nodata=NODATA, algorithm="lander-sun-longest-period-by-start-month",
        configuration=configuration, horizon_inventory_identity=store_identity(directory))
    product = ResumableTiledProduct(
        output_path, job, overwrite=overwrite, start_fresh=start_fresh, backend="cpu",
        dem_elevation=dem.elevation_m)
    all_patches = enumerate_patches(dem.width, dem.height)
    if patches is not None:
        wanted = {(int(r), int(c)) for r, c in patches}
        all_patches = [p for p in all_patches if (p.tile_y // PATCH_SIZE, p.tile_x // PATCH_SIZE) in wanted]
    report.patches_total = len(all_patches)
    pending = [p for p in all_patches if not product.is_complete(p.tile_y, p.tile_x)]
    report.patches_already_done = len(all_patches) - len(pending)
    logger.info("Lander sun duration: %d patches, %d already done, %d bands, threshold %s, %d workers",
                len(all_patches), report.patches_already_done, report.bands, settings.threshold, settings.workers)

    def reduce(patch):
        opened = directory.open_patch(patch.tile_y // PATCH_SIZE, patch.tile_x // PATCH_SIZE)
        if opened is None:                      # no horizon tile, so never lit: no valid terrain here
            return patch, None
        with opened:
            return patch, reduce_patch(opened, plane, first_step, stop_step, month_of, report.bands,
                                       patch.height, patch.width, compiled=compiled)

    last_status = wall_clock.monotonic()
    done = 0
    in_flight: collections.deque = collections.deque()
    queue = iter(pending)
    limit = 2 * settings.workers
    interrupted = False
    with _graceful_interrupt() as interrupts, concurrent.futures.ThreadPoolExecutor(settings.workers) as pool, \
            product.batch_writer(settings.checkpoint_patches) as writer:
        while True:
            while len(in_flight) < limit and not interrupts:
                patch = next(queue, None)
                if patch is None:
                    break
                in_flight.append(pool.submit(reduce, patch))
            if not in_flight:
                break
            patch, tiles = in_flight.popleft().result()
            if tiles is None:
                writer.write_patch_with_timings(patch.tile_y, patch.tile_x, (), valid=False)
                report.patches_without_store_file += 1
            else:
                writer.write_patch_with_timings(patch.tile_y, patch.tile_x, tiles, valid=True)
            report.patches_written += 1
            done += 1
            now = wall_clock.monotonic()
            if settings.status_seconds > 0 and now - last_status >= settings.status_seconds:
                last_status = now
                rate = done / (now - started)
                logger.info("Lander sun duration: %d of %d patches (%.2f/s, about %.0f min left)",
                            report.patches_already_done + done, report.patches_total, rate,
                            (len(pending) - done) / max(rate, 1e-9) / 60)
        interrupted = bool(interrupts) and done < len(pending)
    if interrupted:
        report.status = "interrupted"
    elif patches is None:
        product.finalize()
    else:
        report.status = "completed"
        report.notes.append("a patch subset was written; the product was not finalized")
    report.elapsed_seconds = wall_clock.monotonic() - started
    return report

