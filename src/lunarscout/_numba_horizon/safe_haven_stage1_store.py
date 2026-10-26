"""Stage 1 of the safe-haven analysis fed from the stage-0 light store (private).

Stage 1 normally lights every patch from its horizons. With a light store it reads each patch's three
stored bit-planes instead (Sun fraction at least the shadow threshold, at least the operating
threshold, Earth above the outage elevation) and runs the same per-pixel state machine on them, so a
horizon tile is never decoded. The machine only compares the Sun fraction with two thresholds and Earth's
margin with one, so the planes are turned back into values on the right side of each threshold: a
fraction of 0 (shadow), 0.5 (partial) or 1 (operating), and a margin of -90 or +90 degrees.

The degenerate-outage rule is the one input the planes do not carry: it looks for minima of Earth's
elevation above a flat horizon. That series changes by a few thousandths of a degree across a patch
(2.5 km on a sphere of 1,737 km), so it is computed once, at one pixel near the patch centre, and used for
every pixel of the patch. A minimum that lies within a step of a tie can fall one step differently for a
pixel away from the centre; the rule's effect at that precision is not what stage 1 screens for.

The product's time samples must be the store's grid (same step, a start on the grid and inside the
stored range), and the thresholds given to stage 1 must be the ones the store was made with; otherwise the
planes would answer a different question, and the functions here refuse.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone

import numpy as np
import numpy.typing as npt

from ..errors import InputError
from .file_format import AZIMUTH_COUNT, PATCH_SIZE
from .lightmap_cpu import LightmapCpuSession
from .safe_haven import reduce_safe_haven_products_batches
from .safe_haven_flood_packed import unpack_bits
from .safe_haven_lightstore import LightStore
from .safe_haven_stage0 import LightStoreDirectory
from .safe_haven_windows import FloodGrid

#: Steps unpacked and reduced per call (a multiple of the default chunk of 96). A batch of this many
#: steps is three float32 arrays of 384 x 128 x 128, about 25 MB each.
STEPS_PER_BATCH = 384
PARTIAL_FRACTION = 0.5
#: Steps per device batch: one stored chunk at the default 96 steps, so a batch decodes one chunk.
CUDA_STEPS_PER_BATCH = 96
CLEAR_MARGIN_DEG = 90.0

_INVALID_NODE_M = -20000.0


def store_identity(directory: LightStoreDirectory) -> str:
    """A fingerprint of what the store holds (settings, time range, DEM), for a product's resume check."""
    document = {
        "settings": asdict(directory.settings),
        "first_step": directory.first_step, "step_count": directory.step_count,
        "patch_grid": list(directory.patch_grid), "dem": directory.dem,
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str).encode("ascii")
    return f"light-store-sha256:{hashlib.sha256(payload).hexdigest()}"


def _epoch_microseconds(stamp: datetime | str | np.datetime64) -> int:
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if isinstance(stamp, datetime):
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return int(round(stamp.timestamp() * 1_000_000))
    return int(np.datetime64(stamp, "us").astype(np.int64))


def first_store_step(directory: LightStoreDirectory, timestamps: Sequence[datetime | str],
                     time_step_hours: float) -> int:
    """The global store step of the product's first sample; refuses a grid the store cannot answer.

    The samples must be spaced by the store's step, start on its grid and lie inside its stored range.
    """
    step_minutes = directory.settings.step_minutes
    if abs(time_step_hours * 60.0 - step_minutes) > 1e-6:
        raise InputError("the time step of the product is not the light store's step",
                         code="safe_haven_store_step_mismatch",
                         details={"product_step_minutes": time_step_hours * 60.0, "store_step_minutes": step_minutes})
    grid = FloodGrid(step_minutes)
    first_us = _epoch_microseconds(timestamps[0])
    first = grid.floor_index(first_us)
    if first * grid.step_us != first_us:
        raise InputError("the first time sample is not on the light store's step grid",
                         code="safe_haven_store_time_misaligned", details={"first": str(timestamps[0])})
    last = first + len(timestamps) - 1
    if first < directory.first_step or last >= directory.stop_step:
        raise InputError("the time samples are not inside the light store's range",
                         code="safe_haven_store_time_range",
                         details={"first_step": first, "last_step": last,
                                  "store": [directory.first_step, directory.stop_step - 1]})
    return first


def check_thresholds(directory: LightStoreDirectory, *, sunlight_threshold: float,
                     operating_sunlight_fraction: float, earth_threshold_deg: float) -> None:
    """Refuse thresholds other than those the store's planes were made with."""
    settings = directory.settings
    asked = {"sunlight_threshold": float(sunlight_threshold),
             "operating_sunlight_fraction": float(operating_sunlight_fraction),
             "earth_threshold_deg": float(earth_threshold_deg)}
    stored = {"sunlight_threshold": float(settings.shadow_fraction),
              "operating_sunlight_fraction": float(settings.operating_fraction),
              "earth_threshold_deg": float(settings.earth_elevation_deg)}
    if asked != stored:
        raise InputError("the thresholds differ from those the light store was made with",
                         code="safe_haven_store_thresholds", details={"asked": asked, "stored": stored})


def flat_earth_elevation(dem, earth_vectors_m: npt.ArrayLike, valid_height: int, valid_width: int,
                         tile_y: int, tile_x: int, session: LightmapCpuSession) -> npt.NDArray[np.float32]:
    """Earth's elevation above a flat horizon at the patch's valid pixel nearest its centre, per step."""
    window = np.asarray(dem.elevation_m[tile_y:tile_y + valid_height, tile_x:tile_x + valid_width])
    with np.errstate(invalid="ignore"):
        valid = window > _INVALID_NODE_M
    vectors = np.asarray(earth_vectors_m)
    if not valid.any():
        return np.zeros(len(vectors), dtype=np.float32)
    rows, columns = np.nonzero(valid)
    nearest = int(np.argmin((rows - (valid_height - 1) / 2.0) ** 2 + (columns - (valid_width - 1) / 2.0) ** 2))
    return np.asarray(session.pixel_series(
        dem, np.zeros(AZIMUTH_COUNT, dtype=np.float32), vectors, row=tile_y + int(rows[nearest]),
        column=tile_x + int(columns[nearest]), margin=True, flat_elevation=True), dtype=np.float32)


def _signal_batches(store: LightStore, first_step: int, count: int, valid_height: int, valid_width: int,
                    flat: npt.NDArray[np.float32] | None, check_cancelled) -> Iterator[tuple]:
    """``(fractions, margins, flat)`` batches of ``(steps, valid_height, valid_width)`` float32."""
    done = 0
    while done < count:
        stop = min(done + STEPS_PER_BATCH, count)
        check_cancelled()
        planes = store.read_steps(first_step + done, first_step + stop)
        partial = unpack_bits(planes[0], PATCH_SIZE)[:, :valid_height, :valid_width]
        operating = unpack_bits(planes[1], PATCH_SIZE)[:, :valid_height, :valid_width]
        earth = unpack_bits(planes[2], PATCH_SIZE)[:, :valid_height, :valid_width]
        fractions = np.where(operating, np.float32(1.0),
                             np.where(partial, np.float32(PARTIAL_FRACTION), np.float32(0.0)))
        margins = np.where(earth, np.float32(CLEAR_MARGIN_DEG), np.float32(-CLEAR_MARGIN_DEG))
        batch_flat = None
        if flat is not None:
            batch_flat = np.broadcast_to(flat[done:stop, None, None], fractions.shape)
        yield fractions, margins, batch_flat
        done = stop


def reduce_patch_from_store(
    store: LightStore, first_step: int, time_count: int, valid_height: int, valid_width: int, *,
    flat_elevation: npt.NDArray[np.float32] | None, month_bands, month_index_of,
    sunlight_threshold: float, earth_threshold_deg: float, operating_sunlight_fraction: float,
    merge_gap_hours: float, time_step_hours: float, invalid_pixels, degenerate_window_samples: int,
    check_cancelled=lambda: None,
):
    """The three products of one patch, from its stored planes: ``(hibernation, joint_total, joint_longest)``.

    ``first_step`` is the global store step of the first sample and ``time_count`` the number of samples.
    ``flat_elevation`` is the per-step Earth elevation for the degenerate-outage rule (needed when
    ``degenerate_window_samples`` is positive). ``check_cancelled`` is called before each batch and
    raises to stop the patch.
    """
    window = int(degenerate_window_samples)
    if window > 0 and (flat_elevation is None or len(flat_elevation) != time_count):
        raise InputError("the degenerate-outage rule needs one flat-horizon elevation per time sample",
                         code="safe_haven_store_flat_elevation")
    batches = _signal_batches(store, first_step, time_count, valid_height, valid_width,
                              flat_elevation if window > 0 else None, check_cancelled)
    # The reducer takes the three signals in lockstep, one batch of each per step of its loop.
    for_fractions, for_margins, for_flat = itertools.tee(batches, 3)
    fraction_batches = (batch[0] for batch in for_fractions)
    margin_batches = (batch[1] for batch in for_margins)
    flat_batches = (batch[2] for batch in for_flat) if window > 0 else None
    return reduce_safe_haven_products_batches(
        fraction_batches, margin_batches, time_count, month_bands,
        month_index_of=month_index_of, sunlight_threshold=sunlight_threshold,
        earth_threshold_deg=earth_threshold_deg, operating_sunlight_fraction=operating_sunlight_fraction,
        merge_gap_hours=merge_gap_hours, time_step_hours=time_step_hours, invalid_pixels=invalid_pixels,
        earth_flat_elevation_batches=flat_batches, degenerate_window_samples=window,
    )


__all__ = ["CLEAR_MARGIN_DEG", "CUDA_STEPS_PER_BATCH", "PARTIAL_FRACTION", "STEPS_PER_BATCH", "check_thresholds", "first_store_step",
           "flat_earth_elevation", "reduce_patch_from_store", "store_identity"]
