"""Check safe havens against a horizon traced by ray casting (private).

The safe-haven products rest on two things that nothing re-derives: the horizon tiles (made by the CUDA horizon
generator from fitted rays) and the light store of stage 0 (made from those tiles). This module re-derives both for
a few pixels with ``direct_reference_trace``'s geometry (exact spherical slopes along straight rays, no fitted
segments, hierarchy or near-field shortcut) and compares:

* **the lighting.** For a window around a seed's outage it lights the pixel three ways, with the patch kernel's own
  arithmetic (``LightmapCpuSession.pixel_series``): from the traced reference horizon, from the production horizon
  tile, and it reads the stored bits. Stored against production-tile lighting tests stage 0 (lighting and packing);
  production-tile against reference lighting tests the horizons as the lighting sees them; stored against reference
  is the whole chain. A stored bit that the reference does not reproduce is *explained* when it lies between the
  reference bits for a horizon shifted down and up by a tolerance (a lower horizon lights more, so the bits are
  monotone in the shift): the horizon generator is not exact, and the tolerance says how inexact it may be.
* **the haven score.** ``CandidateEvaluator``, the code stage 2 ran on the store, runs on the reference lighting and
  on the stored lighting, and both are compared with the seed's table row (score, outage start and end, kind).

Nothing here uses CUDA. Importing the module does nothing.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time as wall_clock
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ..errors import InputError, ProductStorageError
from .file_format import AZIMUTH_COUNT, PATCH_SIZE, HorizonTileStore
from .geometry import DemGrid, enu_to_moon_matrix, inverse_stereographic, lat_lon_to_vector, vector_to_lat_lon
from .lightmap_cpu import LightmapCpuSession
from .safe_haven_lightstore import LightStoreSettings, patch_geometry, plane_thresholds
from .safe_haven_stage0 import LightStoreDirectory
from .safe_haven_stage1_store import _INVALID_NODE_M, flat_earth_elevation
from .safe_haven_plan import month_labels
from .safe_haven_stage3_store import StoreSignals
from .safe_haven_windows import CandidateEvaluator, PixelSignals, TimeContext, WindowParameters

logger = logging.getLogger(__name__)

REPORT_FORMAT = "lunarscout-safe-haven-lighting-check-v1"
AZIMUTH_STEP_DEG = 360.0 / AZIMUTH_COUNT
HORIZON_LIMIT_DEG = 50.0                         # stored horizons are clamped to this
PLANES = ("lit_partial", "lit_full", "earth_visible")
FLAT_BASES = ("patch_centre", "pixel")
REGIONS = ("all", "upper_right_edges")
CHAIN_COLUMNS = ("seed_id", "month", "row", "column", "hibernation_hours", "outage_kind", "outage_start_utc",
                 "outage_end_utc", "chain_length")


# ---------------------------------------------------------------------------
# The reference horizon
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceSettings:
    """How densely a ray is sampled: ``near_step_fraction`` of a pixel out to ``near_range_m``, then
    ``far_step_fraction`` of a pixel (the defaults are converged: on rugged terrain a quarter-pixel step near the
    observer put the horizon 0.5 degrees below a 0.5 m step, a 0.05 pixel step is within 0.1 of it). A ray ends where it leaves the DEM, at the DEM's ray limit (1.2 times its
    smaller side) or at ``max_range_m``, the horizon generator's maximum distance."""

    near_range_m: float = 5000.0
    near_step_fraction: float = 0.05
    far_step_fraction: float = 0.25
    chunk_samples: int = 16384
    max_range_m: float = 1_000_000.0

    def __post_init__(self) -> None:
        if not (self.near_step_fraction > 0 and self.far_step_fraction > 0 and self.chunk_samples >= 1
                and self.near_range_m >= 0 and self.max_range_m > 0):
            raise InputError("the trace settings must be positive", code="check_trace_settings_invalid")


def _project(latitude: npt.NDArray, longitude: npt.NDArray, parameters) -> tuple[npt.NDArray, npt.NDArray]:
    """``project_stereographic`` for arrays."""
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_origin, cos_origin = np.sin(parameters.latitude_origin_rad), np.cos(parameters.latitude_origin_rad)
    delta = longitude - parameters.longitude_origin_rad
    denominator = 1.0 + sin_origin * sin_lat + cos_origin * cos_lat * np.cos(delta)
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    scale = 2.0 * parameters.scale * parameters.radius_m / denominator
    x = scale * cos_lat * np.sin(delta) + parameters.false_easting_m
    y = scale * (cos_origin * sin_lat - sin_origin * cos_lat * np.cos(delta)) + parameters.false_northing_m
    return x, y


def _crs_to_pixel(dem: DemGrid, x: npt.NDArray, y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """``DemGrid.crs_to_pixel`` for arrays."""
    transform = dem.geo_transform
    determinant = transform[1] * transform[5] - transform[2] * transform[4]
    dx, dy = x - transform[0], y - transform[3]
    column = (transform[5] * dx - transform[2] * dy) / determinant
    row = (-transform[4] * dx + transform[1] * dy) / determinant
    return column - 0.5, row - 0.5


def _bilinear(dem: DemGrid, column: npt.NDArray, row: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """``DemGrid.elevation`` for arrays (columns and rows inside the DEM), and a mask of the samples whose four
    surrounding nodes are all valid."""
    x1, y1 = column.astype(np.int64), row.astype(np.int64)
    x2, y2 = np.minimum(x1 + 1, dem.width - 1), np.minimum(y1 + 1, dem.height - 1)
    q11, q12 = dem.elevation_m[y1, x1].astype(np.float64), dem.elevation_m[y2, x1].astype(np.float64)
    q21, q22 = dem.elevation_m[y1, x2].astype(np.float64), dem.elevation_m[y2, x2].astype(np.float64)
    height = (q11 * (x2 - column) * (y2 - row) + q21 * (column - x1) * (y2 - row)
              + q12 * (x2 - column) * (row - y1) + q22 * (column - x1) * (row - y1))
    with np.errstate(invalid="ignore"):
        valid = np.minimum(np.minimum(q11, q12), np.minimum(q21, q22)) > _INVALID_NODE_M
    return height, valid & np.isfinite(height)


def trace_ray(dem: DemGrid, observer_vector_m: npt.ArrayLike, direction_moon_centered: npt.ArrayLike,
              distances_m: npt.ArrayLike, rotation: npt.NDArray | None = None) -> npt.NDArray[np.float64]:
    """``direct_reference_trace`` for all the distances at once: rows ``(distance, column, row, elevation, slope)``
    up to the first sample outside the DEM, which is the first with a column or row at or past the last node: the
    horizon tracer stops there too, and ``DemGrid.elevation`` is 0 in that last strip (both bilinear weights vanish
    when the second node is clamped to the first), which would read as a wall at the edge. A sample with an invalid node among the four that surround it has a NaN
    slope. ``rotation`` is the observer's ``enu_to_moon_matrix`` when the caller has it."""
    observer = np.asarray(observer_vector_m, dtype=np.float64)
    direction = np.asarray(direction_moon_centered, dtype=np.float64)
    distances = np.asarray(distances_m, dtype=np.float64)
    if rotation is None:
        rotation = enu_to_moon_matrix(*vector_to_lat_lon(observer))
    walker = observer + direction * distances[:, None]
    longitude = np.arctan2(walker[:, 1], walker[:, 0])
    longitude = np.where(longitude < 0.0, longitude + 2.0 * np.pi, longitude)
    latitude = np.arctan2(walker[:, 2], np.hypot(walker[:, 0], walker[:, 1]))
    x, y = _project(latitude, longitude, dem.projection)
    column, row = _crs_to_pixel(dem, x, y)
    inside = (column >= 0.0) & (column < dem.width - 1) & (row >= 0.0) & (row < dem.height - 1)
    if not inside.all():
        count = int(np.argmin(inside))
        distances, latitude, longitude, column, row = (a[:count] for a in (distances, latitude, longitude, column, row))
    if column.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    elevation, valid = _bilinear(dem, column, row)
    radius = dem.projection.radius_m + elevation
    cos_lat = np.cos(latitude)
    surface = np.stack((radius * cos_lat * np.cos(longitude), radius * cos_lat * np.sin(longitude),
                        radius * np.sin(latitude)), axis=1)
    local = (surface - observer) @ rotation.T
    slope = local[:, 2] / np.hypot(local[:, 0], local[:, 1])
    slope = np.where(valid, slope, np.nan)
    return np.column_stack((distances, column, row, elevation, slope))


def _distance_chunks(pixel_size_m: float, settings: TraceSettings, start_m: float = 1.0, limit_m: float | None = None):
    """Sample distances along a ray in chunks from ``start_m`` to ``limit_m``: dense near the observer (the first
    ``near_range_m``), coarser beyond."""
    block = settings.chunk_samples
    limit = settings.max_range_m if limit_m is None else min(limit_m, settings.max_range_m)
    near_step = pixel_size_m * settings.near_step_fraction
    far_step = pixel_size_m * settings.far_step_fraction
    near = np.arange(start_m, min(settings.near_range_m, limit), near_step)
    for first in range(0, near.size, block):
        yield near[first:first + block]
    first_far = max(settings.near_range_m, start_m)
    done = 0
    while True:
        distances = first_far + far_step * (done + np.arange(block))
        distances = distances[distances <= limit]
        if distances.size == 0:
            return
        yield distances
        done += block


def _ray_maximum_slope(dem: DemGrid, observer: npt.NDArray, rotation: npt.NDArray, direction: npt.NDArray,
                       settings: TraceSettings, start_m: float = 1.0,
                       limit_m: float | None = None) -> tuple[float, float]:
    """``(largest slope, distance reached)`` of a ray through one DEM from ``start_m``: the distance of the last
    sample inside the DEM, or ``start_m`` when the ray starts outside it."""
    maximum, reached = -np.inf, start_m
    for distances in _distance_chunks(dem.map_resolution_m, settings, start_m, limit_m):
        samples = trace_ray(dem, observer, direction, distances, rotation)
        if samples.shape[0]:
            reached = float(samples[-1, 0])
            slopes = samples[:, 4]
            slopes = slopes[~np.isnan(slopes)]
            if slopes.size:
                maximum = max(maximum, float(slopes.max()))
        if samples.shape[0] < distances.size:
            break                                         # the ray left the DEM
    return maximum, reached


def dem_ray_limit_m(dem: DemGrid, settings: TraceSettings) -> float:
    """How far a ray is traced through a DEM: 1.2 times its smaller side, and no more than the maximum distance."""
    return min(settings.max_range_m, 1.2 * min(dem.width, dem.height) * dem.map_resolution_m)


def observer_vector(dem: DemGrid, row: int, column: int, observer_height_m: float = 0.0) -> npt.NDArray[np.float64]:
    """The Moon-centred position of the observer at a pixel centre, ``observer_height_m`` above the surface."""
    latitude, longitude = inverse_stereographic(*dem.pixel_to_crs(column, row), dem.projection)
    return lat_lon_to_vector(latitude, longitude, dem.projection.radius_m + float(dem.elevation_m[row, column])
                             + float(observer_height_m))


def reference_horizon(dems: Sequence[DemGrid], row: int, column: int, *, observer_height_m: float = 0.0,
                      azimuth_indices: npt.ArrayLike | None = None,
                      settings: TraceSettings = TraceSettings()) -> npt.NDArray[np.float32]:
    """The horizon elevation in degrees at the pixel ``(row, column)`` of ``dems[0]`` for azimuth bins
    ``azimuth_indices`` (default all 1,440: ``0.25 * k`` degrees clockwise from north).

    Each bin is the largest exact spherical slope along the ray, clamped to the stored range of plus and minus 50
    degrees. The ray runs through the DEMs in turn as in the horizon generator: through the first from 1 m to where
    it leaves the DEM (or its ray limit), through each later DEM from where the ray left the one before, so a later
    DEM adds only the terrain beyond the earlier ones.
    """
    primary = dems[0]
    if not (0 <= row < primary.height and 0 <= column < primary.width):
        raise InputError("the pixel lies outside the DEM", code="check_pixel_outside", details={"row": row, "column": column})
    observer = observer_vector(primary, row, column, observer_height_m)
    rotation = enu_to_moon_matrix(*vector_to_lat_lon(observer))
    indices = np.arange(AZIMUTH_COUNT) if azimuth_indices is None else np.asarray(azimuth_indices, dtype=np.int64)
    result = np.empty(indices.size, dtype=np.float32)
    for position, index in enumerate(indices):
        azimuth = np.deg2rad(float(index) * AZIMUTH_STEP_DEG)
        direction = np.array((np.sin(azimuth), np.cos(azimuth), 0.0)) @ rotation
        direction = direction / np.linalg.norm(direction)
        slope, start = -np.inf, 1.0
        for dem in dems:
            limit = dem_ray_limit_m(dem, settings)
            largest, reached = _ray_maximum_slope(dem, observer, rotation, direction, settings, start, limit)
            slope = max(slope, largest)
            start = min(limit, reached)
            if start >= settings.max_range_m:
                break
        result[position] = np.float32(np.clip(np.degrees(np.arctan(slope)), -HORIZON_LIMIT_DEG, HORIZON_LIMIT_DEG))
    return result


# ---------------------------------------------------------------------------
# Signals from the reference horizon
# ---------------------------------------------------------------------------


class ReferenceSignals:
    """A pixel's Sun fraction and Earth margin from a traced reference horizon (``PixelSignals``' interface).

    The horizon of a pixel is traced once and kept. The lighting arithmetic is the patch kernel's own, so a
    difference from the stored lighting comes from the horizon. ``flat_basis`` says where Earth's elevation above a
    flat horizon (for the degenerate-outage rule) is taken: ``"patch_centre"``, the pixel nearest the centre of the
    pixel's patch, is what stages 1 to 3 used; ``"pixel"`` is the pixel's own. ``shifted(delta)`` is the same
    signals with every horizon raised by ``delta`` degrees.
    """

    def __init__(self, dems: Sequence[DemGrid], *, observer_height_m: float = 0.0, flat_basis: str = "patch_centre",
                 trace: TraceSettings = TraceSettings(), _shared=None, _offset_deg: float = 0.0) -> None:
        if flat_basis not in FLAT_BASES:
            raise InputError(f"flat_basis must be one of {FLAT_BASES}", code="check_flat_basis_invalid",
                             details={"flat_basis": flat_basis})
        self._dems = list(dems)
        self._dem = self._dems[0]
        self._observer_height = float(observer_height_m)
        self._flat_basis = flat_basis
        self._trace = trace
        self._offset = float(_offset_deg)
        if _shared is None:
            _shared = (LightmapCpuSession(time_batch_size=1), {})
        self._session, self._horizons = _shared

    def shifted(self, delta_deg: float) -> "ReferenceSignals":
        return ReferenceSignals(self._dems, observer_height_m=self._observer_height, flat_basis=self._flat_basis,
                                trace=self._trace, _shared=(self._session, self._horizons),
                                _offset_deg=self._offset + float(delta_deg))

    def horizon_column(self, row: int, column: int) -> npt.NDArray[np.float32]:
        key = (int(row), int(column))
        if key not in self._horizons:
            self._horizons[key] = reference_horizon(self._dems, *key, observer_height_m=self._observer_height,
                                                    settings=self._trace)
        horizon = self._horizons[key]
        return horizon if self._offset == 0.0 else (horizon + np.float32(self._offset)).astype(np.float32)

    def series(self, row: int, column: int, vectors_m: npt.ArrayLike, *, margin: bool, flat_elevation: bool = False,
               times_us: npt.ArrayLike | None = None) -> npt.NDArray[np.float32]:
        row, column = int(row), int(column)
        if flat_elevation and self._flat_basis == "patch_centre":
            tile_y, tile_x, valid_height, valid_width = patch_geometry(
                self._dem, row // PATCH_SIZE, column // PATCH_SIZE)
            return flat_earth_elevation(self._dem, vectors_m, valid_height, valid_width, tile_y, tile_x, self._session)
        horizon = (np.zeros(AZIMUTH_COUNT, dtype=np.float32) if flat_elevation else self.horizon_column(row, column))
        return self._session.pixel_series(self._dem, horizon, vectors_m, row=row, column=column, margin=margin,
                                          flat_elevation=flat_elevation)


# ---------------------------------------------------------------------------
# Lighting comparison
# ---------------------------------------------------------------------------


def plane_bits(fraction: npt.ArrayLike, margin: npt.ArrayLike, store: LightStoreSettings) -> npt.NDArray[np.bool_]:
    """The three store planes, ``(3, steps)`` booleans, of a Sun fraction and an Earth margin series: the
    thresholds and comparison of ``light_patch_planes`` (a float64 value at least the threshold)."""
    shadow, operating, earth = plane_thresholds(store)
    fraction = np.asarray(fraction, dtype=np.float64)
    margin = np.asarray(margin, dtype=np.float64)
    return np.stack((fraction >= shadow, fraction >= operating, margin >= earth))


def stored_bits(stored: StoreSignals, row: int, column: int, times_us: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """The planes as stored for a pixel, ``(3, steps)`` booleans, read through ``StoreSignals``."""
    sun = stored.series(row, column, None, margin=False, times_us=times_us)
    earth = stored.series(row, column, None, margin=True, times_us=times_us)
    if sun is None or earth is None:
        raise ProductStorageError(f"pixel ({row}, {column}) has no light-store file", code="check_no_stored_lighting",
                                  details={"row": row, "column": column})
    sun = np.asarray(sun, dtype=np.float64)
    return np.stack((sun >= 0.5, sun >= 1.0, np.asarray(earth, dtype=np.float64) > 0.0))


@dataclass
class PlaneComparison:
    """One plane compared over a window."""

    plane: str
    steps: int
    stored_set: int
    stored_vs_production: int = 0              # stage 0: the stored bits against the production tile's lighting
    production_vs_reference: int = 0           # the horizons: production tile's lighting against the reference's
    stored_vs_reference: int = 0               # the whole chain
    unexplained_vs_reference: int = 0          # of those, not reproduced by the reference with the horizon shifted
    unexplained_vs_production: int = 0         # stored_vs_production beyond the arithmetic tolerance
    worst: list[dict] = field(default_factory=list)      # the first unexplained differences

    @property
    def ok(self) -> bool:
        return self.unexplained_vs_reference == 0 and self.unexplained_vs_production == 0


def compare_lighting(
    stored: npt.NDArray[np.bool_],
    reference: tuple[npt.NDArray, npt.NDArray],
    production: tuple[npt.NDArray, npt.NDArray] | None,
    low: tuple[npt.NDArray, npt.NDArray],
    high: tuple[npt.NDArray, npt.NDArray],
    store: LightStoreSettings,
    *,
    times_us: npt.ArrayLike,
    arithmetic_tolerance: float = 1e-3,
    detail_limit: int = 20,
) -> list[PlaneComparison]:
    """Compare the stored planes with lighting from the reference horizon (``(fraction, margin)`` series),
    from the production tile (or ``None``), and with the reference horizon lowered (``low``) and raised
    (``high``) by the tolerance. A bit is explained when it lies between the reference bits for ``high`` and
    ``low``. A difference from the production tile is an arithmetic tie, not an error, when the value is within
    ``arithmetic_tolerance`` (a fraction, or degrees) of the threshold."""
    ref_bits = plane_bits(*reference, store)
    low_bits, high_bits = plane_bits(*low, store), plane_bits(*high, store)
    prod_bits = None if production is None else plane_bits(*production, store)
    thresholds = plane_thresholds(store)
    values = (np.asarray(reference[0], dtype=np.float64), np.asarray(reference[0], dtype=np.float64),
              np.asarray(reference[1], dtype=np.float64))
    production_values = None if production is None else (
        np.asarray(production[0], dtype=np.float64), np.asarray(production[0], dtype=np.float64),
        np.asarray(production[1], dtype=np.float64))
    stamps = np.asarray(times_us, dtype=np.int64).astype("datetime64[us]")
    result = []
    for plane, name in enumerate(PLANES):
        item = PlaneComparison(name, int(stored.shape[1]), int(stored[plane].sum()))
        differs = stored[plane] != ref_bits[plane]
        explained = (stored[plane] >= high_bits[plane]) & (stored[plane] <= low_bits[plane])
        item.stored_vs_reference = int(differs.sum())
        bad = differs & ~explained
        item.unexplained_vs_reference = int(bad.sum())
        if prod_bits is not None:
            item.production_vs_reference = int((prod_bits[plane] != ref_bits[plane]).sum())
            gap = stored[plane] != prod_bits[plane]
            item.stored_vs_production = int(gap.sum())
            tie = np.abs(production_values[plane] - thresholds[plane]) <= arithmetic_tolerance
            item.unexplained_vs_production = int((gap & ~tie).sum())
        for step in np.flatnonzero(bad)[:detail_limit]:
            item.worst.append({"time": str(stamps[step]) + "Z", "stored": bool(stored[plane, step]),
                               "reference_value": float(values[plane][step]), "threshold": float(thresholds[plane])})
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# What a pixel's evaluation says
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """What ``CandidateEvaluator.evaluate`` gave for a pixel and month: a haven record or the reason it has none."""

    reason: str = ""                       # "" for a record
    score_hours: float = math.nan
    outage_start_utc: str = ""
    outage_end_utc: str = ""
    kind: str = ""
    outage_first: int = -1
    outage_last: int = -1

    @property
    def has_record(self) -> bool:
        return not self.reason

    def is_haven(self, survival_limit_hours: float) -> bool:
        return self.has_record and self.score_hours <= survival_limit_hours

    def key(self) -> tuple:
        return (self.reason, self.score_hours, self.outage_start_utc, self.outage_end_utc, self.kind)


def _stamp(microseconds: int) -> str:
    return str(np.datetime64(int(microseconds), "us")) + "Z"


def evaluate_outcome(evaluator: CandidateEvaluator, row: int, column: int, month: int) -> Outcome:
    found = evaluator.evaluate(row, column, month)
    if isinstance(found, str):
        return Outcome(reason=found)
    return Outcome(score_hours=float(found.score_hours), outage_start_utc=_stamp(found.outage_start_us),
                   outage_end_utc=_stamp(found.outage_end_us), kind=found.outage_kind,
                   outage_first=int(found.outage_first), outage_last=int(found.outage_last))


def table_outcome(row: dict) -> Outcome:
    """The seed table's own values for a seed, as an :class:`Outcome`."""
    return Outcome(score_hours=float(row["hibernation_hours"]), outage_start_utc=row["outage_start_utc"],
                   outage_end_utc=row["outage_end_utc"], kind=row["outage_kind"])


def judge_outcome(expected: Outcome, reference: Outcome, low: Outcome, high: Outcome, step_hours: float,
                  survival_limit_hours: float, *, haven: bool) -> str:
    """``match`` when the reference gives the expected score, outage and kind; ``within_tolerance`` when the
    expected values lie within what the reference gives with the horizon lowered and raised, give or take one time
    step (the samples are ``step_hours`` apart); ``differs`` otherwise.

    For a seed (``haven``) the expected outcome is its table row; for a non-haven it is "no haven": the reference
    must find no haven, and ``within_tolerance`` means a shifted horizon would make it one.
    """
    if not haven:
        if not reference.is_haven(survival_limit_hours):
            return "match" if not (low.is_haven(survival_limit_hours) or high.is_haven(survival_limit_hours)) \
                else "within_tolerance"
        return "within_tolerance" if not (low.is_haven(survival_limit_hours)
                                          and high.is_haven(survival_limit_hours)) else "differs"
    if reference.key()[1:] == expected.key()[1:] and reference.has_record:
        return "match"
    records = [o for o in (low, reference, high) if o.has_record]
    if not records or not expected.has_record:
        return "differs"
    scores = [o.score_hours for o in records]
    if not min(scores) - step_hours - 1e-9 <= expected.score_hours <= max(scores) + step_hours + 1e-9:
        return "differs"
    for name in ("outage_start_utc", "outage_end_utc"):
        stamps = np.array([np.datetime64(getattr(o, name).rstrip("Z"), "us") for o in records])
        wanted = np.datetime64(getattr(expected, name).rstrip("Z"), "us")
        slack = np.timedelta64(int(round(step_hours * 3_600_000_000)), "us")
        if not stamps.min() - slack <= wanted <= stamps.max() + slack:
            return "differs"
    return "within_tolerance"


# ---------------------------------------------------------------------------
# Choosing pixels
# ---------------------------------------------------------------------------


def read_chain_table(path: str | Path) -> list[dict]:
    """The columns of the final chain table that the check needs, one dict per seed."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in CHAIN_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            raise InputError(f"{path} lacks the columns {missing}", code="check_table_unreadable",
                             details={"path": str(path), "missing": missing})
        rows = []
        for record in reader:
            rows.append({"seed_id": int(record["seed_id"]), "month": record["month"], "row": int(record["row"]),
                         "column": int(record["column"]), "hibernation_hours": float(record["hibernation_hours"]),
                         "outage_kind": record["outage_kind"], "outage_start_utc": record["outage_start_utc"],
                         "outage_end_utc": record["outage_end_utc"], "chain_length": int(record["chain_length"])})
    return rows


def region_filter(region: str, height: int, width: int, band_pixels: int = 256):
    """A function ``(row, column) -> bool`` for the pixels of a region of a DEM ``height`` by ``width``.

    ``"all"``: every pixel. ``"upper_right_edges"``: the upper-right quadrant (the top half of the rows, the right
    half of the columns) within ``band_pixels`` of the top edge or of the right edge, an L-shaped strip."""
    if region not in REGIONS:
        raise InputError(f"region must be one of {REGIONS}", code="check_region_invalid", details={"region": region})
    if band_pixels < 1:
        raise InputError("band_pixels must be positive", code="check_region_invalid", details={"band_pixels": band_pixels})
    if region == "all":
        return lambda row, column: True
    return lambda row, column: (row < height // 2 and column >= width // 2
                                and (row < band_pixels or column >= width - band_pixels))


def _score_class(hours: float) -> str:
    return "0" if hours <= 0 else ("1-24" if hours <= 24 else ">24")


def _chain_class(length: int) -> str:
    return "1" if length <= 1 else ("2-5" if length <= 5 else "6+")


def strata_of(row: dict) -> dict[str, str]:
    """The properties a sample is spread over."""
    return {"kind": row["outage_kind"], "score": _score_class(row["hibernation_hours"]),
            "year": row["month"][:4], "chain": _chain_class(row["chain_length"])}


def stratified_sample(rows: Sequence[dict], count: int, *, random_seed: int = 0, pool: int = 4000) -> list[dict]:
    """``count`` seeds spread over the outage kinds, score classes, years and chain lengths of the table.

    Each pick takes, from a random pool of the rows left, the row whose strata are the least represented so far
    (ties broken at random), so every stratum is reached before any is repeated. The result is the same for the
    same table and ``random_seed``."""
    if count < 1 or not rows:
        return []
    rng = np.random.default_rng(random_seed)
    count = min(count, len(rows))
    taken: set[int] = set()
    seen = {name: Counter() for name in ("kind", "score", "year", "chain")}
    picked: list[dict] = []
    while len(picked) < count:
        candidates = [int(i) for i in rng.choice(len(rows), size=min(pool, len(rows)), replace=False)
                      if int(i) not in taken]
        if not candidates:
            break
        costs = []
        for index in candidates:
            strata = strata_of(rows[index])
            costs.append(sum(seen[name][value] for name, value in strata.items()) + rng.random() * 0.5)
        best = candidates[int(np.argmin(costs))]
        taken.add(best)
        picked.append(rows[best])
        for name, value in strata_of(rows[best]).items():
            seen[name][value] += 1
    return picked


def find_negative(evaluator: CandidateEvaluator, seed: dict, month: int, survival_limit_hours: float, *,
                  max_radius: int = 40, height: int, width: int, allowed=None) -> tuple[int, int, Outcome] | None:
    """The nearest pixel to the seed that has a hibernation in the month and is not a haven (its score is above the
    survival limit), searched in rings out to ``max_radius`` pixels among the pixels ``allowed(row, column)``
    accepts (default: all); ``None`` when there is none."""
    row0, column0 = seed["row"], seed["column"]
    for radius in range(2, max_radius + 1):
        ring = [(row0 + dr, column0 + dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)
                if max(abs(dr), abs(dc)) == radius]
        for row, column in sorted(ring, key=lambda p: (p[0] - row0) ** 2 + (p[1] - column0) ** 2):
            if not (0 <= row < height and 0 <= column < width) or (allowed is not None and not allowed(row, column)):
                continue
            try:
                outcome = evaluate_outcome(evaluator, row, column, month)
            except InputError:
                continue
            if outcome.has_record and outcome.score_hours > survival_limit_hours:
                return row, column, outcome
    return None


# ---------------------------------------------------------------------------
# One pixel
# ---------------------------------------------------------------------------


@dataclass
class PixelCheck:
    """The result of checking one pixel in one month."""

    kind: str                                  # "haven" (a table seed) or "non_haven"
    seed_id: int
    month: str
    row: int
    column: int
    strata: dict = field(default_factory=dict)
    horizon: dict = field(default_factory=dict)
    window: dict = field(default_factory=dict)
    planes: list[PlaneComparison] = field(default_factory=list)
    expected: Outcome = field(default_factory=Outcome)
    stored: Outcome = field(default_factory=Outcome)
    reference: Outcome = field(default_factory=Outcome)
    reference_low: Outcome = field(default_factory=Outcome)
    reference_high: Outcome = field(default_factory=Outcome)
    outcome_verdict: str = ""                  # match, within_tolerance, differs
    stored_reproduces_table: bool = True
    explaining_shift_deg: float | None = None  # when the lighting is not reproduced: the smallest shift that does
    seconds: float = 0.0

    @property
    def lighting_ok(self) -> bool:
        return all(plane.ok for plane in self.planes)

    @property
    def ok(self) -> bool:
        return self.lighting_ok and self.outcome_verdict != "differs" and self.stored_reproduces_table


@dataclass(frozen=True)
class CheckSettings:
    """How the comparison is made."""

    window_days: float = 15.0                  # the lighting is compared this long either side of the outage
    horizon_tolerance_deg: float = 0.3         # the horizon generator's error on real terrain (worst ray 0.29,
                                               # docs/numba-horizon-terrain-interpolation.md)
    explaining_shifts_deg: tuple = (0.1, 0.25, 0.5, 1.0, 2.0)    # larger shifts tried for a pixel that fails
    arithmetic_tolerance: float = 1e-3         # stored against the same horizon: how near a threshold a tie is
    near_horizon_deg: float = 3.0              # horizon bins that matter: a body this close to the horizon there
    flat_basis: str = "patch_centre"
    trace: TraceSettings = field(default_factory=TraceSettings)

    def __post_init__(self) -> None:
        if not (self.window_days > 0 and self.horizon_tolerance_deg >= 0 and self.arithmetic_tolerance >= 0):
            raise InputError("the check settings must be non-negative", code="check_settings_invalid")


@dataclass
class CheckContext:
    """Everything a pixel's check reads, made once."""

    dems: list
    time: TimeContext
    parameters: WindowParameters
    store_directory: LightStoreDirectory
    horizon_store: HorizonTileStore | None
    settings: CheckSettings
    months: list[str]
    earth_threshold_deg: float = 2.0
    sunlight_fraction_threshold: float = 0.2
    merge_gap_hours: float = 24.0
    observer_height_m: float = 0.0

    def __post_init__(self) -> None:
        dem = self.dems[0]
        self.reference = ReferenceSignals(self.dems, observer_height_m=self.observer_height_m,
                                          flat_basis=self.settings.flat_basis, trace=self.settings.trace)
        self.shifted_down = self.reference.shifted(-self.settings.horizon_tolerance_deg)
        self.shifted_up = self.reference.shifted(+self.settings.horizon_tolerance_deg)
        self.stored = StoreSignals(self.store_directory, dem)
        self.production = (None if self.horizon_store is None
                           else PixelSignals(dem, self.horizon_store, self.observer_height_m))
        self._evaluators = {}

    def evaluator(self, name: str) -> CandidateEvaluator:
        if name not in self._evaluators:
            signals = {"stored": self.stored, "reference": self.reference, "low": self.shifted_down,
                       "high": self.shifted_up}[name]
            self._evaluators[name] = CandidateEvaluator(
                signals, self.time, self.parameters, earth_elevation_threshold_deg=self.earth_threshold_deg,
                sunlight_fraction_threshold=self.sunlight_fraction_threshold, merge_gap_hours=self.merge_gap_hours)
        return self._evaluators[name]

    def close(self) -> None:
        self.stored.close()


def _body_bins(dem: DemGrid, row: int, column: int, vectors: npt.NDArray, observer_height_m: float):
    """``(azimuth_bin, elevation_deg)`` of a body at each vector, seen from the pixel."""
    observer = observer_vector(dem, row, column, observer_height_m)
    rotation = enu_to_moon_matrix(*vector_to_lat_lon(observer))
    local = (np.asarray(vectors, dtype=np.float64) - observer) @ rotation.T
    norm = np.linalg.norm(local, axis=1)
    azimuth = np.degrees(np.arctan2(local[:, 0], local[:, 1])) % 360.0
    elevation = np.degrees(np.arcsin(np.clip(local[:, 2] / norm, -1.0, 1.0)))
    return np.rint(azimuth / AZIMUTH_STEP_DEG).astype(np.int64) % AZIMUTH_COUNT, elevation


def horizon_statistics(context: CheckContext, row: int, column: int, lo: int, hi: int) -> dict:
    """The reference horizon against the production tile's: over all azimuths and over the bins where the Sun or
    Earth is near the horizon in the window (``near_horizon_deg``). ``{}`` when there is no production tile."""
    reference = context.reference.horizon_column(row, column).astype(np.float64)
    out = {"reference_min_deg": float(reference.min()), "reference_max_deg": float(reference.max())}
    if context.production is None:
        return out
    tile = context.production.horizon_column(row, column)
    if tile is None:
        return out
    difference = np.asarray(tile, dtype=np.float64) - reference           # production minus reference
    out.update(all_mean_abs_deg=float(np.abs(difference).mean()), all_max_abs_deg=float(np.abs(difference).max()),
               all_mean_signed_deg=float(difference.mean()))
    relevant = np.zeros(AZIMUTH_COUNT, dtype=bool)
    for vectors in (context.time.sun_vectors, context.time.earth_vectors):
        bins, elevation = _body_bins(context.dems[0], row, column, vectors[lo:hi], context.observer_height_m)
        near = np.abs(elevation - reference[bins]) <= context.settings.near_horizon_deg
        relevant[bins[near]] = True
    if relevant.any():
        out.update(relevant_bins=int(relevant.sum()), relevant_max_abs_deg=float(np.abs(difference[relevant]).max()),
                   relevant_mean_signed_deg=float(difference[relevant].mean()))
    return out


def check_pixel(context: CheckContext, row: int, column: int, month: int, *, kind: str, seed_id: int = 0,
                expected: Outcome | None = None, strata: dict | None = None) -> PixelCheck:
    """Check one pixel in one month: its lighting around the stored evaluator's outage, and its score."""
    started = wall_clock.monotonic()
    time, settings, parameters = context.time, context.settings, context.parameters
    result = PixelCheck(kind=kind, seed_id=seed_id, month=context.months[month], row=row, column=column,
                        strata=strata or {}, expected=expected or Outcome())
    result.stored = evaluate_outcome(context.evaluator("stored"), row, column, month)
    if kind == "haven" and expected is not None:
        result.stored_reproduces_table = result.stored.key()[1:] == expected.key()[1:] and result.stored.has_record
    count = len(time.times_us)
    pad = int(math.ceil(settings.window_days * 24.0 / time.step_hours))
    if result.stored.has_record:
        first, last = result.stored.outage_first, result.stored.outage_last
    else:
        first, last = int(time.month_first[month]), int(time.month_last[month])
    lo, hi = max(0, first - pad), min(count, last + 1 + pad)
    stamps = time.times_us[lo:hi]
    result.window = {"first_utc": _stamp(int(stamps[0])), "last_utc": _stamp(int(stamps[-1])), "steps": int(hi - lo)}
    result.horizon = horizon_statistics(context, row, column, lo, hi)

    def series(signals):
        sun = signals.series(row, column, time.sun_vectors[lo:hi], margin=False, times_us=stamps)
        earth = signals.series(row, column, time.earth_vectors[lo:hi], margin=True, times_us=stamps)
        return sun, earth

    production = None
    if context.production is not None and context.production.horizon_column(row, column) is not None:
        production = series(context.production)
    result.planes = compare_lighting(
        stored_bits(context.stored, row, column, stamps), series(context.reference), production,
        series(context.shifted_down), series(context.shifted_up), context.store_directory.settings,
        times_us=stamps, arithmetic_tolerance=settings.arithmetic_tolerance)
    if not result.lighting_ok:
        stored_planes = stored_bits(context.stored, row, column, stamps)
        reference_series = series(context.reference)
        for shift in sorted(s for s in settings.explaining_shifts_deg if s > settings.horizon_tolerance_deg):
            wider = compare_lighting(
                stored_planes, reference_series, None, series(context.reference.shifted(-shift)),
                series(context.reference.shifted(+shift)), context.store_directory.settings, times_us=stamps)
            if all(plane.unexplained_vs_reference == 0 for plane in wider):
                result.explaining_shift_deg = float(shift)
                break
    result.reference = evaluate_outcome(context.evaluator("reference"), row, column, month)
    result.reference_low = evaluate_outcome(context.evaluator("low"), row, column, month)
    result.reference_high = evaluate_outcome(context.evaluator("high"), row, column, month)
    haven = kind == "haven"
    result.outcome_verdict = judge_outcome(
        expected if haven and expected is not None else Outcome(), result.reference, result.reference_low,
        result.reference_high, time.step_hours, parameters.survival_limit_hours, haven=haven)
    result.seconds = wall_clock.monotonic() - started
    return result


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------


@dataclass
class CheckReport:
    format: str = REPORT_FORMAT
    seeds_checked: int = 0
    non_havens_checked: int = 0
    non_havens_not_found: int = 0
    lighting_ok: int = 0
    lighting_failed: int = 0
    verdicts: dict = field(default_factory=dict)
    stored_not_reproducing_table: int = 0
    explained_by_shift: dict = field(default_factory=dict)
    plane_totals: dict = field(default_factory=dict)
    horizon: dict = field(default_factory=dict)
    passed: bool = False
    seconds: float = 0.0
    settings: dict = field(default_factory=dict)
    pixels: list[dict] = field(default_factory=list)


def summarize(results: Sequence[PixelCheck], report: CheckReport) -> None:
    report.lighting_ok = sum(r.lighting_ok for r in results)
    report.lighting_failed = len(results) - report.lighting_ok
    report.verdicts = dict(Counter(f"{r.kind}:{r.outcome_verdict}" for r in results))
    report.stored_not_reproducing_table = sum(not r.stored_reproduces_table for r in results)
    shifts = Counter()
    for r in results:
        if not r.lighting_ok:
            shifts["not explained" if r.explaining_shift_deg is None else f"<= {r.explaining_shift_deg:g} deg"] += 1
    report.explained_by_shift = dict(shifts)
    totals = {name: Counter() for name in PLANES}
    for result in results:
        for plane in result.planes:
            for key in ("steps", "stored_set", "stored_vs_production", "production_vs_reference",
                        "stored_vs_reference", "unexplained_vs_reference", "unexplained_vs_production"):
                totals[plane.plane][key] += getattr(plane, key)
    report.plane_totals = {name: dict(counter) for name, counter in totals.items()}
    relevant = [r.horizon["relevant_max_abs_deg"] for r in results if "relevant_max_abs_deg" in r.horizon]
    signed = [r.horizon["relevant_mean_signed_deg"] for r in results if "relevant_mean_signed_deg" in r.horizon]
    if relevant:
        report.horizon = {"pixels": len(relevant), "relevant_max_abs_deg": float(max(relevant)),
                          "relevant_median_of_max_abs_deg": float(np.median(relevant)),
                          "relevant_mean_signed_deg": float(np.mean(signed))}
    report.passed = all(r.ok for r in results) and bool(results)
    report.pixels = [_pixel_dict(r) for r in results]


def _json_safe(value):
    """``value`` with NaN and infinity (not JSON) replaced by ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _pixel_dict(result: PixelCheck) -> dict:
    document = asdict(result)
    document["ok"] = result.ok
    return _json_safe(document)


def _write_text(path: Path, text: str) -> None:
    staging = path.with_name(path.name + f".partial-{os.getpid()}")
    try:
        staging.write_text(text, encoding="utf-8", newline="\n")
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


CSV_COLUMNS = ("kind", "seed_id", "month", "row", "column", "outage_kind", "table_score_hours", "stored_score_hours",
               "reference_score_hours", "outcome_verdict", "stored_reproduces_table", "steps",
               "stored_vs_production", "production_vs_reference", "stored_vs_reference", "unexplained_vs_reference",
               "unexplained_vs_production", "horizon_relevant_max_abs_deg", "horizon_relevant_mean_signed_deg", "ok")


def write_csv(path: Path, results: Sequence[PixelCheck]) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for r in results:
        sums = {key: sum(getattr(p, key) for p in r.planes) for key in (
            "steps", "stored_vs_production", "production_vs_reference", "stored_vs_reference",
            "unexplained_vs_reference", "unexplained_vs_production")}
        writer.writerow([
            r.kind, r.seed_id, r.month, r.row, r.column, r.strata.get("kind", r.stored.kind),
            "" if r.kind != "haven" else f"{r.expected.score_hours:g}", f"{r.stored.score_hours:g}" if r.stored.has_record else r.stored.reason,
            f"{r.reference.score_hours:g}" if r.reference.has_record else r.reference.reason, r.outcome_verdict,
            r.stored_reproduces_table, sums["steps"], sums["stored_vs_production"], sums["production_vs_reference"],
            sums["stored_vs_reference"], sums["unexplained_vs_reference"], sums["unexplained_vs_production"],
            f"{r.horizon['relevant_max_abs_deg']:.4f}" if "relevant_max_abs_deg" in r.horizon else "",
            f"{r.horizon['relevant_mean_signed_deg']:.4f}" if "relevant_mean_signed_deg" in r.horizon else "",
            r.ok])
    _write_text(path, buffer.getvalue())


def write_markdown(path: Path, report: CheckReport, results: Sequence[PixelCheck]) -> None:
    lines = [
        "# Safe-haven lighting check", "",
        f"Result: **{'PASS' if report.passed else 'FAIL'}**. {report.seeds_checked} seeds and "
        f"{report.non_havens_checked} non-haven pixels checked in {report.seconds / 60:.1f} minutes.", "",
        f"* Lighting reproduced for {report.lighting_ok} of {len(results)} pixels "
        f"(a stored bit is explained when the reference reproduces it with the horizon shifted by "
        f"{report.settings.get('horizon_tolerance_deg')} degrees).",
        (f"* Pixels whose lighting needs a larger horizon shift to be reproduced: {report.explained_by_shift}."
         if report.explained_by_shift else "* Every pixel's lighting is reproduced within the tolerance."),
        f"* Haven score and outage: {report.verdicts}.",
        f"* The stored lighting reproduces the table row for all but {report.stored_not_reproducing_table} seeds.", "",
        "## Lighting planes, summed over the sample", "",
        "| plane | steps | stored set | stored vs tile | tile vs reference | stored vs reference | unexplained (ref) | unexplained (tile) |",
        "|---|---|---|---|---|---|---|---|"]
    for name, totals in report.plane_totals.items():
        lines.append(f"| {name} | {totals.get('steps', 0)} | {totals.get('stored_set', 0)} | "
                     f"{totals.get('stored_vs_production', 0)} | {totals.get('production_vs_reference', 0)} | "
                     f"{totals.get('stored_vs_reference', 0)} | {totals.get('unexplained_vs_reference', 0)} | "
                     f"{totals.get('unexplained_vs_production', 0)} |")
    if report.horizon:
        lines += ["", "## Horizon, production tile minus reference, where the Sun or Earth is near the horizon", "",
                  f"Over {report.horizon['pixels']} pixels: largest difference {report.horizon['relevant_max_abs_deg']:.3f} "
                  f"degrees, median of the per-pixel largest {report.horizon['relevant_median_of_max_abs_deg']:.3f}, "
                  f"mean signed {report.horizon['relevant_mean_signed_deg']:+.4f}."]
    failing = [r for r in results if not r.ok]
    if failing:
        lines += ["", "## Pixels that did not pass", ""]
        for r in failing:
            lines.append(f"* {r.kind} seed {r.seed_id} {r.month} pixel ({r.row}, {r.column}): outcome {r.outcome_verdict}, "
                         f"lighting {'ok' if r.lighting_ok else 'NOT reproduced'}, "
                         f"stored reproduces table {r.stored_reproduces_table}")
    _write_text(path, "\n".join(lines) + "\n")


def run_check(
    output_directory: str | Path,
    *,
    dems: Sequence[DemGrid],
    light_store: LightStoreDirectory,
    horizon_store: HorizonTileStore | None,
    chain_rows: Sequence[dict],
    times: npt.ArrayLike,
    sun_vectors_at,
    earth_vectors_at,
    parameters: WindowParameters,
    seeds: int = 40,
    non_havens: int = 8,
    random_seed: int = 0,
    settings: CheckSettings = CheckSettings(),
    earth_elevation_threshold_deg: float = 2.0,
    sunlight_fraction_threshold: float = 0.2,
    merge_gap_hours: float = 24.0,
    observer_height_m: float = 0.0,
    region: str = "all",
    edge_band_pixels: int = 256,
    overwrite: bool = False,
    progress=None,
) -> CheckReport:
    """Check a stratified sample of seeds and some nearby non-havens; write ``lighting_check.json``, ``.csv`` and
    ``.md`` in ``output_directory``. Nothing is written when one exists and ``overwrite`` is false. ``region`` (see
    :func:`region_filter`) limits the seeds and the non-havens to part of the DEM."""
    output_directory = Path(output_directory)
    paths = {suffix: output_directory / f"lighting_check.{suffix}" for suffix in ("json", "csv", "md")}
    for path in paths.values():
        if path.exists() and not overwrite:
            raise InputError(f"{path} exists; pass overwrite=True to replace it", code="check_output_exists",
                             details={"path": str(path)})
    if seeds < 1 or non_havens < 0:
        raise InputError("seeds must be positive and non_havens not negative", code="check_counts_invalid")
    output_directory.mkdir(parents=True, exist_ok=True)
    started = wall_clock.monotonic()
    time = TimeContext.build(times, sun_vectors_at, earth_vectors_at)
    months = month_labels(time)
    context = CheckContext(list(dems), time, parameters, light_store, horizon_store, settings, months,
                           earth_elevation_threshold_deg, sunlight_fraction_threshold, merge_gap_hours,
                           observer_height_m)
    allowed = region_filter(region, context.dems[0].height, context.dems[0].width, edge_band_pixels)
    in_region = [r for r in chain_rows if allowed(r["row"], r["column"])]
    if not in_region:
        raise InputError("no seed of the chain table lies in the region", code="check_region_empty",
                         details={"region": region, "edge_band_pixels": edge_band_pixels})
    if progress:
        progress(f"{len(in_region)} of {len(chain_rows)} seeds lie in the region {region!r}")
    picked = stratified_sample(in_region, seeds, random_seed=random_seed)
    report = CheckReport(settings={**asdict(settings), "seeds": seeds, "non_havens": non_havens,
                                   "random_seed": random_seed, "survival_limit_hours": parameters.survival_limit_hours,
                                   "region": region, "edge_band_pixels": edge_band_pixels,
                                   "seeds_in_region": len(in_region)})
    results: list[PixelCheck] = []
    try:
        for number, seed in enumerate(picked, start=1):
            month = months.index(seed["month"])
            results.append(check_pixel(context, seed["row"], seed["column"], month, kind="haven", seed_id=seed["seed_id"],
                                       expected=table_outcome(seed), strata=strata_of(seed)))
            report.seeds_checked += 1
            if progress:
                progress(f"seed {number}/{len(picked)}: {results[-1].month} pixel ({seed['row']}, {seed['column']}) "
                         f"lighting {'ok' if results[-1].lighting_ok else 'NOT reproduced'}, "
                         f"score {results[-1].outcome_verdict} ({results[-1].seconds:.0f} s)")
        negatives_from = [picked[i] for i in np.random.default_rng(random_seed + 1).permutation(len(picked))]
        for seed in negatives_from:
            if report.non_havens_checked >= non_havens:
                break
            month = months.index(seed["month"])
            found = find_negative(context.evaluator("stored"), seed, month, parameters.survival_limit_hours,
                                  height=context.dems[0].height, width=context.dems[0].width, allowed=allowed)
            if found is None:
                report.non_havens_not_found += 1
                continue
            row, column, _outcome = found
            results.append(check_pixel(context, row, column, month, kind="non_haven", seed_id=seed["seed_id"],
                                       strata=strata_of(seed)))
            report.non_havens_checked += 1
            if progress:
                progress(f"non-haven {report.non_havens_checked}/{non_havens}: {results[-1].month} pixel ({row}, {column}) "
                         f"lighting {'ok' if results[-1].lighting_ok else 'NOT reproduced'}, "
                         f"score {results[-1].outcome_verdict} ({results[-1].seconds:.0f} s)")
    finally:
        context.close()
    report.seconds = wall_clock.monotonic() - started
    summarize(results, report)
    _write_text(paths["json"], json.dumps(_json_safe(asdict(report)), indent=1) + "\n")
    write_csv(paths["csv"], results)
    write_markdown(paths["md"], report, results)
    return report


__all__ = [
    "CheckContext", "CheckReport", "CheckSettings", "Outcome", "PixelCheck", "PlaneComparison", "ReferenceSignals",
    "TraceSettings", "check_pixel", "compare_lighting", "evaluate_outcome", "find_negative", "judge_outcome",
    "observer_vector", "plane_bits", "read_chain_table", "reference_horizon", "region_filter", "run_check", "stored_bits",
    "stratified_sample", "table_outcome", "trace_ray",
]
