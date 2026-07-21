from __future__ import annotations

import csv
import json
import types
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from ..errors import MapAlgebraDTypeError, MapAlgebraError
from ..raster import Raster
from ._validation import _require_common_grid

StatisticName = Literal[
    "count", "valid_count", "invalid_count", "sum", "mean",
    "min", "max", "range", "std", "variance", "median",
    "p25", "p75", "p90",
]

_VALID_STATS = frozenset({
    "count", "valid_count", "invalid_count", "sum", "mean",
    "min", "max", "range", "std", "variance", "median",
    "p25", "p75", "p90",
})

_DEFAULT_STATS: tuple[StatisticName, ...] = (
    "count", "invalid_count", "valid_count", "sum", "mean",
    "min", "max", "range", "std", "variance", "median",
)

_INTEGER_STATS = frozenset({"count", "valid_count", "invalid_count"})


@dataclass(frozen=True, slots=True)
class ZonalStatistics:
    zone_ids: np.ndarray
    columns: tuple[str, ...]
    values: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    units: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_ids": [int(z) for z in self.zone_ids],
            "columns": list(self.columns),
            "values": {k: [int(x) if k in _INTEGER_STATS else float(x) for x in v]
                        for k, v in self.values.items()},
            "valid": {k: v.tolist() for k, v in self.valid.items()},
            "units": {k: v for k, v in self.units.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_records(self) -> tuple[types.MappingProxyType[str, Any], ...]:
        records: list[dict[str, Any]] = []
        for i, zid in enumerate(self.zone_ids):
            rec: dict[str, Any] = {"zone": int(zid)}
            for col in self.columns:
                if self.valid[col][i]:
                    v = self.values[col][i]
                    rec[col] = int(v) if col in _INTEGER_STATS else float(v)
                else:
                    rec[col] = None
            records.append(rec)
        return tuple(types.MappingProxyType(r) for r in records)

    def write_csv(self, path: str) -> None:
        headers = ["zone"] + list(self.columns)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for rec in self.to_records():
                w.writerow([rec[h] for h in headers])


def _validate_zones(zones: Raster) -> None:
    if not np.issubdtype(zones.values.dtype, np.integer) and not np.issubdtype(
        zones.values.dtype, np.bool_
    ):
        raise MapAlgebraDTypeError(
            "Zone raster must have integer or boolean dtype.",
            code="map_algebra_invalid_zone_dtype",
            details={"dtype": str(zones.values.dtype)},
        )


def _compute_count_stat(
    stat: StatisticName,
    n_zones: int,
    all_indices: list[np.ndarray],
    value_indices: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    out_vals = np.zeros(n_zones, dtype=np.int64)
    out_valid = np.ones(n_zones, dtype=np.bool_)

    for i, (all_idx, val_idx) in enumerate(zip(all_indices, value_indices)):
        if stat == "count":
            out_vals[i] = len(all_idx)
        elif stat == "valid_count":
            out_vals[i] = len(val_idx)
        elif stat == "invalid_count":
            out_vals[i] = len(all_idx) - len(val_idx)
    return out_vals, out_valid


def _compute_value_stat(
    stat: StatisticName,
    flat_values: np.ndarray,
    value_indices: list[np.ndarray],
    n_zones: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_vals = np.zeros(n_zones, dtype=np.float64)
    out_valid = np.ones(n_zones, dtype=np.bool_)

    for i, idx in enumerate(value_indices):
        n = len(idx)
        if n == 0:
            out_valid[i] = False
            continue
        fd = flat_values[idx].astype(np.float64, copy=False)
        if stat == "sum":
            out_vals[i] = float(np.sum(fd))
        elif stat == "mean":
            out_vals[i] = float(np.mean(fd))
        elif stat == "min":
            out_vals[i] = float(np.min(fd))
        elif stat == "max":
            out_vals[i] = float(np.max(fd))
        elif stat == "range":
            out_vals[i] = float(np.max(fd) - np.min(fd))
        elif stat == "std":
            out_vals[i] = float(np.std(fd, ddof=0))
        elif stat == "variance":
            out_vals[i] = float(np.var(fd, ddof=0))
        elif stat == "median":
            out_vals[i] = float(np.median(fd))
        elif stat == "p25":
            out_vals[i] = float(np.percentile(fd, 25, method="linear"))
        elif stat == "p75":
            out_vals[i] = float(np.percentile(fd, 75, method="linear"))
        elif stat == "p90":
            out_vals[i] = float(np.percentile(fd, 90, method="linear"))
    return out_vals, out_valid


def _validate_include_ids(
    ids: Sequence[int],
    zone_dtype: np.dtype,
) -> np.ndarray:
    try:
        arr = np.asarray(list(ids), dtype=zone_dtype)
    except (OverflowError, ValueError, TypeError) as exc:
        raise MapAlgebraError(
            f"include_zone_ids value cannot be represented in zone dtype {zone_dtype}.",
            code="map_algebra_invalid_zone_id",
            details={"zone_dtype": str(zone_dtype), "error": str(exc)},
        ) from exc
    return arr


def zonal_stats(
    values: Raster,
    zones: Raster,
    *,
    statistics: Sequence[StatisticName] | None = None,
    include_zone_ids: list[int] | None = None,
    zone_nodata: int | None = None,
) -> ZonalStatistics:
    _require_common_grid([values, zones])
    _validate_zones(zones)

    if statistics is None:
        requested = list(_DEFAULT_STATS)
    else:
        requested = list(statistics)

    for s in requested:
        if s not in _VALID_STATS:
            raise MapAlgebraError(
                f"Unknown zonal statistic: '{s}'. Valid: {sorted(_VALID_STATS)}.",
                code="map_algebra_invalid_statistic",
                details={"statistic": s},
            )

    flat_zones = zones.values.ravel()
    zone_valid = zones.valid.ravel()
    if zone_nodata is not None:
        zone_valid = zone_valid & (flat_zones != zone_nodata)
    flat_values = values.values.ravel()
    value_valid = values.valid.ravel()

    zone_ids = np.unique(flat_zones[zone_valid])
    if include_zone_ids is not None:
        inc = _validate_include_ids(include_zone_ids, zone_ids.dtype)
        extra = inc[~np.isin(inc, zone_ids)]
        zone_ids = np.sort(np.concatenate([zone_ids, extra]))

    n_zones = len(zone_ids)

    if n_zones == 0:
        return ZonalStatistics(
            zone_ids=np.array([], dtype=np.int64),
            columns=tuple(requested),
            values={s: np.array([], dtype=np.int64 if s in _INTEGER_STATS else np.float64) for s in requested},
            valid={s: np.array([], dtype=np.bool_) for s in requested},
            units={s: values.units for s in requested},
        )

    zone_indices: list[np.ndarray] = []
    zone_all_indices: list[np.ndarray] = []
    for zid in zone_ids:
        zone_mask = (flat_zones == zid) & zone_valid
        idx_all = np.where(zone_mask)[0]
        zone_all_indices.append(idx_all)
        idx_valid = np.where(zone_mask & value_valid)[0]
        zone_indices.append(idx_valid)

    result_values: dict[str, np.ndarray] = {}
    result_valid: dict[str, np.ndarray] = {}
    result_units: dict[str, str | None] = {}

    for stat in requested:
        if stat in ("count", "invalid_count", "valid_count"):
            vals, vld = _compute_count_stat(stat, n_zones, zone_all_indices, zone_indices)
            result_values[stat] = vals
            result_valid[stat] = vld
        else:
            vals, vld = _compute_value_stat(stat, flat_values, zone_indices, n_zones)
            result_values[stat] = vals
            result_valid[stat] = vld
        result_units[stat] = values.units

    zone_ids.flags.writeable = False
    for arr in result_values.values():
        arr.flags.writeable = False
    for arr in result_valid.values():
        arr.flags.writeable = False

    return ZonalStatistics(
        zone_ids=zone_ids,
        columns=tuple(requested),
        values=result_values,
        valid=result_valid,
        units=result_units,
    )


def zonal_raster(
    values: Raster,
    zones: Raster,
    statistic: StatisticName = "mean",
) -> Raster:
    if statistic not in _VALID_STATS:
        raise MapAlgebraError(
            f"Unknown zonal statistic: '{statistic}'. Valid: {sorted(_VALID_STATS)}.",
            code="map_algebra_invalid_statistic",
            details={"statistic": statistic},
        )

    zs = zonal_stats(values, zones, statistics=[statistic])
    out_values = np.zeros(values.shape, dtype=np.float64)
    out_valid = np.zeros(values.shape, dtype=np.bool_)

    flat_zones = zones.values.ravel()
    zone_valid = zones.valid.ravel()

    for i, zid in enumerate(zs.zone_ids):
        if not zs.valid[statistic][i]:
            continue
        mask = (flat_zones == zid) & zone_valid
        out_values.ravel()[mask] = zs.values[statistic][i]
        out_valid.ravel()[mask] = True

    return Raster(
        values=out_values,
        georef=values.georef,
        valid=out_valid,
        units=values.units,
        name=values.name,
    )
