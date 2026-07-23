from __future__ import annotations

import csv
import json
import types
from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
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
            "values": {
                key: [
                    int(value) if values.dtype.kind in "biu" else float(value)
                    for value in values
                ]
                for key, values in self.values.items()
            },
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
                    rec[col] = (
                        int(v) if self.values[col].dtype.kind in "biu" else float(v)
                    )
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


def _value_stat_dtype(stat: StatisticName, source_dtype: np.dtype[Any]) -> np.dtype[Any]:
    if stat == "sum" and source_dtype.kind in "biu":
        return np.dtype(np.uint64 if source_dtype.kind == "u" else np.int64)
    if stat in {"min", "max"}:
        return source_dtype
    if stat == "range" and source_dtype.kind in "biu":
        if source_dtype.kind == "b":
            return np.dtype(np.uint8)
        return np.dtype(f"uint{source_dtype.itemsize * 8}")
    return np.dtype(np.float64)


def _compute_value_stat(
    stat: StatisticName,
    flat_values: np.ndarray,
    value_indices: list[np.ndarray],
    n_zones: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_dtype = flat_values.dtype
    output_dtype = _value_stat_dtype(stat, source_dtype)
    out_vals = np.zeros(n_zones, dtype=output_dtype)
    out_valid = np.ones(n_zones, dtype=np.bool_)

    for i, idx in enumerate(value_indices):
        n = len(idx)
        if n == 0:
            out_valid[i] = False
            continue
        data = flat_values[idx]
        if data.dtype.kind in "biu":
            exact = sorted(int(value) for value in data)
            total = sum(exact)
            if stat == "sum":
                out_vals[i] = np.sum(data, dtype=output_dtype)
            elif stat == "mean":
                out_vals[i] = float(Fraction(total, n))
            elif stat == "min":
                out_vals[i] = exact[0]
            elif stat == "max":
                out_vals[i] = exact[-1]
            elif stat == "range":
                out_vals[i] = exact[-1] - exact[0]
            elif stat in {"std", "variance"}:
                sum_squares = sum(value * value for value in exact)
                variance = Fraction(n * sum_squares - total * total, n * n)
                out_vals[i] = sqrt(float(variance)) if stat == "std" else float(variance)
            elif stat == "median":
                middle = n // 2
                value = (
                    Fraction(exact[middle - 1] + exact[middle], 2)
                    if n % 2 == 0 else Fraction(exact[middle])
                )
                out_vals[i] = float(value)
            elif stat in {"p25", "p75", "p90"}:
                quantile = {"p25": 25, "p75": 75, "p90": 90}[stat]
                rank = Fraction(quantile * (n - 1), 100)
                lower = rank.numerator // rank.denominator
                upper = -(-rank.numerator // rank.denominator)
                if lower == upper:
                    value = Fraction(exact[lower])
                else:
                    weight = rank - lower
                    value = exact[lower] * (1 - weight) + exact[upper] * weight
                out_vals[i] = float(value)
            continue

        fd = data.astype(np.float64, copy=False)
        if stat == "sum":
            out_vals[i] = np.sum(fd)
        elif stat == "mean":
            out_vals[i] = np.mean(fd)
        elif stat == "min":
            out_vals[i] = np.min(fd)
        elif stat == "max":
            out_vals[i] = np.max(fd)
        elif stat == "range":
            out_vals[i] = np.max(fd) - np.min(fd)
        elif stat == "std":
            out_vals[i] = np.std(fd, ddof=0)
        elif stat == "variance":
            out_vals[i] = np.var(fd, ddof=0)
        elif stat == "median":
            out_vals[i] = np.median(fd)
        elif stat == "p25":
            out_vals[i] = np.percentile(fd, 25, method="linear")
        elif stat == "p75":
            out_vals[i] = np.percentile(fd, 75, method="linear")
        elif stat == "p90":
            out_vals[i] = np.percentile(fd, 90, method="linear")
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
            zone_ids=np.array([], dtype=zones.dtype),
            columns=tuple(requested),
            values={
                statistic: np.array(
                    [],
                    dtype=(
                        np.dtype(np.int64)
                        if statistic in _INTEGER_STATS
                        else _value_stat_dtype(statistic, values.dtype)
                    ),
                )
                for statistic in requested
            },
            valid={s: np.array([], dtype=np.bool_) for s in requested},
            units={s: None if s in _INTEGER_STATS else values.units for s in requested},
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
        result_units[stat] = None if stat in _INTEGER_STATS else values.units

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
    out_values = np.zeros(values.shape, dtype=zs.values[statistic].dtype)
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
        units=zs.units[statistic],
        name=values.name,
    )
