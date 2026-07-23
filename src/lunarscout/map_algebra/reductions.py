from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from typing import Any, Literal

import numpy as np

from ..errors import MapAlgebraOperationError
from ..raster import Raster


@dataclass(frozen=True, slots=True)
class RasterStatistics:
    count: int
    invalid_count: int
    sum: int | float
    mean: float
    min_val: int | float
    max_val: int | float
    range_val: int | float
    variance: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count, "invalid_count": self.invalid_count,
            "sum": self.sum, "mean": self.mean,
            "min": self.min_val, "max": self.max_val,
            "range": self.range_val, "variance": self.variance,
            "std": self.std,
        }


def statistics(raster: Raster) -> RasterStatistics:
    """Compute summary statistics from valid pixels."""
    valid_mask = raster.valid
    count = int(np.sum(valid_mask))
    invalid = raster.invalid_count

    if count == 0:
        raise MapAlgebraOperationError(
            "Cannot compute statistics on a raster with no valid pixels.",
            code="map_algebra_empty_reduction",
        )

    valid_data = raster.values[valid_mask]
    if valid_data.dtype.kind in "biu":
        exact = [int(value) for value in valid_data]
        s: int | float = sum(exact)
        mn: int | float = min(exact)
        mx: int | float = max(exact)
        avg = float(Fraction(s, count))
        sum_squares = sum(value * value for value in exact)
        variance_fraction = Fraction(count * sum_squares - s * s, count * count)
        var = float(variance_fraction)
        sd = sqrt(var)
        range_value: int | float = mx - mn
    else:
        floating = valid_data.astype(np.float64, copy=False)
        s = float(np.sum(floating))
        mn = float(np.min(floating))
        mx = float(np.max(floating))
        avg = s / float(count)
        var = float(np.var(floating, ddof=0))
        sd = float(np.std(floating, ddof=0))
        range_value = mx - mn

    return RasterStatistics(
        count=count, invalid_count=invalid, sum=s, mean=avg,
        min_val=mn, max_val=mx, range_val=range_value,
        variance=var, std=sd,
    )


def histogram(
    raster: Raster,
    *,
    bins: int | np.ndarray = 10,
    range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = raster.valid
    if not np.any(valid_mask):
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    valid_data = raster.values[valid_mask]
    if valid_data.dtype.kind in "iu" and np.any(
        np.abs(valid_data.astype(object)) > 2**53
    ):
        if isinstance(bins, (int, np.integer)) or np.asarray(bins).dtype.kind == "f":
            raise MapAlgebraOperationError(
                "Large integers require explicit integer histogram edges.",
                code="map_algebra_inexact_histogram_edges",
                details={"dtype": valid_data.dtype.name},
            )
    try:
        counts, edges = np.histogram(valid_data, bins=bins, range=range)
    except (TypeError, ValueError) as exc:
        raise MapAlgebraOperationError(
            "Invalid histogram bins or range.",
            code="map_algebra_invalid_histogram",
            details={"error": str(exc)},
        ) from exc
    return counts.astype(np.int64), edges


def percentile(
    raster: Raster,
    q: float | list[float] | np.ndarray,
    *,
    method: Literal["exact", "approximate"] = "exact",
) -> int | float | np.integer[Any] | np.floating[Any] | np.ndarray:
    """Compute percentile(s) of valid pixels.

    ``method="exact"`` uses NumPy linear interpolation (in-memory, full
    precision).  ``method="approximate"`` uses NumPy nearest-rank selection
    (also in-memory; it trades interpolation quality for simplicity, not
    for reduced memory).  Neither method bounds memory independently of
    raster size.

    Integer samples are ordered before any floating conversion. Percentiles
    that select an observed integer exactly therefore preserve that integer;
    interpolated results are returned as float64.
    """
    valid_mask = raster.valid
    if not np.any(valid_mask):
        raise MapAlgebraOperationError(
            "Cannot compute percentile on a raster with no valid pixels.",
            code="map_algebra_empty_reduction",
        )
    if method not in {"exact", "approximate"}:
        raise MapAlgebraOperationError(
            "method must be 'exact' or 'approximate'.",
            code="map_algebra_invalid_percentile_method",
            details={"method": method},
        )
    valid_data = raster.values[valid_mask]
    numpy_method = "linear" if method == "exact" else "nearest"
    if valid_data.dtype.kind not in "biu":
        return np.percentile(valid_data, q, method=numpy_method)  # type: ignore[no-any-return]

    quantiles = np.asarray(q, dtype=np.float64)
    if np.any(~np.isfinite(quantiles)) or np.any((quantiles < 0.0) | (quantiles > 100.0)):
        raise MapAlgebraOperationError(
            "Percentiles must be finite values from 0 through 100.",
            code="map_algebra_invalid_percentile",
            details={"q": quantiles.tolist()},
        )
    ordered = sorted(int(value) for value in valid_data)
    results: list[int | float] = []
    all_observed = True
    for quantile in quantiles.reshape(-1):
        rank = float(quantile) * (len(ordered) - 1) / 100.0
        if method == "approximate":
            index = int(np.rint(rank))
            results.append(ordered[index])
            continue
        lower = int(np.floor(rank))
        upper = int(np.ceil(rank))
        if lower == upper:
            results.append(ordered[lower])
            continue
        all_observed = False
        weight = Fraction.from_float(rank - lower)
        interpolated = (
            Fraction(ordered[lower]) * (1 - weight)
            + Fraction(ordered[upper]) * weight
        )
        results.append(float(interpolated))

    if all_observed:
        result = np.asarray(results, dtype=valid_data.dtype).reshape(quantiles.shape)
    else:
        result = np.asarray(results, dtype=np.float64).reshape(quantiles.shape)
    if quantiles.ndim == 0:
        return result[()]
    return result


def unique_counts(
    raster: Raster,
    *,
    max_unique: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = raster.valid
    if not np.any(valid_mask):
        return np.array([], dtype=raster.values.dtype), np.array([], dtype=np.int64)

    valid_data = raster.values[valid_mask]
    uniq, counts = np.unique(valid_data, return_counts=True)
    if max_unique is not None and len(uniq) > max_unique:
        raise MapAlgebraOperationError(
            f"Number of unique values ({len(uniq)}) exceeds max_unique ({max_unique}).",
            code="map_algebra_max_unique_exceeded",
            details={"unique_count": int(len(uniq)), "max_unique": max_unique},
        )
    return uniq, counts.astype(np.int64)
