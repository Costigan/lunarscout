from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..errors import MapAlgebraOperationError
from ..raster import Raster


@dataclass(frozen=True, slots=True)
class RasterStatistics:
    count: int
    invalid_count: int
    sum: float
    mean: float
    min_val: float
    max_val: float
    range_val: float
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
    """Compute summary statistics from valid pixels.

    Integer values beyond 2^53 may lose precision when converted to float64.
    """
    valid_mask = raster.valid
    count = int(np.sum(valid_mask))
    invalid = raster.invalid_count

    if count == 0:
        raise MapAlgebraOperationError(
            "Cannot compute statistics on a raster with no valid pixels.",
            code="map_algebra_empty_reduction",
        )

    valid_data = raster.values[valid_mask].astype(np.float64, copy=False)
    s = float(np.sum(valid_data))
    mn = float(np.min(valid_data))
    mx = float(np.max(valid_data))
    avg = s / float(count)
    var = float(np.var(valid_data, ddof=0))
    sd = float(np.std(valid_data, ddof=0))

    return RasterStatistics(
        count=count, invalid_count=invalid, sum=s, mean=avg,
        min_val=mn, max_val=mx, range_val=mx - mn,
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
    valid_data = raster.values[valid_mask].astype(np.float64, copy=False)
    counts, edges = np.histogram(valid_data, bins=bins, range=range)
    return counts.astype(np.int64), edges.astype(np.float64)


def percentile(
    raster: Raster,
    q: float | list[float] | np.ndarray,
    *,
    method: Literal["exact", "approximate"] = "exact",
) -> float | np.ndarray:
    """Compute percentile(s) of valid pixels.

    ``method="exact"`` uses NumPy linear interpolation (in-memory, full
    precision).  ``method="approximate"`` uses NumPy nearest-rank selection
    (also in-memory; it trades interpolation quality for simplicity, not
    for reduced memory).  Neither method bounds memory independently of
    raster size.

    Integer values beyond 2^53 may lose precision when converted to float64.
    """
    valid_mask = raster.valid
    if not np.any(valid_mask):
        raise MapAlgebraOperationError(
            "Cannot compute percentile on a raster with no valid pixels.",
            code="map_algebra_empty_reduction",
        )
    valid_data = raster.values[valid_mask].astype(np.float64, copy=False)
    numpy_method = "linear" if method == "exact" else "nearest"
    return np.percentile(valid_data, q, method=numpy_method)  # type: ignore[no-any-return]


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
