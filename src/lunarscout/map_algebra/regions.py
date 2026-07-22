from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MapAlgebraDTypeError, MapAlgebraOperationError
from ..raster import Raster
from ..regions import (
    CleanupMode,
    Comparator,
    Connectivity,
    filter_regions_by_size as _array_filter_regions_by_size,
    find_borders as _array_find_borders,
    label_regions as _array_label_regions,
    region_sizes as _array_region_sizes,
)


def _require_boolean_raster(value: Any, *, operation: str) -> Raster:
    if not isinstance(value, Raster):
        raise MapAlgebraOperationError(
            f"ma.{operation}() requires an eager Raster operand.",
            code="map_algebra_invalid_region_operand",
            details={"operation": operation, "type": type(value).__name__},
        )
    if value.dtype != np.dtype(np.bool_):
        raise MapAlgebraDTypeError(
            f"ma.{operation}() requires a Boolean Raster mask.",
            code="map_algebra_requires_boolean",
            details={"operation": operation, "dtype": str(value.dtype)},
        )
    return value


def _masked_values(raster: Raster) -> np.ma.MaskedArray:
    return np.ma.array(raster.values, mask=~raster.valid, copy=False)


def _result_raster(
    source: Raster,
    result: np.ndarray | np.ma.MaskedArray,
    *,
    dtype: np.dtype[Any],
) -> Raster:
    values = np.asarray(np.ma.getdata(result), dtype=dtype)
    valid = ~np.ma.getmaskarray(result)
    return Raster(
        values=values,
        georef=source.georef,
        valid=valid,
        units=None,
        name=source.name,
    )


def label_regions(
    mask: Raster,
    *,
    cleanup: CleanupMode = "none",
    iterations: int = 0,
    connectivity: Connectivity = 8,
) -> Raster:
    """Label connected valid true cells with deterministic int32 IDs."""
    source = _require_boolean_raster(mask, operation="label_regions")
    result, _ = _array_label_regions(
        _masked_values(source),
        source.georef,
        nodata=None,
        cleanup=cleanup,
        iterations=iterations,
        connectivity=connectivity,
    )
    return _result_raster(source, result, dtype=np.dtype(np.int32))


def region_sizes(
    mask: Raster,
    *,
    cleanup: CleanupMode = "none",
    iterations: int = 0,
    connectivity: Connectivity = 8,
) -> Raster:
    """Return each valid true cell's connected-region size in pixels."""
    source = _require_boolean_raster(mask, operation="region_sizes")
    result, _ = _array_region_sizes(
        _masked_values(source),
        source.georef,
        nodata=None,
        cleanup=cleanup,
        iterations=iterations,
        connectivity=connectivity,
    )
    return _result_raster(source, result, dtype=np.dtype(np.int32))


def filter_regions_by_size(
    mask: Raster,
    *,
    threshold: float,
    comparator: Comparator = ">=",
    cleanup: CleanupMode = "none",
    iterations: int = 0,
    connectivity: Connectivity = 8,
) -> Raster:
    """Keep connected regions selected by their pixel count."""
    source = _require_boolean_raster(mask, operation="filter_regions_by_size")
    result, _ = _array_filter_regions_by_size(
        _masked_values(source),
        source.georef,
        threshold=threshold,
        comparator=comparator,
        nodata=None,
        cleanup=cleanup,
        iterations=iterations,
        connectivity=connectivity,
    )
    return _result_raster(source, result, dtype=np.dtype(np.bool_))


def find_borders(
    mask: Raster,
    *,
    connectivity: Connectivity = 8,
) -> Raster:
    """Return the internal border cells of valid true regions."""
    source = _require_boolean_raster(mask, operation="find_borders")
    result, _ = _array_find_borders(
        _masked_values(source),
        source.georef,
        nodata=None,
        connectivity=connectivity,
    )
    return _result_raster(source, result, dtype=np.dtype(np.bool_))
