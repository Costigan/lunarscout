from __future__ import annotations

import warnings
from fractions import Fraction
from math import sqrt
from typing import Any, Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from ..errors import (
    MapAlgebraDTypeError,
    MapAlgebraError,
)
from ..raster import Raster

EdgeMode = Literal["invalid", "constant", "nearest", "reflect", "wrap"]
ValidNeighbor = Literal["require_all", "ignore_invalid", "propagate_center"]

_VALID_EDGE_MODES = frozenset({"invalid", "constant", "nearest", "reflect", "wrap"})
_VALID_NEIGHBOR = frozenset({"require_all", "ignore_invalid", "propagate_center"})

_SCIPY_MODE_MAP: dict[str, str] = {
    "invalid": "constant",
    "constant": "constant",
    "nearest": "nearest",
    "reflect": "reflect",
    "wrap": "wrap",
}


def _validate_edge(edge: str) -> EdgeMode:
    if edge not in _VALID_EDGE_MODES:
        raise MapAlgebraError(
            f"Unknown edge mode: '{edge}'. Must be one of {sorted(_VALID_EDGE_MODES)}.",
            code="map_algebra_invalid_edge_mode",
        )
    return edge  # type: ignore[return-value]


def _validate_valid_neighbor(vn: str) -> ValidNeighbor:
    if vn not in _VALID_NEIGHBOR:
        raise MapAlgebraError(
            f"Unknown valid_neighbor: '{vn}'. Must be one of {sorted(_VALID_NEIGHBOR)}.",
            code="map_algebra_invalid_valid_neighbor",
        )
    return vn  # type: ignore[return-value]


def _validate_min_valid_count(
    min_valid_count: int | None,
    *,
    valid_neighbor: ValidNeighbor,
    footprint_count: int,
) -> int | None:
    if min_valid_count is None:
        return None
    if not isinstance(min_valid_count, int) or isinstance(min_valid_count, bool):
        raise MapAlgebraError(
            "min_valid_count must be an integer or None.",
            code="map_algebra_invalid_min_valid_count",
            details={"min_valid_count": repr(min_valid_count)},
        )
    if valid_neighbor != "ignore_invalid":
        raise MapAlgebraError(
            "min_valid_count is only valid with valid_neighbor='ignore_invalid'.",
            code="map_algebra_invalid_min_valid_count",
            details={
                "min_valid_count": min_valid_count,
                "valid_neighbor": valid_neighbor,
            },
        )
    if min_valid_count < 1 or min_valid_count > footprint_count:
        raise MapAlgebraError(
            "min_valid_count must be between 1 and the footprint cell count.",
            code="map_algebra_invalid_min_valid_count",
            details={
                "min_valid_count": min_valid_count,
                "minimum": 1,
                "maximum": footprint_count,
            },
        )
    return min_valid_count


def _validate_ddof(ddof: Any) -> int:
    if (
        not isinstance(ddof, (int, np.integer))
        or isinstance(ddof, (bool, np.bool_))
        or ddof < 0
    ):
        raise MapAlgebraError(
            "ddof must be a non-negative integer.",
            code="map_algebra_invalid_ddof",
            details={"ddof": repr(ddof)},
        )
    return int(ddof)


def _validate_footprint(
    size: int | tuple[int, int] | None,
    footprint: np.ndarray | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if footprint is not None:
        fp = np.asarray(footprint, dtype=np.bool_)
        if fp.ndim != 2:
            raise MapAlgebraError("Footprint must be a two-dimensional array.",
                                   code="map_algebra_invalid_footprint")
        if fp.shape[0] % 2 != 1 or fp.shape[1] % 2 != 1:
            raise MapAlgebraError("Footprint dimensions must be odd.",
                                   code="map_algebra_invalid_footprint")
        if fp.size == 0 or not np.any(fp):
            raise MapAlgebraError("Footprint must contain at least one active cell.",
                                   code="map_algebra_invalid_footprint")
        return fp, (fp.shape[0] // 2, fp.shape[1] // 2)
    if size is not None:
        if isinstance(size, int):
            s = size
            if s < 1 or s % 2 != 1:
                raise MapAlgebraError("Window size must be a positive odd integer.",
                                       code="map_algebra_invalid_footprint")
            fp = np.ones((s, s), dtype=np.bool_)
            return fp, (s // 2, s // 2)
        h, w = size
        if h < 1 or w < 1 or h % 2 != 1 or w % 2 != 1:
            raise MapAlgebraError("Window dimensions must be positive odd integers.",
                                   code="map_algebra_invalid_footprint")
        fp = np.ones((h, w), dtype=np.bool_)
        return fp, (h // 2, w // 2)
    raise MapAlgebraError("Either size or footprint must be provided.",
                           code="map_algebra_invalid_footprint")


def _validate_focal_expression_parameters(arguments: dict[str, Any]) -> None:
    """Validate public focal parameters without executing a raster kernel."""
    if "kernel" in arguments:
        kernel = _validate_convolution_kernel(arguments["kernel"])
        footprint_count = int(kernel.size)
    else:
        footprint, _ = _validate_footprint(
            arguments.get("size"), arguments.get("footprint"),
        )
        footprint_count = int(np.count_nonzero(footprint))

    if "edge" not in arguments and "valid_neighbor" not in arguments:
        return
    _validate_edge(arguments.get("edge", "invalid"))
    valid_neighbor = _validate_valid_neighbor(
        arguments.get("valid_neighbor", "require_all"),
    )
    _validate_min_valid_count(
        arguments.get("min_valid_count"),
        valid_neighbor=valid_neighbor,
        footprint_count=footprint_count,
    )
    if "ddof" in arguments:
        _validate_ddof(arguments["ddof"])


def _validate_convolution_kernel(kernel: Any) -> np.ndarray:
    candidate = np.asarray(kernel)
    if (
        candidate.ndim != 2
        or candidate.shape[0] % 2 != 1
        or candidate.shape[1] % 2 != 1
    ):
        raise MapAlgebraError(
            "Convolution kernel must be two-dimensional with odd dimensions.",
            code="map_algebra_invalid_kernel",
        )
    if candidate.dtype.kind not in "biuf" or not np.all(np.isfinite(candidate)):
        raise MapAlgebraError(
            "Convolution kernel must contain only finite real numeric values.",
            code="map_algebra_invalid_kernel",
        )
    return candidate.astype(np.float64, copy=False)


def _pad_array(
    values: np.ndarray,
    valid: np.ndarray,
    halo: tuple[int, int],
    edge: EdgeMode,
    cval: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    pad_h, pad_w = halo
    pad_width = ((pad_h, pad_h), (pad_w, pad_w))
    if edge == "invalid":
        padded = np.pad(values, pad_width, mode="constant", constant_values=cval)
        padded_mask = np.pad(valid, pad_width, mode="constant", constant_values=False)
    elif edge == "constant":
        padded = np.pad(values, pad_width, mode="constant", constant_values=cval)
        padded_mask = np.pad(valid, pad_width, mode="constant", constant_values=False)
    elif edge == "nearest":
        padded = np.pad(values, pad_width, mode="edge")
        padded_mask = np.pad(valid, pad_width, mode="edge")
    elif edge == "reflect":
        padded = np.pad(values, pad_width, mode="reflect")
        padded_mask = np.pad(valid, pad_width, mode="reflect")
    elif edge == "wrap":
        padded = np.pad(values, pad_width, mode="wrap")
        padded_mask = np.pad(valid, pad_width, mode="wrap")
    else:
        raise MapAlgebraError(f"Unknown edge mode: {edge}",
                               code="map_algebra_invalid_edge_mode")
    crop_slice = (slice(pad_h, pad_h + values.shape[0]),
                  slice(pad_w, pad_w + values.shape[1]))
    return padded, padded_mask, crop_slice


def _apply_edge_invalid(
    valid: np.ndarray,
    halo: tuple[int, int],
) -> None:
    hh, hw = halo
    if hh > 0:
        valid[:hh, :] = False
        valid[-hh:, :] = False
    if hw > 0:
        valid[:, :hw] = False
        valid[:, -hw:] = False


def _output_dtype(source_dtype: np.dtype[Any], operation: str) -> np.dtype[Any]:
    src = np.dtype(source_dtype)
    if operation in {"sum", "mean", "std", "count", "min", "max"}:
        from ._dtypes import accumulator_dtype

        return accumulator_dtype(src, operation=f"focal.{operation}")
    if operation in ("range", "median"):
        if src == np.dtype(np.float64) or (
            src.kind in "iu" and src.itemsize >= 4
        ):
            return np.dtype(np.float64)
        return np.dtype(np.float32)
    if operation in ("min", "max"):
        return src
    return src


def _neighborhood_windows(
    raster: Raster,
    footprint: np.ndarray,
    *,
    edge: EdgeMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-precision value windows and active validity windows."""
    halo = (footprint.shape[0] // 2, footprint.shape[1] // 2)
    # Constant/invalid padding is canonically invalid in the established focal
    # contract, so its payload is deliberately zero and never participates.
    padded, padded_valid, _ = _pad_array(
        raster.values, raster.valid, halo, edge, cval=0.0,
    )
    windows = sliding_window_view(padded, footprint.shape)
    valid_windows = sliding_window_view(padded_valid, footprint.shape)
    return windows, valid_windows & footprint


def _neighborhood_validity(
    raster: Raster,
    footprint: np.ndarray,
    active: np.ndarray,
    *,
    edge: EdgeMode,
    valid_neighbor: ValidNeighbor,
    min_valid_count: int | None,
    ddof: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    count = np.sum(active, axis=(-2, -1), dtype=np.uint32)
    footprint_count = int(np.count_nonzero(footprint))
    if valid_neighbor == "ignore_invalid":
        threshold = 1 if min_valid_count is None else min_valid_count
        result_valid = count >= threshold
    elif valid_neighbor == "propagate_center":
        result_valid = raster.valid.copy()
    else:
        result_valid = count == footprint_count
    if ddof >= 0:
        result_valid &= count > ddof
    if edge == "invalid":
        _apply_edge_invalid(
            result_valid,
            (footprint.shape[0] // 2, footprint.shape[1] // 2),
        )
    return count, result_valid


def _integer_fractional_reduction(
    windows: np.ndarray,
    active: np.ndarray,
    result_valid: np.ndarray,
    *,
    operation: Literal["mean", "std"],
    ddof: int = 0,
) -> np.ndarray:
    """Reduce integer neighborhoods without first converting payloads to FP64."""
    result = np.zeros(windows.shape[:2], dtype=np.float64)
    for index in np.ndindex(result.shape):
        if not result_valid[index]:
            continue
        values = windows[index][active[index]]
        count = int(values.size)
        total = sum(int(value) for value in values)
        if operation == "mean":
            result[index] = float(Fraction(total, count))
            continue
        sum_squares = sum(int(value) * int(value) for value in values)
        numerator = count * sum_squares - total * total
        denominator = count * (count - ddof)
        result[index] = sqrt(float(Fraction(numerator, denominator)))
    return result


def _focal_reduction(
    raster: Raster,
    footprint: np.ndarray,
    *,
    operation: Literal["sum", "mean", "std"],
    edge: EdgeMode,
    valid_neighbor: ValidNeighbor,
    min_valid_count: int | None,
    ddof: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    windows, active = _neighborhood_windows(raster, footprint, edge=edge)
    _, result_valid = _neighborhood_validity(
        raster, footprint, active, edge=edge,
        valid_neighbor=valid_neighbor, min_valid_count=min_valid_count,
        ddof=ddof if operation == "std" else -1,
    )
    output_dtype = _output_dtype(raster.dtype, operation)
    if operation == "sum":
        selected = np.where(active, windows, raster.dtype.type(0))
        return (
            np.sum(selected, axis=(-2, -1), dtype=output_dtype),
            result_valid,
        )
    if raster.dtype.kind in "biu":
        return (
            _integer_fractional_reduction(
                windows, active, result_valid, operation=operation, ddof=ddof,
            ),
            result_valid,
        )
    selected = np.where(active, windows, output_dtype.type(0))
    count = np.sum(active, axis=(-2, -1), dtype=np.uint32)
    if operation == "mean":
        total = np.sum(selected, axis=(-2, -1), dtype=output_dtype)
        result = np.zeros(total.shape, dtype=output_dtype)
        np.divide(total, count.astype(output_dtype), out=result, where=count > 0)
        return result, result_valid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.std(
            windows, axis=(-2, -1), dtype=output_dtype, ddof=ddof,
            where=active,
        )
    return result.astype(output_dtype, copy=False), result_valid


def _focal_extreme(
    raster: Raster,
    footprint: np.ndarray,
    *,
    operation: Literal["min", "max"],
    edge: EdgeMode,
    valid_neighbor: ValidNeighbor,
    min_valid_count: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    windows, active = _neighborhood_windows(raster, footprint, edge=edge)
    _, result_valid = _neighborhood_validity(
        raster, footprint, active, edge=edge,
        valid_neighbor=valid_neighbor, min_valid_count=min_valid_count,
    )
    if raster.dtype.kind == "f":
        initial = np.inf if operation == "min" else -np.inf
    elif raster.dtype.kind in "iu":
        info = np.iinfo(raster.dtype)
        initial = info.max if operation == "min" else info.min
    else:
        initial = True if operation == "min" else False
    reducer = np.min if operation == "min" else np.max
    result = reducer(
        windows, axis=(-2, -1), where=active, initial=initial,
    )
    return result.astype(raster.dtype, copy=False), result_valid


# ---------------------------------------------------------------------------
# Core focal filter
# ---------------------------------------------------------------------------


def _focal_generic(
    raster: Raster,
    footprint: np.ndarray,
    *,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    func: Any,
    neutral: float = 0.0,
    cval: float = 0.0,
    output_dtype: np.dtype | None = None,
    op_name: str = "focal",
    min_valid_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    halo = (footprint.shape[0] // 2, footprint.shape[1] // 2)
    vals = raster.values.astype(np.float64, copy=False)
    vld = raster.valid.copy()
    padded, padded_mask, crop = _pad_array(vals, vld, halo, edge, cval=cval)
    scipy_mode = _SCIPY_MODE_MAP[edge]
    ftp_count = float(np.sum(footprint))

    if valid_neighbor == "ignore_invalid":
        padded[~padded_mask] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result_padded = ndimage.generic_filter(
                padded, func, footprint=footprint, mode=scipy_mode, cval=np.nan,
            )
        count_padded = ndimage.generic_filter(
            padded_mask.astype(np.float64), np.sum, footprint=footprint,
            mode=scipy_mode, cval=0.0,
        )
        threshold = 1 if min_valid_count is None else min_valid_count
        result_valid = count_padded[crop] >= threshold
    elif valid_neighbor == "propagate_center":
        padded[~padded_mask] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result_padded = ndimage.generic_filter(
                padded, func, footprint=footprint, mode=scipy_mode, cval=np.nan,
            )
        result_valid = vld.copy()
    else:
        padded[~padded_mask] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result_padded = ndimage.generic_filter(
                padded, func, footprint=footprint, mode=scipy_mode, cval=np.nan,
            )
        count_padded = ndimage.generic_filter(
            padded_mask.astype(np.float64), np.sum, footprint=footprint,
            mode=scipy_mode, cval=0.0,
        )
        result_valid = np.isclose(count_padded[crop], ftp_count)

    if edge == "invalid":
        _apply_edge_invalid(result_valid, halo)

    final_values = result_padded[crop]
    if output_dtype is not None:
        final_values = final_values.astype(output_dtype)

    return final_values, result_valid


def _focal_special(
    raster: Raster,
    footprint: np.ndarray,
    scipy_filter: Any,
    *,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    neutral: float = 0.0,
    cval: float = 0.0,
    output_dtype: np.dtype | None = None,
    min_valid_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    halo = (footprint.shape[0] // 2, footprint.shape[1] // 2)
    vals = raster.values.astype(np.float64, copy=False)
    vld = raster.valid.copy()
    padded, padded_mask, crop = _pad_array(vals, vld, halo, edge, cval=cval)
    scipy_mode = _SCIPY_MODE_MAP[edge]
    ftp_count = float(np.sum(footprint))

    if valid_neighbor == "ignore_invalid":
        padded[~padded_mask] = neutral
        result_padded = scipy_filter(padded, footprint=footprint, mode=scipy_mode,
                                      cval=neutral if edge in ("invalid", "constant") else 0.0)
        count_padded = ndimage.generic_filter(
            padded_mask.astype(np.float64), np.sum, footprint=footprint,
            mode=scipy_mode, cval=0.0,
        )
        threshold = 1 if min_valid_count is None else min_valid_count
        result_valid = count_padded[crop] >= threshold
    elif valid_neighbor == "propagate_center":
        padded[~padded_mask] = neutral
        result_padded = scipy_filter(padded, footprint=footprint, mode=scipy_mode,
                                      cval=neutral if edge in ("invalid", "constant") else 0.0)
        result_valid = vld.copy()
    else:
        padded[~padded_mask] = neutral
        result_padded = scipy_filter(padded, footprint=footprint, mode=scipy_mode,
                                      cval=neutral if edge in ("invalid", "constant") else 0.0)
        count_padded = ndimage.generic_filter(
            padded_mask.astype(np.float64), np.sum, footprint=footprint,
            mode=scipy_mode, cval=0.0,
        )
        result_valid = np.isclose(count_padded[crop], ftp_count)

    if edge == "invalid":
        _apply_edge_invalid(result_valid, halo)

    final_values = result_padded[crop]
    if output_dtype is not None:
        final_values = final_values.astype(output_dtype)

    return final_values, result_valid


# ---------------------------------------------------------------------------
# Public focal statistics
# ---------------------------------------------------------------------------


def _common_kwargs(
    size=None, footprint=None, edge="invalid", vn="require_all",
    cval=0.0, op_name="focal", min_valid_count=None,
):
    fp, halo = _validate_footprint(size, footprint)
    edge_v = _validate_edge(edge)
    vn_v = _validate_valid_neighbor(vn)
    minimum = _validate_min_valid_count(
        min_valid_count,
        valid_neighbor=vn_v,
        footprint_count=int(np.count_nonzero(fp)),
    )
    return fp, halo, edge_v, vn_v, minimum


def focal_sum(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    min_valid_count: int | None = None,
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn, minimum = _common_kwargs(
        size, footprint, edge, valid_neighbor, cval,
        min_valid_count=min_valid_count,
    )
    vals, mask = _focal_reduction(
        raster, fp, operation="sum", edge=e, valid_neighbor=vn,
        min_valid_count=minimum,
    )
    return Raster(values=vals, georef=raster.georef, valid=mask,
                  units=raster.units, name=raster.name)


def focal_mean(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    min_valid_count: int | None = None,
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn, minimum = _common_kwargs(
        size, footprint, edge, valid_neighbor, cval,
        min_valid_count=min_valid_count,
    )
    vals, mask = _focal_reduction(
        raster, fp, operation="mean", edge=e, valid_neighbor=vn,
        min_valid_count=minimum,
    )
    return Raster(values=vals, georef=raster.georef, valid=mask,
                  units=raster.units, name=raster.name)


def focal_min(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    min_valid_count: int | None = None,
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn, minimum = _common_kwargs(
        size, footprint, edge, valid_neighbor, cval,
        min_valid_count=min_valid_count,
    )
    vals, mask = _focal_extreme(
        raster, fp, operation="min", edge=e, valid_neighbor=vn,
        min_valid_count=minimum,
    )
    return Raster(values=vals, georef=raster.georef, valid=mask,
                  units=raster.units, name=raster.name)


def focal_max(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    min_valid_count: int | None = None,
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn, minimum = _common_kwargs(
        size, footprint, edge, valid_neighbor, cval,
        min_valid_count=min_valid_count,
    )
    vals, mask = _focal_extreme(
        raster, fp, operation="max", edge=e, valid_neighbor=vn,
        min_valid_count=minimum,
    )
    return Raster(values=vals, georef=raster.georef, valid=mask,
                  units=raster.units, name=raster.name)


def focal_range(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    min_valid_count: int | None = None,
    cval: float = 0.0,
) -> Raster:
    min_r = focal_min(
        raster, size=size, footprint=footprint, edge=edge,
        valid_neighbor=valid_neighbor, min_valid_count=min_valid_count,
        cval=cval,
    )
    max_r = focal_max(
        raster, size=size, footprint=footprint, edge=edge,
        valid_neighbor=valid_neighbor, min_valid_count=min_valid_count,
        cval=cval,
    )
    dst = _output_dtype(raster.dtype, "range")
    if raster.dtype.kind in "biu":
        # Subtract in Python's exact integer domain.  Subtracting extrema in
        # the source dtype can overflow (for example int8: 127 - -128), while
        # converting uint64 extrema to float first can erase a small range.
        values = np.fromiter(
            (
                int(high) - int(low)
                for high, low in zip(max_r.values.flat, min_r.values.flat)
            ),
            dtype=dst,
            count=raster.values.size,
        ).reshape(raster.shape)
    else:
        values = np.subtract(max_r.values, min_r.values, dtype=dst)
    return Raster(
        values=values,
        georef=raster.georef,
        valid=min_r.valid & max_r.valid,
        units=raster.units,
        name=raster.name,
    )


def focal_std(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    min_valid_count: int | None = None,
    cval: float = 0.0,
    ddof: int = 0,
) -> Raster:
    fp, halo, e, vn, minimum = _common_kwargs(
        size, footprint, edge, valid_neighbor, cval,
        min_valid_count=min_valid_count,
    )
    ddof = _validate_ddof(ddof)
    vals, mask = _focal_reduction(
        raster, fp, operation="std", edge=e, valid_neighbor=vn,
        min_valid_count=minimum, ddof=ddof,
    )
    return Raster(values=vals, georef=raster.georef, valid=mask,
                  units=raster.units, name=raster.name)


def focal_count(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "ignore_invalid",
    min_valid_count: int | None = None,
    cval: float = 0.0,
) -> Raster:
    fp, halo = _validate_footprint(size, footprint)
    e = _validate_edge(edge)
    vn = _validate_valid_neighbor(valid_neighbor)
    minimum = _validate_min_valid_count(
        min_valid_count,
        valid_neighbor=vn,
        footprint_count=int(np.count_nonzero(fp)),
    )
    _, active = _neighborhood_windows(raster, fp, edge=e)
    count, result_valid = _neighborhood_validity(
        raster, fp, active, edge=e, valid_neighbor=vn,
        min_valid_count=minimum,
    )
    if vn == "ignore_invalid" and minimum is None:
        result_valid = np.ones(raster.shape, dtype=np.bool_)
        if e == "invalid":
            _apply_edge_invalid(result_valid, halo)
    final = count.astype(np.int64)
    return Raster(values=final, georef=raster.georef, valid=result_valid,
                  units=None, name=raster.name)


def focal_median(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    min_valid_count: int | None = None,
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn, minimum = _common_kwargs(
        size, footprint, edge, valid_neighbor, cval,
        min_valid_count=min_valid_count,
    )
    windows, active = _neighborhood_windows(raster, fp, edge=e)
    _, mask = _neighborhood_validity(
        raster, fp, active, edge=e, valid_neighbor=vn,
        min_valid_count=minimum,
    )
    output_dtype = _output_dtype(raster.dtype, "median")
    vals = np.zeros(raster.shape, dtype=output_dtype)
    for index in np.ndindex(raster.shape):
        if not mask[index]:
            continue
        selected = windows[index][active[index]]
        if selected.size == 0:
            continue
        if raster.dtype.kind in "biu":
            ordered = sorted(int(value) for value in selected)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                value = float(ordered[middle])
            else:
                value = float(Fraction(ordered[middle - 1] + ordered[middle], 2))
            vals[index] = value
        else:
            vals[index] = np.median(selected)
    return Raster(values=vals, georef=raster.georef, valid=mask,
                  units=raster.units, name=raster.name)


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------


def convolve(
    raster: Raster,
    kernel: np.ndarray,
    *,
    normalize: bool = False,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    min_valid_count: int | None = None,
    cval: float = 0.0,
) -> Raster:
    kernel = _validate_convolution_kernel(kernel)

    e = _validate_edge(edge)
    vn = _validate_valid_neighbor(valid_neighbor)
    minimum = _validate_min_valid_count(
        min_valid_count,
        valid_neighbor=vn,
        footprint_count=int(kernel.size),
    )
    if normalize:
        s = float(np.sum(np.abs(kernel)))
        if s > 0:
            kernel = kernel / s

    fp = np.ones(kernel.shape, dtype=np.bool_)
    halo = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    output_dtype = _output_dtype(raster.dtype, "mean")
    vals = raster.values.astype(output_dtype, copy=False)
    vld = raster.valid.copy()
    padded, padded_mask, crop = _pad_array(vals, vld, halo, e, cval=cval)
    scipy_mode = _SCIPY_MODE_MAP[e]

    if vn == "ignore_invalid":
        padded[~padded_mask] = 0.0
    else:
        padded[~padded_mask] = 0.0

    execution_kernel = kernel.astype(output_dtype, copy=False)
    result_padded = ndimage.convolve(
        padded, execution_kernel, output=output_dtype,
        mode=scipy_mode, cval=cval,
    )

    if vn == "require_all":
        mask_count = ndimage.convolve(
            padded_mask.astype(np.float64), np.ones(kernel.shape),
            mode=scipy_mode, cval=0.0,
        )
        result_valid = np.isclose(mask_count[crop], float(np.sum(fp)))
    elif vn == "propagate_center":
        result_valid = vld.copy()
    else:
        if minimum is None:
            result_valid = np.ones(raster.shape, dtype=np.bool_)
        else:
            mask_count = ndimage.convolve(
                padded_mask.astype(np.float64), np.ones(kernel.shape),
                mode=scipy_mode, cval=0.0,
            )
            result_valid = mask_count[crop] >= minimum

    if e == "invalid":
        _apply_edge_invalid(result_valid, halo)

    return Raster(
        values=result_padded[crop],
        georef=raster.georef, valid=result_valid,
        units=raster.units, name=raster.name,
    )


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------


def _require_boolean_raster(raster: Raster) -> None:
    if raster.values.dtype != np.dtype(np.bool_):
        raise MapAlgebraDTypeError(
            "Morphology operations require boolean raster inputs.",
            code="map_algebra_requires_boolean",
            details={"dtype": str(raster.dtype)},
        )


def _masked_morph(
    raster: Raster,
    struct: np.ndarray | None,
    op_fn: Any,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
) -> Raster:
    _require_boolean_raster(raster)
    fp, _ = _validate_footprint(size, footprint)
    struct_el = struct if struct is not None else fp
    values = raster.values.copy()
    values[~raster.valid] = False
    result = op_fn(values.astype(np.uint8), structure=struct_el).astype(np.bool_)
    return Raster(values=result, georef=raster.georef,
                  valid=raster.valid, units=raster.units, name=raster.name)


def dilate(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    structure: np.ndarray | None = None,
) -> Raster:
    return _masked_morph(raster, structure, ndimage.binary_dilation,
                         size=size, footprint=footprint)


def erode(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    structure: np.ndarray | None = None,
) -> Raster:
    return _masked_morph(raster, structure, ndimage.binary_erosion,
                         size=size, footprint=footprint)


def opening(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    structure: np.ndarray | None = None,
) -> Raster:
    return _masked_morph(raster, structure, ndimage.binary_opening,
                         size=size, footprint=footprint)


def closing(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    structure: np.ndarray | None = None,
) -> Raster:
    return _masked_morph(raster, structure, ndimage.binary_closing,
                         size=size, footprint=footprint)


def majority(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
) -> Raster:
    _require_boolean_raster(raster)
    fp, _ = _validate_footprint(size, footprint)
    values = raster.values.copy()
    values[~raster.valid] = False
    mean_vals = ndimage.generic_filter(
        values.astype(np.float64), np.mean, footprint=fp,
        mode="reflect",
    )
    result_values = mean_vals > 0.5
    return Raster(values=result_values, georef=raster.georef,
                  valid=raster.valid, units=raster.units, name=raster.name)
