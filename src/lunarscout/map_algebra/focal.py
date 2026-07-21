from __future__ import annotations

from typing import Any, Literal

import numpy as np
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
        if fp.size == 0:
            raise MapAlgebraError("Footprint must not be empty.",
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


def _output_dtype(raster: Raster, operation: str) -> np.dtype:
    src = raster.values.dtype
    if operation in ("sum", "count"):
        if np.issubdtype(src, np.floating):
            return np.dtype(np.float64)
        return np.dtype(np.int64)
    if operation in ("mean", "std", "range", "median"):
        if src == np.dtype(np.float64):
            return np.dtype(np.float64)
        return np.dtype(np.float32)
    if operation in ("min", "max"):
        return src
    return src


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
) -> tuple[np.ndarray, np.ndarray]:
    halo = (footprint.shape[0] // 2, footprint.shape[1] // 2)
    vals = raster.values.astype(np.float64, copy=False)
    vld = raster.valid.copy()
    padded, padded_mask, crop = _pad_array(vals, vld, halo, edge, cval=cval)
    scipy_mode = _SCIPY_MODE_MAP[edge]
    ftp_count = float(np.sum(footprint))

    if valid_neighbor == "ignore_invalid":
        padded[~padded_mask] = np.nan
        result_padded = ndimage.generic_filter(
            padded, func, footprint=footprint, mode=scipy_mode, cval=np.nan,
        )
        count_padded = ndimage.generic_filter(
            padded_mask.astype(np.float64), np.sum, footprint=footprint,
            mode=scipy_mode, cval=0.0,
        )
        result_valid = count_padded[crop] > 0
    elif valid_neighbor == "propagate_center":
        padded[~padded_mask] = np.nan
        result_padded = ndimage.generic_filter(
            padded, func, footprint=footprint, mode=scipy_mode, cval=np.nan,
        )
        result_valid = vld.copy()
    else:
        padded[~padded_mask] = np.nan
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
        result_valid = count_padded[crop] > 0
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
    cval=0.0, op_name="focal",
):
    fp, halo = _validate_footprint(size, footprint)
    edge_v = _validate_edge(edge)
    vn_v = _validate_valid_neighbor(vn)
    return fp, halo, edge_v, vn_v


def focal_sum(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn = _common_kwargs(size, footprint, edge, valid_neighbor, cval)
    vals, mask = _focal_special(
        raster, fp,
        lambda x, **kw: ndimage.generic_filter(x, np.sum, **kw),
        edge=e, valid_neighbor=vn,
        neutral=0.0, cval=cval,
        output_dtype=_output_dtype(raster, "sum"),
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
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn = _common_kwargs(size, footprint, edge, valid_neighbor, cval)
    vals, mask = _focal_generic(
        raster, fp, edge=e, valid_neighbor=vn,
        func=lambda w: np.nanmean(w), neutral=np.nan, cval=cval,
        output_dtype=_output_dtype(raster, "mean"), op_name="mean",
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
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn = _common_kwargs(size, footprint, edge, valid_neighbor, cval)
    vals, mask = _focal_special(raster, fp, ndimage.minimum_filter,
                                 edge=e, valid_neighbor=vn,
                                 neutral=np.inf, cval=cval,
                                 output_dtype=_output_dtype(raster, "min"))
    return Raster(values=vals, georef=raster.georef, valid=mask,
                  units=raster.units, name=raster.name)


def focal_max(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn = _common_kwargs(size, footprint, edge, valid_neighbor, cval)
    vals, mask = _focal_special(raster, fp, ndimage.maximum_filter,
                                 edge=e, valid_neighbor=vn,
                                 neutral=-np.inf, cval=cval,
                                 output_dtype=_output_dtype(raster, "max"))
    return Raster(values=vals, georef=raster.georef, valid=mask,
                  units=raster.units, name=raster.name)


def focal_range(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    cval: float = 0.0,
) -> Raster:
    min_r = focal_min(raster, size=size, footprint=footprint, edge=edge,
                       valid_neighbor=valid_neighbor, cval=cval)
    max_r = focal_max(raster, size=size, footprint=footprint, edge=edge,
                       valid_neighbor=valid_neighbor, cval=cval)
    from .local import subtract
    result = subtract(max_r, min_r)
    dst = _output_dtype(raster, "range")
    if result.dtype != dst:
        from .local import cast
        result = cast(result, dst, casting="unsafe")
    return result


def focal_std(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    cval: float = 0.0,
    ddof: int = 0,
) -> Raster:
    fp, halo, e, vn = _common_kwargs(size, footprint, edge, valid_neighbor, cval)
    ddof_val = int(ddof)
    vals, mask = _focal_generic(
        raster, fp, edge=e, valid_neighbor=vn,
        func=lambda w: np.nanstd(w, ddof=ddof_val), neutral=np.nan, cval=cval,
        output_dtype=_output_dtype(raster, "std"), op_name="std",
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
    cval: float = 0.0,
) -> Raster:
    fp, halo = _validate_footprint(size, footprint)
    e = _validate_edge(edge)
    vn = _validate_valid_neighbor(valid_neighbor)
    vals = raster.values.astype(np.float64, copy=False)
    vld = raster.valid.copy()
    padded, padded_mask, crop = _pad_array(vals, vld, halo, e, cval=cval)
    scipy_mode = _SCIPY_MODE_MAP[e]
    ftp_count = float(np.sum(fp))

    if vn == "ignore_invalid":
        count_padded = ndimage.generic_filter(
            padded_mask.astype(np.float64), np.sum, footprint=fp,
            mode=scipy_mode, cval=0.0,
        )
        result_valid = np.ones(raster.shape, dtype=np.bool_)
    elif vn == "propagate_center":
        count_padded = ndimage.generic_filter(
            padded_mask.astype(np.float64), np.sum, footprint=fp,
            mode=scipy_mode, cval=0.0,
        )
        result_valid = vld.copy()
    else:
        count_padded = ndimage.generic_filter(
            padded_mask.astype(np.float64), np.sum, footprint=fp,
            mode=scipy_mode, cval=0.0,
        )
        result_valid = np.isclose(count_padded[crop], ftp_count)

    if e == "invalid":
        _apply_edge_invalid(result_valid, halo)

    final = count_padded[crop].astype(np.int64)
    return Raster(values=final, georef=raster.georef, valid=result_valid,
                  units=None, name=raster.name)


def focal_median(
    raster: Raster,
    *,
    size: int | tuple[int, int] | None = None,
    footprint: np.ndarray | None = None,
    edge: EdgeMode = "invalid",
    valid_neighbor: ValidNeighbor = "require_all",
    cval: float = 0.0,
) -> Raster:
    fp, halo, e, vn = _common_kwargs(size, footprint, edge, valid_neighbor, cval)
    vals, mask = _focal_generic(
        raster, fp, edge=e, valid_neighbor=vn,
        func=lambda w: np.nanmedian(w), neutral=np.nan, cval=cval,
        output_dtype=_output_dtype(raster, "median"), op_name="median",
    )
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
    cval: float = 0.0,
) -> Raster:
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise MapAlgebraError("Convolution kernel must be two-dimensional.",
                               code="map_algebra_invalid_kernel")
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise MapAlgebraError("Convolution kernel dimensions must be odd.",
                               code="map_algebra_invalid_kernel")
    if not np.all(np.isfinite(kernel)):
        raise MapAlgebraError("Convolution kernel must contain only finite values.",
                               code="map_algebra_invalid_kernel")

    e = _validate_edge(edge)
    vn = _validate_valid_neighbor(valid_neighbor)
    if normalize:
        s = float(np.sum(np.abs(kernel)))
        if s > 0:
            kernel = kernel / s

    fp = np.ones(kernel.shape, dtype=np.bool_)
    halo = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    vals = raster.values.astype(np.float64, copy=False)
    vld = raster.valid.copy()
    padded, padded_mask, crop = _pad_array(vals, vld, halo, e, cval=cval)
    scipy_mode = _SCIPY_MODE_MAP[e]

    if vn == "ignore_invalid":
        padded[~padded_mask] = 0.0
    else:
        padded[~padded_mask] = 0.0

    result_padded = ndimage.convolve(padded, kernel, mode=scipy_mode, cval=cval)

    if vn == "require_all":
        mask_count = ndimage.convolve(
            padded_mask.astype(np.float64), np.ones(kernel.shape),
            mode=scipy_mode, cval=0.0,
        )
        result_valid = np.isclose(mask_count[crop], float(np.sum(fp)))
    elif vn == "propagate_center":
        result_valid = vld.copy()
    else:
        result_valid = np.ones(raster.shape, dtype=np.bool_)

    if e == "invalid":
        _apply_edge_invalid(result_valid, halo)

    return Raster(
        values=result_padded[crop].astype(_output_dtype(raster, "mean")),
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
