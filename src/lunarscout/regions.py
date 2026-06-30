from __future__ import annotations

from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .errors import RegionOperationError
from .georeference import GeoReference


CleanupMode: TypeAlias = Literal["none", "erosion", "opening"]
Comparator: TypeAlias = Literal[">=", "<="]
NodataArgument: TypeAlias = int | float | None | Literal["auto"]

_CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def _validate_georef_shape(mask: Any, georef: GeoReference | None) -> np.ndarray:
    values = np.asanyarray(mask)
    if values.ndim != 2:
        raise RegionOperationError(
            "Region operations require a two-dimensional mask.",
            code="region_invalid_array_shape",
            details={"shape": list(values.shape)},
        )
    if georef is not None:
        expected = (int(georef.height), int(georef.width))
        if values.shape != expected:
            raise RegionOperationError(
                "Mask shape does not match GeoReference dimensions.",
                code="region_shape_mismatch",
                details={"shape": list(values.shape), "expected_shape": list(expected)},
            )
    return values


def _resolve_nodata(
    georef: GeoReference | None,
    nodata: NodataArgument,
) -> int | float | None:
    if isinstance(nodata, str):
        if nodata != "auto":
            raise RegionOperationError(
                "nodata must be 'auto', None, or a numeric value.",
                code="region_invalid_argument",
                details={"argument": "nodata", "value": nodata},
            )
        return None if georef is None else georef.nodata
    return nodata


def _validate_cleanup(cleanup: str, iterations: int) -> tuple[CleanupMode, int]:
    mode = str(cleanup).strip().lower()
    if mode not in {"none", "erosion", "opening"}:
        raise RegionOperationError(
            "cleanup must be 'none', 'erosion', or 'opening'.",
            code="region_invalid_argument",
            details={"argument": "cleanup", "value": cleanup},
        )
    if isinstance(iterations, bool):
        valid_iterations = False
    else:
        try:
            value = int(iterations)
            valid_iterations = value == iterations and value >= 0
        except (TypeError, ValueError, OverflowError):
            valid_iterations = False
            value = -1
    if not valid_iterations:
        raise RegionOperationError(
            "iterations must be an integer greater than or equal to zero.",
            code="region_invalid_argument",
            details={"argument": "iterations", "value": iterations},
        )
    return mode, value  # type: ignore[return-value]


def _mask_and_invalid(
    mask: Any,
    georef: GeoReference | None,
    nodata: NodataArgument,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], int | float | None, bool]:
    values = _validate_georef_shape(mask, georef)
    effective_nodata = _resolve_nodata(georef, nodata)
    if np.ma.isMaskedArray(values):
        raw_values = np.asarray(values.data)
        invalid = np.ma.getmaskarray(values).astype(bool, copy=True)
        masked_input = True
    else:
        raw_values = np.asarray(values)
        invalid = np.zeros(raw_values.shape, dtype=bool)
        masked_input = False
    if not (
        np.issubdtype(raw_values.dtype, np.bool_)
        or (
            np.issubdtype(raw_values.dtype, np.number)
            and not np.issubdtype(raw_values.dtype, np.complexfloating)
        )
    ):
        raise RegionOperationError(
            "Region masks must use a real numeric or Boolean NumPy dtype.",
            code="region_unsupported_datatype",
            details={"dtype": str(raw_values.dtype)},
        )
    if effective_nodata is not None:
        if isinstance(effective_nodata, float) and np.isnan(effective_nodata):
            if np.issubdtype(raw_values.dtype, np.inexact):
                invalid |= np.isnan(raw_values)
        else:
            invalid |= raw_values == effective_nodata
    mask_bool = np.asarray(raw_values != 0, dtype=bool)
    mask_bool[invalid] = False
    nodata_active = effective_nodata is not None or masked_input
    return mask_bool, invalid, effective_nodata, nodata_active


def _clean_mask(
    mask: NDArray[np.bool_],
    *,
    cleanup: CleanupMode,
    iterations: int,
) -> NDArray[np.bool_]:
    if cleanup == "none" or iterations == 0:
        return np.array(mask, dtype=bool, copy=True)
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=bool)
    if cleanup == "erosion":
        return np.asarray(
            ndimage.binary_erosion(mask, structure=structure, iterations=iterations),
            dtype=bool,
        )
    return np.asarray(
        ndimage.binary_opening(mask, structure=structure, iterations=iterations),
        dtype=bool,
    )


def _label(mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    from scipy import ndimage

    labels, _count = ndimage.label(mask, structure=_CONNECTIVITY)
    return np.asarray(labels, dtype=np.int32)


def _output_georef(
    georef: GeoReference | None,
    effective_nodata: int | float | None,
) -> GeoReference | None:
    if georef is None:
        return None
    return georef.with_nodata(effective_nodata)


def _numeric_output(
    values: NDArray[np.int32],
    *,
    invalid: NDArray[np.bool_],
    effective_nodata: int | float | None,
    nodata_active: bool,
) -> NDArray[np.int32] | np.ma.MaskedArray:
    if not nodata_active:
        return values
    if effective_nodata is None:
        return np.ma.array(values, mask=invalid, copy=False)
    if (
        isinstance(effective_nodata, (int, np.integer))
        or (
            isinstance(effective_nodata, (float, np.floating))
            and np.isfinite(effective_nodata)
            and float(effective_nodata).is_integer()
        )
    ):
        numeric_nodata = int(effective_nodata)
        limits = np.iinfo(np.int32)
        if int(limits.min) <= numeric_nodata <= int(limits.max):
            output = np.array(values, copy=True)
            output[invalid] = numeric_nodata
            return output
    raise RegionOperationError(
        "Nodata cannot be represented by the int32 region output dtype.",
        code="region_unrepresentable_nodata",
        details={"nodata": effective_nodata, "dtype": "int32"},
    )


def _boolean_output(
    values: NDArray[np.bool_],
    *,
    invalid: NDArray[np.bool_],
    nodata_active: bool,
) -> NDArray[np.bool_] | np.ma.MaskedArray:
    if not nodata_active:
        return values
    return np.ma.array(values, mask=invalid, copy=False)


def label_regions(
    mask: Any,
    georef: GeoReference | None = None,
    *,
    nodata: NodataArgument = "auto",
    cleanup: CleanupMode = "none",
    iterations: int = 0,
) -> tuple[NDArray[np.int32] | np.ma.MaskedArray, GeoReference | None]:
    """Label eight-neighbor connected true regions."""

    mode, iteration_count = _validate_cleanup(cleanup, iterations)
    mask_bool, invalid, effective_nodata, nodata_active = _mask_and_invalid(
        mask,
        georef,
        nodata,
    )
    labels = _label(
        _clean_mask(mask_bool, cleanup=mode, iterations=iteration_count)
    )
    return (
        _numeric_output(
            labels,
            invalid=invalid,
            effective_nodata=effective_nodata,
            nodata_active=nodata_active,
        ),
        _output_georef(georef, effective_nodata),
    )


def region_sizes(
    mask: Any,
    georef: GeoReference | None = None,
    *,
    nodata: NodataArgument = "auto",
    cleanup: CleanupMode = "none",
    iterations: int = 0,
) -> tuple[NDArray[np.int32] | np.ma.MaskedArray, GeoReference | None]:
    """Return each true pixel's eight-neighbor connected-region size."""

    mode, iteration_count = _validate_cleanup(cleanup, iterations)
    mask_bool, invalid, effective_nodata, nodata_active = _mask_and_invalid(
        mask,
        georef,
        nodata,
    )
    labels = _label(
        _clean_mask(mask_bool, cleanup=mode, iterations=iteration_count)
    )
    counts = np.bincount(labels.ravel())
    sizes = np.asarray(counts[labels], dtype=np.int32)
    sizes[labels == 0] = 0
    return (
        _numeric_output(
            sizes,
            invalid=invalid,
            effective_nodata=effective_nodata,
            nodata_active=nodata_active,
        ),
        _output_georef(georef, effective_nodata),
    )


def filter_regions_by_size(
    mask: Any,
    georef: GeoReference | None = None,
    *,
    threshold: float,
    comparator: Comparator = ">=",
    nodata: NodataArgument = "auto",
    cleanup: CleanupMode = "none",
    iterations: int = 0,
) -> tuple[NDArray[np.bool_] | np.ma.MaskedArray, GeoReference | None]:
    """Keep original regions selected by cleanup-aware seed-region sizes."""

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RegionOperationError(
            "threshold must be a finite number greater than or equal to zero.",
            code="region_invalid_argument",
            details={"argument": "threshold", "value": threshold},
        ) from exc
    if not np.isfinite(threshold_value) or threshold_value < 0.0:
        raise RegionOperationError(
            "threshold must be a finite number greater than or equal to zero.",
            code="region_invalid_argument",
            details={"argument": "threshold", "value": threshold},
        )
    if comparator not in {">=", "<="}:
        raise RegionOperationError(
            "comparator must be '>=' or '<='.",
            code="region_invalid_argument",
            details={"argument": "comparator", "value": comparator},
        )
    mode, iteration_count = _validate_cleanup(cleanup, iterations)
    mask_bool, invalid, effective_nodata, nodata_active = _mask_and_invalid(
        mask,
        georef,
        nodata,
    )
    original_labels = _label(mask_bool)
    seed_labels = _label(
        _clean_mask(mask_bool, cleanup=mode, iterations=iteration_count)
    )
    seed_counts = np.bincount(seed_labels.ravel())
    keep_seed_ids = (
        seed_counts >= threshold_value
        if comparator == ">="
        else seed_counts <= threshold_value
    )
    if keep_seed_ids.size:
        keep_seed_ids[0] = False
    kept_original_ids = np.unique(original_labels[keep_seed_ids[seed_labels]])
    keep_original_ids = np.zeros(int(original_labels.max()) + 1, dtype=bool)
    if kept_original_ids.size:
        keep_original_ids[kept_original_ids] = True
    keep_original_ids[0] = False
    filtered = keep_original_ids[original_labels]
    filtered[invalid] = False
    return (
        _boolean_output(filtered, invalid=invalid, nodata_active=nodata_active),
        _output_georef(georef, effective_nodata),
    )


def find_borders(
    mask: Any,
    georef: GeoReference | None = None,
    *,
    nodata: NodataArgument = "auto",
) -> tuple[NDArray[np.bool_] | np.ma.MaskedArray, GeoReference | None]:
    """Return the eight-neighbor internal border of true mask regions."""

    from scipy import ndimage

    mask_bool, invalid, effective_nodata, nodata_active = _mask_and_invalid(
        mask,
        georef,
        nodata,
    )
    structure = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(
        mask_bool,
        structure=structure,
        border_value=0,
    )
    borders = np.logical_and(mask_bool, np.logical_not(eroded))
    borders[invalid] = False
    return (
        _boolean_output(borders, invalid=invalid, nodata_active=nodata_active),
        _output_georef(georef, effective_nodata),
    )
