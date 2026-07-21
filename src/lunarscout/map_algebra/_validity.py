from __future__ import annotations

from typing import Any, Literal

import numpy as np

_NumericErrorsPolicy = Literal["invalid", "keep", "raise"]


def intersect_validity(*masks: np.ndarray[Any, np.dtype[np.bool_]]) -> np.ndarray[Any, np.dtype[np.bool_]]:
    result = masks[0].copy()
    for m in masks[1:]:
        result = result & m
    return result


def where_validity(
    condition_values: np.ndarray[Any, np.dtype[np.bool_]],
    condition_valid: np.ndarray[Any, np.dtype[np.bool_]],
    x_valid: np.ndarray[Any, np.dtype[np.bool_]],
    y_valid: np.ndarray[Any, np.dtype[np.bool_]],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    x_branch_valid = condition_values & x_valid
    y_branch_valid = ~condition_values & y_valid
    return condition_valid & (x_branch_valid | y_branch_valid)


def coalesce_validity(
    *validity_masks: np.ndarray[Any, np.dtype[np.bool_]],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    result = np.zeros(validity_masks[0].shape, dtype=np.bool_)
    for m in validity_masks:
        result = result | m
    return result


def apply_numeric_domain(
    values: np.ndarray[Any, Any],
    valid: np.ndarray[Any, np.dtype[np.bool_]],
    *,
    operation: str,
    policy: _NumericErrorsPolicy = "invalid",
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    if policy == "keep":
        return valid.copy()
    if policy == "invalid":
        nonfinite = ~np.isfinite(values)
        new_invalid = valid & ~nonfinite
        return new_invalid
    if policy == "raise":
        if np.any(~np.isfinite(values[valid])):
            raise ValueError(
                f"Operation '{operation}' produced non-finite values at valid pixels."
            )
        return valid.copy()
    raise ValueError(f"Unknown numeric errors policy: {policy}")
