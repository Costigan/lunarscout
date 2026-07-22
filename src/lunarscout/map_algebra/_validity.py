from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..errors import MapAlgebraOperationError

NumericErrorsPolicy = Literal["invalid", "keep", "raise"]
_NumericErrorsPolicy = NumericErrorsPolicy


def normalize_numeric_errors(value: str) -> NumericErrorsPolicy:
    if value not in {"invalid", "keep", "raise"}:
        raise MapAlgebraOperationError(
            "numeric_errors must be 'invalid', 'keep', or 'raise'.",
            code="map_algebra_invalid_numeric_errors",
            details={"numeric_errors": value},
        )
    return value  # type: ignore[return-value]


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
    domain_errors: np.ndarray[Any, np.dtype[np.bool_]] | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    policy = normalize_numeric_errors(policy)
    nonfinite = ~np.isfinite(values)
    failing = nonfinite if domain_errors is None else (nonfinite | domain_errors)
    if policy == "keep":
        return valid.copy()
    if policy == "invalid":
        new_invalid = valid & ~failing
        return new_invalid
    if policy == "raise":
        affected = valid & failing
        count = int(np.count_nonzero(affected))
        if count:
            raise MapAlgebraOperationError(
                f"Operation '{operation}' produced non-finite values at valid pixels.",
                code="map_algebra_numeric_error",
                details={"operation": operation, "affected_pixels": count},
            )
        return valid.copy()
    raise AssertionError("unreachable")
