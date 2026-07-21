from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import MapAlgebraExpressionError


CANONICAL_SCHEMA_VERSION = 3
NORMALIZATION_VERSION = 1


def normalize_canonical(value: Any) -> Any:
    """Return a typed, deterministic JSON value or reject the input."""
    if value is None:
        return {"type": "null"}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            encoded = "nan"
        elif np.isposinf(number):
            encoded = "+infinity"
        elif np.isneginf(number):
            encoded = "-infinity"
        else:
            encoded = number.hex()
        return {"type": "float", "hex": encoded}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if isinstance(value, np.dtype):
        return {"type": "dtype", "value": value.str}
    if isinstance(value, Enum):
        return {
            "type": "enum",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": normalize_canonical(value.value),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            _unsupported(value)
        return {
            "type": "ndarray",
            "dtype": value.dtype.str,
            "shape": [str(size) for size in value.shape],
            "values": normalize_canonical(value.tolist()),
        }
    if isinstance(value, (tuple, list)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [normalize_canonical(item) for item in value],
        }
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            _unsupported(value, reason="Canonical dictionary keys must be strings.")
        return {
            "type": "mapping",
            "items": [
                {"key": key, "value": normalize_canonical(value[key])}
                for key in sorted(value)
            ],
        }
    _unsupported(value)


def normalize_crs_wkt(value: str) -> dict[str, str]:
    """Normalize CRS text to the supported canonical WKT2 representation."""
    try:
        from pyproj import CRS

        canonical = CRS.from_user_input(value).to_wkt(version="WKT2_2019", pretty=False)
    except Exception as exc:
        raise MapAlgebraExpressionError(
            "CRS text cannot be represented canonically.",
            code="map_algebra_uncanonical_parameter",
            details={"type": "crs", "error": str(exc)},
        ) from exc
    return {"type": "crs_wkt2_2019", "value": canonical}


def _unsupported(value: Any, *, reason: str | None = None) -> None:
    raise MapAlgebraExpressionError(
        reason or f"Value of type {type(value).__name__} cannot be represented canonically.",
        code="map_algebra_uncanonical_parameter",
        details={"type": type(value).__name__},
    )
