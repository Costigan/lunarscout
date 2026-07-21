from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from .errors import RasterValidationError
from .georeference import GeoReference

_VALID_DTYPES = {
    np.dtype(np.bool_),
    np.dtype(np.uint8),
    np.dtype(np.int8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.uint32),
    np.dtype(np.int32),
    np.dtype(np.uint64),
    np.dtype(np.int64),
    np.dtype(np.float32),
    np.dtype(np.float64),
}


def _validate_raster_dtype(dtype: np.dtype[Any]) -> None:
    if np.dtype(dtype) not in _VALID_DTYPES:
        raise RasterValidationError(
            "Raster values must have a supported numeric or boolean dtype.",
            code="raster_unsupported_dtype",
            details={"dtype": str(dtype)},
        )


def _validate_nodata_representable(
    nodata: int | float | None,
    dtype: np.dtype[Any],
) -> int | float | None:
    if nodata is None:
        return None
    if np.issubdtype(dtype, np.integer):
        if not isinstance(nodata, (int, np.integer)) or isinstance(nodata, bool):
            raise RasterValidationError(
                "Integer raster nodata must be an integer value.",
                code="raster_unrepresentable_nodata",
                details={"dtype": str(dtype), "nodata": repr(nodata)},
            )
        int_nodata = int(nodata)
        limits = np.iinfo(dtype)
        if int_nodata < int(limits.min) or int_nodata > int(limits.max):
            raise RasterValidationError(
                "Nodata value is out of range for the raster dtype.",
                code="raster_unrepresentable_nodata",
                details={
                    "dtype": str(dtype),
                    "nodata": int_nodata,
                    "min": int(limits.min),
                    "max": int(limits.max),
                },
            )
        return int_nodata
    if np.issubdtype(dtype, np.floating):
        converted = dtype.type(nodata)
        if isinstance(nodata, (int, float)) and np.isfinite(float(nodata)) and not np.isfinite(converted):
            raise RasterValidationError(
                "Nodata value cannot be exactly represented by the raster dtype.",
                code="raster_unrepresentable_nodata",
                details={"dtype": str(dtype), "nodata": repr(nodata)},
            )
        return float(converted)
    return nodata


def _valid_from_nodata(
    values: NDArray[Any],
    nodata: int | float | None,
) -> NDArray[np.bool_]:
    if nodata is None:
        return np.ones(values.shape, dtype=np.bool_)
    if np.issubdtype(values.dtype, np.floating) and (
        isinstance(nodata, (int, float)) and np.isnan(float(nodata))
    ):
        return ~np.isnan(values)
    if np.issubdtype(values.dtype, np.integer) and isinstance(nodata, (int, float)):
        return values != np.asarray(nodata, dtype=values.dtype)
    return values != np.asarray(nodata, dtype=values.dtype)


@dataclass(frozen=True, slots=True, eq=False)
class Raster:
    """Eager, in-memory raster value with explicit spatial and validity metadata.

    ``eq=False`` is deliberate because ``==`` returns a Boolean ``Raster``
    of per-cell comparisons.  Use the named helpers ``Raster.array_equal()``,
    ``Raster.allclose()``, ``Raster.same_grid()``, and
    ``Raster.same_metadata()`` for whole-raster comparison.

    Implicit truth testing is unavailable by design.  ``bool(raster)``
    raises ``TypeError`` with actionable guidance.
    """

    values: NDArray[Any]
    georef: GeoReference
    valid: NDArray[np.bool_]
    units: str | None = None
    name: str | None = None
    validity_provenance: str | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise RasterValidationError(
                "Raster values must be a two-dimensional NumPy array.",
                code="raster_invalid_shape",
                details={"ndim": int(self.values.ndim)},
            )
        _validate_raster_dtype(self.values.dtype)
        if self.valid.ndim != 2:
            raise RasterValidationError(
                "The validity mask must be a two-dimensional NumPy array.",
                code="raster_invalid_validity_shape",
                details={"ndim": int(self.valid.ndim)},
            )
        if self.valid.dtype != np.dtype(np.bool_):
            raise RasterValidationError(
                "The validity mask must have bool dtype.",
                code="raster_invalid_validity_dtype",
                details={"dtype": str(self.valid.dtype)},
            )
        expected_shape = (int(self.georef.height), int(self.georef.width))
        if self.values.shape != expected_shape:
            raise RasterValidationError(
                "Raster values shape does not match GeoReference dimensions.",
                code="raster_shape_mismatch",
                details={
                    "shape": list(self.values.shape),
                    "expected_shape": list(expected_shape),
                },
            )
        if self.valid.shape != expected_shape:
            raise RasterValidationError(
                "Validity mask shape does not match GeoReference dimensions.",
                code="raster_validity_shape_mismatch",
                details={
                    "shape": list(self.valid.shape),
                    "expected_shape": list(expected_shape),
                },
            )

    # ------------------------------------------------------------------
    # Truth testing
    # ------------------------------------------------------------------

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "Raster does not support implicit truth testing. "
            "Use .all_valid, .array_equal(), .allclose(), "
            "or an explicit named helper."
        )

    # ------------------------------------------------------------------
    # Cell-by-cell comparison operators
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> Raster:  # type: ignore[override]
        if not isinstance(other, Raster):
            return NotImplemented
        from .alignment import require_same_grid

        require_same_grid(self.georef, other.georef)
        result_valid = self.valid & other.valid
        result_values = np.where(
            result_valid,
            self.values == other.values,
            False,
        )
        return Raster(
            values=result_values,
            georef=self.georef,
            valid=result_valid,
            name=self.name,
        )

    def __ne__(self, other: object) -> Raster:  # type: ignore[override]
        if not isinstance(other, Raster):
            return NotImplemented
        from .alignment import require_same_grid

        require_same_grid(self.georef, other.georef)
        result_valid = self.valid & other.valid
        result_values = np.where(
            result_valid,
            self.values != other.values,
            False,
        )
        return Raster(
            values=result_values,
            georef=self.georef,
            valid=result_valid,
            name=self.name,
        )

    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        total = self.shape[0] * self.shape[1]
        parts = [
            f"Raster(shape={self.shape}, dtype={self.dtype.name}",
            f"valid=all" if self.all_valid else f"valid={total - self.invalid_count}/{total}",
        ]
        if self.units is not None:
            parts.append(f"units={self.units!r}")
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        if self.validity_provenance is not None:
            parts.append(f"provenance={self.validity_provenance!r}")
        return ", ".join(parts) + ")"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.georef.height), int(self.georef.width))

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    @property
    def height(self) -> int:
        return int(self.georef.height)

    @property
    def width(self) -> int:
        return int(self.georef.width)

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes + self.valid.nbytes)

    @property
    def all_valid(self) -> bool:
        return bool(np.all(self.valid))

    @property
    def invalid_count(self) -> int:
        return int(np.sum(~self.valid))

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def copy(self) -> Raster:
        return Raster(
            values=self.values.copy(),
            georef=self.georef,
            valid=self.valid.copy(),
            units=self.units,
            name=self.name,
            validity_provenance=self.validity_provenance,
        )

    def readonly(self) -> Raster:
        values = self.values.copy()
        values.flags.writeable = False
        valid = self.valid.copy()
        valid.flags.writeable = False
        return Raster(
            values=values,
            georef=self.georef,
            valid=valid,
            units=self.units,
            name=self.name,
            validity_provenance=self.validity_provenance,
        )

    def filled(self, value: int | float) -> NDArray[Any]:
        fill = np.asarray(value, dtype=self.values.dtype)
        result = self.values.copy()
        result[~self.valid] = fill
        return result

    def masked(self) -> np.ma.MaskedArray[Any, Any]:
        return np.ma.array(self.values, mask=~self.valid)

    # ------------------------------------------------------------------
    # Non-mutating metadata helpers
    # ------------------------------------------------------------------

    def with_name(self, name: str | None) -> Raster:
        return Raster(
            values=self.values,
            georef=self.georef,
            valid=self.valid,
            units=self.units,
            name=name,
            validity_provenance=self.validity_provenance,
        )

    def with_units(self, units: str | None) -> Raster:
        return Raster(
            values=self.values,
            georef=self.georef,
            valid=self.valid,
            units=units,
            name=self.name,
            validity_provenance=self.validity_provenance,
        )

    def with_validity(self, valid: NDArray[np.bool_]) -> Raster:
        if valid.shape != self.shape:
            raise RasterValidationError(
                "Validity mask shape must match the raster shape.",
                code="raster_validity_shape_mismatch",
                details={"shape": list(valid.shape), "expected_shape": list(self.shape)},
            )
        if valid.dtype != np.dtype(np.bool_):
            raise RasterValidationError(
                "Validity mask must have bool dtype.",
                code="raster_invalid_validity_dtype",
                details={"dtype": str(valid.dtype)},
            )
        return Raster(
            values=self.values,
            georef=self.georef,
            valid=valid,
            units=self.units,
            name=self.name,
            validity_provenance=self.validity_provenance,
        )

    def with_georef(self, georef: GeoReference) -> Raster:
        expected_shape = (int(georef.height), int(georef.width))
        if self.shape != expected_shape:
            raise RasterValidationError(
                "GeoReference dimensions do not match raster shape.",
                code="raster_shape_mismatch",
                details={"shape": list(self.shape), "expected_shape": list(expected_shape)},
            )
        return Raster(
            values=self.values,
            georef=georef,
            valid=self.valid,
            units=self.units,
            name=self.name,
            validity_provenance=self.validity_provenance,
        )

    # ------------------------------------------------------------------
    # Grid and metadata comparison
    # ------------------------------------------------------------------

    def same_grid(self, other: Raster) -> bool:
        from .alignment import same_grid as _same_grid

        return _same_grid(self.georef, other.georef)

    def same_metadata(self, other: Raster) -> bool:
        return (
            self.same_grid(other)
            and self.dtype == other.dtype
            and self.units == other.units
            and self.name == other.name
        )

    def array_equal(
        self,
        other: Raster,
        *,
        equal_invalid_payload: bool = False,
    ) -> bool:
        from .alignment import require_same_grid

        require_same_grid(self.georef, other.georef)
        valid_equal = bool(np.array_equal(self.valid, other.valid))
        if not valid_equal:
            return False
        if equal_invalid_payload:
            return bool(np.array_equal(self.values, other.values))
        return bool(np.array_equal(self.values[self.valid], other.values[self.valid]))

    def allclose(
        self,
        other: Raster,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        from .alignment import require_same_grid

        require_same_grid(self.georef, other.georef)
        valid_equal = bool(np.array_equal(self.valid, other.valid))
        if not valid_equal:
            return False
        return bool(
            np.allclose(
                self.values[self.valid],
                other.values[self.valid],
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
        )
