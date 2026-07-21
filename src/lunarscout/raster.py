from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .errors import RasterValidationError
from .georeference import GeoReference

if TYPE_CHECKING:
    from .map_algebra._model import RasterExpression


def _is_expression(obj: object) -> bool:
    clsname = type(obj).__qualname__
    return clsname == "RasterExpression"


def _is_temporal_expression(obj: object) -> bool:
    clsname = type(obj).__qualname__
    return clsname == "TemporalRasterExpression"

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
        if isinstance(other, Raster):
            from .alignment import require_same_grid
            require_same_grid(self.georef, other.georef)
            result_valid = self.valid & other.valid
            result_values = np.where(
                result_valid,
                self.values == other.values,
                False,
            )
            return Raster(values=result_values, georef=self.georef, valid=result_valid, name=self.name)
        if isinstance(other, (int, float)):
            result_valid = self.valid.copy()
            result_values = np.where(result_valid, self.values == other, False)
            return Raster(values=result_values, georef=self.georef, valid=result_valid, name=self.name)
        if _is_temporal_expression(other):
            return other.__eq__(self)  # type: ignore[return-value]
        return NotImplemented

    def __ne__(self, other: object) -> Raster:  # type: ignore[override]
        if isinstance(other, Raster):
            from .alignment import require_same_grid
            require_same_grid(self.georef, other.georef)
            result_valid = self.valid & other.valid
            result_values = np.where(
                result_valid,
                self.values != other.values,
                False,
            )
            return Raster(values=result_values, georef=self.georef, valid=result_valid, name=self.name)
        if isinstance(other, (int, float)):
            result_valid = self.valid.copy()
            result_values = np.where(result_valid, self.values != other, False)
            return Raster(values=result_values, georef=self.georef, valid=result_valid, name=self.name)
        if _is_temporal_expression(other):
            return other.__ne__(self)  # type: ignore[return-value]
        return NotImplemented

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: object) -> Raster:
        if _is_expression(other):
            from .map_algebra._sources import constant
            return constant(self).__add__(other)  # type: ignore[return-value]
        if _is_temporal_expression(other):
            return other.__radd__(self)  # type: ignore[return-value]
        from .map_algebra.local import add as _add
        return _add(self, other)  # type: ignore[return-value]

    def __radd__(self, other: object) -> Raster:
        if _is_expression(other):
            return other.__add__(self)  # type: ignore[return-value]
        if _is_temporal_expression(other):
            return other.__add__(self)  # type: ignore[return-value]
        from .map_algebra.local import add as _add
        return _add(other, self)  # type: ignore[return-value]

    def __sub__(self, other: object) -> Raster:
        if _is_expression(other):
            from .map_algebra._sources import constant
            return constant(self).__sub__(other)  # type: ignore[return-value]
        if _is_temporal_expression(other):
            return other.__rsub__(self)  # type: ignore[return-value]
        from .map_algebra.local import subtract as _sub
        return _sub(self, other)  # type: ignore[return-value]

    def __rsub__(self, other: object) -> Raster:
        if _is_expression(other):
            return other.__sub__(self)  # type: ignore[return-value]
        if _is_temporal_expression(other):
            return other.__sub__(self)  # type: ignore[return-value]
        from .map_algebra.local import subtract as _sub
        return _sub(other, self)  # type: ignore[return-value]

    def __mul__(self, other: object) -> Raster:
        if _is_expression(other):
            from .map_algebra._sources import constant
            return constant(self).__mul__(other)  # type: ignore[return-value]
        if _is_temporal_expression(other):
            return other.__rmul__(self)  # type: ignore[return-value]
        from .map_algebra.local import multiply as _mul
        return _mul(self, other)  # type: ignore[return-value]

    def __rmul__(self, other: object) -> Raster:
        if _is_expression(other):
            return other.__mul__(self)  # type: ignore[return-value]
        if _is_temporal_expression(other):
            return other.__mul__(self)  # type: ignore[return-value]
        from .map_algebra.local import multiply as _mul
        return _mul(other, self)  # type: ignore[return-value]

    def __truediv__(self, other: object) -> Raster:
        if _is_expression(other):
            from .map_algebra._sources import constant
            return constant(self).__truediv__(other)  # type: ignore[return-value]
        if _is_temporal_expression(other):
            return other.__rtruediv__(self)  # type: ignore[return-value]
        from .map_algebra.local import divide as _div
        return _div(self, other)  # type: ignore[return-value]

    def __rtruediv__(self, other: object) -> Raster:
        if _is_expression(other):
            return other.__truediv__(self)  # type: ignore[return-value]
        if _is_temporal_expression(other):
            return other.__truediv__(self)  # type: ignore[return-value]
        from .map_algebra.local import divide as _div
        return _div(other, self)  # type: ignore[return-value]

    def __floordiv__(self, other: object) -> Raster:
        from .map_algebra._kernels import _floor_divide as _kernel
        if isinstance(other, Raster):
            from .map_algebra._eager import _dispatch_binary_raster_raster
            from .map_algebra._validation import _require_common_grid
            _require_common_grid([self, other])
            return _dispatch_binary_raster_raster(self, other, _kernel, operation="floordiv")
        elif isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            return _dispatch_binary_raster_scalar(self, other, _kernel, operation="floordiv")
        return NotImplemented

    def __rfloordiv__(self, other: object) -> Raster:
        from .map_algebra._kernels import _floor_divide as _kernel
        if isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            return _dispatch_binary_raster_scalar(
                self, other,
                lambda arr, s: _kernel(np.full(arr.shape, s, dtype=np.float64), arr),
                operation="floordiv",
            )
        return NotImplemented

    def __mod__(self, other: object) -> Raster:
        from .map_algebra._kernels import _remainder as _kernel
        if isinstance(other, Raster):
            from .map_algebra._eager import _dispatch_binary_raster_raster
            from .map_algebra._validation import _require_common_grid
            _require_common_grid([self, other])
            return _dispatch_binary_raster_raster(self, other, _kernel, operation="mod")
        elif isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            return _dispatch_binary_raster_scalar(self, other, _kernel, operation="mod")
        return NotImplemented

    def __rmod__(self, other: object) -> Raster:
        from .map_algebra._kernels import _remainder as _kernel
        if isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            return _dispatch_binary_raster_scalar(
                self, other,
                lambda arr, s: _kernel(np.full(arr.shape, s, dtype=np.float64), arr),
                operation="mod",
            )
        return NotImplemented

    def __pow__(self, other: object) -> Raster:
        from .map_algebra._kernels import _power as _kernel
        if isinstance(other, Raster):
            from .map_algebra._eager import _dispatch_binary_raster_raster
            from .map_algebra._validation import _require_common_grid
            _require_common_grid([self, other])
            return _dispatch_binary_raster_raster(self, other, _kernel, operation="power")
        elif isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            return _dispatch_binary_raster_scalar(self, other, _kernel, operation="power")
        return NotImplemented

    def __rpow__(self, other: object) -> Raster:
        if isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            return _dispatch_binary_raster_scalar(
                self, other,
                lambda arr, s: np.power(np.full(arr.shape, np.float64(s)), arr),
                operation="power",
            )
        return NotImplemented

    def __neg__(self) -> Raster:
        from .map_algebra.local import negative as _neg
        return _neg(self)

    def __pos__(self) -> Raster:
        return self

    def __abs__(self) -> Raster:
        from .map_algebra.local import absolute as _abs
        return _abs(self)

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------

    def __lt__(self, other: object) -> Raster:  # type: ignore[override]
        if isinstance(other, Raster):
            from .map_algebra._eager import _dispatch_binary_raster_raster
            from .map_algebra._kernels import _less
            from .map_algebra._validation import _require_common_grid
            _require_common_grid([self, other])
            return _dispatch_binary_raster_raster(self, other, _less, operation="less")
        elif isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            from .map_algebra._kernels import _less
            return _dispatch_binary_raster_scalar(self, other, _less, operation="less")
        elif _is_temporal_expression(other):
            return other.__gt__(self)  # type: ignore[return-value]
        return NotImplemented

    def __le__(self, other: object) -> Raster:  # type: ignore[override]
        if isinstance(other, Raster):
            from .map_algebra._eager import _dispatch_binary_raster_raster
            from .map_algebra._kernels import _less_equal
            from .map_algebra._validation import _require_common_grid
            _require_common_grid([self, other])
            return _dispatch_binary_raster_raster(self, other, _less_equal, operation="less_equal")
        elif isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            from .map_algebra._kernels import _less_equal
            return _dispatch_binary_raster_scalar(self, other, _less_equal, operation="less_equal")
        elif _is_temporal_expression(other):
            return other.__ge__(self)  # type: ignore[return-value]
        return NotImplemented

    def __gt__(self, other: object) -> Raster:  # type: ignore[override]
        if isinstance(other, Raster):
            from .map_algebra._eager import _dispatch_binary_raster_raster
            from .map_algebra._kernels import _greater
            from .map_algebra._validation import _require_common_grid
            _require_common_grid([self, other])
            return _dispatch_binary_raster_raster(self, other, _greater, operation="greater")
        elif isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            from .map_algebra._kernels import _greater
            return _dispatch_binary_raster_scalar(self, other, _greater, operation="greater")
        elif _is_temporal_expression(other):
            return other.__lt__(self)  # type: ignore[return-value]
        return NotImplemented

    def __ge__(self, other: object) -> Raster:  # type: ignore[override]
        if isinstance(other, Raster):
            from .map_algebra._eager import _dispatch_binary_raster_raster
            from .map_algebra._kernels import _greater_equal
            from .map_algebra._validation import _require_common_grid
            _require_common_grid([self, other])
            return _dispatch_binary_raster_raster(self, other, _greater_equal, operation="greater_equal")
        elif isinstance(other, (int, float)):
            from .map_algebra._eager import _dispatch_binary_raster_scalar
            from .map_algebra._kernels import _greater_equal
            return _dispatch_binary_raster_scalar(self, other, _greater_equal, operation="greater_equal")
        elif _is_temporal_expression(other):
            return other.__le__(self)  # type: ignore[return-value]
        return NotImplemented

    # ------------------------------------------------------------------
    # Boolean / bitwise operators
    # ------------------------------------------------------------------

    def __and__(self, other: object) -> Raster:
        if isinstance(other, Raster):
            from .map_algebra.local import logical_and as _la
            return _la(self, other)
        return NotImplemented

    def __rand__(self, other: object) -> Raster:
        if isinstance(other, Raster):
            from .map_algebra.local import logical_and as _la
            return _la(other, self)
        return NotImplemented

    def __or__(self, other: object) -> Raster:
        if isinstance(other, Raster):
            from .map_algebra.local import logical_or as _lo
            return _lo(self, other)
        return NotImplemented

    def __ror__(self, other: object) -> Raster:
        if isinstance(other, Raster):
            from .map_algebra.local import logical_or as _lo
            return _lo(other, self)
        return NotImplemented

    def __xor__(self, other: object) -> Raster:
        if isinstance(other, Raster):
            from .map_algebra.local import logical_xor as _lx
            return _lx(self, other)
        return NotImplemented

    def __rxor__(self, other: object) -> Raster:
        if isinstance(other, Raster):
            from .map_algebra.local import logical_xor as _lx
            return _lx(other, self)
        return NotImplemented

    def __invert__(self) -> Raster:
        from .map_algebra.local import logical_not as _ln
        return _ln(self)

    # ------------------------------------------------------------------
    # Rounding
    # ------------------------------------------------------------------

    def __round__(self, ndigits: int | None = None) -> Raster:
        from .map_algebra.local import round_half_even as _r
        return _r(self, ndigits=ndigits or 0)

    def __floor__(self) -> Raster:
        from .map_algebra.local import floor as _f
        return _f(self)

    def __ceil__(self) -> Raster:
        from .map_algebra.local import ceil as _c
        return _c(self)

    def __trunc__(self) -> Raster:
        from .map_algebra.local import trunc as _t
        return _t(self)

    def expression(self) -> RasterExpression:  # type: ignore[return-value]
        from .map_algebra._sources import constant
        return constant(self)

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
