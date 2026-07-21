from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..errors import (
    GeoTiffOpenError,
    RasterValidationError,
)
from ..georeference import GeoReference
from ..geotiff import read_geotiff as _read_geotiff
from ..raster import (
    Raster,
    _validate_nodata_representable,
    _valid_from_nodata,
)

from .local import (
    absolute,
    add,
    arccos,
    arcsin,
    arctan,
    arctan2,
    cast,
    ceil,
    clip,
    coalesce,
    cos,
    degrees,
    divide,
    equal,
    exp,
    fill_invalid,
    floor,
    floor_divide,
    greater,
    greater_equal,
    hypot,
    invalid,
    is_invalid,
    is_valid,
    isclose,
    less,
    less_equal,
    log,
    log10,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    negative,
    not_equal,
    positive,
    power,
    radians,
    remainder,
    round_half_even as round,
    set_invalid,
    sin,
    sqrt,
    square,
    subtract,
    tan,
    trunc,
    where,
)


def raster(
    values: NDArray[Any],
    georef: GeoReference,
    *,
    valid: NDArray[np.bool_] | None = None,
    nodata: int | float | None | Literal["auto"] = "auto",
    units: str | None = None,
    name: str | None = None,
    validity_provenance: str | None = None,
) -> Raster:
    """Construct a ``Raster`` from explicit values and georeferencing.

    When ``valid`` is provided it is used directly.  Otherwise validity is
    derived from ``nodata``: ``"auto"`` uses ``georef.nodata``, an explicit
    value uses that nodata, and ``None`` marks every pixel valid.
    """
    if np.ma.isMaskedArray(values):
        masked = np.ma.asarray(values)
        data = np.ma.getdata(masked)
        mask = np.ma.getmaskarray(masked)
        if valid is not None:
            raise RasterValidationError(
                "Cannot supply both a masked array and an explicit valid mask.",
                code="raster_conflicting_validity",
                details={},
            )
        resolved_nodata = georef.nodata if nodata == "auto" else nodata
        nodata_valid = _valid_from_nodata(data, resolved_nodata)
        valid = nodata_valid & ~mask
        if validity_provenance is None:
            validity_provenance = "masked-array+nodata"
        return Raster(
            values=data,
            georef=georef,
            valid=valid,
            units=units,
            name=name,
            validity_provenance=validity_provenance,
        )

    values = np.asarray(values)
    if valid is not None:
        valid = np.asarray(valid, dtype=np.bool_)
        if validity_provenance is None:
            validity_provenance = "explicit-caller"
    else:
        resolved_nodata = georef.nodata if nodata == "auto" else nodata
        resolved_nodata = _validate_nodata_representable(resolved_nodata, values.dtype)
        valid = _valid_from_nodata(values, resolved_nodata)
        if validity_provenance is None:
            if resolved_nodata is not None:
                validity_provenance = "nodata"
            else:
                validity_provenance = "all_valid"
    return Raster(
        values=values,
        georef=georef,
        valid=valid,
        units=units,
        name=name,
        validity_provenance=validity_provenance,
    )


def from_masked_array(
    values: np.ma.MaskedArray[Any, Any],
    georef: GeoReference,
    *,
    units: str | None = None,
    name: str | None = None,
) -> Raster:
    """Construct a ``Raster`` from a NumPy masked array.

    The mask becomes the validity array (inverted: masked means invalid).
    """
    values = np.ma.asarray(values)
    if values.ndim != 2:
        raise RasterValidationError(
            "Masked array values must be two-dimensional.",
            code="raster_invalid_shape",
            details={"ndim": int(values.ndim)},
        )
    data = np.ma.getdata(values)
    valid = ~np.ma.getmaskarray(values)
    return Raster(
        values=data,
        georef=georef,
        valid=valid,
        units=units,
        name=name,
        validity_provenance="masked-array",
    )


def from_existing(
    values: NDArray[Any],
    georef: GeoReference,
    *,
    units: str | None = None,
    name: str | None = None,
) -> Raster:
    """Wrap existing bare ``(values, georef)`` results in a ``Raster``.

    Validity is derived from ``georef.nodata``.
    """
    values = np.asarray(values)
    nodata = _validate_nodata_representable(georef.nodata, values.dtype)
    valid = _valid_from_nodata(values, nodata)
    provenance = "nodata" if nodata is not None else "all_valid"
    return Raster(
        values=values,
        georef=georef,
        valid=valid,
        units=units,
        name=name,
        validity_provenance=provenance,
    )


def to_existing(
    raster_obj: Raster,
    *,
    nodata: int | float | None = None,
) -> tuple[NDArray[Any], GeoReference]:
    """Convert a ``Raster`` back to a bare ``(values, georef)`` tuple.

    Invalid cells are filled with ``nodata`` and the returned
    ``GeoReference`` carries that nodata value.
    """
    validated_nodata = _validate_nodata_representable(nodata, raster_obj.values.dtype)
    if validated_nodata is not None:
        values = raster_obj.filled(validated_nodata)
    else:
        values = raster_obj.values.copy()
    georef = raster_obj.georef.with_nodata(validated_nodata)
    return values, georef


def _read_rasterio_validity_provenance(
    mask_flags: list[Any] | None,
    band_idx: int,
) -> str:
    if mask_flags is None:
        return "all_valid"
    flags = mask_flags[band_idx]
    if not hasattr(flags, "value"):
        return "all_valid"
    all_valid = getattr(flags.__class__, "all_valid", None)
    per_dataset = getattr(flags.__class__, "per_dataset", None)
    per_band = getattr(flags.__class__, "per_band", None)
    alpha = getattr(flags.__class__, "alpha", None)
    nodata_flag = getattr(flags.__class__, "nodata", None)
    if all_valid is not None and (int(flags) & int(all_valid)):
        return "all_valid"
    parts = []
    if per_dataset is not None and (int(flags) & int(per_dataset)):
        parts.append("per_dataset")
    if per_band is not None and (int(flags) & int(per_band)):
        parts.append("per_band")
    if alpha is not None and (int(flags) & int(alpha)):
        parts.append("alpha")
    if nodata_flag is not None and (int(flags) & int(nodata_flag)):
        parts.append("nodata")
    return "+".join(parts) if parts else "all_valid"


def read(
    path: str | Path,
    *,
    band: int = 1,
    units: str | None = None,
    name: str | None = None,
) -> Raster:
    """Read a single-band GeoTIFF as a ``Raster``.

    Combines the GDAL band mask, dataset mask, alpha, and declared nodata
    into a canonical validity mask and preserves the native band values.
    """
    import rasterio as _rasterio

    path = Path(path).expanduser().resolve()
    values, georef = _read_geotiff(path, band=band)
    if georef is None:
        raise GeoTiffOpenError(
            "GeoTIFF is not georeferenced; cannot construct a Raster.",
            code="geotiff_unreferenced",
            details={"path": str(path)},
        )

    dataset = _rasterio.open(path)
    with dataset:
        mask_flags = dataset.mask_flag_enums if hasattr(dataset, "mask_flag_enums") else None
        read_mask: NDArray[np.bool_] | None = None
        try:
            mask_arrays = dataset.read_masks(band)
            if mask_arrays.ndim == 3 and mask_arrays.shape[0] == 1:
                mask_arrays = mask_arrays[0]
            if mask_arrays.ndim == 2:
                read_mask = np.asarray(mask_arrays, dtype=np.bool_)
        except Exception:
            read_mask = None

    flag_provenance = _read_rasterio_validity_provenance(mask_flags, band - 1)

    if read_mask is not None and flag_provenance not in ("all_valid", "nodata"):
        valid = read_mask
        provenance = flag_provenance
    elif "nodata" in flag_provenance or flag_provenance == "nodata":
        valid = _valid_from_nodata(values, georef.nodata)
        if read_mask is not None and flag_provenance != "nodata":
            valid = valid & read_mask
        provenance = flag_provenance
    else:
        valid = _valid_from_nodata(values, georef.nodata)
        provenance = flag_provenance

    georef = georef.with_nodata(None)
    raster_name = name if name is not None else path.stem
    return Raster(
        values=values,
        georef=georef,
        valid=valid,
        units=units,
        name=raster_name,
        validity_provenance="geotiff:" + provenance,
    )
