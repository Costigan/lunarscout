from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS, Transformer

from .errors import CoordinateTransformError, GeoReferenceError


Coordinate: TypeAlias = float | NDArray[np.float64]
Anchor: TypeAlias = Literal["center", "corner"]


def _coordinate_inputs(
    first: ArrayLike,
    second: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    first_array, second_array = np.broadcast_arrays(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
    )
    scalar = first_array.ndim == 0
    return first_array, second_array, scalar


def _coordinate_outputs(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    *,
    scalar: bool,
) -> tuple[Coordinate, Coordinate]:
    if scalar:
        return float(first), float(second)
    return first, second


def _validate_anchor(anchor: str) -> Anchor:
    if anchor not in {"center", "corner"}:
        raise GeoReferenceError(
            "Pixel anchor must be 'center' or 'corner'.",
            code="georeference_invalid_anchor",
            details={"anchor": anchor},
        )
    return anchor  # type: ignore[return-value]


@dataclass(frozen=True)
class GeoReference:
    """Immutable geospatial interpretation of a two-dimensional raster band."""

    projection_wkt: str
    projection_proj4: str
    affine_transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float
    nodata: int | float | None

    def __post_init__(self) -> None:
        if not str(self.projection_wkt).strip():
            raise GeoReferenceError(
                "Projection WKT is required.",
                code="georeference_missing_projection_wkt",
            )
        if not str(self.projection_proj4).strip():
            raise GeoReferenceError(
                "Projection PROJ.4 text is required.",
                code="georeference_missing_projection_proj4",
            )
        if len(self.affine_transform) != 6:
            raise GeoReferenceError(
                "The affine transform must contain six coefficients.",
                code="georeference_invalid_affine",
            )
        if not np.all(np.isfinite(np.asarray(self.affine_transform, dtype=np.float64))):
            raise GeoReferenceError(
                "The affine transform must contain finite coefficients.",
                code="georeference_invalid_affine",
            )
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise GeoReferenceError(
                "Raster width and height must be positive.",
                code="georeference_invalid_dimensions",
                details={"width": self.width, "height": self.height},
            )
        if float(self.pixel_size_x) != float(self.affine_transform[1]):
            raise GeoReferenceError(
                "pixel_size_x must equal affine_transform[1].",
                code="georeference_inconsistent_pixel_size",
            )
        if float(self.pixel_size_y) != float(self.affine_transform[5]):
            raise GeoReferenceError(
                "pixel_size_y must equal affine_transform[5].",
                code="georeference_inconsistent_pixel_size",
            )
        determinant = (
            self.affine_transform[1] * self.affine_transform[5]
            - self.affine_transform[2] * self.affine_transform[4]
        )
        if abs(determinant) < 1e-15:
            raise GeoReferenceError(
                "The affine transform is not invertible.",
                code="georeference_noninvertible_affine",
            )

    @cached_property
    def _inverse_affine(self) -> tuple[float, float, float, float, float, float]:
        gt = self.affine_transform
        determinant = gt[1] * gt[5] - gt[2] * gt[4]
        return (
            -gt[0] * gt[5] / determinant + gt[2] * gt[3] / determinant,
            gt[5] / determinant,
            -gt[2] / determinant,
            gt[0] * gt[4] / determinant - gt[1] * gt[3] / determinant,
            -gt[4] / determinant,
            gt[1] / determinant,
        )

    @cached_property
    def _projected_crs(self) -> CRS:
        try:
            return CRS.from_wkt(self.projection_wkt)
        except Exception as exc:
            raise CoordinateTransformError(
                "Unable to construct the projected coordinate reference system.",
                details={"error": str(exc)},
            ) from exc

    @cached_property
    def _geographic_crs(self) -> CRS:
        try:
            geographic = self._projected_crs.geodetic_crs
            if geographic is None:
                raise ValueError("Projected CRS has no geographic CRS")
            return geographic
        except CoordinateTransformError:
            raise
        except Exception as exc:
            raise CoordinateTransformError(
                "Unable to construct the lunar geographic coordinate reference system.",
                details={"error": str(exc)},
            ) from exc

    @cached_property
    def _to_geographic(self) -> Transformer:
        try:
            return Transformer.from_crs(
                self._projected_crs,
                self._geographic_crs,
                always_xy=True,
            )
        except CoordinateTransformError:
            raise
        except Exception as exc:
            raise CoordinateTransformError(
                "Unable to construct projected-to-geographic transformation.",
                details={"error": str(exc)},
            ) from exc

    @cached_property
    def _to_projected(self) -> Transformer:
        try:
            return Transformer.from_crs(
                self._geographic_crs,
                self._projected_crs,
                always_xy=True,
            )
        except CoordinateTransformError:
            raise
        except Exception as exc:
            raise CoordinateTransformError(
                "Unable to construct geographic-to-projected transformation.",
                details={"error": str(exc)},
            ) from exc

    def pixel_to_projected(
        self,
        column: ArrayLike,
        row: ArrayLike,
        *,
        anchor: Anchor = "center",
    ) -> tuple[Coordinate, Coordinate]:
        anchor = _validate_anchor(anchor)
        columns, rows, scalar = _coordinate_inputs(column, row)
        if anchor == "center":
            columns = columns + 0.5
            rows = rows + 0.5
        gt = self.affine_transform
        easting = gt[0] + columns * gt[1] + rows * gt[2]
        northing = gt[3] + columns * gt[4] + rows * gt[5]
        return _coordinate_outputs(easting, northing, scalar=scalar)

    def projected_to_pixel(
        self,
        easting: ArrayLike,
        northing: ArrayLike,
        *,
        anchor: Anchor = "center",
    ) -> tuple[Coordinate, Coordinate]:
        anchor = _validate_anchor(anchor)
        eastings, northings, scalar = _coordinate_inputs(easting, northing)
        inverse = self._inverse_affine
        columns = inverse[0] + eastings * inverse[1] + northings * inverse[2]
        rows = inverse[3] + eastings * inverse[4] + northings * inverse[5]
        if anchor == "center":
            columns = columns - 0.5
            rows = rows - 0.5
        return _coordinate_outputs(columns, rows, scalar=scalar)

    def _transform_coordinates(
        self,
        first: ArrayLike,
        second: ArrayLike,
        *,
        transformation,
        direction: str,
    ) -> tuple[Coordinate, Coordinate]:
        first_array, second_array, scalar = _coordinate_inputs(first, second)
        try:
            first_transformed, second_transformed = transformation.transform(
                first_array.ravel(),
                second_array.ravel(),
            )
        except Exception as exc:
            raise CoordinateTransformError(
                f"Unable to transform {direction} coordinates.",
                details={"error": str(exc), "point_count": int(first_array.size)},
            ) from exc
        output_first = np.asarray(first_transformed, dtype=np.float64).reshape(first_array.shape)
        output_second = np.asarray(second_transformed, dtype=np.float64).reshape(second_array.shape)
        return _coordinate_outputs(output_first, output_second, scalar=scalar)

    def projected_to_lonlat(
        self,
        easting: ArrayLike,
        northing: ArrayLike,
    ) -> tuple[Coordinate, Coordinate]:
        return self._transform_coordinates(
            easting,
            northing,
            transformation=self._to_geographic,
            direction="projected-to-geographic",
        )

    def lonlat_to_projected(
        self,
        longitude: ArrayLike,
        latitude: ArrayLike,
    ) -> tuple[Coordinate, Coordinate]:
        return self._transform_coordinates(
            longitude,
            latitude,
            transformation=self._to_projected,
            direction="geographic-to-projected",
        )

    def pixel_to_lonlat(
        self,
        column: ArrayLike,
        row: ArrayLike,
        *,
        anchor: Anchor = "center",
    ) -> tuple[Coordinate, Coordinate]:
        easting, northing = self.pixel_to_projected(column, row, anchor=anchor)
        return self.projected_to_lonlat(easting, northing)

    def lonlat_to_pixel(
        self,
        longitude: ArrayLike,
        latitude: ArrayLike,
        *,
        anchor: Anchor = "center",
    ) -> tuple[Coordinate, Coordinate]:
        easting, northing = self.lonlat_to_projected(longitude, latitude)
        return self.projected_to_pixel(easting, northing, anchor=anchor)

    @overload
    def contains_pixel(self, column: float, row: float) -> bool: ...

    @overload
    def contains_pixel(
        self,
        column: ArrayLike,
        row: ArrayLike,
    ) -> NDArray[np.bool_]: ...

    def contains_pixel(self, column: ArrayLike, row: ArrayLike):
        columns, rows, scalar = _coordinate_inputs(column, row)
        result = (
            (columns >= 0.0)
            & (columns < float(self.width))
            & (rows >= 0.0)
            & (rows < float(self.height))
        )
        if scalar:
            return bool(result)
        return result

    def projected_bounds(self) -> tuple[float, float, float, float]:
        columns = np.asarray([0.0, self.width, 0.0, self.width], dtype=np.float64)
        rows = np.asarray([0.0, 0.0, self.height, self.height], dtype=np.float64)
        eastings, northings = self.pixel_to_projected(columns, rows, anchor="corner")
        assert isinstance(eastings, np.ndarray)
        assert isinstance(northings, np.ndarray)
        return (
            float(np.min(eastings)),
            float(np.min(northings)),
            float(np.max(eastings)),
            float(np.max(northings)),
        )

    def with_nodata(self, nodata: int | float | None) -> GeoReference:
        """Return the same spatial grid with a different band nodata value."""

        return replace(self, nodata=nodata)
