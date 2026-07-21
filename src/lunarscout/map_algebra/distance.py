from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from ..errors import DistanceFieldError, MapAlgebraDTypeError, MapAlgebraError
from ..raster import Raster


Metric = Literal["euclidean", "taxicab", "chessboard"]
DistanceUnit = Literal["pixels", "physical"]
InvalidOutput = Literal["preserve", "compute"]

_PHYSICAL_QUERY_CHUNK_SIZE = 1_000_000


def _validate_metric(metric: str) -> Metric:
    if metric not in ("euclidean", "taxicab", "chessboard"):
        raise MapAlgebraError(
            f"Unknown distance metric: '{metric}'.",
            code="map_algebra_invalid_distance_metric",
            details={"metric": metric},
        )
    return metric  # type: ignore[return-value]


def _validate_units(units: str) -> DistanceUnit:
    if units not in ("pixels", "physical"):
        raise MapAlgebraError(
            f"Unknown distance units: '{units}'. Must be 'pixels' or 'physical'.",
            code="map_algebra_invalid_distance_units",
            details={"units": units},
        )
    return units  # type: ignore[return-value]


def _validate_invalid_output(value: str) -> InvalidOutput:
    if value not in ("preserve", "compute"):
        raise MapAlgebraError(
            "invalid_output must be 'preserve' or 'compute'.",
            code="map_algebra_invalid_output_policy",
            details={"invalid_output": value},
        )
    return value  # type: ignore[return-value]


def _validate_max_distance(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DistanceFieldError(
            "max_distance must be a finite, non-negative number.",
            code="map_algebra_invalid_max_distance",
            details={"max_distance": repr(value)},
        ) from exc
    if not np.isfinite(normalized) or normalized < 0:
        raise DistanceFieldError(
            "max_distance must be a finite, non-negative number.",
            code="map_algebra_invalid_max_distance",
            details={"max_distance": normalized},
        )
    return normalized


def _physical_unit_name(georef: object) -> str:
    from pyproj import CRS

    try:
        crs = CRS.from_wkt(georef.projection_wkt)  # type: ignore[attr-defined]
    except Exception as exc:
        raise DistanceFieldError(
            "Unable to parse CRS for physical distance validation.",
            code="map_algebra_invalid_crs",
        ) from exc

    if crs.is_geographic:
        raise DistanceFieldError(
            "Physical distance on geographic/angular CRS is not supported "
            "in 0.2. Use units='pixels' or supply a projected CRS.",
            code="map_algebra_geographic_distance_unsupported",
        )
    if not crs.is_projected:
        raise DistanceFieldError(
            "Physical distance requires a projected two-dimensional CRS.",
            code="map_algebra_physical_crs_unsupported",
            details={"crs_type": crs.type_name},
        )

    axes = crs.axis_info
    if len(axes) < 2:
        raise DistanceFieldError(
            "The projected CRS does not declare two coordinate-axis units.",
            code="map_algebra_physical_units_unknown",
        )
    first_name = str(axes[0].unit_name or "").strip()
    second_name = str(axes[1].unit_name or "").strip()
    first_factor = float(axes[0].unit_conversion_factor or np.nan)
    second_factor = float(axes[1].unit_conversion_factor or np.nan)
    if (
        not first_name
        or not second_name
        or first_name != second_name
        or not np.isfinite(first_factor)
        or not np.isfinite(second_factor)
        or first_factor <= 0
        or second_factor <= 0
        or not np.isclose(first_factor, second_factor)
    ):
        raise DistanceFieldError(
            "The projected CRS must use one consistent linear unit on both axes.",
            code="map_algebra_physical_units_unsupported",
            details={
                "axis_units": [first_name or None, second_name or None],
                "axis_unit_factors": [first_factor, second_factor],
            },
        )
    return first_name


def _compute_pixel_distance(
    seeds: np.ndarray,
    metric: Metric,
    max_distance: float | None,
) -> np.ndarray:
    if metric == "euclidean":
        dist = distance_transform_edt(~seeds).astype(np.float64)
    elif metric == "taxicab":
        dist = _manhattan_distance(seeds)
    else:
        dist = _chessboard_distance(seeds)
    if max_distance is not None:
        np.minimum(dist, max_distance, out=dist)
    return dist


def _physical_points(
    flat_indices: np.ndarray,
    *,
    width: int,
    affine_transform: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    rows, cols = np.divmod(flat_indices, width)
    points = np.empty((flat_indices.size, 2), dtype=np.float64)
    points[:, 0] = (
        cols * float(affine_transform[1])
        + rows * float(affine_transform[2])
    )
    points[:, 1] = (
        cols * float(affine_transform[4])
        + rows * float(affine_transform[5])
    )
    return points


def _compute_physical_euclidean_distance(
    seeds: np.ndarray,
    affine_transform: tuple[float, float, float, float, float, float],
    max_distance: float | None,
) -> np.ndarray:
    """Return exact affine-plane distances in the projected CRS's units."""
    height, width = seeds.shape
    seed_indices = np.flatnonzero(seeds)
    seed_points = _physical_points(
        seed_indices,
        width=width,
        affine_transform=affine_transform,
    )
    tree = cKDTree(seed_points)
    result = np.empty(height * width, dtype=np.float64)
    for start in range(0, result.size, _PHYSICAL_QUERY_CHUNK_SIZE):
        stop = min(start + _PHYSICAL_QUERY_CHUNK_SIZE, result.size)
        query_indices = np.arange(start, stop, dtype=np.int64)
        query_points = _physical_points(
            query_indices,
            width=width,
            affine_transform=affine_transform,
        )
        result[start:stop] = tree.query(query_points, k=1, workers=1)[0]
    if max_distance is not None:
        np.minimum(result, max_distance, out=result)
    return result.reshape((height, width))


def _manhattan_distance(seeds: np.ndarray) -> np.ndarray:
    height, width = seeds.shape
    distance = np.full((height, width), np.inf, dtype=np.float64)
    distance[seeds] = 0.0
    for row in range(height):
        for col in range(width):
            if row > 0:
                distance[row, col] = min(distance[row, col], distance[row - 1, col] + 1)
            if col > 0:
                distance[row, col] = min(distance[row, col], distance[row, col - 1] + 1)
    for row in range(height - 1, -1, -1):
        for col in range(width - 1, -1, -1):
            if row < height - 1:
                distance[row, col] = min(distance[row, col], distance[row + 1, col] + 1)
            if col < width - 1:
                distance[row, col] = min(distance[row, col], distance[row, col + 1] + 1)
    return distance


def _chessboard_distance(seeds: np.ndarray) -> np.ndarray:
    height, width = seeds.shape
    distance = np.full((height, width), np.inf, dtype=np.float64)
    distance[seeds] = 0.0
    for row in range(height):
        for col in range(width):
            if row > 0:
                distance[row, col] = min(distance[row, col], distance[row - 1, col] + 1)
            if col > 0:
                distance[row, col] = min(distance[row, col], distance[row, col - 1] + 1)
            if row > 0 and col > 0:
                distance[row, col] = min(distance[row, col], distance[row - 1, col - 1] + 1)
            if row > 0 and col < width - 1:
                distance[row, col] = min(distance[row, col], distance[row - 1, col + 1] + 1)
    for row in range(height - 1, -1, -1):
        for col in range(width - 1, -1, -1):
            if row < height - 1:
                distance[row, col] = min(distance[row, col], distance[row + 1, col] + 1)
            if col < width - 1:
                distance[row, col] = min(distance[row, col], distance[row, col + 1] + 1)
            if row < height - 1 and col < width - 1:
                distance[row, col] = min(distance[row, col], distance[row + 1, col + 1] + 1)
            if row < height - 1 and col > 0:
                distance[row, col] = min(distance[row, col], distance[row + 1, col - 1] + 1)
    return distance


def distance_to(
    seeds: Raster,
    *,
    metric: Metric = "euclidean",
    units: DistanceUnit = "pixels",
    max_distance: float | None = None,
    invalid_output: InvalidOutput = "preserve",
) -> Raster:
    """Measure center-to-center distance to the nearest valid seed pixel.

    ``seeds`` must be a Boolean eager :class:`Raster`; only valid True cells
    are seeds. Pixel-unit Euclidean, taxicab, and chessboard metrics are
    supported. Physical distance is Euclidean in the projected CRS and uses
    both affine basis vectors, including anisotropy, rotation, and skew.

    ``max_distance`` is expressed in the selected output units. By default,
    input-invalid cells remain invalid without acting as barriers. Set
    ``invalid_output="compute"`` to mark calculated values there valid.

    This eager CPU implementation allocates output proportional to raster area
    and, for physical distance, a seed-coordinate tree plus bounded query
    batches. File-backed and CUDA execution are not implemented.
    """
    if seeds.values.dtype != np.dtype(np.bool_):
        raise MapAlgebraDTypeError(
            "Seeds raster must have boolean dtype.",
            code="map_algebra_requires_boolean",
            details={"dtype": str(seeds.values.dtype)},
        )
    metric = _validate_metric(metric)
    units = _validate_units(units)
    invalid_output = _validate_invalid_output(invalid_output)
    max_distance = _validate_max_distance(max_distance)

    valid_seeds = seeds.values & seeds.valid
    if not np.any(valid_seeds):
        raise DistanceFieldError(
            "No valid seed pixels found. At least one valid True pixel is required.",
            code="map_algebra_no_valid_seeds",
        )

    if units == "physical":
        if metric != "euclidean":
            raise DistanceFieldError(
                f"Physical units with metric='{metric}' are not supported. "
                "Taxicab and chessboard metrics are pixel-unit only.",
                code="map_algebra_physical_requires_euclidean",
            )
        output_units = _physical_unit_name(seeds.georef)
        values = _compute_physical_euclidean_distance(
            valid_seeds,
            seeds.georef.affine_transform,
            max_distance,
        )
    else:
        output_units = "pixels"
        values = _compute_pixel_distance(valid_seeds, metric, max_distance).astype(np.float32)

    output_valid = (
        seeds.valid.copy()
        if invalid_output == "preserve"
        else np.ones(seeds.shape, dtype=np.bool_)
    )
    return Raster(
        values=values,
        georef=seeds.georef,
        valid=output_valid,
        units=output_units,
    )


def signed_distance(
    mask: Raster,
    *,
    metric: Metric = "euclidean",
    units: DistanceUnit = "pixels",
    max_distance: float | None = None,
    invalid_output: InvalidOutput = "preserve",
) -> Raster:
    """Measure signed center-to-center distance between Boolean classes.

    Valid True cells receive positive distance to the nearest valid False
    cell; valid False cells receive negative distance to the nearest valid
    True cell. Invalid cells do not act as barriers or targets. Their output
    validity is preserved unless ``invalid_output="compute"`` is selected, in
    which case their stored Boolean payload chooses the sign/class.

    Metrics, units, affine behavior, clipping, execution mode, and memory
    behavior match :func:`distance_to`.
    """
    if mask.values.dtype != np.dtype(np.bool_):
        raise MapAlgebraDTypeError(
            "Signed distance mask must have boolean dtype.",
            code="map_algebra_requires_boolean",
            details={"dtype": str(mask.values.dtype)},
        )
    metric = _validate_metric(metric)
    units = _validate_units(units)
    invalid_output = _validate_invalid_output(invalid_output)
    max_distance = _validate_max_distance(max_distance)

    valid_true = mask.values & mask.valid
    valid_false = ~mask.values & mask.valid
    if not np.any(valid_true):
        raise DistanceFieldError(
            "No valid True pixels for signed distance.",
            code="map_algebra_no_valid_true",
        )
    if not np.any(valid_false):
        raise DistanceFieldError(
            "No valid False pixels for signed distance.",
            code="map_algebra_no_valid_false",
        )

    if units == "physical":
        if metric != "euclidean":
            raise DistanceFieldError(
                f"Physical units with metric='{metric}' are not supported. "
                "Taxicab and chessboard metrics are pixel-unit only.",
                code="map_algebra_physical_requires_euclidean",
            )
        output_units = _physical_unit_name(mask.georef)
        dist_to_true = _compute_physical_euclidean_distance(
            valid_true,
            mask.georef.affine_transform,
            max_distance,
        )
        dist_to_false = _compute_physical_euclidean_distance(
            valid_false,
            mask.georef.affine_transform,
            max_distance,
        )
        dtype = np.float64
    else:
        output_units = "pixels"
        dist_to_true = _compute_pixel_distance(valid_true, metric, max_distance)
        dist_to_false = _compute_pixel_distance(valid_false, metric, max_distance)
        dtype = np.float32

    values = np.where(mask.values, dist_to_false, -dist_to_true).astype(dtype)
    output_valid = (
        mask.valid.copy()
        if invalid_output == "preserve"
        else np.ones(mask.shape, dtype=np.bool_)
    )
    return Raster(
        values=values,
        georef=mask.georef,
        valid=output_valid,
        units=output_units,
    )
