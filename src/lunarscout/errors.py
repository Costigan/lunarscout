from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class LunarscoutError(Exception):
    """Base class for stable, structured Lunarscout errors."""

    default_code = "lunarscout_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code or self.default_code)
        self.details = dict(details or {})


class InputError(LunarscoutError):
    """An ordinary public operation argument is invalid."""

    default_code = "input_error"


class GridError(InputError):
    """Raster grids are invalid or incompatible for an operation."""

    default_code = "grid_error"


class VectorError(InputError):
    """Celestial vectors are invalid or inconsistent with their time axis."""

    default_code = "vector_error"


class ProductTimeError(InputError):
    """A product time axis or interval contract is invalid."""

    default_code = "product_time_error"


class HorizonError(LunarscoutError):
    """Base class for stored-horizon and horizon-generation failures."""

    default_code = "horizon_error"


class HorizonFormatError(HorizonError):
    """A stored horizon does not satisfy the public file contract."""

    default_code = "horizon_format_error"


class HorizonGenerationError(HorizonError):
    """Horizon calculation or publication failed."""

    default_code = "horizon_generation_error"


class ComputeBackendError(LunarscoutError):
    """The requested compute backend cannot complete the operation."""

    default_code = "compute_backend_error"


class CudaError(ComputeBackendError):
    """CUDA capability, initialization, compilation, or execution failed."""

    default_code = "cuda_error"


class ProductError(LunarscoutError):
    """Base class for horizon-derived product failures."""

    default_code = "product_error"


class ProductCalculationError(ProductError):
    """A horizon-derived scientific calculation failed."""

    default_code = "product_calculation_error"


class ProductStorageError(ProductError):
    """Product staging, persistence, or publication failed."""

    default_code = "product_storage_error"


class OperationCancelledError(LunarscoutError):
    """Cooperative cancellation was observed at a bounded work boundary."""

    default_code = "operation_cancelled"


class GeoTiffError(LunarscoutError):
    default_code = "geotiff_error"


class GeoTiffOpenError(GeoTiffError):
    default_code = "geotiff_open_error"


class GeoTiffBandError(GeoTiffError):
    default_code = "geotiff_band_error"


class GeoTiffDataTypeError(GeoTiffError):
    default_code = "geotiff_unsupported_datatype"


class GeoTiffMetadataError(GeoTiffError):
    default_code = "geotiff_metadata_error"


class GeoTiffWriteError(GeoTiffError):
    default_code = "geotiff_write_error"


class OutputExistsError(GeoTiffWriteError):
    default_code = "geotiff_output_exists"


class GeoReferenceError(LunarscoutError):
    default_code = "georeference_error"


class CoordinateTransformError(GeoReferenceError):
    default_code = "coordinate_transform_error"


class TerrainOperationError(LunarscoutError):
    default_code = "terrain_operation_error"


class RegionOperationError(LunarscoutError):
    default_code = "region_operation_error"


class AlignmentError(LunarscoutError):
    default_code = "alignment_error"


class GridMismatchError(AlignmentError):
    default_code = "grid_mismatch"


class ScenarioError(LunarscoutError):
    default_code = "scenario_error"


class ScenarioPathError(ScenarioError):
    default_code = "scenario_path_error"


class ScenarioStateError(ScenarioError):
    default_code = "scenario_state_error"


class ProductCatalogError(LunarscoutError):
    default_code = "product_catalog_error"


class SpiceError(LunarscoutError):
    default_code = "spice_error"


class SpiceKernelError(SpiceError):
    default_code = "spice_kernel_error"


class SpiceGeometryError(SpiceError):
    default_code = "spice_geometry_error"


class TemporalError(LunarscoutError):
    default_code = "temporal_error"


class TimeRangeError(TemporalError):
    default_code = "time_range_error"


class TemporalCubeError(TemporalError):
    default_code = "temporal_cube_error"


class TemporalOperationError(TemporalError):
    default_code = "temporal_operation_error"


class TemporalSeriesError(TemporalError):
    default_code = "temporal_series_error"


class TemporalSeriesOpenError(TemporalSeriesError):
    default_code = "temporal_series_open_error"


class TemporalSeriesWriteError(TemporalSeriesError):
    default_code = "temporal_series_write_error"


class TemporalLookupError(TemporalSeriesError):
    default_code = "temporal_lookup_error"


class MapAlgebraError(LunarscoutError):
    """Base class for map-algebra failures."""

    default_code = "map_algebra_error"


class RasterValidationError(MapAlgebraError):
    """A Raster value does not satisfy the public data contract."""

    default_code = "raster_validation_error"


class MapAlgebraGridError(MapAlgebraError):
    """Grids are incompatible for the requested map-algebra operation."""

    default_code = "map_algebra_grid_error"


class MapAlgebraDTypeError(MapAlgebraError):
    """The requested dtype is unsupported for the map-algebra operation."""

    default_code = "map_algebra_dtype_error"


class MapAlgebraUnitError(MapAlgebraError):
    """Units are incompatible or missing for the requested operation."""

    default_code = "map_algebra_unit_error"


class MapAlgebraExpressionError(MapAlgebraError):
    """A RasterExpression is invalid or cannot be planned."""

    default_code = "map_algebra_expression_error"


class MapAlgebraOperationError(MapAlgebraError):
    """A map-algebra operation failed during execution."""

    default_code = "map_algebra_operation_error"


class MapAlgebraStorageError(MapAlgebraError):
    """Map-algebra product staging, persistence, or publication failed."""

    default_code = "map_algebra_storage_error"


class DistanceFieldError(MapAlgebraError):
    """A distance-field operation failed."""

    default_code = "distance_field_error"
