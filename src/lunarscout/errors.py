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


class NativeError(LunarscoutError):
    default_code = "native_error"


class NativeUnavailableError(NativeError):
    default_code = "native_unavailable"


class NativeBootstrapError(NativeError):
    default_code = "native_bootstrap_error"


class NativeInputError(NativeError):
    default_code = "native_input_error"


class NativeTemporalError(NativeError):
    default_code = "native_temporal_error"


class NativeAllocationError(NativeTemporalError):
    default_code = "native_allocation_error"


class NativeProductError(NativeError):
    default_code = "native_product_error"


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
