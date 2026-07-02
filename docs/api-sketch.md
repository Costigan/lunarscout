# Lunarscout Python API Sketch

Status: evolving implemented API with executable examples; later managed-state
surfaces remain design proposals.

This document narrows the first public `lunarscout` surface described in
`LUNARSCOUT_PYTHON_SURFACE_PLAN.md`. The executable programs under
`examples/` validate implemented local APIs and continue to
serve as design specifications for later managed surfaces.

## Design Rules

- NumPy arrays are the public representation of raster values.
- NumPy dtypes are the public representation of raster scalar types.
- `GeoReference` carries the geospatial interpretation of an array.
- The local API is eager and does not require a public `Raster` expression
  type.
- GDAL owns GeoTIFF input/output and CRS transformations.
- Managed computation uses an explicit plan, separate from local NumPy
  evaluation, and reuses Lunar Analyst's existing job contracts.
- FastAPI remains the authoritative state owner in the normal application
  topology.
- Filesystem-only code may create files but may not mutate `scenario.db`,
  register products, or publish layers.
- Alignment and reprojection are always explicit.
- Temporal calculations store and compute time in UTC.
- Importing `lunarscout` does not initialize `pythonnet`.
- Production native calls pass through `MoonlibBridge`.

## Imports and Public Types

The common notebook import is:

```python
import lunarscout as ls
import numpy as np
```

The proposed public types are:

```python
ls.GeoReference
ls.Scenario
ls.TimeRange
ls.ComputationPlan
ls.Job
ls.ScenarioState
ls.LocalScenarioState
ls.RemoteScenarioState
```

There is no public `Raster`, `MaterializedRaster`, `ScalarType`, or custom
array type in the initial design. A GeoTIFF band is read as a normal
`numpy.ndarray` using the TIFF band's native dtype. Spatial metadata travels
as a separate `GeoReference` value.

## GeoTIFF Input

The primary local raster input function is:

```python
def read_geotiff(
    filename: str | Path,
    band: int = 1,
) -> tuple[np.ndarray, GeoReference | None]: ...
```

Example:

```python
dem, georef = ls.read_geotiff("primary_dem.tif")
```

Required behavior:

- `band` uses GDAL's one-based band numbering and defaults to band 1.
- Exactly one band is read. The returned array has shape `(height, width)`.
- The returned array uses the selected TIFF band's native NumPy dtype. The
  function does not normalize values to `float32` or any other dtype.
- `GeoReference.nodata` is the selected band's actual GDAL nodata value. The
  function does not replace it with `NaN`, invent a sentinel, or reinterpret
  valid values.
- Missing nodata is represented by `None`.
- The GDAL dataset is closed before the function returns.
- A readable TIFF without projection or affine georeferencing returns
  `(array, None)` instead of a partial `GeoReference`.
- A nonexistent file, non-GeoTIFF input, unsupported datatype, invalid band,
  or out-of-range band produces a specific public exception.

Masking is not implicit. Callers that want a masked array can construct one
from the returned array and `georef.nodata`, or a later explicit helper can be
added.

## `GeoReference`

`GeoReference` is an immutable description of the geospatial interpretation
of a two-dimensional array.

```python
@dataclass(frozen=True)
class GeoReference:
    projection_wkt: str
    projection_proj4: str
    affine_transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float
    nodata: int | float | None
```

Field contracts:

- `projection_wkt` preserves the dataset's WKT projection.
- `projection_proj4` is the equivalent PROJ.4 representation produced from
  the same spatial reference.
- `affine_transform` preserves all six GDAL geotransform coefficients in GDAL
  order.
- `width` and `height` are raster dimensions in pixels.
- `pixel_size_x` and `pixel_size_y` expose the signed pixel-size coefficients
  from the affine transform. The full affine transform remains authoritative,
  especially for rotated rasters.
- `nodata` is exactly the nodata value reported for the selected band, or
  `None` if the band does not declare one.

The array dtype is deliberately not duplicated in `GeoReference`; it is
available from `array.dtype`. Nodata is included because it is required to
interpret and round-trip the selected band.

### Coordinate Conventions

- Pixel coordinates are expressed as `(column, row)`, never `(row, column)`.
- Projected coordinates are `(easting, northing)`.
- Geographic coordinates are `(longitude, latitude)` in degrees.
- Coordinate transformations use traditional GIS axis order even if the CRS
  definition declares another axis order.
- Integer `(column, row)` values identify pixel centers by default.
- `anchor="corner"` exposes raw GDAL pixel-corner affine behavior.
- Reverse transformations return floating-point pixel coordinates and never
  silently round or clamp them.
- Pixel bounds are `0 <= column < width` and `0 <= row < height` when using
  center-based pixel coordinates.
- Longitude/latitude conversion uses the lunar geographic CRS associated with
  the projected CRS. It must not silently use Earth EPSG:4326.

### Coordinate Helpers

```python
class GeoReference:
    def pixel_to_projected(
        self,
        column: float | np.ndarray,
        row: float | np.ndarray,
        *,
        anchor: Literal["center", "corner"] = "center",
    ) -> tuple[float | np.ndarray, float | np.ndarray]: ...

    def projected_to_pixel(
        self,
        easting: float | np.ndarray,
        northing: float | np.ndarray,
        *,
        anchor: Literal["center", "corner"] = "center",
    ) -> tuple[float | np.ndarray, float | np.ndarray]: ...

    def projected_to_lonlat(
        self,
        easting: float | np.ndarray,
        northing: float | np.ndarray,
    ) -> tuple[float | np.ndarray, float | np.ndarray]: ...

    def lonlat_to_projected(
        self,
        longitude: float | np.ndarray,
        latitude: float | np.ndarray,
    ) -> tuple[float | np.ndarray, float | np.ndarray]: ...

    def pixel_to_lonlat(
        self,
        column: float | np.ndarray,
        row: float | np.ndarray,
        *,
        anchor: Literal["center", "corner"] = "center",
    ) -> tuple[float | np.ndarray, float | np.ndarray]: ...

    def lonlat_to_pixel(
        self,
        longitude: float | np.ndarray,
        latitude: float | np.ndarray,
        *,
        anchor: Literal["center", "corner"] = "center",
    ) -> tuple[float | np.ndarray, float | np.ndarray]: ...

    def contains_pixel(
        self,
        column: float | np.ndarray,
        row: float | np.ndarray,
    ) -> bool | np.ndarray: ...

    def projected_bounds(self) -> tuple[float, float, float, float]: ...
```

`projected_bounds()` returns `(min_easting, min_northing, max_easting,
max_northing)` and evaluates all four raster corners so rotated rasters are
handled correctly.

The implementation should use GDAL/OSR transformation facilities and the full
affine transform. Coordinate transformation objects may be created lazily and
cached by the immutable object. Failures to construct a lunar geographic
transform must be reported explicitly.

Array inputs follow NumPy broadcasting rules and return arrays with the
broadcast shape. Scalar inputs return scalar floats.

Example:

```python
dem, georef = ls.read_geotiff("primary_dem.tif", band=1)

easting, northing = georef.pixel_to_projected(100, 200)
longitude, latitude = georef.pixel_to_lonlat(100, 200)
column, row = georef.lonlat_to_pixel(longitude, latitude)
```

### Existing Implementation Sources

There is no existing Python class with this complete contract. The closest
current pieces are:

- `backend.jobs.raster_transform.GridSpec`, which carries only CRS, transform,
  width, and height;
- `backend.jobs.map_algebra.TargetGrid`, which is similarly narrow and tied to
  a DEM path;
- image georeferencing contracts and pixel-readout conversion logic in
  `backend.contracts.models` and `backend.api.routers.v1`;
- native C# `ElevationMap`, which has related conversion methods but is coupled
  to native elevation processing and is not an approved direct Python entry
  surface.

The reusable Python implementation lives in the standalone `lunarscout`
package. A later slice should make the image metadata/readout API consume it
rather than maintaining a second coordinate-conversion implementation.

## GeoTIFF Output

The corresponding local output function is:

```python
def write_geotiff(
    filename: str | Path,
    array: np.ndarray,
    georef: GeoReference,
    *,
    overwrite: bool = False,
) -> Path: ...
```

Required behavior:

- Accept a two-dimensional NumPy array.
- Preserve the array's NumPy dtype as the output band's TIFF datatype.
- Validate `array.shape == (georef.height, georef.width)`.
- Write `georef.projection_wkt`, the full affine transform, and
  `georef.nodata` without changing their meaning.
- Default to 128 x 128 tiles, DEFLATE compression, predictor 2 for integers,
  predictor 3 for floating point, and GDAL `BIGTIFF=IF_SAFER`.
- Write through a same-directory temporary file and atomically replace the
  destination on supported Linux filesystems. Reject an existing output unless
  `overwrite=True`.
- Return the absolute output `Path`.
- Do not register a product, publish a layer, or mutate `scenario.db`.

Example:

```python
dem, georef = ls.read_geotiff("primary_dem.tif")
slope_deg = ls.slope(dem, georef)
candidate = np.where(slope_deg <= 8.0, 1, georef.nodata)

output = ls.write_geotiff(
    "analysis/candidate_flat_sites.tif",
    candidate,
    georef,
)
```

The example must handle `georef.nodata is None` explicitly rather than
assuming every input declares a nodata value.

## UTC Construction

Lunarscout provides a UTC-aware constructor so callers do not need to repeat
`tzinfo=timezone.utc`:

```python
start = ls.utc_datetime(2027, 1, 1)
```

It returns a standard timezone-aware Python `datetime`. Lunarscout APIs treat
naive datetime inputs and timezone-free ISO strings as UTC by default; aware
non-UTC inputs are converted to UTC. Temporal NumPy arrays use UTC
`numpy.datetime64`, with explicit conversion at the .NET bridge boundary.

## Scenario State Ownership

A `Scenario` is initially a safe provider of scenario naming conventions and
paths. Scenario state mutation is delegated to an explicit `ScenarioState`
dependency.

```python
scenario = ls.open_scenario("/data/mons_mouton")
dem_path = scenario.dem_path()
dem, georef = ls.read_geotiff(dem_path)
```

This filesystem-only form can resolve standard paths and create output files.
It cannot register products or publish layers. The canonical paths in the
active scenario contract are `dem.tif` for the primary DEM and
`horizons` for horizon tiles.

An attached notebook uses FastAPI as the state owner:

```python
state = ls.RemoteScenarioState("http://127.0.0.1:8000")
scenario = ls.open_scenario("/data/mons_mouton", state=state)
```

Attached state is designed but not implemented in v0.1. Passing a non-`None`
`state` currently raises `ScenarioStateError` rather than silently bypassing
the FastAPI ownership boundary.

A standalone process may become the authoritative owner explicitly:

```python
with ls.LocalScenarioState(workspace_root="/data") as state:
    scenario = state.open_scenario("mons_mouton")
```

`LocalScenarioState` should be extracted from the existing scenario-service
implementation so FastAPI and standalone code share behavior. It must use an
inter-process ownership lock or equivalent writer coordination. A process
singleton is insufficient because Lunar Analyst uses multiple processes.

`lunarscout` must not directly update `scenario.db` as an alternative
implementation.

## Scenario API

```python
def open_scenario(
    path: str | Path,
    *,
    state: ScenarioState | None = None,
) -> Scenario: ...
```

```python
class Scenario:
    @property
    def root(self) -> Path: ...

    def path(self, relative_path: str | Path) -> Path: ...
    def dem_path(self) -> Path: ...
    def horizons_path(self) -> Path: ...
    def output_path(self, relative_path: str | Path) -> Path: ...

    # Later managed-state slice:

    def register_geotiff(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        publish: bool = False,
    ) -> ProductRef: ...

    def submit(
        self,
        plan: ComputationPlan,
        *,
        output: str | Path,
        overwrite: bool = False,
        publish: bool = False,
        title: str | None = None,
    ) -> Job: ...
```

All relative paths are normalized and checked against `Scenario.root`.
Absolute paths are rejected by scenario-relative methods. Path methods do not
create files and do not require their returned paths to exist. Existing
symlinks are resolved for containment checks, so a symlink cannot be used to
escape the scenario root. `open_scenario()` requires an existing directory
but does not require a complete scenario database or DEM.

`register_geotiff()` requires an attached state owner and never recomputes the
raster. With `publish=True`, registration and publication occur through that
owner, normally FastAPI.

## Explicit Alignment

NumPy arithmetic has no geospatial awareness. Callers must verify compatible
georeferencing before combining arrays:

```python
if ls.same_grid(left_georef, right_georef):
    combined = left + right

ls.require_same_grid(left_georef, right_georef)
```

Grid equality requires exact width and height, semantically equivalent CRS as
reported by GDAL/OSR, and exact affine coefficients by default. Nodata is not
part of the grid. Both checks accept an explicit non-negative
`affine_tolerance`; the default is `0.0` so small grid shifts are not hidden.
`require_same_grid()` raises `GridMismatchError` with the differing grid
fields.

Alignment is explicit:

```python
def align(
    source: np.ndarray,
    source_georef: GeoReference,
    *,
    to: GeoReference,
    resampling: str = "nearest",
    output_nodata: int | float | None | Literal["auto"] = "auto",
    output_dtype: np.dtype | type | str | None = None,
) -> tuple[np.ndarray, GeoReference]: ...
```

Example:

```python
source, source_georef = ls.read_geotiff("source.tif")
reference, reference_georef = ls.read_geotiff("reference.tif")

aligned, aligned_georef = ls.align(
    source,
    source_georef,
    to=reference_georef,
    resampling="bilinear",
)
```

The returned georeference matches `to`, including CRS, affine transform,
width, and height. By default, the result retains the source dtype and source
nodata. `output_nodata=None` explicitly disables destination nodata, and a
numeric value explicitly replaces it. An explicit `output_dtype` permits a
conversion; alignment never chooses a different dtype automatically. Nodata
must be representable by that output dtype.

`available_resampling_algorithms()` reports the algorithms exposed by the
installed GDAL. The stable names supported when their corresponding GDAL
constant is available are `nearest`, `bilinear`, `cubic`, `cubicspline`,
`lanczos`, `average`, `mode`, `max`, `min`, `median`, `q1`, `q3`, `sum`, and
`rms`.

The operation records or returns alignment lineage when used in a managed
workflow. No arithmetic, write, or submission method silently aligns inputs.

## Local Operations

The local API operates eagerly on NumPy arrays. Functions receive
`GeoReference` only when the calculation requires spatial scale or CRS.

```python
ls.slope(
    array,
    georef,
    output_nodata=...,
    units="degrees",
    compute_edges=False,
    scale=1.0,
) -> tuple[np.ndarray, GeoReference]

ls.aspect(
    array,
    georef,
    output_nodata=...,
    compute_edges=False,
) -> tuple[np.ndarray, GeoReference]

ls.hillshade(
    array,
    georef,
    output_nodata=...,
    azimuth=315.0,
    altitude=45.0,
    compute_edges=False,
    scale=1.0,
    z_factor=1.0,
) -> tuple[np.ndarray, GeoReference]

ls.label_regions(
    mask,
    georef=None,
    nodata="auto",
    cleanup="none",
    iterations=0,
) -> tuple[np.ndarray, GeoReference | None]

ls.region_sizes(
    mask,
    georef=None,
    nodata="auto",
    cleanup="none",
    iterations=0,
) -> tuple[np.ndarray, GeoReference | None]

ls.filter_regions_by_size(
    mask,
    georef=None,
    threshold=...,
    comparator=">=",
    nodata="auto",
    cleanup="none",
    iterations=0,
) -> tuple[np.ndarray, GeoReference | None]

ls.find_borders(
    mask,
    georef=None,
    nodata="auto",
) -> tuple[np.ndarray, GeoReference | None]
```

Normal NumPy syntax supplies arithmetic, comparisons, boolean composition,
selection, reductions, and scalar dtype conversion:

```python
slope_deg, slope_georef = ls.slope(
    dem,
    georef,
    output_nodata=-9999.0,
)
terrain_mask = (slope_deg != slope_georef.nodata) & (slope_deg <= 8.0)
candidate = np.where(terrain_mask, 1, 0).astype(np.uint8)
candidate_georef = slope_georef.with_nodata(0)
mean_illumination = np.mean(illumination.values, axis=0)
```

Terrain functions use GDAL's Horn algorithms. Slope defaults to degrees and
also supports percentage slope. GDAL's standard edge behavior is retained by
default; callers opt into computed edges explicitly. Because these operations
change output dtype, `output_nodata` is required and must be representable by
the result dtype. Every operation returns a new immutable `GeoReference`, even
when its spatial grid is unchanged.

`lunarscout` should not duplicate `np.where`, `np.mean`, `np.min`, `np.max`, or
NumPy scalar types merely to create its own vocabulary.

Region operations use eight-neighbor connectivity initially. Size thresholds
are pixel counts, not physical area. An area-based operation should have a
distinct name and explicit physical units.

Region nodata resolution defaults to `"auto"`: use `georef.nodata` when a
georeference is supplied, otherwise do no nodata processing. A numeric
`nodata` explicitly overrides the metadata, while `nodata=None` explicitly
disables nodata processing. Numeric masks interpret nonzero valid pixels as
true. Boolean masks use their values directly.

Labels and region sizes use `int32`. Filtered masks and borders are ordinary
Boolean arrays when nodata processing is inactive. When it is active, Boolean
results are `numpy.ma.MaskedArray` values so nodata remains distinct from both
true and false. All operations return `(array, georef_or_none)`.

## Temporal Arrays

Temporal raster values use a small immutable named container. The contained
values remain an ordinary NumPy array shaped `(time, height, width)` and use
ordinary scalar dtypes such as `np.float32` or `np.int64`:

```python
@dataclass(frozen=True)
class TemporalCube:
    values: np.ndarray
    times: np.ndarray  # one-dimensional datetime64[us], interpreted as UTC
    georef: GeoReference
```

The frozen container prevents rebinding its fields; it does not make the
potentially large `values` array read-only or copy it. `times` is normalized
to a read-only array. The container validates `(time, y, x)` dimensions,
spatial dimensions against `GeoReference`, matching time counts, no `NaT`,
and strictly increasing coordinates. It exposes `shape`, `dtype`,
`time_count`, `height`, `width`, `nbytes`, and
`dimensions == ("time", "y", "x")`.

```python
def times(
    start,
    stop,
    *,
    step_hours: float,
    source_timezone: str | None = None,
) -> TimeRange: ...
```

The domain includes `start`, advances by `step_hours`, and never exceeds
`stop`; an aligned stop is included. UTC-aware strings and datetimes are
accepted. Naive datetimes and timezone-free ISO strings default to UTC.
`source_timezone` explicitly interprets naive inputs in another IANA timezone.
Stored coordinates use `datetime64[us]` and are interpreted as UTC.

Temporal reducers have explicit names and reduce only the time axis:

```python
mean, georef = ls.temporal_mean(cube)
minimum, georef = ls.temporal_min(cube)
maximum, georef = ls.temporal_max(cube)
standard_deviation, georef = ls.temporal_std(cube)
```

Their nodata convention matches the other eager operations: `nodata="auto"`
uses `cube.georef.nodata`, a numeric argument overrides it, and `None`
disables nodata processing.

The local native temporal API makes storage explicit. Small results may return
an in-memory `TemporalCube`. Large results use the file-backed timestamped
GeoTIFF series described below. An unsafe in-memory request must be rejected
with an estimated byte count and a suggestion to select file-backed storage;
the API does not silently change storage or return a hidden lazy `Raster`.

```python
estimate = ls.native.estimate_temporal_allocation(
    signal="sun_fraction",
    times=time_range,
    georef=georef,
    storage="memory",
)

cube = scenario.sun_fraction(
    times=time_range,
    storage="memory",
)

series = scenario.sun_fraction(
    times=time_range,
    storage="geotiff_series",
    output="analysis/sun_fraction.temporal",
)
```

`storage` is required and must be exactly `"memory"` or
`"geotiff_series"`. File-backed storage requires a scenario-relative output;
memory storage rejects one. The default in-memory allocation limit is 2 GiB
and can be changed explicitly. File-backed generation preflights scratch and
output disk space using the uncompressed result size before starting native
work.

Native V2 streaming is patch-major. Memory mode assembles those chunks in the
requested NumPy cube. File-backed mode assembles them in a temporary disk
memmap, then feeds complete time layers into `TemporalGeoTiffSeriesWriter`.
The scratch file is always removed. This avoids a full-cube RAM allocation
without requiring thousands of writable GDAL datasets to remain open.

The initial local signals are `sun_fraction`, `sun_over_horizon_deg`, and
`earth_over_horizon_deg`. `sun_fraction` converts native uint8 encoding to
float32 fractions in `[0, 1]`. Horizon-margin signals are float32 degrees.
Native tiles must cover the complete requested time axis and spatial grid;
missing, overlapping, out-of-order, or malformed tiles fail the operation
rather than leaving implicit zero-valued regions. Progress and cooperative
cancellation propagate through the native stream and file-writing phases.

## File-Backed Timestamped GeoTIFF Series

The persistent temporal format uses one single-band, tiled,
compressed GeoTIFF per UTC sample. This preserves direct interoperability:
each timestamp can be selected by filename and opened independently in QGIS
or any GDAL application. A series with thousands of samples is acceptable;
the series is treated as one logical product rather than thousands of
independent scenario products.

Canonical layout:

```text
sun_fraction.temporal/
  manifest.json
  series.vrt
  layers/
    20270101T000000.000000Z.tif
    20270101T020000.000000Z.tif
    20270101T040000.000000Z.tif
  COMPLETE
```

Timestamp filenames are UTC, lexically sortable, contain six fractional
second digits, contain no colon characters, and are unique within the series.
All backing paths in the manifest and VRT are relative to the series root so
the directory is movable. Layer TIFFs use the existing Lunarscout GeoTIFF
rules: exact NumPy dtype, common nodata, common grid, 128 x 128 tiling,
datatype-sensitive DEFLATE prediction, and `BIGTIFF=IF_SAFER`.

### Manifest Authority

`manifest.json` is the authoritative temporal interpretation. It includes at
least:

- a format identifier and integer format version;
- signal name and units when known;
- common dtype, nodata, dimensions, CRS, and affine transform;
- ordered UTC `datetime64[us]`-compatible time coordinates;
- for every zero-based Python layer index, its relative TIFF path;
- the relative VRT path when generated;
- creation/provenance metadata that does not imply scenario registration.

The manifest uses standard JSON values only. Non-finite nodata values use an
explicit tagged encoding rather than non-standard JSON `NaN` or infinity
tokens. Dtype uses a canonical NumPy dtype string. CRS includes both WKT and
PROJ.4 representations, while WKT remains authoritative for GDAL use. Each
layer entry contains its index, exact UTC timestamp, and relative path.

Times must be unique and strictly increasing. Every layer must be single-band
and match the manifest dtype, nodata, width, height, CRS, and affine transform.
Readers reject missing files, unsafe relative paths, duplicate times, grid
mismatches, and series without a valid completion state.

Writers build in a staging directory on the destination filesystem. Layer
files are completed atomically, the final manifest is written after all
layers validate, and `COMPLETE` is written last. The completion record binds
to the final manifest version and SHA-256 digest so a stale marker cannot
validate a different manifest. Readers never treat a series without a valid
completion record as complete. Overwrite and recovery behavior must preserve
the prior completed series until the replacement is complete.

### VRT and QGIS

`series.vrt` is a derived, rebuildable GDAL VRT with one VRT band per backing
TIFF, in manifest order. The backing TIFFs are single-band, so VRT band
`n + 1` represents Python layer index `n`. Each VRT band receives its UTC
timestamp as its description and metadata.

QGIS can open the VRT and render a selected band while GDAL reads the required
blocks from the corresponding source TIFF. The VRT does not itself define a
standard temporal axis and QGIS is not assumed to connect these bands to its
temporal controller automatically. The manifest remains authoritative for
time lookup; the VRT exists for QGIS and general GDAL interoperability.

In a future managed-state slice, the series and VRT may be registered as one
logical product. Individual timestamp TIFFs are backing assets, not thousands
of independently published layers. Serving or exporting the VRT must keep its
relative backing assets available.

### Python API

The file-backed `TemporalGeoTiffSeries` type is distinct from in-memory
`TemporalCube`; it does not expose a `.values` property that could
unexpectedly materialize the complete series.

```python
def open_temporal_cube(
    path: str | Path,
    *,
    layer_cache_bytes: int = ...,
    max_open_datasets: int = ...,
) -> TemporalGeoTiffSeries: ...

def write_temporal_cube(
    path: str | Path,
    cube: TemporalCube,
    *,
    signal_name: str | None = None,
    units: str | None = None,
    provenance: dict | None = None,
    overwrite: bool = False,
    create_vrt: bool = True,
) -> TemporalGeoTiffSeries: ...

class TemporalGeoTiffSeriesWriter:
    def __init__(
        self,
        path,
        *,
        georef,
        dtype,
        signal_name=None,
        units=None,
        provenance=None,
        overwrite=False,
        create_vrt=True,
        progress_callback=None,
        cancellation_requested=None,
    ): ...

    def write_layer(self, time, array) -> Path: ...
    def finalize(self) -> TemporalGeoTiffSeries: ...
    def abort(self) -> None: ...

series = ls.open_temporal_cube("sun_fraction.temporal")

series.time_count
series.times
series.georef
series.dtype
series.shape                 # (time, y, x)
series.vrt_path

array, georef = series.read_layer(125)
array, georef = series.read_time(ls.utc_datetime(2027, 1, 11, 10))

layer = series.layer_for_time(time, method="exact")
time = series.time_for_layer(layer)
```

Large producers use `TemporalGeoTiffSeriesWriter` so they never need to
allocate an in-memory cube:

```python
with ls.TemporalGeoTiffSeriesWriter(
    "sun_fraction.temporal",
    georef=georef,
    dtype=np.float32,
    progress_callback=on_progress,
    cancellation_requested=is_cancelled,
) as writer:
    for time, layer in generated_layers:
        writer.write_layer(time, layer)

series = writer.result
```

Every layer must match the declared spatial shape and exact NumPy dtype, and
times must be unique and strictly increasing. A successful context exit
finalizes automatically. `finalize()` can also be called explicitly. Any
layer failure, cancellation, exceptional context exit, or `abort()` removes
the staging directory and leaves an existing completed destination intact.
Progress is reported only after a layer GeoTIFF has been written successfully.
Cancellation is cooperative and checked before layers and before publication.
`write_temporal_cube()` is a convenience wrapper over this writer.

Python layer indexes are zero-based. GDAL/VRT band indexes are one-based and
remain an internal translation. `read_layer()` reads its backing TIFF
directly rather than routing normal Python reads through the many-band VRT.
It returns an ordinary two-dimensional NumPy array and `GeoReference`.

Time lookup methods are `exact`, `nearest`, `before`, and `after`. Out-of-range
or unavailable lookups raise a structured temporal lookup error rather than
silently clamping. `nearest` chooses the earlier layer when a target is
exactly halfway between two samples.

### Caching and Reductions

GDAL owns compressed tile decoding and its raster block cache. Lunarscout does
not duplicate that block cache. The file-backed reader may additionally use:

- a bounded LRU of open, read-only GDAL datasets to control repeated opens and
  file-descriptor use; and
- an optional byte-budgeted LRU of fully decoded, read-only layers for
  interactive repeated access by time or layer number.

The full-layer cache is an optimization only. It has a configurable byte
budget, never contains dirty output state, and is not required for
correctness. Eviction simply releases the NumPy array.

Temporal reducers support the file-backed series without materializing the
full `(time, y, x)` cube or a full-cube nodata mask. They stream backing layers
or spatial windows into bounded accumulators and return the same
`(array, georef)` result contract as in-memory reducers. Writer and reducer
access order must be benchmarked with representative series, including the
known scale of approximately 3,800 layers and 3.8 GB compressed.

The initial implementation uses existing NumPy, GDAL, GeoTIFF, and VRT
capabilities and adds no new array-storage dependency. Zarr, HDF5, Dask,
memory mapping, and sharded multiband storage remain possible future backends,
not part of this format contract.

PSR initially remains an explicit product-producing native operation:

```python
scenario.psr(
    "analysis/psr.tif",
    *,
    horizons="horizons",
    overwrite=False,
) -> Path
```

It returns the absolute output path and does not register or publish the file.
The native-grid output is uint8 with `255` for pixels where the Sun center never
clears the horizon and `0` for pixels receiving direct Sun during the sampled
period. Validity is separate from the scientific values: GDAL mask `255` marks
a successfully calculated pixel and mask `0` marks unknown coverage from a
missing or unreadable 128 x 128 horizon tile. Unknown payload bytes are
deterministically zero and must be ignored. Complete output uses GDAL's virtual
all-valid mask; partial output stores a 1-bit internal GeoTIFF mask. The data
band has no nodata sentinel. The current native operation uses six-hour samples
from 1970-01-01 through 2044-01-01 and fixes observer elevation at zero.
Publication is atomic; errors and cancellation remove staging output and
preserve an existing destination. Managed PSR uses its existing typed job.

## Managed Computation Plans

An arbitrary NumPy calculation cannot be serialized reliably for remote
execution. Managed execution therefore uses an explicit `ComputationPlan`
rather than a public lazy raster type.

Conceptual example:

```python
plan = ls.transform_plan(
    "result = where(slope(dem) <= 8, 1, nodata())",
    inputs={"dem": scenario.dem_path()},
)

job = scenario.submit(
    plan,
    output="analysis/candidate_flat_sites.tif",
    publish=True,
)
```

The plan must lower to the existing `raster.transform` request. It is a
bounded, serializable, auditable managed-computation specification, not a
general Python execution container. The exact plan-building syntax remains to
be reviewed after the eager local API is settled.

## Job API

```python
class Job:
    id: str

    def status(self) -> JobStatus: ...
    def wait(self, *, timeout: float | None = None) -> JobResult: ...
    def cancel(self) -> None: ...
    def result(self) -> JobResult: ...
```

`wait()` returns the completed `JobResult` or raises `TimeoutError`,
`JobFailedError`, or `JobCancelledError`. `result()` is non-blocking and raises
`JobNotCompleteError` while the job is pending or running.

## Exceptions and Capability Diagnostics

All public exceptions derive from `LunarscoutError`:

```text
LunarscoutError
  GeoTiffError
    GeoTiffOpenError
    GeoTiffBandError
    GeoTiffMetadataError
    GeoTiffWriteError
    OutputExistsError
  GeoReferenceError
    GridMismatchError
    CoordinateTransformError
    CoordinateOutOfBoundsError
  TerrainOperationError
  RegionOperationError
  ScenarioError
    ScenarioPathError
    ScenarioStateUnavailableError
    ScenarioStateOwnershipError
  LoweringError
    UnsupportedOperationError
    UnboundSourceError
  NativeError
    NativeUnavailableError
    NativeBootstrapError
    NativeInputError
  JobError
    JobNotCompleteError
    JobFailedError
    JobCancelledError
```

Errors include a stable string `code`, a human-readable message, and structured
`details`. Backend errors should be translated without discarding their
original code or details.

Capability inspection must not initialize optional runtimes:

```python
ls.capabilities()
ls.describe_operation("sun_fraction")
ls.native.status()
```

`ls.native.status()` returns a structured report distinguishing Python package,
.NET runtime, `moonlib`, CSPICE, and GDAL availability. The native boundary
also provides:

```python
ls.native.is_available()       # discovery only; does not load CLR
ls.native.initialize()         # explicit lazy bootstrap and smoke checks
```

`initialize(force=False, verify=True)` delegates to Lunar Analyst's existing
native bootstrap and returns the same structured status after initialization.
It does not implement a second DLL resolver. Native operation wrappers create
only `MoonlibBridge` instances through this boundary; they do not expose or
directly instantiate other moonlib runtime types.

## Current v0.1 Boundary

Implemented:

- `read_geotiff()` for one band with native dtype and actual nodata
- immutable `GeoReference` with projection, affine, dimensions, pixel sizes,
  nodata, bounds, and coordinate-conversion helpers
- `write_geotiff()`
- `utc_datetime()`
- stable base GeoTIFF and georeferencing exceptions
- GDAL-compatible slope, aspect, and hillshade
- eight-neighbor region labeling, sizes, filtering, borders, and optional
  cleanup/nodata processing
- strict grid comparison and explicit GDAL alignment/resampling
- filesystem-only `Scenario` paths with root-containment enforcement
- import-safe native capability diagnostics and explicit lazy initialization
- UTC `TimeRange`, named in-memory `TemporalCube`, and eager temporal reducers
- explicit native temporal storage selection, allocation preflight, and local
  `Scenario` generation for solar fraction and Sun/Earth horizon margins
- patch-major native assembly into memory or preflighted disk scratch, with
  progress, cancellation, and complete-coverage validation
- slope and region examples revised to use NumPy plus `GeoReference`

Designed for later slices:

- remaining eager NumPy operations beyond terrain and connected regions
- authoritative `LocalScenarioState` extraction
- `RemoteScenarioState`, `Job`, and managed transform plans
- PSR wrapper
- publication and product registration

The earlier question about oversized native temporal results is resolved:
callers explicitly select an in-memory `TemporalCube` or file-backed
`TemporalGeoTiffSeries`; the implementation never changes storage implicitly.
