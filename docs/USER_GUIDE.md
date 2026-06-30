# Lunarscout User Guide

Status: Draft user guide for the standalone `lunarscout` library.

This guide is the first place to read when learning what Lunarscout is, how to
install it, how to use its current Python API, and which parts of the
implementation are mature. Some sections are intentionally short because the
public package surface is still settling.

## What Lunarscout Is

Lunarscout is a Python library for lunar terrain, raster, temporal, and
optional native lighting analysis. It is designed for notebook and script
workflows where raster values are ordinary NumPy arrays and geospatial metadata
is carried explicitly.

The current package includes:

- single-band GeoTIFF input and output;
- georeferencing and coordinate conversion;
- GDAL-compatible slope, aspect, and hillshade operations;
- explicit grid comparison and raster alignment;
- connected-region labeling and filtering;
- filesystem-safe scenario paths;
- UTC-aware temporal arrays;
- file-backed timestamped GeoTIFF series; and
- optional native lighting and permanent-shadow products.

Lunarscout was split from Lunar Analyst so the calculation library can mature
independently of the agent, web UI, FastAPI service, application job framework,
scenario database, and notebook-runner code.

Dependency direction is intentionally one-way:

```text
lunar_analyst -> lunarscout
```

Lunarscout must not import Lunar Analyst application modules.

## Installation

Lunarscout requires Python 3.11 or newer.

From this repository, install the package in editable mode:

```bash
.venv/bin/python -m pip install -e .
```

For development tools:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

For optional native calculations, install the native Python extra:

```bash
.venv/bin/python -m pip install -e '.[native]'
```

### GDAL Requirement

Lunarscout uses the GDAL Python bindings supplied by the supported runtime
environment. GDAL is not listed as a normal PyPI dependency because the Python
package must match the installed native GDAL library.

TODO: Document supported GDAL installation paths for common platforms.

### Native Runtime Requirement

Pure-Python installation works without building native code. Native features
currently require either:

- a local native build, for example:

  ```bash
  dotnet build native/moonlib/moonlib.csproj
  ```

- or an explicit `LUNARSCOUT_MOONLIB_DLL` environment variable pointing at the
  built `moonlib.dll`.

Prebuilt native wheels are deferred.

TODO: Document exact native runtime setup once distribution packaging is
finalized.

## Quick Start

```python
import lunarscout as ls

scenario = ls.open_scenario("/data/mons_mouton")
dem, georef = ls.read_geotiff(scenario.dem_path())

if georef is None:
    raise ValueError("The DEM is not georeferenced.")

slope_deg, slope_georef = ls.slope(
    dem,
    georef,
    output_nodata=-9999.0,
)

ls.write_geotiff("analysis/slope.tif", slope_deg, slope_georef)
```

`Scenario` currently provides naming and containment only. Its standard paths
are `dem.tif` and `lighting/horizons`. It does not read `scenario.db`,
register products, publish layers, create directories, or own application
state.

## Core Concepts

### GeoTIFFs and GeoReference

GeoTIFF reads return a NumPy array plus optional `GeoReference` metadata:

```python
array, georef = ls.read_geotiff("dem.tif")
```

Use the `GeoReference` when converting between raster coordinates and map
coordinates, comparing grids, or writing derived products.

TODO: Add a complete georeferencing example with coordinate conversion.

### Terrain Products

Lunarscout provides slope, aspect, and hillshade-style operations for
georeferenced rasters:

```python
slope_deg, slope_georef = ls.slope(dem, georef)
aspect_deg, aspect_georef = ls.aspect(dem, georef)
shade, shade_georef = ls.hillshade(dem, georef)
```

TODO: Document nodata handling and units for each terrain operation.

### Grid Alignment

Grid compatibility is never inferred from array shape alone. Verify compatible
rasters before combining them:

```python
ls.require_same_grid(left_georef, right_georef)
```

Align explicitly when grids differ:

```python
aligned, aligned_georef = ls.align(
    source,
    source_georef,
    to=right_georef,
    resampling="bilinear",
)
```

### Region Analysis

Connected-region helpers can label, measure, filter, and outline raster masks:

```python
labels, labels_georef = ls.label_regions(candidate_mask, georef)
sizes, sizes_georef = ls.region_sizes(candidate_mask, georef)
large_regions, large_regions_georef = ls.filter_regions_by_size(
    candidate_mask,
    georef,
    threshold=100,
)
borders, borders_georef = ls.find_borders(large_regions, large_regions_georef)
```

TODO: Document connectivity behavior and recommended mask conventions.

## Temporal Data

Time domains use UTC coordinates and include an aligned stop value:

```python
time_range = ls.times(
    "2027-01-01T00:00:00Z",
    "2027-01-02T00:00:00Z",
    step_hours=2,
)
```

`TemporalCube` stores an in-memory `(time, y, x)` NumPy array with UTC time
coordinates and a spatial `GeoReference`:

```python
cube = ls.TemporalCube(values, time_range, georef)
mean, mean_georef = ls.temporal_mean(cube)
```

Large or persistent time series can be stored as one tiled, compressed,
single-band GeoTIFF per timestamp, plus a manifest and optional VRT:

```python
series = ls.write_temporal_cube(
    "sun_fraction.temporal",
    cube,
    signal_name="sun_fraction",
    units="fraction",
)

layer, layer_georef = series.read_time(
    "2027-01-01T12:00:00Z",
    method="nearest",
)

streamed_mean, mean_georef = ls.temporal_mean(series)
```

`TemporalGeoTiffSeries` has no `.values` property. Direct reads use bounded
caches, and temporal reducers stream layers without constructing a full
three-dimensional cube.

Incremental producers can write a file-backed series without first creating a
`TemporalCube`:

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

The writer validates each layer, writes into a staging directory, and publishes
the completed series only after the manifest, VRT, and completion digest are
ready.

## Optional Native Features

Native compute is optional. Pure-Python imports and pure-Python functionality
must not initialize Python.NET, CLR, GDAL native bindings, SPICE, or `moonlib`.

Capability checks do not initialize CLR:

```python
status = ls.native.status()

if status["available"]:
    loaded_status = ls.native.initialize()
```

Native temporal generation requires an explicit storage choice:

```python
time_range = ls.times("2027-01-01", "2027-01-08", step_hours=2)

illumination = scenario.sun_fraction(
    times=time_range,
    storage="memory",
)

illumination_series = scenario.sun_fraction(
    times=time_range,
    storage="geotiff_series",
    output="analysis/sun_fraction.temporal",
)
```

Memory requests are rejected before native initialization when their exact
output allocation exceeds the configured limit. Storage is never changed
automatically.

Native permanent-shadow generation is an explicit file-producing operation:

```python
psr_path = scenario.psr("analysis/psr.tif", overwrite=False)
```

The current PSR product is a native-grid `uint8` GeoTIFF. Value `255` means the
Sun center never clears the local horizon across the native operation's
six-hour samples from 1970-01-01 through 2044-01-01. Value `0` means it clears
the horizon at least once. A GDAL validity mask distinguishes unknown pixels
whose required horizon tile was missing or unreadable.

TODO: Document native input data requirements, supported products, and
platform support in a stable public format.

## Examples

Executable examples live in `examples/`. They are ordinary Python programs and
are indexed in `examples/README.md`.

Start with:

```bash
.venv/bin/python examples/00_geotiff_and_coordinates.py
```

Then continue through terrain, regions, alignment, temporal cubes,
file-backed series, streaming reductions, and native examples as needed.

TODO: Add a recommended learning path once the example set is finalized.

## Implementation Maturity

Lunarscout currently uses Semantic Versioning and is pre-`1.0.0`.

Before `1.0.0`, public APIs are provisional and breaking changes may occur in
minor releases. Intentional breaking changes should be recorded in
`CHANGELOG.md`. Patch releases should not intentionally break documented
behavior.

More mature areas:

- pure-Python GeoTIFF I/O;
- `GeoReference` metadata handling;
- terrain products;
- grid alignment;
- region operations;
- UTC temporal ranges and `TemporalCube`;
- file-backed temporal GeoTIFF series; and
- source boundary separation from Lunar Analyst application code.

Less mature or explicitly provisional areas:

- high-level native temporal APIs;
- native binary packaging;
- public native product contracts;
- native source layout after extraction from Lunar Analyst;
- CI and release automation;
- platform-specific installation documentation.

Public API includes:

- names exported from `lunarscout.__init__`;
- documented functions and classes in public modules;
- documented file formats and manifests written by the package;
- documented exception classes and error codes.

Not public API:

- names beginning with `_`;
- tests;
- examples;
- C# internals unless documented as Python-callable library API.

## Architecture Overview

The public Python package lives under `src/lunarscout`.

Core modules:

- `georeference.py`: CRS, affine transform, and raster grid metadata.
- `geotiff.py`: GeoTIFF read and write helpers.
- `terrain.py`: slope, aspect, and hillshade-style array operations.
- `alignment.py`: grid compatibility and resampling.
- `regions.py`: connected-region analysis.
- `temporal.py`: UTC time ranges and in-memory temporal cubes.
- `temporal_store.py`: file-backed temporal GeoTIFF series.
- `scenario.py`: filesystem-safe scenario path helpers only.
- `native.py`: native capability discovery and public native entry points.
- `native_temporal.py`: native temporal and lightmap-buffer APIs.
- `native_product.py`: native file-producing products such as PSR rasters.
- `_native_runtime/`: private runtime discovery and bootstrap implementation.

Native source currently lives in:

```text
native/moonlib
```

The initial standalone extraction includes `moonlib` wholesale. This is a
temporary migration posture. The native source and tests should be trimmed
later to the library-owned surface after the public Python/native API is
better understood.

The package must not contain FastAPI routes, web UI code, assistant/RAG logic,
Lunar Analyst job handlers, scenario database mutation, or notebook-runner
helpers.

## Testing

Pure-Python tests live in `tests/`. Native C# tests live in `native/tests/`.

Representative commands:

```bash
PYTHONPATH=src python -m pytest tests -q
dotnet build native/moonlib/moonlib.csproj
dotnet test native/tests/HorizonGen.Tests/HorizonGen.Tests.csproj --filter FullyQualifiedName~FillLightmapBuffersTests
```

TODO: Replace representative local commands with release-quality verification
instructions once CI and packaging are finalized.

## Roadmap

Current open work:

- replace or redesign the old high-level temporal reducer default path so it
  uses standalone native APIs rather than Lunar Analyst streaming adapters;
- mature the high-level native temporal APIs;
- decide native binary and wheel packaging after license and redistribution
  review;
- add CI;
- trim native source and tests to the minimal library-owned surface after
  behavior is stable; and
- expand the user guide with complete installation, data, and product
  reference sections.

## Reference Stubs

The following sections are expected but not ready to fill in:

- API reference;
- supported raster formats;
- supported coordinate reference systems;
- nodata and mask conventions;
- performance guidance;
- memory and disk sizing guidance;
- native runtime troubleshooting;
- release and compatibility policy;
- contribution guide; and
- security and data provenance notes.
