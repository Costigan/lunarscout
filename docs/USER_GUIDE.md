# Lunarscout User Guide

Status: Draft user guide for the standalone `lunarscout` library.

This guide is the first place to read when learning what Lunarscout is, how to
install it, how to use its current Python API, and which parts of the
implementation are mature. Some sections are intentionally short because the
public package surface is still settling.

## What Lunarscout Is

Lunarscout is a Python library for lunar terrain, raster, temporal, and
lighting analysis. It is designed for notebook and script
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
- file-backed timestamped GeoTIFF series;
- SPICE-backed Sun and Earth local-frame histories;
- CUDA-accelerated horizon generation; and
- patch-streamed lightmap, permanent-shadow, safe-haven, and landed
  mission-duration product implementations with CPU fallbacks.

Lunarscout was split from Lunar Analyst so the calculation library can mature
independently of the agent, web UI, FastAPI service, application job framework,
scenario database, and notebook-runner code.

Dependency direction is intentionally one-way:

```text
lunar_analyst -> lunarscout
```

Lunarscout must not import Lunar Analyst application modules.

## Examples

Runnable example scripts are provided in the repository under `examples/`.
They demonstrate every public API capability in increasing order of scope:

| Domain | Scripts |
|--------|---------|
| GeoTIFF I/O, coordinates, terrain, regions, alignment | `01`–`04` |
| Temporal cubes, file-backed series, streaming reducers | `05`–`08` |
| QGIS VRT inspection, landing-site screening | `09`–`10` |
| SPICE vectors, azimuth/elevation | `11` |
| Body/horizon plots, synthetic lightmap | `12`–`13` |
| PSR, horizon generation, downstream products | `15`–`17` |

Most examples work on synthetic data without a GPU or real scenario.
A synthetic 256×256 DEM with pregenerated horizon tiles is downloaded
automatically on first use for examples that need horizons.  See
[`examples/README.md`](../examples/README.md) for complete setup instructions,
a data-requirements table, and detailed guidance for each script.

## Function Overview

The package root is the normal user-facing API:

```python
import lunarscout as ls
```

### Common Root Functions

| Function                                                                   | Summary                                                                             |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `ls.read_geotiff(path, band=1)`                                            | Read one GeoTIFF band and return `(array, georef)`.                                 |
| `ls.write_geotiff(path, array, georef, overwrite=False)`                   | Write a single-band tiled GeoTIFF from a NumPy array and `GeoReference`.            |
| `ls.slope(elevation, georef)`                                              | Compute terrain slope from an elevation raster.                                     |
| `ls.aspect(elevation, georef)`                                             | Compute terrain aspect from an elevation raster.                                    |
| `ls.hillshade(elevation, georef, ...)`                                     | Compute a hillshade raster from elevation and lighting angles.                      |
| `ls.same_grid(left, right, ...)`                                           | Return whether two `GeoReference` grids are compatible.                             |
| `ls.require_same_grid(left, right, ...)`                                   | Raise a structured error unless two grids are compatible.                           |
| `ls.available_resampling_algorithms()`                                     | List supported raster resampling algorithm names.                                   |
| `ls.align(array, source_georef, target_georef, ...)`                       | Resample a raster from one grid onto another.                                       |
| `ls.label_regions(mask, georef=None, ...)`                                 | Label connected boolean regions.                                                    |
| `ls.region_sizes(labels)`                                                  | Return pixel counts for labeled regions.                                            |
| `ls.filter_regions_by_size(mask, georef=None, ...)`                        | Keep or remove connected regions by size.                                           |
| `ls.find_borders(mask, georef=None, ...)`                                  | Return border pixels for connected regions.                                         |
| `ls.utc_datetime(...)`                                                     | Construct a timezone-aware UTC `datetime`.                                          |
| `ls.times(start, stop, step_hours=...)`                                    | Construct an inclusive UTC `TimeRange`.                                             |
| `ls.temporal_mean(cube, ...)`                                              | Reduce a temporal cube to a per-pixel mean raster.                                  |
| `ls.temporal_min(cube, ...)`                                               | Reduce a temporal cube to a per-pixel minimum raster.                               |
| `ls.temporal_max(cube, ...)`                                               | Reduce a temporal cube to a per-pixel maximum raster.                               |
| `ls.temporal_std(cube, ...)`                                               | Reduce a temporal cube to a per-pixel standard-deviation raster.                    |
| `ls.write_temporal_cube(path, cube, ...)`                                  | Write a temporal cube as a timestamped GeoTIFF series.                              |
| `ls.open_temporal_cube(path)`                                              | Open a file-backed temporal GeoTIFF series.                                         |
| `ls.iter_times(start, stop, step)`                                         | Iterate UTC datetimes between start and stop using a `timedelta`.                   |
| `ls.body_vectors_ned(point, body, times, ...)`                             | Return Sun or Earth vectors in local north/east/down coordinates.                   |
| `ls.body_vectors_ned_dataframe(point, body, times, ...)`                   | Return local body vectors as a pandas DataFrame.                                    |
| `ls.body_azimuth_elevation(point, body, times, ...)`                       | Return a NumPy array of body azimuth and elevation.                                 |
| `ls.body_azimuth_elevation_dataframe(point, body, times, ...)`             | Return body azimuth/elevation as a pandas DataFrame.                                |
| `ls.body_azimuth_elevation_over_horizon(point, body, times, horizon, ...)` | Return body azimuth and elevation above an interpolated horizon.                    |
| `ls.plot_body_elevation(point, body, times, ...)`                          | Plot one body's elevation over time.                                                |
| `ls.plot_body_elevations(point, bodies, times, horizon=None, ...)`         | Plot one or more body elevations, optionally over a provided horizon.               |
| `ls.load_map_product_catalog(path)`                                        | Load a map-product catalog JSON file.                                               |
| `ls.search_map_products(catalog, ...)`                                     | Search products in a loaded catalog.                                                |
| `ls.map_product_scenario_name(product)`                                    | Build a scenario name from map-product metadata.                                    |
| `ls.filesystem_safe_scenario_name(name)`                                   | Sanitize a scenario name for filesystem use.                                        |
| `ls.map_product_download_directory(root, product)`                         | Return the expected download directory for a map product.                           |
| `ls.download_map_product(product, output_root, ...)`                       | Download a map product into a local scenario-style directory.                       |
| `ls.open_scenario(path)`                                                   | Open a filesystem-backed `Scenario`.                                                |
| `ls.generate_lightmap(dem, horizons, output, ...)`                         | Generate timestamped uint8 visible-solar-fraction bands.                            |
| `ls.generate_psr(dem, horizons, output, ...)`                              | Generate a permanent-shadow classification and validity mask.                       |
| `ls.generate_sun_elevation(dem, horizons, output, ...)`                    | Generate Sun-center terrain-relative elevation bands.                               |
| `ls.generate_earth_elevation(dem, horizons, output, ...)`                  | Generate Earth-center terrain-relative elevation bands.                             |
| `ls.generate_safe_havens(dem, horizons, output, ...)`                      | Generate longest low-Sun durations for Earth outages.                               |
| `ls.mission_duration_from_sunlight(...)`                                   | Generate landed-duration bands from a sunlight-fraction threshold.                  |
| `ls.mission_duration_from_sun_elevation(...)`                              | Generate landed-duration bands from a Sun-elevation threshold.                      |
| `ls.mission_duration_from_sunlight_and_earth_elevation(...)`               | Generate durations satisfying sunlight and Earth-elevation thresholds.              |
| `ls.mission_duration_from_sun_and_earth_elevation(...)`                    | Generate durations satisfying Sun- and Earth-elevation thresholds.                  |

### Scenario Methods

| Method                                                                  | Summary                                                                                  |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `scenario.root_path()`                                                  | Return the resolved scenario root directory.                                             |
| `scenario.path(relative_path)`                                          | Resolve a scenario-relative path, allowing the root itself.                              |
| `scenario.output_path(relative_path)`                                   | Resolve a non-empty scenario-relative output path.                                       |
| `scenario.dem_path()`                                                   | Return the canonical primary DEM path, `dem.tif`.                                        |
| `scenario.horizons_path()`                                              | Return the canonical horizons directory, `horizons/`.                                    |
| `scenario.generate_horizons(...)`                                      | Generate resumable CUDA horizon tiles from the primary and surrounding DEMs.             |
| `scenario.hillshade_path()`                                             | Return `hillshade.tif` in the scenario root.                                             |
| `scenario.slope_path()`                                                 | Return `slope.tif` in the scenario root.                                                 |
| `scenario.aspect_path()`                                                | Return `aspect.tif` in the scenario root.                                                |
| `scenario.roughness_path()`                                             | Return `roughness.tif` in the scenario root.                                             |
| `scenario.horizon_patch_pixel(x, y)`                                    | Convert DEM pixel coordinates to pixel coordinates inside a 128x128 horizon patch.       |
| `scenario.horizon_patch_row_col(x, y)`                                  | Return the horizon patch row and column containing a DEM pixel.                          |
| `scenario.horizon_file_path(x, y, observer_height_decimeters)`          | Return the matching `.cbin` or `.bin` horizon file path, preferring `.cbin`, or `None`.  |
| `scenario.horizon_from_open_file(file_handle, patch_x, patch_y)`        | Read one horizon from an open horizon file as a 1440-sample `float32` array.             |
| `scenario.horizon_for_pixel(x, y, observer_height_decimeters)`          | Fetch one DEM pixel horizon, caching one open file handle.                               |
| `scenario.close_horizon_file()`                                         | Close the cached open horizon file handle.                                               |
| `scenario.lightmap(output, ...)`                                        | Generate a Python lightmap using canonical DEM and horizon paths.                        |
| `scenario.psr(output, ...)`                                             | Generate a Python permanent-shadow product.                                               |
| `scenario.sun_elevation(output, ...)`                                   | Generate Sun-center terrain-relative elevation bands.                                    |
| `scenario.earth_elevation(output, ...)`                                 | Generate Earth-center terrain-relative elevation bands.                                  |
| `scenario.safe_havens(output, ...)`                                     | Generate safe-haven duration bands.                                                       |
| `scenario.lonlat_to_dem_pixel(point)`                                   | Convert a `LonLat` to DEM pixel coordinates.                                             |
| `scenario.plot_azimuth_elevation_axes(...)`                             | Create an empty azimuth/elevation Matplotlib axis.                                       |
| `scenario.plot_horizon(point, ...)`                                     | Plot the stored horizon for a lon/lat point.                                             |
| `scenario.body_azimuth_elevation_over_horizon(point, body, times, ...)` | Fetch the scenario horizon and return body elevation over that horizon.                  |
| `scenario.plot_body_elevations(point, bodies, times, ...)`              | Plot body elevations, optionally fetching the scenario horizon with `over_horizon=True`. |
| `scenario.plot_body_position(ax, point, body, time, ...)`               | Overlay a body center point or apparent limb on an azimuth/elevation axis.               |
| `scenario.plot_body_path(ax, point, body, times, ...)`                  | Overlay a body center path and/or translucent limb band.                                 |
| `scenario.plot_zoomed_body_path(point, bodies, times, ...)`             | Plot one or more body limb paths against the horizon in an equal-scale zoomed view.      |

### File-Backed Temporal Objects

| Method                                                 | Summary                                                     |
| ------------------------------------------------------ | ----------------------------------------------------------- |
| `TemporalGeoTiffSeriesWriter.write_layer(time, array)` | Write one timestamped GeoTIFF layer.                        |
| `TemporalGeoTiffSeriesWriter.finalize()`               | Commit a temporal GeoTIFF series and return an open reader. |
| `TemporalGeoTiffSeriesWriter.abort()`                  | Remove an in-progress temporal series staging directory.    |
| `TemporalGeoTiffSeries.time_for_layer(index)`          | Return the timestamp for a layer index.                     |
| `TemporalGeoTiffSeries.layer_for_time(time, ...)`      | Find the layer index nearest or matching a time.            |
| `TemporalGeoTiffSeries.read_layer(index)`              | Read one layer as `(array, georef)`.                        |
| `TemporalGeoTiffSeries.read_time(time, ...)`           | Read one layer selected by time.                            |
| `TemporalGeoTiffSeries.close()`                        | Close cached datasets held by the series.                   |

SPICE kernel management helpers are available under `ls.spice`.

## Installation

Lunarscout requires Python 3.11 or newer.

From this repository, install the package in editable mode:

```bash
.venv/bin/python -m pip install -e .
```

The base installation supports CPU execution. On a supported NVIDIA system,
install the CUDA execution profile:

```bash
python -m pip install "lunarscout[cuda]"
```

Both profiles are imported as `import lunarscout as ls`. The `cuda` extra
installs the validated Numba-CUDA CUDA 12 user-space runtime; it does not
install an NVIDIA driver.

For development tools:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

### GDAL Requirement

Lunarscout uses the GDAL Python bindings supplied by the supported runtime
environment. GDAL is not listed as a normal PyPI dependency because the Python
package must match the installed GDAL library.

TODO: Document supported GDAL installation paths for common platforms.

### Compute Backends

Importing `lunarscout` does not import Numba, initialize CUDA, or load SPICE
kernels. Those components are loaded only when a function needs them. This
keeps ordinary scripts, notebooks, documentation builds, and CPU-only
environments lightweight.

Production horizon generation requires a supported NVIDIA GPU and the `cuda`
installation profile. Reading stored horizons and running horizon-derived
calculations do not: lightmaps,
permanent-shadow maps, safe-haven maps, and landed mission-duration maps have
CPU implementations, with CUDA acceleration selected when requested and
available. `ls.cuda.is_available()` explicitly probes whether the supported
runtime and a usable device are available. `ls.cuda.status()` additionally
reports the Numba, Numba-CUDA, CUDA toolkit, CUDA driver API, device, compute
capability, and free/total device-memory values available to that process.
Missing runtimes and devices use stable `CudaError` codes, while driver, PTX,
JIT, and kernel-execution failures are reported as structured CUDA execution
errors and never trigger a silent CPU retry.

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

`Scenario` provides filesystem-safe access to standard paths below an existing
scenario root. Its standard paths are `dem.tif`, `horizons`, `hillshade.tif`,
`slope.tif`, `aspect.tif`, and `roughness.tif`. It does not read
`scenario.db`, register products, publish layers, create directories, or own
application state.

```python
scenario.dem_path()
scenario.root_path()
scenario.horizons_path()
scenario.hillshade_path()
scenario.slope_path()
scenario.aspect_path()
scenario.roughness_path()
scenario.output_path("analysis/result.tif")
```

Scenario-relative path methods reject absolute paths, parent traversal, and
symlink escapes.

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

## SPICE Local-Frame Histories

Lunarscout can calculate Sun and Earth position histories at a lunar surface
point using SpiceyPy and NAIF kernels. The notebook-facing API uses
planetocentric lunar longitude/latitude in degrees:

```python
from datetime import timedelta

point = ls.LonLat(longitude=0.0, latitude=-89.0)
sample_times = list(
    ls.iter_times(
        "2027-01-01T00:00:00Z",
        "2027-01-02T00:00:00Z",
        timedelta(hours=1),
    )
)
```

`iter_times` returns timezone-aware UTC `datetime` values and includes the stop
time when it is aligned to the step.

### Kernel Loading

SPICE kernels are not loaded at package import time. By default, SPICE-backed
functions call `ls.spice.ensure_default_kernels()`, which downloads any missing
default NAIF kernels, verifies their SHA-256 checksums from the checked-in
manifest, generates a temporary meta-kernel, and furnishes it through
SpiceyPy.

Default kernels are cached under:

```text
$LUNARSCOUT_SPICE_KERNEL_DIR
```

or, when that is unset:

```text
$XDG_DATA_HOME/lunarscout/spice/kernels
```

with fallback:

```text
~/.local/share/lunarscout/spice/kernels
```

Download explicitly when desired:

```python
downloaded = ls.spice.download_default_kernels()
```

Use `overwrite=True` to refresh cached files:

```python
ls.spice.download_default_kernels(overwrite=True)
```

If you manage kernels yourself, furnish a path or list of paths. By default,
this disables default autoloading for the current Python process:

```python
ls.spice.furnish("/data/spice/my_project.tm")
ls.spice.furnish(["/data/spice/a.bsp", "/data/spice/b.tpc"])
```

You can also set `LUNARSCOUT_SPICE_META_KERNEL` before Python starts to use a
specific meta-kernel as Lunarscout's default. Kernel state helpers are available
under `ls.spice`, including:

```python
ls.spice.ensure_default_kernels()
ls.spice.reload_default_kernels()
ls.spice.unload_default_kernels()
ls.spice.clear_kernels()
ls.spice.default_kernels_loaded()
ls.spice.autoload_enabled()
ls.spice.set_autoload_enabled(True)
```

### Vectors and Angles

The local Cartesian frame is lunar NED:

- `x`: north, tangent to increasing planetocentric latitude;
- `y`: east, tangent to increasing longitude; and
- `z`: down, toward the Moon center and opposite local up.

Returned vectors are position/range vectors in kilometers, not unit vectors.
Supported body names are currently `"sun"` and `"earth"`.

```python
vectors = ls.body_vectors_ned(point, "sun", sample_times)
```

`vectors` is a `float64` NumPy array shaped `(time, 3)` with columns `x`, `y`,
and `z`.

DataFrame output is also available:

```python
df = ls.body_vectors_ned_dataframe(point, "earth", sample_times)
```

The azimuth/elevation convention is:

- azimuth `0 deg` is north;
- azimuth `90 deg` is east; and
- elevation increases upward, with `+90 deg` straight up.

```python
angles = ls.body_azimuth_elevation(point, "sun", sample_times)
angle_df = ls.body_azimuth_elevation_dataframe(point, "sun", sample_times)
```

`angles` is a `float64` NumPy array shaped `(time, 2)` with columns `azimuth`
and `elevation`, in degrees. DataFrame columns are `time`, `azimuth`, and
`elevation`.

Pass `ensure_kernels=False` to SPICE-backed functions when you have already
furnished kernels and do not want Lunarscout to check defaults:

```python
vectors = ls.body_vectors_ned(
    point,
    "sun",
    sample_times,
    ensure_kernels=False,
)
```

### Plotting

Matplotlib helpers return `(fig, ax)` for notebook customization:

```python
fig, ax = ls.plot_body_elevation(point, "sun", sample_times, grid=True)
```

Overlay multiple bodies:

```python
fig, ax = ls.plot_body_elevations(
    point,
    ["sun", "earth"],
    sample_times,
    grid=True,
)
```

## Horizons and Lighting Products

### Current API Status

Scenario helpers for generating, locating, reading, and plotting horizon tiles
are public. Python-only public functions also generate lightmaps, PSR,
Sun/Earth terrain-relative elevation, safe havens, and all four landed
mission-duration products.

The exact candidate signatures, defaults, shared behavior, open cleanup
recommendations, and approval checklist are collected in
[`PUBLIC_API_REVIEW.md`](PUBLIC_API_REVIEW.md).

Do not import `_numba_horizon` modules in user code. Use the root functions or
the corresponding `Scenario` conveniences. All downstream functions default
to `backend="auto"`; use `"cpu"` to avoid touching CUDA or `"cuda"` to require
CUDA without fallback. They return the completed `Path`.

```python
output = scenario.lightmap(
    "analysis/lightmap.tif",
    times=ls.times(
        "2029-01-01T00:00:00Z",
        "2029-01-02T00:00:00Z",
        step_hours=6,
    ),
    backend="auto",
    verbose=True,
)
```

Supplying `sun_vectors_m=` or `earth_vectors_m=` with a matching `TimeRange`
avoids SPICE import and kernel loading. Generate those exact product-ready
Moon-ME meter vectors explicitly with `ls.body_vectors_moon_me("sun", times)`
or `ls.body_vectors_moon_me("earth", times)`. `verbose=False` is the default. Applications
can instead use `progress_callback` for a monotonic durable fraction and
`progress_event_callback` for immutable stage, backend, patch, and path detail.
Cancellation uses `cancellation_requested`. Compatible staged jobs resume by
default; `start_fresh=True` discards staged state, while `overwrite=True`
protects an existing completed output until replacement publication.

### Generating Horizons

Horizon generation is CUDA-only and therefore has no `backend` argument. The
first DEM supplies the output grid; later DEMs extend the terrain coverage in
the order supplied. The root function returns the resolved output directory:

```python
horizons = ls.generate_horizons(
    "/data/site/horizons",
    [
        "/data/site/dem.tif",
        "/data/regional-dem.tif",
    ],
    observer_height_m=0.0,
    compress=True,
    verbose=True,
)
```

The Scenario convenience automatically places the canonical primary DEM first:

```python
horizons = scenario.generate_horizons(
    surrounding_dems=["surrounding/regional.tif"],
    compress=True,
)
```

Valid existing tiles are structurally checked and skipped by default.
`overwrite=True` regenerates all tiles. Explicit cancellation and both progress
callback forms use the same contracts as downstream products. If a compatible
NVIDIA CUDA device cannot be initialized, the call raises `ls.CudaError` and
does not fall back to a CPU generator.

### Reading Stored Horizons

Scenario helpers locate and read existing horizon tiles. DEM pixel
coordinates are zero-based, with `x` as column and `y` as row. Horizon patches
are 128 by 128 pixels.

```python
patch_x, patch_y = scenario.horizon_patch_pixel(x=257, y=130)
patch_row, patch_col = scenario.horizon_patch_row_col(x=257, y=130)

horizon_file = scenario.horizon_file_path(
    x=257,
    y=130,
    observer_height_decimeters=0,
)

horizon = scenario.horizon_for_pixel(
    x=257,
    y=130,
    observer_height_decimeters=0,
)
if horizon is not None:
    assert horizon.shape == (1440,)
    assert str(horizon.dtype) == "float32"

scenario.close_horizon_file()
```

`horizon_file_path()` and `horizon_for_pixel()` prefer compressed `.cbin`
tiles, fall back to `.bin`, and return `None` when neither file exists. The
single-pixel horizon reader supports both formats and caches one open tile file
handle for repeated calls.

The same data can be accessed from lunar longitude/latitude points:

```python
point = ls.LonLat(longitude=-45.0, latitude=-85.0)
x, y = scenario.lonlat_to_dem_pixel(point)

fig, ax = scenario.plot_horizon(
    point,
    observer_height_decimeters=0,
    center_azimuth=0.0,
)
```

Horizon plots use 1440 azimuth samples. Sample 0 is north, sample 360 is east,
and `center_azimuth` selects the center of the displayed
`[center - 180, center + 180)` azimuth window.

Sun and Earth positions can be overlaid on a horizon plot. Overlay helpers use
the existing Matplotlib axis azimuth window by default, so they can be layered
onto a horizon plot or onto an empty azimuth/elevation background. Sun overlays
default to gold and Earth overlays default to blue.

```python
sample_times = ls.times(
    "2027-01-01T00:00:00Z",
    "2027-01-02T00:00:00Z",
    step_hours=2,
)
```

Use `body_azimuth_elevation()` when you need numeric body positions. It returns
a NumPy array with shape `(time_count, 2)`: column `0` is azimuth in degrees,
and column `1` is elevation in degrees. Azimuth `0` is north and azimuth `90`
is east.

```python
angles = ls.body_azimuth_elevation(point, body="sun", times=sample_times)
```

Use `body_azimuth_elevation_over_horizon()` to subtract a local 1440-sample
horizon from the body elevation. The horizon is sampled at `0.25` degree
azimuth spacing, with sample `0` north and sample `360` east. The function
linearly interpolates between the two horizon samples surrounding the body's
true azimuth, then returns `(azimuth, elevation_over_horizon)`:

```python
dem_x, dem_y = scenario.lonlat_to_dem_pixel(point)
horizon = scenario.horizon_for_pixel(
    round(dem_x),
    round(dem_y),
    observer_height_decimeters=0,
)
angles_over_horizon = ls.body_azimuth_elevation_over_horizon(
    point,
    body="sun",
    times=sample_times,
    horizon=horizon,
)
```

For scenario-backed lookups, use the Scenario convenience method. It converts
the lon/lat point to the nearest DEM pixel, fetches the matching horizon file,
and returns the same `(azimuth, elevation_over_horizon)` array:

```python
angles_over_horizon = scenario.body_azimuth_elevation_over_horizon(
    point,
    body="sun",
    times=sample_times,
    observer_height_decimeters=0,
)
```

```python
fig, ax = scenario.plot_horizon(point, center_azimuth=90.0)

scenario.plot_body_position(
    ax,
    point,
    body="sun",
    time="2027-01-01T12:00:00Z",
    style="center",
)

scenario.plot_body_position(
    ax,
    point,
    body="earth",
    time="2027-01-01T12:00:00Z",
    style="limb",
)
```

Use `plot_azimuth_elevation_axes()` when you want the same plotting coordinate
system without drawing a horizon line:

```python
fig, ax = scenario.plot_azimuth_elevation_axes(
    center_azimuth=180.0,
    elevation_limits=(-10.0, 20.0),
)
```

Use `plot_body_path()` to overlay a body path across a sequence of datetimes.
The `style` argument controls what is drawn:

- `style="center"` draws one line through the body center.
- `style="limbs"` draws a filled band between the apparent lower and upper
  limbs. The band uses the body color with `alpha=0.5` by default.
- `style="center_and_limbs"` draws both the center line and the filled limb
  band.

The Sun apparent diameter is modeled as `0.536` degrees, and the Earth apparent
diameter is modeled as `2.0` degrees. `times` may be any iterable of datetimes
or the `TimeRange` returned by `ls.times()`.

```python
scenario.plot_body_path(
    ax,
    point,
    body="sun",
    times=sample_times,
    style="center_and_limbs",
    label="Sun",
)

scenario.plot_body_path(
    ax,
    point,
    body="earth",
    times=sample_times,
    style="limbs",
    label="Earth",
)
```

`plot_body_path()` returns the Matplotlib artist it created. For
`style="center_and_limbs"`, it returns `(center_line, limb_band)`. Plot keyword
arguments are passed through to Matplotlib; for example, `alpha=0.25` changes
the limb band opacity and `color="black"` overrides the default body color.

Use `plot_zoomed_body_path()` when you want a focused horizon view around one
or more Sun/Earth paths. It fetches the scenario horizon for the lon/lat point,
frames all body centers sampled by the `TimeRange`, keeps horizontal and
vertical degrees at the same scale, and leaves the horizon visible across the
bottom of the plot. Each body is drawn as a full-opacity limb at the first time
sample and as a translucent filled band between its lower and upper limbs over
the full time range:

```python
fig, ax = scenario.plot_zoomed_body_path(
    point,
    bodies=["sun", "earth"],
    times=sample_times,
    observer_height_decimeters=0,
)
```

The standalone `plot_body_elevations()` helper plots body elevation over time.
Pass `horizon=` to plot elevation over that horizon instead of absolute
elevation:

```python
fig, ax = ls.plot_body_elevations(
    point,
    bodies=["sun", "earth"],
    times=sample_times,
    horizon=horizon,
)
```

The Scenario method accepts the same plotting arguments and can fetch the
horizon for the lon/lat point when `over_horizon=True`:

```python
fig, ax = scenario.plot_body_elevations(
    point,
    bodies=["sun", "earth"],
    times=sample_times,
    over_horizon=True,
    observer_height_decimeters=0,
)
```

If `horizon=` is provided, that explicit horizon is used and no scenario
horizon is fetched, even when `over_horizon=True`.

### Horizon Generation Contract

Horizon generation enumerates the DEM in row-major 128 by 128 patches,
including partial patches along the right and bottom edges. Every horizon file
still has the fixed 128 by 128 shape. Pixels outside the valid DEM extent in an
edge file are filled with `-50` degrees; downstream raster products also mark
those pixels invalid so the padding cannot be mistaken for measured terrain.

Each valid pixel has 1,440 `float32` elevation-angle samples at 0.25-degree
azimuth spacing. Sample 0 is north and sample 360 is east. Files use a
pixel-major layout with azimuth samples contiguous within each pixel:

```text
horizon[pixel_y, pixel_x, azimuth_index]
```

**Directory layout and file naming.**  Horizon tiles are organised in
row-major directories below the scenario ``horizons/`` root.  The directory
hiearchy mirrors the DEM grid: a tile at pixel origin
``(tile_x, tile_y)`` lives under ``horizons/{tile_y:05d}/``.  Each tile is
identified by an observer height in decimetres:

```text
horizons/
  00000/horizon_00000_00000_000.cbin
  00000/horizon_00000_00128_000.cbin
  00128/horizon_00128_00000_000.cbin
  ...
```

The file stem is ``horizon_{tile_y:05d}_{tile_x:05d}_{height_dm:03d}`` where
``height_dm`` is the observer height rounded toward zero to three zero-padded
decimal digits.  The maximum nameable observer height is 99.9 metres (999 dm).

**File format selection.**  Readers prefer compressed ``.cbin`` tiles.  When
only a ``.bin`` file exists it is read in preference to an absent ``.cbin``.
For each tile, lookup checks the ``{tile_y:05d}`` subdirectory first, then
the horizon root; within a directory, ``.cbin`` is preferred.  Files without
``.bin`` or ``.cbin`` extensions are ignored.

**Uncompressed ``.bin`` format.**  Each ``.bin`` file contains the logical
horizon array ``(128, 128, 1440)`` written pixel-major (outermost axis is y)
in little-endian ``float32`` with no header.  Total size: exactly
``128 × 128 × 1,440 × 4 = 94,371,840`` bytes.

**Compressed ``.cbin`` format.**  Each pixel's 1,440-sample horizon is encoded
independently as a variable-length byte block.  The file consists of one
unsigned 16-bit little-endian length prefix per block, followed by the block's
encoded bytes.  There is no file header or trailer.

The compression scheme quantises ``float32`` degrees to a signed 16-bit
integer: ``quantised = round(saturate(horizon, -50.0, 50.0) × 32767 / 50.0)``.
The first sample is stored as a signed 16-bit big-endian value.  Subsequent
samples use a variable-length signed delta:

- If the delta fits in 7 signed bits (−64 to 63) it occupies one byte.
- Otherwise it occupies two bytes in big-endian with the high bit of the
  first byte set.

The maximum block length is ``2 × 1,440`` bytes.

The complete file is ``2 × 128 × 128`` length-prefix bytes followed by a
variable number of block payloads.  Readers decode block-by-block and
validate that exactly 1,440 samples are produced from each block.  A valid
tile is the only proof of completion; structural validation rejects
truncated blocks and out-of-range lengths.

The generator combines an ordered list of DEMs cumulatively: the horizon from
an earlier DEM participates in hierarchy culling for later DEMs.
Maximum-elevation pyramids remain resident on the GPU across patches. CPU
preparation, CUDA work, compression, and file writing stream the region instead
of retaining a regional horizon cube. The current scheduler uses a one-item
prepared-patch queue, a one-item writer queue, one CUDA stream, and a reusable
pool of segment and output device buffers. Multiple streams were measured but
did not materially improve throughput on the evaluation workload.

The one-patch queue bounds keep preparation and compression/writing hidden in
steady state. Neighboring-patch segment caching is intentionally disabled:
segment generation is already hidden by CUDA execution, so a row-sized cache
would currently add memory and eviction complexity without measured benefit.
This decision must be revisited if a future workload leaves the CUDA consumer
waiting for prepared work; parallel CPU preparation should be measured at the
same time.

Completed files are skipped by default. New files are staged and atomically
published so a failed calculation cannot look complete or destroy an existing
completed file.

There is intentionally no production CPU fallback for horizon generation; the
CPU implementation is too slow for operational use. CUDA kernels cannot be
interrupted while running, so cancellation is checked between bounded patch
units. Kernel compilation is lazy. The CPU and CUDA functions request Numba
disk caching so compatible compiled artifacts can be reused by a later Python
process; requiring a long-lived worker would not suit notebooks or short
scripts. In the measured fresh-process case, cache reuse reduced the first CPU
segment generation plus first CUDA execution from 13.85 to 7.31 seconds. CUDA
context creation and the real first kernel execution still have startup cost.

On the representative 16-patch sustained benchmark, the selected pipeline
processed 0.179 patches per second, used about 9.02 GB of host memory, and
peaked at 4,458 MiB of GPU memory. These figures include compression and
staged file writes, but are hardware- and terrain-specific; see the
[Phase 6 pipeline evaluation](numba-horizon-phase-6-production-pipeline.md)
for scope and detailed timings.

### Scientific Algorithm Identifiers

Every downstream product records the algorithm that produced it in
staged-job manifests and durable restart metadata.  These identifiers are
public compatibility promises for ``0.1.0rc1``:

| Algorithm identifier                          | Product family       | Description |
| --------------------------------------------- | -------------------- | ----------- |
| ``lightmap-builder-sun-fraction``             | lightmap             | 16-slice solar disk, 0.27° half-angle, truncating uint8 encoding |
| ``psr-upper-solar-limb``                      | PSR                  | Five-viewpoint vector reduction, upper-solar-limb semantics |
| ``sun-center-local-horizon-elevation``        | Sun elevation        | Sun-center terrain-relative elevation at interpolated horizon azimuth |
| ``earth-center-local-horizon-elevation``      | Earth elevation      | Earth-center terrain-relative elevation at interpolated horizon azimuth |
| ``safe-haven-per-pixel-monthly-bands``        | safe havens          | Per-pixel Earth outage detection, monthly calendar bands, streaming reducer |
| ``landed-mission-sunlight-duration``          | mission duration     | Longest sunlight-fraction run per candidate interval |
| ``landed-mission-sun-elevation-duration``     | mission duration     | Longest Sun terrain-relative elevation run per candidate interval |
| ``landed-mission-sunlight-earth-elevation-duration`` | mission duration | Longest combined sunlight + Earth elevation run per candidate interval |
| ``landed-mission-sun-earth-elevation-duration`` | mission duration   | Longest combined Sun + Earth elevation run per candidate interval |

The shared algorithm version is ``phase6b-v1``.  These identifiers and the
version appear in ``.manifest.json`` sidecars, the ``algorithm`` and
``algorithm_version`` fields of restart metadata, and may appear in future
GeoTIFF metadata tags.  Callers should not parse these fields for control
flow, but they may use them to verify that a staged or completed product was
computed with the expected algorithm.

### Shared Tiled-Product Pipeline

Horizon-derived products are computed patch-major because reading a horizon
tile is often more expensive than applying all requested times or reductions
to that tile. The shared pipeline follows this pattern:

1. load one horizon file;
2. calculate every requested result for that 128 by 128 patch; and
3. write the corresponding compressed GeoTIFF tile or tiles.

Outputs cover the full DEM grid and use compressed 128 by 128 tiles by default;
`compress=True` is the default on every downstream operation. Missing,
unreadable, and out-of-DEM horizon pixels are marked invalid with the dataset
mask. Byte products physically store zero at invalid pixels by default (or the
caller's explicit `invalid_value`), but that payload remains a potentially
valid science value and is not a nodata sentinel. Float products physically
store and declare `nodata=np.nan` by default. Consumers should use the dataset
mask as the authoritative validity representation for both families.

Multi-time products are BigTIFF files with one band per UTC time. Each band is
made of 128 by 128 tiles, compressed by default. Pass `compress=False` for
tiled but uncompressed output. The band stores its ISO-8601 UTC timestamp
and the dataset stores the complete ordered timestamp list. This layout is
intended for mission-scale histories, such as two years sampled every six
hours, and stays within TIFF's 65,535-band limit. A 74-year Metonic history is
reduced while calculating a PSR map rather than stored as a time-series TIFF.

Downstream operations can transform each valid calculated patch just before it
is written by supplying `output_transform`, `output_dtype`, and optionally
`output_transform_id`. NumPy dtype forms such as `np.uint16`,
`np.dtype("uint16")`, and `"uint16"` are equivalent. A supplied transform must
preserve the patch shape and return exactly the requested dtype. An omitted
transform ID matches another omitted ID on restart; callers may supply a
stable ID when they want restart metadata to guard against inadvertently
changing transform behavior. Integer conversion of a float product also
requires an explicitly representable integer `nodata` value because NaN cannot
be stored in an integer dtype.

Product backends use `backend="auto"`, `"cpu"`, or `"cuda"`. `auto` selects
CUDA when it is usable and otherwise runs the CPU implementation. CPU support
is an operational fallback for every horizon-derived product described below,
not merely a diagnostic reference. Small CPU/CUDA floating-point differences
are acceptable; values close to a hard threshold can amplify a tiny numerical
difference into a larger duration difference.

### Sun and Earth Vectors

Product-level functions accept explicit Moon-ME Cartesian vectors as
`float64` arrays shaped `(time, 3)`, paired with UTC timestamps. Supplying
vectors overrides time-driven vector generation, which makes controlled tests
and externally generated ephemerides possible.

At the higher level, Lunarscout generates realistic vectors with SpiceyPy.
Exact per-timestamp UTC-to-ephemeris-time conversion is the default. A faster
anchored conversion may be selected only for a calculation and time range
where its equivalence to exact conversion has been demonstrated. In
particular, mission pointing requires subsecond accuracy; a coarse historical
PSR sampling can tolerate a different error budget, but still needs explicit
product-level justification.

### Time-Series Lightmaps

A lightmap is a `uint8` BigTIFF with one band per requested time. Each pixel is
the modeled visible fraction of the solar disk, encoded by truncating
`255 * fraction`. The current model samples 16 solar-disk slices and uses a
0.27-degree apparent solar half-angle.

The calculation loads one horizon tile, loops over all time batches, and writes
one 128 by 128 output tile to each time band before moving to the next horizon
file. Time batching bounds working memory without changing the file's
band-per-time organization.

### Body-Elevation Products

The shared vector and horizon-sampling primitives also calculate Sun- or
Earth-center elevation relative to the interpolated local terrain horizon at
the body's azimuth. The public `generate_sun_elevation()` and
`generate_earth_elevation()` functions stream these values into `float32`
BigTIFF bands through the same patch-major writer; the calculation also feeds
mission-duration products. Geometric elevation above a smooth local horizontal
plane is available from the SPICE angle APIs, but is not substituted for
terrain-relative elevation in lighting products.

### Permanent Shadow

The permanent-shadow-region product is a single-band `uint8` GeoTIFF. Value
255 means the upper solar limb never clears the interpolated local horizon
under the defined sampling and apparent-diameter model; value 0 means that it
does. Invalid pixels have the configurable invalid payload plus an invalid
dataset mask.

In QGIS, render PSR as a paletted or unique-values layer with both 0 and 255
enabled. Value 0 is valid non-PSR science data, not nodata. Use the dataset mask
for transparency; do not configure zero itself as transparent or nodata.

The calculation does not create a Metonic lightmap cube. It chooses the
highest Sun vector in each horizon azimuth bin as seen from the four DEM
corners and center, unions those candidates, and reduces them directly into
one output tile per horizon patch.

### Safe-Haven Maps

Safe-haven products find Earth outages as maximal half-open intervals during
which the Earth-center elevation relative to the local terrain horizon is
strictly below an Earth threshold. For every pixel and outage, the reducer
records the longest complete contiguous interval whose sunlight fraction is
strictly below the sunlight threshold and whose interval overlaps the outage.
The low-Sun interval may begin before the Earth outage or end after it, so the
output reveals a location that remains shadowed when communication returns.
Durations are `float32`, in hours by default, and no truncation or duration
clamp is applied. Each Earth outage is one output band, timestamped with the
first occurrence of its minimum Earth elevation.

### Landed Mission-Duration Maps

Landed mission-duration products are sunlight calculations. A landing-slope
mask can be combined with their output separately. The public API provides four
separate operations rather than one function with a mode argument:

- longest continuous sunlight fraction greater than or equal to a threshold;
- longest continuous Sun-center elevation greater than or equal to a
  threshold;
- longest continuous sunlight fraction and Earth-center elevation, each
  greater than or equal to its threshold; and
- longest continuous Sun-center and Earth-center elevation, each greater than
  or equal to its threshold.

Sun and Earth elevation here always means body-center elevation relative to
the local terrain horizon at the body's azimuth, not elevation above a smooth
local horizontal plane. Every threshold comparison is inclusive.

The caller supplies one overall evaluation interval, a `datetime.timedelta`
sampling step, and a list of smaller candidate-start intervals. The result is a multi-band `float32` GeoTIFF with
one band per candidate-start interval and durations in hours or days. Helpers
construct common monthly, weekly, and fixed-width interval lists.

The condition sampled at time `t[i]` applies over `[t[i], t[i+1])`, clipped to
the overall evaluation stop. A qualifying run may begin at the first sample
inside a candidate-start interval even if the condition was already true, and
may continue beyond that smaller interval. A run still true at the overall
stop receives credit only through that stop; the result does not claim the run
ended there.

The patch reducer streams time and carries compact state rather than retaining
a full time cube. The same stateful-reducer design can later support bounded
outages, battery state-of-charge models, and user-supplied output quantizers,
but those extensions are not implemented.

The `0.1.0` scientific products do not silently apply slope suitability,
battery or power simulation, thermal modeling, traverse planning, or other
application policy. Callers may combine independent masks and models with the
generated products explicitly.

### Resume, Overwrite, and Failure Behavior

Long-running products resume by default. The writer keeps a hidden staged
BigTIFF, a manifest describing the immutable job inputs, and a durable
per-patch completion journal. For a multi-band product, one horizon patch is
the recovery unit: if any result for that patch is missing, all of its bands
are recomputed from the one loaded horizon file. A start-fresh option discards
compatible staging state and begins again.

Final output publication is atomic. A failed overwrite preserves the prior
completed product. Cancellation and exceptions leave resumable staging state
but never publish an incomplete file. Progress events are emitted and flushed
at patch boundaries with completed, skipped, and total counts.

The completion journal is authoritative for restart. Lunarscout does not infer
completion from physical TIFF blocks when journal records are missing; such
blocks are safely recomputed. Physical block recovery is deferred beyond
`0.1.0`.

Lightmap, elevation, safe-haven, and mission-duration products durably journal
each completed horizon patch, so restart recomputes at most the patch that was
in progress. The accepted CUDA PSR path checkpoints at most 16 patches and may
recompute only the unjournaled tail of that bounded batch. It does not reopen
the staged TIFF for every patch.

To discard resumable state safely, rerun the same operation with
`start_fresh=True`; Lunarscout removes the exact staged TIFF, manifest, journal,
and mask sidecar before creating a replacement. Do this only when no process is
still writing the product. The hidden files are named from the requested
output, for example `.result.tif.lunarscout-partial.tif` and its
`.manifest.json`, `.journal.json`, and optional `.tif.msk` companions. Prefer
`start_fresh=True` over manual deletion so partial sidecars cannot be left
behind. `overwrite=True` is separate: it permits replacement of a completed
output while preserving that output until the replacement is published.

### Product Troubleshooting

- A base installation intentionally reports CUDA unavailable. Install
  `lunarscout[cuda]`, then inspect `ls.cuda.status()` for the runtime versions,
  selected device, compute capability, driver API, reason, and GPU memory.
- `backend="cpu"` never probes CUDA. `backend="auto"` falls back only when a
  CUDA session cannot be initialized. `backend="cuda"` and CUDA execution,
  driver, PTX, JIT, or kernel failures raise structured `ls.CudaError` values
  and never silently retry on CPU.
- A GPU hidden by a container or sandbox is reported as unavailable to that
  process; this is not proof that the host lacks a GPU. Probe from the intended
  runtime environment.

Expected CUDA failure behavior is consistent across downstream products:

| Condition | `ls.cuda.status()` | `backend="auto"` | `backend="cuda"` |
| --- | --- | --- | --- |
| Base install without CUDA runtime | unavailable, with the `lunarscout[cuda]` install hint | CPU | structured product-specific unavailable error |
| No GPU or GPU hidden from the process | unavailable, with the runtime reason | CPU | structured product-specific unavailable error |
| Missing/incompatible driver or CUDA initialization failure | unavailable, with the probe or initialization reason | CPU if no CUDA session was created | structured product-specific unavailable error |
| PTX/JIT/kernel failure after CUDA execution starts | the earlier probe may have succeeded | structured execution error; no retry | structured execution error; no retry |

Horizon generation is CUDA-only, so every unavailable or execution-failure
row raises a structured horizon CUDA error instead of selecting CPU.

- Generated Sun/Earth vectors require configured SPICE kernels. Supplying
  explicit Moon-ME vectors avoids SPICE import and kernel loading. Kernel and
  geometry failures use `ls.SpiceKernelError` or `ls.SpiceGeometryError`.
- DEMs and output rasters require a Rasterio/GDAL installation compatible with
  the Python environment. Lunarscout rejects missing georeferencing and grid
  mismatches rather than combining arrays based on shape alone.
- Missing, corrupt, truncated, or incompatible horizon files are format or
  product errors. Invalid or absent patches remain invalid in the dataset mask;
  they are not silently interpreted as illuminated terrain.

## Examples

Executable examples live in `examples/`. They are ordinary Python programs and
are indexed in `examples/README.md`.

Start with:

```bash
.venv/bin/python examples/01_geotiff_and_coordinates.py
```

Then continue through terrain, regions, alignment, temporal cubes,
file-backed series, streaming reductions, and lighting examples as needed.

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
- file-backed temporal GeoTIFF series;
- SPICE-backed local Sun/Earth histories;
- horizon `.bin` and `.cbin` readers;
- validated horizon-generation mathematics and CUDA traversal;
- patch-streamed, resumable tiled-product storage;
- CPU and CUDA lightmap and landed mission-duration calculations;
- permanent-shadow and safe-haven product calculations; and
- source boundary separation from Lunar Analyst application code.

Less mature or explicitly provisional areas:

- default SPICE kernel selection and descriptions;
- compute-backend packaging and cached first-use behavior;
- safe-haven performance validation;
- CI and release automation;
- platform-specific installation documentation.

### Tested `0.1.0rc1` Candidate Matrix

The initial candidate is intentionally narrow. CPU installation and ordinary
tests have been exercised on Linux x86-64 with CPython 3.11 and 3.12. The
validated NVIDIA profile is Linux x86-64 with CPython 3.12, Numba 0.66.0,
Numba-CUDA 0.30.4, the CUDA 12.9.2 user-space toolkit installed by
`lunarscout[cuda]`, and an NVIDIA GeForce RTX 5090 Laptop GPU with compute
capability 12.0. The observed host driver was 580.159.03 and exposed CUDA
driver API 13.0.

This evidence does not claim support for Windows, macOS, Linux architectures
other than x86-64, Python versions outside 3.11 and 3.12, CUDA 13 user-space
toolkits, other NVIDIA GPU generations, AMD/Intel GPUs, or multi-GPU execution.
Users may test additional environments, but should report the exact wheel
version and `ls.cuda.status()` result.

Known candidate limitations:

- Horizon generation is NVIDIA-CUDA-only and requires `lunarscout[cuda]`.
- CPU implementations of every downstream lighting product are supported, but
  mission-scale CPU processing can be substantially slower than CUDA.
- Representative correctness and short installed-wheel smoke timings exist
  for safe havens and mission duration; comprehensive regional performance
  characterization is deferred until limited-user feedback shows which
  workloads matter.
- The restart journal is authoritative. Physical TIFF blocks are not recovered
  independently when journal records are missing.
- Multi-time GeoTIFF products cannot exceed TIFF's 65,535-band limit.
- HDF5 is not a public product format or installed dependency.
- Only the documented CUDA 12 installation profile is accepted for the first
  candidate; CUDA 13 support requires separate driver/toolchain validation.
- Map algebra, distance fields, path planning, battery/power simulation,
  thermal modeling, and traverse policy are outside `0.1.0rc1`.

Public API includes:

- names exported from `lunarscout.__init__`;
- documented functions and classes in public modules;
- documented file formats and manifests written by the package;
- documented exception classes and error codes.

Not public API:

- names beginning with `_`;
- tests;
- examples; and
- prototype implementations until promoted through documented public
  facades.

## Map Algebra (0.2.0rc1)

The `map_algebra` module provides an array-oriented analysis surface for
registered raster data. It is imported as a subpackage:

```python
import lunarscout as ls

ma = ls.map_algebra
```

Map algebra treats rasters as spatially registered fields and combines them
with a small number of operation families: local (cell-by-cell), focal
(neighborhood), zonal (grouped by zone), global (whole-raster), and distance
fields. It preserves explicit grids, validity masks, dtypes, and units
through every operation rather than treating map algebra as unqualified
NumPy arithmetic.

### Value Types

Two value types carry the spatial and scientific metadata needed to combine
rasters safely:

- **`Raster`** is an eager, already-materialized value. It stores a 2-D
  NumPy array together with a `GeoReference`, a boolean validity mask,
  optional units, and an optional name.
- **`RasterExpression`** is an immutable description of a calculation that has
  not yet run. It carries an inferred grid, dtype, units, and halo, but no
  computed arrays or open datasets.

`eq=False` on `Raster` is deliberate: `==` returns a cell-by-cell Boolean
`Raster`, not a whole-raster truth value. Implicit truth testing is
unavailable by design. Use comparing helpers `Raster.array_equal()`,
`Raster.allclose()`, `Raster.same_grid()`, and `Raster.same_metadata()`.

```python
raster = ma.raster(values, georef, units="metres", name="elevation")
raster.valid      # boolean mask, True means scientifically valid
raster.units      # optional trimmed string
raster.all_valid  # True if every pixel is valid
raster.shape      # (height, width)
raster.dtype      # NumPy dtype
```

### Eager and File-Backed Workflows

The API has two modes with visibly different entry points:

1. **Eager mode** accepts `Raster`, explicit NumPy arrays through adapters,
   or scalars, and returns `Raster`. All values are in memory.
2. **File-backed mode** starts with `ma.source(path)` or
   `ma.temporal_source(series)` and constructs a lazy expression.
   `ma.compute()` explicitly materializes a result. For supported local,
   coordinate, terrain, and resampling expressions, `ma.write()` reads and
   writes bounded windows; unsupported general focal/global/zonal/distance/
   temporal nodes fail during planning rather than silently materializing the
   complete raster.

Further large-raster execution coverage is deferred by project decision. Its
completed foundations and remaining tiled, streaming, reconciliation, temporal,
and resource-scaling work are tracked in
`docs/map-algebra-large-raster-plan.md`.

Never pass a file path to an eager operation. Use `ma.read()` to read a
GeoTIFF as an in-memory `Raster`, or `ma.source()` to defer reading.

```python
slope = ma.read("slope.tif", units="degrees")
sun = ma.source("mean_sun.tif", units="fraction")

candidate = (slope <= 8.0) & (sun >= 0.60)
```

Both modes use the same operation specifications, dtype inference, unit
rules, and validity rules. A `Raster` used inside a lazy expression is
automatically wrapped as an in-memory constant node.

### Constructors and Adapters

| Function | Description |
|---|---|
| `ma.raster(values, georef, *, valid, nodata, units, name)` | Construct a `Raster` from explicit arrays. |
| `ma.from_masked_array(masked, georef, *, units, name)` | Construct from a NumPy masked array. |
| `ma.from_existing(values, georef, *, units, name)` | Wrap bare `(values, georef)` results. |
| `ma.to_existing(raster, *, nodata)` | Convert back to `(values, georef)`, filling invalid cells. |
| `ma.read(path, *, band, units, name)` | Read a single-band GeoTIFF as a `Raster`. |
| `ma.source(path, *, band, units, identity)` | Read metadata only; returns `RasterExpression`. |
| `ma.compute(expression)` | Materialize a `RasterExpression` as a `Raster`. |
| `ma.write(path, expression, *, overwrite, start_fresh, dtype, invalid_value, window_width, window_height, progress_callback, cancellation_requested, checkpoint_interval)` | Evaluate a supported local, coordinate, terrain, or resampling expression in bounded windows and write a staged GeoTIFF with a GDAL mask. Supports progress reporting, cancellation, and durable checkpoint journaling for resume. |

### Grids and Grid Validation

Every non-scalar operand in an operation must share the same grid. Shape
equality alone is never accepted as grid compatibility. Use
`ls.require_same_grid()` or `Raster.same_grid()` before combining
georeferenced rasters.

Scalars broadcast over any raster. A length-one or one-dimensional array is
not a scalar and is rejected. The existing array API `ls.align()` remains
available.  Map-algebra `ma.align()` and `ma.resample_to()` are explicit
cross-grid operations that are never inserted implicitly by other
map-algebra operations.

### Validity and Mask Rules

Raster validity is a boolean mask stored alongside the values. A nodata
payload is an encoding detail of the source GeoTIFF, not a value that every
operation repeatedly compares. In-memory validity is always derived once at
ingestion from the GDAL band mask, dataset mask, alpha, and declared nodata.

Default validity rules:
- **Unary operations** preserve input validity, then invalidate newly
  undefined results (e.g. sqrt of a negative number).
- **Multi-raster operations** use the intersection of operand validity masks.
- **Comparisons** at invalid pixels are invalid, not false.
- **Boolean AND/OR/XOR** use strict validity intersection.
- **`where(condition, x, y)`** is valid where the condition is valid and the
  selected branch is valid. Invalidity in the unselected branch does not
  invalidate the result.
- **`coalesce(a, b, ...)`** selects the first valid value per pixel.
- **`fill_invalid(raster, value)`** fills invalid cells and marks them valid.
- **`set_invalid(raster, mask)`** invalidates via a Boolean condition.

Validity provenance is recorded in `Raster.validity_provenance`, which
distinguishes GDAL-band-mask, dataset-mask, alpha, nodata-derived, caller-
supplied, and all-valid sources.

### Dtype Rules

Supported dtypes are `bool`, signed and unsigned integers at standard
widths, `float32`, and `float64`. Object, string, datetime, and complex
dtypes are rejected.

- Ordinary arithmetic uses NumPy 2.x promotion (`np.result_type`).
- Comparisons and Boolean operations return `bool` in memory.
- True division returns at least `float32`; `float64` if an operand is
  `float64` or safe scalar inference requires it.
- Integer overflow uses `overflow="raise"` by default; `"promote"` widens to
  a safe dtype, and `"wrap"` follows NumPy.
- `ma.cast()` supports `casting="safe"`, `"same_kind"`, and `"unsafe"`.

### Unit Rules

Units are conservative metadata. Add, subtract, and comparison operations
require exact unit equality when both raster operands have units. A numeric
scalar threshold is interpreted in the raster operand's units. If two raster
operands are used and only one has units, an error is raised unless
`allow_unknown_units=True`. Multiplication and division of two unit-bearing
rasters require explicit `output_units`. Trigonometric operations require
`"degrees"` or `"radians"`. No operation infers metres merely because a
coordinate number is large.

### Progress and Cancellation for Windowed Writes

``ma.write()`` accepts optional progress and cancellation callbacks for
long-running expressions:

```python
def on_progress(completed: int, total: int, window_idx: int) -> None:
    print(f"Window {completed}/{total} done (idx {window_idx})")

def is_cancelled() -> bool:
    return stop_requested

ma.write(
    "result.tif", expr,
    progress_callback=on_progress,
    cancellation_requested=is_cancelled,
)
```

Progress is reported after each output window whose data and validity mask
have been successfully written. Values are monotonic. The
final completed window is reported exactly once with ``completed == total``
and its zero-based ``window_idx``. A resume with every window already
checkpointed reports ``window_idx == -1`` because no window is recomputed. If the
callback raises, the exception propagates, resources close, and no partial
output is published. The matching staged TIFF and journal are retained so a
later call can resume safely.

Cancellation is checked before execution begins and between windows. If the
``cancellation_requested`` callback returns ``True``, a structured
``OperationCancelledError`` (code ``map_algebra_cancelled``) is raised with
the completed-window count and total in its ``details``. Cancellation never
publishes an incomplete output and cleans up dataset handles and caches
deterministically. A failed overwrite preserves the previously completed
destination.

### Restart and Resumable Writes

Windowed writes are resumable by default. The writer keeps a hidden staged
GeoTIFF (``.{name}.lunarscout-partial.tif``) and a durable checkpoint
completion journal (``.{name}.lunarscout-partial.journal.json``) alongside
the output. The journal stores a compact count representing the contiguous
row-major prefix of completed windows. Its identity binds the expression
scientific identity, complete destination grid, output dtype, invalid fill
value, window layout, checkpoint interval, validity encoding, and enforced
GeoTIFF write options.
The staged TIFF carries the same identity and its dtype, CRS, transform,
nodata, block layout, and dimensions are validated before any window is
skipped.

On restart, the completed prefix in a matching journal is skipped.
Uncommitted or ambiguous windows are recomputed. A journal from a different
expression, dtype, nodata, grid, or window layout is never reused.
Truncated, malformed, stale, or out-of-range journal state is safely ignored
and all windows are recomputed in a newly staged TIFF.

The journal is atomically updated at checkpoint boundaries (default every 16
windows). At each checkpoint, the staged TIFF is closed to flush data to
disk, the journal is written atomically to a temporary file and renamed,
and the parent directory is synchronized. This provides crash-safe updates:
an interruption never leaves the journal syntactically corrupt, and only
windows recorded after the last checkpoint need recomputation.

After all windows complete, the staged TIFF and staged manifest are published
with paired exception rollback, then the journal and staging files are
removed. With
``overwrite=True``, the previous complete output is preserved until the
replacement is ready. A publication exception restores both the previous
output and its previous manifest while retaining the completed stage for
retry. With
``start_fresh=True``, existing staging, journal, manifest, and output files
are removed before execution.

The finished output is accompanied by a ``{name}.manifest.json`` sidecar
recording the expression scientific identity, output dtype, invalid fill
value, and grid dimensions for future restart identity checks.

### Local Operations

Local operations calculate each output pixel from the corresponding input
pixel or pixels. Both function calls and Python operators are supported.

**Arithmetic:** `ma.add`, `ma.subtract`, `ma.multiply`, `ma.divide`,
`ma.floor_divide`, `ma.remainder`, `ma.power`, `ma.negative`, `ma.positive`,
`ma.absolute`, `ma.square`.

**Comparisons:** `ma.equal`, `ma.not_equal`, `ma.less`, `ma.less_equal`,
`ma.greater`, `ma.greater_equal`, `ma.isclose`.

**Boolean:** `ma.logical_not`, `ma.logical_and`, `ma.logical_or`,
`ma.logical_xor`. Require Boolean operands; nonzero-integer truthiness is
not supported. Python `and`/`or`/chained comparisons raise actionable errors
explaining use of `&`/`|` and parentheses.

**Conditional and validity:** `ma.where`, `ma.coalesce`, `ma.is_valid`,
`ma.is_invalid`, `ma.set_invalid`, `ma.fill_invalid`. The sentinel
`ma.invalid` may appear in `where` branches to mark that side as invalid.

**Range and conversion:** `ma.clip`, `ma.cast`, `ma.round`, `ma.floor`,
`ma.ceil`, `ma.trunc`.

**Math:** `ma.sqrt`, `ma.exp`, `ma.log`, `ma.log10`, `ma.sin`, `ma.cos`,
`ma.tan`, `ma.arcsin`, `ma.arccos`, `ma.arctan`, `ma.arctan2`, `ma.degrees`,
`ma.radians`, `ma.hypot`.

**Classification:** `ma.reclassify_values` maps exact values,
`ma.reclassify_ranges` maps half-open `[lower, upper)` ranges, `ma.digitize`
assigns ordered bin numbers, and `ma.one_hot` returns one Boolean raster per
caller-supplied class. Reclassification's `default` may be a value,
`"preserve"`, or `"invalidate"` (the default). Input validity is always
preserved; unmatched valid cells follow the selected default behavior.

**Normalization:** `ma.normalize_minmax` and `ma.standardize` use only valid
cells when statistics are omitted. Supplying the minimum/maximum or mean/
standard deviation makes the statistics explicit. Results are `float64` and
dimensionless. A zero normalization range or zero standard deviation produces
an all-invalid result rather than assigning an arbitrary scientific value.

```python
candidate = (slope <= 8.0) & (sun >= 0.60)
score = ma.where(candidate, 0.4 * sun + 0.6 * (1.0 - slope / 8.0), ma.invalid)
```

Every classification and normalization function accepts either an eager
`Raster` or a `RasterExpression`. `one_hot` returns a tuple in the same order
as its `classes` argument; each tuple member has the same eager or expression
mode as its input.

### Terrain Operations (slope, aspect, hillshade)

Terrain operations compute gradient-derived rasters from an elevation source.
They accept a ``Raster`` to compute eagerly or a ``RasterExpression`` to return
a lazy expression node.  Expression nodes carry a one-pixel halo so that
bounded window execution produces seamless results across internal tile
boundaries.

```python
dem = ma.read("dem.tif", units="metres")

slope_deg = ma.slope(dem)
aspect_deg = ma.aspect(dem, compute_edges=True)
shade = ma.hillshade(dem, azimuth=315.0, altitude=45.0)
```

Eager results are in-memory ``Raster`` values.  For large rasters, construct
an expression and use ``ma.write()`` (or ``ma.compute()``):

```python
dem_expr = ma.source("dem.tif", units="metres")
slope_expr = ma.slope(dem_expr, units="degrees")
ma.write("slope.tif", slope_expr)
```

All terrain operations require a one-pixel source halo.  ``ma.write()`` reads
each output window with a one-pixel expansion, evaluates the terrain kernel,
and crops back to the exact output window.  This produces identical results to
``ma.compute()`` across internal window seams.

**``ma.slope(raster, *, output_nodata=np.nan, units="degrees",
compute_edges=False, scale=1.0)``**

Returns ``float32`` slope.  ``units`` may be ``"degrees"`` (default) or
``"percent"``. ``scale`` is the positive horizontal-to-vertical unit ratio
used by the established terrain kernel; elevation is divided by this value
before its gradient is calculated.

**``ma.aspect(raster, *, output_nodata=np.nan, compute_edges=False)``**

Returns ``float32`` azimuth in degrees.  Flat cells (zero gradient in both
directions) are invalid independent of the ``output_nodata`` value.

**``ma.hillshade(raster, *, output_nodata=0, azimuth=315.0, altitude=45.0,
compute_edges=False, scale=1.0, z_factor=1.0)``**

Returns ``uint8`` shaded relief.  ``azimuth`` is degrees clockwise from north;
``altitude`` is degrees above the horizon.  ``z_factor`` exaggerates vertical
relief.  Output values range from 0 to 255; a valid hillshade pixel at zero
is distinct from an invalid fill cell.

Canonical validity for all terrain products is computed from the source
elevation and neighbourhood gradient, independent of the ``output_nodata``
sentinel value.  For example, a valid aspect of 270 degrees may equal
an ``output_nodata`` of 270, and that does not make the pixel invalid.

When ``compute_edges=False`` (the default), the one-pixel border is marked
invalid because the gradient kernel cannot be applied at the raster edge.
Set ``compute_edges=True`` to calculate boundary gradients where source and
neighbourhood validity otherwise permit it.

### Resampling and Alignment

``ma.resample_to()`` is an explicit cross-grid resampling node.  It is never
inserted implicitly by other map-algebra operations.  ``ma.align()`` is the
eager ``Raster`` adapter.

```python
aligned = ma.align(dem, to=target_grid, resampling="bilinear")
resampled = ma.resample_to(dem_expr, target_grid, resampling="nearest")
```

**``ma.resample_to(raster, grid, *, resampling="nearest",
output_dtype=None, validity_coverage_threshold=None,
categorical=None, allow_unsafe=False)``**

Accepts a ``Raster`` (returns materialised ``Raster``) or a
``RasterExpression`` (returns an ``alignment.resample_to`` expression node).
Supported resampling names include ``nearest``, ``bilinear``, ``cubic``,
``cubicspline``, ``lanczos``, ``average``, ``mode``, and the full set listed
by ``ls.available_resampling_algorithms()``.

**Categorical vs. continuous safety.**  By default, integer and Boolean
source dtypes are treated as categorical: only ``nearest`` and ``mode`` are
allowed.  Interpolating or aggregating a categorical source raises
``AlignmentError`` unless ``allow_unsafe=True``.  ``mode`` on explicitly
continuous data also requires ``allow_unsafe=True``.  Boolean interpolation
always requires ``allow_unsafe=True`` because the resulting fractional values
are rarely meaningful. Override with ``categorical=False`` or
``categorical=True``. Categorical inference always uses the source dtype, not
``output_dtype``. Continuous interpolation into an integer output is rejected
unless ``allow_unsafe=True`` because it can round or truncate results. Other
unsafe dtype conversions likewise require the explicit override.

**Validity resampling.**  By default, validity is resampled with nearest-
neighbour semantics (categorical validity).  When
``validity_coverage_threshold`` is supplied as a float between 0 and 1, each
output pixel must have at least that fraction of valid source coverage to be
considered valid, using the ``average`` resampling algorithm on the source
validity mask.

**Exact 64-bit nearest sampling.**  ``nearest`` resampling uses a custom
implementation that preserves exact ``int64`` and ``uint64`` payloads beyond
the 53-bit mantissa precision of ``float64``.  GDAL's built-in nearest
resampling casts large integers through ``float64`` and can round them.

**``ma.align(raster, *, to, resampling="nearest",
output_nodata="auto", output_dtype=None, validity_coverage_threshold=None,
categorical=None, allow_unsafe=False)``**

Eagerly resample a ``Raster`` onto a destination grid.  Accepts a ``Raster``
only; ``RasterExpression`` operands must use ``ma.resample_to()``.  This is
the map-algebra adapter corresponding to the root-level ``ls.align()``.
``output_nodata="auto"`` preserves the source grid's nodata metadata; a
numeric value sets destination nodata and ``None`` disables it. Canonical
validity is always carried separately from this encoding metadata.

**No implicit resampling.**  Binary or local operations on rasters with
mismatched grids raise ``GridMismatchError`` during expression construction.
Callers must explicitly insert a ``resample_to`` node before combining
rasters from different grids.

### Coordinate Expressions

Coordinate constructors return lazy `RasterExpression` source nodes. Use
`ma.compute()` when an in-memory coordinate raster is required:

```python
rows = ma.compute(ma.row_indices(georef))
x = ma.compute(ma.projected_x(georef, anchor="center"))
lon = ma.compute(ma.longitude(georef))
```

`row_indices` and `column_indices` are zero-based pixel coordinates.
`projected_x` and `projected_y` use the affine transform and report the axis
unit declared by the grid CRS; they do not assume metres. `longitude` and
`latitude` transform through the grid's own geodetic CRS with traditional
longitude/latitude axis order and degree units. They never introduce WGS84
implicitly. `anchor` may be `"center"` or `"corner"`.

### Expression Inspection and Operation Discovery

`expression.describe()` gives a concise human-readable summary.
`expression.to_canonical_json()` returns the complete versioned, typed JSON
representation used for deterministic scientific identity. Integers are
stored as exact decimal text and finite floats use hexadecimal encoding;
unsupported parameter types are rejected rather than serialized with `repr`.

The sealed built-in operation catalog is available without executing kernels:

```python
ma.describe_operation("local.normalize_minmax")
ma.list_operations(category="coordinate")
ma.list_operations(execution_mode="file_backed")
```

The catalog cannot be extended by user callbacks in `0.2`.

`ma.plan(expression)` also reports total windows, journal/progress/cancellation
capability flags, the actually resumable execution stages, and the default
write journal identity with its enforced inputs. Planning is read-only: it does
not execute kernels or create restart artifacts.

### Focal Operations

Focal operations use a neighborhood around each output pixel. Size is odd
and positive. Edge modes: `"invalid"` (default), `"constant"`, `"nearest"`,
`"reflect"`, `"wrap"`. Valid-neighbor policies: `"require_all"`,
`"ignore_invalid"` (with `min_valid_count`), `"propagate_center"`.

**Statistics:** `ma.focal_sum`, `ma.focal_mean`, `ma.focal_min`,
`ma.focal_max`, `ma.focal_range`, `ma.focal_std` (with `ddof`),
`ma.focal_count`, `ma.focal_median`.

**Convolution:** `ma.convolve(kernel, *, normalize=False)` with finite 2-D
numeric kernels.

**Morphology (Boolean input only):** `ma.dilate`, `ma.erode`, `ma.opening`,
`ma.closing`, `ma.majority`.

```python
smoothed = ma.focal_mean(raster, size=5, edge="nearest")
opened = ma.opening(mask, size=3)
```

Focal operations are eager-only in `0.2`.

### Zonal Operations

Zonal operations group pixels by an explicitly supplied integer zone raster:

```python
stats = ma.zonal_stats(values, zones, statistics=["mean", "sum", "count"])
stats.to_dict()     # {zone_id: {"mean": ..., "count": ...}, ...}
stats.to_json()     # JSON string
stats.write_csv("zonal_summary.csv")
```

Supported statistics: `count`, `valid_count`, `invalid_count`, `sum`,
`mean`, `min`, `max`, `range`, `std`, `variance`, `median`, `p25`, `p75`,
`p90`. Zone IDs must be integer or Boolean. Invalid value cells are excluded
from statistics; invalid zone cells are not assigned to any zone.

`ma.zonal_raster(values, zones, statistic="mean")` broadcasts one statistic
back to valid zone cells.

### Global Reductions

Global reductions collapse a complete raster to statistics or summaries:

```python
stats = ma.statistics(raster)
hist = ma.histogram(raster, bins=20)
pct = ma.percentile(raster, [5, 50, 95], method="exact")
counts = ma.unique_counts(raster, max_unique=1_000)
```

`ma.statistics()` returns count, invalid count, sum, mean, min, max, range,
variance, and standard deviation. `ma.unique_counts()` fails predictably when
a safety bound is exceeded. `ma.percentile()` supports `"exact"` (linear)
and `"approximate"` (nearest) methods.

### Distance Fields

Distance fields measure proximity to Boolean seed pixels:

```python
dist = ma.distance_to(seeds, metric="euclidean", units="physical", max_distance=500.0)
signed = ma.signed_distance(mask, metric="euclidean", units="pixels")
```

Supported metrics: `"euclidean"`, `"taxicab"`, `"chessboard"`. Units: `"pixels"`
or `"physical"`. Physical Euclidean distance honours anisotropic and rotated
affine basis vectors. Physical taxicab and chessboard distances are
unsupported. Geographic (angular) CRS grids raise a structured error for
physical distance. Distance fields are eager-only in `0.2`.

### Temporal Map Algebra

Temporal adapters extend map algebra to time-series rasters. Time is a named
axis with UTC coordinates, not an extra spatial band.

**TemporalRaster** is the eager, in-memory temporal value. It stores a 3-D
array shaped `(time, y, x)`, a 1-D `datetime64` time array, a `GeoReference`,
a boolean validity mask of the same shape, optional units, signal name, and
name. Times must be non-empty, strictly increasing, and free of NaT.

An adapter creates one from a `TemporalCube`:

```python
tr = ma.from_temporal_cube(cube, units="fraction")
```

Or equivalently at root level:

```python
tr = ls.from_temporal_cube_to_raster(cube, units="fraction")
```

**TemporalRasterExpression** is the lazy counterpart. Use
`ma.temporal_source()` to create one from an in-memory `TemporalRaster`, a
`TemporalCube`, or a file-backed `TemporalGeoTiffSeries`:

```python
temporal_expr = ma.temporal_source(series)
```

Layer-wise local operations combine temporal expressions with static spatial
rasters or scalars. A spatial `Raster` or `RasterExpression` broadcasts
across every time layer. All operands must share the same spatial grid; two
temporal operands must have matching UTC time coordinates.

```python
sun_expr = ma.temporal_source(sun_series)
candidate = (sun_expr >= 0.60)  # TemporalRasterExpression
```

**Temporal reductions** collapse the time axis into a composable spatial
`RasterExpression`:

```python
mean_sun = ma.temporal_mean(sun_expr)    # RasterExpression (spatial)
valid = (mean_sun >= 0.40) & (slope <= 8.0)
valid_raster = ma.compute(valid)         # explicit whole-raster materialization
ma.write("candidate.tif", valid_raster.expression(), dtype="uint8", invalid_value=0)
```

Reductions are evaluated eagerly when applied directly to a `TemporalRaster`
(returning `Raster`), or deferred when applied to a `TemporalRasterExpression`
(returning `RasterExpression`). Supported reductions: `ma.temporal_mean`,
`ma.temporal_min`, `ma.temporal_max`, `ma.temporal_std` (with `ddof`),
`ma.temporal_sum`, `ma.temporal_count`. Reducer output dtypes use `float64`
accumulators for mean, std, and floating sum; `int64` for integer sum and
count; and the source dtype for min and max. `temporal_count` has no units.
The bounded spatial writer does not yet execute temporal reduction nodes;
materialize them explicitly with `ma.compute()` before writing.

Use `ma.compute_temporal()` to materialize a `TemporalRasterExpression` as
an in-memory `TemporalRaster`. Eager computation is suitable for small-to-
medium datasets; file-backed temporal expressions that would produce full
cubes are only materialized when the caller explicitly requests it.

### Lunar Constraints

The map-algebra implementation is planetary-neutral. It does not:

- assume WGS84, mean sea level, north-up grids, or square pixels;
- download or consult Earth basemaps, SRTM, or weather datasets;
- infer metres from large coordinate numbers;
- substitute Earth geodesics for lunar distance measurements; or
- fold mission policy (thresholds, weights, suitability rules) into
  operations; these must be caller inputs.

Operations that require a specific lunar datum, radius, or body model live in
the terrain or lunar-science API with explicit parameters, not in the generic
algebra.

### Error Handling

Map-algebra errors use structured subclasses of `MapAlgebraError` with stable
`code=` values and repair-oriented `details=` dictionaries. Principal error
classes include `RasterValidationError`, `MapAlgebraGridError`,
`MapAlgebraDTypeError`, `MapAlgebraUnitError`, `MapAlgebraExpressionError`,
`MapAlgebraOperationError`, `MapAlgebraStorageError`, and `DistanceFieldError`.

## Architecture Overview

The public Python package lives under `src/lunarscout`. The normative design is
described in [ARCHITECTURE.md](ARCHITECTURE.md). Public modules validate user
inputs and present notebook-friendly APIs; private engines handle bounded
patch scheduling, CPU/CUDA sessions, horizon storage, and resumable GeoTIFF
products. Heavy dependencies remain lazy so `import lunarscout` is lightweight.

Core modules:

- `georeference.py`: CRS, affine transform, and raster grid metadata.
- `geotiff.py`: GeoTIFF read and write helpers.
- `terrain.py`: slope, aspect, and hillshade-style array operations.
- `alignment.py`: grid compatibility and resampling.
- `regions.py`: connected-region analysis.
- `temporal.py`: UTC time ranges and in-memory temporal cubes.
- `temporal_store.py`: file-backed temporal GeoTIFF series.
- `spice.py`: SPICE kernel download, cache, and furnishing helpers.
- `spice_geometry.py`: local-frame Sun/Earth vector and angle histories.
- `scenario.py`: filesystem-safe scenario path helpers only.
- `_numba_horizon/`: private horizon generation, CPU/CUDA lighting kernels,
  product reducers, vector resolution, and resumable tiled-product machinery.

The package must not contain FastAPI routes, web UI code, assistant/RAG logic,
Lunar Analyst job handlers, scenario database mutation, or notebook-runner
helpers.

## Testing

Python tests live in `tests/`. Ordinary tests must run without a GPU. CUDA
integration tests are explicitly enabled and distinguish GPU visibility,
backend selection, and actual kernel execution.

Representative commands:

```bash
.venv/bin/python -m pytest -q

LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 \
PYTHONPATH=src \
.venv/bin/python -m pytest tests/numba_horizon -q -p no:cacheprovider
```

Release artifacts must not be built directly in a checkout that may contain a
stale `build/lib` directory. Install the development extra, require a clean
working tree, and build into a new or empty directory outside the repository:

```bash
python -m pip install -e '.[dev]'
python scripts/build_release_artifacts.py \
    /tmp/lunarscout-0.1.0rc1 \
    --upload-target testpypi
```

The script copies only Git-visible source into a temporary directory, builds a
wheel and sdist in isolation, runs Twine, enforces the distribution-content
policy, and writes `release-artifacts.json` with the source commit, environment,
target index, filenames, sizes, hashes, and entry counts. It never uploads.
Release mode refuses a dirty tree or a nonempty output directory. The
`--allow-dirty` and `--skip-twine` switches are diagnostic conveniences and
always produce a report with `candidate_artifacts: false`. A true value means
only that artifact construction passed; it does not supersede the other release
gates in `docs/PLAN1.md`.

The repository CI definition runs the ordinary CPU suite on Python 3.11 and
3.12. Its separate packaging job uses the same release-artifact script, rebuilds
a wheel from the sdist, installs the wheel into a fresh environment, runs
`pip check`, verifies lightweight import outside the checkout, and executes the
installed public smoke tests. CI has no .NET or managed-runtime step. Real CUDA
acceptance remains explicitly gated and must run on the documented NVIDIA host.

## Roadmap

Current open work:

- complete safe-haven performance measurements and longer end-to-end horizon
  benchmarks with identical compression and write scope;
- complete the CPU/CUDA scientific and operational failure matrices;
- validate clean Python 3.11, Python 3.12, and NVIDIA installations;
- finalize cache behavior and package metadata;
- add CI; and
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
- CPU and CUDA backend troubleshooting;
- release and compatibility policy;
- contribution guide; and
- security and data provenance notes.
