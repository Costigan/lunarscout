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
| `ls.mission_duration_from_sunlight_and_earth(...)`                         | Generate durations satisfying sunlight and Earth-elevation thresholds.              |
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

Production horizon generation requires a supported NVIDIA GPU. Reading stored
horizons and running horizon-derived calculations do not: lightmaps,
permanent-shadow maps, safe-haven maps, and landed mission-duration maps have
CPU implementations, with CUDA acceleration selected when requested and
available. Distribution extras for these product APIs will be documented when
the public wrappers are promoted.

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

Supplying `sun_vectors_m=` or `earth_vectors_m=` with matching `times=` avoids
SPICE import and kernel loading. `verbose=False` is the default. Applications
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

Uncompressed `.bin` files contain little-endian `float32` values. Compressed
`.cbin` files preserve the established Lunarscout horizon format. Readers
prefer `.cbin` when both forms exist.

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

### Shared Tiled-Product Pipeline

Horizon-derived products are computed patch-major because reading a horizon
tile is often more expensive than applying all requested times or reductions
to that tile. The shared pipeline follows this pattern:

1. load one horizon file;
2. calculate every requested result for that 128 by 128 patch; and
3. write the corresponding compressed GeoTIFF tile or tiles.

Outputs cover the full DEM grid and use compressed 128 by 128 tiles. Missing,
unreadable, and out-of-DEM horizon pixels receive a deterministic payload and
are marked invalid with the dataset mask. The payload is configurable and
defaults to zero. Consumers must use the validity mask rather than infer
validity from the payload value.

Multi-time products are BigTIFF files with one band per UTC time. Each band is
made of compressed 128 by 128 tiles. The band stores its ISO-8601 UTC timestamp
and the dataset stores the complete ordered timestamp list. This layout is
intended for mission-scale histories, such as two years sampled every six
hours, and stays within TIFF's 65,535-band limit. A 74-year Metonic history is
reduced while calculating a PSR map rather than stored as a time-series TIFF.

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
the body's azimuth. Separate private Sun and Earth product functions stream
these values into `float32` BigTIFF bands and the same patch-major writer; the
calculation also feeds mission-duration products. Their public facade remains
provisional. Geometric elevation above a smooth local horizontal plane is
available from the SPICE angle APIs, but is not substituted for
terrain-relative elevation in lighting products.

### Permanent Shadow

The permanent-shadow-region product is a single-band `uint8` GeoTIFF. Value
255 means the upper solar limb never clears the interpolated local horizon
under the defined sampling and apparent-diameter model; value 0 means that it
does. Invalid pixels have the configurable invalid payload plus an invalid
dataset mask.

The calculation does not create a Metonic lightmap cube. It chooses the
highest Sun vector in each horizon azimuth bin as seen from the four DEM
corners and center, unions those candidates, and reduces them directly into
one output tile per horizon patch.

### Safe-Haven Maps

Safe-haven products find Earth outages as maximal half-open intervals during
which the Earth-center elevation relative to the local terrain horizon is
strictly below an Earth threshold. For every pixel and outage, the reducer
records the longest contiguous interval whose sunlight fraction is strictly
below the sunlight threshold. Durations are `float32`, in hours by default,
and no truncation or duration clamp is applied. Each Earth outage is one output
band, timestamped with the first occurrence of its minimum Earth elevation.

### Landed Mission-Duration Maps

Landed mission-duration products are sunlight calculations. A landing-slope
mask can be combined with their output separately. The private engine
implements four separate top-level operations rather than one function with a
mode argument:

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

The caller supplies one overall evaluation interval and a list of smaller
candidate-start intervals. The result is a multi-band `float32` GeoTIFF with
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

## Examples

Executable examples live in `examples/`. They are ordinary Python programs and
are indexed in `examples/README.md`.

Start with:

```bash
.venv/bin/python examples/00_geotiff_and_coordinates.py
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
- public horizon-generation API and installed-wheel validation of the promoted
  horizon-derived product APIs;
- compute-backend packaging and cached first-use behavior;
- structured public exceptions for product failures;
- safe-haven performance validation;
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
- examples; and
- prototype implementations until promoted through documented public
  facades.

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

TODO: Replace representative local commands with release-quality verification
instructions once CI and packaging are finalized.

## Roadmap

Current open work:

- promote the validated Python horizon generator behind a small public function
  and `Scenario` convenience;
- finish structured exceptions, path preflight, progress, and cancellation
  contracts for those public functions;
- complete safe-haven performance measurements and longer end-to-end horizon
  benchmarks with identical compression and write scope;
- package CPU dependencies and optional CUDA acceleration for ordinary
  notebooks and short-lived scripts, including reusable compiled caches;
- add CI;
- remove transitional implementation and dependency artifacts after the
  Python product APIs satisfy their retirement gates; and
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
