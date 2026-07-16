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
- file-backed timestamped GeoTIFF series;
- SPICE-backed Sun and Earth local-frame histories; and
- optional native lighting and permanent-shadow products.

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

### Root Functions

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
| `ls.GenerateHorizons(...)`                                                 | Low-level native horizon-generation wrapper. Prefer `scenario.generate_horizons()`. |

### Scenario Methods

| Method                                                                  | Summary                                                                                  |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `scenario.root_path()`                                                  | Return the resolved scenario root directory.                                             |
| `scenario.path(relative_path)`                                          | Resolve a scenario-relative path, allowing the root itself.                              |
| `scenario.output_path(relative_path)`                                   | Resolve a non-empty scenario-relative output path.                                       |
| `scenario.dem_path()`                                                   | Return the canonical primary DEM path, `dem.tif`.                                        |
| `scenario.horizons_path()`                                              | Return the canonical horizons directory, `horizons/`.                                    |
| `scenario.hillshade_path()`                                             | Return `hillshade.tif` in the scenario root.                                             |
| `scenario.slope_path()`                                                 | Return `slope.tif` in the scenario root.                                                 |
| `scenario.aspect_path()`                                                | Return `aspect.tif` in the scenario root.                                                |
| `scenario.roughness_path()`                                             | Return `roughness.tif` in the scenario root.                                             |
| `scenario.create_hillshade(...)`                                        | Create the native hillshade GeoTIFF.                                                     |
| `scenario.create_slope(...)`                                            | Create the native slope GeoTIFF.                                                         |
| `scenario.create_aspect(...)`                                           | Create the native aspect GeoTIFF.                                                        |
| `scenario.create_roughness(...)`                                        | Create the native roughness GeoTIFF.                                                     |
| `scenario.generate_horizons(...)`                                       | Generate native horizon files into `scenario.horizons_path()`.                           |
| `scenario.horizon_patch_pixel(x, y)`                                    | Convert DEM pixel coordinates to pixel coordinates inside a 128x128 horizon patch.       |
| `scenario.horizon_patch_row_col(x, y)`                                  | Return the horizon patch row and column containing a DEM pixel.                          |
| `scenario.horizon_file_path(x, y, observer_height_decimeters)`          | Return the matching `.cbin` or `.bin` horizon file path, preferring `.cbin`, or `None`.  |
| `scenario.horizon_from_open_file(file_handle, patch_x, patch_y)`        | Read one horizon from an open horizon file as a 1440-sample `float32` array.             |
| `scenario.horizon_for_pixel(x, y, observer_height_decimeters)`          | Fetch one DEM pixel horizon, caching one open file handle.                               |
| `scenario.close_horizon_file()`                                         | Close the cached open horizon file handle.                                               |
| `scenario.lonlat_to_dem_pixel(point)`                                   | Convert a `LonLat` to DEM pixel coordinates.                                             |
| `scenario.plot_azimuth_elevation_axes(...)`                             | Create an empty azimuth/elevation Matplotlib axis.                                       |
| `scenario.plot_horizon(point, ...)`                                     | Plot the stored horizon for a lon/lat point.                                             |
| `scenario.body_azimuth_elevation_over_horizon(point, body, times, ...)` | Fetch the scenario horizon and return body elevation over that horizon.                  |
| `scenario.plot_body_elevations(point, bodies, times, ...)`              | Plot body elevations, optionally fetching the scenario horizon with `over_horizon=True`. |
| `scenario.plot_body_position(ax, point, body, time, ...)`               | Overlay a body center point or apparent limb on an azimuth/elevation axis.               |
| `scenario.plot_body_path(ax, point, body, times, ...)`                  | Overlay a body center path and/or translucent limb band.                                 |
| `scenario.plot_zoomed_body_path(point, bodies, times, ...)`             | Plot one or more body limb paths against the horizon in an equal-scale zoomed view.      |
| `scenario.sun_fraction(...)`                                            | Generate a native temporal sun-fraction product.                                         |
| `scenario.sun_over_horizon_deg(...)`                                    | Generate a native temporal Sun-over-horizon product.                                     |
| `scenario.earth_over_horizon_deg(...)`                                  | Generate a native temporal Earth-over-horizon product.                                   |
| `scenario.psr(...)`                                                     | Generate a native permanent-shadow product.                                              |

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

Native runtime helpers are available under `ls.native`, and SPICE kernel
management helpers are available under `ls.spice`.

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

Native GDAL-backed terrain products can also be generated directly from a
scenario DEM when the native runtime is available:

```python
scenario.create_hillshade(overwrite=False)
scenario.create_slope(overwrite=False)
scenario.create_aspect(overwrite=False)
scenario.create_roughness(overwrite=False)
```

These methods write to the canonical scenario paths: `hillshade.tif`,
`slope.tif`, `aspect.tif`, and `roughness.tif`.

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

Scenario helpers can locate and read existing horizon tiles. DEM pixel
coordinates are zero-based, with `x` as column and `y` as row. Horizon patches
are 128 by 128 pixels.

```python
scenario.generate_horizons(
    compress_horizons=True,
    surrounding_dems=[
        "surrounding/mosaic_outer.tif",
    ],
)

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

`generate_horizons()` writes to `scenario.horizons_path()`. Pass explicit
`dem_paths` when you want full control over the ordered DEM list, or pass
`surrounding_dems` to prepend `scenario.dem_path()` automatically. Relative DEM
paths are resolved below the scenario root.

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
- file-backed temporal GeoTIFF series;
- SPICE-backed local Sun/Earth histories; and
- source boundary separation from Lunar Analyst application code.

Less mature or explicitly provisional areas:

- default SPICE kernel selection and descriptions;
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
- `spice.py`: SPICE kernel download, cache, and furnishing helpers.
- `spice_geometry.py`: local-frame Sun/Earth vector and angle histories.
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
