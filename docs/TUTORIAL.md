# Lunarscout Tutorial

Lunarscout is a notebook-first Python library for lunar terrain, raster,
temporal, horizon, lighting, visibility, and landed-mission analysis. This
tutorial explains why the library is designed the way it is and then uses the
repository's example programs as a guided tour of its public API.

This document is a learning path, not an exhaustive API reference. Use the
[User Guide](USER_GUIDE.md) when you need exact parameter contracts, file
formats, numeric and validity rules, restart behavior, error codes, or the
current maturity of a feature. Use the
[examples index](../examples/README.md) for a compact requirements table and
command summary.

## Why Lunarscout Exists

Lunar analysis often joins several kinds of work:

- georeferenced elevation and derived terrain products;
- raster masks and connected candidate regions;
- time-varying illumination or visibility;
- Sun and Earth geometry from SPICE;
- terrain horizons and lighting products; and
- combinations of those products into an auditable analysis.

Those calculations should be usable from an ordinary Python script or
notebook without requiring a web application, database, job service, or agent
runtime. Lunarscout was therefore separated from Lunar Analyst as a standalone
calculation library. The dependency direction remains one-way:

```text
lunar_analyst -> lunarscout
```

This boundary is useful even if you never use Lunar Analyst. It keeps the
scientific API independent of application state and makes calculations easier
to test, reproduce, package, and call from other software.

Several design choices follow from that goal.

### NumPy values, explicit spatial meaning

Raster values remain ordinary NumPy arrays. Their spatial meaning is carried
by a `GeoReference`, which records the projection, affine transform, width,
height, pixel size, and optional nodata metadata. An array shape alone never
proves that two rasters describe the same place. Lunarscout requires callers
to compare grids or align them explicitly.

### Validity is not just a payload value

A valid measurement may be zero, `False`, or class ID zero. Conversely, an
invalid pixel may still contain a plausible numeric payload. The map-algebra
API therefore carries a Boolean validity array separately from raster values.
GeoTIFF output uses dataset masks where necessary so GIS software can preserve
that distinction.

### Scientific choices are visible

Units, dtype promotion, integer overflow, non-finite values, resampling,
neighborhood edges, connectivity, thresholds, and compute backends can change
an analysis. Lunarscout makes these choices explicit instead of hiding them in
application defaults. Structured exceptions report failures with stable codes
and inspectable details.

### Small and large workflows use different value types

An eager `Raster` holds values in memory and is convenient for exploration.
A `RasterExpression` describes a calculation without immediately evaluating
it. Supported expressions can be inspected, planned, computed in memory, or
written in bounded windows. Likewise, a `TemporalCube` holds a complete
`(time, y, x)` array, while a `TemporalGeoTiffSeries` streams timestamped
layers from storage.

### Lunar assumptions stay lunar

The library does not silently introduce WGS84 or assume that coordinate
numbers are metres. Coordinate transformations use the raster's declared CRS.
Physical distance and angular operations require the metadata needed to
interpret them safely.

### Expensive capabilities are explicit

Importing `lunarscout` does not initialize CUDA, load SPICE kernels, open
rasters, or write files. Horizon generation is explicitly CUDA-only.
Downstream products expose `backend="auto"`, `"cpu"`, and `"cuda"`; explicit
CUDA requests never silently fall back. File-producing operations have
overwrite, staging, progress, cancellation, and restart contracts appropriate
to their cost.

### Scenarios organize files without becoming application state

A `Scenario` provides safe, conventional paths such as `dem.tif`,
`horizons/`, and scenario-relative analysis outputs. It is a filesystem helper
and product facade, not a database or workflow engine. This keeps the same
calculation usable from a notebook, command-line program, service, or other
calling application.

## Before You Start

The examples are part of the source repository, so this tutorial assumes that
you are running from a checkout. From the repository root:

```bash
cd /e/projects/lunarscout
export PYTHONPATH="$PWD/src"
export LUNARSCOUT_EXAMPLE_WORKSPACE=/tmp/lunarscout_tutorial
```

For ordinary library use outside a checkout, install the package into your
environment with `python -m pip install lunarscout`. The source-tree commands
below use the repository environment so they always run the examples beside
the code they document.

Use the repository environment:

```bash
.venv/bin/python -c "import lunarscout as ls; print(ls.__version__)"
```

Most examples accept `--workspace` as an alternative to the environment
variable:

```bash
.venv/bin/python examples/01_geotiff_and_coordinates.py \
  --workspace /tmp/lunarscout_tutorial
```

Examples 01–10 share deterministic fixtures in the workspace. Later
synthetic examples reuse those fixtures when appropriate. It is helpful to
keep one workspace while following the tutorial so that you can inspect the
accumulating outputs.

The scripts are ordinary Python programs with a `main()` function. They are
also intentionally short enough to open beside a notebook and copy one
conceptual section at a time.

## Choose a Learning Route

You do not have to run every example in numeric order.

| Goal                                                                   | Recommended examples |
| ---------------------------------------------------------------------- | -------------------- |
| Learn raster, terrain, region, alignment, and time-series fundamentals | 01–10                |
| Learn Sun/Earth geometry, horizons, and production lighting products   | 11–17                |
| Learn eager and expression-based map algebra                           | 18–22, 25, 27, 31    |

The numbering reserves space for the broader example portfolio, so some
numbers are intentionally absent. There is currently no example 14, 23, 24,
26, or 28–30.

The requirements also change as you progress:

| Examples                 | Additional requirement                                                      |
| ------------------------ | --------------------------------------------------------------------------- |
| 01–10, 18–22, 25, 27, 31 | None beyond the normal CPU installation; inputs are synthetic               |
| 11                       | Network access on first use to obtain and cache SPICE kernels               |
| 12                       | Synthetic horizon bundle download, SPICE geometry, and Matplotlib           |
| 13                       | Synthetic horizon bundle download; explicit vectors avoid SPICE loading     |
| 15                       | A real scenario with `dem.tif` and `horizons/`; long production calculation |
| 16                       | User DEMs, the CUDA installation profile, and a compatible NVIDIA GPU       |
| 17                       | A real scenario with `dem.tif` and `horizons/`; SPICE; CPU or CUDA          |

## How the Example Directory Is Organized

The numbered `.py` files are the supported command-line learning sequence.
[The examples README](../examples/README.md) is its compact index.

Several other files support that sequence:

- [`_example_support.py`](../examples/_example_support.py) creates the
  deterministic lunar grid, DEM, UTC illumination series, and shared argument
  parser. It also downloads and verifies the synthetic horizon bundle. These
  helpers keep example setup short; they are not part of Lunarscout's public
  library API.
- [`data/synthetic_horizon_manifest.json`](../examples/data/synthetic_horizon_manifest.json)
  records the expected synthetic-horizon asset and checksums.
- [`horizon_examples.ipynb`](../examples/horizon_examples.ipynb) is a
  supplemental exploratory notebook for body paths over a real scenario. It
  contains a developer-specific scenario path and assumes an interactive
  environment, so edit the path and establish `scenario` before running its
  horizon cells. Examples 11 and 12 are the self-contained starting point for
  the same concepts.

## Part 1: Raster and Spatial Foundations

The first four examples use the package-root API:

```python
import lunarscout as ls
```

They return ordinary arrays plus explicit georeferencing. Map algebra, covered
later, builds a richer `Raster` object on top of the same spatial contracts.

### Example 01: GeoTIFFs and coordinates

Run:

```bash
.venv/bin/python examples/01_geotiff_and_coordinates.py
```

[Example 01](../examples/01_geotiff_and_coordinates.py) creates a deterministic
64 by 64 lunar scenario, reads its DEM, converts selected pixels to projected
and longitude/latitude coordinates, and writes a copy to:

```text
$LUNARSCOUT_EXAMPLE_WORKSPACE/synthetic_scenario/analysis/dem_copy.tif
```

The central pattern is:

```python
dem, georef = ls.read_geotiff(path)
if georef is None:
    raise RuntimeError("The DEM must be georeferenced.")
```

The explicit `georef is None` check matters. Lunarscout can read raster values
whose spatial metadata is incomplete, but spatial analysis must not pretend
that such values are registered. The example also shows scalar and vectorized
pixel conversion; the same `GeoReference` handles both.

Before continuing, inspect the printed dtype, shape, nodata value, affine
coordinates, and lunar longitude/latitude. Notice that the transform is driven
by the declared lunar CRS rather than an Earth default.

### Example 02: Terrain products

Run:

```bash
.venv/bin/python examples/02_terrain_products.py
```

[Example 02](../examples/02_terrain_products.py) computes slope, aspect, and
hillshade from the synthetic DEM. Each operation returns both values and the
product grid:

```python
slope_values, slope_georef = ls.slope(
    dem,
    georef,
    output_nodata=-9999.0,
)
```

The example writes three products beneath
`synthetic_scenario/analysis/terrain/`. It is worth opening them together in a
GIS and checking that their extent and pixel grid agree. Slope and aspect are
scientific quantities; hillshade is a visualization. Do not treat hillshade
intensity as a physical illumination product.

### Example 03: Connected regions

Run:

```bash
.venv/bin/python examples/03_region_filtering.py
```

[Example 03](../examples/03_region_filtering.py) turns a slope threshold into
a candidate mask, then:

1. labels connected components;
1. assigns each candidate cell its region size;
1. removes regions below a size threshold;
1. applies morphological opening; and
1. extracts the remaining borders.

The outputs under `analysis/regions/` demonstrate the difference between a
per-pixel condition and an area that is spatially coherent enough to inspect.
The threshold of 80 pixels and the slope threshold of 8 degrees are teaching
choices, not landing-site recommendations.

Try changing `comparator`, `cleanup`, or the connectivity parameters described
in the user guide. The scientific question should determine these choices;
they should not be selected merely because one result looks smoother.

### Example 04: Grid comparison and alignment

Run:

```bash
.venv/bin/python examples/04_alignment.py
```

[Example 04](../examples/04_alignment.py) creates two arrays with the same
shape but shifted spatial grids. It first shows that `ls.same_grid()` returns
false, then aligns the shifted raster explicitly:

```python
aligned, aligned_georef = ls.align(
    source,
    source_georef,
    to=reference_georef,
    resampling="bilinear",
    output_nodata=-9999.0,
)
ls.require_same_grid(aligned_georef, reference_georef)
```

This is one of Lunarscout's most important contracts: alignment is an analysis
step, not an incidental array operation. Choose nearest-neighbor resampling
for categorical values and a reviewed continuous method for continuous
measurements. The library rejects several unsafe categorical or
integer-interpolation combinations unless the caller explicitly accepts them.

## Part 2: Time as a First-Class Coordinate

Temporal arrays use UTC coordinates and the in-memory shape
`(time, y, x)`. The next four examples show the progression from a small cube
to a streamed file-backed series.

### Example 05: An in-memory temporal cube

Run:

```bash
.venv/bin/python examples/05_temporal_cube.py
```

[Example 05](../examples/05_temporal_cube.py) constructs six synthetic
illumination layers in a `TemporalCube`, prints its shape, dtype, memory use,
and UTC range, then calculates mean, minimum, maximum, and standard deviation.

This is the simplest temporal model: the complete cube is in memory and each
reducer collapses the time axis into one spatial raster. Use it for modest
arrays and exploratory notebook work.

### Example 06: A file-backed temporal series

Run:

```bash
.venv/bin/python examples/06_file_backed_series.py
```

[Example 06](../examples/06_file_backed_series.py) stores the same conceptual
data as one single-band GeoTIFF per timestamp plus an authoritative manifest
and an optional VRT. It demonstrates:

- reading by zero-based layer index;
- selecting the nearest layer to a UTC time;
- inspecting signal name, dtype, shape, and grid; and
- locating the generated VRT.

A `TemporalGeoTiffSeries` deliberately has no `.values` property. That absence
prevents code from accidentally materializing a large time series merely to
inspect it.

### Example 07: Incremental temporal writing

Run:

```bash
.venv/bin/python examples/07_incremental_writer.py
```

[Example 07](../examples/07_incremental_writer.py) writes generated layers
directly through `TemporalGeoTiffSeriesWriter` instead of first constructing a
`TemporalCube`. Its progress callback reports each committed timestamp, and
the context manager finalizes the series only after all layers succeed.

This pattern is intended for real producers: simulation, lighting, or
instrument code can yield one layer at a time while memory remains
proportional to a layer rather than the complete time domain. The full writer
contract also includes cancellation, staging, abort, and failed-overwrite
behavior; see the user guide before adapting it to a long job.

### Example 08: Streaming temporal reductions

Run:

```bash
.venv/bin/python examples/08_streaming_reductions.py
```

[Example 08](../examples/08_streaming_reductions.py) passes a file-backed
series to the same public temporal reducer names used for an in-memory cube.
The implementation streams layers and maintains only the required
accumulators.

Compare examples 05 and 08. The scientific operation is the same, but the
storage and memory strategy differs. This is a recurring Lunarscout design
pattern: keep domain names stable while making materialization explicit.

## Part 3: GIS Inspection and a First Workflow

### Example 09: Inspect a temporal VRT in QGIS

Run:

```bash
.venv/bin/python examples/09_qgis_vrt.py
```

[Example 09](../examples/09_qgis_vrt.py) prints the generated VRT, its band
descriptions, and one timestamp GeoTIFF. Open both paths in QGIS. VRT band
`n + 1` corresponds to Python layer index `n`.

Set the QGIS project CRS from the lunar raster. QGIS cannot safely transform a
custom lunar CRS into its default WGS 84 project merely because both use
longitude and latitude terminology. The VRT supplies band organization and
descriptions; it does not configure QGIS's temporal controller automatically.

### Example 10: Terrain and illumination screening

Run:

```bash
.venv/bin/python examples/10_landing_site_screening.py
```

[Example 10](../examples/10_landing_site_screening.py) combines several
earlier lessons:

1. read a DEM;
1. stream the temporal mean illumination;
1. verify grids;
1. calculate slope;
1. combine slope and illumination thresholds;
1. remove small disconnected candidates; and
1. write candidate and border rasters.

This example uses the lower-level array and masked-array API. Later examples
22 and 31 express related calculations with map-algebra `Raster` objects and
validity masks. Comparing the versions is a useful way to understand what map
algebra adds.

Again, the thresholds are illustrative. Lunarscout supplies calculation and
validation mechanisms; it does not decide that a site is safe or suitable.

## Part 4: Sun, Earth, Horizons, and Lighting

Examples 11–17 move from small deterministic raster work toward celestial
geometry and production products. Read the requirements before running them.

### Example 11: SPICE vectors and angles

Run:

```bash
.venv/bin/python examples/11_spice_vectors.py
```

[Example 11](../examples/11_spice_vectors.py) evaluates Sun and Earth geometry
for a lunar south-polar point. It shows:

- UTC iteration with `ls.iter_times()` and `ls.times()`;
- north/east/down vectors;
- NumPy and DataFrame return forms; and
- azimuth/elevation conventions.

The first run may download and cache the configured SPICE kernels. Kernel
loading is lazy: importing Lunarscout alone does not touch the SPICE pool.
Vector units and reference frames are part of the scientific contract, so read
the corresponding user-guide section before supplying vectors to another
tool.

### Example 12: Body paths over terrain horizons

Run:

```bash
.venv/bin/python examples/12_body_and_horizon_plots.py
```

[Example 12](../examples/12_body_and_horizon_plots.py) downloads a checked
synthetic horizon scenario on first use and writes PNG plots beneath:

```text
$LUNARSCOUT_EXAMPLE_WORKSPACE/analysis/plots/
```

It progresses from body elevation over time to polar
azimuth/elevation axes, a stored terrain horizon, instantaneous Sun/Earth
positions, limb bands, and a zoomed path against the horizon. Matplotlib uses a
noninteractive backend, so the script works in a terminal or CI environment.

The important conceptual shift is from elevation relative to an ideal local
horizontal plane to elevation relative to actual surrounding terrain. A body
above zero geometric elevation may still be hidden by a ridge.

### Example 13: A lightmap from explicit vectors

Run:

```bash
.venv/bin/python examples/13_synthetic_lightmap.py
```

[Example 13](../examples/13_synthetic_lightmap.py) reuses the synthetic
horizons but supplies explicit Moon-ME Sun vectors. It therefore avoids SPICE
kernel loading during the product call and executes the lightmap on CPU.

The output is a multi-band BigTIFF with one `uint8` band per time sample.
The encoded values represent visible solar fraction according to the product
contract; they are not Boolean illumination flags. The script reads each band
back and prints its range and valid-pixel count.

This explicit-vector path is valuable for reproducibility and integration:
another trusted geometry source can provide vectors without changing the
terrain-lighting engine.

### Example 15: A production permanent-shadow map

[Example 15](../examples/15_python_psr.py) is intentionally different from the
small synthetic examples. It expects a real scenario containing `dem.tif` and
compatible `horizons/`, evaluates a long 1970–2044 time range, and defaults to
CUDA.

Supply paths explicitly rather than relying on its developer-oriented
defaults:

```bash
.venv/bin/python examples/15_python_psr.py \
  --scenario /data/mons_mouton \
  --output /data/mons_mouton/analysis/psr.tif \
  --backend cpu
```

Use `--backend cuda` only with the CUDA profile and a supported NVIDIA device.
Use `--overwrite` only when replacing an existing completed product is
intentional. The progress reporter estimates remaining time, and the final
inspection counts PSR, illuminated, and invalid pixels using the GeoTIFF
dataset mask.

This may be a substantial calculation. Do not run it merely to complete the
tutorial.

### Example 16: Generate terrain horizons

Horizon generation is the one core capability that is deliberately CUDA-only.
Run [Example 16](../examples/16_generate_horizons.py) with your own primary
DEM:

```bash
.venv/bin/python examples/16_generate_horizons.py \
  --primary-dem /data/site/dem.tif \
  --surrounding-dem /data/regional-dem.tif \
  --output /data/site/horizons
```

The primary DEM defines the output grid. Repeat `--surrounding-dem` to extend
terrain coverage. Structurally complete tiles are reusable, so an interrupted
run can resume; `--overwrite` explicitly requests regeneration. The script
checks CUDA availability before starting and emits structured progress by
tile.

Surrounding terrain matters because a distant ridge can define a local
horizon. Providing more DEMs is not equivalent to changing the primary output
grid.

### Example 17: Downstream lighting and mission products

[Example 17](../examples/17_downstream_products.py) is the production product
menu. It accepts a scenario with `dem.tif` and `horizons/` and can generate:

- lightmaps and permanent-shadow maps;
- Sun and Earth terrain-relative elevation;
- safe-haven duration;
- two single-signal mission-duration products; and
- two combined Sun/Earth mission-duration products.

Start with one short CPU lightmap:

```bash
.venv/bin/python examples/17_downstream_products.py \
  /data/site \
  --product lightmap \
  --start 2029-01-01T00:00:00Z \
  --stop 2029-01-02T00:00:00Z \
  --step-hours 6 \
  --backend cpu
```

Outputs default to the scenario-relative
`analysis/example-products/` directory. Passing `--product all` is convenient
but potentially expensive. Review the thresholds and candidate-start
intervals in the script before treating any duration as mission policy.

## Part 5: Eager Map Algebra

Map algebra is normally imported through the package root:

```python
import lunarscout as ls

ma = ls.map_algebra
```

An eager `Raster` holds:

- a two-dimensional NumPy `values` array;
- a matching Boolean `valid` array;
- a `GeoReference`;
- optional units; and
- an optional analyst-facing name.

Operations return new rasters and do not mutate their inputs.

### Example 18: Raster and local-algebra basics

Run:

```bash
.venv/bin/python examples/18_map_algebra_basics.py
```

[Example 18](../examples/18_map_algebra_basics.py) prints every important part
of a small slope raster, including its values, validity, grid, units, dtype,
valid count, and memory use. It then demonstrates non-mutating metadata and
validity helpers, filled and masked interchange forms, arithmetic,
`minimum`, `clip`, `sqrt`, comparisons, strict Boolean operations, and
`is_valid`/`is_invalid`.

Pay particular attention to the difference between:

```python
slope.values
```

and:

```python
slope + 1.0
```

The first is an ordinary array. The second is registered map algebra and
preserves the spatial and scientific metadata. Use NumPy for nonspatial arrays
and map algebra when grid, validity, dtype, or units must follow the result.

Raster Boolean logic uses `&`, `|`, and `~`. Python `and` and `or` perform
scalar truth testing and are intentionally rejected.

### Example 19: Validity, `where`, and `coalesce`

Run:

```bash
.venv/bin/python examples/19_map_algebra_validity.py
```

[Example 19](../examples/19_map_algebra_validity.py) is the most important
validity lesson. Its inputs contain:

- a valid zero;
- invalid zero payloads; and
- invalid nonzero payloads that look plausible.

Ordinary arithmetic intersects operand validity. `where` requires the
condition and the selected branch to be valid. `ma.invalid` creates an
explicit invalid branch. `coalesce` takes the first valid value, while
`set_invalid` removes cells and `fill_invalid` converts missing cells into
valid caller-supplied values.

Reversing the order of `coalesce` changes its meaning, as the script makes
visible. Filling with zero also changes meaning: the cells cease to be missing.
Do not use filling merely to make a plot look complete.

### Example 20: Grids and explicit alignment

Run:

```bash
.venv/bin/python examples/20_map_algebra_grids.py
```

[Example 20](../examples/20_map_algebra_grids.py) revisits example 04 with
`Raster` values. Direct algebra rejects two same-shaped rasters on shifted
grids. `ma.align()` materializes an explicitly aligned eager raster, after
which addition is safe.

The second half computes row, column, projected-x, and projected-y rasters.
Coordinate constructors are expressions, so the example calls `ma.compute()`
to materialize them. Their units come from the declared lunar CRS.

For lazy file-backed work, use `ma.resample_to()` rather than `ma.align()`.
The distinction keeps resampling visible in the expression graph.

### Example 21: Units and numerical policies

Run:

```bash
.venv/bin/python examples/21_map_algebra_numerics.py
```

[Example 21](../examples/21_map_algebra_numerics.py) demonstrates why dtype
and unit policy belong in a scientific API:

- addition requires compatible units;
- scalar thresholds are interpreted in the raster's units;
- multiplying two unit-bearing rasters requires declared derived units;
- trigonometric functions require angle units;
- checked integer arithmetic can raise, wrap, or promote;
- casts distinguish safe and explicitly unsafe conversion; and
- non-finite results can become invalid, be retained, or raise.

Run the same operation under all three `numeric_errors` policies and compare
both values and validity. A retained `NaN` is still a valid payload under
`"keep"`; under `"invalid"` the validity mask carries the failure.

The example uses tiny `int8` values so overflow is obvious. The same policies
also protect exact `int64` and `uint64` boundaries without routing integers
through floating point.

## Part 6: Composing Map-Algebra Workflows

### Example 22: Weighted suitability

Run:

```bash
.venv/bin/python examples/22_map_algebra_suitability.py
```

[Example 22](../examples/22_map_algebra_suitability.py) adapts terrain and
temporal outputs into eager map-algebra rasters. It constructs hard slope and
illumination constraints, calculates a caller-supplied weighted score, and
uses:

```python
ma.where(candidate, score, ma.invalid)
```

to preserve scores only for candidate cells.

The example then writes expressions derived from the eager rasters. Inspect
both `candidate.tif` and `candidate_score.tif` in the scenario's
`analysis/screening/` directory. The weights and thresholds are illustrative,
not library recommendations.

### Example 25: Focal cleanup and distance

Run:

```bash
.venv/bin/python examples/25_map_algebra_focal.py
```

[Example 25](../examples/25_map_algebra_focal.py) introduces neighborhood and
distance operations:

- a 3 by 3 focal mean smooths slope;
- morphological opening removes small Boolean features; and
- Euclidean distance measures pixels to the opened steep-area mask.

Focal results depend on window size, edge policy, and invalid-neighbor policy.
Morphology depends on footprint and connectivity. Distance can be measured in
pixels or physical units when the grid supports the requested interpretation.
Review these parameters in the user guide before applying the pattern to
mission data.

The current file-backed planner does not advertise general focal or distance
nodes from file-backed sources. This example evaluates small eager rasters;
unsupported lazy plans fail explicitly rather than materializing silently.

### Example 27: Terrain expressions and bounded writes

Run:

```bash
.venv/bin/python examples/27_map_algebra_terrain_resample.py
```

[Example 27](../examples/27_map_algebra_terrain_resample.py) is the bridge from
eager exploration to file-backed expressions. It:

1. calculates eager slope and hillshade references;
1. creates a lazy DEM source with `ma.source()`;
1. builds terrain expression nodes with their required halo;
1. writes results in bounded windows;
1. checks the written results against eager calculation;
1. explicitly resamples hillshade to a different grid; and
1. composes terrain and illumination into a windowed score.

The script's validity check demonstrates why output fill and validity are
separate. A stored fill of `-1.0` identifies invalid payload bytes for
interchange, while the dataset mask remains authoritative.

Before adapting this example to a regional raster, inspect the calculation:

```python
print(ma.explain(expression))
print(ma.plan(expression, output=destination))
```

Planning is read-only and rejects unsupported operations before creating the
destination. `ma.compute()` is the explicit whole-raster path;
`ma.write()` is the bounded path for the supported expression inventory.

### Example 31: Temporal map algebra

Run:

```bash
.venv/bin/python examples/31_map_algebra_temporal.py
```

[Example 31](../examples/31_map_algebra_temporal.py) opens the synthetic
illumination series with `ma.temporal_source()`, reduces it with
`ma.temporal_mean()`, and combines the resulting spatial expression with an
eager slope constraint.

This makes the materialization boundary visible: the temporal source is
file-backed, the reduction streams its layers, `ma.compute()` creates the
spatial result, and ordinary spatial map algebra handles the final candidate
mask.

Use a `TemporalRasterExpression` when you want temporal composition and a
`RasterExpression` after a time-reducing operation produces one spatial field.
The user guide documents which temporal nodes are layer-wise, reducing,
streamed, or materializing.

## What to Inspect After Running the Examples

After the CPU/synthetic path, your workspace should contain a useful miniature
scenario:

```text
/tmp/lunarscout_tutorial/
├── synthetic_scenario/
│   ├── dem.tif
│   └── analysis/
│       ├── alignment/
│       ├── focal/
│       ├── regions/
│       ├── screening/
│       ├── synthetic_sun.temporal/
│       ├── temporal/
│       └── terrain/
├── terrain_resample/
├── horizon_scenario/     # after examples 12 or 13
└── analysis/plots/       # after example 12
```

Open representative files with `gdalinfo`, Rasterio, or QGIS and verify:

- CRS and affine transform;
- raster extent and dimensions;
- dtype and nodata metadata;
- dataset mask or alpha-like transparency;
- band count and descriptions;
- valid zero values;
- timestamp ordering; and
- whether the threshold and units printed by the script match your intent.

Do not use visual agreement alone as a correctness test. Two rasters can look
similar while differing in grid, mask, units, or numeric policy.

## Common Mistakes

### Combining equal-shaped arrays without checking grids

Use `ls.same_grid()` or `ls.require_same_grid()`. Align or resample explicitly
when grids differ.

### Treating nodata, zero, and invalid as synonyms

They are different concepts. In map algebra, inspect `raster.valid`. In stored
GeoTIFFs, inspect the dataset mask as well as nodata metadata.

### Letting QGIS assume an Earth CRS

Set the project CRS from the lunar layer and avoid an implicit transformation
to WGS 84.

### Confusing an expression with a computed raster

Building a `RasterExpression` does not calculate its pixels. Use
`ma.explain()` and `ma.plan()` to review it, `ma.compute()` to materialize it,
or `ma.write()` for supported bounded execution.

### Assuming all operations are windowed

The eager API is broader than the current file-backed planner. Planning an
unsupported focal, global, zonal, distance, or temporal composition raises a
structured error before output creation.

### Requesting CUDA as an informal preference

`backend="cuda"` is a requirement and never falls back. Use `"auto"` when CPU
fallback is acceptable, and use `"cpu"` when CUDA should not even be probed.
Horizon generation has no CPU fallback.

### Overwriting completed products casually

Long-running product writers protect completed outputs. Supply overwrite only
after reviewing the exact destination and accepting replacement.

### Treating tutorial thresholds as mission policy

Slope, illumination, Earth visibility, connectivity, and duration thresholds
in these examples exist to teach the API. Site safety and mission policy
require independent scientific and engineering review.

## Where to Go Next

Use the [User Guide](USER_GUIDE.md) for:

- the complete public function inventory;
- exact raster, grid, validity, dtype, and unit contracts;
- temporal manifests and storage behavior;
- SPICE frames, units, and kernel handling;
- horizon and product algorithms;
- progress, cancellation, restart, and overwrite behavior;
- map-algebra operation families and registry inspection; and
- structured exceptions and stable error codes.

Use [Architecture](ARCHITECTURE.md) when you need to understand package
boundaries, pipeline internals, resource management, or the relationship
between eager, windowed, temporal, CPU, and CUDA execution.

When beginning your own analysis, start with the smallest eager raster that
expresses the question. Make grids, validity, units, thresholds, and numeric
policies explicit. Once the calculation is understood, move persistent or
regional inputs to file-backed sources, inspect the expression and plan, and
only then write the full product.
