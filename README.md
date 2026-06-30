# Lunarscout

Lunarscout is the notebook-first Python surface for lunar mission analysis in
Lunar Analyst.

Executable capability examples are indexed in
[`examples/README.md`](../../examples/README.md).

The initial package provides single-band GeoTIFF input/output, georeferencing
and coordinate conversion, GDAL-compatible slope/aspect/hillshade operations,
connected-region analysis, explicit grid alignment, filesystem-safe scenario
paths, UTC-aware temporal arrays, and file-backed timestamped GeoTIFF series.
Raster values are ordinary NumPy arrays.

## Installation in this repository

Lunarscout uses the GDAL Python bindings supplied by the supported Lunar
Analyst runtime. GDAL is an external runtime prerequisite rather than a PyPI
dependency because its Python package must match the installed native GDAL
library.

```bash
.venv/bin/python -m pip install -e packages/lunarscout
```

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
```

`Scenario` currently provides naming and containment only. Its standard paths
are `dem.tif` and `lighting/horizons`; it does not read `scenario.db`, register
products, publish layers, or create directories.

Grid compatibility is never inferred from array shape alone. Verify it or
align explicitly before combining rasters:

```python
ls.require_same_grid(left_georef, right_georef)

aligned, aligned_georef = ls.align(
    source,
    source_georef,
    to=right_georef,
    resampling="bilinear",
)
```

## Optional native runtime

Install Python.NET support only when native calculations are needed:

```bash
.venv/bin/python -m pip install -e 'packages/lunarscout[native]'
```

Capability checks do not initialize CLR:

```python
native_status = ls.native.status()
if native_status["available"]:
    loaded_status = ls.native.initialize()
```

Initialization delegates to Lunar Analyst's existing native bootstrap and
keeps all production moonlib access behind `MoonlibBridge`.

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
output allocation exceeds the configured limit; storage is never changed
automatically. File-backed generation uses preflighted temporary disk scratch
because the native stream is patch-major, then atomically publishes the
timestamped series. Progress and cancellation callbacks are supported.
`sun_fraction` returns float32 values in `[0, 1]`; Sun/Earth horizon-margin
methods return float32 degrees.

Native permanent-shadow generation is an explicit file-producing operation:

```python
psr_path = scenario.psr("analysis/psr.tif", overwrite=False)
```

It atomically publishes a native-grid uint8 GeoTIFF without registering it in
scenario state. Value `255` means the Sun center never clears the local horizon
across the native operation's six-hour samples from 1970-01-01 through
2044-01-01; value `0` means it clears the horizon at least once. Observer
elevation is currently fixed at zero. A GDAL validity mask distinguishes
unknown pixels whose 128 x 128 horizon tile was missing or unreadable: mask
`255` is valid and mask `0` is unknown. Complete output uses GDAL's virtual
all-valid mask, so no physical mask or nodata sentinel is stored. Progress and
cancellation callbacks are supported, and failed overwrites preserve the prior
output.

## Temporal arrays

Time domains use UTC coordinates and include an aligned stop value:

```python
time_range = ls.times(
    "2027-01-01T00:00:00Z",
    "2027-01-02T00:00:00Z",
    step_hours=2,
)
cube = ls.TemporalCube(values, time_range, georef)
mean, mean_georef = ls.temporal_mean(cube)
```

`TemporalCube.values` is a normal NumPy array shaped `(time, y, x)`. The
container also carries its read-only UTC `datetime64[us]` coordinates and
spatial `GeoReference`.

Large or persistent time series can be stored as one tiled, compressed,
single-band GeoTIFF per timestamp, plus an authoritative manifest and an
optional portable VRT:

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

`TemporalGeoTiffSeries` has no `.values` property. Direct reads use a bounded
open-dataset LRU and optional byte-budgeted full-layer LRU; temporal reducers
stream layers without constructing the full three-dimensional cube.

Producers that generate large series incrementally can avoid creating a
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

The writer validates every layer, writes into a staging directory, and only
publishes the directory after its manifest, VRT, and completion digest are
ready. Errors and cooperative cancellation remove staging output while
preserving any previously completed series at the destination.
