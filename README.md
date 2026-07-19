# Lunarscout

Lunarscout is a standalone, notebook-first Python library for lunar terrain,
horizon, lighting, visibility, and landed-mission analysis.

Executable capability examples are indexed in
[`examples/README.md`](examples/README.md).

The Python/Numba production implementation generates CUDA horizon tiles and
CPU/CUDA lightmaps, permanent-shadow maps, Sun/Earth terrain-relative
elevation, safe havens, and four landed mission-duration products. It also
provides GeoTIFF I/O, georeferencing, terrain operations, connected-region
analysis, explicit grid alignment, filesystem-safe scenario paths, UTC-aware
temporal arrays, and file-backed timestamped GeoTIFF series.

## Installation

Install the release candidate from the configured package index:

```bash
python -m pip install lunarscout
```

For source-tree development in this repository, use the repository virtual
environment and source path:

```bash
export PYTHONPATH="$PWD/src"
/e/projects/lunarscout/.venv/bin/python -m pytest -q
```

Rasterio supplies Lunarscout's maintained Python GDAL boundary. SpiceyPy is a
core dependency because generated Sun/Earth vectors are part of the supported
product surface, but it and the SPICE kernel pool remain lazy. Numba is also a
core dependency; installing Lunarscout does not require the CUDA toolkit or an
NVIDIA driver. Only horizon generation and explicitly requested CUDA product
execution require a compatible NVIDIA device and driver at runtime.

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

`Scenario` provides filesystem-safe conventional paths and delegates public
product operations. It does not read `scenario.db`, register products, publish
layers, or become an application state container.

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

## Horizons and lighting products

Generate horizons with the ordered primary and surrounding DEMs. This operation
is deliberately CUDA-only and resumes structurally complete tiles:

```python
horizons = ls.generate_horizons(
    "/data/site/horizons",
    ["/data/site/dem.tif", "/data/regional-dem.tif"],
)
```

Every downstream product accepts `backend="auto"`, `"cpu"`, or `"cuda"` and
defaults to `"auto"`. CPU never probes CUDA; explicit CUDA never falls back;
automatic selection falls back to CPU when CUDA cannot initialize. All product
functions return `Path`, remain quiet unless `verbose=True`, and support simple
fraction progress, structured progress events, cancellation, durable restart,
and failed-overwrite protection.

```python
mission_times = ls.times(
    "2029-01-01T00:00:00Z",
    "2029-02-01T00:00:00Z",
    step_hours=6,
)
lightmap = scenario.lightmap(
    "analysis/lightmap.tif",
    times=mission_times,
    backend="auto",
)
psr = scenario.psr(
    "analysis/psr.tif",
    times=mission_times,
    backend="cpu",
)
```

Explicit Moon-ME vectors prevent SpiceyPy import and SPICE kernel loading. See
the [user guide](docs/USER_GUIDE.md) for vector units, file formats, masks,
restart semantics, and the complete product API.

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
