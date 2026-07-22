# Lunarscout Architecture

Status: Normative target architecture for the Python-only Lunarscout library.

## 1. Architectural decision

Lunarscout is a standalone, in-process Python library. Its implementation does
not use Python.NET, the CLR, .NET, or `moonlib`. The former C# source tree, C#
build, managed-runtime bootstrap, and managed wrappers were removed after the
archival boundary commit `c9c4e66` and remain available from repository history.

The normal execution model is deliberately simple:

```python
import lunarscout as ls

scenario = ls.open_scenario("/data/mons_mouton")
scenario.generate_horizons()
scenario.psr("analysis/psr.tif")
```

Both calls run in the caller's Python process. This model must work in a
Jupyter or marimo kernel, a short-lived script, a test process, and an
agent-launched process. A daemon or long-lived worker may be offered by an
application, but Lunarscout neither requires nor owns one.

"Python-only" describes the library, its orchestration, and its extension
boundary. It does not mean that every numerical instruction is interpreted
Python. Lunarscout uses NumPy, Numba, Rasterio/GDAL, PyProj, SciPy, SpiceyPy/
CSPICE, and an NVIDIA CUDA driver where the selected operation requires them.
These dependencies are reached through maintained Python APIs; there is no
second Lunarscout implementation hosted in another runtime.

## 2. Scope and dependency direction

Lunarscout owns reusable lunar terrain, raster, temporal, ephemeris, horizon,
lighting, visibility, and landed-mission calculations. It also owns the file
formats and resumable product-writing machinery needed by those calculations.

Lunarscout does not own web routes, user-interface state, assistant or RAG
logic, application job records, scenario database mutation, or notebook
execution infrastructure. Dependency direction remains one-way:

```text
lunar_analyst and other applications
                  |
                  v
              lunarscout
                  |
        +---------+----------+
        |         |          |
      NumPy   Python I/O   optional compute/data providers
                         (Numba CUDA, Rasterio, SpiceyPy)
```

No module in `src/lunarscout` imports Lunar Analyst application code.

## 3. Design principles

The implementation follows these rules:

1. **One public Python surface.** Normal scripts use `import lunarscout as ls`.
   Public functions and `Scenario` methods call Python implementations rather
   than exposing backend-specific entry points.
2. **Lazy capabilities.** Importing `lunarscout` initializes no CUDA context,
   SPICE kernel pool, or GDAL dataset and performs no network access. Heavy or
   optional packages are imported at the operation boundary.
3. **Explicit data contracts.** NumPy dtypes, array axes, units, coordinate
   frames, georeferencing, masks, and timestamps are part of each API contract.
4. **Patch-major streaming.** Horizon-derived products load one horizon patch
   and calculate all requested results for that patch before releasing it.
   Regional time cubes are not retained in memory.
5. **Bounded resources.** Queue capacities, time-batch sizes, device buffers,
   caches, and open datasets have explicit bounds independent of region size.
6. **Durable products.** File-producing operations preflight inputs, stage
   output, journal completed patches, resume interrupted jobs, and publish
   atomically. A failed overwrite preserves the prior completed product.
7. **Scientific behavior before mechanical parity.** Preserved C# file and
   calculation behavior is documented and tested. Known defects are not
   perpetuated merely for byte parity, and intentional differences have named
   semantics and independent tests.
8. **Useful CPU execution.** Horizon-derived calculations have CPU fallbacks.
   CUDA acceleration is optional for them. Horizon generation is the deliberate
   exception: its production implementation requires a supported NVIDIA GPU
   because the CPU algorithm is not operationally useful.

## 4. Package layers

The package is divided into four conceptual layers. Imports point downward;
storage and compute backends never define the public API.

```text
Public API and Scenario facade
  horizon generation | lightmaps | PSR | safe havens | mission duration
                               |
Domain models and algorithms
  georeference | temporal | vectors | horizon math | product reductions
                               |
Bounded execution and durable storage
  patch scheduler | CPU/CUDA sessions | horizon store | GeoTIFF product store
                               |
Python ecosystem providers
  NumPy/Numba | Rasterio/GDAL | PyProj | SpiceyPy/CSPICE
```

### 4.1 Core modules

The existing core modules remain backend-independent:

- `georeference.py` owns CRS, affine transform, dimensions, and coordinate
  conversion through `GeoReference`.
- `geotiff.py` owns ordinary GeoTIFF reads and writes.
- `terrain.py`, `alignment.py`, and `regions.py` own array-oriented terrain,
  resampling, and connected-region operations.
- `temporal.py` owns UTC time ranges and in-memory `(time, y, x)` cubes.
- `temporal_store.py` owns the existing directory-backed timestamped GeoTIFF
  series for general temporal arrays.
- `spice.py` owns lazy kernel discovery, verified download, furnishing, and
  kernel-pool lifecycle.
- `spice_geometry.py` owns public local-frame vector and angular histories.
- `scenario.py` owns filesystem-safe scenario paths and delegates calculations
  to domain services. It does not become an application state container.

### 4.2 Horizon and lighting modules

The validated implementation currently lives under the private
`_numba_horizon` package while its public contracts mature. In the final
layout, public facades such as `horizon.py`, `lightmap.py`, and `product.py`
delegate to focused private modules for:

- host geometry and ray-segment generation;
- hierarchy construction and CUDA traversal;
- `.bin`/`.cbin` encoding and decoding;
- patch enumeration and bounded scheduling;
- Sun/Earth vector resolution;
- CPU and CUDA lightmap, PSR, safe-haven, and mission-duration kernels; and
- resumable tiled GeoTIFF storage.

Names beginning with `_` remain implementation details. Users do not import
Numba kernels, CUDA sessions, queue payloads, or storage journals directly.

### 4.3 Removed runtime layer

The following have no counterpart in the final architecture:

- `_native_runtime` CLR discovery and bootstrap;
- a `native.status()` or `native.initialize()` lifecycle;
- `native_horizon.py`, `native_temporal.py`, `native_product.py`, and
  `native_terrain.py` wrappers around managed code;
- `pythonnet` and `moonlib.dll` configuration;
- .NET SDK/runtime requirements and C# binary packaging.

The former managed public names are not compatibility aliases. Applications
must use the Python product functions and structured domain exceptions.

## 5. Public API and lifecycle

The package root remains curated. High-level operations intended for notebooks
and scripts are exported from `lunarscout.__init__`; capability-specific detail
stays in focused modules. Representative target calls are:

```python
import lunarscout as ls

scenario = ls.open_scenario("/data/site")

scenario.generate_horizons(
    surrounding_dems=["surrounding/outer.tif"],
    compress=True,
)

scenario.lightmap(
    "analysis/mission-light.tif",
    start="2029-01-01T00:00:00Z",
    stop="2029-02-01T00:00:00Z",
    step_hours=6,
    backend="auto",
)

scenario.psr("analysis/psr.tif", backend="auto")
scenario.safe_havens("analysis/safe-havens.tif", backend="auto")
scenario.mission_duration("analysis/mission-hours.tif", backend="auto")
```

Each product also has a lower-level function accepting paths, georeferencing,
timestamps, and explicit vectors. Explicit vectors override vector-generation
arguments. This makes controlled tests and externally generated ephemerides
possible without changing the high-level API.

Objects that own resources support context management or an explicit `close()`.
CUDA sessions, open horizon tiles, Rasterio datasets, and SPICE kernel state do
not depend on garbage-collection timing. Lunarscout never globally shuts down
CUDA or clears a kernel pool it did not establish.

## 6. Canonical data contracts

### 6.1 Raster grids

Raster values are NumPy arrays. `GeoReference` carries width, height, CRS WKT,
and the six-value GDAL affine transform. Combining two georeferenced arrays
requires `same_grid` or `require_same_grid`; matching array shapes alone do not
establish compatibility.

Raster axes are `(y, x)`. Pixel coordinates use zero-based `x` columns and `y`
rows. Windows and patch origins are expressed as `(tile_x, tile_y)` to avoid
silently exchanging row and column.

### 6.2 Time

Public timestamps are timezone-aware UTC `datetime` values or ISO-8601 strings
with offsets. Stored timestamps use normalized ISO-8601 UTC text. In-memory
temporal cubes have shape `(time, y, x)`.

Time-series BigTIFF products use one band per time. Band `i + 1` represents
timestamp `i`; the timestamp is stored on that band and in one ordered
dataset-level timestamp list. The file is band-interleaved and each band is
tiled in 128 by 128 pixel blocks.

### 6.3 Celestial vectors

Product kernels consume C-contiguous `float64[time, 3]` Cartesian position
vectors in Moon-ME, in meters, paired one-to-one with UTC timestamps. The
vectors are geometric positions relative to the Moon center. Their magnitudes
are retained because apparent disk geometry may depend on range.

Notebook-facing local-body APIs continue to expose lunar NED vectors where
appropriate. Conversion between Moon-ME and patch-local frames occurs once at
the product boundary, not independently in every pixel kernel.

### 6.4 Horizon tiles

The stable horizon patch contract is:

- 128 by 128 observer pixels;
- 1,440 azimuth samples per observer at 0.25-degree spacing;
- sample 0 north and sample 360 east;
- `float32` angular elevation in degrees after decoding;
- pixel-major logical shape `(128, 128, 1440)`; and
- azimuth contiguous within each pixel.

The uncompressed `.bin` representation is little-endian `float32` in this
order. The `.cbin` representation preserves the established per-horizon
quantization and signed 7/15-bit delta encoding with little-endian block
lengths. Readers structurally validate the entire tile rather than trusting a
file name or extension.

Partial right and bottom DEM patches still produce full 128 by 128 horizon
files so existing random-access readers retain a fixed record contract.
Observers outside the valid DEM rectangle receive deterministic `-50` degree
horizons, the compressed format's minimum representable value. Those padded
pixels never become valid pixels in a derived GeoTIFF.

## 7. Horizon generation

### 7.1 Inputs and preprocessing

Horizon generation accepts an ordered collection of aligned elevation models,
their geospatial metadata, an observer height, and an output directory. It
validates grid compatibility and output paths before initializing CUDA.

For each DEM, the host builds or loads a factor-four maximum-elevation pyramid.
Pyramid cache identity includes the source DEM identity and must evolve to
include the algorithm/cache format version; array length alone is not adequate
production validation. Immutable pyramids are uploaded once and remain
device-resident across patches.

Patches are enumerated row-major with `x` advancing fastest. Host preparation
generates the subpatch-center and halo ray segments used by all pixels in that
patch. Segment preparation is compiled with Numba for CPU execution.

### 7.2 CUDA traversal

Horizon traversal is implemented with Numba CUDA. One linear CUDA index maps
to:

```text
pixel   = linear_index % pixel_count
azimuth = linear_index // pixel_count
```

Consequently a warp normally traces one azimuth for 32 adjacent observer
pixels. This preserves locality for the dominant DEM, pyramid, observer, and
segment reads. It deliberately does not coalesce the smaller final output
writes, because pixel-major output places adjacent observers' same-azimuth
values 1,440 floats apart.

The kernel performs adaptive level-zero and hierarchical traversal, bilinear
sampling, cumulative multi-DEM slope accumulation, and final `float32`
slope-to-degree conversion. Hierarchy culling at a bilinear cell boundary uses
the maximum of all four samples participating in interpolation. Non-cullable
level-zero traversal uses the validated adaptive approximation and 1 mm
boundary nudge. Grid-convergence and optional near-field reference behavior are
not silently added to the mechanical production contract.

There is no production CPU fallback for horizon generation. If a compatible
NVIDIA device and CUDA driver are unavailable, the operation raises a
structured capability error before modifying outputs. Existing horizon files
remain readable and usable on CPU-only machines.

### 7.3 Bounded horizon pipeline

The default pipeline has:

- one CPU segment producer;
- a prepared-patch queue with capacity one;
- one CUDA consumer on the default stream;
- a writer queue with capacity one; and
- one writer that converts, compresses, stages, and publishes each tile.

Queue sizes are configuration points for measurement, not functions of patch
count. Fixed-shape segment and output device buffers are reused. Immutable
pyramids are shared. Results are written immediately, so memory never contains
a regional horizon cube.

Multiple CUDA streams remain an evaluated option, but are not the default:
two and four streams increased memory without materially increasing sustained
throughput on the reference workload. The pixel-fast within-warp mapping is an
invariant even if scheduling changes.

Neighboring-patch segment caching is also not enabled by default. With the
current single-stream pipeline, preparation is hidden behind CUDA execution
and the prepared queue stays full. A row-sized FIFO cache would add substantial
memory and eviction complexity without measured throughput benefit. It may be
introduced only after measurement shows CUDA starvation, and must be compared
with multicore segment generation. CUDA concurrency alone is not evidence that
a segment cache is needed.

### 7.4 Horizon completion

Existing files are skipped only after complete structural validation. Lookup
preserves compressed-before-raw and partitioned-before-legacy naming rules.
Each new tile is written to a unique sibling staging path, flushed and
synchronized, and atomically replaces its destination. Failed generation or a
failed overwrite removes staging data and preserves any prior complete tile.

Cancellation is checked before and between patch preparation, CUDA execution,
and writing. It does not interrupt an already running CUDA kernel.

## 8. Shared horizon-derived product engine

Lightmaps, PSR maps, elevation/visibility products, safe-haven maps, and landed
mission-duration maps use one patch-major engine:

```text
enumerate output patches (row-major)
        |
        +-- completed in durable journal? -- yes --> skip
        |
       no
        v
read and decode one horizon tile
        |
build bounded patch-local vector frames
        |
run CPU or CUDA calculation in bounded time batches
        |
write every band/window for this patch
        |
flush dataset, then journal patch completion
```

The horizon file is loaded once because decompression and disk access can cost
more than the calculation. All requested times and reductions for the patch
are processed before the horizon array is released.

Every downstream algorithm supports `backend="auto"`, `"cpu"`, and `"cuda"`:

- `auto` chooses CUDA when a compatible NVIDIA backend can be initialized and
  otherwise uses CPU;
- `cpu` never probes or initializes CUDA; and
- `cuda` fails with an actionable structured error rather than falling back.

The CPU path is compiled with Numba where it provides a useful speedup and has
the same bounded, patch-major interface as CUDA. CPU and CUDA results need not
be bit-identical. Tests and benchmarks report encoded mismatch counts and
maximum scientific deltas; small accepted differences do not force slower
arithmetic solely for byte parity.

CUDA sessions retain the current horizon, vector data, and bounded result
buffers. CPU sessions yield the same result batches. Neither backend allocates
`time_count * region_height * region_width` storage.

## 9. Product algorithms

### 9.1 Time-series lightmaps

A lightmap stores sunlight fraction as `uint8`, with one BigTIFF band per UTC
sample and one 128 by 128 compressed tile per horizon patch per band. Encoding
uses truncation of `255 * visible_fraction`; 0 is fully obscured and 255 is
fully visible.

The calculation models the apparent solar disk using 16 vertical slices and a
0.27-degree solar half-angle, interpolates the 0.25-degree horizon at the true
solar azimuth, and computes one tile for each Sun vector. A configurable time
batch bounds device and host output. Writing remains patch-major even though
the file is band-major: load a horizon once, loop over time batches, and write
the corresponding tile to every time band.

### 9.2 Permanent-shadow maps

The PSR product is a single-band `uint8` map: 255 indicates that the Sun center
never clears the local horizon under the defined sampling/model, and 0
indicates that it clears at least once. Validity is represented separately by
the dataset mask.

PSR does not materialize a Metonic lightmap series. It reduces the candidate
Sun vectors first: at the four DEM corners and center, retain the
highest-elevation vector in each of 1,440 azimuth bins, then take the union of
the retained indices. The reduced vectors are applied directly to each loaded
horizon tile to produce one output tile.

### 9.3 Safe-haven maps

Safe-haven products identify locations that remain shadowed during intervals
in which Earth is below the communication threshold.  The algorithm operates
in two phases: a calendar-month band structure computed once, and a per-pixel
streaming reducer applied to every horizon tile.

**Band structure (calendar months).**  The evaluation timestamps are grouped
into calendar months in UTC.  Each month becomes one ``float32`` output band
labeled with its ``[start_utc, stop_utc)`` interval.  Month boundaries are
determined purely from the timestamps; no SPICE geometry is needed.

**Per-pixel streaming reducer.**  For each 128×128 horizon tile the reducer
walks through time, consuming two parallel streams:

1. **sunlight fraction tiles** from the shared CPU or CUDA lightmap session
   (the same 16-slice solar-disk model used by lightmaps).
2. **Earth terrain-relative elevation tiles** from the body-center
   margin session, which computes Earth elevation relative to each pixel's
   *own* interpolated terrain horizon at the Earth's azimuth.

The fraction and elevation generators both support CPU and CUDA backends via
``LightmapCpuSession`` / ``LightmapCudaSession`` and select accordingly.
The streaming reducer that combines them (``reduce_safe_haven_patch_stream``)
is implemented in plain NumPy and runs on CPU regardless of backend.  This is
acceptable for ``0.1.0rc1`` because the heavy computation — the 16-slice
solar-disk model applied per-pixel per-timestep — is handled by the upstream
session, while the reducer's per-timestep work is simple boolean masking and
max-reduction on fixed-size 128×128 patches.

At every timestep the reducer maintains per-pixel state:

- ``run_length`` — current contiguous low-sun sample count (independent of
  Earth).  A pixel is low-sun when its sunlight fraction is strictly below
  the configured threshold.
- ``outage_active`` — whether the pixel is currently in an Earth outage
  (Earth elevation relative to its own terrain horizon is strictly below
  the Earth threshold).
- ``best[month]`` — the longest qualifying run seen for each calendar month.
- ``had_outage[month]`` / ``was_above[month]`` — per-month NODATA sentinels.

A run qualifies when it overlaps a pixel-local Earth outage.  Because a run
may begin before the outage, end after it, and span multiple months, the
``best[month]`` accumulator is updated continuously at every timestep while
the pixel is in an outage AND the current run is active.  When the run ends
(sun comes back), the final run length is applied to every month the run
touched via right-censoring.

**NODATA semantics.**  After the stream completes, pixels receive ``NaN``
(NODATA) in two cases:

- Earth *never* went below the threshold during that month (always
  communicable — the safe-haven question is ill-posed).
- Earth *never* went *above* the threshold during the entire month
  (permanent Earth shadow — no safe haven exists).

Pixels where Earth crossed the threshold but no qualifying low-sun run
overlapped any outage receive zero hours with a valid mask.

**Design rationale.**  The original C# implementation computed Earth outage
intervals from the DEM center pixel's terrain horizon, using those
center-calendar intervals for every pixel.  That version also named bands by
the timestamp of the minimum Earth elevation within the center pixel's
outage.  The current Python implementation was reworked for three reasons:

1. **Per-pixel correctness.**  Outages anchored to the center pixel's
   terrain can miss, clip, or misalign outage edges at pixels whose local
   horizon differs materially.  The per-pixel approach detects each pixel's
   own outage transitions from its own horizon, removing the center-pixel
   dependence.

2. **Regular calendar bands.**  Calendar months are unambiguous, predictable,
   and decoupled from any single pixel's internal elevation curve.  Each
   band represents one month in UTC; the interval is stored in per-band
   metadata.

3. **Streaming memory bounds.**  The original C# approach allocated a full
   boolean ``[time]`` array per pixel for Earth and Sun state.  The
   streaming reducer uses fixed per-patch arrays proportional to
   ``(bands × y × x)`` and consumes one fraction tile and one elevation
   tile per timestep.  Memory does not grow with the timeline length.

### 9.4 Landed mission-duration maps

Mission-duration products use the same bounded fraction and body-center-margin
streams. Four separate public functions produce:

- longest continuous sunlight-fraction duration;
- longest continuous Sun-center local-horizon-margin duration;
- longest continuous duration satisfying both sunlight fraction and
  Earth-center local-horizon margin; and
- longest continuous duration satisfying both Sun-center and Earth-center
  local-horizon margins.

These are separate functions, not values passed to one public mode argument.
They share interval validation, signal generation, thresholding, streaming
reduction, and product storage internally. Every threshold comparison is
inclusive (`>=`). "Elevation" in this product family means body-center
elevation relative to the interpolated local terrain horizon at the body's
azimuth, not elevation above an unobstructed local horizontal plane.

Each request defines one overall half-open evaluation interval and one or more
half-open candidate-start intervals inside it. The condition sampled at time
`t[i]` applies to `[t[i], t[i + 1])`, clipped at the evaluation stop; the
evaluation stop is the final boundary when it is not itself sampled. A mission
may start at any qualifying sample in a candidate interval, including when the
underlying qualifying period began earlier, and may continue beyond the
candidate interval. A qualifying period still active at the evaluation stop is
right-censored: it receives credit through the stop, without claiming that the
real condition ends there. Helpers construct clipped calendar-month,
start-anchored week, and arbitrary fixed-duration candidate intervals.

One output band represents each candidate-start interval and stores both
interval boundaries. Results are `float32` hours or days; days are accumulated
from actual sample-to-sample durations and divided by 24 after reduction.
Silent truncation or clipping is not allowed.

Later reducers may accept a bounded total condition outage or integrate a
battery state-of-charge model. Both retain the patch-major signal stream and
GeoTIFF writer. A battery reducer may require several active candidate states
or a specialized dominance algorithm, so constant reducer state independent of
time is not assumed before that algorithm is designed and measured.

### 9.5 Elevation and visibility products

Sun-over-horizon, Earth-over-horizon, elevation, and related visibility maps
also use the shared engine. They select the required body vector, interpolate
the horizon at its azimuth, apply apparent-limb behavior if the product calls
for it, and emit the requested dtype. A time series uses timestamped bands; a
reduction uses a single band or a small fixed set of bands.

## 10. Vector and SPICE architecture

Product APIs have two input levels:

1. a high-level UTC interface accepts `times` or `start`, `stop`, and `step`
   and lazily generates Sun/Earth Moon-ME vectors with SpiceyPy; and
2. a low-level interface accepts explicit Moon-ME vectors and their UTC
   timestamps.

Explicit vectors take precedence over generation arguments. This supports
deterministic tests, external ephemeris sources, and environments where SPICE
is intentionally not installed or furnished.

SPICE kernels are not loaded at `import lunarscout` time. The first operation
that requests generated vectors ensures the configured kernels are available,
verifies managed downloads against checked-in hashes, and furnishes them.
Callers may instead furnish their own kernels or disable default management.

UTC-to-ephemeris conversion defaults to calling `utc2et` for each timestamp.
An anchored conversion may add ordinary elapsed seconds to one converted UTC
epoch only for a calculation and time domain where equivalence to per-sample
`utc2et` has been demonstrated. In particular, an anchored mission-time path
after all published leap seconds may be used when that equivalence is proven;
crossing an intervening leap second invalidates it. PSR may tolerate a coarser
error budget, but it still requires product-level evidence before selecting
the anchored optimization.

## 11. GeoTIFF product contract

Horizon-derived raster products are tiled, compressed BigTIFF files aligned to
the full primary DEM grid:

- block size is 128 by 128;
- right and bottom blocks use partial valid windows;
- interleave is `band`;
- integer and floating dtypes are supported;
- a floating-point predictor is used for floating dtypes and an integer
  predictor for integer dtypes;
- time-series bands carry individual UTC timestamps and the dataset carries an
  ordered timestamp list;
- all pixels have deterministic payload; and
- a GDAL validity mask distinguishes calculated pixels from pixels whose
  horizon was missing, invalid, or outside the DEM.

The invalid payload is a user option, defaults to zero, and must be exactly
representable by the selected dtype. Payload alone never determines validity.

Multi-band time series are limited to the TIFF band-count limit of 65,535.
They are intended for mission-scale series, such as two years at six-hour
steps, not a full 74-year Metonic series. PSR is a reduced single-band product.

## 12. Resumption, overwrite, and failure safety

Long calculations are restartable by default. A hidden staged TIFF is
accompanied by a manifest and durable per-patch completion journal. The
manifest binds the job to:

- grid, transform, and CRS;
- dtype, band count, ordered timestamps, compression, and invalid value;
- algorithm name, algorithm version, and calculation configuration; and
- horizon inventory identity.

An incompatible staged job is rejected. `start_fresh=True` explicitly discards
staged state and begins again.

For a multi-band product, the horizon patch is the recovery unit. The engine
writes or overwrites every band for that patch, flushes and synchronizes the
staged dataset, and only then advances the journal. If interruption occurs
after some bands were written but before the journal update, restart recomputes
all bands for that patch. It does not assume that the last band alone proves
completion and does not scan payload values for completion.

Missing or unreadable horizons are themselves a completed patch result: every
band receives the configured invalid payload and the mask receives zero. This
is journaled like calculated data, making restart deterministic.

The staged TIFF is published only after all expected patch keys are journaled,
metadata and masks are finalized, and the dataset closes successfully. Publish
uses an atomic replacement where the filesystem supports it. With
`overwrite=True`, the old complete output remains in place until the new
staged product is ready; failure cannot masquerade as success or destroy it.

## 13. Progress, cancellation, and errors

Every long-running operation reports structured progress with a stable stage,
completed and total work counts, percentage, message, and current path or
patch where applicable. Text progress is derived from the same events and is
flushed immediately. Library code does not print unless explicitly given a
progress stream.

Cancellation is cooperative and checked between bounded units: vector batches,
patch preparation, horizon reads, CPU/CUDA calls, writes, flushes, and publish.
An executing CUDA kernel is allowed to finish. Cancellation leaves resumable
staging state and never publishes an incomplete product.

Public failures use structured subclasses of `LunarscoutError`, stable `code`
values, and machine-readable `details`. The post-.NET taxonomy is domain based,
for example:

- input/grid/vector/time errors;
- compute capability or CUDA initialization errors;
- horizon format and horizon generation errors;
- product calculation and product storage errors; and
- cancellation.

The obsolete `Native*` exception names are not part of the Python-only public
API. No error message instructs the user to install .NET, configure a DLL, or
initialize a managed runtime.

## 14. Resource management and performance

Memory bounds are expressed in terms of fixed patch state, not region size:

```text
host memory ~= DEMs + pyramids + bounded prepared patches
              + one/few decoded horizons + bounded result batches

GPU memory  ~= resident pyramids + per-worker segment/output buffers
              + one horizon-derived session's bounded vector/result buffers
```

Horizon generation keeps one resident slope buffer through all DEM passes so
later hierarchy culling sees the accumulated horizon. Allocating and merging a
separate slope buffer per DEM is both incorrect and slower.

Downstream time batching is configurable. Defaults are conservative and may be
tuned from measured memory and throughput, but allocation never scales with the
full regional cube. Queue depth and worker count are exposed only where there
is evidence that tuning is useful.

Performance reports distinguish:

- GPU visibility from successful Numba CUDA initialization;
- initialization and JIT time from warm kernel time;
- transfer, synchronization, kernel, compression, and write time;
- single-patch latency from bounded and sustained throughput;
- host RSS from GPU allocation; and
- calculation-only work from end-to-end file-producing work.

## 15. Compilation and caches

Numba CPU and CUDA functions request on-disk compilation caching. A fresh
short-lived Python process can reuse compatible cached artifacts; Lunarscout
does not solve startup time by requiring a persistent worker. If no writable
cache location is available, execution falls back to ordinary JIT compilation
with an actionable warning or diagnostic rather than failing import.

Cache identity and invalidation are owned by Numba plus Lunarscout's algorithm
and data-cache versions. Packaging tests cover writable installed-wheel cache
placement, clean environments, Python/Numba/toolchain changes, and relevant GPU
target changes. A compiled cache does not eliminate CUDA context creation, DEM
loading, or the first real kernel execution; those costs are measured and
reported separately.

DEM pyramid and other data caches include source identity, dimensions, dtype,
algorithm version, and integrity metadata. They are never accepted solely
because an array has the expected length.

## 16. Import and optional-dependency boundaries

`import lunarscout` must succeed on a CPU-only machine and must not:

- import Numba CUDA or create a CUDA context;
- import or initialize SpiceyPy/CSPICE;
- furnish or download SPICE kernels;
- open Rasterio/GDAL datasets; or
- perform filesystem writes or network access.

Raster, SPICE, and CUDA dependencies are imported by focused modules at first
use. Capability discovery is read-only and has no unrelated side effects.
Selecting `backend="cpu"` does not touch CUDA. Supplying explicit vectors does
not import SpiceyPy or touch the SPICE kernel pool. Reading an existing horizon
on a CPU-only machine does not touch CUDA.

SpiceyPy is a core installation dependency because generated Sun/Earth vectors
are part of the supported product surface, while its runtime import and kernel
loading remain lazy. Numba provides CPU execution from the base install. CUDA
support is selected with the `cuda` extra, which installs the validated
Numba-CUDA CUDA 12 user-space runtime but not an NVIDIA driver. The base
installation does not resolve CUDA runtime packages. HDF5 is not a supported
product format or declared dependency. `pythonnet` and .NET artifacts are
absent from the base install and every optional group.

## 17. Testing and evidence

Tests are organized by what they prove:

1. **Ordinary CPU tests** cover import boundaries, core arrays, formats,
   synthetic scientific cases, CPU products, restart, masks, partial edges,
   errors, and cancellation. They run without a GPU, .NET, or SPICE network
   access.
2. **CUDA integration tests** are explicitly gated. They prove that a real
   kernel executed, compare CPU and CUDA within scientific tolerances, check
   buffer reuse and memory bounds, and exercise failure handling.
3. **File-compatibility tests** decode and validate `.bin`, `.cbin`, and
   BigTIFF products independently through Python and GDAL/Rasterio.
4. **SPICE tests** use controlled furnished kernels or immutable vector
   fixtures and separately test optional kernel management.
5. **Real-terrain and performance tests** record hardware, software versions,
   input/cache hashes, cold and warm timings, throughput, host RSS, GPU memory,
   and output hashes.

C# parity fixtures remain useful immutable migration evidence, but running C#
is not part of the shipped test suite or acceptance environment. Every retained
algorithm ultimately has a Python reference or independently specified
scientific test so correctness does not depend on an executable moonlib oracle.

## 18. Packaging and deployment

The distributed artifact is a Python wheel plus ordinary Python metadata. It
contains Python modules, Numba source, small static manifests, and no managed
assemblies. Installation and use require no .NET SDK or runtime.

CPU-only installations can read horizons and run all downstream product
families. Horizon generation and explicit CUDA acceleration additionally
require the `cuda` installation profile and a supported NVIDIA GPU and driver.
Errors identify the missing or incompatible layer without implying that the
machine lacks a GPU merely because one sandbox cannot see it.

Rasterio/GDAL and SpiceyPy/CSPICE remain native system-facing dependencies, but
their versions and installation requirements are expressed as Python package
capabilities. Wheels and clean-environment tests validate the supported Linux,
Python, Numba, CUDA-driver, GDAL, and CSPICE combinations.

## 19. Migration completion criteria

Removal of the managed implementation is complete when all of the following
are true:

- public horizon generation delegates to the validated Python/Numba pipeline;
- existing `.bin` and `.cbin` products remain readable and new files satisfy
  the documented contract;
- lightmap, PSR, elevation/visibility, safe-haven, and mission-duration
  products pass their CPU gates, with CUDA comparison where implemented;
- full-grid BigTIFF output, masks, timestamps, partial edges, restart,
  overwrite preservation, progress, and cancellation are verified;
- representative notebook and one-shot-script runs load no `clr`, `pythonnet`,
  or `moonlib` modules;
- a built wheel passes clean GPU and CPU-only installation tests;
- startup/cache behavior and end-to-end sustained performance are documented;
- public exceptions and documentation contain no managed-runtime concepts; and
- `pythonnet`, native wrapper modules, C# projects/tests, managed build steps,
  and DLL configuration are removed from the repository and package metadata.

Until all criteria are implemented, this document still defines the chosen
architecture: gaps are migration work toward the Python-only system, not a
reason to preserve a second production runtime.

## 20. Map-algebra system

### 20.1 Value model

The map-algebra subsystem introduces two core value types defined in
`raster.py` and `map_algebra/_model.py`:

**`Raster`** is the eager, already-materialized value. It keeps ordinary
2-D NumPy `values` together with a `GeoReference`, a boolean `valid` mask,
optional `units`, `name`, and `validity_provenance`. Construction validates
shape, grid dimensions, validity shape, and read-only metadata (supported
dtypes are `bool`, signed/unsigned integers, `float32`, `float64`; object,
string, datetime, and complex are rejected). `from_masked_array` and
`from_temporal_cube` provide lossless adapters from existing types.

**`RasterExpression`** is an immutable description of a calculation that has
not yet run. It contains an operation identifier, immutable normalized
parameters, operands, inferred grid/dtype/units/halo, and a versioned
semantic identifier. Users obtain expressions from `ma.source()`,
`Raster.expression()`, coordinate constructors, or registered operators.
Python arithmetic, comparison, and bitwise Boolean operators build the
expression graph.

**`TemporalRaster`** extends the model to time series: a 3-D `(time, y, x)`
array with a 1-D `datetime64` time axis, spatial `GeoReference`, and
per-pixel validity mask. An adapter (`from_temporal_cube`) creates one from
an existing `TemporalCube`. **`TemporalRasterExpression`** is the lazy
temporal counterpart, supporting layer-wise local operations with static
raster broadcasting and temporal reductions (mean, min, max, std, sum,
count) that produce composable spatial `RasterExpression` nodes.

### 20.2 Operation registry

All map-algebra operations are defined in a sealed internal registry. Each
operation spec carries an identifier, arity, category, dtype inference rule,
unit inference rule, halo, validity rule, eager kernel, and window kernel.
Registration occurs at import time from static library code only; users
cannot register arbitrary kernels.

The registry drives both execution (by selecting the right kernel for each
node) and introspection (by powering `ma.describe_operation()` and
`ma.list_operations()`). The same metadata feeds the explain/plan tools and
the docstrings, preventing silent divergence.

### 20.2.1 Numeric precision and accelerator constraint

Map-algebra numeric policy is designed for consumer-grade GPUs as the normal
accelerator environment. ``float32`` inputs and operations whose inferred
output is ``float32`` therefore remain in FP32. The implementation must not
use FP64 intermediates as a general-purpose mechanism for overflow detection,
domain validation, eager/window parity, or accelerator implementation. FP64
is used only when an input, explicit requested dtype, accumulator contract, or
documented scientific result requires it.

Consumer NVIDIA GPUs also do not provide a native general-purpose 64-bit
integer ALU path; 64-bit integer arithmetic is software-emulated and may be
slower than FP64. Consequently, CUDA map-algebra hot paths target FP32 and
Boolean/8/16/32-bit integer operations. FP64, ``int64``, and ``uint64`` remain
supported CPU correctness, interchange, identity, and storage types, but
accelerator planning must not depend on them. A future CUDA planner must reject
or route a 64-bit operation to an explicit CPU path unless a separately
benchmarked kernel establishes an acceptable contract; it must never silently
run an emulated 64-bit regional workload merely because compilation succeeds.

Integer correctness is independent of floating-point precision. Signed and
unsigned overflow checks operate in the integer domain with exact boundary
comparisons; ``int64`` and ``uint64`` values, especially those beyond
``2**53``, are never converted to ``float64`` to decide whether a result is
representable. ``overflow="promote"`` selects an exact supported integer dtype
when one exists. If no supported integer dtype can represent the result, the
operation raises a structured map-algebra error rather than silently selecting
an inexact floating dtype. These rules apply equally to eager, windowed, and
future CUDA kernels; on CUDA, a required 64-bit exact operation makes the node
ineligible for the normal GPU fast path.

Integer power follows the same contract through bounded repeated squaring and
per-multiply integer boundary checks; it does not call a floating-point power
kernel to decide overflow. Cast safety has two independent layers: NumPy's
type-level ``casting`` rule, and a value-level ``overflow`` rule. The default
value rule rejects valid values outside the destination range using exact
signed/unsigned comparisons and source-dtype floating boundaries. Explicit
wrapping is limited to integer-to-integer casts. Invalid payloads do not cause
overflow failures because their encoded value is not scientific data.

Selection operations use the same central dtype engine in eager expression
construction and execution. ``where`` and ``coalesce`` treat Python integer
branches as exact values rather than NumPy weak scalars, so a fallback such as
300 cannot wrap into ``uint8``. They copy selected values directly into the
inferred output dtype and never stage them through FP64. If signed ``int64``
and ``uint64`` domains have no common exact supported dtype, construction
raises ``map_algebra_no_exact_promotion``. Raster branches/operands must have
matching units, which are preserved by every execution mode. Their operation
semantic versions are 2 because these rules change scientific dtype, value,
unit, and identity behavior from the earlier implementation.

Nodata and invalid-fill encoding is likewise centralized, but remains separate
from scientific payload promotion. `_validate_nodata_representable` requires a
finite encoding to round-trip exactly through its destination dtype, permits
NaN only for floating output, and rejects infinities and Boolean-as-integer
fills. It normalizes integral floating metadata reported by GDAL and exact 0/1
Boolean metadata. `_valid_from_nodata` validates before payload comparison;
`_fill_invalid_exact` performs a safe dtype conversion into a new array and
then fills invalid cells. Raster construction and conversion, eager
`fill_invalid`, GeoTIFF validation, and windowed output all use these helpers.
Public storage boundaries translate `RasterValidationError` into their own
structured GeoTIFF or map-algebra storage errors while retaining stable domain
codes. No encoding check promotes FP32 work or converts integer payloads
through FP64.

The common numeric-domain helper implements ``invalid``, ``keep``, and
structured ``raise`` behavior for pointwise arithmetic and math kernels.
Policies are expression parameters, so they participate in canonical
scientific and restart identity and are replayed unchanged by eager and
windowed execution. No helper introduces FP64 or 64-bit integer arithmetic
solely to perform these checks.

Variadic layer combinations are compositional public helpers rather than a
separate kernel family. ``sum_layers`` and ``mean_layers`` expand into the
ordinary checked add/divide graph; ``min_layers`` and ``max_layers`` expand
into pairwise minimum/maximum nodes. This keeps eager, expression, windowed,
identity, unit, validity, and numeric-policy behavior on the same enforced
paths and avoids materializing a ``(layer, y, x)`` temporary stack.

Eager focal statistics share one validated ``min_valid_count`` contract.
The parameter is meaningful only with ``ignore_invalid`` and is bounded by the
number of active footprint cells. Expression construction validates and
records it, but executing general focal expressions in bounded windows remains
deferred to the large-raster plan.

Connected-region algorithms remain owned by the array-oriented
``lunarscout.regions`` module. Its public functions retain eight-neighbor
defaults and now accept explicit four-neighbor connectivity. The eager
map-algebra adapters require Boolean ``Raster`` inputs, pass canonical invalid
cells as a mask rather than interpreting payload/nodata again, delegate to the
same algorithms, and return ``Raster`` results on the unchanged grid. Labels
and sizes are ``int32``; filters and borders are Boolean. These operations are
whole-raster/global-cost and have no expression or file-backed execution claim.

### 20.3 Execution architecture

Execution has two strategies on one operation specification:

1. **Eager** (`Raster` in, `Raster` out) delegates to the per-operation
   eager kernel (typically NumPy or SciPy). Validity masks are combined
   according to the operation's declared validity rule, and numeric domains
   are applied to invalidate undefined results.

2. **File-backed / windowed** (`RasterExpression` in, evaluated by
   ``ma.write()``) topologically sorts the expression DAG, validates the graph
   (rejecting unsupported focal/global/zonal/distance/temporal nodes), infers
   one output grid and dtype, enumerates output windows (default 128 by 128),
   and writes results block by block. Consecutive local operations are
   evaluated within the same window pass without full-raster intermediate
   materialization; explicit local fusion is not implemented. Source datasets
   are opened lazily during execution and closed deterministically with
   bounded cache eviction.

``ma.compute()`` materializes an expression as a ``Raster`` (suitable for small
rasters). ``ma.write()`` evaluates in bounded windows and produces a staged,
atomically-published GeoTIFF with a GDAL validity mask and a restart
manifest.

**Recursive per-node window requests.**  Each operation node in the expression
DAG requests the window it needs from its operands.  Leaf nodes (sources,
constants, coordinates) serve data from the bounded ``SourceWindowCache``;
intermediate nodes compute on the windows returned by their operands.  A
per-window memo table prevents redundant recomputation when a sub-expression
is referenced by multiple consumers.

**Halo-aware terrain execution.**  Terrain nodes (``terrain.slope``,
``terrain.aspect``, ``terrain.hillshade``) declare a one-pixel halo.  During
windowed execution, each terrain node expands its request by one pixel on each
edge, evaluates the full terrain kernel through the same existing scientific
implementation used by eager compute, and then crops the result back to the
exact output window.  This expand-and-crop cycle occurs once per terrain node
(for example, two nested terrain nodes expand by two pixels at the leaf). The
``ExecutionPlan.maximum_halo`` field reports the largest cumulative halo
required from any source.

**Cumulative halo planning.**  The planner walks the DAG in reverse
topological order and propagates each node's halo requirement to its
operands.  For example, a terrain node with ``halo=1`` that is itself an
operand of a second terrain node requires a cumulative halo of 2 from the
source.  Resampling nodes (``alignment.resample_to``) break the halo chain
because they map into a different pixel coordinate system; their interpolation
support is accounted for by the destination-window-to-source-window mapper.

**Destination-to-source window mapping.**  For resampling, an output window is
mapped conservatively into the source pixel grid by forming the destination
window's spatial envelope, transforming its bounds with 21-point edge
densification, adding
interpolation support pixels (one for bilinear/nearest, two for cubic, three
for lanczos), and clipping to the source extent.  Source reads are therefore
bounded to the footprint actually needed for each output window.

**Exact nearest-neighbour for 64-bit integers.**  ``nearest`` resampling uses
a pure-NumPy implementation that avoids GDAL's ``float64`` intermediate
casting.  The custom path directly indexes the source array with integer
coordinates, preserving exact ``int64`` and ``uint64`` payloads above the
53-bit mantissa precision of IEEE 754 double-precision floating point.

**Deferred capabilities.**  General focal kernel window execution (halo from
arbitrary footprint sizes), local fusion across consecutive nodes,
cross-window connected-region reconciliation, and global/zonal/distance/
temporal bounded execution remain
deferred to a later milestone.

### 20.4 Temporal execution

Temporal computation follows the same pattern with an additional time axis:

- **Eager temporal compute** (`TemporalRasterExpression` via
  `compute_temporal()`) topologically evaluates nodes: constant nodes
  provide their in-memory values; source nodes read from
  `TemporalGeoTiffSeries` layer by layer with nodata-derived validity;
  broadcast nodes tile a spatial raster across time; local nodes apply
  per-layer kernels; and reducing nodes aggregate the time axis.

- **File-backed temporal** source nodes retain an open
  `TemporalGeoTiffSeries` handle so that eager compute can stream layers
  without constructing a full cube. Reductions that compose with spatial
  algebra are evaluated by first computing the temporal source, then
  applying the reduction eagerly. This ensures no temporal helper
  constructs a full file-backed cube unless the caller explicitly requests
  materialization.

### 20.5 Storage flow

Expression output through `ma.write()` uses the same durable storage
patterns as the horizon-derived product engine:

1. Validate the expression graph, output paths, output dtype, and exact invalid
   fill encoding. Encoding failure occurs before staging artifacts are changed.
2. Create a hidden staged GeoTIFF and compact checkpoint journal. The TIFF
   and journal carry the same execution identity.
3. Process output windows with bounded source reads, computational work,
   and GDAL block writes.
4. At each checkpoint boundary (default 16 windows), close the staged
   TIFF to flush data, atomically update the checkpoint journal, and
   reopen the TIFF.
5. Write each window's values and GDAL mask before advancing the completed
   row-major prefix. Validate staged dtype, CRS, transform, nodata, block
   layout, dimensions, and identity before resuming it.
6. Publish the staged TIFF and staged manifest with paired exception rollback.
   Clean up journal and staging artifacts only after both swaps succeed. A
   failed overwrite restores the prior complete output and prior manifest.

The restart manifest binds completed output to the expression scientific
identity, output dtype, invalid fill value, and grid dimensions. The checkpoint
journal identity additionally binds the complete destination grid, window
layout, checkpoint interval, validity encoding, and enforced GeoTIFF write
options. The journal
stores a compact contiguous completed-window prefix rather than an area-sized
set. A matching journal and identically tagged, structurally validated TIFF
allow resumption from the last checkpoint. Either artifact alone, or an
incompatible, malformed, stale, or truncated journal, is discarded and its
windows are recomputed.

Progress and cancellation callbacks are supported: progress reports
completed-window count, total, and current window index after each successfully
written window, including completion exactly once; cancellation is
checked before execution and between windows, checkpoints prior completed
work, raises ``OperationCancelledError``, and never publishes a partial output.

### 20.6 Dispatch rules

Mixed-mode operand dispatch follows a single rule: if any operand is a
`RasterExpression`, the operation returns `RasterExpression`; if any
operand is a `TemporalRasterExpression`, the result is
`TemporalRasterExpression`; otherwise (all `Raster` or scalar), the
result is `Raster`. A path is never implicitly converted to an expression;
use `ma.source()` explicitly.

Python operators on `Raster` detect `RasterExpression` and
`TemporalRasterExpression` operands and delegate to the expression's
reverse operator, creating the appropriate lazy node. The `map_algebra`
subpackage re-exports wrapped functions that perform the equivalent
dispatch, so `ma.sqrt(expr)` and `ma.add(expr_a, expr_b)` build
expression nodes from any combination of eager and lazy operands.

### 20.7 Modules

Further map-algebra work whose primary purpose is processing rasters too large
for memory is deferred by project decision. The implemented bounded subset
remains supported; its status and the deferred expansion are tracked in
`map-algebra-large-raster-plan.md`.

```text
src/lunarscout/
  raster.py                      # public eager Raster value
  map_algebra/
    __init__.py                  # curated public namespace and dispatch
    local.py                     # public local functions
    focal.py                     # public neighborhood functions
    zonal.py                     # public zonal functions and results
    regions.py                   # eager Boolean Raster region adapters
    reductions.py                # public global reductions
    distance.py                  # public distance functions
    coordinates.py               # public coordinate expressions
    expression.py                # compute, explain, plan for RasterExpression
    temporal.py                  # temporal_source, temporal reductions
    _model.py                    # RasterExpression node model
    _temporal_model.py           # TemporalRaster, TemporalRasterExpression
    _sources.py                  # GeoTIFF and in-memory sources
    _writer.py                   # staged GeoTIFF expression output (windowed)
    _planner.py                  # graph validation, pass enumeration, window plan
    _windows.py                  # window enumeration, bounded source cache, coordinate windows
    _windowed.py                 # per-window expression execution (local/coordinate/terrain/resample kernels)
    _spatial.py                  # terrain expression construction, resampling expression construction,
                                 #   destination-to-source window mapping, eager terrain/resample evaluators
    _eager.py                    # eager dispatch
    _kernels.py                  # spatial NumPy kernels
    _dtypes.py                   # promotion, casting, overflow
    _units.py                    # conservative unit rules
    _validity.py                 # mask combination and numeric domains
    _validation.py               # operand and grid validation
```
