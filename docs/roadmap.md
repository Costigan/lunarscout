# Lunarscout Roadmap

> **Superseded:** This roadmap records the former managed-runtime direction.
> `docs/PLAN1.md` and `docs/ARCHITECTURE.md` govern current development; C#,
> .NET, Python.NET, `moonlib`, and HDF5 product proposals below are historical
> context, not installation or release guidance.

Date: 2026-06-29

## Progress Snapshot

- **§0 Example Script Suite:** complete. Fourteen ordered examples, deterministic
  smoke tests, native lifecycle/integrity validation, representative
  performance benchmarking, manual QGIS inspection.
- **§1 Native End-to-End Validation:** complete. All three native signals
  (solar fraction, Sun/Earth horizon margin) validated for memory/file parity,
  cancellation/restart, metadata, VRT, and manual QGIS inspection.
- **§2 Representative Performance Testing:** complete. 3,800-layer local-NVMe
  baseline established for the existing timestamped-GeoTIFF series design.
- **§3 Scenario State Ownership:** direction changed. Lunarscout is now the
  primary human-facing scenario authoring library and may create, load, modify,
  and delete scenarios, including scenario database updates.
- **§4 Native API Completion:** PSR wrapper complete. All three built-in
  temporal reducers now publish native GeoTIFFs through
  `NativeReduceGeoTiffWriter` with byte-exact parity to the C# buffer stream.
  Remaining: station-over-horizon signals, per-operation capability
  diagnostics, progress/units/provenance normalization.
- **§5 Time-Series Storage Redesign:** initial prototype added. Replace
  directory-backed timestamped GeoTIFF series with a low-file-count two-file
  product optimized for National Research Platform CephFS: BigTIFF for map
  frames plus HDF5 for compressed light-curve analysis.
- **§6 Lunarscout Computation API Growth:** ongoing. The standalone library
  should expose normal in-process Python APIs, including pythonnet-backed
  operations where needed. Web-launched jobs still use `JobHandlers`.
- **§7 API Stabilization:** deferred until the Lunarscout API matures.
- **§8 Agent Adoption:** deferred until the Lunarscout API matures.

## Purpose

Lunarscout represents a change of direction for this project.  My
goal is to develop and mature a python library (which uses pythonnet
for some calculations) that is intended for a human analyst to use to
do the map algebra and other core calculations that lunar_analyst
needs.  The maturity of this library will be tested and demonstrated
via a set of python script and notebook examples.  Then we will return
to the llm-based agent and, perhaps, refine its design.

This roadmap orders the remaining work required to turn the current
`lunarscout` implementation into the primary notebook and scripting analysis
surface for Lunar Analyst. The library should mature first through human
analyst use, examples, and notebooks. After that, the LLM-based agent design
can be revisited and aligned to the stable library surface.

The current baseline includes eager NumPy raster operations, GeoTIFF I/O,
georeferencing, explicit alignment, terrain and region operations, UTC time
domains, in-memory temporal cubes, file-backed timestamped GeoTIFF series,
streaming temporal reducers, and native temporal generation with explicit
memory or file-backed storage. The timestamped GeoTIFF series remains useful
evidence and a compatibility/reference path, but it is no longer the target
large time-series storage design.

## Roadmap Principles

- Human-facing examples and API ergonomics come before additional abstraction.
- Example code uses public `lunarscout` APIs, never backend implementation
  modules.
- Native and large-data behavior is validated on real scenarios before the API
  is expanded further.
- Storage selection is explicit; the library never silently changes execution
  or storage modes.
- Lunarscout may create, load, modify, and delete scenario directories and may
  write `scenario.db` when running as a user script or notebook.
- When the web application launches or observes scripts/notebooks, it must
  detect scenario filesystem and database changes and refresh its view of state
  rather than assuming exclusive ownership.
- SQLite writes from separate processes are acceptable for the expected
  single-user workflow, provided writes are short, use appropriate timeouts,
  and tolerate serialization.
- Standalone Lunarscout calculations should look like ordinary in-process
  Python API calls. This includes pythonnet-backed calculations now that the
  Rasterio and MaxRev GDAL stacks are pinned to a compatible line.
- Web-application jobs still use `JobHandlers` and the existing worker/job
  contracts for queued, observable, cancellable execution.

## 0. Example Script Suite

Status: initial suite implemented, deterministic smoke coverage passing, and
the native examples validated against the 512 x 512 `test_scenario`, including
manual QGIS 3.44.7 inspection of the generated TIFF and VRT outputs.

Create a coherent set of executable scripts demonstrating the capabilities of
the new library. These scripts are the first priority because they validate
the public surface, provide scientist-facing recipes, and become reference
material for future agent guidance.

Initial scripts:

1. GeoTIFF reading, writing, metadata inspection, and coordinate conversion.
2. Slope, aspect, and hillshade generation.
3. Region labeling, size filtering, cleanup, and border extraction.
4. Explicit grid comparison and raster alignment.
5. In-memory temporal cubes and time-axis reductions.
6. File-backed temporal series creation, layer/time lookup, and caching.
7. Streaming reductions over a file-backed temporal series.
8. Native solar-fraction generation with explicit memory storage.
9. Native solar and Earth horizon-margin generation with file-backed storage.
10. GDAL/QGIS inspection through individual timestamp TIFFs and `series.vrt`.
11. End-to-end landing-site screening combining terrain, illumination, and
    region constraints.

Requirements:

- Scripts live under the repository-level `examples/` directory.
- Every script states its required scenario files, native-runtime needs,
  expected outputs, and approximate resource requirements.
- Scripts use `import lunarscout as ls` and public package APIs only.
- Non-native examples should run with small deterministic fixtures in CI.
- Native or large-data examples must fail with clear prerequisite messages and
  support an explicit configured scenario path.
- Examples must demonstrate UTC, nodata, alignment, storage, overwrite, and
  output-path conventions correctly.
- Selected examples should become integration tests once their API shape is
  accepted.

Completion criteria:

- [x] A new user can execute the non-native sequence in order.
- [x] At least one configured real scenario completes the native examples.
- [x] Generated GeoTIFFs and temporal VRTs open correctly with GDAL.
- [x] Temporal TIFFs and VRTs are manually validated in QGIS.
- [x] The examples expose no backend-internal imports. Scenario database
  mutation, when examples add it, should go through public Lunarscout APIs.

Implemented artifacts:

- Fourteen ordered scripts under `examples/`, including repeatable native
  lifecycle/integrity validation and representative performance benchmarking.
- Shared deterministic fixture and native prerequisite support.
- A script index with setup, native-runtime, storage, and QGIS guidance.
- Subprocess smoke tests for all nine deterministic scripts.
- Clear missing-scenario failures for all five native scripts.
- Real-scenario execution for a three-sample UTC range, producing an in-memory
  solar-fraction summary and file-backed Sun/Earth horizon-margin series.
- Independent Python/GDAL readback of both three-layer VRTs, including exact
  timestamp lookup and timestamp band descriptions.

## 1. Native End-to-End Validation

Validate the current native temporal implementation with real scenario data
before adding more native operations.

Status: automated real-scenario validation is complete for all three native
signals. Solar-fraction memory/file parity, conversion, cancellation/restart,
integrity, metadata, VRT, resource, and manual QGIS evidence are recorded.

Deliverables:

- Run solar fraction, solar horizon margin, and Earth horizon margin against a
  representative scenario.
- Compare memory and file-backed results pixel-for-pixel.
- Verify native uint8 solar-fraction conversion to float32 `[0, 1]`.
- Exercise cancellation, cleanup, overwrite preservation, and restart.
- Validate manifests, completion digests, TIFF metadata, and VRT behavior.
- Capture runtime, memory, scratch-disk, and output-size evidence.

Completion criteria:

- [x] Results agree between storage modes.
- [x] Cancellation leaves no published partial series or scratch files.
- [x] Real generated outputs are inspectable in Python, GDAL, and QGIS.

Validation evidence (2026-06-20):

- `examples/11_native_end_to_end_validation.py` ran against
  `/d/lunar_analyst_scenarios/test_scenario` for three hourly UTC samples from
  `2027-01-01T00:00:00Z` through `2027-01-01T02:00:00Z` on a 512 x 512 grid.
- The 3 x 512 x 512 memory cube and restarted file-backed series were exactly
  equal pixel-for-pixel. All solar-fraction values were float32, within
  `[0, 1]`, and bit-exact with conversion of reconstructed native uint8 values
  using the production `uint8 * (1 / 255)` operation.
- Cancellation was requested after the first native spatial tile during an
  overwrite. It raised `native_temporal_cancelled`, preserved every file in the
  prior completed series byte-for-byte, left no staging directory or native
  scratch file, and a subsequent overwrite restart completed successfully.
- The SHA-256 completion digest matched the exact manifest bytes. Independent
  GDAL reads confirmed three single-band Float32 GeoTIFFs with 512 x 512
  dimensions, no nodata value, matching CRS/geotransform, and exact pixel
  agreement. The three-band VRT opened successfully, matched each backing TIFF,
  and carried the expected UTC band descriptions.
- Measured runtimes were 1.348 s in memory, 1.070 s for initial file-backed
  generation, 0.529 s to cancel the overwrite, and 1.045 s for restart.
  Allocation and observed peak scratch were both 3,145,728 bytes. Final output
  was 375,111 bytes across six files. Process peak RSS was 3,254,360 KiB; this
  includes the loaded Python.NET/.NET/Moonlib process and is not an incremental
  cube-only measurement.
- Machine-readable evidence is stored at
  `analysis/native_validation_report.json` in the test scenario. The validation
  does not read or mutate `scenario.db` and performs no product registration.
- Manual inspection with QGIS 3.44.7-Solothurn confirmed the solar-fraction
  TIFF and three-band VRT dimensions, Float32 type, unset nodata, CRS, extent,
  band descriptions, and rendering. QGIS reported pixel size `1, -1`, which is
  the expected north-up affine convention. Its three WGS 84-to-custom-CRS
  warnings were expected because the default Earth project CRS has no valid
  transformation to the lunar CRS; validation used the native layer CRS and
  performed no reprojection.

## 2. Representative Performance Testing

Status: representative 3,800-layer local-NVMe baseline complete. Current
patch-major scratch is acceptable at this scale on local NVMe; target CephFS or
deployment storage still requires its own run before production budgeting.

Measure the known large-series use case of approximately 3,800 layers and
3.8 GB compressed.

Deliverables:

- [x] Benchmark native computation, patch-major scratch assembly, GeoTIFF writing,
  VRT creation, series validation, random-time reads, and temporal reductions.
- [x] Record peak RAM, peak scratch disk, final size, file count, and elapsed time.
- [x] Compare application-cache-cold and warm behavior. Host-wide Linux page
  cache eviction was intentionally not used.
- [x] Determine whether current patch-major disk scratch is acceptable for the
  measured local-NVMe case.
- [x] Decide whether a time-major native stream or tiled direct writer is
  currently necessary. The measurements do not justify a redesign at this
  scale; target-storage measurements remain a deployment prerequisite.

Measured baseline (2026-06-21):

- `examples/12_native_performance_benchmark.py` generated 3,800 hourly
  `sun_over_horizon_deg` layers for the 512 x 512 `test_scenario` grid from
  `2027-01-01T00:00:00Z` through `2027-06-08T07:00:00Z`.
- The managed environment exposed `/d` read-only, so the required 1.1 MB DEM
  and 17-file, 378,633,332-byte horizon set were copied to `/tmp`. `cmp`,
  recursive `diff`, sizes, file counts, and manifest/DEM SHA-256 digests
  verified the copy. Output used ext4 on local NVMe; `scenario.db` was neither
  copied nor accessed.
- Native stream and scratch assembly took approximately 15.20 s. GeoTIFF
  writing dominated at 96.20 s, and VRT finalization plus validation took
  approximately 3.95 s. Total generation was 115.74 s.
- Peak scratch matched the exact 3,984,588,800-byte float32 allocation. Final
  compressed output was 2,978,231,334 bytes across 3,803 files, so measured
  peak scratch plus published/staged data was approximately 6.96 GB. No scratch
  files remained. Observed whole-process peak RSS was 6,208,565,248 bytes from
  a 99,422,208-byte pre-bootstrap baseline.
- Full 3,800-layer validation took 3.93 s, metadata-only open 0.069 s, and VRT
  open 0.083 s. Mean/min/max reducers took 19.48/19.92/19.94 s; standard
  deviation took 22.28 s.
- Thirty-two deterministic random reads took 0.162 s with a new Lunarscout
  reader and empty full-layer cache, then 0.000244 s from that reader's warm
  cache with an identical checksum. The OS page cache was not dropped, so this
  is explicitly an application-cache comparison rather than a physical-disk
  cold-start measurement.
- The benchmark shows no immediate need for time-major native streaming or a
  direct tiled writer at this scale on local NVMe. The existing preflight's
  conservative scratch-plus-uncompressed-output free-space requirement remains
  appropriate. GeoTIFF publication is the dominant optimization target, and
  the 3,803-file layout should be remeasured on target CephFS before setting a
  production budget.
- Machine-readable evidence is committed at
  `docs/benchmarks/lunarscout_native_temporal_2026-06-21.json`.

Completion criteria:

- [x] Performance and local-NVMe resource budgets are documented.
- [x] Any required storage or native-stream redesign is supported by measured
  evidence rather than assumptions.

## 3. Scenario State Ownership

Status: direction changed. Lunarscout is the primary human-facing scenario
authoring library. Local scripts and notebooks may create, load, modify, and
delete scenarios, write and delete scenario files, and update `scenario.db`.
The web application is no longer assumed to be the sole authoritative owner of
scenario state.

Define scenario state behavior for standalone Lunarscout use and for web
application observation of externally changed scenarios.

Deliverables:

- Define `ls.Scenario` creation, open, save/refresh, delete, and mutation
  semantics.
- Define the minimal scenario database API Lunarscout needs for products,
  layers, provenance, and scenario metadata.
- Add SQLite connection hygiene for local multi-process use: `busy_timeout`,
  short transactions, idempotent writes where practical, and clear recovery
  behavior after interrupted writes.
- Define how the web application detects scenario changes made by external
  scripts or notebooks, including filesystem changes and `scenario.db` changes.
- Decide whether the web application should actively watch scenario folders,
  poll sentinel mtimes, expose an explicit refresh operation, or combine these.
- Enforce scenario-root containment for file writes and deletes in Lunarscout
  as well as in the web application.
- Define deletion, overwrite, and recovery behavior for ordinary raster
  products and larger logical products.

Completion criteria:

- Scripts and notebooks can manage scenario state without FastAPI.
- The web application reliably recognizes externally changed scenario state
  after launching or observing a script/notebook.
- Scenario database writes from Lunarscout and the web application serialize
  cleanly for the expected single-user workflow.
- Scenario-root containment and protected-file rules are enforced consistently.

## 4. Native API Completion

Status: PSR product wrapper and real-scenario raster validation are complete
with explicit native time/elevation and byte-mask semantics. Native progress is
delivered synchronously across the Python.NET boundary so no callbacks remain
queued after bridge return. Partial horizon coverage uses an internal GDAL
validity mask, while complete coverage uses the virtual all-valid mask. The
successful validation process still exposed an intermittent segmentation fault
during `pythonnet.unload()`; teardown diagnosis, station-over-horizon, and
capability normalization remain. As the first temporal-reduction publication
slice, average solar fraction now writes its final Float32 GeoTIFF directly in
C# using Moonlib's GDAL rather than transferring reduced tiles into Python
GDAL. The Earth-above-terrain cumulative-duration and combined Sun/Earth
max-contiguous reducers have since been migrated onto the same native writer.
All three built-in native temporal reducers now publish through
`NativeReduceGeoTiffWriter`; the older Python GDAL writer is fully retired for
this code path.

Complete the initial native product set after the existing temporal operations
are validated.

Deliverables:

- Add station-over-horizon temporal signals.
- [x] Add the PSR product wrapper.
- [x] Decide and document partial-horizon coverage behavior.
- Add stable per-operation capability diagnostics.
- Normalize native progress, cancellation, errors, units, and provenance.
- [x] Write average solar-fraction native reductions directly as atomic,
  compressed 128 x 128 tiled GeoTIFFs in C#.
- [x] Migrate the Earth-above-terrain cumulative-duration reducer onto the same
  native writer.
- [x] Migrate the combined Sun/Earth max-contiguous-duration reducer onto the
  same native writer, retiring the Python GDAL writer for this code path.

Completion criteria:

- Native operations have explicit input, output, unit, storage, and capability
  contracts.
- Production Python calls remain behind `MoonlibBridge`.

Average solar-fraction direct-writer evidence (2026-06-22):

- The native writer uses 128 x 128 tiles matching horizon-tile geometry,
  `DEFLATE` compression with floating-point predictor 3, Float32 bands, and
  explicit `-9999` nodata for regions without horizon coverage.
- Publication uses a same-directory temporary TIFF and atomic rename. Automated
  native tests verify missing-tile nodata, CRS/geotransform propagation,
  compression/block metadata, corrupt-input scratch cleanup, and preservation
  of an existing output. Python tests verify bridge cancellation/disposal and
  that this path does not initialize Python `osgeo.gdal`.
- A three-sample real-scenario run on a `/tmp` copy of the 512 x 512
  `test_scenario` wrote 16 horizon-aligned tiles in 1.83 s, produced a 160,037
  byte GeoTIFF, and reached 1,732,544 KiB whole-process peak RSS. The copy was
  required because the managed environment exposes `/d` read-only;
  `scenario.db` was not copied or accessed.
- Independent readback found exact pixel-for-pixel equality with the existing
  C# NativeReduce buffer stream (`0` mismatches), matching CRS/geotransform,
  values in `[0, 1]`, 128 x 128 blocks, DEFLATE compression, and predictor 3.
- The user manually inspected the C#-written output
  `analysis/native_mean_sun_fraction_csharp.tif` and confirmed it renders
  correctly. The older `analysis/native_mean_sun_fraction.tif` is not a
  bit-exact oracle: it was produced through a different temporal-mean route and
  differs only by float rounding (maximum absolute difference approximately
  `1.1920929e-07`).

Earth-above-terrain cumulative-duration migration evidence (2026-06-22):

- `generate_earth_above_terrain_duration_raster` now sets
  `write_output_in_native=True` and publishes through the same
  `NativeReduceGeoTiffWriter` as average solar fraction. The writer is
  signal-agnostic, so no writer changes were required; only the handler routing
  changed.
- A new deterministic .NET test writes a cumulative-duration GeoTIFF using a
  sun-fraction predicate with synthetic Sun vectors (no SPICE dependency). It
  confirms covered pixels accumulate exactly the expected duration (a value
  outside `[0, 1]`, here `3.0` hours over three hourly samples), missing-tile
  regions remain `-9999` nodata, and the output is Float32 / 128 x 128 tiled /
  DEFLATE with the expected nodata, followed by clean scratch removal. The
  Earth-margin reduction path itself is unchanged and continues to share the
  already-validated reduction routine used by the buffer-stream reference path.
- New Python handler coverage asserts the Earth handler routes through the
  native writer (no Python `osgeo.gdal` dataset created), forwards
  `use_spice_earth_vectors`, carries the Earth-center-margin predicate, and
  registers the artifact under
  `generate_earth_above_terrain_duration_raster`.
- Test counts after this migration: .NET `140` passed; focused Python `22`
  passed; lunarscout + focused worker `214` passed / `1` skipped; complete
  backend worker suite `498` passed / `1` skipped.
- Pending validation gate: a real-scenario `/tmp`-copy parity run comparing the
  C# direct-writer Earth-duration output against the existing C# NativeReduce
  buffer stream (with manual visual inspection) is still recommended before
  this reducer is treated as fully demonstrated, matching the bar applied to
  average solar fraction.

Combined Sun/Earth max-contiguous migration evidence (2026-06-22):

- `generate_combined_sun_earth_max_contiguous_duration_raster` now sets
  `write_output_in_native=True`. The writer and reduction routine are unchanged;
  only handler routing changed. All three built-in native temporal reducers now
  publish through `NativeReduceGeoTiffWriter`.
- The existing Python handler test for this reducer was rewritten to use the
  native-writer FakeClient pattern, asserting native routing, Earth-vector
  forwarding, the combined predicate structure, artifact registration, and that
  no Python `osgeo.gdal` dataset is created.
- No new .NET test was added: the combined reducer requires Earth SPICE vectors
  and the .NET test environment does not load SPICE. The writer code is
  signal-agnostic and was already validated by the average-sun and
  cumulative-duration .NET parity tests.
- Test counts after this migration: focused Python `22` passed; lunarscout +
  focused worker `214` passed / `1` skipped; complete backend worker suite
  `498` passed / `1` skipped.
- Pending validation gate: a real-scenario `/tmp`-copy parity run with Earth
  SPICE vectors comparing the C# direct-writer combined-reducer output against
  the existing C# NativeReduce buffer stream, with manual visual inspection,
  is recommended to complete the demonstration, matching the bar applied to the
  two preceding reducers.

## 5. Time-Series Storage Redesign

Status: initial synthetic prototype added in
`examples/14_timeseries_two_file_prototype.py`. The existing file-backed
timestamped GeoTIFF series is no longer the preferred large time-series target.
The target deployment includes the National Research Platform's CephFS
filesystem, which does not handle directories containing many files well. The
desired direction is therefore a low-file-count storage product that supports
both fast spatial frame access and fast per-pixel or small-neighborhood
temporal access.

Prototype a storage format for large lunar lighting cubes such as
`time=5000`, `y=5000`, `x=5000`, `uint8`, approximately 125 GB uncompressed on
local NVMe/SSD or CephFS.

Leading candidate: a two-file representation containing two copies of the same
data optimized for different access patterns:

- `shadow_maps.tif`: BigTIFF optimized for spatial frame reads, with one band
  per time step, 128 x 128 tiled blocks aligned to horizon patches, and
  lossless compression appropriate to the pixel datatype.
- `light_curves.h5`: HDF5 optimized for temporal reads, with dataset shape
  `(y, x, time)` and chunks that span the full time axis over a small spatial
  neighborhood, initially `(16, 16, time_count)`.

This design intentionally does not require the light-curve representation to be
a GeoTIFF. It should be NumPy-oriented and optimized for analysis rather than
GIS interoperability.

Generation model:

- Native or CPU workers generate one 128 x 128 spatial patch at a time.
- CPU paths may compute per-pixel `1 x T` arrays across many cores.
- GPU paths are expected to produce one `128 x 128 x T` array per patch.
- In-process Lunarscout runs should not load a native HDF5 stack through C#.
  Python allocates reusable NumPy buffers shaped `(128, 128, time_count)` in
  C order and passes their writable memory to C# through pythonnet. C# fills the
  caller-owned buffers and returns a completion acknowledgement; Python owns
  buffer lifetime, reuse, and HDF5 writes.
- For `uint8` lighting products, the canonical in-process memory offset for
  local patch pixel `(y, x)` and time index `t` is
  `((y * 128 + x) * time_count) + t`. The CUDA path may therefore write each
  pixel's `T`-byte light curve contiguously while still producing a standard
  C-contiguous NumPy view shaped `(128, 128, T)`.
- Python writes each completed patch buffer to
  `light_curves[y0:y0+128, x0:x0+128, :]`. The HDF5 dataset may still use
  smaller temporal-analysis chunks such as `(16, 16, time_count)`; HDF5 will
  split the 128 x 128 patch write across the 64 destination chunks.
- Separate-process native workers may use C# native HDF5 directly, because that
  mode isolates the C# HDF5 library from Python's loaded `h5py` stack. This
  path is intended for batch generation where the worker writes scenario-owned
  HDF5 products directly and reports only progress, status, and result
  metadata.
- `shadow_maps.tif` writes each patch as a 128 x 128 tile for each time band.
- `light_curves.h5` writes each patch into `dataset[y0:y1, x0:x1, :]`.

Canonical compression policy:

- BigTIFF map file:
  - `uint8`, `int8`, and Boolean-like masks: `ZSTD` if available, otherwise
    `DEFLATE`, with no predictor.
  - multi-byte integer types: `ZSTD` if available, otherwise `DEFLATE`, with
    horizontal predictor 2 where supported.
  - floating-point types: `ZSTD` if available, otherwise `DEFLATE`, with
    floating-point predictor 3 where supported.
- HDF5 light-curve file:
  - `uint8`, `int8`, and Boolean-like masks: Blosc/LZ4, compression level 5,
    no shuffle.
  - `uint16`, `int16`, `uint32`, `int32`, `float32`, and `float64`:
    Blosc/Zstd, compression level 3, byte shuffle enabled.
  - Avoid `float64` unless required by the scientific product.

The in-process path should prefer Python `h5py` for HDF5 creation and writes.
This avoids loading native HDF5 through C# in the same process. The HDF5 Blosc
filters imply an `hdf5plugin` dependency unless benchmark results justify
falling back to built-in gzip or LZF. If plugin availability is an operational
concern, benchmark a built-in compression fallback before standardizing the
format. The separate-process C# writer should start with built-in HDF5 deflate
unless and until a C#-side plugin strategy is proven compatible on NRP.

Directory-backed Zarr is no longer the leading candidate for NRP/CephFS because
ordinary chunk-per-file layouts can create too many files. Zarr remains worth a
small comparison only if sharding or another container-backed store keeps file
count low.

Deliverables:

- [x] Add a small experimental writer that converts generated patch arrays into
  the two-file layout.
- Extend the experimental writer to convert sequential 2D GeoTIFFs or native
  lightmap streams into the two-file layout.
- Store CRS, affine transform, time coordinates, signal name, units, nodata or
  validity semantics, and provenance metadata in both files or in a shared
  sidecar manifest.
- [x] Implement prototype readers for frame access, point light curves, and
  small-neighborhood light-curve queries.
- Add overview access after deciding where spatial overviews should live.
- Benchmark access patterns on local NVMe and NRP/CephFS:
  - one full-resolution frame,
  - one overview frame,
  - one point through all time,
  - one 10 x 10 neighborhood through all time,
  - threshold/mask interval queries over a point or small area.
- Measure total storage, write time, read amplification, compression ratio,
  CPU cost, file-count behavior, and concurrent/staged generation behavior.
- Compare against the existing timestamped GeoTIFF series for the same data.
- Compare against a single BigTIFF-only baseline to quantify the benefit of the
  separate HDF5 light-curve file.
- [x] Add a local synthetic benchmark for HDF5 light-curve chunks
  `(8, 8, time_count)`, `(16, 16, time_count)`, and
  `(32, 32, time_count)`.
- Run the HDF5 chunk benchmark on representative local NVMe and NRP/CephFS
  datasets before standardizing.
- Decide whether HDF5 with `hdf5plugin` is operationally acceptable on target
  systems, or whether a built-in-compression fallback must be the default.
- Prototype the in-process buffer-fill contract:
  - Python allocates a reusable pool of `(128, 128, time_count)` C-order NumPy
    buffers with datatype-specific element size.
  - C# fills a borrowed buffer using the canonical `((y * 128 + x) * T) + t`
    address mapping and never owns or frees the memory.
  - C# returns buffer id, patch coordinates, dtype, shape, status, and timing.
  - Python writes completed buffers to `light_curves.h5` and returns them to the
    reusable pool.
- Prototype the separate-process C# native-HDF5 writer using the same compute
  kernel, isolated from Python, with outputs restricted to scenario-owned
  product paths.
- Decide whether an optional spatial overview pyramid belongs in
  `shadow_maps.tif`, a separate overview TIFF, or the web-map derivative cache.

Completion criteria:

- The selected format supports interactive frame access and fast light-curve
  analysis on representative local NVMe and NRP/CephFS storage.
- Storage footprint remains acceptable for laptop-scale scenarios.
- Scenario registration treats the cube as one logical product with two primary
  files, not thousands of independent files.
- The writer has a safe staged-publication story for interrupted generation.
- The format has a clear migration or compatibility story relative to the
  existing timestamped GeoTIFF series examples.

## 6. Lunarscout Computation API Growth

Status: ongoing. Standalone Lunarscout should expose normal-looking Python APIs
that run calculations in process. The library may use pythonnet for native
calculations. The web application continues to route jobs through
`JobHandlers`.

Grow the stable local vocabulary and implementation surface used by human
analysts.

Deliverables:

- Add ordinary Lunarscout APIs for the core calculations Lunar Analyst needs:
  terrain, map algebra, lighting, horizon-derived products, PSR, temporal
  reductions, and site-screening workflows.
- Keep in-process APIs ergonomic for scripts and notebooks while still exposing
  progress, cancellation, storage selection, overwrite behavior, and provenance
  where operations are long-running.
- Use pythonnet-backed Moonlib operations directly from Lunarscout where that
  is the natural local API.
- Keep web-launched execution on `JobHandlers` so queued/cancellable web jobs
  remain observable and compatible with the existing backend.
- Avoid duplicating handler contracts only to serve the web app; prefer shared
  implementation helpers beneath both Lunarscout and `JobHandlers` when
  practical.

Completion criteria:

- Human analysts can perform the core Lunar Analyst calculations from
  Lunarscout scripts and notebooks without importing backend internals.
- The same scientific concepts remain recognizable in local Lunarscout APIs and
  web-app jobs.
- Long-running local operations provide enough progress/cancellation/reporting
  for practical notebook use.

## 7. API Stabilization

Status: deferred. There is no near-term release freeze target. Continue adding
to and refining the API while examples and notebooks expose what feels stable
or awkward.

Deliverables:

- Continue refining operation names, result objects, exception classes, units,
  nodata rules, and storage/result contracts.
- Use examples and notebooks as the practical test of API clarity.
- Defer formal v0.1 operation selection, compatibility rules, and deprecation
  policy until the library has more real use.
- Track unstable or experimental APIs explicitly when they appear in examples.

Completion criteria:

- The API has enough real notebook/script use to justify freezing a first
  supported subset.
- Public examples and tests identify the eventual supported surface.

## 8. Agent Adoption

Status: deferred until the Lunarscout API is more mature. The LLM-agent surface
will be reconsidered after the human-facing library has stabilized through
examples and notebook workflows.

Deliverables:

- Reconsider whether agents should author pure Lunarscout scripts, call web
  tools, submit managed jobs, or use a mixed strategy.
- Update RAG documents and startup guidance after the library surface matures.
- Add evaluation cases for the stable Lunarscout API and scenario-state model.
- Prefer example-backed recipes over backend-internal implementation guidance.

Completion criteria:

- Agents consistently produce public `lunarscout` code.
- Agent-authored workflows obey CRS, scenario-state, storage, cancellation,
  and native-boundary rules.

## Work Remaining Outside This Roadmap

The roadmap does not authorize unrelated UI redesign, database migrations,
deployment changes, or broad native algorithm rewrites. Those require their
own scoped plans, approval, tests, and rollback strategy.
