# Changelog

## Versioning policy

Lunarscout uses Semantic Versioning. Before 1.0, public APIs are provisional and breaking changes may occur in minor releases. Intentional breaking changes must be recorded here. Patch releases should not intentionally break documented behavior. The 1.0.0 milestone is reserved for a documented stable API surface and standalone native runtime story.

## Unreleased

- Added the private reference/storage slice for time-series lightmaps. It ports
  the 16-slice C# `BuilderSunFraction` solar-disk calculation, encodes visible
  fraction as truncating `uint8(255 * fraction)`, and processes horizons
  patch-first while lazily yielding one 128 by 128 tile per time. The staged
  BigTIFF writer writes each yielded tile directly to its timestamped band, so
  neither a patch time cube nor a regional time cube is retained. Initial tests
  cover full, half, and zero illumination, timestamp metadata, band interleave,
  missing-patch invalid payloads, masks, and partial output edges. Numba CUDA
  time batching and a C# numerical oracle remain open.
- Added the first private Phase 6B downstream-product vertical slice. Python
  now reads complete `.bin`/`.cbin` horizon tiles, accepts explicit timestamped
  Moon-ME vectors or lazily generates geometric SpiceyPy vectors, reproduces
  the five-viewpoint PSR reduction, and computes PSR with CPU-reference and
  reusable-buffer Numba CUDA paths. A dtype-generic staged BigTIFF store writes
  128 by 128 band-interleaved tiles with UTC band metadata, configurable
  invalid payloads, validity masks, durable per-patch restart journals, and
  partial-edge support. Deterministic Python CPU and Numba CUDA outputs match
  the actual C#/ILGPU PSR kernel byte-for-byte, including compressed-horizon
  quantization. A fresh-process compressed-horizon-to-GeoTIFF run loads no
  Python.NET, CLR, or moonlib modules. Real SPICE evidence now covers 108,113
  six-hour samples from 1970 through the start of 2044. Exact per-timestamp
  `utc2et` conversion remains the default, including for future mission
  periods after all published leap seconds. An explicit anchored-linear mode
  exactly reproduces C#, but is selected only where equivalence to `utc2et` is
  demonstrated for the intended calculation. That equivalence was established
  for the retained 16-patch real-terrain PSR product: both modes produced
  byte-identical results at about 1.19 patches/second. The downstream
  scheduler is still serial, and lightmap, safe-haven, and mission-duration
  products are not yet implemented.
- Expanded the Python/Numba replacement evaluation to require a shared bounded
  downstream product pipeline for time-series lightmaps, optimized Metonic PSR,
  safe-haven maps, landed mission-duration maps, and dtype-generic
  horizon/vector reductions. The new gate requires a patch-major pipeline that
  loads one horizon tile, computes a 128 by 128 tile for each requested time,
  and writes it to that time's timestamped band in a tiled compressed BigTIFF.
  It also requires high-level SpiceyPy vector generation with explicit-vector
  override, full-DEM validity masks with configurable deterministic invalid
  payloads, durable per-patch restart journals, file staging, C#-oracle parity
  where available, and fresh-process execution without Python.NET or moonlib
  before downstream C# product code can be retired.
- Added the first private Phase 6 Numba horizon production pipeline: row-major
  full and partial patch enumeration, structurally validated skip/resume,
  bounded CPU preparation and CUDA work queues, cancellation boundaries,
  immediately flushed progress, fixed-contract partial-edge padding, and
  staged atomic `.bin`/`.cbin` writes. Python compressed files are readable by
  moonlib and existing Scenario readers; a 23,592,960-value real tile has at
  most `0.0007639` degrees of expected compression quantization error. In a
  matched warm four-patch, four-DEM compressed run, the serial pipeline reaches
  `0.1385` patches/second and the one-item-ahead pipeline reaches `0.1673`
  patches/second. A sustained 16-patch run with a bounded writer queue and
  reusable device buffers reaches `0.1793` patches/second, 63.8 percent of the
  matched C# throughput. It uses `4,458` MiB peak GPU memory and `9.02` GB peak
  host RSS. Preparation remains fully hidden after initial pipeline fill, so
  neighboring segment-cache reuse is intentionally deferred. Two- and
  four-stream runs produce byte-identical files but no material throughput gain,
  so the selected default remains one stream. Cross-process Numba disk caching
  reduces combined first CPU/CUDA-call time by `6.54` seconds with a `2.33` MB
  cache and falls back safely when no writable cache locator exists. The full
  failure matrix remains production-pipeline work; the implementation is still
  private.
- Added diagnostic Numba CUDA mechanics on a real GPU, including lazy device
  selection, launch/copy synchronization, pixel/azimuth indexing, C#-matching
  arithmetic helpers, fixed-step and adaptive level-0 traversal, exact
  factor-four max pyramids, and hierarchical traversal with C#/CPU/CUDA traces.
  The prototype pins NVIDIA's external CUDA target and a driver-compatible CUDA
  12.9 toolchain. It records narrow terrain skipped by the primary-DEM
  adaptive step floor. An inherited hierarchy defect at bilinear cell
  boundaries is corrected in both C# and Numba with four-cell culling bounds
  and boundary-capped level-0 steps. Production-shaped device subpatch
  interpolation, full and partial patches, multi-resolution DEM accumulation,
  and final degree buffers now match the selected C# fixtures. A
  hierarchy-enabled 16 by 16 LOLA patch differs by at most `5.9605e-8`
  degrees across 368,640 values. A bounded real two-DEM stack differs by at
  most `4.0412e-5` degrees, about 124 times below the adopted `0.005` degree
  angular acceptance limit; a uniform solar-disk model bounds the corresponding
  sunlight-fraction difference at `1.0291e-4`. The fixture also exposed a
  prototype orchestration bug, now corrected, where later passes did not carry
  prior horizon slopes into hierarchy culling as production C# does.
  Phase 5 preserves ILGPU's pixel-fast warp organization in a 256-thread Numba
  launch, removes local interpolation storage and most unintended float64 PTX,
  retains immutable pyramids on the GPU, and overlaps CPU segment preparation
  with CUDA execution. On a matched four-patch RTX 5090 Laptop benchmark, warm
  CUDA latency is `5.146` seconds, pipelined throughput is `0.1635` patches per
  second, peak GPU memory is `5,558` MiB, and peak host memory is `8.95` GB.
  These are respectively 1.196 times C# bounded wall time per patch, 70.3% of
  C# throughput, 1.141 times C# GPU memory, and 63.4% of C# host memory, passing
  all provisional Phase 5 gates. File output and product integration remain
  unimplemented.
- Ported the experimental horizon host-side geometry to Python/NumPy and an
  optional Numba CPU path, with C# oracle parity for sampling, polynomial ray
  segments, multi-DEM continuity, subpatch halos, deterministic cache reuse,
  real-terrain fitted paths, and bounded performance/memory evidence. This
  completed preprocessing Phase 3 before the diagnostic CUDA work began.
- Defined the private Python/NumPy horizon host/device data contract, including
  checked precision conversions, dense segment tensors, flattened pyramid
  storage, kernel/configuration validation, slope-buffer semantics, verified
  oracle loading, and Phase 1 artifact round-trip tests. CUDA and horizon
  algorithm implementation remain deferred to later evaluation phases.
- Established Phase 0 and Phase 1 evidence for evaluating a full Python/Numba
  horizon-generator port: reproducible C# production baselines and warm
  multi-patch benchmarks, GPU-memory sampling, independent reference-ray
  oracles, synthetic and real-terrain fixture manifests, immutable NPZ test
  artifacts, per-DEM and final horizon buffers, and selected CUDA hierarchy
  traversal traces. No horizon algorithm has been ported yet.
- Created standalone Lunarscout repository skeleton from Lunar Analyst packages/lunarscout.
- Replaced the internal architecture guide with a draft user guide covering purpose, installation, usage, maturity, architecture, examples, and roadmap stubs.
- Added a Lunarscout-specific `requirements.in` and removed inherited Lunar Analyst application dependencies.
- Updated package metadata so core installs include raster/geospatial, native bridge, and HDF5 dependencies.
- Verified the new local virtual environment with `pytest -q`: 195 passed, 1 skipped.
- Added map product catalog support, including product/region models, catalog loading, ordered text search, scenario-safe naming, download-directory resolution, file download helper, public exports, and tests.
- Added lunar map product support assets and utility scripts, including a south-pole overview GeoTIFF, a product catalog JSON file, catalog maintenance/download scripts, and Git LFS tracking for `data/product_overview.tif`.
- Added `ls.GenerateHorizons(...)` and `ls.native.GenerateHorizons(...)` as Python.NET wrappers around `QuadTreeHorizonGenerator.GenerateHorizonsForPatches`, with skip-existing patch filtering, compression selection, progress callbacks, cancellation checks, and tests.
- Added `scripts/run_generate_horizons.py` as an editable local runner for manually validating native horizon generation.
- Added `AGENTS.md` with project-specific guidance for future coding agents.
- Updated the native MaxRev GDAL package-version test to locate the extracted Lunarscout repository layout and scan `native/` projects.
- Added SPICE-backed lunar local-frame APIs for `LonLat`, inclusive datetime iteration, Sun/Earth NED vectors, azimuth/elevation histories, pandas DataFrames, and matplotlib elevation plots.
- Added `ls.spice` kernel helpers for furnishing, lazy default loading, reload/unload/clear state, NAIF kernel download/cache, generated meta-kernels, and SHA-256 verification from the checked-in default kernel manifest.
- Added default SPICE kernel manifests under `data/spice/` and package data, plus implementation tracking in `docs/spice-local-frame-api-plan.md`.
- Updated the user guide with SPICE kernel setup, local NED frame conventions, vector/angle APIs, DataFrame helpers, and plotting examples.
- Added `scripts/get_link_tree.py` for recursive same-site HTML link discovery.
- Updated the lunar map product GUI to display overview-map CRS and longitude/latitude mouse coordinates and copy either coordinate pair to the clipboard with keyboard shortcuts.
- Changed the canonical Scenario horizon directory from `lighting/horizons` to root-level `horizons`, with matching tests, examples, README, and guide updates.
- Added Scenario canonical path helpers for `root_path()`, `hillshade_path()`, `slope_path()`, `aspect_path()`, and `roughness_path()`.
- Added native GDAL-backed terrain product generation through `Scenario.create_hillshade()`, `create_slope()`, `create_aspect()`, and `create_roughness()`, backed by a new `moonlib.TerrainProducts` static helper.
- Added `Scenario.generate_horizons()` as a scenario-aware wrapper around native horizon generation, including `dem_paths` and `surrounding_dems` handling with scenario-relative DEM resolution.
- Added Python horizon tile access helpers on `Scenario`, including patch coordinate conversion, horizon file lookup, `.bin`/`.cbin` single-pixel horizon reads, one-file handle caching, and cache close support.
- Added Scenario longitude/latitude DEM pixel lookup and horizon plotting helpers, including centered azimuth windows and empty azimuth/elevation axes.
- Added Scenario Sun/Earth horizon plot overlays for center points, apparent limb markers, center paths, and upper/lower limb paths; Sun defaults to gold and Earth defaults to blue.
- Suppressed the noisy pyproj WKT-to-PROJ.4 information-loss warning during Lunarscout GeoTIFF metadata reads.
- Expanded `docs/USER_GUIDE.md` with Scenario path, native terrain, horizon generation/access, horizon plotting, and Sun/Earth overlay examples.
- Updated SPICE body geometry helpers and plotting helpers to accept `TimeRange` values returned by `ls.times(...)` directly.
- Changed body path limb rendering from upper/lower dashed lines to a translucent filled limb band, while preserving explicit Matplotlib style overrides.
- Added `ls.body_azimuth_elevation_over_horizon(...)` and `Scenario.body_azimuth_elevation_over_horizon(...)` for azimuth plus elevation over an interpolated 1440-sample horizon.
- Added `horizon=` support to `ls.plot_body_elevations(...)` and added `Scenario.plot_body_elevations(...)` with `over_horizon=True` to fetch scenario horizons automatically.
- Expanded `docs/USER_GUIDE.md` with a succinct function overview table covering root functions, Scenario methods, and file-backed temporal object methods.
