# Numba Horizon Phase 0 Inventory

**Inventory date:** 2026-07-16

**C# baseline commit:** `f3b21b5a7d510162783c8e6a1aa01ca2edc61277`

**Prototype branch:** `spike/numba-horizon`

**Primary implementation:**
`native/moonlib/horizon/QuadTreeHorizonGenerator.cs`

This inventory freezes the implementation boundary that the Numba prototype
must evaluate. It describes current behavior; it does not endorse every current
choice as scientifically correct or as the desired Python design.

## 1. Baseline Status

- A forced rebuild of `native/moonlib/moonlib.csproj` succeeds with zero warnings
  and zero errors.
- The native `Fast` test tier passes 122 tests and skips one HDF5 compatibility
  test explicitly.
- The separate `GpuBaseline` test asserts that the generator selected an ILGPU
  CUDA accelerator. On this host it passes alongside an `nvidia-smi` probe for
  the RTX 5090 Laptop GPU. The `Fast` tier remains CPU-compatible and does not
  itself require CUDA.
- The machine-readable evidence is in
  `docs/numba-horizon-phase-0-baseline.json`.
- The fresh real-terrain production capture is in
  `docs/numba-horizon-phase-0-production-baseline.json`.
- The bounded cold/warm four-patch production benchmark is in
  `docs/numba-horizon-phase-0-multi-patch-benchmark.json`.
- The capture command is:

```bash
python3 scripts/numba_horizon/capture_phase0_baseline.py \
  --baseline-commit f3b21b5a7d510162783c8e6a1aa01ca2edc61277 \
  --run-native-checks \
  --run-gpu-probe
```

The production capture tool is reproduced with an empty work directory:

```bash
LUNARSCOUT_BASELINE_COMMIT=f3b21b5a7d510162783c8e6a1aa01ca2edc61277 \
dotnet run --project scripts/numba_horizon/CSharpBaselineCapture.csproj -- \
  /tmp/lunarscout-numba-horizon-baseline 0 0 0 \
  /d/lunar_analyst_scenarios/test_scenario/dem.tif \
  /d/datasets/viper_v71_2024_medium/other/dem.tif \
  /d/viper/maps/gsfc/site_20v2/Site20v2_final_adj_5mpp_surf.tif \
  /d/viper/maps/lola/LDEM_80S_20M-2017-06-15-processed.tif
```

It copies all DEMs into the work directory, forcing fresh pyramid generation.
Two independent runs produced identical pyramid hashes and the same compressed
horizon SHA-256:
`d8b6cff735fb7a244a80993aefbef7080e7d8007c0bcca96ff2c098a4dd9b666`.
The captured one-patch generation scope was 10.56 seconds and peak host working
set was 10,060,210,176 bytes. This is a single-stream latency datum, not a
sustained-throughput benchmark.

The separate multi-patch benchmark uses the first four row-major patches,
hierarchy enabled, four GPU workers, queue capacity six, four DEM passes, and
compressed output. It runs twice in the same process and generator. The cold
run starts with copied DEMs and no pyramid files; the warm run uses the pyramid
files and reusable generator resources retained from the cold run. Reproduce it
with:

```bash
python3 scripts/numba_horizon/run_phase0_production_benchmark.py \
  --baseline-commit f3b21b5a7d510162783c8e6a1aa01ca2edc61277 \
  --patch-count 4 --gpu-concurrency 4 --segment-queue-size 6 \
  --output docs/numba-horizon-phase-0-multi-patch-benchmark.json \
  /tmp/lunarscout-numba-horizon-multipatch \
  /d/lunar_analyst_scenarios/test_scenario/dem.tif \
  /d/datasets/viper_v71_2024_medium/other/dem.tif \
  /d/viper/maps/gsfc/site_20v2/Site20v2_final_adj_5mpp_surf.tif \
  /d/viper/maps/lola/LDEM_80S_20M-2017-06-15-processed.tif
```

The retained report selected the `NVIDIA GeForce RTX 5090 Laptop GPU` CUDA
accelerator. Cold elapsed time was 14.7443 seconds (0.2713 patches/second), and
warm elapsed time was 14.7224 seconds (0.2717 patches/second). The cold and warm
combined output hash was
`64c3fb6a61da4db5da85e7e24d7abc53489f591f49ddf0427ef2ea1b9846c6bb`,
and every individual patch hash matched between runs. Peak sampled per-process
GPU memory was 4,556 MiB in both generation scopes. Peak sampled host working
set was 11,130,077,184 bytes cold and 14,923,546,624 bytes warm; the higher warm
peak is retained as observed rather than being hidden by the cold/warm label.

GPU memory is measured by polling `nvidia-smi` for the benchmark process at a
50 ms interval. It is distinct from device-wide memory use and may miss a
shorter-lived allocation peak. Per-stage timings are retained for every patch
and as run aggregates. Because producer and GPU stages overlap, stage totals do
not sum to wall time; `kernel_launch_seconds` is asynchronous enqueue time and
`stream_sync_seconds` includes waiting for CUDA work.

## 2. Production Entry Path

The supported Python entry point is
`src/lunarscout/native_horizon.py::GenerateHorizons`.

The production path is:

1. Python validates callbacks, DEM paths, output-directory shape, and initial
   cancellation.
2. Python lazily bootstraps Python.NET and moonlib.
3. `MoonlibBridge.EnsureGdalInitialized` initializes the moonlib GDAL runtime.
4. Each input path becomes a C# `ElevationMap`. The first DEM is primary and
   remaining DEMs are progressively surrounding coverage.
5. `QuadTreeHorizonGenerator.GeneratePatchList` creates row-major 128 by 128
   patches. Both primary DEM dimensions must be exact multiples of 128.
6. `RemoveCompletedPatches` optionally removes patches already represented by a
   horizon tile for the selected observer elevation.
7. Python constructs `QuadTreeHorizonGenerator(disable_hierarchy)`. The Python
   default is `False`, so the production default has hierarchy enabled.
8. Python calls `GenerateHorizonsForPatches`, waits for the returned .NET task,
   translates errors, and disposes the generator.

The following C# entry paths exist but are not the current public Python batch
path:

- `GenerateHorizonsInternal` and `LaunchRayCasting` use older or diagnostic
  generation paths.
- `GenerateHorizonsWithSubpatches` is a direct in-memory subpatch entry point.
- `QuadTreeRayCastKernel` is not the kernel launched by the production Python
  batch path.
- `ReferenceHorizonGenerator`, `ReferenceRayEmulator`, and related emulators are
  correctness or diagnostic oracles, not production engines.

The initial Numba prototype must target the batch path through
`GenerateHorizonsForPatches` and `QuadTreeSubpatchRayCastKernel`.

## 3. Generator Initialization

The constructor performs eager native work:

1. Creates an ILGPU context with algorithms and kernel debug symbols.
2. Selects CUDA first, NVIDIA OpenCL second, any OpenCL third, and an ILGPU CPU
   accelerator last.
3. Creates the selected accelerator.
4. Creates a reusable GPU buffer pool.
5. Compiles `QuadTreeSubpatchRayCastKernel`.
6. Creates a pool of accelerator streams.

Current defaults:

| Setting | Value | Status |
| --- | ---: | --- |
| Concurrent GPU workers/streams | 4 | Production default |
| Segment queue capacity | 6 | Production default |
| Patch width and height | 128 | Production contract |
| Azimuth bins | 1,440 | Production contract |
| Azimuth spacing | 0.25 degrees | Derived from 1,440 bins |
| Subpatch size | 8 pixels | Production default, environment-overridable |
| Maximum ray distance | 1,000,000 m | Hard-coded batch value |
| Hierarchy | Enabled | Python default |
| Near-field reference merge | Disabled | Constructor default |
| Shared subpatch segment cache | Disabled | Hard-coded local constant |
| Output compression | Disabled | Python default, caller-selectable |

Initialization is not lazy inside the C# class. A Python replacement should
preserve lazy initialization at the Lunarscout API boundary even if its private
generator object also initializes eagerly.

## 4. Batch Pipeline

`GenerateHorizonsForPatches` has four functional stages despite its two-stage
producer/consumer description.

### 4.1 Pyramid Preparation

For each DEM, `BuildOrLoadPyramid`:

- Flattens the level-0 elevation raster in row-major order as `float32`.
- Builds max-reduction levels with a factor of four in X and Y until both
  dimensions reach one.
- Stores level 0 separately and concatenates all higher levels into one flat
  mip buffer with `LevelInfo.Offset`, `Width`, and `Height` metadata.
- Treats NaN, infinity, and values at or below -20,000 m as invalid.
- Writes -32,000 as the max-pyramid value for a block containing no valid
  samples.
- Builds missing levels with an ILGPU downsample kernel, not CPU code.
- Caches higher levels beside a file-backed DEM as `.pyr.bin`.
- Uploads level 0, mips, level metadata, map parameters, and projection
  parameters to the accelerator.

The cache validation currently checks only expected element count. It does not
record a source DEM hash, mtime, transform, algorithm version, or nodata policy.
The Numba evaluation must not use a stale pyramid cache as reference evidence.

### 4.2 CPU Segment Producer

One producer task processes patches sequentially. For each patch it calls
`CalculateSubpatchRaySegments`, records timing, and writes a `PatchWorkItem` to
a bounded channel.

For the default 128-pixel patch and 8-pixel subpatch:

- Interior subpatch grid: 16 by 16.
- Interpolation halo: one subpatch center on every side.
- Segment-center grid: 18 by 18, or 324 centers.
- Logical segment layout:
  `[azimuth][subpatch-center][DEM]`.
- Segment count per DEM per patch: `1,440 * 324 = 466,560`.

The optional `SubpatchSegmentCache` is instantiated with a local constant set
to `false`, so production currently recomputes all centers for every patch.

### 4.3 GPU Workers

Four worker tasks consume complete patch work items. Each worker:

1. Acquires reusable output, segment, and diagnostic device buffers.
2. Initializes a host slope array to negative infinity and copies it to the
   accumulation buffer.
3. Uploads all ray segments for the patch.
4. Acquires one accelerator stream.
5. Launches `QuadTreeSubpatchRayCastKernel` once per DEM on the same stream.
6. Accumulates the maximum horizon slope in the common output buffer.
7. Synchronizes only that stream.
8. Copies the slope array to the host.
9. Converts each slope to degrees using `atan(slope) * 180 / pi`.
10. Returns the stream and buffers to their pools.

The launch grid is logically `(pixel, azimuth)`, with 16,384 pixels and 1,440
azimuths for a full patch. The flattened output is pixel-major:

```text
output[pixel_index * 1440 + azimuth_index]
```

### 4.4 Output

Each completed patch is written immediately through `HorizonTileStore` using a
directory layout partitioned by Y. The result contains degrees, not slopes.
Compressed writes fall back to uncompressed output on compression failure.
Progress is reported after the file write.

## 5. Host Geometry and Segment Call Graph

The host side is primarily double precision until coefficients are stored in a
`RaySegment` for the `float32` device kernel.

`CalculateSubpatchRaySegments` performs:

1. Validates subpatch size and constructs the halo grid.
2. Computes tile-center grid convergence and per-pixel convergence gradients
   for stereographic maps.
3. Gets or creates a `SubpatchSegmentCache`.
4. Clamps each requested halo center to a legal DEM center.
5. Calls `SubpatchSegmentCache.GetCenterSegments`.
6. Copies each center's `[azimuth][DEM]` segments into the complete
   `[azimuth][subpatch][DEM]` patch array.

`SubpatchSegmentCache.ComputeCenterSegments` performs:

1. Converts the center pixel to projected coordinates and then latitude and
   longitude.
2. Bilinearly samples center terrain and adds observer elevation.
3. Constructs the Moon-centered observer vector and local east/north/up
   rotation.
4. Uses `Parallel.For` across 1,440 azimuths.
5. For each azimuth, calls `ComputeDirectionVector`.
6. For each nested DEM, calls `BuildRaySamples` over that DEM's distance range.
7. Calls `FitRaySegment` and advances the next DEM's start distance.

`BuildRaySamples` and its dependencies perform:

- Chord sampling from the observer vector.
- Moon-centered vector to latitude/longitude conversion.
- DEM-specific longitude/latitude to row/column conversion.
- Bounds detection and terrain interpolation.
- A maximum of 16 stored samples.
- A minimum of four samples spanning at least 100 m where possible.

`FitRaySegment` performs:

- A four-term quartic fit for pixel X versus distance.
- A four-term quartic fit for pixel Y versus distance.
- A cubic fit converting planar pixel displacement to chord distance.
- A zero-length fallback segment when fewer than three samples exist.
- Conversion of fitted coefficients to `float32` in the returned segment.

Key fit helpers are `FitQuartic4TermsDouble`, `SolveLinearSystem4`,
`FitPlanarToChordCubicWithTerrain`, `FitCubicNoIntercept`, and the spherical
chord-distance helpers. These functions need independent intermediate parity
tests before a complete CUDA comparison is meaningful.

## 6. Device Data Contract

### 6.1 `RaySegment`

Each segment contains:

- `StartPixel.X`, `StartPixel.Y`: starting pixel in the active DEM.
- `DemId`: logical source DEM index.
- `X0`, `Y0`: polynomial intercepts.
- `A1..A4`: X pixel coefficients.
- `B1..B4`: Y pixel coefficients.
- `SStart`, `SEnd`: fitted distance interval in kilometers.
- `SStartChord`: chord distance at the segment start in kilometers.
- `PlanarToChordC1..C3`: planar-meters to chord-meters cubic coefficients.

The prototype must measure the actual C# sequential struct size and field
offsets before declaring a binary-compatible NumPy dtype. Binary compatibility
is not otherwise required if both the Python host and CUDA kernel use the new
layout.

### 6.1 Phase 1 Artifact Decision

Phase 1 oracle artifacts will not export raw C# `RaySegment` struct bytes.
They will use a language-neutral structured representation: named NumPy arrays
in `.npz` with versioned JSON metadata that records axes, units, dtypes, and
precision boundaries. This avoids making CLR padding or ILGPU struct ABI part
of the oracle contract. Consequently, `Marshal.SizeOf` and C# field-offset
measurement are not Phase 0 requirements. They become necessary only if a later
experiment explicitly selects raw binary struct interchange.

### 6.2 Pyramid

- Level 0: row-major `float32[height * width]`.
- Mips: concatenated row-major `float32` levels 1 through N.
- Level metadata: offset, width, height, and currently unused cell-size fields.
- Map parameters: lunar radius, stereographic scale/false origins, inverse
  transform determinant, and six affine transform components.
- Projection parameters: lunar radius, projection origin, scale, false easting,
  and false northing.

### 6.3 Kernel Parameters

- Observer elevation in meters.
- Minimum traversal distance in kilometers.
- Diagnostic azimuth index.
- Grid-convergence center and gradients in radians.
- Debug flags.
- Primary DEM width and height.

The production subpatch kernel receives grid-convergence fields but currently
sets `correctedAzIdx = azIdx` and does not apply them. The older
`QuadTreeRayCastKernel` does apply convergence gradients. This difference must
be treated as current production behavior and investigated scientifically; it
must not be silently "fixed" during a mechanical port.

### 6.4 Output

The device output is `float32` horizon slope. Negative infinity means no valid
horizon was found. DEM passes update the same buffer by retaining the maximum
slope. Degree conversion happens once on the CPU after all passes.

## 7. Production Subpatch Kernel

For each `(pixel, azimuth)` thread, `QuadTreeSubpatchRayCastKernel`:

1. Computes the pixel's position within the patch.
2. Identifies four neighboring subpatch centers.
3. Loads four ray segments using
   `[azimuth][subpatch][DEM]` indexing.
4. Shifts each center segment to the current pixel using the primary-to-active
   DEM resolution ratio.
5. Bilinearly interpolates all segment fields.
6. Samples observer terrain from the primary DEM and adds observer elevation.
7. Marches from the segment start to end.
8. Evaluates quartic pixel X/Y at each step.
9. Stops at NaN or active-DEM bounds.
10. Uses direct distance below 500 m and polynomial-corrected chord distance
    beyond it.
11. Chooses either level-0 adaptive traversal or hierarchical traversal.
12. Updates the maximum slope and stores it back to the shared output bin.

### 7.1 Level-0 Path

When hierarchy is disabled, the kernel:

- Bilinearly samples level 0.
- Uses a flat local slope below 500 m.
- Uses a spherical law-of-cosines formulation beyond 500 m.
- Selects step size from pixel-path tangent, horizon margin, angular budget,
  distance, DEM resolution, and the primary-DEM far-field floor.

### 7.2 Hierarchical Path

When hierarchy is enabled, the kernel:

- Selects a starting max-pyramid level based on distance and map resolution.
- Locates the containing block by factor-four bit shifts.
- Computes a tangent-linear estimate of the distance to the block exit.
- Computes a conservative possible slope from the block maximum and a nearer
  distance bound.
- Skips a block when that possible slope cannot exceed the current horizon.
- Descends to finer levels otherwise.
- Bilinearly samples level 0 and advances adaptively when descent reaches zero.

The dynamic ray length, per-thread hierarchy descent, bounds exits, nodata
skips, and adaptive advances are the main CUDA divergence risks.

## 8. Constants That Affect Parity

| Constant | Current value | Meaning |
| --- | ---: | --- |
| `PYR_DOWNSAMPLE_FACTOR` | 4 | Pyramid reduction per dimension |
| `DEFAULT_SUBPATCH_SIZE` | 8 px | Segment-center spacing |
| `DEBUG_FIXED_STEP_METERS` | 1.2 m | Diagnostic fixed step |
| `COMPARISON_EPSILON` | 0 | Hierarchy culling margin |
| `ADAPTIVE_EPSILON_C0/C1` | 0 / 0 | Additional adaptive culling margin |
| `GUARD_BAND_PIXELS` | 0.5 px | Block-bound inflation in older kernel helpers |
| `PRIMARY_DEM_FAR_MIN_STEP_DISTANCE_METERS` | 100 m | Primary far-step threshold |
| `MIN_ADAPTIVE_STEP_RESOLUTION_FACTOR` | 0.5 | General minimum step versus resolution |
| `PRIMARY_DEM_FAR_MIN_STEP_RESOLUTION_FACTOR` | 0.8 | Primary far-field minimum step |
| `INV_TAN_MAX_SLOPE` | 1.732 | Margin-step terrain assumption |
| `ANGULAR_STEP_FACTOR` | 0.00151 | Angular step-error budget |
| Near/far threshold | 0.5 km | Switch in slope/distance formulation |
| `MIN_RAY_SAMPLE_COUNT` | 4 | Host segment-fit target |
| `MIN_RAY_SAMPLE_SPAN_METERS` | 100 m | Host segment-fit target |
| `MAX_RAY_SAMPLE_CAPACITY` | 16 | Host fit buffer size |
| `MIN_PLANAR_SPAN_FOR_CHORD_FIT_METERS` | 500 m | Chord-fit condition |

Every constant above must initially be copied exactly and covered by a targeted
case. Tuning belongs after baseline parity.

## 9. Runtime and Diagnostic Modes

| Control | Current effect | Prototype treatment |
| --- | --- | --- |
| Python `disable_hierarchy` | Sets hierarchy debug flag; default `False` | Required diagnostic and parity mode |
| `QUADTREE_FORCE_FIXED_STEPS` | Forces 1.2 m march steps | Required diagnostic mode |
| `QUADTREE_PIPELINE_SUBPATCH_SIZE` | Selects 2, 4, 8, 16, 32, 64, or 128 | Preserve for comparisons; target 8 first |
| `QUADTREE_PIPELINE_PROFILE` | Enables host pipeline timing logs | Preserve equivalent measurements |
| `QUADTREE_TRAVERSAL_PROFILE` | Compile-time traversal counters | Reproduce with optional diagnostic arrays |
| `QUADTREE_DEBUG_AZ` | Selects diagnostic azimuth/sample output in non-batch paths | Replace with explicit diagnostic arguments |
| `QUADTREE_DEBUG_DEM` | Selects diagnostic DEM | Replace with explicit diagnostic arguments |
| Near-field reference merge | Optional constructor mode, off in Python path | Evaluate after core parity |
| Shared segment cache | Hard-coded off | Measure before deciding Python behavior |

Environment variables are acceptable for preserving the baseline but should
not become the only interface to new Python diagnostic behavior.

## 10. Current Correctness Oracles

The existing fast tests establish useful but incomplete evidence:

- `QuadTreeProductionRegressionTests`: flat terrain, east obstacle, tile
  boundary, and multi-DEM outer obstacle through the production generator.
- `NativeHorizonScenarioRegressionTests`: independent reference rays for flat,
  obstacle, and multi-DEM cases.
- `CoordinateConversionComparisonTests`: comparison with GDAL transforms.
- `HorizonCompressorTests`, `HorizonFileTests`, and `HorizonTileStoreTests`:
  output encoding and layout behavior.
- Bounding-box and projection tests: supporting coordinate behavior.

Known gaps before porting can be called scientifically equivalent:

- No committed intermediate arrays for pyramids, samples, segments, traversal,
  per-DEM slopes, or final degree buffers.
- No automated comparison of the production quadtree output with reference-ray
  output over a complete horizon.
- The accelerator probe confirms CUDA selection but does not yet attach the
  selected accelerator name to every individual regression result.
- Production regression tests now explicitly construct
  `QuadTreeHorizonGenerator(disableHierarchy: false)`, matching the Python
  production default.
- No real-terrain committed fixture or recorded external fixture provenance.
- No accepted numerical or downstream lighting error budget.
- The Phase 0 production report covers one production patch and a bounded
  four-patch batch. It does not yet establish variance, whole-region
  throughput, or scaling across concurrency settings; those remain Phase 5
  work.

Phase 1 fixtures must continue to exercise
`new QuadTreeHorizonGenerator(disableHierarchy: false)` and retain separate
fixed-step and hierarchy-disabled diagnostic comparisons.

## 11. Replacement Boundary

The minimum algorithm replacement consists of:

- Max-pyramid construction and validated caching.
- Projection and Moon-centered host geometry.
- Ray sampling and segment fitting.
- Subpatch halo construction and interpolation.
- Single- and multi-DEM fixed-step, adaptive, and hierarchical traversal.
- Slope accumulation and degree conversion.
- Patch scheduling, bounded concurrency, cancellation, progress, and resumption.
- Horizon file compatibility or a deliberate versioned replacement.

GDAL file loading, file writing, and the public wrapper can remain outside the
first kernel prototype. They must be included before claiming that .NET can be
removed.

## 12. Phase 0 Completion State

Completed:

- [x] Record the clean C# baseline commit.
- [x] Capture host, GPU, driver, CUDA toolkit, Python, and .NET facts.
- [x] Capture a zero-warning moonlib rebuild.
- [x] Capture the native `Fast` test result.
- [x] Select the local four-DEM test scenario as the first real baseline.
- [x] Capture a fresh hierarchy-enabled production patch twice with matching
      pyramid and output hashes.
- [x] Trace the public Python-to-C# production call path.
- [x] Inventory production host geometry and device helpers.
- [x] Record array layouts, units, precision boundaries, sentinels, and modes.
- [x] Confirm that Python defaults to hierarchy enabled.
- [x] Make production regression tests exercise hierarchy-enabled traversal.
- [x] Add a separate test that requires and records CUDA accelerator selection.
- [x] Add reproducible manifest capture tooling.
- [x] Capture peak sampled per-process GPU memory for cold and warm production
      generation scopes.
- [x] Capture bounded four-patch cold and warm throughput with production
      concurrency, per-stage timing, input/output hashes, accelerator identity,
      and host/device memory.
- [x] Select language-neutral Phase 1 oracle artifacts instead of raw C# struct
      interchange.

Phase 0 evidence is complete on the dedicated evaluation branch. Formal
deliverable state:

- [x] Commit the inventory and machine-readable manifests.

Phase 1 reference artifact generation and the synthetic/real fixture matrix
were completed after the Phase 0 baseline and remain distinct from it.

The Phase 1 reference-ray artifact now lives at
`tests/data/numba_horizon/phase1_reference_rays.npz` with adjacent JSON
metadata. Its schema is documented in
`docs/numba-horizon-phase-1-oracle-schema.md`, and external real-terrain
fixtures in `docs/numba-horizon-phase-1-real-terrain-fixtures.json`. These do
not change the Phase 0 baseline evidence or begin the Numba implementation.
