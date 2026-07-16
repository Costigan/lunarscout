# Numba Horizon Prototype and Evaluation Plan

**Plan date:** 2026-07-15

**Status:** proposed

**Target branch:** `spike/numba-horizon`

## 1. Purpose

Evaluate whether the production quadtree horizon-generation algorithm in
`native/moonlib/horizon/QuadTreeHorizonGenerator.cs` can be implemented in
Python with Numba CPU compilation and Numba CUDA kernels well enough to remove
the .NET runtime from Lunarscout.

The prototype is successful only if it establishes scientific correctness,
production-scale performance, memory feasibility, maintainability, and a
reproducible installation path. Merely launching a CUDA kernel or matching a
small synthetic example is not sufficient.

The initial GPU target is NVIDIA CUDA on Linux. CPU execution is required for
preprocessing and may provide a slow reference or fallback, but a fast CPU-only
horizon generator is not required for this evaluation. AMD and Apple GPU
support are future investigations and must not constrain the first prototype.

## 2. Decision to Be Made

At the end of the prototype, choose one of these outcomes:

1. **Proceed with replacement.** The Numba implementation is a credible basis
   for replacing moonlib horizon generation and ultimately removing .NET.
2. **Continue targeted research.** The approach is promising, but a bounded
   correctness, performance, packaging, or maintainability problem remains.
3. **Retain moonlib.** The prototype demonstrates that replacing the current
   implementation would cause unacceptable scientific, performance, or
   operational regressions.

The prototype must produce evidence for this decision rather than assume that
removing .NET is inherently worth any regression.

## 3. Scope

### Included

- [ ] Production `QuadTreeHorizonGenerator` behavior with hierarchy enabled.
- [ ] CPU construction and reuse of quadtree metadata, elevation pyramids,
      subpatch ray segments, and grid-convergence parameters.
- [ ] Numba CUDA implementation of the subpatch horizon ray-casting kernel.
- [ ] Fixed-step, adaptive level-0, and hierarchical traversal modes needed to
      isolate correctness problems.
- [ ] Single-DEM and multi-DEM horizon generation.
- [ ] Subpatch interpolation, including the interpolation halo and boundary
      clamping.
- [ ] Observer elevation, nodata, DEM edges, tile edges, and lunar-curvature
      calculations.
- [ ] Patch-level generation for the production 128 by 128 patch size and 1,440
      azimuth bins.
- [ ] Comparison with the current C# reference and ILGPU implementations.
- [ ] End-to-end scheduling, streaming, cancellation, progress, caching, and
      horizon-file compatibility after the core algorithm passes.
- [ ] Installation and import behavior in a clean Python environment.

### Excluded From the First Prototype

- [ ] AMD/HIP and Apple/Metal GPU backends.
- [ ] Rewriting unrelated moonlib lighting, SPICE, GDAL, or product-generation
      code.
- [ ] Removing C#, Python.NET, or .NET before the final evaluation is accepted.
- [ ] Changing Lunarscout's public horizon API during the algorithm prototype.
- [ ] Adding Numba or CUDA dependencies to published package metadata before
      the packaging and adoption gate.
- [ ] General optimization of algorithms unrelated to horizon generation.
- [ ] Treating bit-for-bit equality across GPU architectures as a requirement.

## 4. Working Method and Repository Layout

Perform the work in the existing repository so the C# implementation, native
tests, Python wrappers, and file formats remain available as comparison
oracles. Use a dedicated branch and preferably a second Git worktree. Do not
create a separate project or delete the existing implementation during the
evaluation.

Recommended private prototype layout:

```text
src/lunarscout/_numba_horizon/
    __init__.py
    contracts.py
    geometry.py
    pyramid.py
    segments.py
    cuda_kernel.py
    generator.py
    diagnostics.py
tests/numba_horizon/
tests/data/numba_horizon/
scripts/numba_horizon/
```

The exact module split may change as the port reveals better boundaries. The
following constraints should remain:

- `_numba_horizon` is private and is not exported from `lunarscout.__init__`
  during the evaluation.
- Importing `lunarscout` must not import Numba, initialize CUDA, select a GPU,
  or compile a kernel.
- Core calculation functions accept NumPy arrays and explicit scalar metadata;
  they do not open GDAL datasets or depend on Python.NET.
- CPU and CUDA functions are kept distinct. Shared conventions should be
  represented by constants and tests rather than device-incompatible Python
  abstractions.
- Large generated results and local DEMs remain outside Git. Small synthetic
  fixtures and compact reference results may be committed.
- Prototype-only dependencies are recorded separately and do not silently
  become Lunarscout runtime dependencies.

## 5. Reproducible Evaluation Environment

The evaluation is not reproducible unless another clean environment can run
the same fixtures and benchmarks.

- [ ] Record the checkpoint commit containing the C# baseline.
- [ ] Record the prototype commit for every published comparison report.
- [ ] Select one initial Python version already supported by Lunarscout.
- [ ] Select and pin compatible NumPy, Numba, CUDA Python package, and testing
      versions in a prototype-specific input and lock file.
- [ ] Record the Linux distribution, kernel, NVIDIA driver, CUDA runtime, GPU
      model, GPU memory, CPU model, RAM, and storage used for benchmarks.
- [ ] Add a smoke script that reports the toolchain and executes a trivial
      compiled CPU function and CUDA kernel.
- [ ] Run the environment setup and smoke script from a clean virtual
      environment.
- [ ] Verify the chosen CUDA package does not initialize CUDA during an
      ordinary `import lunarscout`.
- [ ] Decide whether cached compiled kernels are part of the supported setup or
      only an optimization; measure both cold and warm behavior.
- [ ] Store machine-readable environment and benchmark metadata with every
      result set.

Do not spend time on a multi-version dependency matrix until the core kernel
has passed the first correctness gate. The first goal is one completely
specified, repeatable environment.

## 6. Phase 0: Inventory and Freeze the Baseline

Before porting, document exactly which current behavior is being evaluated.
`QuadTreeHorizonGenerator.cs` contains algorithm logic, diagnostic paths,
scheduling, caches, and file production in one large class. These must be
separated conceptually before deciding what belongs in CPU code, CUDA code, or
later integration work.

- [x] Trace the production call path from Python `GenerateHorizons` through
      patch enumeration, segment generation, GPU launch, degree conversion,
      compression, and file writing.
- [x] Inventory every host function called directly or indirectly by
      `CalculateSubpatchRaySegments` and `SubpatchSegmentCache`.
- [x] Inventory every device helper called by
      `QuadTreeSubpatchRayCastKernel`.
- [x] Record the array layouts, dimensions, units, types, sentinels, nodata
      rules, coordinate conventions, and azimuth orientation.
- [x] Record all algorithm constants and environment-controlled behavior,
      including subpatch size, fixed-step mode, hierarchy mode, near-field
      merge, traversal profiling, and pipeline concurrency.
- [x] Determine which modes are production requirements and which exist only
      for diagnosis or historical compatibility.
- [x] Confirm the production Python path uses `disableHierarchy: false` and
      preserve that as the final parity target.
- [x] Capture current C# build and focused test results.
- [x] Capture baseline output and timing before making diagnostic changes to
      the C# implementation.

### Baseline Deliverable

- [x] Commit an algorithm inventory describing the host/device boundary and
      the data passed across it.
- [x] Commit a machine-readable baseline manifest containing source commit,
      parameters, environment, inputs, output hashes, timings, and test results.

The inventory, baseline manifests, and bounded cold/warm production benchmark
are complete and included on the dedicated evaluation branch.

## 7. Phase 1: Establish Independent Oracles and Fixtures

The current ILGPU implementation cannot be the only oracle because reproducing
an existing implementation bug would look like success. Use three levels of
evidence:

1. `ReferenceRayEmulator` or another deliberately simple CPU ray calculation
   for scientific behavior along selected rays.
2. The current hierarchy-enabled ILGPU implementation for production parity.
3. Analytical expectations for synthetic terrain where the answer is known.

### Synthetic Fixture Matrix

- [x] Flat spherical terrain at multiple map resolutions.
- [x] One obstacle in each cardinal and intercardinal direction.
- [x] Multiple peaks along one ray, including a near lower peak and a far
      higher peak.
- [x] Ridge crossing several adjacent azimuth bins.
- [x] Negative elevations and an elevated observer.
- [x] Nodata holes, nodata borders, and an entirely nodata ray segment.
- [x] Observer and obstacle near DEM, tile, and subpatch boundaries.
- [x] Partial tiles and DEM dimensions that are not powers of four.
- [x] Multi-DEM coverage with different resolutions and a horizon-setting
      feature in an outer DEM.
- [x] Locations where grid convergence is materially nonzero.
- [x] Cases just below and above near/far calculation thresholds.

### Real-Terrain Fixture Matrix

- [x] Select a small, redistributable real lunar DEM subset for committed or
      automatically acquired validation.
- [x] Select at least one larger local scenario representing normal production
      terrain, range, and multi-DEM use.
- [x] Include smooth, rugged, boundary, and high-latitude observers.
- [x] Record provenance and hashes for every external DEM.
- [x] Ensure tests skip external fixtures explicitly rather than silently
      substituting synthetic data.

### Reference Artifacts

For each fixture, capture applicable intermediate and final arrays:

- [x] DEM level 0 and all max-pyramid levels.
- [x] Level offsets, widths, heights, transforms, and projection parameters.
- [x] Subpatch centers and grid-convergence values.
- [x] Ray samples before fitting.
- [x] Quartic pixel-path coefficients and planar-to-chord coefficients.
- [x] Per-DEM slope buffers before merging.
- [x] Final slope and degree buffers.
- [x] Selected traversal traces showing levels, cells, samples, and advances.

Use a documented, language-neutral format such as `.npz` plus JSON metadata.
Every artifact schema needs a version and explicit units. Do not parse C# log
text as a data interchange format.

The completed Phase 1 artifact is recorded in
`tests/data/numba_horizon/phase1_reference_rays.npz` with versioned JSON
metadata and schema documentation in
`docs/numba-horizon-phase-1-oracle-schema.md`. It covers 27 compact
independent-reference cases spanning the synthetic matrix, plus nominal-bin
production `BuildRaySamples` inputs and fitted `RaySegment` coefficients. It is
byte-reproducible across repeated captures.
It also captures every max-pyramid level, level offset/dimension record, map
parameter vector, and projection parameter vector for the current DEM fixtures
through the production CUDA path, including an odd-sized NaN, infinity, cutoff,
and all-invalid-block fixture. A bounded corner fixture also captures a complete
halo-inclusive subpatch grid for 16 azimuths and two DEMs, including boundary
clamping and grid convergence. A one-pixel, 1,440-bin, two-DEM CUDA fixture
captures each pass before merge, final slopes and degrees, and 925 selected
hierarchy traversal steps across levels zero and one. Real-terrain selections,
provenance, acquisition, hashes, and explicit external-test gating are recorded
in `docs/numba-horizon-phase-1-real-terrain-fixtures.json`.

## 8. Phase 2: Define the Python Data Contract

Translate C# structs into device-friendly arrays rather than attempting to
mirror an object graph.

- [ ] Define NumPy dtypes and shapes for map parameters, projection parameters,
      pyramid level metadata, kernel parameters, and outputs.
- [ ] Evaluate a structured dtype versus structure-of-arrays storage for ray
      segments; benchmark both only if the choice is not clear from generated
      memory access.
- [ ] Preserve `float64` for host geometry and polynomial fitting where the C#
      implementation uses `double`.
- [ ] Make every conversion to `float32` at the device boundary explicit and
      tested.
- [ ] Define the segment layout equivalent to
      `[azimuth][subpatch][DEM]` and test index calculations independently.
- [ ] Define flattened pyramid storage and offsets without relying on C#
      `ArrayView` semantics.
- [ ] Define slope-buffer sentinels and the single conversion point from slope
      to degrees.
- [ ] Add validation for contiguity, dtype, dimensions, bounds, and supported
      configuration before CUDA initialization.
- [ ] Round-trip all reference artifacts through the Python contract.

### Contract Gate

- [ ] Python can load every reference artifact without Python.NET.
- [ ] Indexing and interpolation select the same segments and pyramid cells as
      C# for targeted diagnostic cases.
- [ ] Units and precision boundaries are documented and covered by tests.

## 9. Phase 3: Port and Test Host-Side Geometry

Port host calculations incrementally. Begin with ordinary Python or NumPy where
that makes comparison easy, then apply `@njit` and `parallel=True` only to
measured CPU bottlenecks and naturally independent loops.

- [ ] Port affine pixel/CRS conversion and stereographic projection helpers.
- [ ] Port latitude/longitude and Moon-centered vector conversions.
- [ ] Port local east/north/up rotation and azimuth direction construction.
- [ ] Port chord sampling and DEM-bound intersection behavior.
- [ ] Port sample placement and minimum-sample rules.
- [ ] Port four-term quartic fitting and singular-system fallback behavior.
- [ ] Port planar-distance to chord-distance fitting.
- [ ] Port DEM segment-context and ray-limit construction.
- [ ] Port subpatch-center clamping and interpolation-halo construction.
- [ ] Port grid-convergence calculation or define a small non-GDAL input
      contract through which Lunarscout supplies it.
- [ ] Port segment caching without assuming that a Python dictionary is usable
      inside compiled code.
- [ ] Add deterministic comparison tests for every intermediate result.
- [ ] Profile serial Python, compiled serial CPU, and compiled parallel CPU
      implementations of segment generation.
- [ ] Confirm parallel execution is deterministic within the accepted numeric
      tolerance and does not oversubscribe the end-to-end pipeline.

### Host-Side Gate

- [ ] All synthetic segment coefficients match the baseline within a justified
      tolerance.
- [ ] Real-terrain segment paths remain within an accepted pixel error over the
      entire fitted distance, not only at sample points.
- [ ] Segment preparation is fast enough to overlap GPU work or is identified
      with a specific optimization plan.
- [ ] Peak host memory is acceptable when caches cover a realistic patch batch.

## 10. Phase 4: Build the CUDA Kernel in Diagnostic Stages

Do not port the complete optimized kernel in one step. Each stage should retain
a selectable diagnostic mode until the next stage is validated.

### Stage 4A: CUDA Mechanics

- [ ] Implement lazy device selection, allocation, launch, synchronization, and
      result-copy helpers.
- [ ] Verify the `(pixel, azimuth)` thread mapping and output indexing.
- [ ] Implement and test device helpers for interpolation, polynomial
      evaluation, tangents, bilinear sampling, validity, and subpatch clamping.
- [ ] Confirm bounds checks and sentinel behavior under CUDA's error model.
- [ ] Add a CUDA-simulator or CPU-helper test only where it tests the same
      arithmetic; do not claim it validates real GPU behavior.

### Stage 4B: Fixed-Step Level-0 Traversal

- [ ] Implement level-0 ray marching with a deliberately fixed, conservative
      step.
- [ ] Compare selected rays against `ReferenceRayEmulator` sample by sample.
- [ ] Validate near-field and spherical far-field slope calculations.
- [ ] Validate quartic evaluation and chord correction independently from
      adaptive stepping.
- [ ] Produce a compact traversal trace for a selected pixel and azimuth.

### Stage 4C: Adaptive Level-0 Traversal

- [ ] Port tangent-based pixel stepping, margin stepping, angular stepping, and
      minimum step floors.
- [ ] Compare adaptive results against the fixed-step implementation.
- [ ] Identify every mismatch caused by skipped terrain rather than merely
      increasing a global tolerance.
- [ ] Test discontinuities around the 500-meter near/far threshold and the
      primary-DEM far-step threshold.

### Stage 4D: Hierarchical Traversal

- [ ] Build max pyramids with the same factor-four reduction and nodata rules.
- [ ] Port start-level selection, block bounds, conservative possible-slope
      calculation, culling, descent, and block-exit advancement.
- [ ] Add optional traversal counters analogous to the C# diagnostic build.
- [ ] Compare hierarchy-enabled output to adaptive level-0 output and the C#
      hierarchy-enabled output.
- [ ] Prove that hierarchy culling does not lower a true horizon beyond the
      accepted scientific tolerance.
- [ ] Test rays tangent to block boundaries and rays with nearly zero X or Y
      tangent components.

### Stage 4E: Full Subpatch and Multi-DEM Operation

- [ ] Port four-segment bilinear interpolation for each pixel and azimuth.
- [ ] Validate edge clamping and interpolation halos for full and partial
      patches.
- [ ] Run one pass per DEM while accumulating the maximum slope.
- [ ] Validate maps with different resolution, extent, and ray limits.
- [ ] Apply degree conversion once, after all DEM passes.
- [ ] Evaluate the optional near-field reference merge separately and decide
      whether it is required for replacement.

### Kernel Correctness Gate

- [ ] All analytical synthetic expectations pass.
- [ ] Fixed-step selected rays agree with the independent reference calculation.
- [ ] Adaptive and hierarchical modes meet the accepted error budget.
- [ ] Single- and multi-DEM real-terrain comparisons meet the accepted error
      budget without unexplained spatial patterns.
- [ ] Repeated warm runs produce stable results on the reference GPU.
- [ ] CUDA memory checking and out-of-bounds diagnostics report no errors.

## 11. Correctness Metrics and Acceptance Policy

Do not choose a broad tolerance before measuring the existing disagreement
between `ReferenceRayEmulator`, fixed-step ILGPU, and production ILGPU. Establish
the baseline first, then adopt thresholds no weaker than the current production
error unless a scientifically reviewed change is intentional.

Report at least:

- [ ] Maximum and mean absolute angular error.
- [ ] Median, 95th, 99th, and 99.9th percentile absolute angular error.
- [ ] Counts above each selected angular threshold.
- [ ] Azimuth and spatial locations of the largest errors.
- [ ] NaN, infinity, sentinel, and missing-bin counts.
- [ ] Directional bias and signed error distribution.
- [ ] Error separated by DEM pass, distance, terrain class, tile edge, subpatch
      edge, nodata proximity, and hierarchy mode.
- [ ] Horizon-setting obstacle distance for selected diagnostic rays.

The comparison harness should fail on shape, metadata, missing values, or
unexplained sentinel differences before computing aggregate statistics.

### Downstream Scientific Tests

Small angular errors matter only through their effect on supported products.
For representative Sun and Earth geometries:

- [ ] Compare lit/shadow classification from C# and Numba horizons.
- [ ] Compare visibility classification near the horizon threshold.
- [ ] Compare accumulated illumination or PSR results for a representative
      temporal interval.
- [ ] Report the number, location, and duration of changed classifications.
- [ ] Obtain an explicit decision on acceptable downstream disagreement before
      declaring algorithm parity.

## 12. Phase 5: Performance and Resource Evaluation

Benchmark correctness-approved implementations only. Use identical inputs,
hardware, output scope, and concurrency settings for C#/ILGPU and Numba.

### Measurements

- [ ] Cold process startup and first kernel compilation.
- [ ] Warm generator initialization.
- [ ] Pyramid construction or cache loading.
- [ ] Segment construction and cache lookup.
- [ ] Host-to-device segment transfer.
- [ ] Kernel execution per DEM pass.
- [ ] Device synchronization and device-to-host transfer.
- [ ] Slope-to-degree conversion.
- [ ] Compression and file writing when end-to-end testing begins.
- [ ] Total latency for one patch and throughput for many patches.
- [ ] Peak host RAM, device RAM, and retained cache size.
- [ ] GPU utilization, occupancy, and major causes of warp divergence.
- [ ] Scaling with patch count and concurrent streams.

### Benchmark Matrix

- [ ] One pixel by all azimuths for diagnostic overhead.
- [ ] Small blocks for correctness-development feedback.
- [ ] One production 128 by 128 patch.
- [ ] A contiguous multi-patch batch with cache reuse.
- [ ] Single-DEM and representative multi-DEM cases.
- [ ] Smooth and rugged real terrain.
- [ ] Hierarchy disabled and enabled.
- [ ] Cold and warm runs, with enough repetitions to report variance.

### Optimization Order

- [ ] Measure before changing the algorithm or memory layout.
- [ ] Remove redundant transfers and allocations.
- [ ] Reuse device buffers and streams.
- [ ] Improve segment and pyramid memory access.
- [ ] Tune launch geometry and register pressure.
- [ ] Evaluate overlap of CPU segment generation, transfers, kernels, and output
      writing.
- [ ] Re-run the complete correctness suite after every optimization that can
      alter arithmetic or traversal.

### Performance Gate

Before benchmarking, record explicit acceptable ratios relative to the current
ILGPU implementation for:

- [ ] Warm single-patch latency.
- [ ] Sustained multi-patch throughput.
- [ ] Peak device memory.
- [ ] Peak host memory.
- [ ] First-use compilation latency.

A performance regression may be accepted in exchange for removing .NET, but
the accepted cost must be stated rather than hidden in an aggregate benchmark.

## 13. Phase 6: Production-Pipeline Prototype

Only begin this phase after the kernel correctness and initial performance
gates pass.

- [ ] Reproduce patch enumeration and partial-edge handling.
- [ ] Reproduce skip-existing behavior based on the final output contract.
- [ ] Implement bounded CPU preparation and GPU work queues.
- [ ] Reuse segment caches across neighboring patches without unbounded growth.
- [ ] Reuse device buffers and CUDA streams safely.
- [ ] Implement progress events with the existing user-visible semantics.
- [ ] Check cancellation between bounded units of work.
- [ ] Stage file outputs so failure cannot corrupt an existing completed product.
- [ ] Write files readable by the existing horizon readers and downstream
      lighting code.
- [ ] Compare compressed and uncompressed output, including metadata and
      completion detection.
- [ ] Test interruption, CUDA failure, disk-full simulation where practical,
      overwrite failure, and restart/skip behavior.
- [ ] Confirm large runs stream results instead of retaining the full regional
      horizon cube in memory.

### Pipeline Gate

- [ ] A representative real scenario completes without Python.NET or moonlib
      after the DEM arrays and metadata have been supplied.
- [ ] Existing downstream code consumes the generated horizon product.
- [ ] Failure and cancellation leave no output that appears complete.
- [ ] Progress, resumption, and overwrite behavior are documented and tested.
- [ ] End-to-end throughput and memory meet the recorded acceptance criteria.

## 14. Phase 7: Packaging and Operational Evaluation

- [ ] Test installation from a built wheel in a clean environment on the
      selected Linux/Python/CUDA combination.
- [ ] Identify which CUDA components come from Python packages and which are
      required from the system or NVIDIA driver.
- [ ] Confirm package metadata can state these requirements truthfully.
- [ ] Verify behavior on a machine with no NVIDIA GPU and on a machine with an
      incompatible or missing driver.
- [ ] Return a structured, actionable error without initializing unrelated
      native systems.
- [ ] Decide whether a CPU horizon fallback is required or whether horizon
      generation is explicitly NVIDIA-only for the first release.
- [ ] Decide whether Numba is imported lazily or placed in a separately loaded
      internal backend.
- [ ] Measure installed dependency size and clean-environment resolution time.
- [ ] Document kernel compilation and cache locations, invalidation, and disk
      use.
- [ ] Repeat the smoke and focused integration tests on a second clean machine
      or independently constructed environment.

No package dependency decision is accepted until this phase demonstrates the
actual installation and failure behavior.

## 15. Test Organization

Use explicit test tiers so ordinary development does not require a real GPU or
external lunar data.

### Tier A: Ordinary CPU Tests

- [ ] Data contracts, indexing, transforms, geometry, fitting, pyramid
      construction, validation, and error handling.
- [ ] Small committed fixtures only.
- [ ] No CUDA initialization and no moonlib requirement unless generating an
      oracle fixture in a separate command.

### Tier B: Focused NVIDIA Tests

- [ ] Small synthetic CUDA kernel tests on a real NVIDIA GPU.
- [ ] Fixed-step, adaptive, hierarchy, subpatch, nodata, boundary, and multi-DEM
      coverage.
- [ ] Marked explicitly so absence of hardware produces a visible skip reason.

### Tier C: C# Parity Tests

- [ ] Load immutable C# reference artifacts for most comparisons.
- [ ] Provide a separate regeneration command that builds moonlib and records
      the exact source commit.
- [ ] Detect stale artifacts rather than regenerating them silently during a
      test.

### Tier D: Real-Scenario and Performance Tests

- [ ] External data and real GPU required.
- [ ] Never included in ordinary unit-test timing claims.
- [ ] Produce machine-readable correctness, timing, memory, and environment
      reports.
- [ ] Retain representative reports with the final evaluation.

### Regression Expectations

- [ ] Every diagnosed mismatch becomes a minimal regression test where
      practical.
- [ ] Tests exercise production hierarchy-enabled behavior, not only the easier
      hierarchy-disabled mode.
- [ ] CUDA tests synchronize explicitly so asynchronous failures are attributed
      to the correct test.
- [ ] Expected skips are enumerated in the report; an unconfigured integration
      tier is not described as passing.

## 16. Risks and Required Experiments

### R1: Host Geometry Dominates Runtime

`SubpatchSegmentCache` computes many double-precision ray samples and quartic
fits. Numba CUDA does not solve a CPU preparation bottleneck.

- [ ] Measure existing and ported segment-generation time separately.
- [ ] Test cache reuse across adjacent patches.
- [ ] Determine whether Numba parallel CPU compilation is sufficient.

### R2: CUDA Control-Flow Divergence

Adaptive hierarchical rays can take different paths and iteration counts.

- [ ] Profile iteration distribution and warp divergence.
- [ ] Compare thread mappings and batching strategies without changing results.

### R3: Precision Changes Alter Traversal

A small arithmetic difference can change a culling decision and produce a
larger horizon difference.

- [ ] Compare intermediate possible slopes, selected levels, and advances at
      the first divergence.
- [ ] Keep host double precision and device float precision boundaries explicit.

### R4: Device-Friendly Layout Becomes Unmaintainable

Flattened arrays and duplicated device helpers can obscure scientific intent.

- [ ] Document every array axis and unit next to its contract.
- [ ] Require CPU unit tests for device arithmetic helpers where feasible.
- [ ] Review the completed kernel for maintainability before optimizing further.

### R5: Numba/CUDA Packaging Is Not Reproducible

- [ ] Build and install from a clean artifact rather than the source checkout.
- [ ] Test supported driver and CUDA combinations explicitly.
- [ ] Record compilation cache and first-use behavior.

### R6: The Prototype Accidentally Narrows Product Behavior

- [ ] Inventory multi-DEM, near-field merge, compression, resumption,
      cancellation, and diagnostic behavior before calling the replacement
      complete.
- [ ] Label anything deliberately deferred in the final decision report.

## 17. Decision Gates

### Gate A: Environment and Oracle Ready

- [ ] A clean environment runs CPU and CUDA smoke tests.
- [ ] Baseline artifacts are versioned, reproducible, and independently checked.
- [ ] Production modes and parameters are documented.

**Failure action:** stop implementation work until the baseline can be trusted.

### Gate B: Host Calculations Credible

- [ ] Segment and pyramid intermediates meet parity requirements.
- [ ] Host preparation time and memory have a viable production path.

**Failure action:** isolate the failed geometry or fitting component and decide
whether to retain a small native implementation; do not proceed by loosening
tolerances without evidence.

### Gate C: CUDA Algorithm Correct

- [ ] Fixed-step, adaptive, and hierarchical modes pass the agreed tests.
- [ ] Downstream scientific comparisons are acceptable.

**Failure action:** classify the mismatch as a port defect, current C# defect,
or algorithm sensitivity before continuing to optimization.

### Gate D: Performance Viable

- [ ] Latency, throughput, memory, and first-use costs meet recorded criteria.

**Failure action:** perform only measured, bounded optimization experiments. If
the remaining gap has no credible remedy, retain moonlib.

### Gate E: Replacement Viable

- [ ] Production pipeline behavior and files are compatible.
- [ ] Clean installation and failure behavior are truthful.
- [ ] A second environment reproduces the principal results.
- [ ] The remaining moonlib dependency inventory shows that removing .NET is
      actually achievable, not merely that horizon generation moved.

**Failure action:** keep the Numba backend experimental or retain moonlib while
documenting the exact blocker.

## 18. Final Evaluation Report

Create `docs/numba-horizon-evaluation.md` at the end of the spike. It must
include:

- [ ] Executive recommendation: replace, continue research, or retain moonlib.
- [ ] Source commits and complete environment manifests.
- [ ] Implemented and deferred behavior.
- [ ] Correctness tables and error visualizations.
- [ ] Downstream scientific comparison results.
- [ ] Cold and warm performance tables with variance.
- [ ] Host and device memory results.
- [ ] Installation, driver, and failure-mode findings.
- [ ] Maintainability assessment and known backend limitations.
- [ ] Remaining .NET/moonlib dependency inventory.
- [ ] Artifact locations and exact reproduction commands.
- [ ] Any proposed changes to `docs/FRESH_PLAN.md`, package dependencies,
      platform claims, and the `0.1.0` milestone.

The report must distinguish tests that ran from tests that were skipped. It
must not claim general NVIDIA support from one GPU without stating the tested
hardware and driver.

## 19. Decisions Needed Before or During the Prototype

Recommended defaults are included so work can begin without resolving every
later product question.

### P1: Initial Hardware and Toolchain

**Recommended:** one available Linux x86-64 NVIDIA development machine and one
supported Lunarscout Python minor version. Pin the full environment after the
minimal smoke kernel succeeds.

- [ ] Record the selected machine, GPU, driver, Python, Numba, and CUDA versions.

### P2: Performance Threshold

**Recommended:** measure the baseline first, then decide the maximum acceptable
warm throughput regression and first-use delay before optimization begins.

- [ ] Record acceptable performance and memory ratios at Gate D.

### P3: Scientific Error Budget

**Recommended:** derive the initial budget from current reference-versus-ILGPU
disagreement and validate it using downstream lighting and visibility effects.

- [ ] Approve angular and downstream classification thresholds at Gate C.

### P4: Near-Field Reference Merge

**Recommended:** test it after the core hierarchy-enabled kernel. Determine from
the production configuration and parity results whether it is required for the
replacement or may remain deferred.

- [ ] Record whether near-field merge is part of the replacement contract.

### P5: CPU Fallback

**Recommended:** require CPU code for preprocessing and tests, but defer a
production-speed CPU horizon backend. If the Numba port succeeds, separately
decide whether systems without NVIDIA GPUs receive a slow fallback or an
explicit unsupported-capability error.

- [ ] Record the first-release behavior on systems without NVIDIA CUDA.

### P6: Adoption Boundary

**Recommended:** keep the implementation private and preserve the existing
public API. Select the backend internally only after Gate E passes.

- [ ] Decide whether the first adopted version replaces moonlib immediately or
      ships both backends for a bounded comparison period.

## 20. Overall Progress Checklist

- [x] Phase 0: baseline inventoried and frozen.
- [x] Phase 1: independent oracles and fixtures established.
- [ ] Phase 2: Python/device data contract validated.
- [ ] Phase 3: host geometry and segment generation validated.
- [ ] Phase 4: CUDA kernel passes correctness gates.
- [ ] Phase 5: performance and resource evaluation passes.
- [ ] Phase 6: production pipeline prototype passes.
- [ ] Phase 7: packaging and operational evaluation passes.
- [ ] Final evaluation report completed.
- [ ] Replacement decision recorded.
