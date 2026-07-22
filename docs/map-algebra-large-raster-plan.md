# Deferred Large-Raster Map-Algebra Execution Plan

Status: **DEFERRED by project decision**

Last updated: 2026-07-22

## Purpose

This document owns map-algebra work whose primary purpose is processing maps
that are too large to materialize comfortably in memory. It separates that
scalability work from the core scientific and public-API plan in
`docs/map-algebra-implementation-plan.md`.

Deferral means no new large-raster operation families are currently scheduled.
The bounded capabilities already implemented remain supported and tested; they
are not being removed or downgraded.

## What Counts as Large-Raster Work

This plan includes:

- window or tile execution that bounds memory independently of raster area;
- halo reads for neighborhood operations at tile boundaries;
- cross-window reconciliation of connected regions;
- streaming whole-raster and per-zone accumulators;
- bounded or explicitly approximate distance-field algorithms;
- spatial-window and time-batch execution for temporal products;
- local-kernel fusion and window/block scheduling optimizations;
- dataset/window-cache, queue, and temporary-storage resource controls;
- restart, cancellation, and progress extensions needed by those stages; and
- empirical memory, throughput, and scaling evidence for large rasters.

Eager NumPy/SciPy operations, scientific definitions, input validation,
operation metadata, structured errors, and notebook-sized examples remain in
the core plan even when a future bounded implementation may reuse them.

## Current Status

### Completed and supported

- Defensive expression planning with node, depth, and source-count limits.
- Streaming row-major output-window enumeration without an area-sized window
  list.
- Bounded source dataset and source-window caches with deterministic cleanup.
- Windowed execution for local cell-by-cell and coordinate expressions.
- One-pixel halo-aware windowed `slope`, `aspect`, and `hillshade`, with eager
  parity across internal tile boundaries.
- Explicit cross-grid `resample_to` window planning and execution, including
  conservative source-window mapping and exact nearest-neighbor 64-bit values.
- Atomic staged single-band GeoTIFF output with deterministic invalid payloads
  and a GDAL validity mask.
- Writer lifecycle control: monotonic progress, cooperative cancellation,
  compact checkpoint journal format 2, validated resume, paired TIFF/manifest
  rollback, and interrupted-publication recovery.
- Read-only planner diagnostics for total windows, estimated per-window memory,
  lifecycle capabilities, journal identity inputs, and resumable stages.
- Explicit rejection of unsupported focal, global, zonal, distance, and
  temporal nodes before bounded execution modifies output.
- Public `compute()` remains the explicit whole-raster materialization path;
  supported `write()` operations do not silently materialize their sources.

### Partial foundations

- Window dimensions are configurable with a 128-by-128 default, but are not
  selected from actual source/output block geometry.
- Planner memory estimates exist, but empirical peak-memory scaling has not
  been measured across increasing raster dimensions.
- Terrain nodes provide fixed symmetric halos; arbitrary footprint-derived and
  asymmetric halos are not implemented.
- Nodes execute within one bounded output task, but consecutive local nodes are
  not explicitly fused into a compiled or single-pass kernel.
- Scientific and restart identities exist; a distinct execution-cache identity
  and golden compatibility fixtures remain incomplete.
- Eager focal, morphology, global, zonal, distance, and temporal capabilities
  exist in useful subsets, but their general bounded executors do not.
- Temporal sources and streaming reducers exist, but the map-algebra writer
  does not schedule spatial windows and bounded time batches together.

## Deferred Work, in Dependency Order

### LR1. General halo and window foundation

- Select windows from block geometry and an explicit memory budget.
- Represent arbitrary rectangular and footprint-derived asymmetric halos.
- Propagate halos through compatible nodes and crop exactly once per operation.
- Define safe behavior for edge modes whose neighborhoods cross dataset edges.
- Add optional local-node fusion only where it preserves dtype, validity,
  overflow, and floating-point contracts.
- Complete separate scientific, restart, and execution-cache identities.
- Add a same-destination concurrency/locking contract for resumable writers.

### LR2. Focal, convolution, and morphology windows

- Execute focal sum/mean/min/max/range/std/count/median and reviewed
  percentiles window by window.
- Execute finite convolution kernels and Boolean morphology with arbitrary
  reviewed footprints.
- Preserve `invalid`, `constant`, `nearest`, `reflect`, and `wrap` edge modes.
- Preserve `require_all`, `ignore_invalid`, and `propagate_center` validity
  policies, including a completed `min_valid_count` contract.
- Compare every bounded kernel with the eager reference over non-divisible
  windows, rotated/anisotropic grids, invalid neighborhoods, cancellation, and
  resume.

### LR3. Connected-region reconciliation

- Add eager `Raster` adapters for labeling, region sizes, filtering, and
  borders only if they have not already been completed in the core plan.
- For bounded labeling, assign provisional per-window labels and reconcile
  equivalence classes where regions touch across window edges and corners.
- Support explicit four- and eight-neighbor connectivity.
- Produce deterministic final labels independent of window order and size.
- Stream region-size/filter information without retaining an unbounded
  per-pixel structure.

### LR4. Global and zonal streaming

- Implement mergeable bounded accumulators for count, sum, min, max, mean,
  variance, standard deviation, and histograms.
- Merge per-zone partial accumulators deterministically across windows.
- Preserve exact integer zone IDs, including `uint64` values.
- Define exact versus approximate memory contracts for median, percentile, and
  unique counts; never switch algorithms silently.
- Verify eager/streaming tolerance, window-order behavior, empty zones,
  all-invalid data, and accumulator overflow boundaries.

### LR5. Bounded distance fields

- Select a genuinely bounded exact Euclidean algorithm or document a tiled
  approximation with a measurable error bound.
- Preserve pixel/physical units, rotated and anisotropic affine behavior,
  invalid-output policy, and empty opposite-class errors.
- Avoid silently invoking whole-raster SciPy transforms from file-backed calls.
- Benchmark representative sparse and dense hazard masks only after scientific
  parity is established.

### LR6. Temporal spatial windows and time batches

- Classify temporal nodes as layer-wise or reducing in the execution plan.
- Select bounded spatial windows and time batches from an explicit memory
  budget.
- Bound simultaneously open datasets for long timestamped series.
- Make temporal reductions directly composable with static spatial expressions
  during windowed writes.
- Map layer-wise expressions into the existing timestamped GeoTIFF-series
  format before considering generic multiband BigTIFF output.
- Exercise a representative approximately 3,000-layer file-backed series.

### LR7. Resource and performance evidence

- Record planning, source reads, kernels, masks, compression, journal, close,
  and publication timings separately where practical.
- Measure peak RSS, dataset handles, cache occupancy, queues, temporary disk,
  and output size at three increasing raster dimensions.
- Prove that supported bounded paths scale with window/graph complexity rather
  than total raster area, aside from explicitly bounded global state.
- Establish eager-size guidance instead of guessing a universal threshold.
- Store reproducible commands, environment details, machine-readable results,
  and a short interpretation.
- Add a runnable large lunar-region example only after these claims are backed
  by measurements.

## Deferred Test Matrix

When this plan resumes, tests must cover:

- one window, many windows, non-divisible edges, and multiple window layouts;
- exact eager/bounded validity and integer parity plus stated floating tolerance;
- arbitrary halos, internal seams, rotated grids, and dataset boundaries;
- cancellation and resume at every durable stage;
- failures between value/mask writes, journal updates, and publication swaps;
- proof that checkpointed kernels are skipped and ambiguous work is recomputed;
- deterministic region/zone identity across window sizes and orders;
- all-invalid inputs, extreme signed/unsigned integers, non-finite values, and
  exact accumulator boundaries;
- bounded dataset handles and resident memory for increasing spatial and
  temporal sizes; and
- fresh-process execution through the public API.

## Acceptance Criteria

This deferred plan is complete only when:

1. Every operation advertised as file-backed has a real bounded executor or an
   explicit, documented bounded approximation.
2. Eager and bounded scientific results agree against independent references.
3. No supported bounded call silently materializes the full spatial raster or
   temporal cube.
4. Progress, cancellation, restart, overwrite, and cleanup remain correct for
   every new execution stage.
5. Resource claims are supported by reproducible empirical evidence.
6. User documentation states availability, passes, memory behavior, temporary
   storage, and limitations for every file-backed operation.

## Work That Remains in the Core Plan

The following work is not deferred by this document:

- scientific and numeric-policy consistency;
- eager API gaps and adapters that do not require bounded execution;
- canonical identity and operation-registry completeness;
- structured error normalization;
- public reference tables and notebook-sized examples;
- adversarial and boundary test coverage; and
- release reconciliation unrelated to large-raster performance claims.
