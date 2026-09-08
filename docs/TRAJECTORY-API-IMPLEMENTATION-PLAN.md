# Trajectory API Implementation Plan

Status: in progress. The static parts of Phase 0/0.5 and the Phase-1 and Phase-2
CPU APIs are implemented. Phase 3A has frozen occupancy/time semantics and a
private exact occupancy oracle. Phase 3B's provider foundation, compiled
dynamic mobility bundle, and explicit Sun-direction specialization are
implemented. Phase 3C's exhaustive GridRunner core and initial public dynamic
path API are implemented; the private compiled dynamic-mobility bundle is not
yet exposed by that API. Phase 3D's exact CPU safe-interval alternative is
implemented and publicly selectable. Phase 4A's public power models, immutable
energy records, source-cell movement sampling, and piecewise-constant Wh
accounting are implemented. Phase 4B's bounded exact multi-label CPU oracle,
catch-up dominance, independent replay, and adversarial/exhaustive validation
are implemented. Phase 4C's scalable approximate SOC planner is next.

Latest verification (2026-09-07): trajectory plus example tests pass (178
tests). The complete ordinary suite reports 2,008 passed, 18 skipped, and one
unrelated pre-existing failure because `tests/test_dependency_boundary.py`
expects package version `0.1.0rc3` while `pyproject.toml` declares `0.1.0rc5`.

This checklist turns the contracts and phased direction in
[`trajectory-api-design.md`](trajectory-api-design.md) into implementation work.
That document remains the source for rationale, scientific semantics, proposed
public signatures, and the compatibility boundary. This plan intentionally does
not restate that explanatory material.

The repository-wide constraints in [`ARCHITECTURE.md`](ARCHITECTURE.md) and the
public conventions in [`USER_GUIDE.md`](USER_GUIDE.md) apply to every phase.

## 1. How to use this plan

- [ ] Treat every checkbox as a separate acceptance criterion; do not mark a
  phase complete from a green aggregate test run alone.
- [ ] Resolve and record each design gate before writing production code that
  depends on it.
- [ ] Keep one implementation PR or review unit focused on one phase or a
  clearly identified subphase.
- [ ] Update this file in the same change that completes or defers an item.
- [ ] Link decision records and benchmark reports from the relevant checkbox.
- [ ] Keep unfinished future APIs private; do not export placeholders from
  `ls.trajectory`.
- [ ] At every public milestone, test the installed/public API in a fresh Python
  process and verify that import remains side-effect free.

Phase completion labels used below:

- **Contract gate:** implementation must not proceed until the checked decision
  is documented and accepted.
- **Public milestone:** the listed names become compatibility commitments.
- **Private milestone:** implementation may ship for testing without expanding
  the public compatibility boundary.
- **Hardware gate:** evidence must come from a host where the required device is
  visible; sandbox visibility is not evidence of host capability.

## 2. Planned repository layout

The exact private split may change during implementation. Any change must retain
the public/private boundary from Sections 4, 9, and 19 of the
[API design](trajectory-api-design.md).

```text
src/lunarscout/
  errors.py
  __init__.py                         # exports the trajectory namespace only
  trajectory/
    __init__.py                       # curated public trajectory API
    static.py                         # Phase-1 public models and functions
    providers.py                      # public provider protocols after Phase 3B
    dynamic.py                        # public dynamic API after its contract gate
    power.py                          # public power models after Phase 4A
    science.py                        # public science/task API after Phase 6 gate
    simulation.py                     # public robustness API after Phase 7 gate
    _validation.py                    # arrays, coordinates, models, grids, times
    _geometry.py                      # affine steps and CRS-unit conversion
    _static_reference.py             # Dijkstra and A* CPU references
    _dispatch.py                      # algorithm/backend compatibility and choice
    _block_scheduler.py               # shared private scheduling concepts
    _block_cpu.py                     # optimized Numba CPU implementation
    _dynamic_reference.py            # independent exact temporal oracle
    _dynamic_gridrunner.py            # GridRunner physics/scheduling integration
    _safe_interval.py                 # safe-interval implementation
    _environment.py                   # adapters and bounded provider caches
    _power_reference.py              # exact multi-label SOC oracle
    _power_greedy.py                 # scalable approximate SOC algorithm
    _block_cuda.py                    # CUDA implementation and sessions
tests/
  trajectory/
    conftest.py                       # reusable projected grids and small oracles
    test_import_boundary.py
    test_static_contracts.py
    test_static_reference.py
    test_static_public.py
    test_block_cpu.py
    test_dynamic_time_contract.py
    test_dynamic_reference.py
    test_providers.py
    test_dynamic_gridrunner.py
    test_safe_interval.py
    test_power_models.py
    test_power_reference.py
    test_power_greedy.py
    test_dispatch.py
    test_science_planning.py
    test_simulation.py
  trajectory_cuda/
    test_gridrunner_cuda.py
    test_power_greedy_cuda.py
examples/
  32_static_trajectory.py
  trajectory_real_terrain.py          # manual/recorded real-DEM validation
```

## 3. Cross-cutting requirements

### 3.1 Public and dependency boundaries

- [x] Export only `trajectory` from the package root; keep trajectory functions
  under `ls.trajectory` as specified in Section 4.1 of the API design.
- [x] Keep Numba, CUDA, SpiceyPy, raster opening, horizon loading, filesystem
  writes, and network activity out of `import lunarscout` and
  `import lunarscout.trajectory`.
- [x] Keep all application, database, job-runner, UI, RAG, and Lunar Analyst
  dependencies outside `src/lunarscout`.
- [x] Ensure explicit-vector/provider workflows do not import SpiceyPy or touch
  the SPICE kernel pool.
- [x] Keep block dimensions, epoch indices, predecessor encoding, queues,
  labels, caches, and device state private.
- [x] Add AST/import-subprocess checks to the existing dependency-boundary
  coverage, including assertions that `numba`, `numba.cuda`, and `spiceypy` are
  absent after importing the namespace.

### 3.2 Validation and errors

- [x] Add the trajectory hierarchy described in Section 13 of the API design to
  `src/lunarscout/errors.py`, with stable default codes and useful `details`.
- [ ] Decide and test exact error classes/codes for unsupported CRS units,
  algorithm/backend combinations, invalid temporal contracts, exhausted
  timelines, power-model failures, and bounded-resource exhaustion before those
  failures become public.
- [ ] Translate lower-layer `GeoReferenceError`, coordinate-transform, provider,
  and backend failures at the public trajectory boundary without losing the
  actionable cause in `details` or exception chaining.
- [x] Represent a valid but unreachable destination in the result object; do not
  raise `NoPathError` from the ordinary path functions.
- [ ] Validate before initializing an optional backend or asking a provider for
  expensive data.
- [ ] Test error type, stable code, and the key repair-oriented detail fields for
  every public failure mode.

### 3.3 Numerical and determinism policy

- [x] Use physical projected-plane distances from both affine basis vectors;
  never infer square, north-up, unrotated, or metre grids.
- [x] Convert declared CRS linear units to metres exactly once at the validated
  geometry boundary.
- [x] Assign `np.inf` deliberately for infeasible transitions and unreachable
  valid cells; reject accidental NaN, negative, or zero-duration edges.
- [ ] Define tolerances by quantity and algorithm comparison rather than using
  one repository-wide tolerance.
- [x] Promise optimal cost, not a unique equal-cost raster path, unless a
  specific tie rule is later added to the public contract.
- [x] Use stable row-major cell/region enumeration and explicit queue keys where
  deterministic diagnostics or benchmark identities require repeatability.
- [ ] Keep CPU/CUDA differences for the same algorithm within documented
  scientific tolerances; never use backend selection to change algorithms.

### 3.4 Resource and lifecycle policy

- [ ] Bound provider caches, active-region batches, temporary arrays, device
  buffers, and future label stores independently of total mission duration and
  raster area wherever the selected algorithm permits.
- [ ] Give provider/session objects that own files, horizon readers, SPICE state,
  or device allocations explicit `close()` and context-manager behavior.
- [ ] Define progress and cooperative-cancellation boundaries before exposing
  mission-scale planners; preserve a complete result or no result on failure.
- [ ] Separate cold initialization/JIT, provider I/O, host planning, device
  execution, transfers, scheduling, and reconstruction in performance reports.

## 4. Phase 0 -- Foundation

**Milestone:** private scaffolding plus the public namespace and errors.

- [x] Add trajectory exception classes and export the intended public exception
  names consistently.
- [x] Create `src/lunarscout/trajectory/__init__.py` with an explicit `__all__`.
- [x] Add `trajectory` to `src/lunarscout/__init__.py` without root-level
  trajectory function aliases.
- [ ] Define private `AlgorithmName`, `BackendName`, and dispatch records without
  committing future public spellings prematurely.
- [ ] Implement one dispatch validator that distinguishes unknown algorithm,
  unknown backend, unsupported combination, unavailable automatic backend, and
  explicit backend initialization failure.
- [ ] Specify that `backend="auto"` chooses only among implementations of the
  selected algorithm.
- [x] Add import-boundary and forbidden-dependency tests.
- [x] Add a fresh-process smoke test for `import lunarscout as ls`,
  `ls.trajectory`, and `import lunarscout.trajectory`.
- [x] Verify that the namespace exposes no dynamic, SOC, science, or simulation
  placeholders.

**Exit evidence**

- [x] Focused foundation tests pass on a CPU-only environment.
- [x] Import tests prove no CUDA probe, SPICE initialization, raster access,
  network access, or working-directory mutation.
- [x] The public exception hierarchy and namespace match Sections 4.1, 13, and
  19 of the API design.

## 5. Phase 0.5 -- Freeze static contracts

**Contract gate:** complete before implementing `static_path` or
`static_travel_time`.

Create a concise decision record linked from this section and from the API
documentation. It must resolve the open static questions in Section 18 and the
following implementation-critical details.

### 5.1 Coordinates, grids, and units

- [x] Freeze whether `LonLat` selects the containing cell or nearest cell
  center.
- [x] Freeze exact boundary, negative-fractional-pixel, and outside-extent
  behavior for `LonLat` conversion.
- [x] Freeze supported projected CRS categories and how `pyproj.CRS.axis_info`
  linear-unit conversion factors are validated.
- [ ] Define behavior for differing horizontal axis units, missing/zero/nonfinite
  conversion factors, engineering/local CRSs, and geographic/angular CRSs.
- [x] Freeze the elevation vertical-unit contract needed for dimensionless
  signed slope, including how callers communicate or guarantee that unit.
- [ ] Define rejection behavior for degenerate affine neighbor vectors even
  though `GeoReference` already rejects a singular transform.

### 5.2 Cells, neighbors, and transitions

- [x] Freeze four/eight-neighbor ordering for deterministic internal traversal.
- [x] Freeze diagonal corner crossing when zero, one, or both adjacent cardinal
  cells are unavailable.
- [x] Freeze whether edge feasibility checks destination occupancy only or both
  endpoints, and apply the rule consistently to every algorithm.
- [x] Define start/goal validation order for type, conversion, bounds, validity,
  traversability, and required elevation.
- [x] Freeze `start == goal` behavior after ordinary cell/model validation.
- [ ] Define accepted dtypes and values for `traversable`, `valid`, and
  `elevation`, including object, complex, non-finite, and zero-sized inputs.
- [x] Define when elevation values are required and whether non-finite elevation
  invalidates a cell or raises an input error.

### 5.3 Mobility model

- [x] Freeze `SlipFunction` construction, knot ordering, interpolation,
  extrapolation, slope-limit, and infeasibility rules.
- [x] Freeze whether a factor may be less than one and expose the minimum
  feasible factor needed by the A* heuristic.
- [x] Reject non-finite, non-positive, or otherwise nonphysical model parameters
  with stable errors.
- [x] Freeze model equality/repr behavior and whether stored NumPy parameter
  arrays are copied and made read-only.
- [x] Define the admissible heuristic calculation for every supported built-in
  model, including the zero-heuristic fallback.

### 5.4 Contract tests

- [x] Write direct geometry/unit tests before search tests.
- [x] Cover cardinal and diagonal distances on north-up, rotated, skewed, and
  anisotropic transforms.
- [ ] Cover a projected non-metre CRS and all rejected CRS categories.
- [x] Cover signed slope at multiple resolutions and in both travel directions.
- [x] Cover each coordinate boundary and diagonal-corner decision explicitly.
- [x] Cover valid/non-traversable/infeasible distinctions and deliberate
  `np.inf` transitions.
- [x] Review the frozen contract against Sections 2--4 and 18--19 of the API
  design before starting Phase 1.

## 6. Phase 1 -- Static CPU reference API

**Public milestone:** the complete Phase-1 compatibility boundary in Section 19
of the API design.

### 6.1 Models and shared mechanics

- [x] Implement immutable `SlipFunction` and `StaticTravelModel` public values.
- [x] Implement validated coordinate normalization from `(x, y)` and `LonLat`
  to an integer cell without loading SPICE.
- [x] Implement reusable affine-neighbor geometry and CRS-to-metre conversion.
- [x] Implement one transition-cost function used by Dijkstra and A*, including
  traversability, elevation, signed slope, corner, and `np.inf` rules.
- [x] Keep model compilation separate from public model construction so later
  Numba/CUDA implementations consume constants/tables rather than callbacks.
- [x] Decide and test ownership/writeability of arrays held by result objects.

### 6.2 Dijkstra travel-time field

- [x] Implement `TravelTimeResult` with float64 times, Boolean reached mask,
  original `GeoReference`, and normalized start cell.
- [x] Implement `static_travel_time` as an independently inspectable heap-based
  single-source reference algorithm.
- [x] Initialize unavailable/unreached cells consistently and ensure
  `reached == np.isfinite(travel_time_hours)` only where that equivalence is part
  of the frozen contract.
- [x] Avoid exposing predecessors, heap state, or direction encodings.
- [x] Test empty frontier, isolated start, partial reachability, complete
  reachability, and barriers.

### 6.3 A* point-to-point path

- [x] Implement `PathResult` exactly at the Phase-1 public boundary.
- [x] Implement A* with a model-aware admissible physical-time heuristic.
- [x] Reconstruct `[x, y]` cells from start through goal and validate endpoint,
  adjacency, occupancy, edge feasibility, and recomputed path cost in tests.
- [x] Return the documented unreachable result for a valid unreachable goal.
- [x] Return the one-cell, zero-hour result for a valid `start == goal`.
- [x] Do not promise a specific path for uncontracted equal-cost ties.

### 6.4 Static test matrix

- [ ] Cover all cases listed under Phase 1 and Section 15.2 of the API design.
- [x] Add randomized small-grid property tests comparing each A* cost to the
  Dijkstra field using fixed seeds and multiple affine/model configurations.
- [x] Include explicit tests for all-invalid rasters, valid zeros, Boolean and
  integer traversability, non-finite elevations, shape mismatches, exact bounds,
  and non-finite model parameters.
- [x] Test path-cost recomputation independently of A*'s accumulated result.
- [x] Run every static public call through `ls.trajectory` in a fresh process.
- [x] Confirm CPU selection does not probe CUDA.

### 6.5 Documentation and example

- [x] Add the static API to the User Guide function overview and a focused
  trajectory section that links to the design for rationale.
- [x] Add `examples/32_static_trajectory.py` using only synthetic CPU data and
  public API names.
- [x] Add the example to `examples/README.md` and the deterministic example test
  sequence.
- [ ] Show reachable, unreachable, non-metre/projected-unit, and path-coordinate
  interpretation without relying on private helpers.
- [x] Update `docs/ARCHITECTURE.md` package layers/modules only after the layout
  exists.

**Exit evidence**

- [x] Focused trajectory tests and deterministic example tests pass.
- [ ] Complete ordinary CPU suite passes.
- [ ] Fresh installed-wheel smoke coverage proves the public import and calls.
- [x] Public docs, `__all__`, signatures, error codes, and tests agree.
- [x] Only the names listed for Phase 1 in design Section 19 are public.

## 7. Phase 2 -- Optimized Numba CPU block engine

**Private milestone:** optimized static execution without changing Phase-1
scientific semantics.

- [x] Specify a private compiled model containing validated affine steps,
  metre conversion, neighbor tables, slip tables, and infeasible sentinels.
- [x] Implement private region decomposition with dimensions chosen by the CPU
  implementation, not by horizon tile size.
- [x] Implement single-label relaxation to local quiescence, boundary-change
  detection, active-region tracking, and neighbor reactivation.
- [x] Preserve the reference transition function's semantics in compiled form.
- [x] Add termination and non-convergence safeguards with structured failures.
- [x] Keep the reference Dijkstra/A* implementation available as an oracle.
- [x] Compare full fields, reachability, goal costs, and reconstructed path
  validity against the reference over canonical 128x128 and 3x3-patch cases.
- [x] Add adversarial cases where revisiting/reactivating a region is necessary.
- [x] Benchmark end-to-end reference and block workflows, including compilation,
  warm execution, scheduler overhead, and peak memory.
- [x] Add optimized public dispatch only if measured behavior justifies it and
  the selection rule can preserve the existing public contract.

**Exit evidence**

- [x] Tolerance policy and parity report are checked in.
- [x] Benchmarks state cases where the block engine does and does not help.
- [x] The ordinary suite still runs CPU-only and namespace import stays lazy.

## 8. Phase 3A -- Dynamic time contract and exact oracle

**Contract gate:** no optimized dynamic planner or public dynamic result ships
until Section 7.1 of the API design is resolved in a reviewed decision record.

- [x] Define sample meaning, half-open interval boundaries, lookup, and any
  interpolation policy.
- [x] Define continuous physical move duration and how an edge crossing one or
  multiple environment boundaries is subdivided and checked.
- [x] Define exact equality at departure, transition, arrival, and timeline end.
- [x] Define wait availability, allowed cells, duration, repeated move/wait
  ordering, and feasibility throughout a wait.
- [x] Define start/goal occupancy at requested departure/arrival times.
- [x] Define timeline exhaustion and whether extension is forbidden, delegated
  to a provider, or represented as an ordinary unreachable result.
- [x] Define provider consistency requirements for repeated and batched reads
  in [`trajectory-provider-contract.md`](trajectory-provider-contract.md).
- [x] Freeze internal oracle state dominance without making its representation
  public.
- [x] Implement an independent small exact CPU search using explicit in-memory
  environment arrays.
- [x] Extend the oracle through Phase 3B's explicit compiled dynamic mobility
  inputs without introducing SPICE or CUDA into oracle tests.
- [x] Add explicit vector-to-mobility compilation with its private local-frame
  and parameterization contract frozen in
  [`trajectory-dynamic-mobility-contract.md`](trajectory-dynamic-mobility-contract.md).
- [x] Add the occupancy and boundary adversarial cases in Section 15.3,
  including exact boundaries and multi-interval movement.
- [x] Add the GridRunner block-reactivation adversarial case in Phase 3C.
- [x] Add brute-force enumeration for tiny cases where it provides an oracle
  independent of the exact search implementation.
- [x] Keep the oracle private unless a separate public-reference use case and
  contract are approved.

**Exit evidence**

- [x] The temporal and dynamic-mobility decision records have no unresolved
  behavior needed by either GridRunner or safe-interval planning.
- [x] Oracle tests require neither SPICE, horizons, Numba, nor CUDA.

## 9. Phase 3B -- Dynamic environment and mobility providers

**Milestone:** reusable provider contracts; public exposure occurs only after
their signatures, lifecycles, and errors are frozen.

- [x] Decide whether each public provider is a `Protocol`, abstract base class,
  concrete adapter family, or combination, and document conformance testing in
  [`trajectory-provider-contract.md`](trajectory-provider-contract.md).
- [x] Implement and test `SunVectorProvider` adapters for explicit vectors and
  lazy existing SPICE-backed vector generation.
- [x] Implement the `SunlightProvider` contract, in-memory interval adapter,
  window/time reads, and equivalent batched reads.
- [x] Add a horizon-backed sunlight adapter whose private batched path reuses
  one loaded horizon tile across times.
- [x] Implement `EarthElevationProvider` separately from communications policy.
- [x] Implement composable `ConfigurationSpaceProvider` policies that convert
  selected physical signals into hard occupancy constraints.
- [x] Ensure missing data/provider failure raises a structured error rather than
  becoming darkness, outage, traversability, or invalidity.
- [x] Validate every returned window's shape, dtype, finiteness where required,
  grid identity, and time coverage.
- [x] Implement bounded, configurable private caches for static, signal, and
  combined configuration-space windows; test eviction and resource closure.
- [x] Ensure trajectory calls never generate horizons as a side effect.
- [x] Define and implement the private built-in dynamic travel model as a
  compiled model, preserving signed slope and supporting explicit deterministic
  hazard factors; see
  [`trajectory-dynamic-mobility-contract.md`](trajectory-dynamic-mobility-contract.md).
- [x] Test scalar conceptual semantics against batched/windowed execution.
- [x] Test explicit providers in a process where SpiceyPy is unavailable.

**Exit evidence**

- [x] Provider contract tests cover timestamps, partial edge windows, failures,
  cache eviction, batching equivalence, and deterministic repeated reads.
- [x] End-to-end exact-oracle cases run through providers without changing their
  expected answers.

## 10. Phase 3C -- GridRunner dynamic planner

- [x] Implement a private space-time physics bundle using the frozen temporal and
  provider contracts.
- [x] Implement UTC/public-time conversion to private interval state without
  exposing integer epochs.
- [x] Reuse active-region scheduling, local relaxation, boundary propagation,
  and reactivation only where their assumptions fit the dynamic state.
- [x] Add an admissible A*-guided region priority and record the lower-bound
  assumptions in
  [`trajectory-gridrunner-private-contract.md`](trajectory-gridrunner-private-contract.md).
- [x] Implement wait and movement transitions exactly as frozen in Phase 3A.
- [x] Define the private implementation's termination as exact/exhaustive, with
  no approximate early-stop mode.
- [x] Compare reachability, arrival/cost, and trajectory feasibility against the
  Phase-3A oracle across synthetic and randomized small cases.
- [x] Add explicit regression tests requiring region reactivation.
- [x] Freeze the dynamic public result, algorithm spelling, defaults, and
  diagnostics boundary before exporting `dynamic_path`.
- [x] Document whether the first public GridRunner algorithm is exact,
  complete, bounded-suboptimal, or otherwise approximate under each policy.

## 11. Phase 3D -- Safe-interval planner

- [x] Implement maximal allowed-interval construction using the exact Phase-3A
  boundary semantics.
- [x] Validate interval construction independently on hand-authored timelines.
- [x] Implement no-SOC safe-interval search using the same static model,
  configuration provider, start/goal rules, movement checks, and waiting rules
  as GridRunner.
- [x] Use the same public dynamic result and clearly distinct algorithm name.
- [x] Compare both optimized planners to the independent exact oracle.
- [x] Add sparse-change, frequent-change, long-timeline, exact-boundary, and
  no-wait cases.
- [x] Benchmark cost, agreement, runtime, peak memory, expanded states,
  interval count, activation count, and sensitivity dimensions from Sections
  10.4 and 15.5 of the API design; see
  [`trajectory-dynamic-benchmark.md`](trajectory-dynamic-benchmark.md).
- [x] Keep both algorithms available when they satisfy their documented
  contracts; do not silently substitute one for the other.

**Phase-3 exit evidence**

- [x] Public time, provider, dynamic result, algorithm, backend, and error
  contracts are documented and tested through `ls.trajectory`.
- [x] Explicit `backend="cpu"` never probes CUDA; `auto` does not change the
  requested algorithm.
- [x] Real-terrain/manual examples record source identities and environment
  sampling without entering the ordinary test suite.

## 12. Phase 4A -- Freeze power and battery semantics

**Contract gate:** resolve Section 8 and power questions 12--14 in Section 18
before implementing SOC search. Question 15 gates Phase 4C; questions 16--17
remain deferred until bounded production multi-label/GPU design.

- [x] Freeze units and validation for battery capacity, initial/minimum energy
  or SOC, drive load, stationary load, solar input, and elapsed time.
- [x] Freeze the first `SolarPowerModel`, including orientation assumptions,
  clipping, conversion efficiency, and invalid signal behavior.
- [x] Freeze battery charge/discharge efficiencies, capacity clipping, charging
  limits, and equality at minimum energy.
- [x] Define integration when movement/waiting crosses environmental boundaries.
- [x] Define drive, idle, charge, and any initial operational modes without
  implying a universal load.
- [x] Define state feasibility and energy tolerance policy at every transition.
- [x] Freeze public power/result structures only after their array ownership,
  units, timeline, and trajectory-event representation are testable.
- [x] Add direct energy-accounting tests independent of search.

## 13. Phase 4B -- Exact SOC CPU oracle

- [x] Implement a private exact nondominated multi-label resource-constrained
  search for small problems.
- [x] Define exact dominance over location, time, energy, and cost under the
  frozen model.
- [x] Retain enough labels to establish exactness within the declared continuous
  or discretized state contract; fail structurally if an explicit oracle bound
  is exceeded.
- [x] Validate every returned trajectory by replaying time, environment, power,
  and battery transitions independently.
- [x] Add all counterexample classes listed in Section 15.4.
- [x] Add tiny exhaustive-state comparisons that do not share the label-store
  implementation.
- [x] Keep mission-scale performance explicitly out of the oracle acceptance
  criteria.

## 14. Phase 4C -- Scalable approximate SOC planner

- [ ] Specify one greedy single-label selection/dominance rule before coding.
- [ ] State its feasibility, completeness, and optimality guarantees and known
  non-guarantees in public algorithm metadata/documentation.
- [ ] Implement the same rule first on CPU using shared energy-transition code.
- [ ] Compare every small case with the exact multi-label oracle.
- [ ] Check in minimal reproducible counterexamples where the approximation
  misses a route or returns an inferior route.
- [ ] Ensure algorithm selection visibly distinguishes exact and greedy SOC;
  backend selection must not cross that boundary.
- [ ] Expose bounded-resource exhaustion separately from ordinary no-path.
- [ ] Add replay validation for every returned power-aware trajectory.

## 15. Phase 5 -- CUDA block engine

**Hardware gate:** final acceptance requires the repository's explicitly gated
real-CUDA environment.

- [ ] Add CUDA implementation support to dispatch without importing or probing
  CUDA during namespace import or explicit CPU execution.
- [ ] Compile the same validated physics/model records used by the corresponding
  CPU algorithm into CUDA-compatible constants/tables.
- [ ] Select and benchmark trajectory-specific block geometry; do not reuse
  128x128 horizon tiles by default.
- [ ] Implement bounded active-region batches, local relaxation, boundary-change
  reporting, neighbor reactivation, and A*-guided scheduling.
- [ ] Bound device state, environment batches, transfer buffers, and scheduler
  queues and report resource failures structurally.
- [ ] Port only supported algorithms: initially GridRunner dynamic and the same
  greedy SOC semantics as CPU; keep exact multi-label SOC CPU-only.
- [ ] Make explicit unsupported CUDA combinations fail without fallback.
- [ ] Make `auto` fall back to CPU only for the same selected algorithm and only
  for capability unavailability covered by the dispatch contract; never hide
  execution/JIT/kernel failures with a CPU retry.
- [ ] Compare CUDA against the same-algorithm CPU implementation, then compare
  both against independent exact oracles on small cases.
- [ ] Add gated tests that prove device initialization and kernel execution, not
  merely simulator or compile success.
- [ ] Benchmark cold/warm complete workflows, transfers, providers, scheduling,
  reconstruction, peak host memory, and peak device memory.

**Exit evidence**

- [ ] Ordinary CPU suite passes with no visible GPU.
- [ ] Real-CUDA suite passes with `LUNARSCOUT_REQUIRE_NUMBA_CUDA=1`.
- [ ] Backend matrix documentation and structured failures match actual support.

## 16. Phase 6 -- Science target, region, and activity planning

**Contract gate:** freeze the future public structures from Section 11 before
exporting them.

- [ ] Define versioned science type, geometry/region, value function, service
  duration, coverage, resource, time-window, and completion-state contracts.
- [ ] Support point tasks first only if the public model can evolve to regions
  and activities without reinterpreting existing fields.
- [ ] Build static target-to-target costs through public static trajectory
  semantics and retain directed costs when slope makes travel asymmetric.
- [ ] Implement service/dwell times and diminishing marginal value.
- [ ] Implement a practical heuristic orienteering solver with explicit
  termination and approximation metadata.
- [ ] Add a tiny exact/bounded comparison solver as private test tooling or a
  clearly optional dependency if justified.
- [ ] Test unreachable targets, asymmetric costs, repeated science types,
  service-time exhaustion, region completion, and deterministic fixed-seed
  behavior.
- [ ] Defer dynamic science planning until lower-level dynamic APIs are stable;
  do not collapse a long dynamic transfer to one departure-time edge.

## 17. Phase 7 -- Robustness simulation and adaptive replanning

- [ ] Freeze fixed-route result metrics, event/state records, and replay format.
- [ ] Implement deterministic delay/scenario models before stochastic wrappers.
- [ ] Implement counter-based random draws keyed by seed, simulation, event
  index/time, and event type so scheduling does not change outcomes.
- [ ] Add fixed-route Monte Carlo for speed, load, solar conversion, slip,
  delays, hazards, faults, and degraded modes as each model is approved.
- [ ] Implement trajectory replay that detects time, occupancy, and SOC
  violations independently of the planner.
- [ ] Implement receding-horizon replanning with explicit observed-state and
  model-update boundaries.
- [ ] Test reproducibility across serial, batched, and parallel execution.
- [ ] Bound ensemble memory and report aggregation independently of ensemble
  size where streaming permits.
- [ ] Keep contingency-tree, MDP/POMDP, reinforcement-learning, and policy-search
  APIs deferred until separately designed.

## 18. Test fixtures, examples, and benchmark evidence

### 18.1 Canonical synthetic fixtures

- [ ] Add tiny hand-built grids for exact geometry, temporal, and SOC behavior.
- [ ] Add a deterministic 128x128 patch for reference/optimized comparisons.
- [ ] Add a deterministic 3x3 arrangement of 128x128 patches that forces
  cross-region propagation and reactivation.
- [ ] Store fixture-generation logic and expected scientific summaries; avoid
  opaque binary expected outputs where a small declarative fixture suffices.
- [ ] Record fixed random seeds and make generated cases reproducible.

### 18.2 Real-terrain examples

- [x] Keep real-DEM workflows under `examples/` or a clearly manual validation
  area, not the ordinary CPU test suite.
- [ ] Make editable DEM, horizons, output, and kernel paths explicit.
- [x] Record input hashes/identifiers, algorithm, backend, model parameters,
  environment sampling, software/hardware versions, result summary, runtime,
  and memory as listed in Section 15.6 of the API design.
- [x] Compare multiple planner implementations on identical inputs where their
  scientific contracts match.
- [ ] State NVIDIA device/driver requirements for every GPU example.

### 18.3 Required verification at each implementation milestone

- [x] Run the smallest focused tests while developing.
- [x] Run `.venv/bin/python -m pytest tests/trajectory -q`.
- [x] Run `.venv/bin/python -m pytest -q` before declaring a CPU milestone
  complete.
- [ ] Run the gated real-CUDA trajectory tests for CUDA milestones.
- [x] Run deterministic public examples affected by the change.
- [x] Run installed-package/public-API smoke tests in a fresh process.
- [x] Run `git diff --check`.
- [x] Inspect `git status --short`, including untracked fixtures and docs.
- [x] Review the actual diff for correctness, regressions, security, resource
  bounds, edge cases, maintainability, and accidental public exports.

## 19. Documentation and compatibility checklist

- [x] Keep `docs/trajectory-api-design.md` as the rationale/contract source and
  this file as the progress tracker; avoid copying long rationale between them.
- [x] Update the User Guide only for behavior that is implemented and tested.
- [ ] Update the User Guide function overview, examples index, error guidance,
  backend matrix, and implementation-maturity section at each public milestone.
- [x] Update Architecture module/layer diagrams when production modules exist,
  not in anticipation of them.
- [x] Identify each exact versus approximate algorithm and its guarantees in
  user-facing documentation.
- [ ] Document supported algorithm/backend pairs from the dispatch registry or
  another single source of truth so docs and behavior cannot drift silently.
- [ ] Maintain a compatibility table distinguishing public names, provisional
  names, private oracles, and deferred capabilities.
- [x] Do not claim test counts, performance, backend support, or completion that
  is not backed by current evidence.

## 20. Final reconciliation checklist

Before declaring the trajectory API implementation complete:

- [ ] Re-read the full [API design](trajectory-api-design.md) and reconcile
  every data contract, planner phase, test category, performance rule, open
  question, and compatibility item against the final implementation.
- [ ] Trace every public capability end to end: namespace import, validation,
  model compilation, algorithm/backend dispatch, execution, result creation,
  documentation, examples, and public fresh-process tests.
- [ ] Verify static, dynamic, SOC, science, and simulation features are described
  only to their actual implementation maturity.
- [ ] Verify optimized implementations retain independent reference oracles and
  do not test themselves solely against shared kernels.
- [ ] Verify all unresolved gates are explicitly deferred and that no public API
  depends on an unresolved interpretation.
- [ ] Verify no private representation listed in design Section 19 has leaked
  into a public signature, result, serialized record, or required provider API.
- [ ] Verify complete CPU and required hardware-gated suites, examples, diff
  checks, and status inspection have current recorded evidence.
- [ ] Obtain final API/diff review before updating the plan status to complete.
