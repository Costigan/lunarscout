# Trajectory Environment Provider Contract

Status: accepted Phase 3B provider-foundation contract.
Date: 2026-09-07.

This record freezes the public provider signatures, lookup behavior, lifecycle,
and errors used by the first dynamic trajectory integrations. See the
[trajectory API design](trajectory-api-design.md) for rationale and the larger
planner architecture. Horizon-backed signal calculation and the dynamic travel
model remain later Phase 3B work.

## Public interfaces and implementations

`SunVectorProvider`, `SunlightProvider`, `EarthElevationProvider`, and
`ConfigurationSpaceProvider` are runtime-checkable structural `Protocol`
classes. A third-party object conforms by implementing the documented members;
subclassing or registration is unnecessary.

The initial concrete adapters are:

- `ExplicitSunVectorProvider` and `SpiceSunVectorProvider`;
- `ArraySunlightProvider`, `HorizonSunlightProvider`, and
  `ArrayEarthElevationProvider`;
- `StaticConfigurationSpaceProvider`;
- `SunlightThresholdProvider` and `EarthElevationThresholdProvider`; and
- `AllOfConfigurationSpaceProvider`.

Signal and configuration providers expose a `georef` property and a scalar
`read(x0, y0, width, height, time)` operation. Sun-vector providers expose
`vectors(times)`. Signal adapters additionally expose `read_many(...)` as a
batch optimization; it has exactly the same result as stacking scalar reads in
the same order.

## Time and data

- Public scalar times and ordinary vector timestamp iterables contain
  timezone-aware `datetime` values. Values are normalized to UTC. A
  `TimeRange` remains accepted by vector providers.
- Array signal adapters receive strictly increasing UTC boundaries and a cube
  shaped `(interval, y, x)`. Interval `i` applies over the half-open range
  `[boundary[i], boundary[i + 1])`. There is no interpolation, nearest lookup,
  or extension beyond the final boundary.
- Explicit Sun vectors use exact normalized UTC timestamp lookup. Missing
  timestamps are errors.
- Sun vectors are finite, nonzero, C-contiguous `float64[time, 3]` values in
  Moon-ME metres. Sunlight windows are exactly `uint8[height, width]`. Earth
  elevation windows are finite `float32[height, width]` degrees above the local
  terrain horizon. Configuration windows are exactly
  `bool[height, width]`.
- Windows have positive integer dimensions and lie wholly inside the provider
  grid. Providers do not clip or pad requests.
- Built-in providers copy constructor inputs and return read-only independent
  arrays. Repeating an identical read returns identical values. Batch reads
  equal ordered scalar reads. Third-party providers must meet the same value,
  shape, dtype, time, and consistency rules even if their returned arrays are
  writable.

## Policies, errors, and grids

Sunlight thresholds are inclusive after converting bytes to fractions by
`value / 255`. Earth-elevation thresholds are inclusive in degrees. The
all-of provider combines one or more Boolean configuration providers using
logical AND. Every child in a composition must have the same complete raster
grid according to `same_grid`; shape equality alone is insufficient.

Invalid caller arguments raise `TrajectoryInputError` with stable
`trajectory_*` codes. A malformed result or a failure while obtaining or
combining occupancy data raises `ConfigurationSpaceError`, retaining the child
provider type and cause where available. A closed SPICE vector adapter raises
`PlanningError`. Missing signal data is never translated into darkness,
outage, or an unavailable cell.

## Caches and lifecycle

Concrete adapters use private entry-bounded LRU caches. `cache_entries=0`
disables caching. Cache keys include the normalized time, interval, and/or
window as applicable. Returned values are copied so callers cannot mutate
cached state.

Providers that own caches expose idempotent `close()` methods and support
context management. Closing clears their caches and prevents further source
reads. Threshold and composition adapters do not own their child providers, so
closing them clears only their own cache. Owners must close shared or
file-backed children explicitly.

SPICE import, kernel furnishing, and vector generation occur only when
`SpiceSunVectorProvider.vectors()` needs an uncached value. Explicit and array
providers neither import SpiceyPy nor touch the kernel pool.

`HorizonSunlightProvider` opens an explicit DEM and existing horizon directory,
loads required horizon tiles through `HorizonTileStore`, and uses the CPU
lightmap session. A batched read computes all requested times while each tile
is resident. A missing or malformed tile is a structured error. It never calls
horizon generation, and it owns neither the vector provider nor SPICE state.

## Conformance evidence

Provider conformance tests cover structural protocol checks, UTC and half-open
boundaries, exact vector timestamps, partial edge windows, dtype/shape/finiteness
validation, grid mismatches, structured child failures, inclusive thresholds,
composition, cache hits and eviction, closure, scalar/batch equivalence,
deterministic repeated reads, explicit operation without SpiceyPy, and private
exact-oracle materialization through a provider.
