# Trajectory Planning: Public API Design and Implementation Plan

Status: design + plan (no implementation yet).
Date: 2026-08-13.

This document records the decisions made in discussion, designs the
public-facing API for the rover path planners being added to lunarscout, and
lays out the phased implementation plan.  It is the working specification for
the trajectory layer; the broader problem-space discussion lives in
`docs/trajectory-planning.md`, the engine design in
`docs/deepseek-trajectory-planning-review.txt` (see the "Generic Engine"
section), and the existing prototype in `docs/gridrunner-summary.md`.

Only the public API is a compatibility promise.  Internal module layout may
change.  Names beginning with `_` are private.

---

## 1. Recorded decisions

1. **Data contract.**  Georeferenced NumPy arrays plus explicit validity masks,
   `GeoReference` for metadata and `LonLat <-> pixel` conversion.  A single
   Sun-vector provider; SpiceyPy is never called directly from the trajectory
   layer.  The provider is named **`SunVectorProvider`** (not `SunProvider`).

2. **Sunlight source / configuration space.**  Horizons are a required
   pre-computed input for now (on-the-fly generation is a later consideration,
   not yet).  The planner consumes a **configuration space provider** that
   returns 128x128 blocks of byte "rover-can-be-here" values (0 = blocked,
   non-zero = traversable), cached in a large LRU cache.  Those blocks are
   produced by thresholding 128x128 byte sun-fraction blocks (0 = no sun to
   255 = full sun) against a sun threshold (e.g. 50%).  When direct-to-ground
   communications are required, the thresholded sun block is AND-ed with a
   128x128 byte block from an Earth communications provider (Earth elevation
   above horizon, thresholded, e.g. > 2 deg).

3. **Cost model / time discretization.**  Drive time is direction-dependent for
   some rover designs: the angle between the drive direction and the Sun
   vector (not compass direction) changes achievable speed.  This is
   approximated by a **travel-time table** for the 8 neighbors of a cell,
   supplied by a provider and re-generated at the same discrete times as the
   sunlight levels (2-hour steps).  Rover speed may also be sensitive to
   elevation change; this is an **optional** `SlipFunction` -- a stepwise
   function of the *signed* elevation change between the two cells.  It takes
   the two pixels' elevations as arguments; reading a slope raster is not
   sufficient because slope is unsigned and loses the direction of elevation
   change.

4. **Scope.**  Phase 1 is static reachability + static A* on CPU with no
   sunlight and no SOC, to lock the data contract, error taxonomy, and test
   oracles cheaply.  Dynamic, SOC, Numba CPU, and CUDA phases follow.

5. **Engine shape.**  One relaxation engine x {FIFO flood, N-parallel A*}
   schedulers x {reachable, time, time+SOC} physics bundles, generic at the
   scheduler/loop-skeleton level and specialized at the physics level.  See
   the review doc's "Generic Engine" section.  N-parallel A* is approximate
   only when it terminates prematurely (an optional early-exit mode), not when
   run to exhaustion; the A* queue supports block reactivation.

---

## 2. Data contract

### 2.1 Grid and validity

Trajectory operations take ordinary NumPy arrays plus a `GeoReference`, with
validity carried as an explicit boolean mask, mirroring the `map_algebra`
convention:

- `traversable` -- `np.ndarray` of shape `(height, width)`; `bool` or integer
  dtype.  True / non-zero means the rover may occupy the cell.
- `georef` -- `ls.GeoReference` describing the grid.
- `valid` -- optional `np.ndarray` of `bool`, same shape; `True` where the
  caller knows the terrain value is meaningful.  Cells that are `invalid` are
  treated as non-traversable and never expanded through.  Default: all valid.

Grid compatibility is never inferred from array shape alone; the planner checks
that `traversable`, `valid`, and any co-registered raster have shape
`(georef.height, georef.width)`.

### 2.2 Coordinates

Start and goal are given as pixel coordinates `(x, y)` with `x` the column and
`y` the row (matching `scenario.lonlat_to_dem_pixel`).  Callers convert
`LonLat` via `georef.lonlat_to_pixel(...)` and round to integer cells; the
planner validates that start/goal are in-bounds, traversable, and valid.
A convenience accepts `LonLat` and performs the conversion.

### 2.3 Time

For static planning there is no time.  Dynamic planning uses UTC `TimeRange`
and 2-hour epochs (the same convention as the existing lightmap products).
All time is UTC; there is no local-time abstraction in this layer.

### 2.4 Costs and distances

All costs are physical travel time in hours, computed from the `GeoReference`
affine transform so rotated and anisotropic grids are handled correctly
(consistent with `map_algebra/distance.py`).  Admissible heuristics use the
affine-aware physical distance.

---

## 3. Provider architecture

These providers are the target design for the dynamic phases.  Only the ones
marked "phase 1" are implemented first.

### 3.1 Block layout

Blocks are 128x128 cells, addressed by `(block_col, block_row)` in the DEM
grid, matching the horizon-tile layout (`horizon_{tile_y:05d}_{tile_x:05d}`).
Edge blocks are padded so every block is exactly 128x128.  Block-keyed
providers return a `(128, 128)` `uint8` array.

### 3.2 `SunVectorProvider`

Provides Sun position vectors; wraps the existing SPICE machinery and caches
results per time range.

```python
class SunVectorProvider:
    def vectors(self, times: TimeRange | Iterable[datetime]) -> NDArray[np.float64]:
        """Return (time, 3) Moon-ME Sun vectors in meters."""
```

Backed by `ls.body_vectors_moon_me("sun", times)` with a per-process cache.
It is the single place SPICE is reached; the rest of the trajectory layer
never imports SpiceyPy.  Callers may instead inject explicit vectors.

### 3.3 `SunlightProvider`

Provides per-block Sun-fraction bytes from pre-computed horizons + Sun vectors.

```python
class SunlightProvider:
    def block(self, block_col: int, block_row: int, epoch: int) -> NDArray[np.uint8]:
        """Return a (128, 128) uint8 sun-fraction block (0 = no sun, 255 = full)."""
```

Computed via `ls.sunlight_fraction(...)` per cell (or the lightmap kernel for
a whole block), using the 128x128 horizon tile for the block and the epoch's
Sun vector.  Epoch-to-UTC mapping is the caller's `TimeRange` (2-hour steps).

### 3.4 `EarthCommsProvider`

Provides per-block Earth-visibility bytes (Earth elevation above horizon,
thresholded).

```python
class EarthCommsProvider:
    def block(self, block_col: int, block_row: int, epoch: int) -> NDArray[np.uint8]:
        """Return a (128, 128) uint8 block; non-zero = Earth above threshold."""
```

Uses `ls.body_azimuth_elevation_over_horizon(...)` and an elevation threshold
(e.g. `> 2 deg`).

### 3.5 `ConfigurationSpaceProvider`

Produces the traversable block the planner actually consumes, and owns the LRU
cache.

```python
class ConfigurationSpaceProvider:
    def __init__(
        self,
        *,
        sun: SunlightProvider,
        sun_threshold: float = 0.5,
        comms: EarthCommsProvider | None = None,
        comms_threshold: float | None = None,
        capacity: int = 4096,
    ) -> None: ...

    def block(self, block_col: int, block_row: int, epoch: int) -> NDArray[np.uint8]:
        """Return (128, 128) uint8: 0 = rover may not be here, non-zero = may."""
```

Semantics: `block = (sun.block(...) >= round(sun_threshold * 255))`; when
`comms` is set, `block &= comms.block(...)`.  The result is cached in an LRU
cache keyed by `(block_col, block_row, epoch)`.

### 3.6 `TravelTimeProvider`

Provides the 8-neighbor travel-time table used as the move cost.

```python
class TravelTimeProvider:
    def table(self, epoch: int) -> NDArray[np.float64]:
        """Return 8 travel times (hours) for the 8 neighbors, in neighbor order."""
```

The table encodes direction-dependent speed (drive direction vs Sun vector)
and changes at the same 2-hour epochs as sunlight.  An optional `SlipFunction`
(section 4.3) may be applied by the cost model on top of this table; it needs
both pixels' elevations, so the cost model must also have access to the
elevation raster.

### 3.7 Phase-1 subset

Phase 1 does not need the block providers.  It consumes a full-grid
`traversable` mask and a static cost model directly.  The provider interface is
specified now so the static planner and the dynamic planner share one cost /
configuration-space abstraction later.

---

## 4. Public API

### 4.1 Namespace

The trajectory layer is a subpackage, `lunarscout/trajectory/`, reachable as:

```python
import lunarscout as ls
from lunarscout import trajectory          # also reachable as ls.trajectory
```

Following the `map_algebra` precedent, the subpackage holds the implementation
and entry points are reached as `ls.trajectory.static_path(...)`.  For now they
are **not** re-exported at the root; that decision is deferred.  (The root
keeps its own flat array-based functions, e.g. `ls.slope`; `map_algebra` is
not re-exported at the root either.)  The subpackage import must not
initialize CUDA, load SPICE kernels, open rasters, or perform network access.

### 4.2 Result types

```python
@dataclass(frozen=True)
class ReachabilityResult:
    reached: NDArray[np.bool_]            # (height, width); True = reached from start
    predecessor: NDArray[np.int8]         # (height, width); direction index, see below
    start: tuple[int, int]
    # convenience
    def path_to(self, cell: tuple[int, int]) -> NDArray[np.int64] | None: ...

@dataclass(frozen=True)
class PathResult:
    reachable: bool                       # whether goal is reachable
    cost: float | None                    # total travel time (hours); None if unreachable
    path: NDArray[np.int64] | None        # (N, 2) [x, y] cells, start -> goal; None if unreachable
    predecessor: NDArray[np.int8]         # full predecessor field for reuse
```

Predecessor convention: `int8` direction index into a fixed neighbor table
(8-connectivity order documented in the module), `-1` at the start cell, and a
sentinel (e.g. `-2`) at unreached cells.  `reached` is authoritative for
"reached vs not"; `predecessor` only carries the parent direction.

### 4.3 Static cost model

```python
@dataclass(frozen=True)
class StaticTravelCost:
    speed_m_per_h: float = 2000.0
    include_diagonals: bool = True
    slip: SlipFunction | None = None      # optional direction-aware time penalty

@dataclass(frozen=True)
class SlipFunction:
    # Stepwise-linear multiplier on travel time as a function of the *signed*
    # elevation change between the two pixels.  The two pixels' elevations
    # (in the vertical unit of the DEM) are supplied to the function; slope
    # is not sufficient because it is unsigned and loses whether the rover
    # climbs or descends.
    def factor(self, from_elevation: float, to_elevation: float) -> float: ...
```

Adjacent cost = `pixel_distance / speed`; diagonal = `sqrt(2) *` adjacent when
`include_diagonals`; each is optionally multiplied by `slip.factor(e_from,
e_to)`, so the cost model must also receive the elevation raster.  The A*
heuristic is `physical_distance_to_goal / speed`, the admissible lower
bound, and must remain a lower bound when `slip` is present (use the slip
factor's minimum over its domain).

A `SlipFunction` therefore implies an `elevation` raster argument on the
planning functions (section 4.4).

### 4.4 Phase-1 functions

```python
def static_reachability(
    traversable: NDArray[Any],
    georef: GeoReference,
    start: tuple[int, int] | LonLat,
    *,
    valid: NDArray[np.bool_] | None = None,
    elevation: NDArray[Any] | None = None,
    cost: StaticTravelCost | None = None,
) -> ReachabilityResult:
    """Flood from start to all reachable traversable cells (static)."""

def static_path(
    traversable: NDArray[Any],
    georef: GeoReference,
    start: tuple[int, int] | LonLat,
    goal: tuple[int, int] | LonLat,
    *,
    valid: NDArray[np.bool_] | None = None,
    elevation: NDArray[Any] | None = None,
    cost: StaticTravelCost | None = None,
) -> PathResult:
    """Lowest-travel-time path from start to goal (static A*)."""
```

`static_path` raises `TrajectoryInputError` for an out-of-bounds or
non-traversable start/goal, and returns `PathResult(reachable=False)` when the
goal is valid but unreachable.  `cost=None` means default uniform speed with
diagonals.  `elevation` is required exactly when `cost.slip` is set, and must
share the grid (`require_same_grid`).

### 4.5 Planned (later) functions

```python
def dynamic_reachability(..., provider: ConfigurationSpaceProvider,
                         start, start_time, time_range, ...) -> ...
def dynamic_path(..., provider: ConfigurationSpaceProvider,
                 start, goal, start_time, time_range, ..., power=None) -> ...
```

Signatures are deferred until the space-time and SOC physics bundles are
pinned (see open questions).  They will use the same result types and the same
two-scheduler engine.

### 4.6 Error taxonomy

Add to `src/lunarscout/errors.py` and export from the package root:

- `TrajectoryError(LunarscoutError)` -- base, `code="trajectory_error"`.
- `TrajectoryInputError(InputError)` -- invalid trajectory arguments,
  `code="trajectory_input_error"`.
- `PlanningError(TrajectoryError)` -- planning failed, `code="planning_error"`.
- `NoPathError(PlanningError)` -- reserved for strict "must reach" modes;
  `code="no_path"`.  Normal unreachability is returned as
  `PathResult(reachable=False)`, not raised.
- `ConfigurationSpaceError(TrajectoryError)` -- c-space provider failure,
  `code="configuration_space_error"`.

---

## 5. Engine design (recap)

One relaxation engine, two schedulers, three physics bundles.  See
`docs/deepseek-trajectory-planning-review.txt` "Generic Engine" for the
`make_relax_kernel` factory sketch and the physics-bundle protocol
(`UNOCCUPIED`, `dominates`, `move`, `charge`, `is_goal`, block-cost
projection).  The scheduler (FIFO flood vs N-parallel A*) and the physics
(state representation + dominance) are independent axes; the public functions
in section 4 are thin wrappers that select a scheduler and, for phase 1, the
static physics bundle.

---

## 6. Implementation plan

Each phase is independently shippable and tested through the public API in a
fresh process (per AGENTS.md completion discipline).  Ordinary tests remain
CPU-only.

### Phase 0 -- Foundation

- Add the error taxonomy to `errors.py` and export it.
- Create `src/lunarscout/trajectory/__init__.py` importing nothing heavy
  (no Numba, no SPICE, no raster I/O) and exposing the public names.
- Add `test_dependency_boundary` coverage that `import lunarscout.trajectory`
  imports no forbidden modules.

### Phase 1 -- Static engine (CPU)

- Implement `StaticTravelCost` / `SlipFunction`, neighbor tables and
  predecessor encoding, and a shared flood/A* relaxation loop.
- Implement `static_reachability` and `static_path`.
- Tests (synthetic grids, no GPU):
  - all-traversable grid: cost matches Manhattan/euclidean-with-diagonal
    expectations; diagonal cost = sqrt(2) x adjacent;
  - barrier grid: reachability stops at the barrier; unreachable goal returns
    `reachable=False`;
  - slope-style exclusion: invalid / non-traversable cells never entered;
  - rotated / anisotropic affine: physical-distance costs and heuristic
    correctness (compare against brute-force or a trusted reference);
  - invalid inputs: out-of-bounds start/goal, non-traversable start, shape
    mismatch, non-finite cost parameters, `connectivity`/dtype validation;
  - deterministic identity: same inputs -> identical path; predecessor field
    reconstructs the exact path for every reached cell.
- Add `ls.trajectory` documentation to `docs/USER_GUIDE.md` and an example
  script under `examples/` (synthetic, no GPU).

### Phase 2 -- Numba CPU two-level

- Decompose the grid into 128x128 blocks, add live-block tracking (modified
  blocks + neighbor reactivation), and port the relaxation to `@njit` so the
  CPU reference matches the GPU shape.
- Parity tests: block-engine results equal phase-1 whole-grid results on
  deterministic cases.

### Phase 3 -- Dynamic space-time (no SOC)

- Implement `SunVectorProvider`, `SunlightProvider`, `EarthCommsProvider`,
  `ConfigurationSpaceProvider` (LRU cache), and `TravelTimeProvider`.
- Implement the space-time physics bundle and the 2-hour epoch advance.
- Implement `dynamic_reachability` and `dynamic_path`.
- Tests use explicit Moon-ME vectors (no SPICE in tests) and synthetic
  horizon tiles.

### Phase 4 -- SOC

- Add the SOC physics bundle (battery, wait/charge, min-fuel tracking) and
  reconcile the dominance rule (greedy single-label first; small nondominated
  set if experiments require it).

### Phase 5 -- CUDA two-level

- Port the relaxation to Numba CUDA, add the A*-guided block scheduler with
  N-parallel batch processing and block reactivation.  Gated behind
  `LUNARSCOUT_REQUIRE_NUMBA_CUDA=1` tests.

---

## 7. Open questions

1. Exact neighbor-order and predecessor encoding (cosmetic; decide in phase 1).
2. The `SlipFunction` parameterization (values/bins) for the stepwise
   function, and whether it is stepwise-linear on signed elevation change or
   on slope direction.
3. The SOC dominance rule: confirm greedy single-label vs small nondominated
   set (deferred to phase 4, but the physics-bundle interface already isolates
   it).
5. Travel-time table shape/provider details (8 scalars vs per-cell factors).
