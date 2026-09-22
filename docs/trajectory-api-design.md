# Trajectory Planning: Public API Design and Implementation Plan

Status: design + plan (no implementation yet).  
Date: 2026-09-07.

This document records the current design decisions for the public-facing rover
trajectory-planning API being added to Lunarscout and lays out the phased
implementation plan.

The broader problem-space discussion lives in:

- `docs/trajectory-planning.md`
- `docs/deepseek-trajectory-planning-review.txt`
- `docs/gridrunner-summary.md`
- `docs/old_code_gridrunner/`

Only the public API is a compatibility promise. Internal module layout,
compiled representations, block sizes, scheduler state, predecessor encoding,
cache layout, CUDA data structures, label storage, dynamic discretization, and
other implementation details may change. Names beginning with `_` are private.

The initial compatibility promise is deliberately small. Static planning is
specified first. Dynamic, power-aware, safe-interval, science-Xtarget, and
stochastic planning APIs remain provisional until their scientific and
algorithmic semantics are sufficiently mature.

---

## 1. Design principles

### 1.1 Small number of planner families

The trajectory subsystem should cover the main lunar rover mission-planning
cases with a small number of reusable algorithm families rather than a separate
planner for every combination of mission objective, environmental model, and
rover execution model.

The important axes are:

1. mission objective:
   - point-to-point travel;
   - travel-time / accessibility fields;
   - science-target, science-region, and activity planning;
2. environment model:
   - static terrain;
   - time-varying constraints such as sunlight and communications;
   - time-varying constraints with explicit power / battery state;
3. rover execution model:
   - deterministic nominal behavior;
   - uncertain performance;
   - faults, delays, or degraded modes.

The deterministic raster planners are the foundation. Stochastic performance
and fault analysis should normally wrap or repeatedly invoke deterministic
planning rather than require completely separate low-level raster engines.

Different algorithms may solve the same conceptual planning problem. In
particular, time-dependent planning may eventually include both:

- GridRunner-derived hierarchical/block-parallel propagation; and
- safe-interval planning.

These are alternative supported algorithms rather than an exploratory
benchmark in which one must necessarily replace the other.

### 1.2 Separate problem concepts from engine representation

The public API should expose physical and mission concepts:

- traversability;
- georeferencing;
- terrain elevation;
- mobility / travel-time models;
- dynamic environmental constraints;
- physical environmental signals;
- rover power state;
- battery behavior;
- trajectories and costs.

It should not expose implementation choices such as:

- GPU block dimensions;
- horizon-tile dimensions;
- predecessor direction encodings;
- integer epoch indices;
- active-block queues;
- CUDA kernel state layouts;
- single-label storage layouts;
- bounded multi-label storage layouts;
- safe-interval internal representations.

Those are private engine details.

### 1.3 CPU reference implementations first

Each major planner should have a deterministic CPU implementation suitable for
unit tests and correctness oracles before introducing performance-oriented
Numba or CUDA implementations.

For static planning, the CPU reference algorithms are ordinary Dijkstra and A*.

For dynamic planning, small exact CPU reference problems should use a
straightforward time-expanded or interval-based search whose implementation is
independent of the GridRunner engine.

For battery/SOC planning, small exact CPU reference problems should use a
multi-label resource-constrained search. This exact solver is an oracle for
testing approximate production algorithms; it is not required to scale to
mission-sized raster problems.

### 1.3.1 Small problems for testing

The GPU implementations are intended to support problems large enough for real
mission design. Some algorithms will not be able to execute on the CPU at that
scale.

For development and oracle-based testing, define a small collection of
canonical synthetic problems. Typical sizes are:

- one 128 by 128 pixel patch;
- a 3 by 3 arrangement of 128 by 128 pixel patches;
- much smaller hand-constructed grids for exact dynamic and SOC tests.

In addition to standard pytest-based testing using synthetic examples, retain a
second level of testing using real DEMs.

These should be called `examples` rather than ordinary tests and need not run
during the normal test suite. Their results should be compared with recorded
expected behavior and, where appropriate, with results from multiple planner
implementations.

### 1.4 Static and dynamic factors remain conceptually separate

Static terrain factors and dynamic environmental factors are distinct even when
a planner ultimately consumes a combined configuration-space representation.

The intended dynamic implementation may maintain separate caches for:

- static terrain tiles;
- dynamic environment tiles;
- combined configuration-space tiles.

That cache decomposition remains behind the configuration-space provider and is
not part of the public compatibility contract.

### 1.5 Separate algorithm semantics from compute backend

The planner algorithm determines the mathematical or approximate planning
semantics.

The compute backend determines where an implementation of that algorithm runs.

These concepts must not be conflated.

For example:

- an exact multi-label SOC algorithm may initially be available only on CPU;
- a greedy single-label SOC algorithm may have CPU and CUDA implementations;
- `backend="auto"` for the greedy algorithm may choose CPU or CUDA;
- requesting `backend="cuda"` for an algorithm that has no CUDA implementation
  should raise a structured unsupported-backend error;
- `backend="auto"` may choose CPU when the selected algorithm is CPU-only.

Thus a user conceptually selects:

```text
planning algorithm
    +
execution backend supported by that algorithm
```

rather than expecting every algorithm to exist on every backend.

Two different algorithms may legitimately return different answers. CPU and
CUDA implementations of the *same algorithm* should implement the same
planning semantics apart from documented numerical tolerances.

### 1.6 Exact and approximate planners are both legitimate

Some mission-scale dynamic and power-aware planners will require
approximations.

Approximation is permitted, but it must be explicit.

Examples include:

- greedy single-label SOC planning;
- bounded multi-label planning;
- bounded-suboptimal search;
- limited block expansion;
- early termination;
- approximate environmental interpolation.

An approximate algorithm should be compared against exact small-problem
reference solvers and should document what optimality, completeness, or
feasibility guarantees it does and does not provide.

---

## 2. Data contract

### 2.1 Grid and validity

Trajectory operations use ordinary NumPy arrays plus a `GeoReference`, matching
existing Lunarscout conventions.

Typical inputs are:

- `traversable`: `np.ndarray` of shape `(height, width)`, Boolean or integer;
  true / non-zero means the rover may occupy the cell;
- `georef`: `ls.GeoReference` describing the grid;
- `valid`: optional Boolean array of the same shape; false cells are treated as
  unavailable regardless of payload values;
- `elevation`: optional co-registered elevation raster when the travel model
  depends on signed slope.

Grid compatibility is never inferred from array shape alone. Co-registered
inputs must correspond to the same `GeoReference` grid.

Payload zeros and NaNs are not used as implicit validity sentinels unless a
specific API explicitly defines them that way.

Invalidity, non-traversability, and transition infeasibility are distinct
concepts:

- an invalid cell has no scientifically usable input value;
- a non-traversable cell may contain valid scientific data but cannot be
  occupied under the current static planning problem;
- an infeasible edge connects cells that may individually be occupiable but
  cannot be traversed under the selected mobility model.

### 2.2 Coordinates

Raster locations are represented internally and in low-level public APIs as
integer pixel coordinates:

```python
(x, y)
```

where `x` is column and `y` is row.

Public planning functions also accept `LonLat` as a convenience. Conversion to
pixel coordinates uses the supplied `GeoReference`.

Before Phase 1 is considered complete, the `LonLat` conversion contract must
pin the exact cell-selection convention, including:

- whether the containing pixel or nearest pixel center is selected;
- behavior for a coordinate exactly on a pixel boundary;
- handling outside the raster extent.

The resulting cell is validated for bounds, input validity, and
traversability.

### 2.3 Time

Static planning has no time coordinate.

Dynamic public APIs use UTC-aware datetimes or an explicit Lunarscout time-axis
object. Public provider interfaces do not expose integer epoch numbers.

The planner may internally map times to integer sample or interval indices for
efficient execution.

Physical trajectory time and environment sample index are conceptually
different quantities.

A rover movement may have an arbitrary physical duration and may cross one or
more environmental sample boundaries.

The current mission-analysis convention of approximately two-hour environment
sampling is an implementation/configuration choice, not a permanent public API
constraint.

Detailed time semantics are a design gate for dynamic planning and are defined
further in Section 7.

### 2.4 Distances, CRS requirements, and units

All travel distances are physical projected-plane distances derived from the
`GeoReference` affine transform so rotated, skewed, and anisotropic grids are
handled correctly.

Trajectory operations expressed in metres or metres per hour require:

- a projected CRS;
- known linear axis units; and
- a defined conversion from those CRS units to metres.

Affine distances are first interpreted in the CRS's declared linear units and
then converted to metres.

A geographic/angular CRS, a CRS without usable linear-unit metadata, or another
grid for which physical metre distances cannot be determined raises a
structured trajectory input error.

No trajectory planner infers that coordinate values are metres merely because
their numerical magnitude appears appropriate.

Travel costs are expressed as physical travel time in hours unless an API
explicitly states otherwise.

A* heuristics must remain admissible under the selected travel model.

### 2.5 Infeasible transition representation

The canonical numerical representation of an infeasible movement edge is
positive infinity:

```python
np.inf
```

Conceptually:

```text
edge_travel_time(from_cell, to_cell) =
    finite positive time      if the transition is feasible
    +infinity                 if the transition is infeasible
```

Dijkstra, A*, relaxation kernels, and related algorithms do not relax
`+infinity` transitions.

This convention is deliberately separate from cell validity:

- invalid cells are rejected or masked before transition evaluation;
- non-traversable cells cannot be occupied;
- an otherwise valid pair of cells may still have an infinite transition cost.

Travel-time result fields also use `np.inf` for valid raster cells that are not
reachable from the origin.

A planner must never rely on accidental floating-point overflow to create the
infinite sentinel. The travel model returns or assigns it deliberately when a
transition is physically disallowed.

---

## 3. Static mobility model

### 3.1 Physical factors

For the intended near-term rover models, short-term achievable rover speed may
depend on:

- signed terrain slope in the direction of travel;
- rover velocity direction relative to the Sun vector;
- estimated hazard frequency or hazard burden.

Orbital DEM roughness is not treated as a useful proxy for hazard frequency and
is not part of the baseline model.

Additional mobility factors may be added later if justified by rover design or
terrain knowledge.

Uncertain slip, uncertain delay, faults, and other stochastic effects are
important for long traverses, but they do not make the basic deterministic
planner probabilistic. Those uncertainties are handled by later robustness,
simulation, or replanning layers.

### 3.2 `SlipFunction`

`SlipFunction` is retained as the public name for the deterministic
slope-dependent mobility component.

Slip is a function of signed along-track slope, not raw elevation difference.
The same elevation change over different horizontal distances must not imply
the same mobility effect.

Conceptually:

```python
signed_slope = (z_to - z_from) / horizontal_distance
```

The initial interface is:

```python
@dataclass(frozen=True)
class SlipFunction:
    def factor(self, signed_slope: float) -> float:
        """Return a multiplicative travel-time factor for signed slope."""
```

Positive slope denotes uphill travel and negative slope downhill travel.

The exact parameterization and interpolation rules remain a design question. A
piecewise-linear representation is expected initially.

Although the public conceptual interface is scalar, arbitrary user Python
callbacks should not be executed for every movement inside Numba or CUDA hot
loops. Built-in slip models should be representable by parameters, tables, or
compiled specialized functions.

A selected `SlipFunction` may also declare slopes for which movement is
infeasible. Such a transition has travel time `np.inf`.

### 3.3 `StaticTravelModel`

The static travel model represents the physical travel-time calculation used by
static point-to-point planning and static travel-time fields.

Initial form:

```python
@dataclass(frozen=True)
class StaticTravelModel:
    speed_m_per_h: float = 36.0
    include_diagonals: bool = True
    slip: SlipFunction | None = None
```

For a feasible move from one neighboring raster cell to another:

```text
base_time = physical_step_distance_m / speed_m_per_h
travel_time = base_time * slip.factor(signed_slope)
```

when a slip function is present.

A transition rejected by traversability, slope limits, the mobility model, or a
future hard edge constraint has:

```text
travel_time = +infinity
```

The public model should remain extensible enough to incorporate additional
deterministic spatial mobility factors later, especially hazard-frequency
estimates, without requiring users to replace the planner.

Risk-related inputs may eventually include:

- deterministic hazard penalties;
- hazard-frequency rasters;
- terrain-class-dependent penalties;
- uncertainty or probability fields.

These extension points need not be implemented in Phase 1.

Arbitrary Python per-cell callbacks should be avoided in the hot-path API
because they are difficult to support efficiently in Numba and CUDA. Built-in
model objects should instead be compilable into arrays, constants, lookup
tables, or specialized kernels.

### 3.4 Neighbor and diagonal semantics

Phase 1 uses a regular raster graph with either:

- four-neighbor connectivity; or
- eight-neighbor connectivity.

When `include_diagonals=True`, diagonal step distance is calculated from the
actual affine basis, not assumed to be `sqrt(2)` times a cardinal step.

Before Phase 1 is frozen, the plan must explicitly define diagonal corner
semantics.

In particular, it must state whether movement from `(x, y)` to
`(x + 1, y + 1)` is permitted when one or both adjacent cardinal cells are
non-traversable.

The default should be scientifically deliberate rather than an accidental
property of the search implementation.

Future footprint-aware, any-angle, or continuous planners may use different
movement primitives without changing the semantics of the initial static
raster planner.

---

## 4. Static planning API

### 4.1 Namespace

Trajectory functionality lives under:

```python
import lunarscout as ls

ls.trajectory
```

The initial public functions are accessed through the subpackage rather than
re-exported from the package root.

Importing `lunarscout.trajectory` must not:

- initialize CUDA;
- import Numba unnecessarily;
- load SPICE kernels;
- open rasters;
- perform network access.

### 4.2 Path results

The normal public point-to-point result contains only mission-relevant output,
not search-engine state.

```python
@dataclass(frozen=True)
class PathResult:
    reachable: bool
    travel_time_hours: float | None
    path: NDArray[np.int64] | None
```

`path` has shape `(N, 2)` and contains `[x, y]` raster cells from start to goal.

If the goal is valid but unreachable:

```python
reachable == False
travel_time_hours is None
path is None
```

If `start == goal`, the result is:

```python
reachable == True
travel_time_hours == 0.0
path == [[x, y]]
```

subject to ordinary validation of that cell.

Predecessor arrays, direction encodings, queue state, and label state are
private implementation details and are not part of `PathResult`.

The Phase-1 result is deliberately a raster-cell path. Future any-angle,
continuous, rover-footprint, or kinodynamic trajectory types are not required
to use this representation.

### 4.3 Static travel-time fields

A pure Boolean reachability flood is not useful enough for mission planning to
justify being the primary static all-destinations API.

The useful operation is a minimum-travel-time field from one origin.

```python
@dataclass(frozen=True)
class TravelTimeResult:
    travel_time_hours: NDArray[np.float64]
    reached: NDArray[np.bool_]
    georef: GeoReference
    start: tuple[int, int]
```

Unreached cells contain:

```python
np.inf
```

in `travel_time_hours`.

The initial function is:

```python
def static_travel_time(
    traversable: NDArray[Any],
    georef: GeoReference,
    start: tuple[int, int] | LonLat,
    *,
    valid: NDArray[np.bool_] | None = None,
    elevation: NDArray[Any] | None = None,
    model: StaticTravelModel | None = None,
) -> TravelTimeResult:
    """Return minimum static travel time from start to every reachable cell."""
```

This is conceptually a single-source shortest-path computation, typically
Dijkstra or an equivalent label-correcting implementation.

It is useful for:

- static mission accessibility analysis;
- target-to-target travel-time precomputation;
- reverse heuristic fields;
- validating block-parallel implementations;
- supporting later science-target/orienteering planners.

A separate Boolean-only reachability function may be added later if a concrete
use case appears, but it is not required for Phase 1.

### 4.4 Static point-to-point path

```python
def static_path(
    traversable: NDArray[Any],
    georef: GeoReference,
    start: tuple[int, int] | LonLat,
    goal: tuple[int, int] | LonLat,
    *,
    valid: NDArray[np.bool_] | None = None,
    elevation: NDArray[Any] | None = None,
    model: StaticTravelModel | None = None,
) -> PathResult:
    """Return the minimum-travel-time static path from start to goal."""
```

The normal implementation is A*.

`elevation` is required when the selected travel model requires slope.

The heuristic is based on an optimistic physical travel-time lower bound. For a
constant maximum possible speed this is:

```text
affine-aware physical distance to goal / maximum possible rover speed
```

If the slip model can reduce travel time below the nominal base value, the
heuristic must account for the minimum possible slip factor.

If no nonzero admissible heuristic can be derived safely for a model, the
planner may use zero, reducing A* to Dijkstra behavior.

### 4.5 Search diagnostics

Search internals are deliberately omitted from the normal result types.

If later debugging or research workflows require predecessor fields, expanded
node counts, frontier statistics, block-activation histories, heuristic
statistics, or label counts, expose those through a separate diagnostic API or
optional diagnostic result object rather than extending `PathResult` with
engine-specific fields.

### 4.6 Tie behavior

The public static planner promises the optimal travel cost under the selected
model.

It does not necessarily promise an identical raster-cell path when multiple
equal-cost optimal paths exist.

Different valid implementations, priority-queue ordering, Numba compilation,
or future parallel algorithms may choose different equal-cost paths.

Tests that require deterministic tie behavior should construct cases where a
specific tie-breaking contract is intentionally being tested.

---

## 5. Dynamic environment provider architecture

The dynamic planners require environmental information that changes with time.

The public interfaces should describe spatial and temporal data requirements
without exposing the planner's internal tiling or epoch representation.

### 5.1 General provider rule

Public providers should accept physical/public coordinates and UTC times or
requested raster windows.

They should not require callers to know:

- internal GPU block dimensions;
- horizon tile dimensions;
- cache tile dimensions;
- integer epoch indices;
- internal padding rules.

The engine is free to adapt these providers to whichever block shape is most
efficient for CPU or GPU execution.

Current Lunarscout lightmap and Earth-elevation products used for trajectory
planning are expected, in practice, to provide complete values over the
planning region. The initial provider interfaces therefore do not add a
separate per-pixel validity return merely for trajectory planning.

Provider failures, missing required source products, missing horizon data, and
other inability to supply requested environmental data are reported as
structured errors rather than silently converted to an environmental value.

### 5.2 `SunVectorProvider`

A single Sun-vector abstraction isolates trajectory planning from direct SPICE
usage.

Conceptually:

```python
class SunVectorProvider:
    def vectors(
        self,
        times: TimeRange | Iterable[datetime],
    ) -> NDArray[np.float64]:
        """Return Moon-ME Sun vectors in meters with shape (time, 3)."""
```

A standard Lunarscout implementation may use the existing SPICE-backed body
vector machinery and cache results.

Callers may provide explicit vectors instead.

The trajectory layer itself does not call SpiceyPy directly.

### 5.3 `SunlightProvider`

A sunlight provider supplies solar illumination over a requested raster region
and time or set of times.

The initial conceptual interface may remain close to the existing lightmap
representation:

```python
class SunlightProvider:
    def read(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
        time: datetime,
    ) -> NDArray[np.uint8]:
        """Return solar-fraction bytes for the requested raster window."""
```

A built-in implementation may generate requested values from precomputed
horizons plus Sun vectors and may internally cache horizon-sized or
planner-sized tiles.

Horizons are currently a required precomputed input. Horizon generation is not
a hidden side effect of trajectory planning.

The implementation may also provide internal batched operations over multiple
times when that is more efficient. In particular, horizon-backed evaluation
should be free to load one horizon tile and evaluate multiple requested times
before releasing it.

The single-time public conceptual interface does not require the underlying
implementation to perform one expensive horizon load for every individual
time query.

### 5.4 Earth communications provider

Communications visibility is kept distinct from sunlight.

The provider exposes a physical quantity; threshold ownership belongs to the
specific planning problem.

Preferred initial design:

```python
class EarthElevationProvider:
    def read(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
        time: datetime,
    ) -> NDArray[np.float32]:
        """Return Earth elevation above local terrain horizon in degrees."""
```

A higher-level configuration-space model may then apply a flight rule such as:

```text
Earth elevation > 2 degrees
```

when the current planning problem requires that condition.

Other planning problems may permit operation during Earth outages and therefore
use the same signal without treating it as an occupancy constraint.

### 5.5 `ConfigurationSpaceProvider`

`ConfigurationSpaceProvider` is a general abstraction for the *problem-specific
hard occupancy constraints* of a dynamic planner.

It is not a universal claim that sunlight or communication loss are always hard
constraints.

Conceptually:

```python
class ConfigurationSpaceProvider:
    def read(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
        time: datetime,
    ) -> NDArray[np.bool_]:
        """Return whether the rover may occupy each requested cell at time."""
```

A standard Lunarscout implementation may combine whichever conditions are hard
constraints for the current planning problem, including:

- static traversability;
- required sunlight under flight rules that prohibit operation in shadow;
- required Earth visibility under flight rules that prohibit an outage;
- temporary exclusion zones;
- thermal or operational exclusions;
- other mission-specific occupancy rules.

For a different planning problem:

- darkness may be allowed and handled through battery state;
- Earth outage may be allowed and handled through rover operational mode;
- those signals therefore need not appear in the configuration-space mask.

Internally, a provider may maintain separate LRU caches for:

- static tiles;
- dynamic environmental tiles;
- combined configuration-space tiles.

Those cache structures, tile dimensions, and keys are private implementation
details.

### 5.6 Environmental signals versus hard constraints

The trajectory architecture distinguishes:

```text
physical environmental signal
```

from:

```text
problem-specific hard constraint derived from that signal
```

For example, Sun fraction can be used in any of the following ways:

- converted to `False` in configuration space because the current flight rules
  prohibit shadow driving;
- retained as a continuous input to a battery model;
- used to alter achievable speed;
- ignored entirely by a purely static planner.

Likewise Earth elevation may be:

- thresholded into a hard communication requirement;
- used only by a higher-level activity model;
- ignored for a traverse that permits autonomous operation during outage.

---

## 6. Dynamic travel-time model

### 6.1 Required physical factors

Dynamic short-term travel time may depend on:

- signed terrain slope;
- rover velocity vector relative to the Sun vector;
- estimated hazard frequency;
- additional future rover-specific factors if justified.

The public conceptual model must not assume that eight direction costs are
constant across an entire map or that they are indexed only by an integer
epoch.

### 6.2 Public conceptual interface

The conceptual operation is equivalent to:

```python
travel_time(
    x: int,
    y: int,
    dx: int,
    dy: int,
    time: datetime,
) -> float
```

with:

```text
finite hours  -> feasible movement
+infinity     -> infeasible movement
```

This does not imply that Python callbacks are executed for every rover step.

Built-in travel models should be compiled or transformed into efficient arrays,
tables, or specialized kernels before entering the hot loop.

For a sufficiently small planning region or a rover model where spatial
variation is negligible, an implementation may optimize the model to eight
per-time direction values.

That representation is private and is not the public provider contract.

### 6.3 Hazard frequency and risk extension points

Hazard frequency should be modeled explicitly when estimates are available.

DEM roughness is not substituted for hazard frequency.

The initial API should leave room for a co-registered hazard-frequency,
hazard-penalty, or risk raster or another built-in spatial model, even if the
first implementation does not use such data.

Probabilistic risk-aware planning is not required on the early implementation
critical path.

A deterministic planner may initially consume a deterministic risk penalty or
conservative hazard mask.

Later robust or stochastic layers may interpret hazard information
probabilistically.

---

## 7. Dynamic point-to-point planning

Dynamic planning handles traverses whose feasibility or travel time changes
while the rover is moving.

Trips may span many environmental samples. They must not be collapsed into a
single target-to-target edge associated with one epoch.

### 7.1 Time semantics are a design gate

Dynamic planner implementation must not proceed until the temporal semantics are
written down precisely and accepted.

At minimum, the design must specify:

1. **Environment sample interpretation**
   - Does a value sampled at `t[i]` apply at one instant?
   - Does it apply over `[t[i], t[i+1])`?
   - Is interpolation permitted?

2. **Movement crossing environment boundaries**
   - What happens when a movement starts in one environment interval and ends in
     another?
   - Is the edge subdivided?
   - Is feasibility checked continuously, at interval boundaries, or under a
     conservative rule?

3. **Travel-time representation**
   - Movement duration is physical real-valued time.
   - It must not be forced to equal one environmental sampling step merely for
     search convenience.

4. **Waiting**
   - Whether waiting is allowed;
   - at which locations it is allowed;
   - whether it may span arbitrarily many environment intervals;
   - how configuration-space changes during a wait are handled.

5. **Move versus wait ordering**
   - whether the planner may wait before moving;
   - move and then wait;
   - or repeatedly alternate them.

6. **Boundary equality**
   - exact semantics when arrival coincides with an environmental transition.

7. **Temporal horizon**
   - behavior when the requested path would extend beyond the supplied
     environmental timeline.

8. **Provider lookup**
   - exact versus nearest lookup;
   - interpolation;
   - or interval indexing.

These choices are part of the planner's scientific semantics and must be tested
independently of the optimized engine.

Where appropriate, the design should use the same half-open interval discipline
already used elsewhere in Lunarscout:

```text
[start, stop)
```

### 7.2 Dynamic planning without explicit battery state

The state is approximately:

```text
(x, y, physical_time)
```

Typical dynamic constraints include:

- sunlight when required by the selected flight rules;
- communications when required by the selected flight rules;
- temporary exclusion zones;
- other time-dependent occupancy restrictions.

Two principal algorithm families are intended.

#### GridRunner-derived planner

The first is the GridRunner-derived hierarchical, A*-guided,
block-parallel label-correcting propagation engine.

This is intended to scale to large raster planning problems and eventually to
CUDA execution.

#### Safe-interval planner

The second is a safe-interval planning approach.

Instead of representing every location at every sampled time, it represents
intervals during which occupation or transition is allowed.

This can be advantageous when environmental constraints change relatively
infrequently compared with the underlying temporal sample rate.

Safe-interval planning is intended as an alternate library-supported algorithm,
not merely as an experiment whose purpose is to choose a single winner.

The two dynamic planners should be benchmarked against each other on:

- runtime;
- memory use;
- number of states/labels expanded;
- solution cost;
- scalability with timeline length;
- scalability with environment transition frequency;
- suitability for CPU and GPU implementation.

Both should also be checked against independent exact small-problem reference
solutions.

### 7.3 Exact dynamic reference solver

Before relying on the optimized GridRunner or safe-interval implementation,
implement a small CPU reference solver.

Candidate reference formulations include:

- explicit time-expanded Dijkstra/A*;
- a simple exact safe-interval implementation;
- another straightforward search whose correctness is easy to inspect.

The reference solver is intended for:

- small synthetic problems;
- regression tests;
- temporal-boundary tests;
- validating optimized implementations.

It is not required to support mission-scale rasters.

### 7.4 Dynamic planning with battery state

The state becomes approximately:

```text
(x, y, time, SOC)
```

The physics model adds:

- drive power;
- stationary power;
- solar generation;
- battery capacity;
- minimum allowable SOC;
- charging;
- waiting.

SOC dominance is subtle.

The first scalable production implementation should nevertheless be allowed to
use a simple, justifiable greedy single-label approximation so that a useful
power-aware planner is available before a significantly more complex
multi-label GPU design is undertaken.

The initial production algorithm must clearly identify itself as approximate
when its dominance rule can discard feasible or superior alternatives.

It should be evaluated systematically against an exact multi-label CPU
reference on small cases.

A later algorithm may use:

- an unrestricted nondominated label set for small CPU problems;
- a bounded nondominated set;
- discretized SOC;
- bucketed resources;
- another controlled approximation.

These are distinct algorithms rather than invisible changes to the compute
backend.

### 7.5 Dynamic result types

Dynamic planners are not required to return the same result type as static
planners.

A dynamic trajectory naturally carries information such as:

- position;
- UTC arrival time;
- drive intervals;
- waiting intervals;
- battery SOC;
- energy input/output;
- possibly rover mode or other state.

The dynamic result API is deferred until time semantics and the initial
state-space contracts are pinned.

No compatibility promise is made that dynamic paths will be represented simply
as an `(N, 2)` pixel array.

---

## 8. Power and battery model

### 8.1 Separation of sunlight from electrical generation

Visible solar fraction is an environmental signal.

Electrical power into the rover is a rover/model-dependent quantity.

The power model therefore requires a mapping conceptually equivalent to:

```text
planning state × Sun fraction -> watts_in
```

The planning state may eventually include information such as:

- position;
- time;
- movement direction;
- rover orientation;
- operational mode;
- deployed hardware configuration.

The initial implementation may deliberately use a simpler subset of this state.

### 8.2 `SolarPowerModel`

The conceptual interface is equivalent to:

```python
watts_in(
    state,
    sun_fraction,
) -> float
```

This does not imply an arbitrary Python callback inside a GPU kernel.

Built-in power models should be compilable into efficient constants, arrays,
lookup tables, or specialized functions.

The first model may make explicit simplifying assumptions such as:

- fixed panel geometry;
- orientation-independent collection;
- constant conversion efficiency;
- no thermal dependence.

Such assumptions must be documented rather than implied.

Later models may account for:

- solar incidence angle;
- rover attitude;
- panel articulation;
- temperature;
- degraded panels;
- charging-power limits.

Adding attitude to the physical model may require adding orientation or mode to
the trajectory state. That extension is intentionally deferred.

### 8.3 Battery model

Battery behavior is separate from solar generation.

The initial battery model should represent at least:

- usable capacity in Wh;
- minimum allowable stored energy or minimum SOC;
- initial energy/SOC;
- energy integration over arbitrary elapsed durations.

Likely future parameters include:

- charge efficiency;
- discharge efficiency;
- maximum charging power;
- self-discharge;
- temperature-dependent usable capacity;
- degradation.

The battery state transition conceptually integrates:

```text
stored_energy_next =
    stored_energy
    + electrical_energy_in
    - electrical_energy_out
```

over the actual elapsed interval.

A state transition is infeasible if the battery state violates a hard minimum
constraint.

### 8.4 Rover power consumption

Power draw should distinguish at least:

- driving power;
- stationary/idle power.

Future operational modes may add:

- science activity power;
- communications power;
- survival/heater mode;
- degraded/fault modes.

The initial point-to-point SOC planner need not implement every rover activity,
but its abstractions should not assume that there is only one universal
constant load.

### 8.5 Exact SOC reference solver

Implement an exact multi-label CPU reference solver for small problems.

Its purpose is to determine whether optimized or greedy algorithms:

- incorrectly report no route;
- discard a feasible route;
- return a substantially inferior route;
- violate battery constraints.

The exact solver is not on the mission-scale performance critical path.

It may be slow and memory intensive.

### 8.6 Initial scalable SOC approximation

The first scalable SOC planner may keep one preferred label per internal state
or cell according to a documented greedy rule.

The exact rule must be specified before implementation and tested against
counterexamples.

Possible criteria include combinations of:

- higher stored energy;
- earlier arrival time;
- lower accumulated travel time.

No greedy dominance rule should be described as exact unless it has been
proved exact for the selected physics.

### 8.7 Future multi-label algorithms

Supporting `0..K` labels per state is structurally different from one label per
state, particularly on CUDA.

The initial generic engine is **not required** to solve this harder problem
before the single-label implementation exists.

When a production multi-label algorithm is introduced, it may require:

- a different private state store;
- different kernels;
- label insertion/removal;
- dominance scans;
- overflow/eviction rules;
- different block memory layouts.

The public planning API should allow that algorithm to coexist with the simpler
single-label planner without requiring the low-level kernels to share identical
state storage.

---

## 9. Engine architecture

The intended implementation retains the central GridRunner-derived design for
the planners to which it applies:

```text
relaxation engine
    × scheduler
    × physics bundle
```

The abstraction should be reused where useful without forcing fundamentally
different algorithms, such as safe-interval planning or future multi-label
search, into an unnatural representation.

### 9.1 Generic scheduler / loop skeleton

The generic GridRunner implementation owns:

- active-region scheduling;
- local relaxation to quiescence;
- boundary-change detection;
- reactivation of affected neighboring regions;
- optional A*-style optimistic ordering;
- termination policy.

The scheduler should not know the full physical meaning of a state.

### 9.2 Specialized physics bundles

Physics-specific code owns:

- state representation;
- unoccupied / invalid state representation;
- dominance rule;
- move transition;
- wait / charge transition where applicable;
- goal test;
- projection used for scheduler priority.

Expected initial variants include:

1. static travel time;
2. dynamic space-time travel;
3. approximate dynamic space-time + SOC.

A future multi-label SOC algorithm may share parts of the scheduler but is not
required to use the same one-label relaxation storage.

### 9.3 Block geometry is private

No public API fixes a block size.

CPU and GPU implementations may select different region dimensions based on:

- shared-memory limits;
- occupancy;
- register pressure;
- cache behavior;
- terrain dimensions;
- hardware capabilities.

The current 128 by 128 horizon-tile organization does not constrain trajectory
planner block dimensions.

### 9.4 A*-guided batch scheduling

The dynamic GPU algorithm may process the best `N` active blocks or regions in
parallel.

A useful optimistic priority is:

```text
priority(P) = g(P) + h(P)
```

where `h(P)` is an admissible lower bound such as:

```text
affine-aware distance to goal / maximum possible speed
```

or a stronger precomputed terrain-only reverse travel-time field.

Block reactivation must remain possible after a block has previously been
processed.

Running a label-correcting process to exhaustion should preserve whatever
correctness properties belong to the selected state and dominance model.

Optional early termination may deliberately trade correctness or optimality for
speed and must be exposed as an algorithmic policy rather than an invisible
backend behavior.

### 9.5 CPU and CUDA specialization

The generic algorithm should be shared at the architectural level, but CPU and
CUDA kernels need not be mechanically identical.

Numba specialization may use:

- kernel factories;
- specialized device functions;
- typed state structures;
- constants captured at compile time.

The important requirement is semantic equivalence for implementations claiming
to implement the same planner algorithm.

### 9.6 State storage

The initial GridRunner engine may use one state/label per cell or per internal
space-time state.

It is not required to solve the future multi-label representation problem
before that implementation exists.

However, public APIs and scheduler abstractions should avoid making the
one-label layout a compatibility promise.

If a later bounded or exact multi-label planner requires a different private
engine or state store, that is acceptable.

---

## 10. Safe-interval planning

Safe-interval planning is a first-class alternate dynamic planning approach.

### 10.1 Motivation

A dense time-expanded representation may create many states for a location even
when its occupancy status changes only occasionally.

For a given raster cell, dynamic constraints may instead be represented by
maximal intervals such as:

```text
allowed:    [t0, t1)
forbidden:  [t1, t2)
allowed:    [t2, t3)
```

A safe-interval planner searches over those intervals rather than every sampled
time index.

### 10.2 Scope

The initial safe-interval planner targets dynamic planning without explicit
battery state.

It may consume the same:

- static travel model;
- configuration-space provider;
- physical time contract;
- start and goal coordinates.

SOC-aware safe-interval planning may be considered later but is not required
for the first implementation.

### 10.3 Common scientific semantics

Safe-interval and GridRunner planners solving the same planning problem must
share the same definitions of:

- physical travel time;
- environment interval semantics;
- traversability;
- configuration-space constraints;
- start and goal validity;
- allowed waiting;
- movement feasibility.

Algorithm choice must not silently redefine the planning problem.

### 10.4 Comparative evaluation

Both algorithms remain supported when useful.

Benchmark dimensions should include:

- raster size;
- timeline length;
- temporal sample spacing;
- number of configuration changes;
- fraction of cells whose state changes frequently;
- allowed waiting;
- path length;
- memory use;
- CPU runtime;
- eventual GPU suitability;
- solution quality.

Exact small-problem reference solutions provide a third point of comparison.

---

## 11. Future science-target, region, and activity planning

Science planning remains a separate high-level algorithm family built on raster
travel planners.

### 11.1 Science value

Science objectives may have types and each type may have a monotonically
increasing, diminishing-marginal-value function:

```text
V_k(n)
```

The marginal value of another observation of type `k` is:

```text
V_k(n + 1) - V_k(n)
```

This remains useful for point targets as well as repeated observations within a
science region.

### 11.2 Science tasks need not be points

The eventual science abstraction should not assume that every objective is one
raster cell.

A science task may represent:

- a point target;
- a polygon or region;
- a required amount of areal coverage;
- a required traverse distance through a region;
- a dwell/activity duration;
- repeated spatially separated observations;
- an instrument-specific operation.

Possible task information includes:

```text
geometry
science type
reward/value function
service duration
coverage requirement
resource requirements
time-window constraints
completion state
```

The exact public structures are deferred.

### 11.3 Static science planning

For static terrain, terrain-aware target-to-target costs can be generated using
`static_path` or `static_travel_time` and then supplied to a higher-level
solver.

Likely practical solver families include:

- large-neighborhood search;
- iterated local search;
- insertion/removal/swap/2-opt heuristics;
- beam search;
- bounded exact solvers for small validation problems.

### 11.4 Dynamic science planning

Dynamic target planning may invoke the dynamic point-to-point planners rather
than assuming one fixed target-to-target cost.

A multi-hour or multi-day transfer cannot generally be collapsed into one edge
associated with the environment at departure.

Dynamic science planning is therefore deferred until the lower-level dynamic
trajectory contracts are stable.

---

## 12. Stochastic performance, delays, risk, and faults

Performance variation and faults are treated primarily as a simulation/policy
layer above deterministic trajectory planning.

The deterministic planner remains useful even when long traverses have many
potential causes of delay.

### 12.1 Fixed-route robustness evaluation

A nominal route may be executed repeatedly under sampled variations in:

- rover speed;
- power consumption;
- solar conversion;
- slip;
- operational delays;
- hazard encounters;
- faults or degraded modes.

Useful metrics include:

- probability of reaching the goal;
- arrival-time distribution;
- probability of violating minimum SOC;
- expected energy margin;
- lower-tail mission performance;
- probability of violating a mission time window.

### 12.2 Deterministic delay models

Not every delay model needs to be probabilistic.

A deterministic planner may incorporate:

- expected slowdown factors;
- fixed operational overheads;
- conservative speed assumptions;
- deterministic hazard penalties;
- scheduled dwell times.

This permits long-range planning under simple assumptions without requiring a
probabilistic planner.

### 12.3 Adaptive replanning

When faults or deviations materially change future rover capability, use a
receding-horizon policy:

1. observe the actual rover state;
2. update the rover/environment model;
3. invoke the appropriate deterministic planner;
4. execute part of the plan;
5. realize variation, delay, or fault events;
6. repeat.

### 12.4 Reproducibility

Deterministic simulation should support reproducible counter-based random
sampling for large ensembles.

Random draws should be keyed by stable identifiers such as:

```text
global_seed
simulation_id
timestep or event index
event_type
```

so execution ordering and parallel scheduling do not change the simulated
scenario.

### 12.5 Future policy methods

Later research may consider:

- contingency trees;
- MDP methods;
- POMDP methods;
- policy search;
- reinforcement learning.

These are not prerequisites for the deterministic trajectory-planning API.

---

## 13. Error taxonomy

Add trajectory-specific exceptions to `src/lunarscout/errors.py` following
existing Lunarscout conventions.

Proposed hierarchy:

- `TrajectoryError(LunarscoutError)`
  - base trajectory-domain exception;
  - `code="trajectory_error"`;
- `TrajectoryInputError(InputError)`
  - invalid trajectory arguments;
  - `code="trajectory_input_error"`;
- `PlanningError(TrajectoryError)`
  - planner execution failure;
  - `code="planning_error"`;
- `NoPathError(PlanningError)`
  - reserved for APIs or modes that explicitly require success;
  - `code="no_path"`;
- `ConfigurationSpaceError(TrajectoryError)`
  - failure obtaining or combining dynamic occupancy data;
  - `code="configuration_space_error"`.

Additional future structured errors may distinguish:

- unsupported planner/backend combinations;
- unsupported CRS units;
- invalid time semantics;
- exhausted temporal horizon;
- power-model failure;
- resource/label-capacity failure.

Normal inability to reach a valid goal is represented by the result object and
is not an exception.

---

## 14. Implementation plan

Each phase should be independently testable and shippable where practical.

### Phase 0 -- Foundation

- Add trajectory exception types.
- Create `src/lunarscout/trajectory/` with lazy imports.
- Ensure importing the subpackage initializes neither CUDA nor SPICE.
- Add dependency-boundary tests.
- Establish algorithm/backend dispatch conventions.
- Ensure algorithm selection and backend selection remain distinct concepts.

### Phase 0.5 -- Freeze static physical contracts

Before implementing the public Phase-1 API, pin:

- projected-CRS requirement;
- CRS linear-unit conversion to metres;
- behavior for geographic/angular CRSs;
- affine-aware neighbor geometry;
- `LonLat` to cell selection semantics;
- start-equals-goal behavior;
- diagonal corner-cutting semantics;
- finite versus `np.inf` transition semantics;
- invalid cell versus non-traversable cell versus infeasible edge;
- `SlipFunction` interpolation/extrapolation behavior;
- static travel model parameter validation.

Add direct unit tests for these contracts before search-algorithm tests.

### Phase 1 -- Static CPU reference planners

Implement:

- `SlipFunction` using signed slope;
- `StaticTravelModel`, defaulting to `speed_m_per_h=36.0`;
- affine-aware neighbor geometry;
- explicit CRS-unit-to-metre conversion;
- transition infeasibility through `np.inf`;
- `static_travel_time` using Dijkstra or equivalent;
- `static_path` using A*;
- `PathResult`;
- `TravelTimeResult` including `GeoReference`.

Do not expose predecessor fields publicly.

Tests should include:

- all-traversable grids;
- barriers and unreachable goals;
- explicitly infeasible movement edges;
- diagonal versus orthogonal physical distances;
- diagonal corner rules;
- signed uphill/downhill slope behavior;
- resolution independence of slope-based slip;
- rotated affine transforms;
- skewed affine transforms;
- anisotropic grids;
- projected CRSs whose linear unit is not metre;
- rejection of geographic/angular grids for metre-based planning;
- invalid and non-traversable start/goal cells;
- `start == goal`;
- shape/grid mismatch;
- non-finite model parameters;
- deliberate `np.inf` transition costs;
- deterministic tie behavior only where explicitly required;
- comparison of A* result cost against the corresponding Dijkstra travel-time
  field.

Add a synthetic CPU-only example to `examples/` and document the API in
`docs/USER_GUIDE.md`.

### Phase 2 -- Numba CPU block engine

- Introduce private region/block decomposition.
- Port the single-label relaxation engine to Numba CPU.
- Implement active-region tracking and neighbor reactivation.
- Permit block dimensions to be selected by the implementation.
- Preserve `np.inf` edge semantics.
- Require exact or tolerance-defined parity with the Phase-1 CPU reference for
  deterministic static cases.
- Benchmark the block implementation against ordinary Dijkstra/A* rather than
  assuming the block engine is faster for every problem.

The initial engine need not implement multi-label SOC state.

### Phase 3A -- Dynamic temporal semantics and exact reference planner

Dynamic optimized implementation is blocked until the Section 7.1 temporal
contract has been completed.

Then implement a small exact CPU reference planner.

Tasks:

- define environment sample interval semantics;
- define movement across environmental sample boundaries;
- define waiting semantics;
- define end-of-timeline behavior;
- define provider time lookup/interpolation rules;
- implement a straightforward time-expanded or interval-based exact search;
- construct adversarial temporal-boundary test cases.

Use explicit environmental arrays and explicit vectors in tests so SPICE and
CUDA are unnecessary.

### Phase 3B -- Dynamic environment providers

Implement:

- `SunVectorProvider`;
- window/time-based `SunlightProvider`;
- Earth elevation / communications provider;
- general problem-specific `ConfigurationSpaceProvider`;
- private caching of static, dynamic, and combined configuration-space data;
- dynamic travel-time model;
- support for efficient internal multi-time evaluation where appropriate.

Horizons remain explicitly precomputed input.

Trajectory planning does not silently invoke horizon generation.

### Phase 3C -- GridRunner dynamic space-time planner

Implement the no-SOC dynamic planner using the GridRunner-derived engine:

- space-time physics bundle;
- UTC/public time to internal interval/index mapping;
- active-region scheduling;
- region reactivation;
- A*-guided scheduling;
- dynamic point-to-point planning.

Compare results against the Phase-3A exact reference solver on small cases.

### Phase 3D -- Safe-interval dynamic planner

Implement safe-interval planning as an alternate supported dynamic algorithm.

Use the same:

- static mobility semantics;
- configuration-space semantics;
- time contract;
- start/goal contract.

Compare:

- safe-interval planner;
- GridRunner dynamic planner;
- exact small-problem oracle.

Record which problem structures favor each approach.

Do not require one algorithm to replace the other.

### Phase 4A -- Power and battery semantics

Before SOC search optimization, define:

- battery capacity;
- minimum SOC or minimum stored energy;
- initial SOC/energy;
- drive consumption;
- stationary consumption;
- solar input model;
- mapping from planning state × Sun fraction to watts in;
- integration across arbitrary elapsed durations;
- waiting/charging transition semantics.

Document simplifying assumptions of the initial solar-generation model,
especially whether rover attitude matters.

### Phase 4B -- Exact SOC CPU oracle

Implement an exact multi-label resource-constrained reference planner for small
problems.

This solver may be slow.

Its purpose is validation, not mission-scale execution.

Build synthetic cases specifically designed to demonstrate failures of naïve
single-label dominance.

### Phase 4C -- Approximate scalable SOC planner

Implement a simple, justifiable single-label SOC algorithm.

- document the greedy selection/dominance rule;
- implement it first on CPU;
- compare against the exact Phase-4B solver;
- record counterexamples and known limitations;
- expose the algorithm as approximate when appropriate.

Do not require a production `0..K` label engine at this stage.

### Phase 5 -- CUDA block engine

Port applicable GridRunner relaxation kernels to Numba CUDA.

- select block geometry according to GPU constraints rather than horizon-tile
  dimensions;
- implement A*-guided batched active-region scheduling;
- preserve region reactivation;
- implement the same greedy SOC algorithm on CPU and CUDA where applicable;
- compare CUDA with the corresponding CPU implementation of the *same*
  algorithm;
- compare both with exact reference solvers on small cases;
- gate hardware-required tests behind existing CUDA test conventions.

Backend behavior should follow the selected algorithm:

Example:

```text
exact multi-label SOC:
    cpu   -> supported
    cuda  -> unsupported
    auto  -> CPU

greedy single-label SOC:
    cpu   -> supported
    cuda  -> supported when available
    auto  -> CUDA when available, otherwise CPU
```

A later multi-label CUDA algorithm is a separate milestone.

### Phase 6 -- Science-target / region / activity planning

- Define public science task and reward data structures.
- Support points first if useful, but design toward regions and activities.
- Use static travel-time/path planning to build target graphs.
- Add service/dwell times.
- Add diminishing-marginal-value objectives.
- Implement a practical heuristic orienteering solver.
- Add a small exact/bounded reference solver only as a development/test tool or
  optional dependency if useful.
- Extend toward coverage and repeated-measurement requirements.

Dynamic science-target planning is deferred until lower-level dynamic planner
APIs are stable.

### Phase 7 -- Ensemble simulation and adaptive replanning

- Deterministic replay with counter-based random sampling.
- Fixed-route Monte Carlo robustness evaluation.
- Deterministic delay/scenario models.
- Hazard and performance uncertainty.
- Fault and degraded-mode models.
- Receding-horizon replanning.
- Large CPU/GPU ensemble execution as appropriate.

Potential later work includes policy search, MDP/POMDP methods, and other
decision-making algorithms.

---

## 15. Testing strategy

### 15.1 Layered correctness

Trajectory tests should form a correctness ladder.

```text
simple mathematical oracle
        ↓
reference CPU planner
        ↓
optimized CPU planner
        ↓
CUDA implementation
```

Agreement between two implementations that share the same optimized engine is
not sufficient evidence of scientific correctness.

### 15.2 Static oracle tests

Static cases compare:

```text
A* point-to-point cost
```

against:

```text
Dijkstra all-destinations travel-time field
```

for the same start, goal, and travel model.

### 15.3 Dynamic oracle tests

Small dynamic cases compare optimized planners against an independently
implemented exact dynamic search.

Important adversarial cases include:

- move crosses one environment transition;
- move crosses multiple environment transitions;
- waiting is required before departure;
- waiting is required after partial progress;
- path exists only by exploiting an exact boundary time;
- apparently short path becomes forbidden midway through an edge;
- GridRunner block reactivation is required to find the best path.

### 15.4 SOC oracle tests

Small SOC cases compare the greedy algorithm with the exact multi-label CPU
oracle.

Cases should intentionally include situations where:

- earlier arrival has worse energy;
- later arrival has better energy;
- waiting changes energy ordering;
- waiting consumes energy in darkness;
- charging reverses a dominance relation;
- an apparently dominated state is the only one able to complete a future
  high-energy segment.

Finding such counterexamples is expected and does not invalidate the simple
planner; it defines its approximation boundary.

### 15.5 Algorithm-versus-algorithm benchmarks

GridRunner and safe-interval planners should be benchmarked on identical
problems.

Measure:

- answer cost;
- reachable/unreachable agreement;
- runtime;
- peak host memory;
- number of expanded states;
- number of intervals;
- number of block activations;
- sensitivity to timeline length;
- sensitivity to frequency of environmental changes.

### 15.6 Real terrain examples

Maintain representative real-DEM examples outside the ordinary fast test suite.

Record:

- input identifiers/hashes;
- planner algorithm;
- backend;
- model configuration;
- environment sampling;
- hardware/software versions;
- resulting travel time;
- path/trajectory identity or summary;
- runtime;
- memory.

---

## 16. Performance and resource policy

Mission-scale planning may involve rasters and time domains too large for
straightforward dense CPU search.

Performance work should therefore preserve Lunarscout's general bounded-resource
principles.

### 16.1 Bounded dynamic data access

Dynamic providers may cache bounded sets of:

- terrain windows;
- horizon-derived environment values;
- configuration-space windows;
- vector batches.

Cache size must not scale without bound with the full mission timeline or raster
extent.

### 16.2 CUDA resource discipline

GPU implementations should explicitly bound:

- active blocks;
- state arrays;
- scheduler batches;
- temporary buffers;
- environment batches;
- label storage for any future bounded-label planner.

### 16.3 Benchmark complete workflows

Performance reports should distinguish:

- provider/environment cost;
- CPU planner cost;
- CUDA kernel cost;
- host/device transfer;
- scheduler overhead;
- path reconstruction;
- JIT/initialization;
- warm steady-state runtime.

A faster inner relaxation kernel does not establish that the end-to-end planner
is faster.

---

## 17. Planner selection and backend selection

The eventual dynamic API should make planner algorithm and backend visibly
different choices.

Exact public spelling is deferred, but conceptually:

```python
dynamic_path(
    ...,
    algorithm="gridrunner",
    backend="auto",
)
```

or:

```python
dynamic_path(
    ...,
    algorithm="safe_interval",
    backend="cpu",
)
```

Likewise SOC planning may conceptually distinguish:

```python
algorithm="soc_exact"
```

from:

```python
algorithm="soc_greedy"
```

Supported combinations are algorithm-specific.

`backend="auto"` means:

> choose an available implementation backend for the selected algorithm.

It does **not** mean:

> silently switch to a different planning algorithm.

If the selected algorithm has only a CPU implementation, `auto` selects CPU.

If the user explicitly asks for an unsupported combination such as:

```text
algorithm = exact multi-label SOC
backend   = CUDA
```

the planner raises a structured error.

---

## 18. Open questions

The following remain intentionally unresolved.

### Static design questions

1. exact parameterization and interpolation rules for `SlipFunction`;
2. exact diagonal corner-crossing rule;
3. exact `LonLat` to raster-cell selection rule;
4. representation of hazard-frequency effects in the travel model.

These must be resolved as specified by the Phase-0.5 gate before the
corresponding public behavior is frozen.

### Dynamic design questions

5. exact public dynamic trajectory/result structure;
6. full time discretization and interval semantics;
7. environmental interpolation versus piecewise-constant semantics;
8. dynamic travel-model compilation strategy for CPU and CUDA;
9. public algorithm-selection spelling;
10. final safe-interval planner API;
11. scheduler batch size and GPU block geometry.

The time semantics in items 6 and 7 must be resolved before Phase 3
implementation proceeds.

### Power design questions

12. initial solar power model;
13. whether initial solar generation depends on rover orientation;
14. initial battery efficiency assumptions;
15. exact greedy single-label SOC rule;
16. eventual bounded multi-label strategy;
17. whether a future multi-label GPU planner shares the GridRunner outer
    scheduler or uses a separate engine.

The complex multi-label production algorithm is deliberately not on the early
critical path.

### Science and simulation questions

18. eventual public target/region/activity API;
19. science coverage representation;
20. stochastic hazard and delay model interfaces;
21. eventual diagnostic API for search statistics;
22. future policy-planning APIs.

---

## 19. Current compatibility boundary

The first compatibility promise should be deliberately small.

Phase 1 should commit only to:

- `ls.trajectory` namespace;
- array + `GeoReference` input conventions;
- projected linear-CRS requirements for physical travel;
- `LonLat` convenience coordinates;
- `SlipFunction` semantics based on signed slope;
- `StaticTravelModel`;
- default nominal speed of `36.0 m/h`;
- `np.inf` semantics for infeasible transitions;
- `static_travel_time`;
- `static_path`;
- `TravelTimeResult`, including its `GeoReference`;
- `PathResult`;
- trajectory exception types.

The following explicitly remain private and changeable:

- predecessor storage;
- neighbor-direction encoding;
- block dimensions;
- scheduler implementation;
- cache tiling;
- dynamic epoch/index representation;
- Numba/CUDA state layouts;
- dynamic trajectory representation;
- SOC label representation;
- safe-interval internal representation;
- multi-label state storage;
- GPU label-storage strategy.

The following are future public capabilities but are not part of the Phase-1
compatibility promise:

- dynamic GridRunner planning;
- safe-interval planning;
- exact dynamic oracle planning;
- battery/SOC planning;
- exact multi-label SOC reference planning;
- science-target/region/activity planning;
- stochastic robustness simulation.

This keeps the initial user-facing API centered on stable mission-planning
concepts while leaving the implementation free to evolve as the GridRunner,
safe-interval, SOC, CPU, and GPU planners are developed.

---

## 20. Summary of intended planner families

The long-term trajectory subsystem should approximately have the following
structure:

```text
STATIC DETERMINISTIC PLANNING
    static_travel_time
        Dijkstra/reference
        optional optimized block implementation

    static_path
        A*

DYNAMIC DETERMINISTIC PLANNING
    exact small-problem reference
        time-expanded / interval-based CPU search

    GridRunner dynamic planner
        CPU
        CUDA

    safe-interval planner
        initially CPU
        future optimization as justified

POWER-AWARE DYNAMIC PLANNING
    exact multi-label reference
        CPU
        small problems only

    greedy single-label production planner
        CPU
        CUDA
        explicitly approximate where applicable

    future bounded/exact multi-label production planners
        separate algorithms
        backend support determined independently

SCIENCE / MISSION PLANNING
    point targets
    regions
    coverage
    activities
    diminishing marginal value
    route-aware orienteering

ROBUSTNESS / EXECUTION
    deterministic delay scenarios
    Monte Carlo ensembles
    faults and degraded modes
    receding-horizon replanning
    future policy methods
```

The central architectural principle is that these algorithms reuse common
Lunarscout terrain, georeferencing, horizon, lighting, temporal, SPICE, and
execution capabilities without turning the public trajectory API into a mirror
of any one CPU/GPU search engine.

The implementation should start with simple, independently verifiable planners,
then add scalable approximations and accelerators while retaining exact
small-problem references against which those later algorithms can be tested.
