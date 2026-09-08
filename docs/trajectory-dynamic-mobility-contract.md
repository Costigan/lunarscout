# Dynamic Trajectory Mobility Contract

Status: accepted private Phase 3B contract.
Date: 2026-09-07.

This record defines the compiled mobility behavior used by the private exact
dynamic oracle. See the [trajectory API design](trajectory-api-design.md) for
rationale and future public API context. No class or function in this record is
public API.

## Compiled representation

The compiler binds one validated static problem to one dynamic occupancy
timeline. It produces copied, read-only C-contiguous arrays containing:

- static base duration for every `(y, x, direction)` edge; and
- multiplicative mobility factors for every
  `(interval, y, x, direction)` edge.

Static base durations come from `StaticProblem.transition_time`, so physical
affine distance, connectivity, corner rules, and signed uphill/downhill slip
remain identical to static planning. Invalid static edges have deliberate
positive-infinity base duration.

Callers may supply either a complete interval edge-factor array or one
`(interval, direction)` table broadcast over the grid. They may also supply a
static `(y, x)` deterministic hazard factor. The hazard factor of the entered
destination cell multiplies an edge's interval factor. This is an explicit
input; DEM roughness is not inferred as hazard frequency.

Factors are positive finite `float64` values or deliberate positive infinity.
NaN, zero, negative, Boolean, complex, malformed, and finite multiplication
overflow inputs are structured trajectory errors. Positive infinity makes an
edge infeasible in that interval and is not produced accidentally by overflow.

## Movement over interval boundaries

Finite factors scale the duration required to complete a whole edge while
that interval's conditions hold. If a move crosses a time boundary, completed
edge fraction is accumulated at each interval's rate until the remaining
fraction reaches zero. The edge's spatial signed-slope base duration does not
change during that integration.

An edge encountering a positive-infinity factor before completion is
infeasible for that departure. The rover does not pause midway between raster
cells. The search may instead wait at the source under the occupancy rules and
try a later boundary departure. Exact arrival at an internal boundary uses the
existing Phase 3A boundary rule; arrival at the timeline's final boundary is
unreachable.

The compiled bundle must match the exact static availability mask, complete
grid, neighbor count, and normalized UTC boundaries of the problem presented
to the oracle. A mismatch is rejected before search.

## Deferred vector specialization

The interval edge-factor representation is the hot-loop form expected to hold
future Sun-direction or rover-specific mobility factors. Conversion of Moon-ME
vectors into those factors remains deferred until its local-frame and public
parameterization contract is frozen. The current implementation does not call
SPICE, Numba, CUDA, or a Python callback per edge.
