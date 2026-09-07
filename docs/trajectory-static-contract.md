# Static Trajectory Contract Decisions

Status: implemented Phase 0.5 contract.
Date: 2026-09-07.

This record freezes the implementation-critical static choices required by
Phase 0.5 of the [trajectory API design](trajectory-api-design.md). Refer to
that document for rationale and broader context.

## Coordinates and grids

- Pixel coordinates are integer `(x, y)` tuples. Boolean and fractional tuple
  values are rejected.
- A `LonLat` selects its containing pixel using corner-anchored inverse affine
  coordinates and `floor`. Raster bounds are half-open: `[0, width) x
  [0, height)`.
- Values within `1e-6` pixel of an integer boundary are snapped to that boundary
  before cell selection. This makes projected/longitude-latitude round trips
  stable at boundaries.
- The selected start and goal cells must be in bounds, valid, and traversable.
  Validation occurs before the `start == goal` special case.
- Physical planning requires a projected CRS with two finite, positive, equal
  horizontal-axis conversion factors. Geographic/angular, missing-unit, and
  mixed-horizontal-unit CRSs are rejected.
- The affine transform supplies both physical basis vectors. Rotated, skewed,
  and anisotropic grids therefore use their actual cardinal and diagonal
  lengths. CRS lengths are converted to metres before travel calculations.
- Elevation array values are metres. A future unit-bearing raster adapter may
  make this declaration explicit without changing the array API.

## Arrays and availability

- `traversable` is a two-dimensional Boolean or integer array matching the
  `GeoReference`; nonzero means occupiable.
- `valid`, when supplied, is a same-shaped Boolean array. A cell is available
  only when it is both valid and traversable.
- `elevation`, when supplied, is a same-shaped real numeric array. It is
  required for a model with `slip` and must be finite at every available cell.
- Start and goal validation order is grid/type, model, CRS units, arrays,
  coordinate conversion and bounds, then availability.
- Both endpoints of every traversed edge must be available.

## Neighbors and transitions

- Cardinal traversal order is east, south, west, north. Diagonals follow as
  southeast, southwest, northwest, northeast.
- Eight-neighbor movement does not cut corners. A diagonal is feasible only if
  both adjacent cardinal cells are available.
- A physically feasible edge has a finite, strictly positive duration in hours.
  An infeasible edge is assigned `np.inf` deliberately.
- Overflow in an otherwise finite edge, accumulated cost, or heuristic raises a
  structured trajectory error; it is never interpreted as infeasibility.
- A valid but unreachable result is not an exception. Unreachable travel-field
  cells contain `np.inf` and have `reached == False`.

## Slip and heuristic

- `SlipFunction` is piecewise linear over at least two strictly increasing,
  finite signed-slope knots with finite positive travel-time factors.
- Positive signed slope is uphill. Interpolation includes both endpoint knots.
- Extrapolation is explicitly either `"infeasible"` (the default, returning
  `np.inf`) or `"constant"` (using the nearest endpoint factor).
- `StaticTravelModel.speed_m_per_h` is finite and positive. Its default is
  `36.0`; `include_diagonals` must be Boolean; `slip` is either a
  `SlipFunction` or `None`.
- Static model dataclasses use field-based equality and representation. Slip
  knots and factors are normalized to immutable tuples rather than retaining
  caller-owned arrays.
- The A* lower bound is affine-aware straight-line distance divided by nominal
  speed, multiplied by the model's minimum feasible slip factor. The current
  built-in slip representation always exposes that finite positive minimum.

## Results and ties

- `start == goal` returns one `[x, y]` cell and zero hours after ordinary input
  validation.
- Public result arrays are owned by the result and read-only.
- Dijkstra and A* use stable internal neighbor and queue ordering, but equal-cost
  path identity is not a compatibility promise. Optimal travel cost is.
