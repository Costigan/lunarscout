# Public Dynamic Trajectory Contract

Status: accepted Phase 3C/3D public contract.
Date: 2026-09-07.

This record freezes the first public dynamic path operation. Refer to the
[trajectory API design](trajectory-api-design.md) for rationale and future
algorithm context, and to the [dynamic time](trajectory-dynamic-time-contract.md)
and [provider](trajectory-provider-contract.md) records for shared semantics.

## Operation

`ls.trajectory.dynamic_path(...)` accepts static traversability and grid inputs
using the same validation, coordinate, affine, diagonal, elevation, and
`StaticTravelModel` rules as `static_path`. It additionally accepts:

- at least two strictly increasing timezone-aware interval boundaries;
- a matching `ConfigurationSpaceProvider`;
- a timezone-aware departure time;
- `algorithm="gridrunner"` or `"safe_interval"`; and
- `backend="auto"`, `"cpu"`, or `"cuda"`.

The configuration provider is read at each interval's start boundary. That
Boolean window applies over the corresponding half-open interval. Reads are
split into bounded private windows and validated before use; window dimensions
are not public API. The current public operation materializes the complete
occupancy cube and is bounded to five million `(interval, y, x)` states.

Both public algorithms are exact under the frozen occupancy and static movement
model. `gridrunner` performs exhaustive block relaxation over cell-interval
states. `safe_interval` searches maximal contiguous allowed intervals. Both
permit waiting and return the earliest arrival before timeline exhaustion.
`backend="auto"` selects the chosen algorithm's CPU implementation without
probing CUDA. Explicit `backend="cuda"` raises a structured unavailable-backend
error; it never falls back. Unknown algorithms and backends have distinct
structured input errors. Backend choice never changes the algorithm.

The private compiled Sun-direction and hazard-factor mobility bundle is not yet
a public argument. The initial public dynamic function uses the selected
`StaticTravelModel` for edge duration while configuration occupancy varies over
time.

## Result

`DynamicPathResult` contains:

- `reachable`;
- final UTC `arrival_time`;
- read-only `int64[N, 2]` raster `cells`;
- one UTC `arrival_times` value per cell; and
- one UTC `departure_times` value per directed leg.

For leg `i`, the rover reaches `cells[i]` at `arrival_times[i]`, may wait, and
leaves at `departure_times[i]`; it then reaches `cells[i + 1]` at
`arrival_times[i + 1]`. `wait_intervals` derives all nonzero waits, and
`travel_time_hours` includes both driving and waiting from initial departure to
final arrival.

An unreachable valid goal returns `reachable=False` and all trajectory data as
`None`; its derived travel time is `None` and wait intervals are empty. When
start equals goal, the result contains one cell, one arrival equal to departure,
no leg departures, and zero travel time.

The result omits block queues, labels, predecessors, activation counts, and
other engine diagnostics. Equal-cost path identity is not promised, so the two
algorithms may return different equal-arrival routes.
