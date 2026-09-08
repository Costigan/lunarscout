# Safe-Interval Planner Implementation Contract

Status: implemented Phase 3D CPU contract.
Date: 2026-09-07.

Refer to the [trajectory API design](trajectory-api-design.md) and the
[dynamic time contract](trajectory-dynamic-time-contract.md) for rationale and
shared scientific semantics. This record describes the private implementation
boundary used by the public `algorithm="safe_interval"` operation.

## State and transitions

`build_safe_intervals` merges each cell's consecutive allowed occupancy samples
into maximal half-open intervals. A search state identifies one raster cell and
one such interval. Its label is the earliest physical arrival in that interval;
with static FIFO edge durations, that arrival dominates later arrivals in the
same interval because the rover may wait continuously while the cell remains
allowed.

For each neighboring destination interval, the planner tests the earliest
departure no earlier than the source arrival or destination-interval start. The
source must remain allowed until movement completion, and the destination must
remain allowed from departure through arrival. Exact sample-boundary handling
uses the shared boundary-snap policy. Arrivals at the open end of a destination
safe interval or at timeline end are rejected.

The frontier is ordered by physical arrival time and terminates when its exact
earliest label for the goal is removed. Positive static edge durations and
earliest-arrival dominance make this an exact no-SOC search under the shared
contract. Equal-arrival path identity is not part of the contract.

## Resource and public boundaries

Provider reads remain bounded windows, but the requested Boolean occupancy cube
is currently materialized before interval construction. Requests above five
million raw `(interval, y, x)` states fail before provider reads. Safe intervals,
frontier state, predecessors, and diagnostics remain private. The public result
is the shared `DynamicPathResult`.

This implementation accepts the public static movement model. The private
compiled time-varying mobility bundle is not yet supported by the safe-interval
engine or exposed by `dynamic_path`.
