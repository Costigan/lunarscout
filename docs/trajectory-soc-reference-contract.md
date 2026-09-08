# Exact SOC Reference Contract

Status: Phase 4B accepted and implemented.
Date: 2026-09-07.

Refer to the [trajectory API design](trajectory-api-design.md) for rationale and
the [initial power contract](trajectory-power-contract.md) for the shared power
semantics. This record defines the scope in which the private Phase 4B oracle is
exact.

## Scope and objective

`_power_reference.exact_soc_path` is a bounded CPU validation oracle, not a
public mission-scale planner. It minimizes UTC arrival time among feasible paths
under the existing dynamic event-time contract. Because departure time is fixed,
elapsed cost and arrival time have the same ordering and are not stored as
independent resources.

The oracle uses continuous float64 time and stored energy in Wh; it does not
bucket SOC. At each reached cell it may move immediately or wait to a later
environmental boundary while the cell remains occupiable. Edge arrivals may
occur between boundaries and become immediate-departure events. The oracle does
not optimize over arbitrary interior charging-departure times that are neither
an arrival nor an environmental boundary.

Occupancy and sunlight inputs must have the same UTC half-open interval
boundaries and the same grid as the static problem. Static movement duration
comes from the validated `StaticTravelModel`. Waiting uses occupied-cell
sunlight, and movement uses source-cell sunlight according to the Phase 4A
contract.

## Labels and dominance

A label contains location, UTC arrival time, and stored energy. Multiple labels
are retained per cell. One label dominates another only when it:

- is at the same cell;
- arrives no later;
- can remain continuously occupiable and battery-feasible while idling to the
  other label's arrival time; and
- has at least the other label's stored energy after that catch-up wait, within
  the Phase 4A energy tolerance.

This catch-up test accounts for charging, idle consumption, capacity clipping,
and environmental steps. Earlier/lower-energy and later/higher-energy labels
therefore remain nondominated when neither can reproduce the other. Equal-cost
ties need not produce a unique raster path.

Computed event times are canonicalized through the UTC datetime representation.
Replay permits `1e-9` hours of round-trip timing error while search and
occupancy retain the existing boundary-snap rule.

## Bounds and replay

The caller supplies or accepts an explicit maximum number of labels. Exceeding
that bound raises `PlanningError` with code
`trajectory_exact_soc_label_limit`; it never becomes an ordinary unreachable
result. The bound limits memory and computation but does not alter dominance.

Every reachable result is reconstructed and independently replayed from the
initial battery energy. Replay checks path timing, occupancy throughout waits
and moves, static edge durations, source-cell sunlight, operating mode, every
battery transition, and final stored energy. A valid but infeasible or
unreachable goal returns an ordinary unreachable private result.

Dynamic mobility factors, arbitrary interior charging departures, provider
streaming, greedy approximation, public SOC dispatch, CUDA, and mission-scale
performance are outside Phase 4B.
