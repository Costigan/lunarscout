# Public SOC Trajectory Contract

Status: Phase 4C accepted and implemented.
Date: 2026-09-08.

Refer to the [trajectory API design](trajectory-api-design.md) for rationale,
the [initial power contract](trajectory-power-contract.md) for Wh accounting,
and the [exact SOC reference contract](trajectory-soc-reference-contract.md)
for oracle scope. This record defines the Phase 4C public behavior.

## Operation and inputs

`ls.trajectory.soc_path(...)` accepts the static trajectory inputs, UTC
half-open environmental boundaries, a `ConfigurationSpaceProvider`, a
`SunlightProvider`, departure time, and explicit solar, battery, and rover power
models. Sunlight bytes are converted to fractions with `value / 255.0`.

Occupancy and sunlight providers must match the trajectory grid. Both timelines
are sampled at each interval start and materialized before CPU planning. The
timeline is limited to five million `(interval, y, x)` states. Movement and
power semantics otherwise follow the existing dynamic and Phase 4A contracts.

## Algorithms and backends

| Algorithm | Backend | Rule | Guarantees |
| --- | --- | --- | --- |
| `exact` | `cpu`, `auto` | Nondominated multi-label oracle | Complete and minimum-arrival within the event-time contract unless its explicit label bound is exceeded. |
| `greedy` | `cpu`, `auto` | One label per `(interval, y, x)`; earliest arrival wins, with higher energy breaking equal-time ties | Every returned path is feasible; completeness and minimum arrival are not guaranteed. |

`greedy` is the default. `auto` selects CPU for the requested algorithm and
never changes algorithms. Explicit CUDA raises `trajectory_backend_unavailable`
for both algorithms in Phase 4C.

The greedy state rule may discard a later label with more energy when both
arrivals fall in the same environmental interval. It can consequently report
unreachable when the exact algorithm finds a route, or return a later route.
The checked-in synthetic counterexample and `examples/34_soc_trajectory.py`
demonstrate the false-negative case.

`max_labels` is a positive explicit bound for either algorithm. Exhausting it
raises `trajectory_exact_soc_label_limit` or
`trajectory_greedy_soc_label_limit`; timeline/state bounds use separate
structured errors. Resource exhaustion is never returned as ordinary
unreachability.

## Results and replay

`SocPathResult` contains a validated `DynamicPathResult`, replayed
`EnergyTimelineResult`, selected algorithm, and resolved backend. Convenience
properties expose cells, UTC times, waits, travel time, final energy, and the
algorithm's `complete` and `optimal` guarantees. Unreachable results contain no
path or energy data.

Every reachable exact and greedy result is replayed before publication. Replay
checks timing, occupancy, movement duration, source-cell sunlight, mode, and all
battery transitions. Result arrays remain owned and read-only through the
dynamic result contract.

Provider streaming, dynamic mobility factors, arbitrary interior charging
departures, CUDA, and mission-scale benchmark claims remain deferred.
