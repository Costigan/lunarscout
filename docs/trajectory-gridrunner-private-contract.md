# Private Dynamic GridRunner Contract

Status: accepted Phase 3C private-core contract.
Date: 2026-09-07.

This record defines the first private block-reactivation implementation. See
the [trajectory API design](trajectory-api-design.md) for architecture and
rationale. It does not freeze a public dynamic function, result type,
algorithm name, backend option, or diagnostic field.

## Physics and state

`SpaceTimePhysics` binds one validated static problem, one half-open occupancy
timeline, and an optional compatible compiled dynamic travel model. It applies
the Phase 3A movement, waiting, exact-boundary, and timeline-exhaustion rules.
UTC departure is converted to private elapsed hours and interval state only
inside the implementation.

Search has one earliest-arrival label for each `(interval, y, x)` state, using
the dominance rule proven by the exact oracle. Predecessors retain source
state and edge departure time so waits and moves can be reconstructed. The
implementation currently materializes the occupancy timeline in memory.
Provider-backed entry reads it in independently validated, block-bounded
windows; it does not yet stream an unbounded mission timeline.

## Block scheduling and termination

A spatial block owns all interval states for its cells. Processing repeatedly
scans those states in stable interval/row/column/direction order until no local
label improves. Improvements entering another block schedule that block. A
versioned queue discards stale priorities, and a previously processed block is
reactivated whenever a later external improvement enters it.

Queue priority is the minimum arrival label in the block plus a lower bound
formed from minimum block-to-goal affine distance, nominal speed, global
minimum slip factor, and global minimum compiled dynamic factor. Obstacles and
waiting can only increase true cost. This priority affects ordering only.

The current policy is exact and exhaustive: search terminates only when the
active-block queue is empty. It has no approximate early-stop mode. Therefore
correctness does not depend on the priority proof. A configured state-array
limit and local-quiescence guard raise structured `PlanningError` failures;
they are not reported as unreachable paths.

## Current boundary

The engine is private Python correctness scaffolding for the production
GridRunner path. Its result combines the private exact path representation with
private scheduling diagnostics. Public exposure waits for the dynamic result,
algorithm spelling, default provider/materialization policy, progress,
cancellation, and diagnostics compatibility gates.

Acceptance compares reachability and earliest arrival against the independent
exact oracle on hand-authored occupancy and dynamic-mobility cases plus seeded
random grids. A fixed adversarial case must produce a genuine processed-block
reactivation, and provider integration must issue only bounded window reads.
