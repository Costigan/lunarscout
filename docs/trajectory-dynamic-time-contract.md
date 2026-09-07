# Dynamic Trajectory Time Contract

Status: accepted Phase 3A contract.
Date: 2026-09-07.

This record resolves the temporal gate in Section 7.1 of the
[trajectory API design](trajectory-api-design.md). Refer there for rationale and
future public API context. The Phase-3A implementation is a private correctness
oracle; this record does not publish a dynamic result type or final algorithm
spelling.

## Timeline and lookup

- A dynamic occupancy timeline has at least two strictly increasing,
  timezone-aware UTC boundary datetimes.
- An occupancy cube has shape `(interval, y, x)`, where interval `i` applies on
  the half-open interval `[boundary[i], boundary[i + 1])`.
- No interpolation or nearest-time lookup occurs. The containing half-open
  interval is authoritative.
- Dynamic occupancy is defined only on `[boundary[0], boundary[-1])`.
  Departure must be in this domain. A move arriving at or after the final
  boundary is unavailable and does not produce a label.
- Internal real-valued hours are snapped to a boundary when they differ by no
  more than `1e-12` hours, making exact-boundary semantics stable under ordinary
  float64 arithmetic.

## Occupancy during movement

- Static availability and static edge feasibility remain governed by the
  Phase-1 [static contract](trajectory-static-contract.md).
- A move has its static model's arbitrary positive real-valued duration; it is
  never rounded to an environment interval.
- For a move on `[departure, arrival)`, both source and destination cells must
  be dynamically occupiable in every environment interval intersecting that
  half-open span. This is the initial conservative edge-occupancy rule.
- The destination must additionally be occupiable in the interval containing
  the arrival instant.
- The source need not be occupiable in a new interval beginning exactly at
  arrival because the rover has left it at that instant.
- A move may cross any number of environment boundaries when these conditions
  remain true.

## Waiting and ordering

- Waiting is allowed at every statically available cell while that cell remains
  dynamically occupiable.
- Waiting may span any number of consecutive allowed intervals but may not cross
  a forbidden interval.
- Search may alternate waiting and moving at every visited cell. Candidate
  departures are the current arrival instant and later environment boundaries
  reachable by a continuous allowed wait.
- Since environment values change only at boundaries and movement duration is
  fixed for a static edge, those candidates cover every earliest feasible
  transition under this contract.

## Boundary and result behavior

- Arrival exactly at an internal boundary uses the new interval for destination
  occupancy and subsequent actions.
- A dynamically unavailable departure cell is an input error. A valid goal that
  cannot be reached before timeline exhaustion is an ordinary unreachable
  result.
- The exact oracle stores the earliest arrival for each `(cell, interval)`.
  Keeping interval-specific labels is required because an earlier label before
  a forbidden gap cannot necessarily wait into a later allowed interval.
- The private oracle returns cell arrival times and edge departure times so wait
  behavior can be replayed independently. No representation in that private
  result is a public compatibility promise.
