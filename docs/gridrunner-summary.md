# GridRunner Path-Planning Algorithm Summary

Source: `docs/old_code_gridrunner/` (C# / ILGPU).  These notes describe what
the existing GridRunner code actually implements, for reference when planning
the Python/Numba lunarscout trajectory layer.

## Overview

GridRunner is **not** A* or Dijkstra.  It is a **GPU-accelerated, two-level,
label-correcting flood propagation over a discretized space-time lattice**: a
wavefront that expands from a start cell and stops when it reaches the goal,
with each grid cell holding a single "best" state.

## State per cell (`StateTuple`)

One struct per grid cell holding a *single* best state (not a Pareto set):

- `TotalWh` — total stored energy (battery + fuel cell); `0` is the
  "unoccupied" sentinel.
- `ArrivalTime` — hours since the simulation epoch.
- `BatteryWh` — usable battery energy (the 20% reserve is implicitly excluded
  from this field).
- `MinFuelCellWhScaledBy20` — worst fuel-cell level seen along the path
  (divided by 20), a per-path tracking field.
- `PreviousCell` — an 8-direction encoding (1-9) pointing at the predecessor;
  `255` is the chain terminator (start), used for path reconstruction.

## Two-level structure

1. **Inner level — GPU blocks ("groups").**  The grid is tiled into square
   blocks (64x64 down to 4x4, chosen by device limits via
   `CalculateOptimalGroupSize`).  `Kernel1_Driving` processes each active
   block to *local quiescence* using an intra-block `do/while` loop.

2. **Outer level — CPU active-group manager.**  An `ActiveGroupManager` keeps
   a `HashSet` of active block coordinates, pushes them to the GPU, and reads
   back a per-block "modified" flag.

## Time decomposition

The simulation advances in fixed 2-hour windows (`TimeStepDurationHours`), up
to `maxSteps=400`.  Each window runs two phases:

### Driving phase (`Kernel1_Driving`)

Spatial motion.  Each thread owns one cell.  It reads each of the 8 neighbors
and forms a candidate move *from* the neighbor *to* this cell: add drive
duration to arrival time, subtract drive energy from battery/total.  A
candidate is rejected if it would arrive after the window end or drain the
battery below 0.  The destination cell's **slope must be <= 15 degrees** (a
cell that is too steep is simply never written into).

`IsBetterState(new, old)` selects the winner — **more `TotalWh` wins, ties
broken by earlier arrival** — and sets `PreviousCell`.  If any cell changed, a
shared `group_idle_flag` is set and the block loops again until no cell
changes (local quiescence).

### Charging phase (`Kernel2_CHARGING`)

Time advance to the window end.  Every occupied cell "waits" until
`stepEndTime`:

- If the wait is <= the 0.33 h solar-array deployment time, the battery just
  drains at stationary power (below 0 -> unoccupied).
- Otherwise the array deploys and the cell charges at
  `sunFraction * full-sun watts - stationary watts` for the remainder, clamped
  to battery capacity.

Arrival time is set to the window end.

### Block reactivation

After the driving loop of a window, blocks whose `iteration_count > 1` (i.e.
something changed) are marked "modified"; the CPU manager reactivates those
blocks **and their 8 neighbors**, so changes at block boundaries propagate.
The driving kernel iterates until no active blocks remain (or 500
iterations), then the charging kernel runs and the next 2-hour window begins.

## Movement and energy model

- 8-connected grid, 20 m cells, 2 km/h rover speed.  Adjacent move =
  `0.02 km / 2 km/h` = 0.01 h; diagonal = x sqrt(2).  Drive energy =
  `DrivingPowerWatts * duration` (3925 W).
- Power: driving 3925 W, stationary 2870 W, full-sun generation 10600 W,
  solar-array deployment 0.33 h.
- Storage: battery (31400 Wh raw, 20% reserve excluded -> 25120 Wh usable)
  **plus** a fuel cell (551000 Wh).  Note: the trajectory-planning design doc
  says "one battery / do not model a fuel cell," but the existing code models
  both.
- The greedy `IsBetterState` (maximize `TotalWh`, then earliest arrival) is
  the single-label dominance rule — no Pareto set.  This is exactly the
  SOC-dominance approximation that the review flags as a correctness hazard.

## Path reconstruction

`EnumeratePath` walks `PreviousCell` backward from the goal to the chain
terminator (255) using a 3x3 delta table, then reverses to produce the
forward route; `WriteRouteGeoJson` serializes it.

## Stubbed / incomplete in this copy

- **Sunlight is not actually driven.**  `gpu_sun` is memset to 255 at startup
  and the per-epoch `sunDataGetter` update block is disabled (`if (false)`),
  so the rover currently sees constant full sun.  This is the on-the-fly
  lightmap hook point for the lunarscout integration.
- **The scheduler is a FIFO/set** (`ActiveGroupManager`), not A*-guided; the
  priority-queue change described in
  `docs/hierarchical-a-star-guided-patch-propagation.md` is not yet present.
- **Stationary-logic drift.**  `ApplyStationaryLogic` (which also tracks the
  fuel cell and min-fuel-cell) exists, but the actual charging kernel is a
  simpler battery-only version.

## One-line characterization

A synchronous 2-hour time-stepped, single-label (greedy energy-then-time
dominance), block-parallelized label-correcting flood fill with
neighbor-block reactivation and a wait/charge time-advance — the "GridRunner"
style planner that the trajectory-planning documents describe as the existing
two-level propagation architecture.
