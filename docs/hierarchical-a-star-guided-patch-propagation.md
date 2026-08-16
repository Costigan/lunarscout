# Hierarchical A*-Guided Patch Propagation

## Purpose

Implement a GPU-accelerated rover path planner in Python using Numba CUDA.

The planner operates on a spatial raster and a discretized time dimension. Conceptually, it performs a constrained flood fill through a 3D space of:

- X position
- Y position
- time

Each reachable state additionally carries rover power information.

The algorithm should preserve the existing two-level propagation architecture:

1. Within a rectangular spatial patch, propagate rover states repeatedly until the patch reaches quiescence.
2. At the global level, maintain active patches and process the best N patches concurrently on the GPU.

The global patch scheduler should be changed from an unordered/FIFO work queue to an A*-guided priority queue so that patches that appear more promising for reaching the requested destination are processed preferentially.

This architecture may be described as:

> Hierarchical A*-guided label propagation in which rover states are relaxed to quiescence within GPU-resident spatial patches, while a CPU-side priority queue schedules batches of active patches according to an optimistic estimate of goal completion cost.

The implementation should preserve correctness-oriented state propagation and dominance semantics independently of the A* scheduling heuristic. Initially, A* should control the order in which patches are processed rather than changing which states are considered valid.

---

# 1. High-Level Problem

Find efficient rover trajectories through a gridded lunar terrain while accounting for:

- terrain traversability;
- slope constraints;
- time-varying sunlight;
- communications constraints;
- rover power consumption while driving;
- rover power consumption while stationary;
- solar power generation;
- battery capacity;
- minimum allowable battery state;
- start location;
- destination location or destination region;
- start time;
- initial battery energy;
- planning time horizon.

The first implementation should use a simplified rover power model containing only a single battery.

Do not model a separate fuel cell or auxiliary energy reserve.

A rover state therefore contains approximately:

- spatial cell;
- arrival time;
- battery energy;
- predecessor information;
- any additional path metrics required for dominance or reconstruction.

The planner should support both:

- transitions that change spatial location;
- transitions that remain in the same spatial cell while advancing time and modifying battery state.

---

# 2. Fundamental State-Space Interpretation

Treat the planner as propagation through a discretized space-time state lattice.

A conceptual state is:

    S = (x, y, t, battery)

The explicit raster dimensions are:

    x
    y
    t

Battery energy is a state label associated with a particular space-time location rather than necessarily being another dense raster dimension.

A drive transition has the form:

    (x, y, t, battery)
        ->
    (x2, y2, t + drive_time, battery - drive_energy)

A wait/charge transition has the form:

    (x, y, t, battery)
        ->
    (x, y, t2, updated_battery)

The second transition is important: state propagation does not necessarily imply spatial motion.

The current architecture should retain discretized planning windows. Within a planning window, rover motion propagates through spatial cells. Once spatial propagation for that window has converged, states may be advanced in time through stationary consumption and solar charging.

The implementation should be structured so that alternative temporal transition schemes can later be introduced without rewriting the entire planner.

---

# 3. Two-Level Propagation Architecture

The defining feature of the algorithm is that propagation occurs at two different spatial scales.

## 3.1 Inner Level: Patch Relaxation

Divide the planning raster into rectangular patches.

Examples might be:

    32 x 32 cells
    64 x 64 cells
    128 x 128 cells

The exact patch size should be configurable and benchmarked.

A patch is the basic GPU work unit.

When a patch is processed, rover states should be repeatedly propagated within the patch until no state in the patch changes.

In other words:

    load/process patch
        ->
    relax reachable cells
        ->
    determine whether any state changed
        ->
    if changed, relax again
        ->
    repeat until quiescent

Quiescence means that another application of the local propagation operator would produce no improved state within the patch.

Do not process a patch only once.

Local convergence is an essential property of this architecture.

The reason is that propagation may cross many cells inside one patch before neighboring patches need to be reconsidered.

This reduces the amount of global scheduling and CPU/GPU coordination.

---

# 4. Patch Boundary Propagation

If state changes reach a patch boundary, neighboring patches may need to be processed.

For each patch, determine whether its relaxation changed states along or near any of its boundaries.

Depending on implementation convenience, either:

- mark specific neighboring patches affected by changed boundary cells;
- or conservatively activate all neighboring patches whenever the patch changes.

For a rectangular tiling, a patch may activate up to eight neighboring patches.

A patch must also be eligible to reactivate after it has previously reached quiescence.

For example:

    Patch A converges.
    Patch B later discovers a better state along their shared boundary.
    Patch A must become active again.

The algorithm is therefore label-correcting rather than a one-pass traversal.

---

# 5. Global Active Patch Scheduler

Maintain a CPU-side collection of active patches.

In the original architecture this may behave like a simple queue.

Replace this with a priority queue.

Each active patch receives an A*-style priority representing the most optimistic known completion cost through that patch.

At each global scheduling iteration:

1. Pop the best N active patches.
2. Dispatch those N patches to the GPU.
3. Relax each patch independently or mostly independently until local quiescence.
4. Retrieve or inspect change information.
5. Activate neighboring patches affected by changed boundaries.
6. Recompute priorities where necessary.
7. Insert newly active or reprioritized patches into the priority queue.
8. Continue until the stopping condition is satisfied.

N should be configurable.

N provides an intentional tradeoff between strict best-first behavior and GPU utilization.

Small N:

- behaves more like conventional A*;
- performs strongly goal-directed exploration;
- may underutilize the GPU.

Large N:

- increases GPU utilization;
- performs more speculative work;
- gradually approaches an unordered parallel flood fill.

Benchmark N rather than assuming one fixed value.

---

# 6. A*-Guided Patch Priority

The initial objective should be earliest feasible arrival at the destination.

For a conventional A* search:

    f = g + h

Use the same conceptual structure for patches.

However, a patch contains many rover states, so g is not a single path cost in the usual sense.

A useful initial definition is:

    g(P) = earliest arrival time of any retained state in patch P

The heuristic should estimate the minimum remaining travel time from the patch to the goal.

Initially use a purely geometric heuristic:

    h(P) = minimum_distance_from_patch_to_goal / maximum_rover_speed

Then:

    priority(P) = g(P) + h(P)

The heuristic should initially ignore:

- sunlight;
- shadow;
- battery level;
- expected charging;
- communication availability;
- future waiting;
- terrain difficulty other than impassability if desired.

This is intentional.

Ignoring these constraints makes the heuristic optimistic.

The heuristic should estimate what would happen under unrealistically favorable conditions:

- unlimited power;
- no waiting;
- maximum rover speed;
- no unfavorable sunlight effects.

That makes it useful as a lower bound on achievable arrival time.

---

# 7. Distance from Patch to Goal

Do not use distance from the patch center to the goal as the formal heuristic distance.

Instead compute the minimum geometric distance between any position in the patch and the destination.

For a point destination:

    d(P, goal) =
        minimum Euclidean distance between the goal and the rectangular footprint of P

If the goal lies inside the patch:

    d = 0

For a destination region:

    d(P, goal_region) =
        minimum distance between the patch rectangle and the goal region

This produces a more conservative lower bound than measuring from the patch center.

For an initial implementation, using the nearest spatial cell in the patch is also acceptable if that is simpler.

---

# 8. Possible State-Based Patch Priority Refinement

A stronger priority may eventually be calculated from individual states rather than using:

    earliest_patch_arrival + patch_distance

For every retained state s in patch P, compute:

    optimistic_goal_time(s)
        =
    arrival_time(s)
        +
    distance(s.position, goal) / maximum_rover_speed

Then define:

    priority(P)
        =
    minimum optimistic_goal_time(s)
    over all retained states s in P

This is potentially tighter because the earliest state in a patch may not be the state spatially closest to the destination.

Do not require this optimization in the first implementation.

Start with a simple patch-level g + h definition and preserve the API so this refinement can be added later.

---

# 9. A* Is Initially a Scheduling Policy

A critical design rule:

Do not initially let A* determine state validity.

The A* heuristic should control:

    which active patch is processed next

It should not initially control:

    whether a rover state is discarded

These mechanisms must remain separate.

State dominance determines which rover states survive.

Patch priority determines which work is performed first.

This separation allows the heuristic to be changed without changing planner correctness.

It also permits comparison against the unordered flood-fill implementation.

If the priority queue were allowed to drain completely, and no additional heuristic pruning were used, the final reachable-state solution should be equivalent to the same algorithm using an unordered patch queue, modulo explicitly configured approximations.

---

# 10. Goal Detection and Termination

Do not automatically stop merely because the goal is reached for the first time unless the stopping condition is justified by the current search ordering.

Initially distinguish:

    first feasible route found

from:

    best route proven under the selected objective

The first reachable goal state provides an upper bound on arrival time:

    best_goal_time

Once a valid admissible lower-bound heuristic exists, it may become possible to terminate when no active patch can possibly produce a better result.

For example, if:

    minimum priority among all active patches >= best_goal_time

then no remaining active patch can improve the current earliest-arrival solution, provided that the priority calculation is a valid lower bound for all states that could emerge from that patch.

This proof obligation should be treated carefully because patch-level relaxation differs from ordinary state-level A*.

Initially, it is acceptable to use A* only to find good solutions quickly while continuing propagation until the conventional algorithmic stopping condition.

---

# 11. Configuration-Space Generation Per Patch

Preserve the existing abstraction that obtains a configuration-space raster for each patch.

The planner should not hard-code all terrain, illumination, and communications logic directly into the propagation kernel.

Provide one or more functions that determine which rover positions are valid for a specified patch and time interval.

Conceptually:

    get_configuration_space(patch, time)

or:

    get_configuration_space(
        patch,
        time_start,
        time_end
    )

The returned data should identify whether each cell is valid for rover occupation or traversal under the relevant operational constraints.

At minimum, configuration-space validity should combine:

- slope;
- terrain exclusion;
- illumination constraints where applicable;
- communications constraints where applicable.

For example:

    valid =
        slope_ok
        AND terrain_ok
        AND sun_constraint_ok
        AND communications_ok

However, sunlight should not necessarily always be treated as a hard validity constraint.

The architecture should distinguish:

- sunlight affecting power generation;
- sunlight being required by a mission rule;
- shadow being physically traversable but energetically costly.

Similarly, communications may be:

- a hard continuous constraint;
- required only at specific times;
- required periodically;
- irrelevant during some motion segments.

Therefore configuration-space generation should be implemented behind an interface or functional abstraction that allows mission policy to change.

---

# 12. Configuration-Space Caching

Configuration-space generation may be expensive.

Allow patch configuration spaces to be cached.

A useful cache key may include:

    patch_id
    time_index
    relevant mission mode

Do not require regeneration of slope masks if slope is static.

Separate static and dynamic constraints where practical.

Possible structure:

    StaticPatchData
        slope_valid
        terrain_valid
        roughness
        fixed exclusions

    DynamicPatchData(time)
        sunlight
        communications
        transient exclusions

Then:

    configuration_space =
        combine(static_data, dynamic_data)

This should reduce repeated CPU and GPU work.

---

# 13. Simplified Battery Model

Replace the previous battery-plus-fuel-cell model with one battery.

A state should contain at least:

    arrival_time
    battery_wh
    predecessor
    path/state validity metadata

Battery constraints:

    0 <= battery_wh <= battery_capacity_wh

or, preferably:

    minimum_battery_wh <= battery_wh <= battery_capacity_wh

where minimum_battery_wh represents the operational reserve.

Driving consumes:

    drive_energy =
        drive_power * drive_duration

Initially constant drive power and constant speed are acceptable.

Later these may depend on:

- slope;
- terrain;
- rover orientation;
- thermal state;
- driving mode.

Do not design the first implementation around these extensions, but keep energy calculation in a function that can later be replaced.

---

# 14. Driving Transition

For each valid neighboring cell:

1. Verify that the destination cell is traversable.
2. Compute movement distance.
3. Compute drive time.
4. Compute battery consumption.
5. Advance arrival time.
6. Reject transitions beyond the current allowed temporal interval.
7. Reject transitions violating battery reserve.
8. Construct the candidate destination state.
9. Apply the destination dominance rule.
10. Store the state if accepted.
11. Record enough predecessor information for route reconstruction.

Use 8-connected movement unless mission requirements specify otherwise.

Distances:

    cardinal = cell_size

    diagonal = sqrt(2) * cell_size

Do not approximate diagonal moves as cardinal moves.

---

# 15. Time Windows and In-Place Temporal Propagation

Retain the existing concept of planning time windows.

For each temporal window:

    [window_start, window_end]

allow spatial driving propagation within the current window.

Once relevant patch propagation reaches quiescence, advance reachable states to the end of the time window using the stationary power model.

This is an explicit same-location state transition:

    (x, y, arrival_time, battery)
        ->
    (x, y, window_end, updated_battery)

The rover may therefore propagate through the state graph without changing spatial position.

This temporal propagation should account for:

- stationary power consumption;
- solar generation;
- battery capacity;
- deployment delays if applicable;
- minimum battery reserve.

---

# 16. Solar Charging

Solar input should be based on spatial position and time.

The implementation should permit either:

- discrete sunlight samples;
- integrated sunlight over an interval;
- a horizon-derived illumination provider;
- precomputed illumination rasters.

Define the charging calculation behind a function such as:

    update_stationary_battery(
        cell,
        arrival_time,
        end_time,
        initial_battery
    )

This function should:

1. subtract stationary consumption;
2. add available solar energy;
3. apply charging efficiency if modeled;
4. cap battery energy at capacity;
5. reject the state if battery reserve is violated before charging can recover it, if that is physically required.

Be explicit about whether transient battery depletion inside a waiting interval is allowed.

Normally, if survival load cannot be met before sunlight begins, the state should be invalid even if the net energy over the complete interval would be positive.

---

# 17. Solar Deployment Delay

If the rover requires a delay before meaningful solar charging begins, preserve that behavior.

For a state arriving at:

    t_arrival

with deployment delay:

    t_deploy

solar generation should begin no earlier than:

    t_arrival + t_deploy

unless deployment state is modeled separately.

Stationary consumption continues during deployment.

The implementation may initially assume that arrays are automatically deployed whenever waiting/charging occurs.

Keep this policy isolated so explicit deployment actions can be introduced later if necessary.

---

# 18. State Dominance

Multiple paths may reach the same space-time cell.

Do not simply overwrite a state because another path reaches the same cell.

At minimum, dominance should recognize the relationship between:

    arrival time
    battery energy

If two states refer to the same cell and equivalent planning time layer, state A conservatively dominates state B if:

    A.arrival_time <= B.arrival_time

and:

    A.battery_wh >= B.battery_wh

with at least one strict inequality.

Then B need not be retained.

However, carefully evaluate whether discretized time windows make arrival time itself future-relevant inside a time layer.

For example, arriving at 10:01 with less battery may be strategically different from arriving at 10:59 with more battery because charging opportunities differ.

Therefore do not assume that one state per spatial cell is always sufficient.

The implementation should support multiple retained labels per cell or provide an abstraction that allows this to be introduced.

---

# 19. Initial Multi-Label Strategy

For correctness testing, implement or plan for a small CPU reference representation that allows multiple labels.

A cell may conceptually contain:

    label 0
        arrival_time
        battery

    label 1
        arrival_time
        battery

    ...

Dominated labels are removed.

For the GPU implementation, a fixed maximum number of labels per cell may eventually be necessary.

Possible strategies include:

- exact Pareto labels for the CPU reference solver;
- fixed K labels per GPU cell;
- time buckets;
- energy buckets;
- epsilon dominance;
- top-K approximate retention.

Do not silently introduce approximate pruning.

Any fixed label cap should be configurable and documented as an approximation unless correctness can be established.

---

# 20. Patch-Level Quiescence

Within a patch, repeatedly apply state relaxation until no accepted state changes occur.

Conceptually:

    changed = true

    while changed:
        changed = false

        for every active/reachable state in patch:
            evaluate neighbor transitions

            if destination state improves:
                store candidate
                changed = true

On the GPU this should be implemented in a parallel manner rather than literally using the pseudocode above.

A patch should report at least:

    did_anything_change

and preferably boundary-specific information such as:

    changed_north
    changed_south
    changed_east
    changed_west

and optionally diagonal boundary changes.

This information drives global patch activation.

---

# 21. GPU Execution Model

Use Numba CUDA for GPU kernels.

Python should manage:

- problem setup;
- priority queue;
- patch metadata;
- batch formation;
- stopping conditions;
- diagnostics;
- path reconstruction;
- configuration-space orchestration.

CUDA kernels should perform highly parallel regular operations such as:

- state relaxation;
- transition evaluation;
- dominance comparison where practical;
- charging updates;
- changed-cell detection;
- patch-boundary change detection.

Avoid excessive CPU/GPU synchronization.

The patch architecture should explicitly reduce synchronization by allowing substantial local GPU work before global scheduling decisions are required.

---

# 22. N-Parallel Patch Processing

At each outer scheduling iteration, select up to N patches with the best priorities.

For example:

    batch = priority_queue.pop_best(N)

The GPU should process these patches in parallel.

The implementation should not assume that strict A* ordering is preserved when N > 1.

This is intentionally a batched best-first algorithm.

If current priorities are:

    P1 = 100
    P2 = 101
    P3 = 102
    P4 = 500

and N = 4, P4 may be processed even though P1 could generate new work with priority 100.5.

This is acceptable.

The goal is to balance:

- A*-like directionality;
- GPU occupancy;
- reduced total explored area.

Treat N as a performance parameter.

Collect benchmarks showing:

    N
    GPU utilization
    patches processed
    states processed
    time to first solution
    time to proven solution
    total runtime

---

# 23. Duplicate Patch Activation

A patch may be activated many times before it is processed.

Do not blindly insert unlimited duplicate entries into the priority queue.

Maintain patch scheduling metadata such as:

    inactive
    queued
    running
    dirty

If a queued patch receives new incoming state:

- update its priority if necessary;
- mark its state dirty;
- avoid creating unnecessary duplicate queue records.

A heap implementation may use lazy priority updates if direct decrease-key operations are inconvenient.

For example, associate each patch with a generation counter and discard stale heap entries when popped.

---

# 24. Concurrency Between Adjacent Patches

N parallel patches may be adjacent.

Avoid race conditions when two GPU patch relaxations can update the same destination state.

Possible approaches include:

1. Disallow simultaneously processing directly adjacent patches.
2. Use atomic state update logic.
3. Give each patch private output buffers and merge results afterward.
4. Use halo/boundary exchange between iterations.
5. Assign ownership of destination cells to a unique patch.

Do not choose an approach solely for convenience without benchmarking it.

A halo-based approach may fit the existing patch architecture well:

    patch interior
    +
    one-cell halo

Each patch locally converges its interior using stable input boundary data.

Boundary changes are exported after convergence.

This naturally limits cross-patch synchronization.

---

# 25. Reverse Spatial Search Heuristic Improvement

After the Euclidean heuristic is working, implement a stronger optional heuristic using a reverse spatial search from the destination.

This reverse search should ignore power.

It should answer:

> Assuming unlimited energy and no waiting requirements, what is the minimum possible remaining travel time from this spatial location to the destination?

Run a reverse Dijkstra or A* search from the destination over the static terrain grid.

The reverse search may respect:

- impassable terrain;
- slope-based exclusion;
- static hazards;
- terrain-dependent maximum speed.

It should ignore:

- battery;
- sunlight;
- charging;
- communications constraints that vary with time;
- waiting.

The result is a static raster:

    minimum_remaining_travel_time[x, y]

Use this raster as the heuristic.

For a state:

    h(s) =
        minimum_remaining_travel_time[s.x, s.y]

For a patch:

    h(P) =
        minimum h(x, y)
        over valid cells in P

or use the minimum heuristic value among currently reachable states in the patch.

This should be a significantly stronger heuristic than straight-line distance when terrain barriers require detours.

Because the reverse search assumes unlimited power and removes time-dependent restrictions, it should remain optimistic.

---

# 26. Reverse Heuristic Precomputation

Cache the reverse heuristic raster for a fixed destination.

If planning repeatedly to the same destination:

    goal
        ->
    reverse terrain-only search
        ->
    heuristic raster
        ->
    many forward power-aware searches

If the destination changes frequently, reverse heuristic computation must be cheap enough to justify itself.

For goal regions rather than one cell, initialize the reverse search with every cell in the goal region at zero cost.

---

# 27. Possible Hierarchical Heuristic Later

Do not require this initially, but keep the architecture compatible with multi-resolution planning.

A future solver might perform:

    coarse spatial route search
        ->
    identify broad candidate corridor
        ->
    fine-resolution hierarchical patch propagation

Do not permanently restrict the fine planner to a narrow corridor because solar power may make substantial detours beneficial.

Any corridor restriction should be expandable.

---

# 28. Predecessor Storage and Path Reconstruction

Store enough information to reconstruct the full trajectory.

A predecessor must identify the actual previous state, not merely a direction, if multiple labels per cell are permitted.

Useful predecessor information includes:

    previous cell
    previous time layer
    previous label index or state ID
    action type
    action duration

Action types may include:

    DRIVE_N
    DRIVE_NE
    DRIVE_E
    DRIVE_SE
    DRIVE_S
    DRIVE_SW
    DRIVE_W
    DRIVE_NW
    WAIT_CHARGE

The reconstructed result should contain:

- spatial path;
- timestamps;
- battery profile;
- sunlight or configuration-space status where useful;
- waiting intervals;
- arrival time.

---

# 29. CPU Reference Solver

Even though the production architecture is GPU-oriented, implement a simple CPU reference solver for small problems.

The CPU solver should emphasize correctness rather than speed.

It should support:

- the same state-transition functions;
- exact or relatively exact dominance handling;
- tiny synthetic maps;
- detailed tracing;
- deterministic behavior.

Use it as an oracle for validating the Numba GPU implementation.

Do not optimize away this reference implementation.

---

# 30. Synthetic Correctness Tests

Create small synthetic scenarios whose expected behavior is obvious.

At minimum test:

## All traversable, always sunlight

The solution should approximate the geometric shortest path.

## All traversable, always shadow

The rover should propagate until battery reserve prevents further travel.

## Sunlit charging station

The optimal feasible route should require waiting and charging.

## Shadow barrier

A route should require sufficient stored battery before entering a long shadow region.

## Two paths

One route is shorter but energy-infeasible.

The other is longer but sunlit and feasible.

The solver should select the feasible route.

## Arrival-time tradeoff

Two states reach the same cell:

    earlier with less battery
    later with more battery

Construct the remainder of the map such that each can be preferable in different cases.

This verifies that an over-aggressive single-label dominance rule fails.

## Patch boundary

Ensure a state entering a neighboring patch causes that patch to activate.

## Patch reactivation

Ensure a previously quiescent patch is processed again when a better incoming state arrives.

## A* queue equivalence

Run the same problem using:

    unordered patch queue

and:

    A*-guided patch priority queue

Without heuristic pruning, the final reachable states should agree.

## N variation

Run the same problem with:

    N = 1
    N = 2
    N = 4
    N = 8
    ...

Verify solution consistency while measuring performance.

---

# 31. Diagnostics

Collect enough statistics to understand both algorithm behavior and GPU performance.

Useful metrics include:

    patches activated
    patches processed
    patch reactivations
    queue maximum size
    average active queue size
    states generated
    states accepted
    states rejected
    states dominated
    local relaxation iterations
    average iterations per patch
    boundary updates
    GPU batch sizes
    time to first feasible goal
    final goal arrival time
    GPU runtime
    CPU scheduling runtime
    configuration-space generation runtime

Also provide optional diagnostic rasters for:

    earliest arrival time
    battery at selected state
    reachability
    patch processing count
    patch activation count
    heuristic value

These are extremely useful for verifying that the A* scheduler is actually directing search toward the destination.

---

# 32. Performance Comparison

Retain a mode using the original unordered patch scheduling.

The new implementation should support at least:

    scheduler = FIFO
    scheduler = ASTAR

This provides a direct experimental comparison.

Measure:

- identical or equivalent solution quality;
- number of patches processed before first solution;
- total patches processed;
- total GPU work;
- wall-clock runtime;
- GPU utilization.

A successful A*-guided scheduler should ideally reach the destination after processing a much smaller subset of the globally reachable terrain.

---

# 33. Suggested Python Module Structure

Use a modular structure similar to:

    gridrunner/
        config.py
        geometry.py
        terrain.py
        illumination.py
        communications.py
        configuration_space.py
        rover.py
        state.py
        dominance.py
        patches.py
        scheduler.py
        heuristic.py
        reverse_heuristic.py
        gpu/
            driving.py
            charging.py
            relaxation.py
            reductions.py
        cpu_reference.py
        planner.py
        reconstruction.py
        diagnostics.py

The exact filenames are not mandatory, but preserve separation between:

- physical rover modeling;
- map/configuration-space generation;
- state transition logic;
- patch scheduling;
- GPU kernels;
- heuristics;
- diagnostics.

Do not put the entire planner into one Numba-oriented module.

---

# 34. Implementation Sequence

Implement in stages.

## Stage 1: Port the domain model

Port:

- terrain representation;
- grid geometry;
- sunlight provider;
- communications provider;
- configuration-space generation;
- rover power model.

Simplify power state to one battery.

## Stage 2: CPU state transitions

Implement and test:

- drive transition;
- stationary transition;
- battery accounting;
- validity checks;
- dominance.

## Stage 3: CPU reference planner

Build a small correct planner for synthetic test cases.

## Stage 4: Patch abstraction

Implement:

- raster tiling;
- patch indexing;
- patch neighbors;
- activation state;
- dirty state;
- boundary tracking.

## Stage 5: Numba GPU patch relaxation

Port the existing local relaxation concept.

A GPU patch must propagate repeatedly until local quiescence.

## Stage 6: FIFO global scheduler

First reproduce the behavior of the existing two-level algorithm using a simple scheduler.

Use this to validate the Python/Numba port.

## Stage 7: A*-guided patch scheduler

Replace the FIFO queue with a priority queue.

Initially use:

    g(P) =
        earliest arrival time in patch

    h(P) =
        minimum Euclidean distance from patch to goal
        /
        maximum rover speed

## Stage 8: N-patch GPU batches

Process the best N patches concurrently.

Benchmark N.

## Stage 9: Goal-bound pruning

After correctness is established, consider using the current best goal arrival time to reject work whose admissible lower bound cannot improve the solution.

Keep this optional until its correctness has been demonstrated.

## Stage 10: Reverse heuristic

Implement the optional reverse terrain-only shortest-path calculation and compare it with Euclidean distance.

---

# 35. Important Architectural Principles

Preserve these principles throughout implementation.

## Separate physics from search

Functions describing:

- movement;
- power consumption;
- charging;
- configuration-space validity

should not depend on whether the scheduler is FIFO or A*.

## Separate dominance from priority

Dominance decides whether a state remains.

Priority decides when work is processed.

Do not confuse them.

## Preserve patch reactivation

Quiescence is only local and temporary.

A patch can become relevant again after receiving improved boundary states.

## Preserve deterministic reference behavior

GPU execution ordering may differ, but deterministic CPU reference results should be available for comparison.

## Prefer admissible heuristics

Start with simple optimistic heuristics rather than sophisticated estimates of power or sunlight delay that may accidentally overestimate the true remaining cost.

## Optimize explored work before micro-optimizing kernels

The primary reason for A*-guidance is to avoid propagating through large portions of irrelevant space-time.

Reducing the amount of terrain processed may produce larger gains than optimizing individual transition arithmetic.

---

# 36. Expected Algorithmic Character

The resulting planner is not ordinary A*.

It is not ordinary flood fill either.

It is best understood as:

> A hierarchical, batched, A*-guided label-correcting propagation algorithm over a discretized space-time rover state graph.

The inner level exploits spatial locality:

    patch
        ->
    GPU relaxation to quiescence

The outer level exploits goal direction:

    active patches
        ->
    A*-guided priority queue
        ->
    best N patches
        ->
    GPU batch

The state-transition model accounts explicitly for power:

    driving
    waiting
    solar charging

The configuration-space model accounts explicitly for operational feasibility:

    slope
    terrain
    sunlight rules
    communications

The initial heuristic intentionally ignores power and dynamic illumination:

    optimistic geometric travel time

A later heuristic may replace Euclidean distance with a reverse, terrain-only shortest-path raster:

    optimistic terrain-aware travel time to goal

This architecture should retain the computational strengths of the existing patchwise GPU flood propagation while making the search strongly goal-directed and substantially reducing propagation into terrain that is unlikely to contribute to a useful route.
