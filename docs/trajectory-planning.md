# Rover Trajectory Planning Problem Cases and Candidate Algorithms

## Purpose

These notes summarize the rover trajectory-planning problem space we have been discussing and the algorithm families that appear most appropriate for each case.

The immediate purpose is to use this as a design discussion document for a Python + Numba lunar mission-planning library. The goal is not to implement every possible planning algorithm. Instead, the library should cover the important classes of early lunar mission-planning problems with a small number of reusable planning engines and simulation layers.

The important distinction is that several largely independent dimensions change the nature of the planning problem:

1. **Mission objective**

   - Point-to-point traversal.
   - Science-target/orienteering traversal.

1. **Environment/configuration-space model**

   - Static configuration space.
   - Time-varying configuration space.
   - Time-varying configuration space with solar power and battery constraints.

1. **Rover execution model**

   - Deterministic nominal rover.
   - Rover with uncertain performance.
   - Rover subject to discrete faults, delays, or degraded operating modes.

These dimensions should preferably be represented through composable abstractions rather than by creating a completely separate planner for every combination.

______________________________________________________________________

# 1. Common Physical Model

The planning region is represented as a raster.

Static terrain properties may include:

- elevation;
- slope;
- roughness;
- crater or rock/hazard presence;
- permanent exclusion regions;
- other terrain-derived constraints.

Dynamic environmental properties may include:

- sunlight;
- solar incidence or solar generation strength;
- communications availability;
- temporary exclusion regions;
- other epoch-dependent operational constraints.

Sunlight and related environment data are currently evaluated at approximately two-hour intervals.

A target-to-target traversal may nevertheless take anywhere from about an hour to several days, so an individual trajectory can cross many environment epochs.

The dynamic problem should therefore not assume that a target-to-target edge belongs to one illumination epoch. A physical traverse may interact with dozens of successive configuration-space states.

For power-aware cases, use a simplified rover energy model initially:

- one battery;
- battery capacity;
- minimum allowable state of charge, for example 20%;
- drive power;
- stationary power;
- solar generation as a function of position and epoch;
- charging efficiency if needed.

Battery state is primarily a feasibility variable rather than a science objective.

______________________________________________________________________

# 2. Configuration-Space Cases

## 2.1 Static Configuration Space

The valid rover positions and movement costs depend only on static terrain properties.

Conceptually:

```
C(x, y) =
    f(
        slope,
        roughness,
        crater/hazard mask,
        permanent exclusions
    )
```

There is no time dependence in traversability.

This is the simplest and most reusable baseline.

Important consequences:

- shortest paths are time-independent;
- target-to-target costs can be precomputed;
- A\* or Dijkstra results can be cached;
- reverse distance/travel-time fields can be reused;
- science-target planning becomes an ordinary orienteering problem;
- rover performance variation can be studied independently of environmental variation.

______________________________________________________________________

## 2.2 Time-Varying Configuration Space Without SOC Constraints

The valid rover positions depend on time because of sunlight, communications, or other operational constraints.

Conceptually:

```
C(x, y, t) =
    f(
        slope,
        roughness,
        sun(x, y, t),
        comm(x, y, t),
        dynamic constraints
    )
```

The rover may, for example, be constrained to remain in illuminated terrain.

If this guarantees that the rover remains power-positive, battery state of charge need not be represented explicitly.

The problem is still dynamic because a route that is feasible at one epoch may not be feasible two hours later.

Long traverses can cross many configuration-space epochs.

______________________________________________________________________

## 2.3 Time-Varying Configuration Space With Power

The environment changes with time and the rover may enter shadow or otherwise experience periods of negative power balance.

The state includes battery SOC.

Conceptually:

```
state =
    (x, y, t, SOC)
```

with a constraint such as:

```
SOC >= 20%
```

Solar generation depends on position and time.

The planner must account for:

- drive energy;
- stationary consumption;
- solar generation;
- waiting/charging;
- battery capacity;
- minimum SOC.

SOC is normally treated as a feasibility/resource variable rather than as part of the trajectory's science value.

______________________________________________________________________

# 3. Rover Execution Cases

The rover model introduces a separate axis from the environment model.

## 3.1 Deterministic Nominal Rover

Examples:

- fixed driving speed;
- fixed drive power;
- fixed charging efficiency;
- deterministic communications;
- deterministic deployment delays;
- no faults.

Given an action and state, the resulting state is deterministic.

This should be the reference mode for all planners.

______________________________________________________________________

## 3.2 Performance Variation

Examples:

- speed varies around nominal;
- drive power varies;
- solar conversion efficiency varies;
- traversal time varies;
- slip causes delays;
- operation durations vary.

The same commanded action may therefore produce a distribution of outcomes.

This can initially be modeled through Monte Carlo simulation without changing the basic deterministic planners.

______________________________________________________________________

## 3.3 Faults and Degraded Modes

Examples:

- temporary mobility degradation;
- permanent speed reduction;
- reduced solar generation;
- deployment failure;
- temporary stoppage;
- communications failure;
- instrument failure;
- wheel or steering degradation;
- delayed recovery.

A fault may change the rover's future transition model.

The rover state may therefore need a mode variable such as:

```
rover_mode =
    nominal
    degraded_mobility
    degraded_power
    instrument_failed
    ...
```

For these cases, a fixed precomputed trajectory may be less useful than a policy that replans from the rover's actual state after a fault.

______________________________________________________________________

# 4. Mission Objective Cases

Two main objective families appear sufficient for the library.

## 4.1 Point-to-Point Planning

Given:

- start location;
- start time;
- initial SOC if relevant;
- goal location or goal region;

find an efficient feasible traverse.

Possible objectives include:

- earliest arrival;
- shortest distance;
- minimum energy use;
- maximize probability of arrival;
- minimize expected arrival time;
- maximize robustness subject to arrival constraints.

______________________________________________________________________

## 4.2 Science-Target / Orienteering Planning

There is a set of science targets.

Each target belongs to one of a limited set of target types.

For each type k, define a value function:

```
V_k(n)
```

where n is the number of targets of type k already visited.

The functions are:

- monotonically increasing;
- diminishing in marginal value.

Therefore:

```
delta_V_k(n) =
    V_k(n + 1) - V_k(n)
```

decreases as n grows.

A trajectory's science value can be expressed approximately as:

```
total_value =
    sum over target types k of V_k(n_k)
```

This is much simpler than retaining arbitrary trajectory history.

The rover still needs a visited-target set so that an individual physical target cannot be collected repeatedly.

______________________________________________________________________

# 5. Core Algorithm Family 1: Static Grid Planning

## Applicability

Best for:

- point-to-point planning;
- static configuration space;
- deterministic rover.

Also useful as a component of more complex planners.

## Recommended Algorithm

Use A\* or Dijkstra on the raster.

A\* should normally be preferred for point-to-point queries.

Possible edge costs include:

- distance;
- slope-dependent traversal time;
- roughness penalty;
- terrain-dependent speed;
- fixed hazard penalties.

A basic admissible heuristic is Euclidean distance divided by maximum rover speed.

A stronger heuristic can be obtained from a reverse terrain-aware search from the goal.

For repeated queries, reverse Dijkstra fields may be worth caching.

## Additional Uses

This planner can also provide:

- static target-to-target costs;
- target-to-target paths;
- lower bounds for more complex algorithms;
- reverse travel-time heuristic fields;
- CPU reference solutions.

______________________________________________________________________

# 6. Core Algorithm Family 2: Hierarchical A\*-Guided Patch Propagation

## Applicability

Best for:

- point-to-point planning;
- time-varying sun/communications;
- long traverses crossing many epochs;
- optional battery/SOC constraints.

This is the natural continuation of the existing GridRunner-style algorithm.

## State-Space Interpretation

The planner propagates through a discretized space-time lattice.

Without explicit battery state:

```
state approximately =
    (x, y, t)
```

With power:

```
state approximately =
    (x, y, t, SOC)
```

Time remains discretized using the existing planning epoch structure.

Propagation can either:

- move spatially;
- remain at the same location while advancing time.

______________________________________________________________________

## Two-Level Propagation

### Inner Level

The terrain is divided into rectangular patches.

Within a patch:

- propagate states repeatedly;
- continue relaxation until local quiescence;
- detect whether boundary states changed.

### Outer Level

Maintain active patches in a CPU-side priority queue.

Each patch gets an optimistic completion priority.

Process the best N patches concurrently.

Conceptually:

```
priority(P) =
    g(P) + h(P)
```

where:

```
g(P)
```

represents the best currently known arrival cost in the patch, and:

```
h(P)
```

is an optimistic estimate of remaining travel time.

Initially:

```
h(P) =
    minimum Euclidean distance from patch to goal
    / maximum rover speed
```

The heuristic deliberately ignores:

- sunlight;
- communications;
- battery;
- charging.

It is therefore optimistic.

______________________________________________________________________

## Reverse-Heuristic Improvement

A stronger optional heuristic can be precomputed by running a reverse terrain-only shortest-path search from the destination.

It may respect:

- static slope exclusions;
- roughness;
- static obstacles;
- terrain-dependent speed.

It should ignore:

- sun;
- communications;
- battery;
- waiting.

The resulting field estimates the minimum possible remaining travel time under unrealistically favorable dynamic conditions.

This can replace Euclidean distance in the patch priority.

______________________________________________________________________

## N-Parallel Patch Processing

At each global iteration:

1. remove the best N patches from the priority queue;
1. process them concurrently;
1. relax each patch locally;
1. detect changed boundaries;
1. activate affected neighboring patches;
1. recompute priorities;
1. continue.

N controls the tradeoff between:

- strict best-first behavior;
- GPU occupancy;
- speculative propagation.

This is not classical A\*.

A useful description is:

> Hierarchical A\*-guided, batched, label-correcting patch propagation.

______________________________________________________________________

# 7. Point-to-Point: Static Environment, Nominal Rover

## Recommended Approach

Use:

```
A* / Dijkstra
```

This should be the simplest point-to-point planner.

It should support:

- static terrain;
- slope;
- roughness;
- crater masks;
- fixed exclusions.

Use this implementation as a reference for more complicated planners.

______________________________________________________________________

# 8. Point-to-Point: Dynamic Environment, Nominal Rover

## Recommended Approach

Use:

```
Hierarchical A*-Guided Patch Propagation
```

State:

```
(x, y, time)
```

The planner advances through successive two-hour environment epochs.

Configuration space may incorporate:

- slope;
- sun;
- communications.

Trips may span many epochs.

Do not reduce a multi-day traverse to a single time-dependent target-to-target edge.

The raster planner itself handles the evolving configuration space.

______________________________________________________________________

# 9. Point-to-Point: Dynamic Environment + Power, Nominal Rover

## Recommended Approach

Use the same:

```
Hierarchical A*-Guided Patch Propagation
```

with battery state added to the physical transition model.

Track:

- SOC;
- drive energy;
- stationary consumption;
- solar generation;
- charging;
- waiting.

Enforce:

```
SOC >= minimum_SOC
```

Initially avoid introducing large Pareto sets of slightly different arrival-time/SOC states unless experiments demonstrate that they are necessary.

The objective can remain earliest feasible arrival.

______________________________________________________________________

# 10. Point-to-Point: Performance Variation

There are two distinct problems.

## 10.1 Evaluate a Fixed Route

First compute a nominal route.

Then execute that route repeatedly under sampled performance variations.

Estimate quantities such as:

```
expected arrival time
arrival-time variance
probability of SOC violation
probability of reaching the destination
expected energy margin
```

This is:

```
Monte Carlo route robustness evaluation
```

No new low-level path planner is required.

______________________________________________________________________

## 10.2 Adaptive Replanning Under Performance Variation

If deviations are large enough that the rover should alter its route, use a receding-horizon policy.

At intervals:

1. observe the actual rover state;
1. invoke the appropriate deterministic point-to-point planner;
1. execute only part of the resulting plan;
1. realize performance variation;
1. update actual state;
1. replan.

The deterministic guidance engine may be:

- A\* in static terrain;
- DynamicPatchPlanner in a dynamic environment;
- power-aware DynamicPatchPlanner when SOC matters.

This can be called:

```
stochastic receding-horizon point-to-point planning
```

______________________________________________________________________

# 11. Point-to-Point: Faults and Degraded Modes

A fixed route may be inadequate if faults change future rover capabilities.

Use a policy that replans from the actual post-fault state.

State may include:

```
position
time
SOC
rover_mode
```

At each replanning point:

1. evaluate current rover mode;
1. update mobility/power/communications models;
1. generate new guidance;
1. execute for a short horizon;
1. sample or realize subsequent events;
1. repeat.

Candidate evaluation metrics include:

```
probability of reaching goal
expected arrival time
lower-tail arrival performance
probability of SOC violation
probability of entering unrecoverable state
```

The deterministic physical planner remains a reusable inner component.

______________________________________________________________________

# 12. Core Algorithm Family 3: Static Orienteering

## Applicability

Best for:

- science-target planning;
- static configuration space;
- nominal rover or simple travel-cost uncertainty.

## Step 1: Build Target Graph

Precompute terrain-aware shortest-path costs between relevant science targets.

For targets i and j:

```
C[i, j] =
    static minimum travel cost
```

Also retain paths if useful.

The raster planning problem is therefore reduced to a target graph.

______________________________________________________________________

## Step 2: Solve Orienteering Problem

Choose:

- which targets to visit;
- in which order;

to maximize science value within mission constraints.

Because rewards have diminishing marginal value by target type:

```
marginal_value(target j) =
    V_type(j)(n + 1) - V_type(j)(n)
```

The incremental utility of another target changes as the route accumulates observations of that type.

## Recommended Practical Solver

Use:

```
Large-Neighborhood Search
or
Iterated Local Search
```

Candidate modifications include:

- insert target;
- remove target;
- replace target;
- swap target order;
- 2-opt;
- remove several targets and reinsert;
- replace one target type with another.

A simple insertion heuristic can compare:

```
marginal science gain
/
incremental travel cost
```

For small problems, also consider an exact or bounded-exact solver such as CP-SAT, MILP, or branch-and-bound as a validation oracle.

______________________________________________________________________

# 13. Dynamic Orienteering

Static target-to-target edge costs no longer apply when sun and communications change during travel.

A target-to-target trip may span many two-hour epochs.

Therefore avoid assuming:

```
C[i, j, epoch]
```

is sufficient.

The actual route may change repeatedly while travelling between two science targets.

Two possible approaches remain attractive.

______________________________________________________________________

# 14. Dynamic Orienteering Approach A: Low-Level Space-Time Transition Oracle

The high-level orienteering solver asks a low-level dynamic raster planner for transitions.

Conceptually:

```
transition(
    source_target,
    destination_target,
    departure_time
)
```

returns:

```
arrival_time
path
feasibility
```

For the power-aware case:

```
transition(
    source_target,
    destination_target,
    departure_time,
    starting_SOC
)
```

may return:

```
arrival_time
ending_SOC
path
```

or a small set of operationally distinct alternatives.

Because this query may itself require a multi-day raster propagation, calls should be cached and reused.

______________________________________________________________________

# 15. One-to-Many Dynamic Propagation

Instead of separately computing:

```
A -> B
A -> C
A -> D
A -> E
```

run one space-time propagation from A.

Record when each interesting science target is reached.

Conceptually:

```
propagate_from(
    source_target,
    departure_time
)
```

returns:

```
target B -> arrival information
target C -> arrival information
target D -> arrival information
...
```

This is well matched to flood/patch propagation because that algorithm is naturally single-source and many-destination.

With power, each reached target may optionally retain a small nondominated set of:

```
(arrival_time, SOC)
```

arrival conditions.

______________________________________________________________________

# 16. Dynamic Orienteering Approach B: Utility-Guided Receding-Horizon Orienteering

This is a different algorithm family that may scale better for science optimization.

Instead of globally solving the combinatorial orienteering problem, simulate a rover that repeatedly makes locally informed decisions.

At each decision point:

1. locate nearby unvisited science targets;
1. determine the marginal science value of each target;
1. estimate travel cost heuristically;
1. perform short or one-step lookahead;
1. select the most promising action;
1. advance using the full physical simulator;
1. repeat.

A simple heuristic target value might resemble:

```
utility_j =
    marginal_science_value_j
    /
    distance_j^alpha
```

Euclidean distance may initially be used.

The growing simulated traverse itself accounts for:

- real slope;
- sun;
- communications;
- battery;
- faults if enabled.

Therefore the heuristic does not need to model all physical effects.

A useful conceptual separation is:

> The heuristic estimates desirability; the simulator determines feasibility.

______________________________________________________________________

# 17. Nearby-Target Lookup

The receding-horizon rover should not evaluate every science target at every movement step.

Use a spatial index to retrieve the N nearest candidate targets.

Possible structures:

- k-d tree;
- quadtree;
- uniform spatial grid;
- spatial hash.

For GPU implementations, a uniform grid or spatial hash may be more suitable than a branch-heavy tree.

Each rover maintains its own visited-target bitset.

When nearby targets are returned:

- ignore targets already visited by that rover;
- compute current marginal reward from target type;
- evaluate only a limited candidate set.

______________________________________________________________________

# 18. One-Step Lookahead

Do not necessarily commit immediately to one science target.

For each feasible immediate rover action:

1. predict the resulting rover state;
1. recompute approximate target utilities from that predicted state;
1. aggregate candidate utilities;
1. choose the action with the best predicted value.

Candidate actions may include:

- 8-connected spatial moves;
- WAIT;
- short macro-moves if later added.

Possible aggregate functions include:

```
maximum target utility

sum of top K target utilities

softmax-like combination
```

A max tends to select one clear target.

A sum or top-K combination can favor motion toward regions containing several useful targets.

______________________________________________________________________

# 19. Parallel Receding-Horizon Orienteering

Run many simulated rovers independently.

Each rover carries approximately:

```
position
time
SOC if needed
visited-target bitset
count per target type
accumulated science value
policy parameters
fault/performance state
deterministic random identity
```

Different simulated rovers can use:

- different utility parameters;
- different candidate-target counts;
- different random perturbations;
- different stochastic action choices;
- different lookahead weights;
- different risk attitudes.

This creates a population of candidate trajectories.

A useful name is:

> Parallel Utility-Guided Receding-Horizon Orienteering

If stochastic choices are included:

> Parallel Stochastic Receding-Horizon Orienteering

______________________________________________________________________

# 20. Stochastic Rover Simulation

Performance variation and faults can be incorporated directly into the simulated-rover population.

Each simulated rover represents a particular realization of:

```
policy parameters
performance variation
fault realization
random decisions
```

The simulator can estimate:

```
expected science return
variance of science return
science-return quantiles
probability of reaching required destinations
probability of SOC violation
expected delays
fault-conditioned performance
```

This turns trajectory generation into policy evaluation under uncertainty.

______________________________________________________________________

# 21. Optimize Policies Rather Than Fixed Trajectories

Once faults and delays cause replanning, the object being evaluated is no longer simply one fixed trajectory.

It is a policy:

```
policy(current mission state)
    -> next action
```

The policy generates different realized trajectories under different fault and environment realizations.

The objective may therefore be:

```
maximize expected science return
```

possibly subject to constraints such as:

```
probability(SOC < 20%) < threshold

probability(mission completion) > threshold
```

or using risk-sensitive objectives such as:

```
expected value
minus
downside-risk penalty
```

______________________________________________________________________

# 22. Deterministic Replay

Millions of simulated rovers should not normally store complete trajectories.

Instead make every simulation reproducible.

A simulation should be determined by:

```
initial state
policy parameters
simulation ID / random key
scenario parameters
```

During a large ensemble run, retain only compact summaries such as:

```
final science value
completion time
minimum SOC
success/failure
termination reason
target counts
target sequence if inexpensive
simulation ID
policy ID
```

Interesting simulations can then be replayed deterministically with detailed logging enabled.

A counter-based random-number scheme is attractive because random events can be generated from values such as:

```
global_seed
simulation_id
timestep
event_type
```

This avoids dependence on GPU scheduling order.

______________________________________________________________________

# 23. Adaptive Population-Based Search

Do not necessarily run every simulated rover to mission completion.

Instead allocate simulation effort adaptively.

Conceptually:

```
large diverse population
    ->
partial simulation
    ->
estimate promise
    ->
discard weak candidates
    ->
retain multiple promising families
    ->
clone / perturb / continue
    ->
repeat
```

This can greatly increase the amount of computation devoted to promising parts of trajectory and policy space.

______________________________________________________________________

# 24. Partial-Trajectory Scoring

A partial trajectory should not be ranked solely by science collected so far.

A rover may currently have little science return because it is travelling toward a valuable distant region.

A partial score may therefore combine:

```
science already obtained
+
approximate future opportunity
-
risk penalties
```

Future opportunity can use:

- nearby unvisited targets;
- marginal science value;
- Euclidean distance;
- approximate remaining mission time.

This value-to-go estimate need not be physically exact.

It is used to decide where additional simulation effort is most promising.

______________________________________________________________________

# 25. Preserve Families of Solutions

Do not simply retain the globally highest-scoring partial trajectories.

That can prematurely eliminate qualitatively different strategies.

For example:

```
family A:
    many nearby moderate-value targets

family B:
    long initial transit to a rich distant region
```

Both may deserve continued simulation.

Possible family descriptors include:

- coarse geographic region;
- current target;
- next intended target;
- sequence of target types;
- science-count vector;
- coarse target sequence;
- policy parameter cluster.

This resembles diversity preservation or niching in evolutionary search.

______________________________________________________________________

# 26. Successive Halving / Statistical Racing

An initial adaptive approach can be simple.

Example:

```
1,000,000 partial simulations
    ->
retain 200,000
    ->
simulate farther
    ->
retain 40,000
    ->
simulate farther
    ->
retain 8,000
    ->
complete missions
```

Likewise, do not evaluate every candidate policy under thousands of fault scenarios.

For example:

```
many policies
    x
few fault seeds
```

then:

```
fewer promising policies
    x
many more fault seeds
```

This is statistical racing.

______________________________________________________________________

# 27. Branching Promising Partial States

A promising partial rover state can be cloned.

Example:

```
                partial mission
                     |
       +-------------+-------------+
       |             |             |
    variant A     variant B     variant C
```

Variants might differ in:

- selected next target;
- utility parameters;
- stochastic action;
- fault realization;
- exploration temperature.

The prefix need not be replayed.

The descendants continue directly from the saved state.

This introduces a beam-search-like structure without abandoning forward simulation.

______________________________________________________________________

# 28. Cross-Entropy / Evolutionary Policy Search

The rover policy can be parameterized by quantities such as:

```
distance exponent
target candidate count
utility aggregation rule
stochastic temperature
SOC conservatism
waiting preference
lookahead weighting
target-type weighting
```

Sample policy parameters from a broad distribution.

Run simulations.

Select elite policies.

Update the sampling distribution toward those elites.

Repeat.

This is closely related to the Cross-Entropy Method.

Because the trajectory simulator is discrete and nonlinear, derivative-free policy optimization is attractive.

Maintain multiple policy families or clusters if necessary to prevent the population from collapsing onto one mediocre compromise between several genuinely different good strategies.

______________________________________________________________________

# 29. Rare Faults and Importance Sampling

Naive Monte Carlo may poorly estimate very rare but consequential failures.

Potential improvement:

- deliberately oversample important fault conditions;
- retain probability weights;
- estimate unbiased mission statistics using weighted samples.

This is importance sampling.

It can be added after the basic stochastic simulator is working.

______________________________________________________________________

# 30. Suggested Small Set of Library Components

The library does not need a separate planner class for every row of the problem matrix.

A compact design could contain approximately the following major components.

## StaticGridPlanner

Responsibilities:

- A\*;
- Dijkstra;
- static point-to-point planning;
- static target-to-target paths;
- reverse travel-time fields;
- reference solutions.

______________________________________________________________________

## DynamicPatchPlanner

Responsibilities:

- time-varying configuration space;
- two-hour environment epochs;
- hierarchical patch propagation;
- local patch relaxation to quiescence;
- A\*-guided patch priority;
- best-N parallel patch processing;
- optional battery/SOC;
- waiting/charging;
- CPU and/or Numba CUDA implementation.

This should implement the Hierarchical A\*-Guided Patch Propagation algorithm.

______________________________________________________________________

## StaticOrienteeringPlanner

Responsibilities:

- science target graph;
- diminishing-return reward functions;
- target subset/order optimization;
- LNS / iterative local search;
- optional small exact solver for validation.

This planner relies on static target-to-target costs from StaticGridPlanner.

______________________________________________________________________

## RecedingHorizonOrienteeringPolicy

Responsibilities:

- nearby-target lookup;
- visited-target bitsets;
- science-count state;
- marginal-value calculation;
- one-step or short-horizon lookahead;
- utility evaluation;
- target/action selection;
- optional stochastic selection.

The physical transition simulator determines whether selected actions are feasible.

______________________________________________________________________

## RoverSimulator

Responsibilities:

- execute policies;
- terrain transitions;
- sun/communications;
- SOC;
- charging;
- delays;
- performance variation;
- faults;
- deterministic replay.

______________________________________________________________________

## EnsemblePlanner

Responsibilities:

- run many simulated rovers;
- Monte Carlo statistics;
- policy comparison;
- deterministic simulation identities;
- adaptive simulation allocation;
- successive halving;
- statistical racing;
- trajectory-family preservation;
- optional CEM/evolutionary policy optimization;
- selection of simulations for replay.

______________________________________________________________________

# 31. Recommended Problem-to-Algorithm Mapping

## Point-to-Point, Static Environment, Nominal Rover

Use:

```
StaticGridPlanner
A*
```

______________________________________________________________________

## Point-to-Point, Dynamic Sun/Comm, Nominal Rover

Use:

```
DynamicPatchPlanner
Hierarchical A*-Guided Patch Propagation
```

without SOC if power is guaranteed.

______________________________________________________________________

## Point-to-Point, Dynamic Sun/Comm + Power, Nominal Rover

Use:

```
DynamicPatchPlanner
Hierarchical A*-Guided Patch Propagation
```

with SOC, solar generation, waiting, and charging.

______________________________________________________________________

## Point-to-Point, Any Environment, Performance Variation

First support:

```
nominal route
    +
Monte Carlo route robustness evaluation
```

For variations large enough to justify replanning, use:

```
receding-horizon point-to-point policy
    +
deterministic physical planner
    +
stochastic RoverSimulator
```

______________________________________________________________________

## Point-to-Point, Any Environment, Faults

Use:

```
adaptive receding-horizon point-to-point policy
    +
fault-aware RoverSimulator
    +
ensemble evaluation
```

Replan when the fault changes actual rover capabilities.

______________________________________________________________________

## Science Planning, Static Environment

Use:

```
StaticOrienteeringPlanner
    +
precomputed StaticGridPlanner target-to-target costs
```

Primary practical algorithm:

```
Large-Neighborhood Search / Iterated Local Search
```

______________________________________________________________________

## Science Planning, Dynamic Environment

Two useful approaches should be considered.

### More global / expensive approach

Use:

```
target-level orienteering
    +
DynamicPatchPlanner as transition oracle
```

Prefer one-to-many propagation where practical.

### Highly scalable approximate approach

Use:

```
RecedingHorizonOrienteeringPolicy
    +
dynamic physical simulation
```

This is likely the more natural basis for very large simulated populations.

______________________________________________________________________

## Science Planning, Dynamic Environment + Power

Use:

```
RecedingHorizonOrienteeringPolicy
    +
SOC-aware RoverSimulator
```

The heuristic science guidance need not understand battery in detail.

The actual state transition enforces battery feasibility.

______________________________________________________________________

## Science Planning With Performance Variation and Faults

Use:

```
Parallel Stochastic Receding-Horizon Orienteering
    +
RoverSimulator
    +
EnsemblePlanner
```

Evaluate:

```
expected science return
downside science return
mission-completion probability
SOC violation probability
robustness to faults
```

Adaptive population search can increasingly focus computation on promising policy and trajectory families.

______________________________________________________________________

# 32. Overall Architecture

A useful conceptual layering is:

```
Ensemble / uncertainty optimization
    |
    | faults, delays, performance distributions
    |
    v
Mission-level policy
    |
    | point-to-point policy
    | or science/orienteering policy
    |
    v
Deterministic physical planning / simulation
    |
    | StaticGridPlanner
    | or DynamicPatchPlanner
    |
    v
Physical/environment models
    |
    | terrain
    | slope
    | roughness
    | craters
    | sunlight
    | communications
    | battery
    | rover modes
```

The important architectural goal is to keep these responsibilities separable.

The science-target policy should not need to understand GPU patch propagation.

The DynamicPatchPlanner should not need to understand diminishing-return science objectives.

The fault simulator should not need to know how A\* is implemented.

The ensemble optimizer should operate on policy parameters, simulation states, and outcome statistics rather than terrain rasters.

______________________________________________________________________

# 33. Overall Recommendation

Avoid implementing a large collection of unrelated trajectory planners.

The problem space can be covered reasonably well by three primary algorithm families:

1. **A* / Dijkstra*\*

   - static deterministic raster planning;
   - static target-to-target costs;
   - heuristic generation;
   - reference solutions.

1. **Hierarchical A*-Guided Patch Propagation*\*

   - dynamic sun/communications;
   - long multi-epoch traverses;
   - optional SOC and solar charging;
   - CPU/GPU parallel implementation.

1. **Orienteering / Receding-Horizon Policy Search**

   - static orienteering via target-graph optimization;
   - dynamic science planning via utility-guided receding-horizon policies;
   - stochastic population rollout for faults and performance uncertainty.

Then provide a reusable:

4. **Simulation and Ensemble Layer**
   - deterministic replay;
   - Monte Carlo fault/performance simulation;
   - expected science-return estimation;
   - adaptive allocation of simulation effort;
   - population/family preservation;
   - optional policy optimization.

This should cover the important early lunar mission-planning cases without creating unnecessary algorithm proliferation.
