# Trajectory Planning Support Review

Date: 2026-10-13.

This review summarizes the traverse-planning cases described in the design
documents, what `ls.trajectory` implements today, what remains unimplemented,
and recommended next steps. It is a point-in-time assessment; the progress
tracker remains
[`TRAJECTORY-API-IMPLEMENTATION-PLAN.md`](TRAJECTORY-API-IMPLEMENTATION-PLAN.md)
and the rationale/contract source remains
[`trajectory-api-design.md`](trajectory-api-design.md).

Verification evidence for this review: `.venv/bin/python -m pytest tests/trajectory -q` reported 189 passed on 2026-10-13. Performance figures are
quoted from [`trajectory-dynamic-benchmark.md`](trajectory-dynamic-benchmark.md).
The site-size example in Section 4 is derived arithmetic, not a measurement.

**Summary.** Point-to-point planning is implemented on CPU for static,
time-varying, and battery-aware cases, with exact reference oracles behind the
optimized planners. Science-target planning, robustness simulation, adaptive
replanning, and every CUDA trajectory planner are designed but not built. The
largest practical gap is problem size: all three time-varying planners are
limited to roughly 5 million interval-cell states.

## 1. Planning cases described

[`trajectory-planning.md`](trajectory-planning.md) and the API design describe
the following cases.

| Dimension           | Cases described                                                                                                                                                                                                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Configuration space | Static terrain (slope, slip, keep-out cells); time-varying terrain (sunlight, Earth visibility for communications); time-varying terrain plus battery state of charge (SOC)                                                                                                                                   |
| Rover execution     | Deterministic nominal rover; performance variation (speed, slip, power); faults and degraded modes                                                                                                                                                                                                            |
| Mission objective   | Point-to-point; science targets and regions with time windows, service/dwell time, and diminishing marginal value (orienteering)                                                                                                                                                                              |
| Algorithm families  | Dijkstra/A\*; GridRunner (A\*-guided patch propagation); safe-interval search; exact multi-label and greedy single-label SOC search; static and dynamic orienteering; Monte Carlo ensembles; receding-horizon replanning; policy, cross-entropy, and evolutionary search; importance sampling for rare faults |

## 2. Implemented capabilities

All public names live under `ls.trajectory`.

| Capability               | Public entry point                                           | Algorithms                                                                                           | Notes                                                                                                     |
| ------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Static travel-time field | `static_travel_time`                                         | Dijkstra; compiled Numba block engine for grids of 16,384 or more cells                              | Signed-slope slip; rotated, skewed, and non-metre projected grids                                         |
| Static path              | `static_path`                                                | A\* (pure Python)                                                                                    | Optimal cost; no unique tie-break is promised                                                             |
| Time-varying path        | `dynamic_path`                                               | `gridrunner` (default) and `safe_interval`, both exact                                               | Earliest arrival with waiting; half-open UTC interval timeline                                            |
| Environment providers    | Sunlight, Earth-elevation, and configuration-space providers | Explicit arrays, horizon-backed sunlight, SPICE-backed Sun vectors, threshold and all-of composition | Missing data raises a structured error instead of becoming darkness, outage, or blocked terrain           |
| Battery-aware path       | `soc_path`                                                   | `exact` (small problems) and `greedy` (default, approximate)                                         | Every returned path is replayed independently; the result reports whether it is complete and optimal      |
| Power models             | `SolarPowerModel`, `BatteryModel`, `RoverPowerModel`         | Not applicable                                                                                       | Orientation-independent solar output; ideal (100%) battery efficiency only; constant drive and idle loads |

Each optimized planner is compared against an independent exact reference on
small cases. Contracts are frozen in decision records before implementation,
namespace import has no side effects, and backend selection never substitutes a
different algorithm.

## 3. Not yet implemented

| Plan phase or item                                                            | Status                                                                                                                                                                                                              |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 5: CUDA block engine (GridRunner, greedy SOC)                           | Marked next in the plan. `backend="cuda"` always raises `trajectory_backend_unavailable`.                                                                                                                           |
| Phase 6: science target, region, and activity planning (orienteering)         | Not started. The public data structures remain open design question 18.                                                                                                                                             |
| Phase 7: robustness simulation, delays, faults, adaptive replanning           | Not started.                                                                                                                                                                                                        |
| Policy, MDP/POMDP, and evolutionary search (planning document Sections 21-29) | Research direction only; explicitly deferred.                                                                                                                                                                       |
| Sun-relative and hazard mobility in public planners                           | A private compiled dynamic-mobility bundle exists and is tested, but `dynamic_path` does not use it.                                                                                                                |
| Power realism                                                                 | No non-ideal battery efficiency, charging attitude, sun-elevation solar dependence, or heater/thermal loads.                                                                                                        |
| Open housekeeping items                                                       | Single-source algorithm/backend support table; progress and cancellation boundaries; end-to-end bounded resources; non-metre projected-grid example; installed-wheel smoke test; complete public error-code matrix. |

## 4. Recommendations

Ordered by expected value.

1. **Raise the problem-size ceiling.** All three time-varying planners reject
   problems where intervals x cells exceeds 5,000,000
   (`src/lunarscout/trajectory/_dynamic_gridrunner.py`,
   `src/lunarscout/trajectory/_safe_interval.py`, and
   `src/lunarscout/trajectory/soc.py`). An hourly sunlight timeline over 14
   days has 336 intervals, which caps the grid at about 15,000 cells: roughly
   120 x 120 cells, or 2.4 km square at 20 m. That is too small for realistic
   polar traverses. Safe-interval search does not need the full interval-cell
   timeline in memory; building per-cell intervals on demand from streamed
   provider windows, then compiling the search with Numba, would likely lift
   the limit further than CUDA GridRunner. Do this before Phase 5.
1. **Make `safe_interval` the default for `dynamic_path`, or document why
   not.** In the Phase 3D benchmark, pure-Python GridRunner ran 5 to 450 times
   slower than safe-interval search in every case (for example 6.669 s versus
   0.0151 s for long timelines with frequent changes). Both are exact, yet
   GridRunner is the default. Its expected advantage depends on a CUDA
   implementation that does not exist yet.
1. **Use the compiled block engine for `static_path` on large grids.** Only
   `static_travel_time` dispatches to the block engine; `static_path` always
   runs pure-Python A\*. For long traverses, callers currently do better by
   computing the full field and tracing back from the goal.
1. **Deliver a first Phase 6 slice before CUDA.** A directed target-to-target
   cost matrix built from `static_travel_time`, plus a heuristic orienteering
   solver with explicit termination metadata, uses only finished components and
   answers the most common mission-planning question: which targets, and in
   what order. It also provides the route substrate Phase 7 needs.
1. **Connect safe-haven products to traverse planning.** `generate_safe_havens`
   and `scripts/threshold_safe_havens.py` already produce safe-haven rasters. A
   configuration-space provider built from them would let `dynamic_path` and
   `soc_path` enforce "always within N hours of a safe haven," a common
   polar-rover operations constraint.
1. **Expose the compiled dynamic-mobility model through `dynamic_path`.** It is
   built and tested, but the public dynamic planners currently ignore
   Sun-relative and hazard effects on mobility.
1. **Add minimal power realism behind the existing contract gates.** Battery
   efficiency below 100% (currently rejected by `BatteryModel`), a
   charge-while-parked mode, and a sun-elevation term in solar output would
   make `soc_path` results usable for operations without changing its
   exact/greedy structure.
1. **Add search diagnostics and progress/cancellation.** Mission-scale CPU runs
   need expanded-state counts, proximity to state limits, and a cancellation
   hook (design open question 21 and implementation plan Section 3.4).
1. **Reconcile the planning documents with the code.**
   - Phase 0 dispatch checkboxes are unchecked, but `_dispatch.py` implements
     them.
   - Planned test files `test_dispatch.py`, `test_static_public.py`, and
     `test_dynamic_time_contract.py` do not exist.
   - The API design still spells SOC algorithms `soc_exact` and `soc_greedy`
     (Section 17); the code uses `exact` and `greedy`.
   - The plan's verification line (2,022 passed, one version-mismatch failure,
     2026-09-08) should be refreshed against current `main`.
