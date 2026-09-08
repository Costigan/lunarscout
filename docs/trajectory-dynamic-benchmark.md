# Dynamic Planner Comparison

Date: 2026-09-07 (America/Los_Angeles).

This report records the Phase 3D comparison required by the
[implementation plan](TRAJECTORY-API-IMPLEMENTATION-PLAN.md). Refer to the
[trajectory API design](trajectory-api-design.md) for the benchmark rationale.
The complete machine-readable output is
[`trajectory-dynamic-benchmark.json`](trajectory-dynamic-benchmark.json).

## Method

Command:

```text
.venv/bin/python scripts/benchmark_trajectory_dynamic.py --repeats 3 \
  --output docs/trajectory-dynamic-benchmark.json
```

The deterministic synthetic matrix varies raster size, timeline length, sample
spacing, configuration-change frequency, and required waiting. Each case runs
GridRunner and safe-interval planning on the same prebuilt in-memory occupancy
timeline and compares both answers with the independent exact oracle.

Runtime is the median of three untraced runs. Peak host allocation is one
separate `tracemalloc` run and therefore covers Python-tracked planner
allocations, not total process RSS. Provider materialization is excluded from
both measurements. The run used Python 3.12.3 and NumPy 2.2.6 on
Linux 7.0.11 x86-64.

## Results

| Case | GridRunner time (s) | Safe-interval time (s) | Grid expanded | Safe expanded | Blocks activated | Safe intervals | Exact cost (h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| short, no wait | 0.420 | 0.0128 | 2,704 | 100 | 5 | 100 | 4.5 |
| long, sparse changes | 4.077 | 0.0131 | 7,970 | 100 | 4 | 124 | 4.5 |
| long, frequent changes | 6.669 | 0.0151 | 15,819 | 125 | 7 | 670 | 4.5 |
| larger, sparse changes | 3.140 | 0.0172 | 14,947 | 442 | 13 | 526 | 9.5 |
| required wait | 0.0631 | 0.0120 | 406 | 12 | 2 | 12 | 3.5 |

All ten optimized-planner answers agreed with the exact oracle on reachability
and earliest-arrival cost. GridRunner reactivation counts ranged from zero to
four. Python-tracked peak allocations ranged from 30,573 to 505,243 bytes for
GridRunner and from 12,787 to 265,539 bytes for safe-interval search; exact
values and state-relaxation counts are in the JSON artifact.

## Interpretation and limits

Safe-interval search is substantially faster in this small CPU/Python matrix,
especially when long timelines compress into few safe intervals. Frequent
changes increase its interval count and work, as expected, but it remains ahead
in these cases. This evidence supports retaining both exact algorithms and
using safe intervals as the practical CPU option for these structures.

These measurements do not establish mission-scale performance, include dynamic
provider I/O, compare compiled kernels, or predict a future parallel/CUDA
GridRunner implementation. They are regression evidence for the present
implementations, not a permanent automatic-selection rule; `backend="auto"`
continues to preserve the caller's selected algorithm.
