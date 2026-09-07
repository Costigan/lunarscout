# Phase 2 Static CPU Block Benchmark

Status: initial implementation evidence.
Date: 2026-09-07.

This report records the first complete-workflow comparison required by Phase 2
of the [trajectory implementation plan](TRAJECTORY-API-IMPLEMENTATION-PLAN.md).
The scientific and architectural rationale remains in the
[trajectory API design](trajectory-api-design.md).

## Environment and command

```text
Python 3.12.3
NumPy 2.2.6
Numba 0.66.0
Linux 7.0.11 x86_64
```

```bash
PYTHONPATH=src .venv/bin/python benchmarks/trajectory_static_cpu.py \
  --sizes 64 128 384 --block-sizes 4 8 16 32 --repeats 3
```

The 384x384 case is a 3x3 arrangement of the canonical 128x128 patch size.
Each case uses the same deterministic skewed affine grid, barriers, elevation,
signed-slope model, start cell, and full-field output. Kernel JIT warmup was
measured separately at 0.086 seconds. Peak process RSS was 226,740 KiB after all
cases; this process-wide high-water mark includes Python, NumPy, PyProj, Numba,
JIT state, reference runs, and every benchmark case.

For a first one-shot 64x64 call, JIT warmup plus block execution was about
0.0866 seconds versus 0.0763 seconds for reference Dijkstra, so the optimized
engine does not help that cold small case. At 128x128 the same cold-cost sum is
already below the measured reference runtime.

## Warm median results

| Grid | Reference Dijkstra | Block 4 | Block 8 | Block 16 | Block 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64x64 | 0.0763 s | 0.00162 s | 0.00093 s | 0.00080 s | 0.00086 s |
| 128x128 | 0.3059 s | 0.00753 s | 0.00470 s | 0.00559 s | 0.00992 s |
| 384x384 | 2.8198 s | 0.05932 s | 0.06151 s | 0.08766 s | 0.25527 s |

All compared fields had identical finite/unreachable masks and a maximum finite
travel-time delta of `0.0` hours. The Phase-2 test acceptance bound is
`rtol=1e-12`; exact equality is observed on these cases but is not required
across future Numba/compiler/platform combinations.

## Scheduler and memory observations

For 128x128, block sizes 4/8/16/32 required 2,985/753/191/51 activations. For
384x384 they required 22,620/7,450/1,996/712 activations. Larger blocks reduce
scheduler activity but spend more time iterating locally to quiescence. The
fastest block size is workload-sensitive; no universal optimum is claimed.

The 384x384 block results retained travel time plus two predecessor arrays and
block diagnostics, using approximately 3.5 MiB of directly counted result
state. Public field-only dispatch omits both predecessor arrays.

## Dispatch decision

- Keep public point-to-point `static_path` on the Phase-1 A* reference.
- Keep small `static_travel_time` calls on reference Dijkstra to avoid one-shot
  Numba initialization cost.
- Use lazy block-CPU field execution at 128x128 cells and above. The selected
  block size is 8x8: it was near the best result across measured sizes without
  the high activation count of 4x4 or the large-grid slowdown of 16x16/32x32.
- Import the private Numba engine only after public validation and only when the
  size threshold selects it. Namespace import and small CPU calls remain free
  of Numba initialization.

These thresholds and private dimensions are performance policy, not public API.
They should be retuned only with parity tests and updated complete-workflow
measurements.
