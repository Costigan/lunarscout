# Numba Horizon Phase 4B Fixed-Step Diagnostics

Phase 4B has begun with a deliberately conservative 1.2 m step and level-0
sampling only. Adaptive stepping and hierarchy are not implemented.

The CUDA kernel evaluates the Phase 3 fitted pixel path, switches from direct
parameter distance to planar-to-chord correction at 500 m, performs clamped
bilinear level-0 sampling, applies the current production near/far slope
formulas, and records every step. Selected flat and obstacle rays use the
immutable C# Phase 1 DEM and segment artifacts.

The real RTX 5090 trace is compared field by field with an independent CPU
implementation for parameter distance, true distance, pixel X/Y, elevation,
sample slope, and running maximum. Small CPU/GPU differences come from repeated
float32 addition and GPU fused arithmetic. At one kilometer, measured bounds
are 0.01 m for true distance, 0.0003 pixel for fitted position, 0.005 m across
the synthetic obstacle interpolation gradient, and `2e-5` for slope.

The report deliberately separates this production-arithmetic comparison from
the C# `ReferenceRayEmulator`. That reference oversamples three azimuth offsets
and uses exact spherical slope at every distance. The production CUDA algorithm
uses a flat-earth slope below 500 m. Consequently:

- the fixed-step production port reports a zero maximum on flat near terrain;
- the C# reference reports a small negative curvature slope;
- the aligned obstacle case also differs in its maximum because fixed-step
  bilinear sampling and reference oversampling are not the same calculation.

No tolerance is used to hide those algorithmic differences. A separate direct
vector implementation reproduces the immutable C# `ReferenceRayEmulator`
traces sample by sample to `1e-9`, then evaluates exact reference geometry at
every GPU fixed-step distance. The report records fitted-path, sampled-height,
and production-versus-exact-slope differences separately.

Regenerate `docs/numba-horizon-phase-4b-fixed-step.json` with:

```bash
env PYTHONPATH=src \
  /e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/capture_phase4b_fixed_step.py
```
