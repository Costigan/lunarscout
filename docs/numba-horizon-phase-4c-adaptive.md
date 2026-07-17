# Numba Horizon Phase 4C Adaptive Level-0 Traversal

Phase 4C ports current production adaptive stepping while retaining the Phase
4B 1.2 m fixed-step traversal as a control. Hierarchy remains disabled.

The port includes tangent-based one-pixel steps, horizon-margin steps, angular
steps, the quarter-step multiplier below 500 m, the half-resolution general
floor, and the 0.8-resolution primary-DEM floor beginning at 100 m. Tests cross
both the 100 m step-floor transition and the 500 m near/far slope transition.
Real CUDA traces match an independent CPU implementation within the measured
float32 bounds established in Phase 4B.

## Skipped-Terrain Finding

The selected one-pixel-wide 150 m obstacle exposes a significant difference
between adaptive and fixed traversal:

- fixed-step maximum slope: about `0.25628`;
- adaptive maximum slope: about `0.22022`;
- adaptive loss: about `0.03606` slope.

At 30 m/pixel, the primary-DEM far-step floor is 24 m. The adaptive samples
bracket the narrow peak without landing as close to its center as the 1.2 m
control, lowering the bilinearly sampled elevation and maximum slope. The JSON
report records the fixed horizon-setting row and neighboring adaptive rows.

This is an inherited consequence of the current C# stepping constants, not a
Numba-specific optimization or a silent algorithm change. It must be included
in the scientific error baseline. Hierarchical culling may not be credited for
or allowed to increase this pre-existing adaptive loss.

Regenerate `docs/numba-horizon-phase-4c-adaptive.json` with:

```bash
env PYTHONPATH=src \
  /e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/capture_phase4c_adaptive.py
```
