# Numba Horizon Phase 4D Hierarchical Traversal

Phase 4D ports the current production factor-four maximum pyramid and
hierarchical ray traversal to an independent NumPy CPU oracle and a Numba CUDA
kernel. It does not yet implement the full patch/subpatch/multi-DEM launch.

## Parity Result

All levels of 29 captured C# DEM pyramids match byte-for-byte, including odd
dimensions, non-finite values, the `-20000` validity cutoff, and the `-32000`
invalid-block sentinel.

The C# and Numba implementations now use the maximum valid elevation from the
current, right, bottom, and bottom-right pyramid cells for hierarchy culling.
When that bound prevents culling at level zero, traversal retains the original
adaptive sampling approximation and uses a 1 mm cell-exit nudge. The dense
bilinearly interpolated surface is not treated as terrain ground truth.

For the corrected production outer-DEM ray at azimuth index 360:

- C# and Numba CUDA both return slope `0.1659297943`;
- the NumPy CPU oracle differs by `1.49e-8` slope;
- all 932 CPU and CUDA trace rows choose the same level, cell, and action as
  C#;
- all recorded values in all 932 CUDA trace rows match C# exactly;
- the action-derived counters are 924 ray iterations, 3 level-0 samples, and
  921 culled blocks; and
- hierarchy slope `0.1659297943` is above the 1.2 m fixed-step level-0
  reference slope `0.1638278216`.

Explicit device `float32` return values are required. Without them, Numba can
promote helper arithmetic and move an exact cell-boundary result by one ULP,
which changes a traversal decision. The checked kernel retains the C#
arithmetic and branch sequence at that boundary.

## Corrected Bilinear-Boundary Defect

The bounded fixture places a 150 m, one-pixel obstacle immediately across the
next bilinear cell boundary. Results are:

- adaptive level-0 slope: `0.2202198207`;
- 1.2 m fixed-step level-0 slope: `0.2562785745`;
- corrected C# hierarchy slope: `0.25802016`;
- corrected NumPy hierarchy slope: `0.2580201626`;
- corrected Numba CUDA hierarchy slope: `0.2580201626`.

Before the four-cell correction, C# and Numba returned about `0.000713` because
the culling bound omitted elevations that bilinear sampling could use. The
four-cell bound prevents that invalid cull. The retained adaptive sampler may
still skip an interior maximum of the artificial bilinear surface; that is an
intentional terrain approximation rather than a culling-safety claim.

A separate ten-case diagnostic matrix covers eight directions, diagonal
interior maxima, odd dimensions, coarse factor-four mip traversal, raster
edges, and invalid neighbors. Its dense 1.2 m bilinear comparison is recorded
to characterize the approximation, but it is not an acceptance gate. C#/Numba
parity and downstream illumination error remain the correctness gates. See
`docs/numba-horizon-phase-4-hierarchy-safety.json`.

## Reproduction

Refresh the C# CUDA artifact first, then the Python/Numba report:

```bash
dotnet run \
  --project scripts/numba_horizon/CSharpPhase4DHierarchyCapture.csproj \
  -- tests/data/numba_horizon/phase4d_production_segments.json

env PYTHONPATH=src \
  /e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/capture_phase4d_hierarchy.py
```

The machine-readable results are in
`docs/numba-horizon-phase-4d-hierarchy.json`.
