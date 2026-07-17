# Numba Horizon Phase 4 Real-Terrain CUDA Parity

This bounded validation compares the production-shaped Numba CUDA kernel with
the C# ILGPU CUDA kernel on a retained 512 by 512 LOLA terrain fixture. It uses
one hierarchy-enabled 16 by 16 patch, all 1,440 azimuth bins, one DEM pass, and
zero-metre observer elevation.

## Result

All 368,640 degree values are byte-for-byte identical. Tiny slope-level
arithmetic differences round to identical degrees. The hashes are:

- C# slopes: `4772be4dfe45efbb3acbcb10ee41ebea80389240a3ee4eaeea6c5608f6f0f715`;
- Numba slopes: `f74413cc8f8d11e87f1c12ad23b45bc3896daf55f46562aef00bd7ce3d1a4baf`;
- shared degrees: `385bd7a01923f7f338945b4fd8b2caf6feca43d189adbd53dc1cb67d912a2d86`.

There are no sentinel mismatches or non-finite values. Maximum, mean, median,
95th, 99th, and 99.9th percentile angular errors are all zero, as are the
signed bias and every spatial and azimuth grouping.

Three subsequent warm calls reproduce the same slope hash. Their diagnostic
elapsed times including transfers were 16.0, 15.0, and 14.8 ms; these are
stability observations, not a sustained-throughput claim. NVIDIA Compute
Sanitizer 2026.1.1 reports zero memcheck errors for the bounded validator; its
command and coverage are recorded in
`docs/numba-horizon-phase-4-cuda-memcheck.json`.

Python-generated ray segments differ from the separately captured C# segment
tensor by at most `8.31e-11` in any field, but produce the same final C# buffer.
The selected diagnostic ray also captures the actual C# device-interpolated
segment; all 18 fields, 117 traversal rows, and the final slope match Numba
CUDA exactly.

## Arithmetic Findings

Exact parity requires reproducing device arithmetic, not a CPU reconstruction:

- kilometre step floors multiply by the C# float32 `0.001f` constant in C#
  expression order;
- segment interpolation uses float32 fused multiply-add, as ILGPU does on the
  reference GPU;
- map-resolution and observer calculations remain float32 at the production
  kernel boundary; and
- slope-to-degree conversion uses `MathF.Atan`-equivalent float32 precision and
  preserves negative-infinity sentinels.

Before those boundaries were made explicit, one-ULP differences could change
a hierarchy-culling decision and amplify into a visible angular outlier.

## Scope

This is a correctness fixture, not a performance benchmark. It covers one
LOLA tile, one DEM, and one NVIDIA RTX 5090 Laptop GPU.

A second bounded fixture uses the real 512-square scenario DEM as primary and
a 1,024-square crop of the overlapping production medium DEM as the outer
pass. Across 368,640 merged values, one Python-generated-segment result differs
by `7.391e-6` degrees; the mean error is `2.47e-11` degrees and no value exceeds
`1e-5` degrees. The accepted maximum is now `0.005` degrees: one percent of
the approximately `0.5` degree apparent solar diameter, chosen as a
conservative proxy for the accepted one-percent sunlight error. The observed
maximum is about 676 times smaller than that limit.
Using the exact C# segments makes the merged slope and degree buffers
byte-for-byte identical, localizing the remaining discrepancy to host segment
generation amplified by a hierarchy branch. Three warm outputs are stable.
Full provenance and metrics are in
`docs/numba-horizon-phase-4-multi-real-terrain.json`.

This fixture exposed and fixed a prototype orchestration error: C# starts every
DEM pass from negative infinity and merges the completed pass buffers, whereas
the prototype had initially seeded a later traversal with the earlier DEM's
horizon. That changed hierarchy culling even though both isolated pass outputs
were correct. A CPU regression test now enforces independent passes followed by
maximum-slope merge.

The outer crop can be regenerated with:

```bash
gdal_translate -q -srcwin 768 3712 1024 1024 \
  /d/datasets/viper_v71_2024_medium/other/dem.tif \
  /tmp/lunarscout-phase4-medium-1024.tif
```

The inherited C# hierarchy bilinear-boundary defect is now corrected in both
implementations, as documented in `docs/numba-horizon-phase-4d-hierarchy.md`.
The bounded real-terrain angular comparison therefore passes. A direct
solar-limb illumination comparison and broader hierarchy-safety evidence still
remain before the overall scientific correctness gate can close.

Reproduce the C# capture and comparison with:

```bash
dotnet run \
  --project scripts/numba_horizon/CSharpPhase4RealTerrainCapture.csproj \
  -- /tmp/lunarscout-phase1-lola-512.tif \
     /tmp/lunarscout-phase4-real-terrain 255 1160

env PYTHONPATH=src \
  /e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/validate_phase4_real_terrain.py \
  /tmp/lunarscout-phase4-real-terrain.json
```

The machine-readable comparison is in
`docs/numba-horizon-phase-4-real-terrain.json`.
