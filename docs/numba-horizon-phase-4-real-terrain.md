# Numba Horizon Phase 4 Real-Terrain CUDA Parity

This bounded validation compares the production-shaped Numba CUDA kernel with
the C# ILGPU CUDA kernel on a retained 512 by 512 LOLA terrain fixture. It uses
one hierarchy-enabled 16 by 16 patch, all 1,440 azimuth bins, one DEM pass, and
zero-metre observer elevation.

## Result

Across all 368,640 values, the maximum angular difference is
`5.9605e-8` degrees and no value exceeds `1e-6` degrees. The hashes are:

- C# slopes: `63465f37ff5b6c1e80c4e9f72bf078dc664a6b5a8cebbe836499afb73c74e5b9`;
- Numba slopes: `8195c94ad6007eac9528a29476d8c245431ec8044a41099d1a8d86ac29a5faec`;
- C# degrees: `aa7c0d4671b77faf539c7f6b3fb848a0932f9ab2ef062dd60b1b8e020f17e431`;
- Numba degrees: `0cd75d802982afa7cf8ef52f1b20aa0e13590d595ca0a049348cd46496eff8b9`.

There are no sentinel mismatches or non-finite values. Mean error is
`3.03e-13` degrees; median, 95th, 99th, and 99.9th percentile errors are zero.

Three subsequent warm calls reproduce the same slope hash. Their diagnostic
elapsed times including transfers were 15.47, 14.94, and 14.81 ms; these are
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
pass. Across 368,640 merged values, the maximum angular difference is
`4.0412e-5` degrees and the mean absolute error is `8.16e-10` degrees. No value
exceeds the accepted `0.005` degree maximum, which is one percent of the
approximately `0.5` degree apparent solar diameter and a conservative proxy
for the accepted one-percent sunlight error. The observed maximum is about
124 times smaller than that limit.

The isolated first and second pass maxima are `3.5763e-7` and `2.3842e-7`
degrees. Exact C# segments retain the same merged `4.0412e-5` degree maximum,
so the larger merged difference comes from a C#/Numba arithmetic difference in
the cumulative-horizon culling path rather than Python segment generation.
Three warm outputs are stable.
Full provenance and metrics are in
`docs/numba-horizon-phase-4-multi-real-terrain.json`.

This fixture exposed and fixed a prototype orchestration error: production C#
seeds each later DEM pass with the accumulated horizon from earlier passes so
terrain already below that horizon can be culled. The prototype initially
started every pass from negative infinity and merged completed buffers on the
host. A CPU regression test now enforces cumulative slope carry across passes.

The outer crop can be regenerated with:

```bash
gdal_translate -q -srcwin 768 3712 1024 1024 \
  /d/datasets/viper_v71_2024_medium/other/dem.tif \
  /tmp/lunarscout-phase4-medium-1024.tif
```

Under a uniform `0.5` degree solar-disk and locally straight horizon model, the
maximum possible absolute sunlight-fraction difference is `1.0291e-4`, or
`0.0103%` of the full disk, well below the accepted one percent. The inherited
C# hierarchy bilinear-boundary defect is corrected in both implementations and
a ten-case directional/coarse-mip matrix records the difference from dense
bilinear sampling as a non-gating approximation diagnostic. These are bounded
results, not a proof over every fitted terrain ray.

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
