# Numba Horizon Phase 3 Host Geometry

Phase 3 ports and validates the host-side geometry that prepares dense
`RaySegment` tensors for a future horizon kernel. It does **not** implement a
Numba CUDA horizon kernel, traverse a terrain pyramid to calculate a horizon,
or reproduce final C# horizon products.

## Implemented Scope

`src/lunarscout/_numba_horizon/geometry.py` contains the readable NumPy oracle
implementation for:

- affine pixel/CRS conversion and spherical stereographic projection;
- latitude/longitude, Moon-centered vectors, ENU rotation, and azimuth rays;
- C#-matching bilinear DEM sampling, boundary intersection, sample placement,
  minimum span, and minimum sample count;
- quartic pixel-path fitting, cubic planar-to-chord fitting, and all current
  fallback rules;
- per-DEM resolution and ray-limit contexts;
- subpatch halo construction, edge clamping, and deterministic center caching;
- complete `[azimuth, center, DEM, field]` segment tensor assembly.

Grid convergence uses a deliberately small language-neutral three-scalar input
contract: center angle, X gradient, and Y gradient. This preserves the existing
production behavior: the values are prepared and passed onward, but the current
C# subpatch CUDA kernel does not apply them. Phase 3 does not silently change
that behavior.

`geometry_numba.py` contains optional Numba 0.66.0 CPU kernels for chord
sampling and segment fitting. The parallel loop is over independent segment
jobs. `build_subpatch_segments_numba` imports that module lazily, deduplicates
clamped halo centers before compilation, preserves multi-DEM distance
continuity, and restores the production tensor order. Normal `import
lunarscout` and importing the NumPy geometry module do not import Numba.

Numba remains a prototype-only dependency in
`scripts/numba_horizon/requirements-prototype.txt`; it has not been added to the
published package dependencies.

## C# Correctness Evidence

The tests load the immutable Phase 1 artifacts captured from the production C#
helpers. They do not regenerate expected values from Python.

- All 29 captured C# fit passes reproduce their double-precision samples to
  less than `1e-10` absolute difference.
- Ordinary NumPy segment coefficients reproduce the stored C# float32 values
  to less than `5e-9` in the targeted fit tests.
- The complete 512-segment synthetic subpatch fixture, including two DEMs and
  duplicate clamped halo centers, matches C# to `6e-8` for NumPy.
- Compiled serial and parallel results are bitwise equal. Their solver order
  can differ slightly from C# and NumPy; stored coefficient differences are
  below `1e-5`, while dense fitted-path error remains below `1e-5` pixel.

The real-terrain gate uses the Phase 1 NASA LOLA 512×512 subset. The C# capture
calls `CalculateSubpatchRaySegmentsForDiagnostics` for a 16×16 patch layout,
64 azimuth bins, one DEM, and a 5 km maximum range. It records 1,024 production
C# segments. The comparison evaluates every Python and C# segment at 257
distances across the shared fitted range:

- ordinary NumPy maximum path error: `3.275e-10` pixel;
- Numba maximum path error: `3.045e-10` pixel;
- accepted maximum: `1e-4` pixel;
- C# selected accelerator: NVIDIA GeForce RTX 5090 Laptop GPU, CUDA.

CUDA selection in this capture is evidence that ILGPU used CUDA while building
the diagnostic pyramid. The compared ray segments themselves are host-prepared
inputs. This is not evidence that Python has generated a horizon.

The reproducible report is
`docs/numba-horizon-phase-3-real-terrain.json`. Regenerate its C# input and run
the validator with:

```bash
dotnet run --project scripts/numba_horizon/CSharpPhase3RealTerrainCapture.csproj -- \
  /tmp/lunarscout-phase1-lola-512.tif \
  /tmp/lunarscout-phase3-csharp-real.json

/e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/validate_phase3_real_terrain.py \
  /tmp/lunarscout-phase3-csharp-real.json
```

The raster payload hash is checked against the Phase 1 fixture manifest.

## Performance and Memory

`docs/numba-horizon-phase-3-host-benchmark.json` separates coefficient-only
fitting from complete sampling plus fitting. On the selected 24-logical-CPU
host, with eight Numba threads and warm compiled functions:

| Work | Serial Python | Numba serial | Numba parallel |
|---|---:|---:|---:|
| Fit precomputed samples | 8,657 jobs/s | 858,023 jobs/s | 6,517,045 jobs/s |
| Sample one synthetic DEM and fit | 4,991 jobs/s | 527,434 jobs/s | 3,636,532 jobs/s |

These microbenchmarks repeat bounded oracle shapes and must not be presented as
end-to-end horizon throughput.

The real-terrain report also measures complete subpatch assembly. A bounded
production-shaped workload with a 128×128 patch layout, 1,440 azimuth bins, one
512×512 real DEM, and a 5 km range creates 466,560 segments. Its eight-thread
warm median is 1.359 seconds. The output tensor is 33,592,320 bytes and the
process maximum RSS after the run is 571,228 KiB. This remains smaller and
shorter-range than the four-DEM production stack.

A full four-DEM production patch contains 1,866,240 segment records and its
float32 tensor payload is 134,369,280 bytes. Six such payloads are 806,215,680
bytes, excluding samples, Python objects, DEM arrays, CUDA buffers, and any
transient duplicate tensor. An actual retention test held six production-shaped
four-DEM tensors containing exactly 806,215,680 payload bytes; process maximum
RSS was 1,448,724 KiB. It repeats the same bounded LOLA DEM and later DEM passes
have no remaining ray distance, so this validates cache/tensor memory rather
than four-DEM compute throughput. The retained size is acceptable on the 98 GB
evaluation host, but queue-wide memory must be measured again with the actual
Phase 4/5 pipeline.

The current optimization plan is to generate one DEM pass at a time, overlap
preparation of the next DEM or patch with the active GPU pass, reuse observer
and direction arrays, cap the Numba pool below all 24 logical CPUs, and avoid
retaining duplicate cache and transfer tensors. Phase 5 must measure the actual
four-DEM ranges and concurrency before accepting production throughput or a
package-wide thread default.

## Phase 3 Conclusion

Host geometry and segment generation satisfy the standalone Phase 3 correctness
and bounded-performance gates. The combined checklist remains open until the
Phase 4 pipeline exists and can prove that the eight-thread CPU pool does not
oversubscribe or delay concurrent CUDA work. The evidence supports beginning
diagnostic Phase 4 CUDA mechanics, but does not support removing C#, claiming
final horizon parity, or adding Numba to normal Lunarscout runtime dependencies.
