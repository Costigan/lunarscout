# Numba Horizon Phase 6B Downstream Products

## Initial private vertical slice

The first Phase 6B work unit implements a private PSR path under
`lunarscout._numba_horizon`. It does not change the public Lunarscout API or
select the Numba path as a production backend.

Implemented behavior:

- complete `.bin` and `.cbin` horizon-tile reads into the fixed
  `float32[128, 128, 1440]` pixel-major contract, with structural validation;
- explicit Moon-ME vector input with timestamps, plus a lazy SpiceyPy path
  using geometric Moon-centered `MOON_ME` positions in meters;
- the exact five-viewpoint, 1,440-azimuth-bin PSR vector-reduction heuristic;
- a correctness-first CPU PSR calculation and a reusable-buffer Numba CUDA
  kernel with the C# upper-solar-limb and horizon-interpolation semantics;
- a dtype-generic, band-interleaved, tiled BigTIFF store with per-band and
  dataset-level UTC timestamps;
- stable staging, a job manifest, a durable per-patch completion journal,
  partial right/bottom windows, a dataset validity mask, and configurable
  deterministic invalid payloads; and
- a serial patch-major PSR pipeline that reads one horizon tile once, computes
  one output tile, durably writes it, and resumes without repeating completed
  patches.

The durable store closes and synchronizes the staged TIFF before it advances
the completion journal for a patch. This conservative first implementation
reopens the TIFF once per patch. Later performance work may checkpoint a
bounded batch, but the journal must remain behind durable TIFF data. If any
band of a patch fails, the patch remains unmarked and every band is recomputed
on restart.

## C# and Numba parity

`Lightmaps.ComputePSRPatchForDiagnostics` is a diagnostic-only boundary around
the existing production `ComputePSRKernel`. The capture utility executes that
ILGPU kernel on caller-supplied arrays; it does not substitute a rewritten CPU
formula for the oracle.

The immutable fixture contains a constant-threshold case, a spatially mixed
interpolation case, and the same mixed case after exact `HorizonCompressor`
quantization. The Python CPU reference matches all three C#/ILGPU outputs
byte-for-byte. The Numba CUDA kernel matches the mixed C#/ILGPU output
byte-for-byte on the RTX 5090 Laptop GPU.

The compressed fresh-process pipeline also matches the compressor-quantized
C#/ILGPU oracle at all 16,384 pixels. It differs from the uncompressed oracle
at 17 deliberately near-threshold pixels. Those changes are caused by the
existing `.cbin` angular quantization, not by Python or CUDA arithmetic; the
C# and Numba kernels agree exactly when given identical quantized inputs.

Artifacts:

- `tests/data/numba_horizon/phase6b_psr_csharp.json`
- `docs/numba-horizon-phase-6b-psr-no-dotnet.json`
- `scripts/numba_horizon/CSharpPhase6BPsrOracle.cs`
- `scripts/numba_horizon/validate_phase6b_psr_no_dotnet.py`

## Fresh-process result

The fresh Python process creates a compressed horizon tile, reads it through
the private full-tile decoder, reduces the supplied vectors, runs Numba CUDA,
writes and publishes a tiled DEFLATE GeoTIFF, reopens the result with Rasterio,
and compares every byte with the quantized C# oracle.

| Measurement | Result |
| --- | ---: |
| CUDA session setup | `0.3348 s` |
| Horizon read, reduction, CUDA calculation, and GeoTIFF publication | `0.3371 s` |
| Compressed horizon bytes | `23,642,112` |
| PSR GeoTIFF bytes | `1,816` |
| C#/Numba mismatches on identical compressed input | `0 / 16,384` |
| Validity mask | `16,384` valid, `0` invalid |
| Loaded `clr`, `pythonnet`, or `moonlib` modules | `0` |

This is a one-patch synthetic correctness and deployment probe, not a
sustained throughput benchmark. The Numba cache had already been populated.
Kernel-only time, decompression time, write time, peak host memory, and peak
GPU memory have not yet been separated for Phase 6B.

## Restart and output semantics

The staged product manifest binds a restart to the grid, datatype, band count
and ordered timestamps, compression, invalid value, algorithm/configuration,
horizon inventory identity, and algorithm version. An incompatible staged job
is rejected unless `start_fresh=True` is explicit.

Completion state is not inferred from payload values or from the final
validity mask. A patch is complete only when its journal entry is durable.
Missing or invalid horizon patches receive the configured invalid payload in
every band and mask value zero. Computed patches receive mask value 255.
Tests cover interrupted multi-band writes, journal-write failure, incompatible
resume, partial edge windows, timestamp metadata, and restart after PSR
cancellation.

## Intentionally incomplete

- The high-level SpiceyPy vector path is implemented, but realistic Moon-ME
  vectors have not yet been compared with C# `SpiceManager` output.
- The current PSR pipeline is serial. It has no bounded horizon-reader/CUDA/
  writer queue yet, and the staged TIFF is reopened once per durable patch.
- Physical TIFF block inspection when a single-band journal is missing is not
  implemented.
- A broader disk-full, process-kill, failed-overwrite, and publication failure
  matrix remains to be run.
- Long-run PSR throughput and host/GPU memory are not measured.
- The full Metonic vector generation/reduction workload is not yet benchmarked.
- Time-series lightmaps, safe-haven maps, landed mission-duration maps, and an
  additional dtype-generic reduction are not implemented.
- No public API, structured public exception mapping, packaging decision, or
  clean-wheel validation is included.

This work supports the conclusion that the downstream tiled-product pattern is
expressible in-process without Python.NET, .NET, or moonlib. It is not yet
sufficient to retire the downstream C# code because only the first PSR slice
has correctness evidence and the shared production scheduler is incomplete.
