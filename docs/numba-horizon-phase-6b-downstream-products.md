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
- `tests/data/numba_horizon/phase6b_spice_csharp.json`
- `docs/numba-horizon-phase-6b-psr-no-dotnet.json`
- `docs/numba-horizon-phase-6b-psr-vectors.json`
- `scripts/numba_horizon/CSharpPhase6BPsrOracle.cs`
- `scripts/numba_horizon/CSharpPhase6BSpiceOracle.cs`
- `scripts/numba_horizon/validate_phase6b_psr_no_dotnet.py`
- `scripts/numba_horizon/benchmark_phase6b_psr_vectors.py`

## SPICE time conversion and full-cycle PSR evidence

The generated-vector boundary has two explicit UTC-to-ephemeris-time modes:

- `utc2et`, the default, converts every UTC timestamp independently. This
  remains the mission-time path, including for future mission periods after
  all published leap seconds.
- `linear_from_anchor` converts the C# epoch, 2023-12-01 UTC, once and adds
  ordinary elapsed `TimeSpan` seconds. It exactly reproduces the current C#
  `SpiceManager` convention. It may be selected only where equivalence to
  per-timestamp `utc2et` has been demonstrated for the intended calculation;
  it does not account for leap seconds crossed relative to the anchor.

The C# diagnostic captures geometric Sun and Earth positions in `MOON_ME` at
selected dates from 1970 through 2044. Python's anchored mode agrees exactly
with those C# positions. Compared with the C# anchored values, exact `utc2et`
has maximum angular differences of `0.00393945` degrees for the Sun and
`0.000472598` degrees for Earth over those selected dates. The largest
differences precede the current leap-second epoch.

A real full-cycle benchmark generated 108,113 Sun positions at six-hour steps
from 1970-01-01 through 2044-01-01. Exact conversion took `1.1517 s`; anchored
conversion took `0.4646 s`, a one-time saving of `0.6871 s`. The modes differed
by at most `0.00397038` degrees, retained 2,251 and 2,253 indices respectively,
and shared 2,234 retained indices. Despite 17 exact-only and 19 anchored-only
indices, their PSR outputs were identical on the real terrain used below.
This establishes product-level equivalence for this retained PSR case; it is
not a general authorization to substitute anchored conversion in other
pointing or lighting calculations.

## Real 16-patch product result

Both full vector sets were independently reduced and run through the staged
single-band GeoTIFF pipeline over 16 real compressed horizon patches. Each
product covers the complete DEM grid; the 262,144 pixels backed by those
horizon patches are valid and all other pixels have deterministic zero payload
and invalid mask.

| Measurement | Exact `utc2et` | Anchored-linear |
| --- | ---: | ---: |
| End-to-end product time | `13.3627 s` | `13.4669 s` |
| Throughput | `1.1974 patches/s` | `1.1881 patches/s` |
| Warm one-patch CUDA median | `0.01480 s` | `0.01490 s` |
| Final GeoTIFF bytes | `139,790` | `139,790` |

The two products have zero pixel mismatches, zero mask mismatches, and the same
file SHA-256. Peak host RSS was `926,576,640` bytes. The retained CUDA session
uses `96,468,992` bytes; including the largest reduced-vector device buffer it
uses `98,566,144` bytes. No `clr`, `pythonnet`, or `moonlib` module was loaded.
The run used a populated Numba cache and therefore does not measure first-use
kernel compilation.

## Initial time-series lightmap slice

The private lightmap reference path defines byte sunlight using the active C#
`LightmapPipeline.GenerateShadows`/`BuilderSunFraction` convention: 16 vertical
solar-disk slices, a 0.27-degree solar half-angle, interpolated 0.25-degree
horizon samples, and truncating `uint8(255 * visible_fraction)` encoding. It
precomputes only the bounded local frames for one horizon patch, then yields one
128 by 128 tile per supplied Sun vector.

`run_lightmap_product` makes the horizon patch the primary work unit. It loads
one `.bin`/`.cbin` tile and passes the lazy time-tile iterator directly to the
already-open staged BigTIFF. Each yielded tile is written to the band carrying
that time's UTC metadata before the next tile is requested. It therefore does
not retain a patch time cube or regional time cube. As with PSR, a missing or
invalid horizon writes the configurable invalid value to every band and marks
the output mask invalid.

The initial independent tests cover full, half, and zero visible solar disks
(`255`, `127`, and `0`), two timestamped band-interleaved tiles, a partial edge,
and a missing horizon patch. This is a correctness-first CPU/storage slice.
The bounded Numba CUDA time-batch implementation, actual C# numerical oracle,
resume interruption within a multi-band patch, and representative time-series
performance measurements remain open.

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

- The current PSR pipeline is serial. It has no bounded horizon-reader/CUDA/
  writer queue yet, and the staged TIFF is reopened once per durable patch.
- Physical TIFF block inspection when a single-band journal is missing is not
  implemented.
- A broader disk-full, process-kill, failed-overwrite, and publication failure
  matrix remains to be run.
- Decompression, transfer, calculation, and compression/write costs have not
  yet been separated within the 16-patch end-to-end result.
- Time-series lightmaps have only a CPU-reference/storage slice; their Numba
  CUDA kernel and performance evidence are not implemented. Safe-haven maps,
  landed mission-duration maps, and an additional dtype-generic reduction are
  not implemented.
- No public API, structured public exception mapping, packaging decision, or
  clean-wheel validation is included.

This work supports the conclusion that the downstream tiled-product pattern is
expressible in-process without Python.NET, .NET, or moonlib. It is not yet
sufficient to retire the downstream C# code because only the first PSR slice
has correctness evidence and the shared production scheduler is incomplete.
