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
Resume interruption within a multi-band patch and representative CPU/CUDA
time-series performance measurements remain open.

The downstream execution contract now requires `auto`, `cpu`, and `cuda`
backends for lightmaps, PSR, elevation products, safe-haven maps, and landed
mission-duration maps. `auto` prefers usable NVIDIA CUDA and otherwise falls
back to CPU without .NET. This requirement does not add a CPU production path
for horizon generation, which remains CUDA-only because its CPU implementation
is too slow.

The lightmap slice now has a deterministic oracle calling the production C#
`LightmapGenerator.BuilderSunFraction` routine. The Python CPU backend matches
all 24 oracle bytes across full, partial, zero, interpolated, and azimuth-wrap
cases. A reusable Numba CUDA session keeps the horizon and vector buffers on
device and bounds output to a configurable time batch; a 2/2/1 batch test is
byte-identical to CPU. Explicit backend selection is wired into the private
pipeline. A Numba-parallel CPU session now provides the same bounded time-batch
contract for explicit CPU use and automatic fallback. Initial warm real-patch
timing is comparable with CUDA for a 64-time batch; longer end-to-end CPU
performance evidence is still required.

The initial real-terrain comparison covered 1,048,576 byte values. CPU and
CUDA differed at six values (`0.000572%`). Every difference was exactly one
byte level: four CPU values were one lower and two were one higher. The maximum
represented sunlight-fraction difference is therefore `1/255`, approximately
`0.003922`. This is accepted as provisional CPU/CUDA agreement. Broader
benchmarks must keep reporting the difference count and maximum delta, but byte
identity is not required while differences remain scientifically small.

The lightmap-specific restart test interrupts a two-band patch after its first
band has reached the staged TIFF but before the patch is journaled. Restart
recomputes and overwrites both bands, not only the missing second band, and the
published result contains the two resumed values. This confirms the required
per-horizon-patch recovery behavior for partial multi-band writes.

## Two-year time-series lightmap benchmark

The matched longer run uses a 256 by 256 real-terrain region, four compressed
horizon patches, 2,921 exact `utc2et` Sun vectors at six-hour intervals, 2,921
timestamped `uint8` BigTIFF bands, and a time-batch size of 32. CPU and CUDA use
the same patch-major reader, staged writer, compression, masks, timestamps, and
output validation. Vector generation (`0.0972 s`) is reported separately from
the product pipeline.

| Measurement | Compiled CPU | Numba CUDA |
| --- | ---: | ---: |
| One-patch calculation | `0.3690 s` | `0.05392 s` |
| Four-patch staged BigTIFF | `3.5691 s` | `2.1780 s` |
| End-to-end throughput | `1.1207 patches/s` | `1.8365 patches/s` |
| Output bytes | `15,954,652` | `15,954,593` |

CUDA is approximately 6.84 times faster for calculation alone and 1.64 times
faster end-to-end; compressed horizon reads and 11,684 compressed tile writes
reduce the end-to-end advantage. CPU is nevertheless a useful fallback rather
than merely a correctness reference.

The two products contain 191,430,656 values. They differ at 2,294 values
(`0.00120%`), always by exactly one byte, and their validity masks are
identical. Streaming band-by-band validation gives a conservative combined
process peak RSS of `1,135,542,272` bytes. The CUDA session retains `98,566,144`
bytes of device buffers on the 24 GB reference GPU. Memory is bounded by one
horizon patch, resident vectors, and the configured 32-time byte and fraction
output batches, not by the total regional cube. Evidence is recorded in
`docs/numba-horizon-phase-6b-lightmap-benchmark.json`.

## Direct body-elevation products

Separate private Sun and Earth product functions now write body-center
elevation relative to the bilinearly interpolated local terrain horizon at the
body's azimuth. The output is a tiled, compressed `float32` BigTIFF with one
UTC-tagged band per supplied vector. Both functions use the same bounded
CPU/CUDA margin stream as the landed mission-duration products, including
`auto` CPU fallback, configurable time batches, patch-level cancellation, and
durable resume.

Synthetic file tests cover both bodies, timestamps, datatype, compressed-input
quantization, forced CUDA unavailability, and interrupted-patch restart. An
explicit real-GPU test runs the complete CPU and CUDA file pipelines on the
same input and confirms matching validity masks and margin values within
`2e-5` degrees. A representative long-workload elevation benchmark is already
included in the landed mission-duration evidence because those products
consume the identical margin stream; direct multi-band write throughput has
not yet been measured separately.

## Initial safe-haven semantics

The C# `GenerateSafeHavenDurations` path identifies center-view intervals where
Earth elevation is below a threshold, then finds each pixel's longest
contiguous low-Sun run within every interval. Its stored interval end is
inclusive, while the calculation loops with an exclusive comparison, omitting
the final below-threshold sample. It also truncates fractional hours and clamps
the result into one byte.

The Python reference uses explicit half-open intervals `[start, stop)`, includes
every below-threshold sample, and uses the first minimum-Earth sample in each
interval as its timestamp. Durations are `float32` hours by default, preserving
fractional steps and values above 255 hours. Synthetic tests cover intervals at
both ends of the time axis, repeated interior intervals, inclusion of the last
sample, multiple pixels, and a 2.5-hour step. A bounded operational patch
reducer now consumes the unquantized `float32` fraction stream online, retaining
only current and longest run counters per outage and pixel. Both compiled CPU
and CUDA provide that bounded stream; `auto` falls back to CPU. A synthetic
end-to-end product writes two correctly timestamped `float32` duration bands,
and CPU/CUDA produce the same duration for the controlled reduction. Real
safe-haven performance remains open. The operational pipeline now emits
immediately flushed patch progress, checks cancellation before and after
horizon reads, within the streamed time calculation, before writes, and before
publication, and resumes an interrupted patch as one durable work unit. Tests
also force CUDA unavailability and verify `auto` CPU fallback. Missing horizon
patches already use the configured invalid payload and validity mask through
the shared product store.

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
cancellation. PSR now uses the same explicit `auto`, `cpu`, and `cuda` backend
policy as the other downstream pipelines: `auto` falls back to the CPU
reference when CUDA is unavailable, while explicit `cuda` propagates the
capability failure instead of silently changing backends.

## Initial landed mission-duration semantics

The initial private landed-duration implementation defines four distinct
product functions rather than one public mode argument: sunlight fraction,
Sun-center elevation, sunlight fraction plus Earth-center elevation, and
Sun-center plus Earth-center elevation. Here elevation is body-center margin
above the bilinearly interpolated local terrain horizon at the body's azimuth.
The shared CPU and CUDA lightmap sessions now emit this margin using the same
pixel frame and horizon interpolation as sunlight calculation. Comparisons are
inclusive, so values exactly equal to any Sun or Earth threshold qualify.

Every product accepts an overall half-open evaluation interval and explicit
half-open candidate-start intervals. The condition at sample `t[i]` owns
`[t[i], t[i + 1])`, using the evaluation stop as the final boundary when the
stop is not sampled. A candidate may begin after an already-qualifying period
enters its start interval, continues beyond that smaller interval while the
condition remains true, and is credited only through the evaluation stop when
still active there. Sample-to-sample durations may be irregular. The output is
one `float32` band per candidate-start interval in hours or days, with start,
stop, unit, and UTC timestamp metadata. Month, start-anchored week, and fixed
duration helpers construct common interval lists.

Synthetic tests cover right-censoring, a qualifying period already active at
the candidate boundary, irregular sample spacing, inclusive thresholds, all
four product functions, missing horizons with configurable invalid payload,
per-band metadata, cancellation, and durable patch-level resume. Controlled
CPU geometry produces the expected local-horizon margin; the real-GPU test
matches CUDA and CPU margin output. A controlled end-to-end combined
Sun-center/Earth-center product is also identical between CPU and CUDA,
including the published `float32` duration tile. The following benchmark adds
representative regional performance and memory evidence.

## Two-year landed mission-duration benchmark

The representative landed-duration run covers a 256 by 256 real-terrain
region, four compressed horizon patches, and the exact half-open interval
`[2027-01-01, 2029-01-01)`. It uses 2,925 Sun and Earth samples at six-hour
spacing and 24 calendar-month candidate-start bands. Sunlight-fraction
thresholds are `>= 0.5`; Sun-center and Earth-center local-horizon-margin
thresholds are `>= 0` degrees. Every output is a tiled, compressed, 24-band
`float32` BigTIFF in days. CPU and CUDA kernels are warmed before measurement;
SPICE vector generation and the separately reported compressed-horizon read are
not included in the product timings.

| Product | CPU calculation, one patch | CUDA calculation, one patch | CPU end-to-end, four patches | CUDA end-to-end, four patches | End-to-end speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sun elevation | `0.3547 s` | `0.1369 s` | `2.1902 s` | `1.3138 s` | `1.67x` |
| Sun + Earth elevation | `0.6362 s` | `0.1733 s` | `3.3358 s` | `1.4544 s` | `2.29x` |
| Sunlight fraction | `0.4127 s` | `0.1390 s` | `2.5748 s` | `1.3350 s` | `1.93x` |
| Sunlight + Earth elevation | `0.6935 s` | `0.1747 s` | `3.5116 s` | `1.4568 s` | `2.41x` |

The CPU fallback sustains `1.14` to `1.83` patches per second end-to-end;
CUDA sustains `2.75` to `3.04` patches per second. Calculation-only CUDA
speedups range from `2.59x` to `3.97x`. Margin-only execution now bypasses the
unused 16-slice solar-disk calculation. The online reducer also skips interval
array work when an interval can neither accept a new start nor has an active
candidate.

CPU and CUDA masks, timestamps, interval metadata, band counts, and dtypes all
match. Across 1,572,864 values per product, only four to nine values differ
(`0.000254%` to `0.000572%`). Most differences are one six-hour sample
(`0.25 day`). The largest is one day at one Sun-margin pixel. This amplification
is expected for a discontinuous thresholded-duration reduction: a very small
CPU/CUDA signal difference at exactly the threshold can split or join a run.
The artifact records every mismatching band, pixel, CPU value, CUDA value, and
delta; it does not characterize the underlying margin difference as one day.

Peak process RSS across all four CPU and CUDA products was `815,747,072` bytes.
Two simultaneously retained CUDA sessions, sufficient for combined Sun/Earth
products, allocated `199,229,440` bytes on the 24 GB reference GPU. Memory is
bounded by one decoded horizon, two fixed-shape signal sessions, the configured
32-time batches, and 24 interval states per patch; it does not scale with the
regional time cube. Exact vectors took `0.0881 s` for the Sun and approximately
`0.03 s` for Earth. One compressed horizon read took approximately `0.52 s`.
No `clr`, `pythonnet`, or `moonlib` module was loaded.

Machine-readable configuration, input identities, environment versions,
output hashes, timings, memory, comparisons, and mismatch samples are in
`docs/numba-horizon-phase-6b-mission-duration-benchmark.json`.

## Intentionally incomplete

- The current PSR pipeline is serial. A full 1,599-patch Mons Mouton run
  completed in `306.45 s` (`5.218 patches/s`) with only about five percent GPU
  utilization. The retained `0.0148 s` warm kernel measurement versus `0.1917
  s` full-run wall time per patch demonstrates material headroom outside the
  kernel. There is no bounded horizon-reader/decompress, CUDA, and writer
  pipeline yet, and the staged TIFF is reopened once per durable patch.
- The follow-up must time every pipeline stage, then measure bounded
  decompression/GPU/write overlap, batched durable checkpoints, pinned and
  asynchronous transfers, multi-patch CUDA submissions, and limited CPU
  decompression parallelism without weakening restart semantics or allowing
  memory to scale with the region.
- Physical TIFF block inspection when a single-band journal is missing is not
  implemented.
- A broader disk-full, process-kill, failed-overwrite, and publication failure
  matrix remains to be run.
- Decompression, transfer, calculation, and compression/write costs have not
  yet been separated within the 16-patch end-to-end result.
- Safe-haven performance remains synthetic; representative regional CPU/CUDA
  throughput and memory evidence are not yet recorded for that product.
- No public API, structured public exception mapping, packaging decision, or
  clean-wheel validation is included.

This work supports the conclusion that the downstream tiled-product pattern is
expressible in-process without Python.NET, .NET, or moonlib. It is not yet
sufficient to retire the downstream C# code because public API/error contracts,
representative safe-haven performance, full operational failure evidence, and
clean-wheel deployment remain incomplete.
