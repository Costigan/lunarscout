# Numba Horizon Phase 6 Production-Pipeline Prototype

## Implemented Scope

The first Phase 6 unit remains private under `lunarscout._numba_horizon`. It
adds production-shaped patch enumeration, a bounded producer/consumer pipeline,
the existing user-visible progress fields, cancellation checks between bounded
work units, structurally validated skip/resume, and staged horizon tile output.
It does not alter the public native horizon API or select Numba as a production
backend.

Aligned DEMs enumerate exactly like C#: row-major 128 by 128 patches with X
advancing fastest. The current C# `GeneratePatchList` rejects dimensions that
are not multiples of 128. Python intentionally covers the partial right and
bottom edges. Because existing readers require 16,384 horizons per tile, only
pixels outside the valid DEM rectangle are padded, using the compressed
format's minimum representable elevation of `-50` degrees.

Skip detection retains the C# lookup order: partitioned `.cbin`, partitioned
`.bin`, legacy flat `.cbin`, then legacy flat `.bin`. Unlike C#, a name alone
does not establish completion. Raw files must have one of the two reader-
accepted sizes. Compressed files must contain exactly 16,384 valid length-
prefixed blocks and no trailing bytes. Each output is written to a uniquely
named sibling staging file, flushed and synchronized, then atomically replaced.
A failed overwrite preserves the prior completed file and removes staging data.

The bounded pipeline has one segment producer, a configurable prepared-item
queue, a configurable worker factory, and an optional bounded writer queue.
The evaluated path has one CUDA consumer on Numba's default stream and one
writer. It writes each completed patch immediately and never retains a regional
horizon cube. `CudaSession` reuses resident immutable pyramids and fixed-shape
segment and output device buffers across patches.

## File Compatibility

Python reproduces the C# `.cbin` quantization, signed 7/15-bit delta encoding,
little-endian block lengths, pixel-major ordering, and `.bin` little-endian
float32 layout. Moonlib successfully read a Python-generated compressed
synthetic partial tile and wrote an uncompressed round trip that the existing
Python Scenario reader consumed.

For the first real pipeline tile, moonlib decoded all 23,592,960 compressed
values. Comparing that decoded output with the matched Python `.bin` gives:

| Measurement | Result |
| --- | ---: |
| Maximum absolute compression error | `0.0007638931` degrees |
| Mean absolute compression error | `0.0003806583` degrees |
| Values above `0.001` degrees | `0` |
| Compressed bytes | `23,672,772` |
| Uncompressed bytes | `94,371,840` |

This is expected format quantization and is below the accepted `0.005` degree
kernel-comparison limit. Machine-readable results are in
`docs/numba-horizon-phase-6-file-compatibility.json`.

## Matched End-to-End Measurements

All measurements use four contiguous 128 by 128 patches, 1,440 azimuths, four
real DEM passes, hierarchy enabled, resident pyramids, and one warm-up patch.
They include degree conversion, staged compression/write, and output hashing.
The compressed serial and pipelined file hashes match exactly for all patches.

| Measurement | Serial compressed | Pipelined compressed | Pipelined raw |
| --- | ---: | ---: | ---: |
| Elapsed | `28.8768 s` | `23.9041 s` | `23.8760 s` |
| Throughput | `0.13852 patch/s` | `0.16734 patch/s` | `0.16753 patch/s` |
| Wall time per patch | `7.2192 s` | `5.9760 s` | `5.9690 s` |
| Write time, four patches | `0.3885 s` | `0.4187 s` | `0.3505 s` |
| Output bytes | `94,627,520` | `94,627,520` | `377,487,360` |
| Peak host RSS | `8.09 GB` | `9.00 GB` | `8.99 GB` |
| Peak GPU memory | `5,338 MiB` | `5,338 MiB` | `5,338 MiB` |

The bounded pipeline improves throughput by 20.8 percent over serial. Against
the matched corrected-C# compressed benchmark (`0.23247 patch/s`), it reaches
72.0 percent throughput. Peak GPU memory is 1.096 times C# and peak host memory
is 0.638 times C#, so the recorded Phase 5 resource and throughput gates remain
satisfied with identical file-producing scope. Compressed output is 25.1
percent of raw size and adds only `0.028 s` to this bounded run.

Reports are stored in:

- `docs/numba-horizon-phase-6-serial.json`
- `docs/numba-horizon-phase-6-pipeline.json`
- `docs/numba-horizon-phase-6-pipeline-uncompressed.json`

## Sustained 16-Patch Measurement

The longer comparison uses all 16 contiguous patches covering the 512 by 512
primary DEM. Both implementations use the same four DEMs, hierarchy, 1,440
azimuths, compression, and file-writing scope. C# uses its production four GPU
workers/streams and queue depth six. Python uses one segment producer, one
default-stream CUDA consumer, a one-item prepared queue, and a one-item writer
queue.

| Measurement | Python, initial | Python, writer | Python, writer + buffer reuse | C# warm |
| --- | ---: | ---: | ---: | ---: |
| Throughput (patch/s) | `0.17266` | `0.17697` | `0.17931` | `0.28093` |
| Wall time (s/patch) | `5.7919` | `5.6508` | `5.5769` | `3.5596` |
| Peak host memory | `9.00 GB` | `9.02 GB` | `9.02 GB` | `18.09 GB` |
| Peak GPU memory | `5,558 MiB` | `5,558 MiB` | `4,458 MiB` | `5,504 MiB` |

The optimized Python path reaches 63.8 percent of sustained C# throughput,
uses 49.9 percent of C# host memory, and uses 81.0 percent of C# GPU memory.
The earlier 72.0 percent result remains valid only for the bounded four-patch
comparison; it is not the sustained ratio. C# gains substantially from its
multi-worker scheduling as the batch grows, while this Python path remains
single-stream.

The prepared queue reaches its one-item bound. Apart from the initial 1.72
second pipeline fill, every consumer dequeue wait is below one millisecond,
while later producer enqueue waits are typically 2.6 to 3.7 seconds. Segment
preparation is therefore fully hidden in steady state. A segment cache or
CPU-parallel preparation cannot improve this single-stream measurement.

The writer queue also reaches its one-item bound, but CUDA-to-writer enqueue
waits remain about 8 to 12 microseconds. Moving degree conversion, compression,
and staged writes to the writer improves throughput by 2.5 percent. Reusing
transient device buffers adds another 1.3 percent and reduces peak device memory
by 1,100 MiB. Across 16 patches, measured CUDA boundary work totals 0.229
seconds for segment uploads, 0.295 seconds for output resets, and 0.248 seconds
for copies back; kernel wall time totals 86.378 seconds. The dominant remaining
gap is kernel execution and concurrent GPU scheduling, not transfers or output
work.

All 16 Python output hashes are identical between the initial, writer, and
buffer-reuse runs. Reports are stored in:

- `docs/numba-horizon-phase-6-sustained-python-baseline.json`
- `docs/numba-horizon-phase-6-sustained-python-writer.json`
- `docs/numba-horizon-phase-6-sustained-python-reuse.json`
- `docs/numba-horizon-phase-6-sustained-csharp.json`

### CUDA stream matrix

The same sustained run was repeated with two and four CUDA workers. Each worker
has one non-default stream and one bounded segment/output device-buffer slot;
the immutable pyramids remain shared.

| CUDA workers/streams | Throughput (patch/s) | Peak host RSS | Peak GPU memory |
| ---: | ---: | ---: | ---: |
| 1 | `0.17931` | `9.02 GB` | `4,458 MiB` |
| 2 | `0.18045` | `9.45 GB` | `4,680 MiB` |
| 4 | `0.17952` | `10.16 GB` | `5,120 MiB` |

All output hashes match. The apparent 0.6 percent two-stream improvement is
within run-to-run variation and disappears with four streams. Concurrent
kernels overlap, but individual calls lengthen to about 11 seconds with two
workers and 22 seconds with four, compared with about 5.4 seconds for one.
More streams therefore add memory without meaningful throughput. One default
stream remains the selected production-prototype setting.

With four workers, preparation causes an additional 3.72 second wait during
initial ramp-up, after which dequeue waits are below one millisecond. Parallel
CPU preparation could shorten that fill interval, as expected, but cannot
improve the measured steady-state GPU throughput because multiple streams do
not outperform one stream. Reports are stored in:

- `docs/numba-horizon-phase-6-sustained-python-streams2.json`
- `docs/numba-horizon-phase-6-sustained-python-streams4.json`

## Fresh-Process Startup and Compiled Cache

Startup was split into data/cache loading, CUDA device/session initialization,
first CPU segment generation, and first CUDA execution. A dedicated empty
`NUMBA_CACHE_DIR` measured population; a second fresh Python process used the
same directory. No persistent worker process is involved.

| Stage | Cache population | Fresh process with cache | Saved |
| --- | ---: | ---: | ---: |
| DEM and pyramid-cache load | `8.211 s` | `8.216 s` | none |
| CUDA session/device initialization | `3.487 s` | `3.441 s` | none |
| First CPU segment generation | `7.002 s` | `1.821 s` | `5.181 s` |
| First CUDA compile and execution | `6.849 s` | `5.488 s` | `1.361 s` |
| Combined first CPU/CUDA calls | `13.851 s` | `7.309 s` | `6.542 s` |

The populated cache contains 21 files totaling 2,325,563 bytes. The first CUDA
figure includes a real four-pass kernel execution; its warm call is about 5.1
seconds. CUDA context/device initialization is separate and cannot be removed
by compiled-kernel caching.

The segment functions and production CUDA kernel now request Numba disk
caching. If Numba reports that no writable cache locator is available, the
decorators fall back to uncached JIT instead of breaking import or execution.
Ordinary tests verify that read-only-source fallback. Numba's cache keys include
the compiled signature, target/code-generation identity, function bytecode,
and closure content; clean-wheel cache placement and cross-toolchain/GPU
invalidation remain packaging-phase checks.

Both fresh processes produced the same output hash, and their benchmark reports
record no loaded `clr`, `pythonnet`, or `moonlib` modules. Artifacts are:

- `docs/numba-horizon-phase-6-startup-uncached.json`
- `docs/numba-horizon-phase-6-startup-cache-populate.json`
- `docs/numba-horizon-phase-6-startup-cache-reuse.json`

## Intentionally Deferred

- Neighboring patch segments are intentionally recomputed, matching the
  hard-coded-off C# shared cache. Preparation is hidden by the current CUDA
  consumer, so cache memory and eviction complexity have no measured
  end-to-end benefit. Revisit caching only if later measurements show a CUDA
  consumer waiting for prepared work. Before adding a cache, also measure
  additional CPU-parallel preparation and its host-memory and memory-bandwidth
  costs.
- Multiple true CUDA streams are available in the private prototype and have
  been measured, but are intentionally not the default because they add memory
  without material throughput improvement on this workload and GPU.
- Cancellation is checked before preparation, compute, and write, but does not
  interrupt a CUDA kernel already executing.
- Simulated cancellation, worker/CUDA failure, and staged write failure are
  tested. Actual disk-full behavior and a broader interruption/restart matrix
  remain open.
- Sixteen patches establish sustained behavior for this 512 by 512 primary DEM;
  larger regional runs and long-run resource stability remain open.
- No public API, structured public failure mapping, backend selection,
  packaging decision, or clean-wheel validation is included.
- A long-lived compiled worker is not an acceptable startup strategy because
  horizon generation must work efficiently in notebooks, throwaway scripts,
  and agent-launched processes. Cross-process CPU and CUDA disk caching is now
  implemented; clean-wheel cache placement and broader toolchain/GPU
  invalidation testing remain deferred to packaging evaluation.
- Downstream illumination, visibility, temporal accumulation, and PSR
  classification comparisons remain part of the scientific adoption gate. In
  addition, Phase 6B now requires a shared no-.NET tiled-product pipeline for
  time-series lightmaps, optimized PSR, safe-haven, landed mission-duration,
  and dtype-generic reductions before the downstream C# code can be retired.

## Reproduction

Run ordinary tests without a GPU:

```bash
PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 \
  /e/projects/lunarscout/.venv/bin/python -m pytest \
  tests/numba_horizon tests/test_scenario.py tests/test_native_horizon.py \
  -q -p no:cacheprovider
```

Run the GPU-visible end-to-end measurements from an escalated host shell:

```bash
PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 \
  /e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/benchmark_phase6_pipeline.py \
  --mode serial --output-json docs/numba-horizon-phase-6-serial.json \
  --output-directory /tmp/lunarscout-phase6-serial --patch-count 4

PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 \
  /e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/benchmark_phase6_pipeline.py \
  --mode pipeline --output-json docs/numba-horizon-phase-6-pipeline.json \
  --output-directory /tmp/lunarscout-phase6-pipeline --patch-count 4
```

Build and run the cross-language reader/writer probe with:

```bash
dotnet build scripts/numba_horizon/CSharpPhase6FileCompatibility.csproj --no-restore
dotnet run --project scripts/numba_horizon/CSharpPhase6FileCompatibility.csproj \
  --no-build -- <python-input.cbin> <csharp-output.bin>
```

## Verification

- Ordinary Numba, Scenario, and native-wrapper suite: `100 passed, 7 skipped`.
- Explicitly CUDA-gated Numba suite: `67 passed, 1 skipped`.
- Focused CUDA subpatch suite after the final locking adjustment: `8 passed`.
- All 33 Numba-horizon JSON artifacts parse successfully.
- All 30 prototype and benchmark Python sources compile with `compile(...)`.
- Ordinary `import lunarscout` loads none of Numba, Python.NET, CLR, or moonlib.
- `git diff --check` passes.
