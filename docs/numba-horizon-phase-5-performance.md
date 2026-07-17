# Numba Horizon Phase 5 Performance Evaluation

Phase 5 compares only correctness-approved C#/ILGPU and Python/Numba hierarchy
implementations. Correctness remains governed by the `0.005` degree angular
limit and the one-percent uniform-solar-disk illumination-fraction limit.

## Provisional Performance Gate

These engineering screening limits were recorded before the Phase 5 Numba
production-scale benchmark was run. They are not scientific tolerances and may
be revised only with an explicit rationale:

| Measurement | Initial acceptable Numba result |
| --- | --- |
| Warm single-patch compute latency | no more than 2.0 times matched C#/ILGPU |
| Warm sustained multi-patch throughput | at least 0.5 times matched C#/ILGPU |
| Peak process GPU memory | no more than 1.5 times matched C#/ILGPU |
| Peak host memory | no more than 1.5 times matched C#/ILGPU |
| First-use compilation plus first patch | no more than 5.0 times matched C#/ILGPU generator initialization plus first patch |

The first serial benchmark exposed redundant device pyramid uploads and the
lack of CPU/GPU overlap. The measured implementation now retains immutable
pyramids on the device and uses a one-item-ahead CPU segment producer. Segment
and output buffers are still allocated per patch and CUDA work still uses one
default stream.

## Benchmark Contract

- Hardware, four input DEMs, hashes, patch coordinates, 128 by 128 patch size,
  1,440 azimuth bins, zero-meter observer elevation, and hierarchy mode match
  the Phase 0 C# production benchmark.
- The existing `.pyr.bin` payloads are read as language-neutral little-endian
  float32 mip arrays for the warm-cache measurement. Their hashes and the DEM
  hashes are recorded independently because the production cache validates
  only array length.
- Cold Numba timing separates process/session startup and first CUDA compilation
  from warm execution. It does not relabel a pre-existing pyramid cache as a
  fresh-pyramid build.
- Single-patch and bounded contiguous four-patch measurements are separate.
- Segment generation, CUDA call, degree conversion, hashing, host memory, and
  sampled per-process GPU memory are reported separately where measurable.
- The Numba report retains a serial four-patch measurement and separately
  measures a one-producer/one-CUDA-consumer pipeline. The latter overlaps
  segment generation for patch N+1 with CUDA work for patch N.
- The corrected C# comparison uses the same four contiguous patches with its
  production four-worker, four-stream, queue-depth-six pipeline. C# elapsed
  time includes compression and writes; the Numba prototype stops after degree
  conversion and hashing. Phase 6 must make the output scope identical.

## Launch Geometry and Device Arithmetic

ILGPU reports a `(768, 1, 1)` group for the production kernel. Its generated
index reconstruction is pixel-fast: consecutive linear lanes map to adjacent
pixels at one azimuth, and linear index 768 advances to the next azimuth. Thus
each warp evaluates 32 spatially adjacent pixels for a common azimuth.

Numba uses the same pixel-fast linear mapping with 256-thread blocks. The exact
block size is not the compatibility requirement; preserving the warp mapping
is. A 768-thread Numba block was slower because the Numba kernel uses more
registers. The retained 256-thread launch preserves the same within-warp
spatial locality while allowing better occupancy. Scalar interpolation removed
local memory, and explicit float32 arithmetic reduced unintended PTX float64
operations from 92 to 37.

Pixel-fast does not make every access coalesced. The output contract is
pixel-major with azimuth as the contiguous dimension, so same-azimuth stores
from adjacent pixel lanes are separated by 1,440 floats. The mapping instead
prioritizes the much heavier input traffic: primary-DEM observer samples are
adjacent, lanes occur in eight-pixel subpatch groups that reuse the same four
segment records, and active-DEM traversal coordinates begin spatially close
and often retain cache locality. An azimuth-fast warp would coalesce the single
output store but send its lanes along 32 different ray directions and widely
separated segment blocks. A transposed device-only output is a possible future
measurement, but it would require a final transpose into the public
pixel-major contract.

The benchmark also exposed an orchestration defect: the initial Numba path
started every DEM pass at negative infinity and merged on the host. Production
C# carries the accumulated slope into later passes, allowing later DEMs to cull
terrain already below the current horizon. Numba now uses one resident slope
buffer across all four passes.

## Results

All provisional gates pass on the RTX 5090 Laptop GPU. Ratios use the matched
four-patch corrected-C# report.

| Measurement | Numba | C#/ILGPU | Numba/C# | Gate | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Warm patch CUDA call / C# bounded wall time per patch | 5.146 s | 4.302 s | 1.196 | <= 2.0 | pass |
| Four-patch pipelined throughput | 0.1635 patch/s | 0.2325 patch/s | 0.703 | >= 0.5 | pass |
| Peak process GPU memory | 5,558 MiB | 4,872 MiB | 1.141 | <= 1.5 | pass |
| Peak host memory | 8.95 GB | 14.10 GB | 0.634 | <= 1.5 | pass |
| First CUDA use / C# initialization plus cold wall time per patch | 7.163 s | 4.779 s | 1.499 | <= 5.0 | pass |

The Numba serial four-patch median is 29.077 seconds, or 0.1376 patches/s.
Overlapping segment construction reduces the median to 24.469 seconds, or
0.1635 patches/s. After the first patch, the CUDA consumer records essentially
zero wait for the segment producer. Segment construction slows from roughly
1.7-2.0 seconds serial to roughly 2.2-2.4 seconds while overlapped, indicating
host-memory contention, but remains hidden behind CUDA work.

Retaining the four immutable pyramids on the device reduced warm CUDA latency
from 5.447 to 5.146 seconds and sampled peak GPU memory from 8,312 to 5,558 MiB.
The earlier 8,312 MiB peak failed the memory gate because a new roughly 3.9 GiB
pyramid set could coexist with allocations pending release from the previous
patch.

These are bounded screening results, not asymptotic regional throughput. The
C# path uses four CUDA streams and writes compressed products; Numba uses one
stream and does not write files. Phase 6 must compare the same file-producing
scope and a longer bounded queue before making a replacement-performance claim.

## Status

The initial Phase 5 performance and resource gate passes. Remaining performance
work belongs with the Phase 6 production pipeline: reusable transient buffers,
multiple CUDA streams if they improve throughput, compression/file output,
cancellation, and a longer end-to-end run.
