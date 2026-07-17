# Numba Horizon Phase 4E Full Subpatch and Multi-DEM Operation

Phase 4E connects the hierarchy traversal to production-shaped CUDA indexing,
four-center interpolation, patch edges, and sequential DEM accumulation. This
is still a private kernel prototype, not the production scheduler or file
writer.

## C# Buffer Parity

The one-pixel production fixture exercises all 1,440 azimuth bins and two DEM
passes. Against the immutable C# hierarchy-enabled buffers:

- primary-pass maximum slope error: `0`;
- outer-pass maximum slope error: `0`;
- accumulated final-slope maximum error: `0`;
- final-degree maximum error: `0` degrees;
- negative-infinity sentinel mismatches: `0`.

The GPU kernel performs the C# order of operations: select four halo-inclusive
subpatch centers, clamp requested centers against the primary DEM, shift each
segment into the current pixel and active-DEM resolution, bilinearly
interpolate all 18 fields, and traverse one active DEM. The first DEM starts
from negative infinity; each later pass reads and updates the same accumulated
slope buffer so existing horizons participate in hierarchy culling. Degrees
are calculated once after every DEM pass.

## Patch and DEM Coverage

Additional CPU/CUDA comparisons cover:

- a 16 by 16 partial patch, including corners, the 8-pixel subpatch seam, and
  center-adjacent pixels: 60 selected comparisons, maximum slope error `0`;
- a full 128 by 128 patch with all 324 halo-inclusive centers: 30 selected
  comparisons, maximum slope error `0`;
- a 30 m primary DEM and 60 m outer DEM with different dimensions and ray
  limits: 32 selected comparisons, maximum slope error `1.49e-8`.

The complete full-patch buffer is executed on CUDA, but selected rays rather
than all 262,144 pixel/azimuth outputs are recomputed through the slow CPU
oracle. Performance and resource claims are deferred to Phase 5.

A separate 16 by 16 LOLA fixture compares every one of its 368,640 outputs
with C# and is byte-for-byte exact. See
`docs/numba-horizon-phase-4-real-terrain.md`.

A bounded two-DEM real-terrain fixture has maximum error `4.0412e-5` degrees
with both Python-generated and captured C# segments. It remains about 124
times below the accepted `0.005` degree limit. The fixture also verifies that
later DEM passes inherit earlier horizon slopes for production hierarchy
culling.

## Near-Field Reference Merge

The optional C# near-field reference merge is disabled by the constructor
default and is not exposed by the current public Python horizon path. It is not
required for initial replacement parity and is deliberately outside this core
kernel stage. A future adoption decision must either leave the optional native
mode explicitly unsupported or evaluate and port it as separate scope.

## Qualification

Phase 4E demonstrates that Python/Numba now generates production-shaped slope
and degree buffers that match the selected C# fixture. It does not yet provide
patch scheduling, streams and queues, cancellation, progress, persistent
caches, compression, or horizon-file output. The inherited hierarchy
bilinear-boundary defect documented in Phase 4D is corrected in C# and Numba.
A ten-case dense-bilinear matrix characterizes the intentional adaptive terrain
approximation without gating it. The uniform-solar-disk model gives a maximum
sunlight-fraction difference of `1.0291e-4`. Broader fitted-ray and actual
temporal Sun-geometry coverage remain adoption evidence rather than kernel-port
blockers.

Regenerate the machine-readable report with:

```bash
env PYTHONPATH=src \
  /e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/capture_phase4e_subpatch.py
```
