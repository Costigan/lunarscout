# Numba Horizon Phase 2 Data Contract

Phase 2 freezes the NumPy representation at the future host/device boundary.
It does not initialize CUDA, import Numba, load Python.NET, or implement horizon
geometry or traversal. The private implementation is in
`src/lunarscout/_numba_horizon/contract.py`.

## Precision Boundary

Host geometry, projection calculations, ray samples, and polynomial fitting
remain C-contiguous `float64`. Conversion to C-contiguous `float32` occurs only
when values enter a device-facing contract. Integer device metadata is
bounds-checked before conversion to `int32`; Phase 1 pyramid metadata is stored
as `int64`, so its adapter performs an explicit checked narrowing conversion.

The frozen field vectors are:

| Data | Dtype | Shape |
| --- | --- | --- |
| Map parameters | `float32` | `(11,)` |
| Projection parameters | `float32` | `(6,)` |
| Pyramid level metadata | `int32` | `(level, 4)` |
| Kernel float parameters | `float32` | `(5,)` |
| Kernel integer parameters | `int32` | `(4,)` |
| Segment values | `float32` | `(azimuth, subpatch, DEM, 18)` |
| Segment DEM IDs | `int32` | `(azimuth, subpatch, DEM)` |
| Slope output | `float32` | `(pixel, azimuth)` |

Field names and order are constants beside the implementation and are covered
by tests. Distances retain the Phase 1 units: host samples use meters, segment
distance and polynomial parameters use kilometers, elevations use meters,
slopes are dimensionless, and angles use degrees.

## Segment Storage Decision

Segments use a dense four-dimensional float tensor plus a separate DEM-ID
tensor. This is intentionally neither an aligned NumPy structured dtype nor a
fully field-split structure of arrays.

The production thread interpolates four neighboring segments and consumes
almost every coefficient. A packed 18-float vector preserves per-segment
locality and the exact C# flattening expression:

```text
((azimuth * subpatch_count) + subpatch) * dem_count + dem
```

A structured dtype would introduce record-alignment and Numba-record support
risk without adding useful semantics. A fully split structure of arrays would
require 18 separate arrays or device arguments despite the all-field access
pattern. The dense tensor is already the language-neutral Phase 1 artifact
representation, and its final field axis remains contiguous.

## Pyramid Storage

Level zero remains a two-dimensional `float32[y, x]` array. Levels above zero
are concatenated into one C-contiguous `float32` mip buffer. Metadata columns
are `(level, offset, width, height)`. Level-one offset is zero; every later
offset is the cumulative size of preceding mip levels. Cell lookup is:

```text
mips[offset + y * width + x]
```

The contract validates every offset, dimension, buffer length, and cell bound
without relying on ILGPU `ArrayView` behavior.

## Configuration and Outputs

The contract validates positive dimensions, supported subpatch sizes, the
128-pixel tile ceiling, C-contiguity, exact dtypes and shapes, finite
coefficients, DEM-axis IDs, and index bounds before any future CUDA
initialization. Subpatch-grid dimensions deliberately reproduce the current C#
rule, including its use of tile width for both grid axes; Phase 2 does not
silently correct production behavior.

Slope buffers initialize to negative infinity. DEM passes update a common
accumulated slope buffer. `HorizonBuffers.degrees()` is the only device-contract conversion to
degrees and is called after merging; it returns a new `float32` array using the
same `MathF.Atan`-equivalent precision as C# and preserves negative-infinity
sentinels.

## Reference Gate

`load_reference_artifact` loads all 822 Phase 1 arrays with `allow_pickle=False`
and verifies names, dtypes, shapes, C-order data hashes, and contiguity without
CUDA or Python.NET. Tests additionally prove:

- all captured pyramids round-trip through flattened mip storage;
- selected pyramid cells reproduce the C# artifact;
- segment flattening matches C-order and the C# index expression;
- the boundary-halo interpolation selects C# centers 0, 1, 4, and 5 and
  reproduces the clamped segment exactly;
- host sample arrays remain `float64` while segments cross the boundary as
  `float32`;
- per-DEM slopes merge exactly and the single degree conversion reproduces the
  production buffer exactly.
