# Numba Horizon Phase 4A CUDA Mechanics

Phase 4A establishes the smallest real-GPU execution boundary before terrain
traversal is ported. It does not calculate horizons.

## Toolchain Finding

Numba 0.66.0 initially used the system CUDA 13.2 toolkit. Device discovery
succeeded for the RTX 5090 (compute capability 12.0), but the first launch
failed with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`: CUDA 13.2 generated PTX 9.2
while driver 580.159.03 accepts PTX through 9.0.

The prototype now pins NVIDIA's external `numba-cuda` 0.30.4 target and CUDA
12.9 PyPI toolchain in
`scripts/numba_horizon/requirements-prototype.txt`. CUDA 12.9 supports Blackwell
and produces PTX that this driver can JIT. This is intentionally not a normal
Lunarscout runtime dependency.

NVIDIA documents `numba-cuda[cu12]` as the supported pip installation path and
notes that PTX applications require a compatible driver/toolkit combination:

- https://nvidia.github.io/numba-cuda/user/installation.html
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

## Implemented and Verified

`cuda_backend.py` performs no module-level Numba import. `CudaSession` is the
explicit point that imports CUDA, verifies availability, selects the device,
and builds kernels.

The real-GPU test and capture validate:

- allocation, host-to-device copy, launch, synchronization, and result copy;
- two-dimensional `X=pixel`, `Y=azimuth` mapping and
  `pixel * azimuth_count + azimuth` output order;
- rounded launch dimensions with explicit bounds checks;
- four-segment shift and bilinear interpolation;
- quartic position, tangent, and planar-to-chord evaluation;
- clamped bilinear DEM sampling and the `-32000` invalid sentinel;
- the strict finite-and-greater-than-`-20000` elevation rule;
- subpatch-center clamping and interpolation selection.

The CPU helpers are arithmetic oracles. Their normal test does not require a
GPU and must not be described as CUDA validation. The separately gated test
requires `LUNARSCOUT_REQUIRE_NUMBA_CUDA=1` and ran on the physical RTX 5090.

Regenerate the report with:

```bash
/e/projects/lunarscout/.venv/bin/python \
  scripts/numba_horizon/capture_phase4a_cuda_mechanics.py
```

The report is `docs/numba-horizon-phase-4a-cuda-mechanics.json`.

## Next Boundary

Stage 4B will implement deliberately fixed-step, level-0 traversal and compare
selected rays sample by sample with the independent Phase 1 reference data.
Adaptive stepping and hierarchy remain out of scope until that fixed-step
diagnostic is correct.
