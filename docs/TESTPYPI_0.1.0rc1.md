# TestPyPI `0.1.0rc1` Candidate Evidence

Status: pre-upload draft; no upload is authorized or complete.

This report records the limited-user-testing release gates. It must be updated
with the exact clean candidate commit and artifact hashes after the current
reviewed changes are committed. A dirty diagnostic artifact is not an upload
candidate.

## Intended artifact and installation profiles

```bash
python -m pip install lunarscout
python -m pip install "lunarscout[cuda]"
```

Both profiles import as `lunarscout`. The base profile supports every
downstream product on CPU and does not install CUDA runtime packages. The
`cuda` profile installs the validated Numba-CUDA CUDA 12 user-space stack but
does not install an NVIDIA driver.

## Completed evidence

- Ordinary source CPU suite with CUDA disabled: 395 passed, 17 skipped.
- Representative restart, cancellation, checkpoint, journal-failure,
  process-exit, failed-write, and public failure suite: 92 passed, 9 gated
  skips.
- Clean Python 3.12 base-wheel suite: 380 passed, 27 optional or real-GPU
  skips; `pip check` passed.
- Clean Python 3.11 base-wheel public/core suites: 27 passed and 83 passed,
  each with one optional skip; `pip check` passed.
- Clean installed CUDA suite: 159 passed, 1 skipped.
- Final public installed-wheel base checks: 35 passed, 2 gated skips.
- Final installed-wheel real-GPU horizon and complete public downstream
  CPU/CUDA comparison: 2 passed.
- Public comparison checked complete arrays, validity masks, integer identity,
  floating-point agreement, and truthful CPU/CUDA metadata for lightmap, PSR,
  both elevation products, safe havens, and all four mission-duration
  operations.
- Read-only configured Numba cache smoke: 2 passed.
- Downstream public example CLI executed from `/tmp` with a clean installed
  base-wheel interpreter and no source-tree `PYTHONPATH`.
- Wheel and sdist builds passed explicit content allowlists and Twine checks;
  the sdist independently rebuilt an allowlisted wheel.
- Import probes loaded no Numba, CUDA, SpiceyPy, CLR, Python.NET, or moonlib,
  opened no raster, and performed no Lunarscout-directed filesystem
  initialization.

The clean Python 3.11 profile resolved to Numba 0.66.0, NumPy 2.4.6, Rasterio
1.4.4/GDAL 3.10.3, SpiceyPy 8.1.2/CSPICE N0067, SciPy 1.17.1, and PyProj
3.7.2. The clean Python 3.12 profile resolved to Numba 0.66.0, NumPy 2.4.6,
Rasterio 1.5.0/GDAL 3.12.1, SpiceyPy 8.1.2/CSPICE N0067, SciPy 1.18.0, and
PyProj 3.7.2.

## Current-tree diagnostic artifacts

The uncommitted tree was copied into an isolated build snapshot and produced
artifacts that passed the content allowlists and Twine checks. These artifacts
were installed outside the checkout on Python 3.11, Python 3.12, and the CUDA
profile. They are diagnostic only because their release record correctly says
`"dirty_worktree": true` and `"candidate_artifacts": false`:

- `lunarscout-0.1.0rc1-py3-none-any.whl`: 157,848 bytes, SHA-256
  `78cd8c2eaa24d89db9bfe356871e7b0828a85ceedea128862065b2cf4e4610fd`;
- `lunarscout-0.1.0rc1.tar.gz`: 134,928 bytes, SHA-256
  `cdea5ca897b1f7978f7bed4e1ce5ddae4189ae65eb47d2e2527e191b6f429ca1`.

The exact diagnostic wheel passed 59 public/Scenario tests with 7 gated skips
on each of Python 3.11 and 3.12, and `pip check` passed in both environments.
The public example CLI ran from `/tmp` using the installed Python 3.12 package
with no source-tree `PYTHONPATH`.

The installed CUDA wheel ran the real-kernel public horizon test and the
complete downstream CPU/CUDA product matrix: 2 passed in 14.26 seconds. The
test covered complete arrays, masks, backend metadata, and actual horizon and
downstream kernels. The imported package path was the clean environment's
`site-packages/lunarscout/__init__.py`, not the checkout.

With `NUMBA_DISABLE_CUDA=1` in that same installed CUDA environment, an
end-to-end automatic lightmap completed on CPU and recorded
`LUNARSCOUT_COMPUTE_BACKENDS=["cpu"]`. An explicit CUDA request raised the
stable `cuda_lightmap_unavailable` error and created no output.

## NVIDIA environment

- Device: NVIDIA GeForce RTX 5090 Laptop GPU.
- Compute capability: 12.0.
- Host driver: 580.159.03.
- CUDA driver API: 13.0.
- Numba: 0.66.0.
- Numba-CUDA: 0.30.4.
- CUDA toolkit package: 12.9.2.0.
- GPU memory observed by `ls.cuda.status()`: 25,146,949,632 bytes total and
  24,870,780,928 bytes free during the recorded idle probe.

Actual-kernel tests covered horizon mapping and traversal, lightmap, PSR,
Sun/Earth elevation, safe havens, and mission-duration signal/reduction paths.
GPU visibility or backend selection alone was not counted as kernel evidence.

## Scientific identity retained

The accepted complete 1,599-patch PSR result remains:

- 62.690 seconds;
- 25.5064 patches/second; and
- SHA-256
  `e246ac369b36d3e5f67f9c6c1f64284f0ddbc26448c17358b69cdd69c9ffed5d`.

This large prototype benchmark was not rerun solely for packaging changes.
Immutable scientific fixtures, complete public CPU/CUDA arrays and masks, and
file-format tests remained unchanged. The candidate report must say so rather
than presenting the historical timing as an installed-wheel benchmark.

## Commands to repeat on the clean candidate commit

```bash
env PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_CUDA=1 \
  /e/projects/lunarscout/.venv/bin/python -m pytest -q -p no:cacheprovider

python scripts/build_release_artifacts.py /tmp/lunarscout-dist \
  --upload-target testpypi

python -m pip check
```

The explicitly gated NVIDIA diagnostic command was:

```bash
env LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 \
  NUMBA_CACHE_DIR=/tmp/lunarscout-pretestpypi-gpu-cache \
  PYTHONDONTWRITEBYTECODE=1 \
  /tmp/lunarscout-cuda-install.fy1gFy/venv/bin/python -m pytest \
  /e/projects/lunarscout-numba-horizon/tests/test_public_horizon.py::test_public_horizon_executes_real_cuda_kernel \
  /e/projects/lunarscout-numba-horizon/tests/test_public_lightmap.py::test_all_public_downstream_cpu_and_cuda_products_agree \
  -q -p no:cacheprovider
```

The final clean-candidate run must use the same gates and installed-wheel
boundary and replace temporary paths in the permanent evidence record.

## Pre-upload items still open

- Record the exact clean commit and final wheel/sdist filenames, sizes, and
  SHA-256 hashes.
- Add reviewed package author/maintainer metadata, or explicitly decide to
  omit it. Do not infer a publishable email address from local Git settings.
- Run the configured GitHub CPU/package workflow or record an explicit manual
  exception for the limited TestPyPI evaluation.
- Create or claim the TestPyPI project and configure a scoped token or trusted
  publisher outside the repository. The public JSON endpoint
  `https://test.pypi.org/pypi/lunarscout/json` returned HTTP 404 on
  2026-07-19, so no existing TestPyPI project was publicly visible.
- Obtain explicit authorization before uploading the reviewed immutable
  artifacts.

## Known limitations

The tested support matrix and candidate limitations are normative in
`docs/USER_GUIDE.md`. In particular, this candidate does not claim Windows,
macOS, CUDA 13, other GPU generations, physical TIFF-block recovery, HDF5,
map-algebra, distance-field, or path-planning support.
