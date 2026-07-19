# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

Lunarscout is a standalone Python library for lunar terrain, raster, temporal,
horizon, lighting, visibility, and landed-mission analysis. The public Python
package lives in `src/lunarscout`; production numerical work uses NumPy and
Numba CPU/CUDA implementations.

Read `docs/USER_GUIDE.md` before making broad API or architecture changes.

## Dependency Boundary

Lunarscout was split from Lunar Analyst. Keep dependency direction one-way:

```text
lunar_analyst -> lunarscout
```

Do not import Lunar Analyst application modules from `src/lunarscout`. This
package must not grow FastAPI routes, web UI code, assistant/RAG logic,
application job handlers, scenario database mutation, or notebook-runner
helpers.

## Public API Shape

The package root is a curated user-facing API:

```python
import lunarscout as ls
```

It is acceptable for high-level functions to be exported from
`src/lunarscout/__init__.py` when they are intended for normal user scripts.
Implementation modules should stay focused by domain, for example
`horizon.py`, `products.py`, and `temporal_store.py`.

CUDA status and diagnostics belong under `ls.cuda`. Scientific functions use
domain names rather than implementation-specific names.

Names beginning with `_` are private. Tests and examples are not public API.

## Runtime Rules

Importing Lunarscout must not initialize CUDA, load SPICE kernels, open raster
datasets, write files, or perform network access. Explicit-vector product calls
must not import SpiceyPy or touch the SPICE kernel pool.

CUDA capability checks remain lazy:

```python
import lunarscout as ls

status = ls.cuda.status()
```

Horizon generation is CUDA-only. Downstream products support `backend="auto"`,
`"cpu"`, and `"cuda"`; explicit CUDA failures never fall back, while automatic
selection may fall back to CPU. CPU selection must not probe CUDA.

## Error Handling

Use structured Lunarscout exceptions from `src/lunarscout/errors.py`.

Use domain exceptions such as `InputError`, `GridError`, `VectorError`,
`HorizonGenerationError`, `CudaError`, `ProductCalculationError`, and
`ProductStorageError`.

Include stable `code=` values and useful `details=` for failures that callers
may inspect.

## Raster and Temporal Conventions

Raster values are ordinary NumPy arrays. Geospatial metadata is carried
explicitly in `GeoReference`.

Do not infer grid compatibility from array shape alone. Use `same_grid` or
`require_same_grid` before combining georeferenced rasters.

Temporal arrays use UTC coordinates. In-memory cubes are shaped
`(time, y, x)`. File-backed temporal series should stream when possible rather
than loading full cubes unnecessarily.

File-producing operations should preflight paths, use staging output when
failure would otherwise corrupt an existing product, and preserve completed
outputs on failed overwrites.

## Tests and Verification

Use the repository virtual environment:

```bash
.venv/bin/python -m pytest -q
```

Run focused Python tests during development, for example:

```bash
.venv/bin/python -m pytest tests/test_public_horizon.py \
    tests/test_public_lightmap.py -q
```

Ordinary tests remain CPU-only. Real CUDA tests must be explicitly gated with
`LUNARSCOUT_REQUIRE_NUMBA_CUDA=1` and run where the NVIDIA device is visible.
Do not treat sandbox GPU visibility failures as evidence that the host lacks a
GPU.

## Examples and Scripts

Executable examples live in `examples/`. Utility and local validation scripts
live in `scripts/`.

Scripts may include editable placeholder paths for local DEMs and output
directories, but they should make it clear which values users must edit before
running.

GPU examples must clearly state that a compatible NVIDIA device and driver are
required.

## Editing Guidelines

Keep changes scoped to the requested behavior. Avoid unrelated refactors,
metadata churn, or formatting sweeps.

Respect dirty working trees. Do not revert changes you did not make.

Use ASCII unless a file already uses another character set or the change clearly
requires non-ASCII text.

Prefer direct, explicit code over new abstractions unless an abstraction removes
real duplication or matches an established local pattern.

When adding public API, update tests and consider whether `docs/USER_GUIDE.md`
or an example script should also be updated.
