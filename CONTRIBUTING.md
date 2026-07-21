# Contributing to Lunarscout

## Quick start

```bash
git clone <repo-url> lunarscout
cd lunarscout
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -c "import lunarscout as ls; print(ls.__version__)"
```

The editable install gives you a working `import lunarscout` while keeping
the source tree as the import target.  If you prefer not to install, set
`PYTHONPATH` instead:

```bash
export PYTHONPATH="$PWD/src"
```

## Running tests

### CPU test suite (no GPU required)

```bash
PYTHONPATH="$PWD/src" .venv/bin/python -m pytest -q
```

Expected: 450+ tests pass, a small number are skipped.  No GPU is needed.

### Focused tests during development

```bash
.venv/bin/python -m pytest tests/test_public_horizon.py \
    tests/test_public_lightmap.py -q
```

### CUDA tests (GPU required)

CUDA tests are explicitly gated behind an environment variable:

```bash
LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 .venv/bin/python -m pytest -q
```

Without this variable, every test that needs a real NVIDIA GPU is skipped.
Do not treat sandbox GPU-visibility failures as evidence that the host lacks
a GPU.

### Other optional test variables

| Variable | Effect |
|----------|--------|
| `LUNARSCOUT_RUN_TEMPORAL_BENCHMARK=1` | Runs a 3,800-layer temporal benchmark |
| `LUNARSCOUT_RUN_PHASE1_EXTERNAL_TERRAIN=1` | Validates against external DEMs |
| `LUNARSCOUT_SPICE_META_KERNEL` | Overrides the default SPICE meta-kernel path |
| `LUNARSCOUT_SPICE_KERNEL_DIR` | Overrides the SPICE kernel cache directory |

### Expected skips

Tests that require a GPU, external terrain files, network access for SPICE
kernel downloads, or a long-running temporal benchmark are skipped by default
and run only when the corresponding environment variable is set.

### Expected warnings

`pyproj` emits `UserWarning` about PROJ.4 string conversion (WKT is the
authoritative CRS representation).  `rasterio` emits `NotGeoreferencedWarning`
in tests that deliberately exercise non-georeferenced files.  These are
understood and scoped to their fixture tests.

## Building

```bash
.venv/bin/python -m build
```

Outputs appear under `dist/`:
- `lunarscout-<version>-py3-none-any.whl`
- `lunarscout-<version>.tar.gz`

Inspect the results:

```bash
.venv/bin/python -m twine check dist/*
unzip -l dist/*.whl
```

No `.dll`, `.exe`, `.pdb`, notebook checkpoints, or local scenario paths
should appear in the wheel.

## Project layout

```text
src/lunarscout/        # package source
  __init__.py          # curated public API, imported as `lunarscout as ls`
  alignment.py         # grid comparison and raster resampling
  cuda.py              # CUDA status and diagnostics (ls.cuda)
  errors.py            # structured LunarscoutError hierarchy
  georeference.py      # GeoReference, coordinate conversion
  geotiff.py           # single-band GeoTIFF read / write
  horizon.py           # public horizon generation facade
  products.py          # public lightmap, PSR, elevation, mission-duration facade
  progress.py          # progress callbacks and event models
  regions.py           # connected-region labeling and filtering
  scenario.py          # filesystem-safe scenario paths and product methods
  spice.py             # SPICE kernel management
  spice_geometry.py    # Sun / Earth local-frame vector histories
  temporal.py          # TimeRange, TemporalCube, time-axis reducers
  temporal_store.py    # file-backed TemporalGeoTiffSeries
  terrain.py           # slope, aspect, hillshade
  _cuda_runtime.py     # private CUDA runtime helpers
  _numba_horizon/      # private Numba / CUDA implementation packages
  _native_runtime/     # historical managed-runtime package (not shipped)
  data/                # packaged static data (SPICE kernel manifests)

tests/                 # test suite
examples/              # executable capability examples
scripts/               # validation, benchmarking, and release helper scripts
docs/                  # architecture, user guide, plans, benchmarks
```

## Code conventions

### Public vs private

Names beginning with `_` are private implementation details.  Tests and
examples are not public API.  The public surface is `import lunarscout as ls`
and everything reachable from `__init__.py`.

### Import laziness

`import lunarscout` must succeed on a CPU-only machine and must not:

- import Numba CUDA or create a CUDA context
- import or initialize SpiceyPy / CSPICE
- furnish or download SPICE kernels
- open Rasterio / GDAL datasets
- perform filesystem writes or network access

Raster, SPICE, and CUDA dependencies are imported by focused modules at
first use.  Selecting `backend="cpu"` must not touch CUDA.  Supplying
explicit Moon-ME vectors must not import SpiceyPy.

### Backend selection

Horizon generation is CUDA-only and has no `backend` argument.  Downstream
products accept `backend="auto"`, `"cpu"`, or `"cuda"`:

- `"cpu"` never probes or initializes CUDA.
- `"cuda"` fails with a structured `CudaError` rather than falling back.
- `"auto"` selects CUDA when available and otherwise falls back to CPU.

### Error handling

Use the structured exception classes from `src/lunarscout/errors.py`.
Every public failure should:

- subclass `LunarscoutError`
- carry a stable `code` value
- include machine-readable `details` where callers may inspect them

Domain exceptions include `InputError`, `GridError`, `VectorError`,
`HorizonGenerationError`, `CudaError`, `ProductCalculationError`, and
`ProductStorageError`.  Never suggest installing .NET or locating a DLL.

### Raster conventions

- Raster values are ordinary NumPy arrays.
- Geospatial metadata is carried explicitly in `GeoReference`.
- Never infer grid compatibility from array shape alone.  Use `same_grid` or
  `require_same_grid` before combining georeferenced rasters.
- Axes are `(y, x)`; pixel coordinates use zero-based `x` columns and `y` rows.
- Windows and patch origins use `(tile_x, tile_y)` to avoid row/column confusion.

### Temporal conventions

- Timestamps are timezone-aware UTC `datetime` values or ISO-8601 strings.
- In-memory cubes are shaped `(time, y, x)`.
- File-backed series should stream rather than load full cubes.
- Naive datetimes and timezone-free strings are interpreted as UTC.

### File safety

File-producing operations must:

1. preflight inputs and output paths before expensive initialization
2. write into a staging location
3. flush, synchronize, and atomically publish
4. preserve the previous completed output on failed overwrite

### Docstrings and type hints

All public functions expected in `__init__.py` should have complete docstrings
and type annotations.  Private helpers should have enough to make intent clear
to maintainers.

### Editing rules

- Keep changes scoped to the requested behaviour.
- Avoid unrelated refactors, metadata churn, or formatting sweeps.
- Respect dirty working trees; do not revert changes you did not make.
- Use ASCII unless a file already uses another character set.
- Prefer direct, explicit code over new abstractions, unless the abstraction
  removes real duplication or matches an established local pattern.
- When adding public API, update tests and consider whether the user guide
  or an example script should also be updated.

## Version bumping

The canonical version lives in `pyproject.toml`:

```toml
version = "0.1.0rc3"
```

When bumping, also update the version assertion in
`tests/test_dependency_boundary.py`.  There is no `_version.py` or
package-level `__version__` attribute.

## Release process

1. **Ensure a clean worktree.**  Commit or stash all changes.

2. **Run the full CPU test suite:**

   ```bash
   PYTHONPATH="$PWD/src" .venv/bin/python -m pytest -q
   ```

3. **Run CUDA tests** (gated: requires a supported NVIDIA GPU):

   ```bash
   LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 PYTHONPATH="$PWD/src" .venv/bin/python -m pytest -q
   ```

4. **Bump the version** in `pyproject.toml` and update the assertion in
   `tests/test_dependency_boundary.py`.

5. **Commit and tag:**

   ```bash
   git add pyproject.toml tests/test_dependency_boundary.py
   git commit -m "Bump version to X.Y.Z"
   git tag -a vX.Y.Z -m "vX.Y.Z"
   ```

6. **Build:**

   ```bash
   rm -rf dist/
   .venv/bin/python -m build
   ```

7. **Inspect:**

   ```bash
   .venv/bin/python -m twine check dist/*
   unzip -l dist/lunarscout-X.Y.Z-py3-none-any.whl
   ```

   Confirm no unintended files (no `.dll`, `.exe`, `.pdb`, notebooks,
   local paths, build artifacts).

8. **Smoke-test the wheel** in a fresh venv:

   ```bash
   python3.11 -m venv /tmp/lunarscout-smoke
   source /tmp/lunarscout-smoke/bin/activate
   pip install dist/lunarscout-X.Y.Z-py3-none-any.whl
   python -c "import lunarscout as ls; print(ls.__version__)"
   deactivate
   ```

9. **Upload to TestPyPI:**

   ```bash
   .venv/bin/python -m twine upload --repository testpypi dist/*
   ```

   This requires a `[testpypi]` section in `~/.pypirc` with a TestPyPI API
   token.  Never commit credentials.

10. **Verify** the project page at `https://test.pypi.org/project/lunarscout/`
    and install from TestPyPI in a fresh environment to confirm the
    installation path works.

Uploaded candidates are **immutable**.  Defects require a new version number;
never overwrite an existing index artifact.

## GPU environment

On a machine with a supported NVIDIA GPU and driver, install the CUDA profile:

```bash
pip install "lunarscout[cuda]"
```

This installs the validated Numba-CUDA CUDA 12 user-space runtime.  It does
not install an NVIDIA driver.  Both `lunarscout` and `lunarscout[cuda]` use
the same import:

```python
import lunarscout as ls
```

Check CUDA status:

```python
status = ls.cuda.status()
print(status.available)
```

Horizon generation is CUDA-only.  All downstream products (lightmaps, PSR,
elevation, safe havens, mission duration) support `backend="auto"`, `"cpu"`,
and `"cuda"` and can run on a CPU-only machine.

## Dependency boundary

Lunarscout is a standalone library.  Dependency direction is one-way:

```text
lunar_analyst -> lunarscout
```

Never import Lunar Analyst application modules from `src/lunarscout`.  The
package must not add FastAPI routes, web UI, RAG logic, job handlers,
scenario database mutation, or notebook-runner helpers.
