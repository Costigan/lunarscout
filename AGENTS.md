# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

Lunarscout is a standalone Python library for lunar terrain, raster, temporal,
and optional native lighting analysis. The public Python package lives in
`src/lunarscout`.

The native C# implementation currently lives under `native/moonlib`, with tests
under `native/tests`. Native features are optional and are accessed from Python
through Python.NET.

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
`native_horizon.py`, `native_temporal.py`, and `native_product.py`.

Native-backed features may also be available through `ls.native`, especially
for status, initialization, and native-specific details. Keep the simple user
path available at the root when that is the expected API.

Names beginning with `_` are private. Tests and examples are not public API.

## Native Runtime Rules

Pure-Python imports and pure-Python functionality must not initialize
Python.NET, CLR, GDAL native bindings, SPICE, or `moonlib`.

Native capability checks should remain lazy:

```python
import lunarscout as ls

status = ls.native.status()
if status["available"]:
    ls.native.initialize()
```

Native functions should validate cheap Python-side inputs before bootstrapping
Python.NET where practical. If a native API needs `moonlib`, import it lazily
through `src/lunarscout/_native_runtime/bootstrap.py` or helpers in
`src/lunarscout/native.py`.

Native features currently require either a local build:

```bash
dotnet build native/moonlib/moonlib.csproj
```

or `LUNARSCOUT_MOONLIB_DLL` pointing at the built `moonlib.dll`.

## Error Handling

Use structured Lunarscout exceptions from `src/lunarscout/errors.py`.

Prefer:

- `NativeInputError` for invalid user inputs detected before native execution.
- `NativeBootstrapError` or `NativeUnavailableError` for runtime/bootstrap
  failures.
- `NativeTemporalError` for temporal native failures.
- `NativeProductError` for file-producing native product failures, including
  horizon and PSR generation.

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
.venv/bin/python -m pytest tests/test_native.py tests/test_native_horizon.py -q
```

Build native C# when changing native code:

```bash
dotnet build native/moonlib/moonlib.csproj
```

Representative native C# tests:

```bash
dotnet test native/tests/HorizonGen.Tests/HorizonGen.Tests.csproj
```

Native Python tests should not require a real GPU or real `moonlib` unless the
test is explicitly an integration test. Prefer fakes for public wrapper
behavior and separate local scripts/examples for long-running native validation.

## Examples and Scripts

Executable examples live in `examples/`. Utility and local validation scripts
live in `scripts/`.

Scripts may include editable placeholder paths for local DEMs and output
directories, but they should make it clear which values users must edit before
running.

For native example runs, ensure the native runtime is built and configured
before Python starts.

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
