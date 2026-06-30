# Lunarscout Architecture

Status: Current architecture guide for the standalone `lunarscout` library.

This document is the first place to read when trying to understand the shape of
the `lunarscout` repository. It describes the intended current architecture and
key decisions. ADRs are historical decision records; some may be proposed,
superseded, or rejected. This document should stay aligned with implemented
code.

## Purpose

Lunarscout is a standalone Python library for lunar terrain, raster, temporal,
and optional native lighting analysis.

It is independent of Lunar Analyst. It must not import or depend on Lunar
Analyst backend, web, assistant, job, scenario database, or notebook-runner
modules. Lunar Analyst may later use Lunarscout as an application dependency.

Dependency direction:

```text
lunar_analyst -> lunarscout
```

Never:

```text
lunarscout -> lunar_analyst
```

## Repository Layout

```text
lunarscout/
  pyproject.toml
  README.md
  CHANGELOG.md
  LICENSE
  docs/
    ARCHITECTURE.md
    ...
  examples/
    ...
  src/
    lunarscout/
      __init__.py
      alignment.py
      errors.py
      georeference.py
      geotiff.py
      native.py
      native_product.py
      native_temporal.py
      regions.py
      scenario.py
      temporal.py
      temporal_store.py
      terrain.py
      _native_runtime/
        bootstrap.py
  native/
    moonlib/
      moonlib.csproj
      ...
    tests/
      HorizonGen.Tests/
  tests/
    ...
  tools/
    ...
```

## Python Package Boundaries

The public Python package lives under `src/lunarscout`.

Core modules:

- `georeference.py`: CRS, affine transform, raster grid metadata.
- `geotiff.py`: GeoTIFF read/write helpers.
- `terrain.py`: slope, aspect, hillshade-style array operations.
- `alignment.py`: grid compatibility and resampling.
- `regions.py`: connected-region analysis.
- `temporal.py`: UTC time ranges and in-memory temporal cubes.
- `temporal_store.py`: file-backed temporal GeoTIFF series.
- `scenario.py`: filesystem-safe scenario path helpers only.
- `native.py`: native capability discovery and public native entry points.
- `native_temporal.py`: native temporal and lightmap-buffer APIs.
- `native_product.py`: native file-producing products such as PSR rasters.
- `_native_runtime/`: private runtime discovery/bootstrap implementation.

The package must not contain FastAPI routes, web UI code, assistant/RAG logic,
Lunar Analyst job handlers, scenario database mutation, or notebook-runner
helpers.

## Public API Policy

Lunarscout uses Semantic Versioning.

Before `1.0.0`, public APIs are provisional and breaking changes may occur in
minor releases. Intentional breaking changes must be recorded in
`CHANGELOG.md`. Patch releases should not intentionally break documented
behavior.

Public API includes:

- names exported from `lunarscout.__init__`;
- documented functions/classes in public modules;
- documented file formats and manifests written by the package;
- documented exception classes and error codes.

Not public API:

- names beginning with `_`;
- tests;
- examples;
- C# internals unless documented as Python-callable library API.

## Native Architecture

Native compute is optional. Pure-Python imports and pure-Python functionality
must not initialize Python.NET, CLR, GDAL native bindings, SPICE, or `moonlib`.

Native source currently lives in:

```text
native/moonlib
```

The initial standalone extraction intentionally includes `moonlib` wholesale.
This is a temporary migration posture. The native source and tests should be
trimmed later to the library-owned surface after the public Python/native API is
better understood.

The Python native runtime code lives in:

```text
src/lunarscout/_native_runtime
```

Current native packaging strategy:

1. Pure-Python package installation works without building native code.
2. Native features require either:
   - a local native build, for example:
     `dotnet build native/moonlib/moonlib.csproj`; or
   - an explicit `LUNARSCOUT_MOONLIB_DLL` environment variable pointing at
     `moonlib.dll`.
3. Prebuilt native wheels are deferred.
4. Runtime discovery should remain flexible enough to support future packaged
   payloads or separate platform-specific native packages without changing
   public calculation APIs.

Before distributing native binaries, the project must audit redistribution
terms for .NET/runtime assumptions, GDAL/PROJ, CSPICE/SPICE assets, HDF5,
HDF.PInvoke, ILGPU, MaxRev.Gdal packages, and bundled static files.

## Native Buffer Streaming

The current low-level native lightmap-buffer API is built around Python-owned
NumPy buffers.

Ownership model:

1. Python allocates a reusable pool of C-contiguous NumPy arrays.
2. Python passes available buffers to C# as `(buffer_id, pointer, byte_length)`
   records.
3. C# owns each passed buffer only while it is queued, being filled, or waiting
   to be returned.
4. `Poll` returns filled/error records identifying buffer IDs.
5. Once C# returns a record, it forgets that buffer.
6. Python may inspect the array, then pass the same buffer again in a later
   poll.

This avoids a permanent native registry of Python pointers and keeps Python as
the owner of memory.

## Scenario Model

Lunarscout's `Scenario` is a filesystem path helper, not an application state
owner.

It may provide safe paths such as DEM and horizon locations. It must not own or
mutate Lunar Analyst `scenario.db`, product catalogs, FastAPI state, job state,
or map-layer registration.

## Tests

Pure-Python tests live in:

```text
tests/
```

Native C# tests live in:

```text
native/tests/
```

The source boundary test must continue to enforce that `src/lunarscout` does
not import `backend` or `lunar_analyst`.

Representative commands:

```bash
PYTHONPATH=src python -m pytest tests -q
dotnet build native/moonlib/moonlib.csproj
dotnet test native/tests/HorizonGen.Tests/HorizonGen.Tests.csproj --filter FullyQualifiedName~FillLightmapBuffersTests
```

## Current Open Work

- Replace or redesign the old high-level temporal reducer default path so it
  uses standalone native APIs rather than Lunar Analyst streaming adapters.
- Treat high-level native temporal APIs as provisional until the library-level
  API is matured.
- Decide future native binary/wheel packaging after license and redistribution
  review.
- Add CI later. The initial public repository commit will not enable CI.
- Trim native source/tests to the minimal library-owned surface after behavior
  is stable.

## Repository Visibility

The standalone GitHub repository is intended to be public. Initially there are
no external committers.
