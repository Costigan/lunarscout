# Lunarscout Interactive Notebook Course

Executable Jupyter notebooks that teach the Lunarscout public API through
analytical stories combining related examples.

## Notebooks

| Notebook                               | Draws from  examples | Purpose                                    |
| -------------------------------------- | -------------------- | ------------------------------------------ |
| `01_raster_foundations.ipynb`           | 01--04               | GeoTIFFs, terrain, regions, grids, alignment |
| `02_temporal_workflows.ipynb`           | 05--10               | Cubes, file-backed series, streaming reducers, screening |
| `03_celestial_geometry.ipynb`           | 11--13               | Sun/Earth geometry, horizons, plots, synthetic lightmap |
| `04_map_algebra_foundations.ipynb`      | 18--21               | Raster values, validity, alignment, units, numerical policy |
| `05_suitability_and_neighborhoods.ipynb` | 22, 25               | Weighted suitability, focal cleanup, morphology, distance |
| `06_lazy_and_temporal_algebra.ipynb`    | 27, 31               | Expressions, explain/plan, bounded writes, temporal reduction |

## Running the Notebooks

From the repository root:

```bash
.venv/bin/jupyter notebook
```

Then navigate to `examples/notebooks/` and open the desired notebook.

Most notebooks use fully synthetic data and run on CPU with zero external
dependencies. `03_celestial_geometry.ipynb` requires SPICE kernel download on
first use and the synthetic horizon data bundle (downloaded automatically).

## Source Maintenance

These notebooks are committed without outputs to keep diffs readable and
outputs stale-free.  Execute them locally or in CI to produce a rendered
version.

A git clean/smudge filter strips outputs automatically on `git add` and
`git diff`.  Set it up once per checkout:

```bash
.venv/bin/pip install nbstripout
.venv/bin/nbstripout --install
```

CI verifies that committed notebooks have no outputs via `nbstripout --verify`.
To regenerate the notebook source, run:

```bash
.venv/bin/python scripts/generate_notebooks.py
```

## Companion Scripts

The command-line example scripts at `examples/` are the reference
implementations. Each notebook consolidates related scripts into a coherent
analytical narrative with inline plots, prose, and interactive exercises.
