# TODO: horizon surface interpolation and stepping

Status: proposed, not started. CPU-only experiments first; no kernel or GPU
work until the results are reviewed. Written 2026-10-30.

## Problem

The safe-haven lighting check (`scripts/check_safe_haven_lighting.py`,
`docs/safe-haven-light-store.md` section 16) found that production horizon
tiles sit below a converged ray-cast reference on rugged terrain near the
upper and right edges of the 20 m DEM: median 0.41 deg, maximum 8.6 deg, with
the biggest gaps on steep rims 100-400 m from the pixel.

A production-oracle experiment on three such rays (fixed steps on the same
fitted ray, DEM 1 only) gave:

| Pixel / bin | crest range | 16 m | 2 m | 0.5 m |
|---|---|---|---|---|
| (125,20429) / 1363 | 200-400 m | 15.1 deg | 18.4 | 18.7 |
| (1024,30144) / 1254 | 200-400 m | 17.4 | 20.3 | 20.3 |
| (8174,30309) / 1212 | 100-200 m | 10.2 | 12.2 | 12.6 |

Mechanism: the terrain model is bilinear between DEM posts. It is piecewise
linear with a kink at every post, and on rugged terrain a crest is usually a
post. A sample `s/2` from a peak post reads about `slope * s/2` too low, which
is 2-3 deg at 300 m for a rim falling 25-30 m per pixel. The curvature bound
inside a cell (about 0.5 deg) does not cover this. The step floor is 10 m
inside 100 m and 16 m beyond on a 20 m DEM (0.5 / 0.8 pixel; the documented
quarter-pixel near-field step is overridden by the floor).

Relevant code: `hierarchy.py` (step floor), `fixed_step.py` (adaptive
oracle), `cuda_backend.py` (three kernel variants and the step guard),
`docs/numba-horizon-terrain-interpolation.md` (earlier decision to keep
bilinear and no cell-exit clamp, based on one 512 px tile).

## Options

| Option | Idea | Concern |
|---|---|---|
| A. Finer step floor | Brute force, about 0.1 px or finer near field | Step guard allows 16 steps per pixel; 1 m on 20 m is 20. GPU cost 1.36-1.46x measured for quarter pixel on one tile only. |
| B. Cell-crossing samples | Force a sample on every grid-line crossing so every kink is evaluated | About two samples per pixel of path. Old clamp stalled beyond about 32 km in float32; needs an ulp-safe, in-kernel-capped nudge. |
| C. Smoother surface (Catmull-Rom / Keys bicubic) | C1 surface, no kinks, crests between posts can rise above posts | 16 taps instead of 4; max-pyramid cull bound assumes no overshoot and needs a derived margin; changes the converged answer, so it is a modelling decision. |
| D. TIN or analytic per-cell maximum | Exact per-cell maximum for the chosen interpolant | New kernel math and cull argument. |
| E. No kernel change | Horizon margin or uncertainty product | Does not fix haven classification. |

B-spline and Lanczos were considered and rejected (flattens peaks; too many
taps).

## Experiment plan (CPU only)

1. **Prototype interpolants in the oracle.** Add Catmull-Rom (and optionally
   TIN) sampling next to `sample_bilinear` in a scratch oracle; keep
   production code untouched.
2. **Sample rays.** A few hundred (pixel, azimuth) rays, stratified by
   rugged/smooth and near/far from the DEM edge, including the three rays above.
3. **Measure per ray:** converged horizon (0.5 m steps) for bilinear and for
   each candidate; the horizon each scheme gives at the production floor
   (16 m), at 2 m, and with cell-crossing samples; sample counts as a cost proxy.
4. **Overshoot and cull margin.** Measure how far Catmull-Rom exceeds the
   posts in a block, and derive the pyramid margin it implies.
5. **Validate against finer terrain.** Check whether a DEM finer than 20 m
   exists near 80 S (LOLA has finer products at some sites; coverage not yet
   checked). If so: downsample to 20 m, compute horizons with each interpolant,
   compare with horizons from the fine DEM. This is the only test of "models
   real terrain better"; without it, the interpolant is an assumption.
6. **Report:** accuracy table against the converged reference and against the
   fine-DEM horizon, sample-count ratios, and a recommendation.

## Decisions to make after the CPU results

- Accuracy target (largest per-pixel horizon difference near the Sun against
  the converged reference) and the GPU slowdown that is acceptable.
- Whether the new profile becomes the default or stays opt-in (legacy profile
  keeps the C#-parity fixtures and pinned values valid).
- Approval of a bounded GPU experiment, described to the user first (in-kernel
  iteration caps; `nvidia-smi` utilization, power and P-state before and
  after; see AGENTS.md GPU safety).

## If the change proceeds

- Kernels: three variants in `cuda_backend.py`, the CPU oracles, and a
  reworked step guard.
- Tests to update: `test_phase4_hierarchy_safety`, `test_hierarchy_cull_bound`,
  `test_phase4_subpatch`, `test_direct_reference_accuracy` (GPU-gated), the
  phase1/phase4d C# parity data, and the pinned safety-matrix values
  (4.77 deg, 0.08315321).
- Provenance: tiles carry no generator version, so new tiles need a new
  directory and a recorded stepping/interpolation profile (also in the light
  store manifest).
- Regeneration scale: 56,178 horizon tiles, 1.36 TB, about 6 s per patch on the
  512 px fixture benchmark; downstream light store, stage 1-4, chain maps and
  viewer data all rerun.
- Acceptance: `check_safe_haven_lighting.py` on the whole DEM and with
  `--region upper_right_edges`, plus the oracle comparison above.

## Open items

- Last-column edge pixels (for example (9635,30335), 8.6 deg in a narrow
  sector) are unexplained.
- (8174,30309): the production oracle matches the tile at only one bin;
  possibly 8 px tile versus 128 px patch subpatch clamping near the edge.
- `DemGrid.elevation` in `geometry.py` returns 0 in the last row/column strip;
  production avoids it by stopping at width-1 / height-1.
