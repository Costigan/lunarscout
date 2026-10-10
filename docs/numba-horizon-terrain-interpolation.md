# Horizon terrain interpolation and level-0 stepping: design note

Status: decided 2026-10-09. Production horizon generation keeps **bilinear terrain** and **level-0
steps that do not stop at cell exits** (the behaviour since `d543e6e`). This note records the
evidence behind that decision, what it costs, and what to do if the question is reopened.

All measurements were taken on 2026-10-09 on an RTX 5090 Laptop GPU and its host CPU, on the
512 x 512, 20 m LOLA fixture (`scripts/numba_horizon/acquire_phase1_real_fixture.py`) unless
stated. Every figure can be reproduced with the scripts listed at the end.

## How the horizon kernel samples terrain

- The DEM is a grid of nodes, one per pixel. Between nodes the kernel treats terrain as
  **bilinear**: each level-0 cell is the square between four neighbouring nodes. A bilinear surface
  never rises above its highest node, so a single raised node is a sharp "tent" one pixel wide.
- A ray descends the max-pyramid until a block might hold terrain above the current horizon, then
  samples the bilinear surface at level 0 and steps forward: about one pixel per step (a quarter
  pixel within 500 m, with a 0.8 pixel floor), longer at long range under the angular budget.
- Before `d543e6e` (2026-07-29) each level-0 step was cut short at the **cell exit**, so a sample
  always fell on the grid line where the ray left the cell. `d543e6e` removed that clamp from
  production for outer-DEM speed. The CPU oracle and the traced kernel kept it until `36738ea`
  (2026-10-09), which made them step like production again.

## Evidence

### 1. Without the cell-exit clamp, a step can straddle a single raised node (synthetic)

The hierarchy safety matrix places a single raised node on flat ground and traces a ray past it.
In the worst case (`level0_boundary_southeast`: a 150 m node about 850 m away, ray through the node):

| Stepping | Samples nearest the node | Horizon |
|---|---|---|
| Production (no clamp) | 75.5 m at 0.29 px before it, 51.1 m at 0.42 px after it | 5.21 deg |
| Stall-safe clamp | 75.5 m, then a sample on the cell exit, which is the node: 150.0 m | 10.12 deg |
| Dense fixed-step reference | | 9.98 deg |

Maximum dense-minus-hierarchy gap over the ten matrix cases and the C#-derived production fixture
(a 5,000 m single-node obstacle):

| Variant | Max gap | Notes |
|---|---|---|
| Production (no clamp) | 4.77 deg | four boundary cases and the fixture miss by 3.5 to 4.8 deg |
| Stall-safe clamp | 0.56 deg | exactly the pre-`d543e6e` value |
| Quarter-pixel steps, no floor | 1.18 deg | shrinks but does not close the misses |
| Clamp + quarter-pixel | 0.14 deg | |

These cases are the bilinear model's worst case: one isolated node, perfectly flat surroundings, a
ray through or beside the node. The clamp works because bilinear crests lie on grid lines; it is a
fix specific to bilinear terrain.

### 2. On real terrain the stepping effect is small

Production versus the bilinear float64 brute force, sun-relevant rays (reference horizon -2 to 5
deg, 463 rays from 40 pixels), with the H-A and H-B fixes:

| Stepping | Mean \|d\| | Rays > 0.05 deg | Worst ray | GPU time vs production (512 px) |
|---|---|---|---|---|
| Production | 0.0149 deg | 6.0% | 0.29 deg | 1.00 |
| + cell-exit clamp | 0.0112 deg | 4.1% | 0.45 deg | about 1.16 |
| + quarter-pixel steps | 0.0061 deg | 1.3% | 0.25 deg | 1.36 to 1.46 |

The clamp changes real-terrain horizons by about 0.005 deg on average and made the worst ray worse.
None of these variants was timed on a large or outer DEM, which is where `d543e6e` found its speed.

### 3. The terrain model matters far more than the stepping

The same real-terrain rays traced through an **interpolating cubic B-spline** (passes through every
node, C2, curvature-minimizing) instead of bilinear terrain, both at quarter-pixel steps:

| Limiting terrain | Sun-relevant rays: cubic - bilinear, mean \|d\| | > 0.05 deg |
|---|---|---|
| 0 to 20 m (52 rays) | 0.37 deg | 79% |
| 20 to 100 m (95) | 0.35 deg | 78% |
| 100 to 500 m (94) | 0.08 deg | 28% |
| 500 m to 2 km (157) | 0.03 deg (p95 0.02) | 3% |
| beyond 2 km (65) | 0.002 deg (p95 0.007) | 0% |

- Over all 815 real-terrain rays: mean \|d\| 0.119 deg, mean signed +0.110 deg (the cubic surface is
  higher; it can rise above nodes on convex ground, bilinear cannot), max 2.85 deg.
- The bilinear reference itself is converged: 1/16 versus 1/4 pixel steps differ by 0.004 deg mean.
- Within 500 m, where over half of the sun-relevant rays are limited, the choice of surface changes
  horizons 10 to 30 times more than production's entire difference from the bilinear reference.
  Beyond 2 km the two effects are the same size.
- Neither surface is known to be right: the DEM carries no information below one pixel. Every
  accuracy figure in the 0.1.0rc5 review is relative to bilinear terrain.

### 4. Cost of a cubic surface

Stand-alone GPU kernels, 26.2 M lookups, median of 7 runs:

| Grid and access | Bilinear (4 taps) | Cubic B-spline (16 taps) | Catmull-Rom (16 taps) |
|---|---|---|---|
| 8,192 px (256 MB), along rays | 34 G lookups/s | x1.88 to x1.95 | x1.68 to x1.72 |
| 512 px (cached), along rays | 31 G lookups/s | x1.77 | x1.58 to x1.66 |
| 8,192 px, scattered | 10 G lookups/s | x2.29 | x2.31 |

- This is the lookup alone. The kernel also traverses, culls and computes slopes, so the end-to-end
  slowdown lies between almost nothing and about x1.9; the share of production time spent in level-0
  lookups has not been measured.
- Implementation matters: a first, naive cubic kernel (weights recomputed with powers inside the
  16-tap loop) was 7 to 13 times slower than bilinear.
- Structural costs beyond the lookup:
  1. **Overshoot breaks the cull bound.** Both cubics can exceed the highest surrounding node. The
     max-pyramid assumes they cannot, so culling would become non-conservative (review finding H-B
     again) unless level 0 of the pyramid bounds the interpolated surface, not the nodes.
  2. **Prefilter (B-spline only).** One pass over each DEM (3.9 ms on the CPU for 512 px; roughly
     15 s for a 30,400 px DEM by linear scaling, unmeasured) and a second DEM-sized array in GPU
     memory. Catmull-Rom needs neither, but is only C1 and its accuracy was not measured.
  3. **Larger nodata footprint.** A 4x4 neighbourhood instead of 2x2 roughly doubles the area each
     invalid node affects.

## Decision

Keep bilinear terrain and production stepping without the cell-exit stop.

- The real-terrain evidence for changing the stepping is weak (about 0.005 deg mean on one tile)
  and measured against a reference that assumes the same bilinear surface.
- The clamp is a bilinear-specific fix, costs outer-DEM speed, and made the worst real ray worse.
- A curvature-minimizing surface is the more realistic model, but it is not established as more
  accurate, it moves near-field horizons by tenths of a degree, and it costs about x1.7 to x1.9 per
  lookup plus a new cull bound.

The safety-matrix values (4.77 deg; production fixture 0.08315321) are pinned as diagnostics of the
bilinear worst case, not acceptance criteria.

## Limits of this evidence

- One 512 px polar LOLA tile at 20 m; 815 and 463 rays from 12 and 40 pixels.
- No comparison with independent observations of illumination.
- No large-DEM or outer-DEM timing of any variant.
- Catmull-Rom accuracy not measured.

## Recommended next steps if the question is reopened

1. **Choose the surface on observational grounds.** Compare predicted shadow and illumination with
   independent observations at known times (for example, orbital images with measurable shadow
   edges at a polar site). Without this, bilinear versus cubic is a modelling choice, not a
   correctness question.
2. **State near-field uncertainty in the meantime.** For products driven by horizons limited within
   500 m (most polar sun-relevant rays), the bilinear-versus-cubic spread (about 0.1 to 0.4 deg mean)
   is a fair estimate of terrain-model uncertainty. Consider documenting it in the user guide, or
   offering it as an optional uncertainty product.
3. **Broaden the evidence before deciding.** Repeat `s8` on several tiles: other latitudes, rougher
   terrain, 5 m and 60 m DEMs, and outer DEMs. CPU only, a few minutes per tile.
4. **Measure end-to-end cost before adopting a cubic.** Swap the production level-0 lookup for
   Catmull-Rom in a scratch copy (the `s6` patching harness), and time it against production on the
   512 px fixture and a 2,048 px or larger DEM, with time limits and the step guard. Timing is valid
   even before the cull bound is fixed; accuracy is not.
5. **If a cubic is adopted:** build level 0 of the max-pyramid from per-cell maxima of the
   interpolated surface (or a proven overshoot bound), widen invalid-node handling to the 4x4
   footprint, prefilter on the GPU if the B-spline is chosen, and add the surface to the CPU oracle
   and the brute-force reference so parity and accuracy tests measure the same model.
6. **If bilinear is kept but tighter agreement with the bilinear reference is wanted:** time
   quarter-pixel steps on a large DEM; on 512 px it cut sun-relevant rays above 0.05 deg from 6.0%
   to 1.3% for 36% to 46% more GPU time. A cell-exit clamp should only return with a
   strictly-advancing nudge (see below).

### Hazard: the original clamp can stall

The pre-`d543e6e` clamp nudged each step 1e-6 km past the cell exit. Beyond about 32 km the float32
ray parameter cannot represent that increment, so a ray sitting on a cell exit stops advancing. An
experiment that used it on a 2,048 px DEM kept the GPU at 100% overnight. The stall-safe version used
here nudges by max(1e-6 km, 4 float32 ulps of the position). Production now caps each ray at 16
steps per pixel of range plus 1000 (`3e20291`), so a stall raises `horizon_step_limit_exceeded`
instead of hanging.

## Reproduction

| Figure | Command | Device | Time |
|---|---|---|---|
| Evidence 1 | `.venv/bin/python review_scripts/s7_stepping_matrix.py` | CPU | under 1 min |
| Evidence 2 | `.venv/bin/python review_scripts/s6_cull_step_isolation.py --dem /tmp/lunarscout-phase1-lola-512.tif --variants baseline,clamp,fine` | GPU | about 3 min |
| Evidence 3 | `.venv/bin/python review_scripts/s8_interpolation_sensitivity.py --dem /tmp/lunarscout-phase1-lola-512.tif` | CPU | about 3 min |
| Evidence 4 | `.venv/bin/python review_scripts/s9_interpolation_cost.py --dem /tmp/lunarscout-phase1-lola-512.tif` | GPU | under 1 min |

Evidence 2's table comes from the 2026-10-09 `s6` runs recorded in `docs/code_review_0.1.0rc5.md`.
Those runs applied the H-A patch as an `s6` variant, before H-A was committed; on the current tree
`baseline` already includes it. `s6`'s `clamp` variant uses the original 1e-6 km nudge, which is
safe on the 512 px fixture (rays under 12.3 km) but stalls beyond about 32 km: do not run it on
larger DEMs. The in-kernel step guard and `--timeout` limit the damage if it is.
