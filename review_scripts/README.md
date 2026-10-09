# review_scripts: checks to run on your machine

Written during the lunarscout 0.1.0rc5 review. They are read-only with respect to the repo (they write only to temp dirs).
Run them with the repo's own virtualenv and paste each script's full output back.

| Script | Needs | What it settles |
|---|---|---|
| `s1_cpu_repros.py` | CPU only | Reproduces 8 findings: int8 multiply overflow, `-inf` horizon -> dark pixel, truncated tile skipped on resume, observer-height file-name collision, stale `.cbin` after rewrite, 50 deg clamp, PSR writer-failure hang, tolerance arithmetic. About 1 minute (the hang check waits 40 s). **I ran this in my sandbox and all 8 reproduced.** |
| `s2_horizon_vs_bruteforce.py` | GPU, DEM | Compares production horizon tiles with an independent float64 ray march. `--selftest` runs on CPU. `--synthetic wall/nan/bowl` builds test terrain from the fixture's CRS. **You ran real terrain, wall and nan on the RTX 5090 (see the report). The `nan` case was badly designed in the first version and has been fixed; re-run it. The output now also reports error by distance to the limiting terrain point and the production range cap.** |
| `s4_farfield_patch_test.py` | GPU, DEM | Patches the far-field slope formula in a temp copy of `src/` and compares production vs brute force before and after. Isolates how much of the error review finding H-A explains. Experiment only, not a proposed fix. **Run on real terrain and wall: isolates H-A (mean abs 0.300° → 0.033°); see the report.** |
| `s5_residual_diagnostics.py` | GPU, DEM | Diagnoses what s4 left: near-field outliers, exact-zero rays, sun-relevant tails. **Run 2026-10-08 by Claude Code on this machine; see the report.** |
| `s6_cull_step_isolation.py` | GPU, DEM | Builds scratch copies with the H-A patch, the cull disabled (H-B) and/or `margin_step` zeroed (M-A), and compares all of them on the same rays against one brute-force reference. **Run 2026-10-08 by Claude Code: isolates H-B; M-A inactive on this fixture; see the report.** |
| `s3_wheel_check.py` | network | Builds wheel and sdist from tracked files, lists contents, installs into a clean venv, tries imports, `pip check`, and the worker entry point. **You ran it: no problems found.** Its pandas probe only checks that an attribute exists, not that calling it fails. |

## What to run next

Isolate the far-field slope finding (H-A). `s4` patches a scratch copy of `src/` (your tree is untouched), then runs `s2` on the unpatched repo and on the patched copy with identical arguments:

```bash
.venv/bin/python review_scripts/s4_farfield_patch_test.py --dem /tmp/lunarscout-phase1-lola-512.tif
.venv/bin/python review_scripts/s4_farfield_patch_test.py --dem /tmp/lunarscout-phase1-lola-512.tif --synthetic wall
```

Add `--check-only` first if you want just the patch and the CPU sanity table (no GPU). If the patch does not apply, or Numba fails to compile the patched kernel, paste the output: I could not compile CUDA in my sandbox, so the first GPU compile happens on your machine.

Other optional runs:

```bash
.venv/bin/python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif
.venv/bin/python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif --synthetic nan
```

The first now prints distance-binned errors and the three most negative samples (to see why production sometimes reports a lower horizon than the reference). The second now tests what it was meant to.

## Full order (for reference)

```bash
cd /e/projects/lunarscout
.venv/bin/python review_scripts/s1_cpu_repros.py
.venv/bin/python review_scripts/s3_wheel_check.py
.venv/bin/python review_scripts/s2_horizon_vs_bruteforce.py --selftest
# GPU (acquire the fixture first if /tmp/lunarscout-phase1-lola-512.tif is missing):
.venv/bin/python scripts/numba_horizon/acquire_phase1_real_fixture.py
.venv/bin/python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif
.venv/bin/python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif --synthetic wall
.venv/bin/python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif --synthetic nan
.venv/bin/python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif --synthetic bowl
```

The acquire script needs `gdal_translate` on PATH and reads `docs/numba-horizon-phase-1-real-terrain-fixtures.json`.

## GPU tool runs (commands only; not scripted)

```bash
# initcheck is the one tool the review found missing from the recorded evidence
compute-sanitizer --tool initcheck  .venv/bin/python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif --pixels 2 --azimuths 8
compute-sanitizer --tool memcheck   .venv/bin/python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif --pixels 2 --azimuths 8
# full real-GPU test suite (the repo's own gate):
LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 .venv/bin/python -m pytest tests/numba_horizon tests/test_public_horizon.py -q
```

Note: `tests/numba_horizon/test_phase4_cuda_mechanics.py` asserts compute capability (12, 0), so it will fail on any other GPU. That is a test-portability issue, not a kernel bug.

## Reading s2's output

* "signed diff (prod - brute)": positive means production reports a higher horizon than the independent march.
* The table by horizon elevation shows whether error grows with angle (review finding on far-field slant range).
* `--synthetic wall`: expect a visible positive error growing with wall elevation if the finding holds.
* `--synthetic nan`: expect large errors at pixels near the NaN node if the finding holds.
* Brute force uses the production pixel-to-CRS convention, so half-pixel registration between mixed-resolution DEMs is not tested.
* If `--selftest` prints GROSS DISAGREEMENT, stop and tell me before trusting the GPU numbers.
