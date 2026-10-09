#!/usr/bin/env python3
"""Isolate review finding H-A (far-field slope) by patching a SCRATCH COPY of the repo.

Your working tree is never modified. The script:
  1. copies <repo>/src to a temp dir (skipping __pycache__, so no stale Numba caches),
  2. rewrites the far-field slope arithmetic in the temp copy of
     _numba_horizon/cuda_backend.py (all six sites, asserted by count),
  3. prints a CPU sanity check of old vs new formula against exact geometry,
  4. runs s2_horizon_vs_bruteforce.py on the UNPATCHED repo, then on the PATCHED copy,
     with identical arguments (s2 is seeded, so both runs use the same pixels/azimuths).

What the patch changes (and what it does not)
  The production code treats `true_m` as the straight-line observer-to-target distance.
  `true_m` is actually the observer-to-GROUND-SPHERE-point distance (geometry.fit_planar_to_chord
  measures to a point on the sphere of radius reference+terrain at the subpatch centre, not to the
  elevated target). With the target's height included, the correct local coordinates are
      q  = 1 - cos(theta) = c^2 / (2 * Ro * Rc)        (Rc ~ Ro; observer height is metres)
      z  = (Rp - Ro) - Rp * q
      x  = Rp * sqrt(q * (2 - q))
      slope = z / x
  with c = true_m, Ro = radius + observer_z, Rp = radius + target height.
  The patched formula is evaluated in float64 inside the kernel (cast back to float32), so a
  float32 cancellation problem cannot be confused with the formula error.
  NOT changed: the near-field (<0.5 km) formula, the hierarchy cull logic (H-B), NaN handling
  (H-C), stepping, and the CPU emulation in hierarchy.py / fixed_step.py.
  This is an EXPERIMENT to measure how much of the error H-A explains, not a proposed fix.

Usage (repo root, repo venv, GPU):
    .venv/bin/python review_scripts/s4_farfield_patch_test.py --dem /tmp/lunarscout-phase1-lola-512.tif
    .venv/bin/python review_scripts/s4_farfield_patch_test.py --dem /tmp/lunarscout-phase1-lola-512.tif --synthetic wall

Any argument this script does not recognise is passed to s2.
Options: --repo PATH, --patched-only, --keep, --check-only (patch + CPU sanity check, no GPU run).
"""
from __future__ import annotations

import argparse
import difflib
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--repo", type=Path, default=HERE.parent)
ap.add_argument("--patched-only", action="store_true", help="skip the unpatched baseline run")
ap.add_argument("--keep", action="store_true", help="keep the temp copy and diff")
ap.add_argument("--check-only", action="store_true", help="patch and sanity-check, do not run s2")
args, s2_args = ap.parse_known_args()
repo = args.repo.resolve()
backend_rel = Path("src/lunarscout/_numba_horizon/cuda_backend.py")
if not (repo / backend_rel).exists():
    sys.exit(f"cannot find {backend_rel} under {repo}")

# --------------------------------------------------------------------------- the patch
DEVICE_FN = '''    @cuda.jit(device=True)
    def far_slope_patched(chord_m, observer_z, radius, height):
        # REVIEW EXPERIMENT (H-A): chord_m is observer -> ground-sphere point; add target height.
        ro = float64(radius) + float64(observer_z)
        rp = float64(radius) + float64(height)
        c = float64(chord_m)
        q = c * c / (2.0 * ro * ro)
        z = (rp - ro) - rp * q
        x2 = q * (2.0 - q)
        x = rp * math.sqrt(x2) if x2 > 0.0 else 1e-6
        return float32(z / x)

'''


def patch(source: str) -> str:
    counts = {}

    def sub(label, pattern, repl, text, expected):
        text, n = re.subn(pattern, repl, text, flags=re.DOTALL)
        counts[label] = (n, expected)
        return text

    text = source
    text = sub("import float64", r"from numba import float32, int32\n",
               "from numba import float32, float64, int32\n", text, 1)
    text = sub("device function", r"(    @cuda\.jit\(device=True\)\n    def valid_elevation\(value\):)",
               lambda m: DEVICE_FN + m.group(1), text, 1)

    # fixed_step_kernel and adaptive_kernel (float64-ish arithmetic, indexed arrays)
    text = sub(
        "fixed/adaptive far-field",
        r"observer_radius = radius\[index\] \+ observer_z\[index\]\n\s+distance_squared = true_m \* true_m\n.*?"
        r"slope = local_z / local_x if local_x != 0 else -1e30",
        "slope = far_slope_patched(true_m, observer_z[index], radius[index], height)", text, 2)

    # hierarchy kernels: cull bound ("possible") and level-0 slope. Two variants each:
    # hierarchy_kernel uses radius[index]/observer_z[index]; hierarchy_maximum uses scalars.
    def possible(m):
        block = m.group(0)
        obs, rad = ("observer_z[index]", "radius[index]") if "observer_z[index]" in block else ("observer_z", "radius")
        bound = "bound_m" if "bound_m * bound_m" in block else "true_near"
        return f"possible = far_slope_patched({bound}, {obs}, {rad}, maximum_height)"

    def level0(m):
        block = m.group(0)
        obs, rad = ("observer_z[index]", "radius[index]") if "radius[index]" in block else ("observer_z", "radius")
        return f"slope = far_slope_patched(true_m, {obs}, {rad}, height)"

    text = sub("cull bound", r"squared = float32\((?:true_near \* true_near|bound_m \* bound_m)\).*?possible = float32\(local_z / local_x\)",
               possible, text, 2)
    text = sub("level-0 slope", r"squared = float32\(true_m \* true_m\).*?slope = float32\(local_z / local_x\)",
               level0, text, 2)
    bad = {k: v for k, v in counts.items() if v[0] != v[1]}
    for k, (n, e) in counts.items():
        print(f"  patch site {k:28s} replaced {n} (expected {e})")
    if bad:
        sys.exit(f"PATCH DID NOT APPLY CLEANLY ({bad}). The source differs from what this script expects; "
                 "paste this output back to the reviewer.")
    return text


work = Path(tempfile.mkdtemp(prefix="ls_farfield_"))
print(f"repo   : {repo}\nscratch: {work}")
shutil.copytree(repo / "src", work / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
src_path = work / backend_rel
original = (repo / backend_rel).read_text()
patched = patch(original)
src_path.write_text(patched)
diff = "".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True), "a/cuda_backend.py", "b/cuda_backend.py"))
(work / "farfield.diff").write_text(diff)
compile(patched, str(src_path), "exec")  # syntax check
print(f"  diff: {diff.count(chr(10))} lines -> {work / 'farfield.diff'}\n")

# --------------------------------------------------------------------------- CPU sanity check
R = 1737400.0


def exact_elevation_deg(ground_range_m, h, zo=0.0):
    theta = ground_range_m / R
    ro, rp = R + zo, R + h
    return math.degrees(math.atan2(rp * math.cos(theta) - ro, rp * math.sin(theta)))


def old_formula_deg(c, h, zo=0.0):
    ro, rp = R + zo, R + h
    z = ((rp - ro) * (rp + ro) - c * c) / (2 * ro)
    x2 = c * c - z * z
    return math.degrees(math.atan2(z, math.sqrt(x2) if x2 > 0 else 1e-6))


def new_formula_deg(c, h, zo=0.0):
    ro, rp = R + zo, R + h
    q = c * c / (2 * ro * ro)
    z = (rp - ro) - rp * q
    return math.degrees(math.atan2(z, rp * math.sqrt(q * (2 - q))))


print("CPU sanity check, observer at ground level, target = step wall; angles in degrees")
print(f"{'range_km':>9} {'wall_h_m':>9} {'exact':>9} {'old':>9} {'new':>9} {'old-exact':>10} {'new-exact':>10}")
for rng_m, h in ((5000, 440), (5000, 880), (5000, 1820), (10000, 3640), (2000, 100)):
    c = 2 * R * math.sin(rng_m / (2 * R))  # observer -> ground point below target
    e, o, n = exact_elevation_deg(rng_m, h), old_formula_deg(c, h), new_formula_deg(c, h)
    print(f"{rng_m/1000:9.1f} {h:9.0f} {e:9.4f} {o:9.4f} {n:9.4f} {o-e:+10.4f} {n-e:+10.6f}")
print("(expect: old-exact grows with elevation, about +1.3 deg near 20 deg; new-exact ~ 0)\n")

if args.check_only:
    print(f"--check-only: stopping. Patched copy at {work}")
    sys.exit(0)

# --------------------------------------------------------------------------- run s2 both ways
s2 = HERE / "s2_horizon_vs_bruteforce.py"


def import_target(root: Path) -> str:
    r = subprocess.run([sys.executable, "-c",
                        f"import sys; sys.path.insert(0, r'{root / 'src'}'); import lunarscout; print(lunarscout.__file__)"],
                       capture_output=True, text=True)
    return r.stdout.strip() or r.stderr.strip()[-300:]


rc = 0
runs = ([] if args.patched_only else [("BASELINE (unpatched repo)", repo)]) + [("PATCHED far-field (scratch copy)", work)]
for label, root in runs:
    where = import_target(root)
    print("=" * 78 + f"\n{label}\n  lunarscout imported from: {where}\n" + "=" * 78, flush=True)
    if str(root / "src") not in where:
        print("  WARNING: not importing from the expected tree (another install shadows it?); results unreliable.")
    rc |= subprocess.run([sys.executable, str(s2), "--repo", str(root), *s2_args]).returncode

if args.keep:
    print(f"\nkept {work}")
else:
    shutil.rmtree(work, ignore_errors=True)
sys.exit(rc)
