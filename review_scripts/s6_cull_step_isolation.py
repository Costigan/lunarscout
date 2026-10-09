#!/usr/bin/env python3
"""Isolate H-B (hierarchy cull bound) from M-A (adaptive level-0 step) as causes of the near-field
misses that remain after the H-A far-field patch (see s4, s5).

Your working tree is never modified. The script builds scratch copies of <repo>/src, one per variant,
each a different edit of _numba_horizon/cuda_backend.py (every site asserted by count):

  baseline        unpatched
  HA              s4's far-field slope patch (imported from s4_farfield_patch_test.py, not duplicated)
  HA+nocull       HA, and the cull test `if possible <= current:` never fires (both hierarchy kernels)
  HA+dense        HA, and margin_step = 0, so level 0 advances one pixel (a quarter pixel within 0.5 km)
  HA+nocull+dense both
Other variants are any '+'-joined combination of the edits HA, nocull, dense and
  clamp           production kernel stops each level-0 step at the cell exit, as the traced kernel and the
                  CPU oracle do (removed for speed in d543e6e)
  fine            quarter-pixel level-0 steps at every range, with no 0.8 / 0.5 pixel step floor
  diagnear        (fix/hierarchy-cull-bound tree only) a block's near distance uses its diagonal

For each variant it generates horizons for the LOLA fixture and the synthetic wall in a child process,
then compares the same rays against s2's float64 brute force (computed once, shared by all variants):
  - the s2 real-terrain sample (seed 0, 12 px x 48 az, range 1e6 m), so 'baseline' and 'HA' must
    reproduce s4's numbers (mean |d| 0.300 and 0.033 deg): a check that the harness is the same;
  - s5's sun-relevant sample (reference horizon in [-2, 5] deg, range capped like production);
  - the wall sample from s2 (--synthetic wall);
  - the specific rays named in the review.
Reading: if HA+nocull removes the near-field negatives, H-B causes them; if HA+dense does, M-A does.

Run from the repo root with the repo venv (GPU and the LOLA fixture, as for s2):
    .venv/bin/python review_scripts/s6_cull_step_isolation.py --dem /tmp/lunarscout-phase1-lola-512.tif
Options: --repo PATH, --variants baseline,HA,HA+clamp,..., --keep. Experiment only; no variant is a proposed fix.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND = Path("src/lunarscout/_numba_horizon/cuda_backend.py")
EDITS = ("HA", "nocull", "dense", "clamp", "fine", "diagnear")
DEFAULT_VARIANTS = "baseline,HA,HA+nocull,HA+dense,HA+nocull+dense"
# (dataset, col, row, azimuth index): rays named in the review
NAMED = [("real", 9, 2, 0), ("real", 39, 5, 180), ("real", 11, 92, 1360), ("real", 11, 92, 1320),
         ("real", 10, 3, 400), ("wall", 74, 64, 300), ("wall", 89, 64, 600), ("wall", 89, 84, 443),
         ("wall", 89, 64, 510)]


def load_s4_patch():
    """Pull DEVICE_FN and patch() out of s4 without running s4's top-level script."""
    tree = ast.parse((HERE / "s4_farfield_patch_test.py").read_text())
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.FunctionDef) and n.name == "patch")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "DEVICE_FN" for t in n.targets))]
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "s4_farfield_patch_test.py", "exec"), ns)
    return ns["patch"]


def sub(text, label, pattern, repl, expected):
    text, n = re.subn(pattern, repl, text)
    print(f"    {label:10s} replaced {n} (expected {expected})")
    if n != expected:
        sys.exit(f"PATCH {label} DID NOT APPLY CLEANLY; the source differs from what this script expects.")
    return text


def build(repo: Path, work: Path, name: str, edits) -> Path:
    root = work / name
    shutil.copytree(repo / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    text = (repo / BACKEND).read_text()
    print(f"  {name}")
    if "HA" in edits:
        text = load_s4_patch()(text)
    if "nocull" in edits:
        text = sub(text, "nocull", r"if possible <= current:", "if False and possible <= current:", 2)
    if "dense" in edits:
        text = sub(text, "dense", r"float32\(float32\(margin \* true_m\) \* float32\(1\.732\)\)", "float32(0.0)", 2)
    if "clamp" in edits:  # production kernel only: stop each level-0 step at the cell exit, as the traced kernel does
        text = sub(text, "clamp", r"(\n( +)advance = float32\(max\(advance, floor\)\)\n)(\s+# The production C# kernel)",
                   lambda m: m.group(1) + f"{m.group(2)}advance = float32(min(advance, float32(distance_to_exit + "
                   f"float32(0.000001)) if distance_to_exit > float32(0.0) else fallback))\n" + m.group(3), 1)
    if "diagnear" in edits:  # fixed tree only: near distance of a block = true_m - diagonal, not - side
        text = sub(text, "diagnear", r"true_near = float32\(max\(float32\(true_m - block_m\), float32\(1\.0\)\)\)",
                   "true_near = float32(max(float32(true_m - float32(block_m * float32(1.4142136))), float32(1.0)))", 1)
    if "fine" in edits:  # quarter-pixel level-0 steps at every range, no 0.8 / 0.5 pixel floor
        text = sub(text, "fine-1", r"if s < float32\(0\.5\):\n( +)advance = float32\(advance \* float32\(0\.25\)\)",
                   r"if True:\n\1advance = float32(advance * float32(0.25))", 2)
        text = sub(text, "fine-2", r"advance = float32\(max\(advance, floor\)\)", "advance = float32(max(advance, float32(1e-6)))", 2)
    compile(text, str(root / BACKEND), "exec")
    (root / BACKEND).write_text(text)
    return root


# --------------------------------------------------------------------------- child: generate + read
def child(a):
    sys.path.insert(0, str(Path(a.child_repo) / "src"))
    import lunarscout as ls
    sys.path.insert(0, str(HERE))
    import s2_horizon_vs_bruteforce as s2
    assert str(Path(a.child_repo) / "src") in ls.__file__, ls.__file__
    keys = json.loads(Path(a.keys).read_text())
    result = {"imported": ls.__file__}
    for ds, dem in (("real", a.dem), ("wall", a.wall)):
        out = Path(tempfile.mkdtemp(prefix=f"ls_s6_{ds}_"))
        t0 = time.perf_counter()
        ls.generate_horizons(out, [dem], observer_height_m=0.0, compress=False, overwrite=True)
        result[f"seconds_{ds}"] = time.perf_counter() - t0
        tile = s2.read_production_tile(out, 0, 0, 0.0)
        result[ds] = [float(tile[r, c, az]) for c, r, az in keys[ds]]
        shutil.rmtree(out, ignore_errors=True)
    Path(a.result).write_text(json.dumps(result))


# --------------------------------------------------------------------------- parent
def samples(s2, real, wall):
    """Ray sets: (label, dataset, [(col,row,az_idx)], brute range)."""
    cap = min(1.0e6, 1.2 * max(real.width, real.height) * real.map_resolution_m)
    pix, azi = s2.sample_points(real, 0, 0, 12, 48, 0)
    s2_real = [(c, r, int(az)) for c, r in pix for az in azi]
    rng = np.random.default_rng(1)  # s5 section 3: seed + 1
    pix5 = [(int(rng.integers(0, 127)), int(rng.integers(0, 127))) for _ in range(40)]
    sun = [(c, r, int(az)) for c, r in pix5 for az in np.arange(0, 1440, 40)]
    focus = [(64 + k, 64 + j) for k in (0, 10, 25) for j in (0, 20)]
    wpix, wazi = s2.sample_points(wall, 0, 0, 12, 48, 0, focus)
    s2_wall = [(c, r, int(az)) for c, r in wpix for az in wazi]
    named = {ds: [(c, r, az) for d, c, r, az in NAMED if d == ds] for ds in ("real", "wall")}
    return [("s2 real", "real", s2_real, 1.0e6), ("sun", "real", sun, cap), ("named", "real", named["real"], cap),
            ("s2 wall", "wall", s2_wall, 1.0e6), ("named", "wall", named["wall"], cap)]


def stats(d, dist):
    ok = np.isfinite(d)
    d, dist = d[ok], dist[ok]
    near, mid = dist < 500.0, (dist >= 500.0) & (dist < 2000.0)
    return {"n": int(d.size), "mean_signed": d.mean(), "mean_abs": np.abs(d).mean(), "max_abs": np.abs(d).max(),
            "f005": (np.abs(d) > 0.005).mean(), "f05": (np.abs(d) > 0.05).mean(),
            "neg05": int((d < -0.05).sum()), "pos05": int((d > 0.05).sum()),
            "near_signed": d[near].mean() if near.any() else np.nan,
            "near_max": np.abs(d[near]).max() if near.any() else np.nan,
            "mid_signed": d[mid].mean() if mid.any() else np.nan,
            "mid_max": np.abs(d[mid]).max() if mid.any() else np.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE.parent)
    ap.add_argument("--dem", default="/tmp/lunarscout-phase1-lola-512.tif")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help=f"comma-separated; each is 'baseline' or edits joined by '+', from {EDITS}")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--timeout", type=float, default=1200.0, help="seconds allowed per variant (default 1200)")
    ap.add_argument("--child-repo"), ap.add_argument("--wall"), ap.add_argument("--keys"), ap.add_argument("--result")
    a = ap.parse_args()
    if a.child_repo:
        return child(a)

    repo = a.repo.resolve()
    names = a.variants.split(",")
    edits = {n: () if n == "baseline" else tuple(n.split("+")) for n in names}
    unknown = [n for n, e in edits.items() if any(x not in EDITS for x in e)]
    if unknown:
        sys.exit(f"unknown variants {unknown}; use 'baseline' or edits joined by '+' from {EDITS}")
    work = Path(tempfile.mkdtemp(prefix="ls_s6_"))
    print(f"repo   : {repo}\nscratch: {work}\nbuilding variants:")
    roots = {n: build(repo, work, n, edits[n]) for n in names}

    sys.path.insert(0, str(HERE))
    import s2_horizon_vs_bruteforce as s2  # inserts <repo>/src; used here only for the DEM loader
    from lunarscout.products import _load_dem
    wall_path = work / "synthetic_wall.tif"
    s2.synthesize(Path(a.dem), "wall", wall_path)
    dems = {"real": _load_dem(Path(a.dem))[0], "wall": _load_dem(wall_path)[0]}
    sets = samples(s2, dems["real"], dems["wall"])

    print("\nbrute force reference (float64, CPU) ...", flush=True)
    t0 = time.perf_counter()
    keys = {"real": [], "wall": []}
    with_refs = []
    for label, ds, rays, rng in sets:
        refs = np.array([s2.brute_horizon_deg([dems[ds]], c, r, az * s2.AZ_STEP_DEG, 0.0, rng) for c, r, az in rays])
        with_refs.append((label, ds, rays, rng, refs, len(keys[ds])))
        keys[ds].extend(rays)
    sets = with_refs
    print(f"  {sum(len(s[2]) for s in sets)} rays in {time.perf_counter() - t0:.0f} s")
    (work / "keys.json").write_text(json.dumps(keys))

    results = {}
    for n in names:
        print(f"\nvariant {n}: generating horizons on the GPU ...", flush=True)
        res = work / f"{n}.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child-repo", str(roots[n]), "--dem", a.dem,
               "--wall", str(wall_path), "--keys", str(work / "keys.json"), "--result", str(res)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            sys.exit(f"variant {n} exceeded --timeout {a.timeout:.0f} s; stopped (child killed)")
        if p.returncode:
            print(p.stdout[-3000:], p.stderr[-3000:])
            sys.exit(f"variant {n} failed")
        results[n] = json.loads(res.read_text())
        print(f"  imported {results[n]['imported']}\n  real {results[n]['seconds_real']:.1f} s, "
              f"wall {results[n]['seconds_wall']:.1f} s")

    def diffs(n, s):
        label, ds, rays, rng, refs, off = s
        prod = np.array(results[n][ds][off:off + len(rays)])
        return prod - refs[:, 0], refs[:, 1], prod, refs[:, 0]

    print("\n" + "=" * 100 + "\nRESULTS (production - brute force, degrees)\n" + "=" * 100)
    for s in sets:
        if s[0] == "named":
            continue
        print(f"\n{s[0]} ({s[1]}){' : rays with reference horizon in [-2, 5] deg' if s[0] == 'sun' else ''}")
        print(f"  {'variant':16s} {'n':>4s} {'mean sgn':>9s} {'mean |d|':>9s} {'max |d|':>8s} {'>0.005':>7s} "
              f"{'>0.05':>6s} {'#<-0.05':>7s} {'#>+0.05':>7s} {'<0.5km sgn':>10s} {'<0.5km max':>10s} "
              f"{'0.5-2km sgn':>11s} {'0.5-2km max':>11s}")
        for n in names:
            d, dist, prod, ref = diffs(n, s)
            if s[0] == "sun":
                m = np.isfinite(ref) & (ref >= -2.0) & (ref <= 5.0)
                d, dist = d[m], dist[m]
            st = stats(d, dist)
            print(f"  {n:16s} {st['n']:4d} {st['mean_signed']:+9.4f} {st['mean_abs']:9.4f} {st['max_abs']:8.4f} "
                  f"{100 * st['f005']:6.1f}% {100 * st['f05']:5.1f}% {st['neg05']:7d} {st['pos05']:7d} "
                  f"{st['near_signed']:+10.4f} {st['near_max']:10.4f} {st['mid_signed']:+11.4f} {st['mid_max']:11.4f}")

    print("\nnamed rays: production per variant (reference, limiting distance)")
    for s in sets:
        if s[0] != "named":
            continue
        for i, (c, r, az) in enumerate(s[2]):
            ref, dist = s[4][i]
            row = "  ".join(f"{n}={diffs(n, s)[2][i]:+.4f}" for n in names)
            print(f"  {s[1]} ({c},{r}) az {az * s2.AZ_STEP_DEG:6.2f}: ref {ref:+.4f} at {dist:.0f} m | {row}")

    print("\nGPU generation time, real + wall (s): "
          + ", ".join(f"{n} {results[n]['seconds_real'] + results[n]['seconds_wall']:.1f}" for n in names))
    print("(first variant includes Numba compile and any one-time pyramid build; compare the later ones)")
    if a.keep:
        print(f"kept {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
