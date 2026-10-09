#!/usr/bin/env python3
"""Diagnose the horizon errors that the far-field patch (s4) did NOT remove.

Three sections, each answers one open question from the review. Needs the GPU and the LOLA fixture
(same prerequisites as s2). Read-only; writes only to temp dirs.

  Section 1  NEAR-FIELD OUTLIERS (real terrain). Production is up to 1 deg LOWER than the reference at
             pixel (9,2) and (39,5), with limiting terrain 50-80 m away. Three hypotheses are tested:
               (a) DEM-edge effect      -> error should grow toward row/col 0 in the sweep
               (b) subpatch-phase effect-> error should repeat with period 8 (row % 8) in BOTH the
                                           edge sweep and the interior control sweep
               (c) production ignores the first N metres of the ray -> production should equal a
                                           brute force that starts at min_range = N
  Section 2  EXACT-ZERO RAYS (synthetic wall). Production reports exactly 0.0000 where the reference
             sees 1.68 deg at pixel (74,64) az 75. Checks whether this is a single ray, a pattern,
             or a fallback value, and counts exact zeros in a whole tile on both DEMs.
  Section 3  SUN-RELEVANT TAILS (real terrain). Rays whose reference horizon lies in [-2, 5] deg:
             how many differ by more than 0.005 / 0.05 deg, and at what limiting distance.

Run from the repo root with the repo venv:
    .venv/bin/python review_scripts/s5_residual_diagnostics.py --dem /tmp/lunarscout-phase1-lola-512.tif
Options: --repo PATH, --only 1,2,3, --out-real DIR / --out-wall DIR (reuse existing horizon dirs).
To test a patched tree, run s4 with --keep and pass its scratch path as --repo (e.g. --repo /tmp/ls_farfield_xxxx).
Paste the full output back.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2_horizon_vs_bruteforce as s2  # noqa: E402  (inserts the default repo src; overridden below)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--repo", type=Path, default=HERE.parent)
ap.add_argument("--dem", default="/tmp/lunarscout-phase1-lola-512.tif")
ap.add_argument("--only", default="1,2,3")
ap.add_argument("--obs-h", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-real", type=Path)
ap.add_argument("--out-wall", type=Path)
args = ap.parse_args()
sections = {int(x) for x in args.only.split(",")}
sys.path.insert(0, str(args.repo.resolve() / "src"))

import lunarscout as ls  # noqa: E402
from lunarscout.products import _load_dem  # noqa: E402

print(f"lunarscout imported from: {ls.__file__}")
print(f"GPU status: {ls.cuda.status()}\n")
AZ = s2.AZ_STEP_DEG
OBS = args.obs_h


def generate(dem_path: Path, out: Path | None):
    out = out or Path(tempfile.mkdtemp(prefix="ls_s5_hz_"))
    if not any(out.glob("*")):
        ls.generate_horizons(out, [dem_path], observer_height_m=OBS, compress=False, overwrite=True)
    return s2.read_production_tile(out, 0, 0, OBS)


def brute(dems, c, r, az_deg, obs_h=0.0, **kw):
    """s2's brute force, with the ray range capped like production (min(1e6, 1.2 x DEM size))."""
    kw.setdefault("max_range_m", RANGE_CAP)
    return s2.brute_horizon_deg(dems, c, r, az_deg, obs_h, **kw)


def pair(tile, dems, c, r, a_idx, **kw):
    """(production, brute, limiting distance m) for one ray in tile (0,0)."""
    prod = float(tile[r, c, a_idx])
    deg, dist = brute(dems, c, r, a_idx * AZ, OBS, **kw)
    return prod, deg, dist


real_dem_path = Path(args.dem)
real = _load_dem(real_dem_path)[0]
real_tile = generate(real_dem_path, args.out_real)
dems_real = [real]
RANGE_CAP = min(1.0e6, 1.2 * max(real.width, real.height) * real.map_resolution_m)
print(f"brute-force range capped at the production cap: {RANGE_CAP:.0f} m")

# ---------------------------------------------------------------------------------------- 1
if 1 in sections:
    print("=" * 78 + "\nSECTION 1: near-field outliers on real terrain\n" + "=" * 78)
    targets = [(9, 2, 0), (9, 2, 7), (9, 2, 11), (39, 5, 180)]
    elev = np.asarray(real.elevation_m, dtype=np.float64)
    print("1a. Production vs brute force restricted to start at min_range (metres). If production matches a")
    print("    large min_range, it is skipping the first part of the ray. Also the cumulative maximum by range.")
    for c, r, a in targets:
        prod = float(real_tile[r, c, a])
        print(f"\n  pixel ({c},{r}) az {a * AZ:.2f} deg: production {prod:.4f}")
        row = [f"min_range={m:>3d}: {brute(dems_real, c, r, a * AZ, OBS, min_range_m=float(m))[0]:8.4f}"
               for m in (0, 20, 40, 60, 80, 100)]
        print("   ", " | ".join(row))
        row = [f"<= {m:>4d} m: {brute(dems_real, c, r, a * AZ, OBS, max_range_m=float(m))[0]:8.4f}"
               for m in (20, 40, 60, 80, 120, 200, 500)]
        print("   ", " | ".join(row))
        r0, c0 = max(r - 3, 0), max(c - 3, 0)
        patch = elev[r0:r + 4, c0:c0 + 7]
        print(f"    DEM relative to the observer node, rows {r0}..{r0 + patch.shape[0] - 1}, cols {c0}..{c0 + patch.shape[1] - 1} (m):")
        for line in patch - elev[r, c]:
            print("     " + " ".join(f"{v:7.2f}" for v in line))

    print("\n1b. Sweeps: mean signed error (production - brute) over near-field rays (limiting distance < 500 m),")
    print("    12 azimuths per pixel. 'edge' rows/cols run toward the DEM border; 'interior' is the control.")
    az_idx = np.arange(0, 1440, 120)

    def sweep(label, pixels):
        print(f"\n  {label}")
        print(f"  {'pixel':>10} {'row%8':>5} {'col%8':>5} {'n_near':>6} {'mean signed':>12} {'max |d|':>9}")
        for c, r in pixels:
            d = []
            for a in az_idx:
                p, b, dist = pair(real_tile, dems_real, c, r, int(a))
                if np.isfinite(p) and np.isfinite(b) and np.isfinite(dist) and dist < 500.0:
                    d.append(p - b)
            if d:
                d = np.array(d)
                print(f"  {f'({c},{r})':>10} {r % 8:5d} {c % 8:5d} {len(d):6d} {d.mean():+12.4f} {np.abs(d).max():9.4f}")
            else:
                print(f"  {f'({c},{r})':>10} {r % 8:5d} {c % 8:5d} {0:6d}")

    sweep("edge sweep, rows 0..31 at col 64 (top border at row 0)", [(64, r) for r in range(0, 32)])
    sweep("interior control, rows 64..95 at col 64", [(64, r) for r in range(64, 96)])
    sweep("edge sweep, cols 0..31 at row 64 (left border at col 0)", [(c, 64) for c in range(0, 32)])
    print("\n  Read: a trend that grows toward row/col 0 = border effect; a pattern repeating every 8 rows/cols")
    print("  that also appears in the interior control = subpatch-phase effect; neither = look at 1a.")

# ---------------------------------------------------------------------------------------- 2
if 2 in sections:
    print("\n" + "=" * 78 + "\nSECTION 2: exact-zero rays\n" + "=" * 78)
    wall_path = Path(tempfile.mkdtemp(prefix="ls_s5_")) / "synthetic_wall.tif"
    s2.synthesize(real_dem_path, "wall", wall_path)
    wall = _load_dem(wall_path)[0]
    dems_wall = [wall]
    wall_tile = generate(wall_path, args.out_wall)

    print("2a. Pixel (74,64), azimuths around 75 deg (index 300): production vs brute")
    print(f"  {'az_deg':>7} {'prod':>9} {'brute':>9} {'diff':>9} {'dist_m':>8}")
    for a in range(284, 317, 2):
        p, b, d = pair(wall_tile, dems_wall, 74, 64, a)
        print(f"  {a * AZ:7.2f} {p:9.4f} {b:9.4f} {p - b:+9.4f} {d:8.0f}")
    print("\n  Same azimuth (75 deg), neighbouring pixels along the row:")
    print(f"  {'pixel':>9} {'prod':>9} {'brute':>9} {'diff':>9} {'dist_m':>8}")
    for c in range(70, 79):
        p, b, d = pair(wall_tile, dems_wall, c, 64, 300)
        print(f"  {f'({c},64)':>9} {p:9.4f} {b:9.4f} {p - b:+9.4f} {d:8.0f}")
    print("\n  Brute force cumulative maximum by range along that ray (where does 1.68 deg come from):")
    print("   ", " | ".join(f"<= {m:>5d} m: {brute(dems_wall, 74, 64, 75.0, OBS, max_range_m=float(m))[0]:7.3f}"
                           for m in (500, 1000, 1400, 1500, 1600, 2000, 4000)))

    print("\n2b. Census of exactly-0.0 production values in tile (0,0), and what the reference says for a sample")
    rng = np.random.default_rng(args.seed)
    for label, tile, dems in (("real terrain", real_tile, dems_real), ("synthetic wall", wall_tile, dems_wall)):
        zeros = np.argwhere(tile == 0.0)
        total = tile.size
        print(f"\n  {label}: exact zeros {len(zeros)} of {total} ({100.0 * len(zeros) / total:.3f}%)")
        if len(zeros) == 0:
            continue
        pick = zeros[rng.choice(len(zeros), size=min(150, len(zeros)), replace=False)]
        rows = []
        for r, c, a in pick:
            deg, dist = brute(dems, int(c), int(r), int(a) * AZ, OBS)
            rows.append((int(c), int(r), int(a) * AZ, deg, dist))
        refs = np.array([x[3] for x in rows])
        fin = np.isfinite(refs)
        print(f"  of {len(rows)} sampled zeros: reference within 0.005 of 0: {int((np.abs(refs[fin]) < 0.005).sum())}, "
              f"reference above 0.05: {int((refs[fin] > 0.05).sum())}, below -0.05: {int((refs[fin] < -0.05).sum())}")
        bad = sorted((x for x in rows if np.isfinite(x[3]) and x[3] > 0.05), key=lambda x: -x[3])[:8]
        for c, r, az, deg, dist in bad:
            print(f"    ({c},{r}) az {az:7.2f}: production 0.0000, reference {deg:.4f} at {dist:.0f} m")

# ---------------------------------------------------------------------------------------- 3
if 3 in sections:
    print("\n" + "=" * 78 + "\nSECTION 3: sun-relevant tails (reference horizon in [-2, 5] deg), real terrain\n" + "=" * 78)
    rng = np.random.default_rng(args.seed + 1)
    pixels = [(int(rng.integers(0, 127)), int(rng.integers(0, 127))) for _ in range(40)]
    az_idx = np.arange(0, 1440, 40)
    rows = []
    for c, r in pixels:
        for a in az_idx:
            p, b, d = pair(real_tile, dems_real, c, r, int(a))
            if np.isfinite(p) and np.isfinite(b) and -2.0 <= b <= 5.0:
                rows.append((c, r, a * AZ, p, b, p - b, d))
    if not rows:
        print("no samples in range")
    else:
        diff = np.array([x[5] for x in rows])
        dist = np.array([x[6] for x in rows])
        print(f"samples: {len(rows)} rays from {len(pixels)} random pixels of tile (0,0)")
        print(f"mean signed {diff.mean():+.4f}  mean |d| {np.abs(diff).mean():.4f}  max |d| {np.abs(diff).max():.4f}")
        for thr in (0.005, 0.0039, 0.05, 0.1):
            print(f"  fraction |d| > {thr:<6}: {100.0 * (np.abs(diff) > thr).mean():6.2f}%")
        print("  by limiting distance:")
        for lo, hi in ((0, 100), (100, 500), (500, 2000), (2000, 5000), (5000, 1e9)):
            m = (dist >= lo) & (dist < hi)
            if m.any():
                print(f"    {lo:>5.0f}-{hi:>9.0f} m: n={int(m.sum()):4d} mean signed {diff[m].mean():+.4f} "
                      f"max |d| {np.abs(diff[m]).max():.4f} frac>0.005 {100.0 * (np.abs(diff[m]) > 0.005).mean():5.1f}%")
        print("  worst 10 (col,row,az: prod / brute / diff / limiting distance m):")
        for c, r, az, p, b, d, dm in sorted(rows, key=lambda x: -abs(x[5]))[:10]:
            print(f"    ({c},{r},{az:.2f}): {p:.4f} / {b:.4f} / {d:+.4f} / {dm:.0f}")
        print(f"\n  Under the repo's uniform-disk model 0.005 deg = {0.005 * 2.0 / (np.pi * 0.25) * 100:.2f}% sunlight error on that ray.")

print("\ndone.")
