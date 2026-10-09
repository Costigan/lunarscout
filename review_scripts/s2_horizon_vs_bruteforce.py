#!/usr/bin/env python3
"""Compare production horizon tiles against an independent float64 brute force.

WHY: the Numba/CUDA horizon code was validated against the earlier C# implementation.
Agreement with C# shows reproduction, not correctness. This script measures the
production output against a dense great-circle ray march that shares no code with the
production kernels except the DEM loader and the stereographic projection definition.

REQUIRES (for the GPU modes): an NVIDIA GPU, `pip install "lunarscout[cuda]"`, and a DEM
(e.g. the LOLA fixture from scripts/numba_horizon/acquire_phase1_real_fixture.py).

USAGE (from the repo root, using the repo's venv)
  # 0. sanity-check this script's own plumbing on CPU (no GPU, no DEM file):
  python review_scripts/s2_horizon_vs_bruteforce.py --selftest

  # 1. real terrain, production kernel on the GPU:
  python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif

  # 2. synthetic terrain built from the fixture's own CRS (tests specific findings):
  python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif --synthetic wall
  python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif --synthetic nan
  python review_scripts/s2_horizon_vs_bruteforce.py --dem /tmp/lunarscout-phase1-lola-512.tif --synthetic bowl

Paste the whole output back. Pass criteria used here: max |difference| <= 0.005 deg (the
repo's accepted tolerance) and <= 0.0039 deg (what 1% sunlight error actually needs).
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

R_MOON = 1737400.0
AZ_STEP_DEG = 0.25  # 1440 bins, index*0.25 deg, 0 = north, clockwise (verified in review)


# ----------------------------------------------------------------------------- brute force
def stereo_fwd(lat, lon, p):
    lat0, lon0 = p.latitude_origin_rad, p.longitude_origin_rad
    k = 2 * p.scale * p.radius_m / (1 + math.sin(lat0) * np.sin(lat) + math.cos(lat0) * np.cos(lat) * np.cos(lon - lon0))
    x = k * np.cos(lat) * np.sin(lon - lon0)
    y = k * (math.cos(lat0) * np.sin(lat) - math.sin(lat0) * np.cos(lat) * np.cos(lon - lon0))
    return x + p.false_easting_m, y + p.false_northing_m


def stereo_inv(x, y, p):
    x, y = x - p.false_easting_m, y - p.false_northing_m
    rho = math.hypot(x, y)
    if rho < 1e-9:
        return p.latitude_origin_rad, p.longitude_origin_rad
    c = 2 * math.atan(rho / (2 * p.scale * p.radius_m))
    lat0 = p.latitude_origin_rad
    lat = math.asin(math.cos(c) * math.sin(lat0) + y * math.sin(c) * math.cos(lat0) / rho)
    lon = p.longitude_origin_rad + math.atan2(
        x * math.sin(c), rho * math.cos(lat0) * math.cos(c) - y * math.sin(lat0) * math.sin(c))
    return lat, lon


def crs_to_pixel_vec(dem, X, Y):
    t = dem.geo_transform
    det = t[1] * t[5] - t[2] * t[4]
    dx, dy = X - t[0], Y - t[3]
    return (t[5] * dx - t[2] * dy) / det, (-t[4] * dx + t[1] * dy) / det


def bilinear_vec(elev, c, r):
    """NaN where outside the grid, or any corner is NaN / <= -20000 (treated as 'no terrain')."""
    h, w = elev.shape
    ok = (c >= 0) & (c <= w - 1) & (r >= 0) & (r <= h - 1)
    c0 = np.clip(np.floor(np.where(ok, c, 0)).astype(np.int64), 0, w - 2)
    r0 = np.clip(np.floor(np.where(ok, r, 0)).astype(np.int64), 0, h - 2)
    tx, ty = c - c0, r - r0
    q = [elev[r0, c0], elev[r0, c0 + 1], elev[r0 + 1, c0], elev[r0 + 1, c0 + 1]]
    q = [x.astype(np.float64) for x in q]
    bad = ~ok
    for x in q:
        bad |= ~np.isfinite(x) | (x <= -20000)
    v = q[0] * (1 - tx) * (1 - ty) + q[1] * tx * (1 - ty) + q[2] * (1 - tx) * ty + q[3] * tx * ty
    v[bad] = np.nan
    return v


def brute_horizon_deg(dems, col, row, az_deg, obs_h=0.0, max_range_m=1.0e6, step_frac=0.25, min_range_m=0.0):
    """Independent float64 horizon: returns (elevation angle deg, distance m to the limiting terrain point)."""
    d0 = dems[0]
    p = d0.projection
    x, y = d0.pixel_to_crs(col, row)  # production's pixel->crs convention (integer index = grid node)
    lat, lon = stereo_inv(x, y, p)
    h0 = float(bilinear_vec(d0.elevation_m, np.array([float(col)]), np.array([float(row)]))[0])
    if not np.isfinite(h0):
        return float("nan"), float("nan")
    ro = R_MOON + h0 + obs_h
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    obs = up * ro
    az = math.radians(az_deg)
    tdir = math.sin(az) * east + math.cos(az) * north
    best = -math.inf
    best_s = float("nan")
    for dem in dems:
        step = dem.map_resolution_m * step_frac
        s = np.concatenate([np.arange(1.0, min(20.0, max_range_m), 1.0), np.arange(20.0, max_range_m, step)])
        s = s[s >= min_range_m]
        if s.size == 0:
            continue
        th = s / R_MOON
        pdir = np.cos(th)[:, None] * up + np.sin(th)[:, None] * tdir
        la = np.arcsin(np.clip(pdir[:, 2], -1, 1))
        lo = np.arctan2(pdir[:, 1], pdir[:, 0])
        X, Y = stereo_fwd(la, lo, dem.projection)
        c, r = crs_to_pixel_vec(dem, X, Y)
        e = bilinear_vec(dem.elevation_m, c, r)
        m = np.isfinite(e)
        if not m.any():
            continue
        P = pdir[m] * (R_MOON + e[m])[:, None] - obs
        hz = np.hypot(P @ east, P @ north)
        ang = np.arctan2(P @ up, hz)
        k = int(np.argmax(ang))
        if float(ang[k]) > best:
            best, best_s = float(ang[k]), float(s[m][k])
    return (math.degrees(best) if best > -math.inf else -math.inf), best_s


# ----------------------------------------------------------------------------- helpers
def summarize(label, prod, ref, pixels, dist=None):
    prod, ref = np.asarray(prod, float), np.asarray(ref, float)
    both = np.isfinite(prod) & np.isfinite(ref)
    d = prod - ref
    print(f"\n=== {label} ===")
    print(f"samples: {len(d)}  comparable: {int(both.sum())}  "
          f"prod finite/ref -inf: {int((np.isfinite(prod) & ~np.isfinite(ref)).sum())}  "
          f"prod -inf/ref finite: {int((~np.isfinite(prod) & np.isfinite(ref)).sum())}")
    if not both.any():
        return False
    dd = d[both]
    ad = np.abs(dd)
    print(f"signed diff (prod - brute) deg: mean {dd.mean():+.5f}  min {dd.min():+.5f}  max {dd.max():+.5f}")
    print(f"|diff| deg: mean {ad.mean():.5f}  p99 {np.percentile(ad, 99):.5f}  max {ad.max():.5f}")
    print(f"fraction > 0.0039 deg: {(ad > 0.0039).mean():.3%}   > 0.005 deg: {(ad > 0.005).mean():.3%}")
    el = ref[both]
    for lo_, hi_ in ((-90, 0), (0, 2), (2, 5), (5, 10), (10, 90)):
        sel = (el >= lo_) & (el < hi_)
        if sel.any():
            print(f"  horizon in [{lo_:>3},{hi_:>3}) deg: n={int(sel.sum()):5d}  mean signed {dd[sel].mean():+.5f}  max |d| {ad[sel].max():.5f}")
    idx = np.flatnonzero(both)
    if dist is not None:
        dist = np.asarray(dist, float)
        print("error by distance to the brute-force limiting terrain point:")
        for lo_, hi_ in ((0, 500), (500, 2000), (2000, 5000), (5000, 1e9)):
            sel = (dist[both] >= lo_) & (dist[both] < hi_)
            if sel.any():
                print(f"  {lo_/1000:>5.1f}-{min(hi_,9e8)/1000:>7.1f} km: n={int(sel.sum()):5d}  mean signed {dd[sel].mean():+.5f}  max |d| {ad[sel].max():.5f}")
        print("most negative 3 (production LOWER than brute: possible missed terrain):")
        for w in np.argsort(dd)[:3]:
            i = idx[w]
            print(f"  {pixels[i]}: prod {prod[i]:.4f} / brute {ref[i]:.4f} / diff {d[i]:+.4f}  limiting point at {dist[i]:.0f} m")
    worst = np.argsort(-ad)[:5]
    print("worst 5 by |diff| (col,row,az_deg: prod / brute / diff" + (" / limiting distance m" if dist is not None else "") + "):")
    for w in worst:
        i = idx[w]
        extra = f" / {dist[i]:.0f}" if dist is not None else ""
        print(f"  {pixels[i]}: {prod[i]:.4f} / {ref[i]:.4f} / {d[i]:+.4f}{extra}")
    return bool(ad.max() <= 0.005)


def sample_points(dem, tile_x, tile_y, n_pixels, n_az, seed, focus=None):
    rng = np.random.default_rng(seed)
    if focus:
        pix = focus
    else:
        pix = [(tile_x + int(rng.integers(0, 128)), tile_y + int(rng.integers(0, 128))) for _ in range(n_pixels)]
    pix = [(min(c, dem.width - 2), min(r, dem.height - 2)) for c, r in pix]
    azi = np.unique(np.concatenate([np.arange(0, 1440, max(1, 1440 // n_az)), rng.integers(0, 1440, n_az // 2)]))
    return pix, azi


def read_production_tile(out_dir, tile_y, tile_x, obs_h):
    from lunarscout._numba_horizon.file_format import HorizonTileStore
    tile = HorizonTileStore(out_dir).read(tile_y, tile_x, obs_h)
    if tile is None:
        raise SystemExit(f"no tile written at y={tile_y} x={tile_x} in {out_dir}")
    return tile  # float32[y, x, azimuth] degrees


def synthesize(dem_path, kind, out_path):
    """Write a synthetic DEM that reuses the fixture's CRS/transform/encoding."""
    import rasterio
    with rasterio.open(dem_path) as src:
        prof = src.profile.copy()
        raw = src.read(1).astype(np.float64)
        scales, offsets = src.scales, src.offsets
        h, w = raw.shape
        finite_min = np.nanmin(raw)
        shift = R_MOON if (finite_min > 0.5 * R_MOON and not (offsets and offsets[0])) else 0.0
        res = abs(src.transform.a)
        z = np.zeros((h, w))
        cc = np.arange(w)[None, :] * np.ones((h, 1))
        rr = np.arange(h)[:, None] * np.ones((1, w))
        if kind == "wall":      # 200 m step wall ~1 km east of column 64 (grid-east); far-field slope test
            z[:, int(64 + 1000 / res):] = 200.0
        elif kind == "nan":     # flat; 100 m plateau on rows < 60 (north of the observers); one NaN node at (148,148)
            z[:60, :] = 100.0   # observers sit in tile (128,128), rows 128+, so the plateau is a real obstacle
            z[148, 148] = np.nan
        elif kind == "bowl":    # smooth crater bowl, 80 m deep; exercises curvature/float32 paths
            d = np.hypot(cc - w / 2, rr - h / 2) * res
            z = 80.0 * (np.exp(-(d / (0.3 * w * res)) ** 2) - 1.0) + 0.0
        else:
            raise SystemExit(f"unknown synthetic kind {kind}")
        prof.update(dtype="float32")
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write((z + shift).astype(np.float32), 1)
            if scales:
                dst.scales = scales
            if offsets:
                dst.offsets = offsets
    return out_path


# ----------------------------------------------------------------------------- modes
def run_gpu(args):
    import lunarscout as ls
    from lunarscout.products import _load_dem

    dem_path = Path(args.dem)
    if args.synthetic:
        syn = Path(tempfile.mkdtemp(prefix="ls_s2_")) / f"synthetic_{args.synthetic}.tif"
        synthesize(dem_path, args.synthetic, syn)
        dem_path = syn
        print(f"synthetic DEM written: {syn}")
    dem = _load_dem(dem_path)[0]
    dems = [dem]
    out = Path(args.out or tempfile.mkdtemp(prefix="ls_s2_hz_"))
    print(f"GPU status: {ls.cuda.status()}")
    ls.generate_horizons(out, [dem_path], observer_height_m=args.obs_h, compress=False, overwrite=True)
    tile = read_production_tile(out, args.tile_y, args.tile_x, args.obs_h)
    focus = None
    if args.synthetic == "wall":
        focus = [(args.tile_x + 64 + k, args.tile_y + 64 + j) for k in (0, 10, 25) for j in (0, 20)]
    elif args.synthetic == "nan":
        focus = [(150, 150), (146, 146), (151, 150), (149, 147), (152, 152), (140, 140), (160, 160)]
    pix, azi = sample_points(dem, args.tile_x, args.tile_y, args.pixels, args.azimuths, args.seed, focus)
    prod, ref, lab, dist = [], [], [], []
    for c, r in pix:
        for a in azi:
            prod.append(float(tile[r - args.tile_y, c - args.tile_x, a]))
            deg, dm = brute_horizon_deg(dems, c, r, float(a) * AZ_STEP_DEG, args.obs_h, args.max_range_m)
            ref.append(deg)
            dist.append(dm)
            lab.append((c, r, float(a) * AZ_STEP_DEG))
    print(f"DEM: {dem.width}x{dem.height} px, resolution {dem.map_resolution_m:.2f} m, "
          f"production ray range cap = min(1e6, 1.2 x DEM size) = {min(1e6, 1.2 * max(dem.width, dem.height) * dem.map_resolution_m):.0f} m")
    ok = summarize(f"production GPU vs brute force ({args.synthetic or 'real terrain'})", prod, ref, lab, dist)
    print("\nRESULT:", "within 0.005 deg" if ok else "EXCEEDS 0.005 deg (see worst samples)")
    print("note: brute force uses the production pixel->CRS convention, so half-pixel registration "
          "(review finding on mixed-resolution DEMs) is not tested here.")


def run_selftest(args):
    """CPU-only: check this script's brute force against the repo's CPU hierarchy emulation."""
    from lunarscout._numba_horizon.geometry import (
        DemGrid, GridConvergenceInput, ProjectionParameters, build_subpatch_segments_numba)
    from lunarscout._numba_horizon.hierarchy import traverse_hierarchy
    from lunarscout._numba_horizon.kernel_math import sample_bilinear
    from lunarscout._numba_horizon.pyramid import build_max_pyramid
    from lunarscout._numba_horizon.subpatch import interpolate_pixel_segment

    rng = np.random.default_rng(1)
    n, res = 160, 20.0
    from scipy.ndimage import gaussian_filter
    z = gaussian_filter(rng.normal(0, 1, (n, n)), 4) * 60
    gt = np.array([-n * res / 2, res, 0, n * res / 2, 0, -res], dtype=np.float64)
    proj = ProjectionParameters(R_MOON, math.radians(-89.0), 0.0, 1.0, 0.0, 0.0)
    dem = DemGrid(np.ascontiguousarray(z, dtype=np.float32), gt, proj)
    tile_c = tile_r = 64
    segs, _, _ = build_subpatch_segments_numba(
        [dem], tile_column=tile_c, tile_row=tile_r, tile_width=16, azimuth_count=1440,
        maximum_distance_m=1e6, observer_elevation_m=0.0, subpatch_size=8,
        grid_convergence=GridConvergenceInput(0, 0, 0))
    py = build_max_pyramid(dem)
    prod, ref, lab = [], [], []
    for c, r in ((70, 70), (75, 68), (66, 77)):
        for a in range(0, 1440, 60):
            seg = interpolate_pixel_segment(
                segs, py, py, tile_column=tile_c, tile_row=tile_r, tile_width=16, subpatch_size=8,
                pixel_column=c - tile_c, pixel_row=r - tile_r, azimuth=a, pass_index=0)
            oz = float(sample_bilinear(dem.elevation_m, float(c), float(r)))
            tr = traverse_hierarchy(seg, py, observer_z_m=oz, radius_m=R_MOON,
                                    map_resolution_m=dem.map_resolution_m, pass_index=0, initial_slope=-np.inf)
            s = float(tr.maximum_slope)
            prod.append(math.degrees(math.atan(s)) if np.isfinite(s) else -np.inf)
            ref.append(brute_horizon_deg([dem], c, r, a * AZ_STEP_DEG, 0.0, 20000.0)[0])
            lab.append((c, r, a * AZ_STEP_DEG))
    summarize("SELFTEST: repo CPU hierarchy emulation vs this script's brute force", prod, ref, lab)
    d = np.abs(np.asarray(prod) - np.asarray(ref))
    gross = float(np.nanmax(d))
    # A wrong azimuth/pixel/projection convention shows up as errors of degrees, not hundredths.
    # Residuals of a few 0.01 deg are the algorithm differences described in the review
    # (float32 radial arithmetic, adaptive stepping, hierarchy culling), not plumbing errors.
    print(f"\nSELFTEST max |diff| = {gross:.4f} deg ->",
          "conventions agree (residual = algorithm differences, not plumbing)" if gross < 0.1
          else "GROSS DISAGREEMENT: conventions differ; do not trust GPU comparison, tell the reviewer")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO, help="lunarscout repo root (default: parent of this folder)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dem", default="/tmp/lunarscout-phase1-lola-512.tif")
    ap.add_argument("--synthetic", choices=["wall", "nan", "bowl"])
    ap.add_argument("--out", help="horizon output directory (default: temp dir)")
    ap.add_argument("--tile-x", type=int, default=0)
    ap.add_argument("--tile-y", type=int, default=0)
    ap.add_argument("--pixels", type=int, default=12)
    ap.add_argument("--azimuths", type=int, default=48)
    ap.add_argument("--obs-h", type=float, default=0.0)
    ap.add_argument("--max-range-m", type=float, default=1.0e6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.synthetic == "nan" and args.tile_x == 0 and args.tile_y == 0:
        args.tile_x = args.tile_y = 128  # NaN scenario lives in tile (128,128)
    sys.path.insert(0, str(args.repo.resolve() / "src"))
    (run_selftest if args.selftest else run_gpu)(args)


if __name__ == "__main__":
    main()
