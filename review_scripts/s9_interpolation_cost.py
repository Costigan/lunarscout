#!/usr/bin/env python3
"""GPU cost of bilinear vs cubic terrain lookups (stand-alone kernels; no production code).

Times 26.2 M height lookups (65,536 threads x 400 samples) with bilinear (4 taps), a prefiltered
cubic B-spline (16 taps) and Catmull-Rom (16 taps, no prefilter), for ray-coherent access (each
thread walks a straight line, like production) and scattered access, on the 512 px fixture (fits in
cache) and an 8,192 px tiling of it (256 MB). Reports the median of 7 runs after a warm-up.
Needs a CUDA GPU; runs for well under a minute.

    .venv/bin/python review_scripts/s9_interpolation_cost.py --dem /tmp/lunarscout-phase1-lola-512.tif
Used for docs/numba-horizon-terrain-interpolation.md (2026-10-09). The "scattered" row on the 512 px
grid is not meaningful: the jumps clamp to the grid edge so often that reads stay in cache.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import cuda, float32
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunarscout.products import _load_dem  # noqa: E402

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--dem", default="/tmp/lunarscout-phase1-lola-512.tif")
_args = _parser.parse_args()

dem = _load_dem(_args.dem)[0].elevation_m.astype(np.float32)
H, W = dem.shape
t0 = time.perf_counter()
coef = ndimage.spline_filter(dem.astype(np.float64), order=3, mode="nearest").astype(np.float32)
prefilter_s = time.perf_counter() - t0

@cuda.jit(device=True, inline=True)
def clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

@cuda.jit(device=True)
def bilinear(e, x, y):
    x0 = int(math.floor(x)); y0 = int(math.floor(y))
    tx = float32(x - x0); ty = float32(y - y0)
    x0 = clampi(x0, 0, e.shape[1] - 2); y0 = clampi(y0, 0, e.shape[0] - 2)
    a = e[y0, x0]; b = e[y0, x0 + 1]; c = e[y0 + 1, x0]; d = e[y0 + 1, x0 + 1]
    top = a + tx * (b - a); bot = c + tx * (d - c)
    return top + ty * (bot - top)

@cuda.jit(device=True)
def cubic16(e, x, y, spline):
    x0 = int(math.floor(x)); y0 = int(math.floor(y))
    tx = float32(x - x0); ty = float32(y - y0)
    tx2 = tx * tx; tx3 = tx2 * tx; ty2 = ty * ty; ty3 = ty2 * ty
    if spline:   # cubic B-spline weights on prefiltered coefficients
        s6 = float32(1.0 / 6.0)
        wx0 = (float32(1.0) - tx) * (float32(1.0) - tx) * (float32(1.0) - tx) * s6
        wx1 = (float32(3.0) * tx3 - float32(6.0) * tx2 + float32(4.0)) * s6
        wx2 = (float32(-3.0) * tx3 + float32(3.0) * tx2 + float32(3.0) * tx + float32(1.0)) * s6
        wx3 = tx3 * s6
        wy0 = (float32(1.0) - ty) * (float32(1.0) - ty) * (float32(1.0) - ty) * s6
        wy1 = (float32(3.0) * ty3 - float32(6.0) * ty2 + float32(4.0)) * s6
        wy2 = (float32(-3.0) * ty3 + float32(3.0) * ty2 + float32(3.0) * ty + float32(1.0)) * s6
        wy3 = ty3 * s6
    else:        # Catmull-Rom
        h = float32(0.5)
        wx0 = h * (-tx3 + float32(2.0) * tx2 - tx); wx1 = h * (float32(3.0) * tx3 - float32(5.0) * tx2 + float32(2.0))
        wx2 = h * (float32(-3.0) * tx3 + float32(4.0) * tx2 + tx); wx3 = h * (tx3 - tx2)
        wy0 = h * (-ty3 + float32(2.0) * ty2 - ty); wy1 = h * (float32(3.0) * ty3 - float32(5.0) * ty2 + float32(2.0))
        wy2 = h * (float32(-3.0) * ty3 + float32(4.0) * ty2 + ty); wy3 = h * (ty3 - ty2)
    hmax = e.shape[0] - 1; wmax = e.shape[1] - 1
    xa = clampi(x0 - 1, 0, wmax); xb = clampi(x0, 0, wmax); xc = clampi(x0 + 1, 0, wmax); xd = clampi(x0 + 2, 0, wmax)
    acc = float32(0.0)
    for j in range(4):
        yy = clampi(y0 + j - 1, 0, hmax)
        wy = wy0 if j == 0 else (wy1 if j == 1 else (wy2 if j == 2 else wy3))
        acc += wy * (wx0 * e[yy, xa] + wx1 * e[yy, xb] + wx2 * e[yy, xc] + wx3 * e[yy, xd])
    return acc

def make(kind):
    @cuda.jit
    def rays(e, x0s, y0s, dxs, dys, steps, out):
        i = cuda.grid(1)
        if i >= x0s.shape[0]: return
        x = x0s[i]; y = y0s[i]; m = float32(-1e30)
        for k in range(steps):
            if kind == 0: h = bilinear(e, x, y)
            elif kind == 1: h = cubic16(e, x, y, True)
            else: h = cubic16(e, x, y, False)
            m = max(m, h)
            x += dxs[i]; y += dys[i]
        out[i] = m
    return rays

rng = np.random.default_rng(0)
n, steps = 1 << 16, 400                      # 65,536 rays x 400 samples = 26.2 M samples
ang = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
step = np.float32(0.8)                       # production's far step floor, in pixels
coherent = dict(x0s=np.full(n, W / 2, np.float32), y0s=np.full(n, H / 2, np.float32),
                dxs=(np.cos(ang) * step * 0.6).astype(np.float32), dys=(np.sin(ang) * step * 0.6).astype(np.float32))
scatter = dict(x0s=rng.uniform(2, W - 3, n).astype(np.float32), y0s=rng.uniform(2, H - 3, n).astype(np.float32),
               dxs=rng.uniform(-37, 37, n).astype(np.float32), dys=rng.uniform(-37, 37, n).astype(np.float32))
# scatter: large pseudo-random jumps, wrapped by clamping -> poor cache reuse

grids = [("512 px LOLA (1 MB, cached)", dem, coef)]
big = np.tile(dem, (16, 16))                                   # 8192 x 8192 float32 = 256 MB
grids.append(("8192 px tiled (256 MB)", big, np.tile(coef, (16, 16))))
out = cuda.device_array(n, np.float32)
print(f"samples per run {n*steps/1e6:.1f} M; host spline prefilter of the 512 px DEM {prefilter_s*1000:.1f} ms (one-time)")
results = {}
for gname, g, gc in grids:
  d_dem, d_coef = cuda.to_device(g), cuda.to_device(gc)
  gh, gw = g.shape
  print(gname)
  base = dict(coherent); base["x0s"] = np.full(n, gw / 2, np.float32); base["y0s"] = np.full(n, gh / 2, np.float32)
  sc = dict(scatter); sc["x0s"] = rng.uniform(2, gw - 3, n).astype(np.float32); sc["y0s"] = rng.uniform(2, gh - 3, n).astype(np.float32)
  sc["dxs"] = rng.uniform(-37, 37, n).astype(np.float32); sc["dys"] = rng.uniform(-37, 37, n).astype(np.float32)
  for pattern, args in (("ray-coherent", base), ("scattered", sc)):
      dargs = {k: cuda.to_device(v) for k, v in args.items()}
      for kind, name, grid in ((0, "bilinear (4 taps)", d_dem), (1, "cubic B-spline (16 taps)", d_coef), (2, "Catmull-Rom (16 taps)", d_dem)):
          k = make(kind)
          k[(n + 127) // 128, 128](grid, dargs["x0s"], dargs["y0s"], dargs["dxs"], dargs["dys"], steps, out)  # compile + warm
          cuda.synchronize()
          times = []
          for _ in range(7):
              start, stop = cuda.event(), cuda.event()
              start.record(); k[(n + 127) // 128, 128](grid, dargs["x0s"], dargs["y0s"], dargs["dxs"], dargs["dys"], steps, out); stop.record()
              stop.synchronize(); times.append(cuda.event_elapsed_time(start, stop))
          ms = float(np.median(times)); results[(gname, pattern, name)] = ms
          print(f"  {pattern:13s} {name:26s} {ms:8.2f} ms  {n*steps/ms/1e6:8.2f} G samples/s  x{ms/results[(gname, pattern,'bilinear (4 taps)')]:.2f}")
