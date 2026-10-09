#!/usr/bin/env python3
"""CPU-only reproductions of findings from the lunarscout 0.1.0rc5 review.

No GPU, no network. Run from anywhere:

    python review_scripts/s1_cpu_repros.py            # repo root = parent of this folder
    python review_scripts/s1_cpu_repros.py --repo /e/projects/lunarscout

Each check prints one line:  REPRODUCED (bug present)  |  not reproduced (fixed or
not triggered)  |  ERROR (script could not run the check; paste the traceback).
Paste the full output back to the reviewer.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
ARGS = parser.parse_args()
REPO = ARGS.repo.resolve()
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    print(f"[{status:>14}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def check(name):
    def wrap(fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            record(name, "ERROR", traceback.format_exc(limit=3).strip().splitlines()[-1])
            traceback.print_exc()
        return fn

    return wrap


# ---------------------------------------------------------------- B1
@check("B1 signed-int multiply overflow undetected (map_algebra/_dtypes.py)")
def _b1():
    from lunarscout.map_algebra._dtypes import _overflow_mask

    missed = {}
    for dt in (np.int8, np.int16):
        info = np.iinfo(dt)
        vals = np.arange(info.min, info.max + 1, dtype=np.int64)
        if dt is np.int16:  # full 2^32 grid is too big; use a dense random sample + edge values
            rng = np.random.default_rng(0)
            a = rng.choice(vals, 2_000_000)
            b = rng.choice(vals, 2_000_000)
        else:
            a, b = (x.ravel() for x in np.meshgrid(vals, vals))
        true_overflow = (a * b < info.min) | (a * b > info.max)
        mask = _overflow_mask(a.astype(dt), b.astype(dt), np.dtype(dt), "multiply")
        missed[dt.__name__] = int((true_overflow & ~mask).sum())
    status = "REPRODUCED" if any(missed.values()) else "not reproduced"
    record("B1 signed-int multiply overflow undetected", status, f"missed overflows: {missed}")


# ---------------------------------------------------------------- horizon -inf in lightmap
def _lightmap_fraction(fill_or_array):
    from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
    from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
    from lunarscout._numba_horizon.psr import _pixel_frame

    dem = DemGrid(
        np.zeros((2, 3), np.float32),
        np.array((1000.0, 20.0, 0.0, -1000.0, 0.0, -20.0)),
        ProjectionParameters(1737400.0, -np.pi / 2, 0.0, 1.0, 0.0, 0.0),
    )
    rot, tr = _pixel_frame(dem, 0, 0)
    el, az = np.deg2rad(5.0), np.deg2rad(90.0)
    local = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)]) * 1.5e11
    vec = ((local - tr) @ rot.T)[None, :]
    h = np.full((128, 128, 1440), -50.0, np.float32) if np.isscalar(fill_or_array) is False else None
    return dem, vec, h, LightmapCpuSession


@check("H-lightmap: -inf horizon bins turn a lit pixel dark")
def _lm():
    dem, vec, _, Session = _lightmap_fraction(None)
    out = {}
    for label, fill in (("-50 deg", -50.0), ("-inf", -np.inf), ("nan", np.nan)):
        h = np.full((128, 128, 1440), fill, np.float32)
        r = np.stack(list(Session(time_batch_size=1).iter_patch_fraction_tiles(
            dem, h, vec, tile_y=0, tile_x=0, valid_height=2, valid_width=3)))
        out[label] = float(r[0, 0, 0])
    bad = out["-50 deg"] > 0.99 and (out["-inf"] < 0.01 or out["nan"] < 0.01)
    record("H-lightmap: -inf horizon bins -> dark pixel", "REPRODUCED" if bad else "not reproduced", str(out))


# ---------------------------------------------------------------- tile files / resume
@check("H2/H3 resume and tile-file behavior (file_format.py, pipeline.py)")
def _tiles():
    from lunarscout._numba_horizon import file_format as ff
    from lunarscout._numba_horizon.file_format import (
        AZIMUTH_COUNT, PIXEL_COUNT, HorizonTileStore, read_horizon_tile)
    from lunarscout._numba_horizon.pipeline import enumerate_patches, run_bounded_pipeline

    tmp = Path(tempfile.mkdtemp(prefix="ls_s1_"))
    st = HorizonTileStore(tmp)
    z = np.zeros((PIXEL_COUNT, AZIMUTH_COUNT), np.float32)

    # A: truncated .cbin is treated as done by resume
    p = st.write(0, 0, 0.0, z, compress=True)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    calls: list[int] = []
    res = run_bounded_pipeline(
        enumerate_patches(128, 128), store=st, prepare_patch=lambda q: 1,
        processor_factory=lambda i: (lambda q, x: calls.append(1) or z), compress=True)
    try:
        read_horizon_tile(p)
        readable = True
    except Exception as exc:  # noqa: BLE001
        readable = False
    record("H2 truncated tile skipped by resume",
           "REPRODUCED" if (res.skipped_patches == 1 and not calls and not readable) else "not reproduced",
           f"skipped={res.skipped_patches} computed={len(calls)} readable={readable}")

    # B: observer height truncated to decimeters in file name
    names = {h: st.build_file_name(0, 0, h, compress=True) for h in (1.5, 1.55, 1.59)}
    record("H3 observer heights 1.5/1.55/1.59 m share one file name",
           "REPRODUCED" if len(set(names.values())) == 1 else "not reproduced", str(set(names.values())))

    # C: overwrite with a different extension leaves the stale tile preferred
    tmp2 = Path(tempfile.mkdtemp(prefix="ls_s1_"))
    st2 = HorizonTileStore(tmp2)
    st2.write(0, 0, 0.0, np.full_like(z, 5.0), compress=True)
    st2.write(0, 0, 0.0, np.full_like(z, 20.0), compress=False)
    got = float(st2.read(0, 0, 0.0)[0, 0, 0])
    record("M2 re-write with compress=False: old .cbin still wins",
           "REPRODUCED" if abs(got - 5.0) < 0.5 else "not reproduced", f"read back {got} (new data is 20.0)")

    # D: values beyond 50 deg are silently clamped
    tmp3 = Path(tempfile.mkdtemp(prefix="ls_s1_"))
    pp = HorizonTileStore(tmp3).write(0, 0, 0.0, np.full_like(z, 60.0), compress=True)
    mx = float(read_horizon_tile(pp).max())
    record("LOW: horizon > 50 deg clamped on write", "REPRODUCED" if mx <= 50.01 else "not reproduced",
           f"60 deg -> {mx}")


# ---------------------------------------------------------------- PSR writer hang (subprocess, 40 s timeout)
PSR_SNIPPET = textwrap.dedent(
    """
    import sys, time, tempfile
    from pathlib import Path
    repo = Path(sys.argv[1]); sys.path[:0] = [str(repo/'src'), str(repo/'tests'), str(repo/'tests'/'numba_horizon')]
    import numpy as np
    import test_phase6b_psr as t
    from lunarscout._numba_horizon.file_format import HorizonTileStore, AZIMUTH_COUNT
    from lunarscout._numba_horizon.psr_pipeline import run_psr_product
    from lunarscout._numba_horizon.product_store import ProductBatchWriter
    tmp = Path(tempfile.mkdtemp()); W = 128*6
    dem = t._dem(width=W, height=1); hs = HorizonTileStore(tmp/'h')
    for i in range(6):
        hs.write(0, 128*i, 0.0, np.zeros((128, AZIMUTH_COUNT), np.float32), compress=True, valid_width=128, valid_height=1)
    def slow_fail(self, *a, **k):
        time.sleep(0.5); raise OSError('disk full (simulated)')
    ProductBatchWriter.write_patch_with_timings = slow_fail
    try:
        run_psr_product(dem=dem, georef=t._georef(W,1), horizon_store=hs, output_path=tmp/'o.tif',
            sun_vectors_m=t._position_for_local_angle(dem,0.0,1.0)[None,:], backend='cpu', pipeline_mode='bounded')
    except Exception as e:
        print('RAISED', type(e).__name__)
    """
)


@check("H1 bounded PSR pipeline hangs when the writer fails")
def _psr():
    if not (REPO / "tests/numba_horizon/test_phase6b_psr.py").exists():
        record("H1 bounded PSR pipeline hang", "ERROR", "tests/numba_horizon/test_phase6b_psr.py not found")
        return
    try:
        r = subprocess.run([sys.executable, "-c", PSR_SNIPPET, str(REPO)], capture_output=True, text=True, timeout=40)
        record("H1 bounded PSR pipeline hang", "not reproduced",
               f"returned in time; stdout={r.stdout.strip()!r} stderr_tail={r.stderr.strip()[-120:]!r}")
    except subprocess.TimeoutExpired:
        record("H1 bounded PSR pipeline hang", "REPRODUCED", "no return after 40 s (expected: raises within ~3 s)")


# ---------------------------------------------------------------- tolerance arithmetic
@check("M tolerance: 0.005 deg is not 1% sunlight error")
def _tol():
    r_deg = 0.25  # solar angular radius used by the docs (0.5 deg diameter)
    dF_ddeg = 2.0 / (np.pi * r_deg)  # d(fraction)/d(horizon offset) at disk center, uniform disk
    pct = 0.005 * dF_ddeg * 100
    record("M tolerance 0.005 deg -> sunlight error", "REPRODUCED" if pct > 1.0 else "not reproduced",
           f"{pct:.2f}% (about {0.01 / dF_ddeg:.4f} deg gives 1%)")


print("\n==== SUMMARY ====")
for n, s, d in RESULTS:
    print(f"{s:>14}  {n}")
