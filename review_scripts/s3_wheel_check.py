#!/usr/bin/env python3
"""Build the wheel + sdist from the real repo and check what actually ships.

Answers the open packaging questions from the review:
  * does the wheel contain _numba_horizon/, map_algebra/, trajectory/, data/spice/*.toml?
  * is LICENSE included; are examples/docs shipped (README links)?
  * does a clean venv install and import, and do pandas/matplotlib features fail cleanly?
  * does `pip check` pass and does the worker entry point start?

Needs network (to build and to install dependencies) and `python -m pip install build`.
Does NOT need a GPU. Run from anywhere:

    python review_scripts/s3_wheel_check.py [--repo /e/projects/lunarscout] [--keep]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
ap.add_argument("--keep", action="store_true", help="keep the temp directory")
args = ap.parse_args()
repo = args.repo.resolve()
work = Path(tempfile.mkdtemp(prefix="ls_wheel_"))
print(f"repo: {repo}\nwork: {work}")
issues: list[str] = []


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


# 1. Build from a clean copy of tracked files only (so stray/untracked files cannot leak in)
src = work / "src_snapshot"
src.mkdir()
ls = run(["git", "-C", str(repo), "ls-files", "-z"])
if ls.returncode == 0:
    import shutil
    for rel in [p for p in ls.stdout.split("\0") if p]:
        f = repo / rel
        if f.is_file():
            (src / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, src / rel)
    dirty = run(["git", "-C", str(repo), "status", "--porcelain"]).stdout.strip()
    if dirty:
        print("WARNING: working tree has uncommitted changes; the build uses the working-tree copy of tracked files:\n" + dirty[:800])
else:
    print("not a git repo here; building from the directory as-is")
    src = repo

dist = work / "dist"
b = run([sys.executable, "-m", "build", "--outdir", str(dist), str(src)])
print(b.stdout[-1500:], b.stderr[-1500:])
wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
if b.returncode or not wheels:
    sys.exit("BUILD FAILED - paste the output above")
whl = wheels[0]
print(f"\nwheel: {whl.name}")

# 2. Inspect contents
names = zipfile.ZipFile(whl).namelist()
required = [
    "lunarscout/__init__.py", "lunarscout/_numba_horizon/__init__.py", "lunarscout/_numba_horizon/worker.py",
    "lunarscout/_numba_horizon/cuda_backend.py", "lunarscout/map_algebra/__init__.py",
    "lunarscout/trajectory/__init__.py", "lunarscout/data/spice/default_kernels.toml",
]
for r in required:
    ok = r in names
    print(f"  {'ok ' if ok else 'MISSING'}  {r}")
    if not ok:
        issues.append(f"wheel missing {r}")
for pat in ("LICENSE", "py.typed"):
    print(f"  {'ok ' if any(pat in n for n in names) else 'absent'}  {pat} (informational)")
junk = [n for n in names if n.endswith((".pyc", "~")) or "__pycache__" in n or ".ipynb_checkpoints" in n or "/#" in n]
print(f"  junk files in wheel: {junk[:10] or 'none'}")
if junk:
    issues.append(f"junk in wheel: {junk[:5]}")
if sdists:
    sn = tarfile.open(sdists[0]).getnames()
    print(f"sdist: {sdists[0].name}, {len(sn)} files; has docs/: {any('/docs/' in n for n in sn)}, "
          f"examples/: {any('/examples/' in n for n in sn)}, tests/: {any('/tests/' in n for n in sn)}")
    print(f"  stray in sdist: {[n for n in sn if n.endswith('~') or '/#' in n or 'checkpoints' in n][:10] or 'none'}")

# 3. Clean venv install + import
venv_dir = work / "venv"
venv.EnvBuilder(with_pip=True).create(venv_dir)
py = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
inst = run([str(py), "-m", "pip", "install", str(whl)])
print(inst.stdout[-600:], inst.stderr[-800:])
if inst.returncode:
    issues.append("pip install of wheel failed")
else:
    for label, code in {
        "import lunarscout": "import lunarscout as ls; print(ls.__version__)",
        "import horizon package": "import lunarscout._numba_horizon.worker, lunarscout._numba_horizon.cuda_backend; print('ok')",
        "no numba/spice at import": "import sys, lunarscout; print('numba' in sys.modules, 'spiceypy' in sys.modules)",
        "spice manifest resolves": "import lunarscout as ls; p=ls.spice; print([n for n in dir(p) if 'manifest' in n.lower()][:3])",
        "pandas feature w/o pandas": "import lunarscout as ls\ntry:\n    ls.body_azimuth_elevation_dataframe\n    print('attr present')\nexcept Exception as e: print(type(e).__name__)",
    }.items():
        r = run([str(py), "-c", code])
        print(f"  [{label}] rc={r.returncode} out={r.stdout.strip()[-200:]!r} err={r.stderr.strip()[-200:]!r}")
        if r.returncode:
            issues.append(f"clean-venv check failed: {label}")
    r = run([str(py), "-m", "pip", "check"])
    print("  pip check:", r.stdout.strip() or r.stderr.strip())
    exe = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "lunarscout-horizon-worker"
    r = run([str(exe), "--help"]) if exe.exists() else None
    print("  worker --help:", "entry point missing" if r is None else f"rc={r.returncode} {(r.stdout or r.stderr).strip()[:200]!r}")
    if r is None or r.returncode:
        issues.append("worker entry point not runnable")

print("\n==== SUMMARY ====")
print("\n".join(f"- {i}" for i in issues) if issues else "no packaging problems found by these checks")
if not args.keep:
    import shutil
    shutil.rmtree(work, ignore_errors=True)
else:
    print(f"kept {work}")
