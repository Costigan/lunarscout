#!/usr/bin/env python3
"""Capture a machine-readable Phase 0 Numba-horizon baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "numba-horizon-phase-0-baseline.json"
NATIVE_PROJECT = "native/moonlib/moonlib.csproj"
NATIVE_TEST_PROJECT = "native/tests/HorizonGen.Tests/HorizonGen.Tests.csproj"


def run_command(
    command: Sequence[str],
    *,
    timeout_seconds: int = 120,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        result = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            env={**os.environ, **(environment or {})},
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return {
            "command": list(command),
            "available": False,
            "error": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(command),
            "available": True,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    return {
        "command": list(command),
        "available": True,
        "exit_code": result.returncode,
        "duration_seconds": round(
            (datetime.now(UTC) - started).total_seconds(), 6
        ),
        "stdout": result.stdout.rstrip(),
        "stderr": result.stderr.rstrip(),
    }


def command_stdout(command: Sequence[str]) -> str | None:
    result = run_command(command)
    if result.get("exit_code") != 0:
        return None
    return str(result.get("stdout", ""))


def parse_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def parse_cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def parse_memory_kib() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    match = re.search(
        r"^MemTotal:\s+(\d+)\s+kB$",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return int(match.group(1)) if match else None


def find_nvcc() -> str | None:
    on_path = shutil.which("nvcc")
    if on_path:
        return on_path
    conventional = Path("/usr/local/cuda/bin/nvcc")
    return str(conventional) if conventional.is_file() else None


def capture_cuda_packages() -> dict[str, str]:
    result = run_command(["dpkg-query", "-W", "-f=${Package}\t${Version}\n"])
    if result.get("exit_code") != 0:
        return {}
    packages: dict[str, str] = {}
    for line in str(result.get("stdout", "")).splitlines():
        if "\t" not in line:
            continue
        name, version = line.split("\t", 1)
        if name.startswith(("cuda-", "nvidia-", "libnvidia-")):
            packages[name] = version
    return packages


def capture_repository(baseline_commit: str | None) -> dict[str, Any]:
    head = command_stdout(["git", "rev-parse", "HEAD"])
    status = command_stdout(["git", "status", "--short", "--branch"])
    branch = command_stdout(["git", "branch", "--show-current"])
    commit = baseline_commit or head
    commit_details = None
    if commit:
        commit_details = command_stdout(
            ["git", "show", "-s", "--format=%H%n%aI%n%s", commit]
        )
    status_lines = status.splitlines() if status else []
    return {
        "baseline_commit": commit,
        "capture_head": head,
        "branch": branch,
        "working_tree_clean": len(status_lines) <= 1,
        "status": status_lines,
        "baseline_commit_details": (
            commit_details.splitlines() if commit_details else None
        ),
    }


def capture_gpu() -> dict[str, Any]:
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    summary = run_command(["nvidia-smi"])
    driver_cuda_compatibility = None
    if summary.get("exit_code") == 0:
        match = re.search(
            r"CUDA Version:\s*([0-9.]+)", str(summary.get("stdout", ""))
        )
        if match:
            driver_cuda_compatibility = match.group(1)
    return {
        "probe_context": (
            "The result describes only the environment in which this script ran. "
            "A sandbox may hide a host GPU."
        ),
        "query": query,
        "driver_reported_cuda_compatibility": driver_cuda_compatibility,
    }


def capture_toolchain() -> dict[str, Any]:
    nvcc = find_nvcc()
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "dotnet": run_command(["dotnet", "--info"]),
        "nvcc_path": nvcc,
        "nvcc": run_command([nvcc, "--version"]) if nvcc else None,
        "cuda_and_nvidia_deb_packages": capture_cuda_packages(),
    }


def run_native_checks(run_gpu_probe: bool) -> dict[str, Any]:
    build = run_command(
        ["dotnet", "build", NATIVE_PROJECT, "-t:Rebuild", "-v", "minimal"],
        timeout_seconds=600,
    )
    tests = run_command(
        [
            "dotnet",
            "test",
            NATIVE_TEST_PROJECT,
            "--no-restore",
            "-v",
            "minimal",
            "--filter",
            "TestCategory=Fast",
        ],
        timeout_seconds=900,
    )
    checks = {"moonlib_rebuild": build, "fast_native_tests": tests}
    if run_gpu_probe:
        checks["ilgpu_cuda_probe"] = run_command(
            [
                "dotnet",
                "test",
                NATIVE_TEST_PROJECT,
                "--no-restore",
                "-v",
                "minimal",
                "--filter",
                "TestCategory=GpuBaseline",
            ],
            timeout_seconds=600,
            environment={"LUNARSCOUT_REQUIRE_CUDA_BASELINE": "1"},
        )
    return checks


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    uname = platform.uname()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "capture_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "repository": capture_repository(args.baseline_commit),
        "host": {
            "platform": platform.platform(),
            "uname": {
                "system": uname.system,
                "node": uname.node,
                "release": uname.release,
                "version": uname.version,
                "machine": uname.machine,
            },
            "os_release": parse_os_release(),
            "cpu_model": parse_cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "memory_total_kib": parse_memory_kib(),
        },
        "gpu": capture_gpu(),
        "toolchain": capture_toolchain(),
        "native_checks": (
            run_native_checks(args.run_gpu_probe) if args.run_native_checks else None
        ),
        "limitations": [
            "GPU visibility is scoped to the process that ran this capture.",
            "The Fast test tier does not require CUDA; the separate GpuBaseline test does.",
            "This manifest does not claim Numba compatibility or contain a Numba environment.",
            "Driver-reported CUDA compatibility is not the installed CUDA toolkit version.",
        ],
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--baseline-commit",
        help="Commit containing the C# horizon baseline; defaults to HEAD.",
    )
    parser.add_argument(
        "--run-native-checks",
        action="store_true",
        help="Rebuild moonlib and run the Fast native test tier.",
    )
    parser.add_argument(
        "--run-gpu-probe",
        action="store_true",
        help="With --run-native-checks, require the ILGPU CUDA baseline test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    manifest = build_manifest(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)

    checks = manifest.get("native_checks") or {}
    failed_checks = [
        name for name, result in checks.items() if result.get("exit_code") != 0
    ]
    return 1 if failed_checks else 0


if __name__ == "__main__":
    raise SystemExit(main())
