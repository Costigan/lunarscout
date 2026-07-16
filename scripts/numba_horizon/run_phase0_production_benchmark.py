#!/usr/bin/env python3
"""Run the Phase 0 C# multi-patch benchmark with peak GPU-memory sampling."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = "scripts/numba_horizon/CSharpMultiPatchBenchmark.csproj"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "numba-horizon-phase-0-multi-patch-benchmark.json"


def run_nvidia_smi(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["nvidia-smi", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )


def parse_gpu_summary() -> dict[str, Any]:
    result = run_nvidia_smi(
        [
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "command": ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits"],
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


class GpuMemorySampler:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.pid: int | None = None
        self.phase = "startup"
        self.samples_by_phase: dict[str, list[int]] = {}
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def set_pid(self, pid: int) -> None:
        with self._lock:
            self.pid = pid

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                pid = self.pid
                phase = self.phase
            if pid is not None:
                try:
                    result = run_nvidia_smi(
                        [
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ]
                    )
                    if result.returncode != 0:
                        self.errors.append(result.stderr.strip() or f"exit code {result.returncode}")
                    else:
                        for line in result.stdout.splitlines():
                            fields = [field.strip() for field in line.split(",")]
                            if len(fields) == 2 and int(fields[0]) == pid:
                                self.samples_by_phase.setdefault(phase, []).append(int(fields[1]))
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    self.errors.append(str(exc))
            self._stop.wait(self.interval_seconds)

    def report(self) -> dict[str, Any]:
        phases: dict[str, Any] = {}
        for phase, samples in sorted(self.samples_by_phase.items()):
            phases[phase] = {
                "sample_count": len(samples),
                "peak_process_gpu_memory_mib": max(samples),
                "minimum_process_gpu_memory_mib": min(samples),
            }
        return {
            "method": "Repeated nvidia-smi --query-compute-apps sampling matched to the benchmark process PID.",
            "sample_interval_seconds": self.interval_seconds,
            "units": "MiB as reported by nvidia-smi",
            "phases": phases,
            "error_count": len(self.errors),
            "errors": self.errors[:20],
            "limitations": [
                "Sampled peaks can understate allocations shorter than the polling interval.",
                "Per-process nvidia-smi memory is CUDA context/allocation accounting, not device-wide memory.used.",
            ],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_directory", type=Path)
    parser.add_argument("dem_paths", nargs="+", type=Path)
    parser.add_argument("--patch-count", type=int, default=4)
    parser.add_argument("--observer-elevation-m", type=float, default=0.0)
    parser.add_argument("--gpu-concurrency", type=int, default=4)
    parser.add_argument("--segment-queue-size", type=int, default=6)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.05)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_interval_seconds <= 0:
        raise ValueError("--sample-interval-seconds must be greater than zero")
    if args.work_directory.exists() and any(args.work_directory.iterdir()):
        raise ValueError(f"work directory is not empty: {args.work_directory}")

    gpu_before = parse_gpu_summary()
    if gpu_before["exit_code"] != 0:
        raise RuntimeError(f"nvidia-smi GPU probe failed: {gpu_before['stderr']}")

    command = [
        "dotnet",
        "run",
        "--project",
        PROJECT,
        "--",
        str(args.work_directory),
        str(args.patch_count),
        str(args.observer_elevation_m),
        str(args.gpu_concurrency),
        str(args.segment_queue_size),
        *(str(path) for path in args.dem_paths),
    ]
    environment = {**os.environ, "LUNARSCOUT_BASELINE_COMMIT": args.baseline_commit}
    sampler = GpuMemorySampler(args.sample_interval_seconds)
    sampler.start()
    output_lines: list[str] = []
    report_path: Path | None = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            print(line, flush=True)
            output_lines.append(line)
            if line.startswith("BENCHMARK_PID "):
                sampler.set_pid(int(line.split(maxsplit=1)[1]))
            elif line.startswith("BENCHMARK_PHASE "):
                marker = line.split(maxsplit=1)[1]
                if marker.endswith("_start"):
                    sampler.set_phase(marker.removesuffix("_start"))
                elif marker.endswith("_end"):
                    sampler.set_phase("between_runs")
            elif line.startswith("BENCHMARK_REPORT "):
                report_path = Path(line.split(maxsplit=1)[1])
        return_code = process.wait()
    finally:
        sampler.stop()
    elapsed_seconds = time.monotonic() - started

    if return_code != 0:
        raise RuntimeError(
            f"C# benchmark failed with exit code {return_code}:\n" + "\n".join(output_lines[-50:])
        )
    if report_path is None or not report_path.is_file():
        raise RuntimeError("C# benchmark did not report a readable JSON artifact")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["gpu_memory"] = sampler.report()
    report["gpu_memory"]["device_before"] = gpu_before
    report["gpu_memory"]["device_after"] = parse_gpu_summary()
    report["launcher"] = {
        "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "command": command,
        "elapsed_seconds": elapsed_seconds,
    }
    report["limitations"] = [
        limitation
        for limitation in report["limitations"]
        if "external Python sampler" not in limitation
    ]
    report["limitations"].append(
        "Peak GPU memory is sampled rather than an allocation-level high-water mark."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
