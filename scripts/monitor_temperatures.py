#!/usr/bin/env python3
"""Watch GPU, CPU, NVMe and fan readings while a long job runs.

    .venv/bin/python scripts/monitor_temperatures.py

Run this in a second terminal when you start a long job (scripts/run_safe_haven_stage3.py,
for example) and watch the first 10 to 20 minutes before you leave it running. It prints one
line every INTERVAL_SECONDS and writes the same readings to a CSV, then prints the minimum,
mean and maximum of each reading when you stop it with Ctrl-C (or after --duration seconds).

It reads what Linux reports and nothing else:
  * GPU: nvidia-smi (temperature, power, utilization, SM clock, performance state, and whether
    the driver reports a hardware or software thermal slowdown, a power brake or a power cap).
  * CPU: the coretemp sensor in /sys/class/hwmon (package and hottest core), plus the
    one-minute load average.
  * NVMe drives: each drive's own composite temperature, labelled with its controller name
    (nvme0, nvme1, ...). A drive that holds one of WATCH_PATHS is marked with *.
  * Fans: every fan that /sys/class/hwmon exposes, in RPM.

A reading is flagged (!!) when it comes within WARN_MARGIN_C degrees of the limit the machine
itself reports for that sensor (the CPU's critical temperature, an NVMe drive's "high"
temperature), or when nvidia-smi reports a GPU thermal slowdown, which is the driver saying the
GPU is already reducing its clocks. WARN_MARGIN_C is a choice made here, not a manufacturer
figure; change it if you want earlier or later warnings. This script changes nothing on the
machine and starts no GPU work; without nvidia-smi the GPU readings show n/a. Importing it starts nothing.
"""

from __future__ import annotations

import argparse
import csv
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Edit these before running.
INTERVAL_SECONDS = 10.0
LOG_DIRECTORY = Path("/e/lunar_analyst_scenarios/70south/logs")
# Drives holding these paths are marked with * in the NVMe readings.
WATCH_PATHS = [Path("/d/viper/maps/lola"), Path("/e/lunar_analyst_scenarios/70south")]
# Flag a reading this many degrees below the limit the sensor itself reports.
WARN_MARGIN_C = 10.0

HWMON_ROOT = Path("/sys/class/hwmon")
GPU_FIELDS = (
    "temperature.gpu", "power.draw", "utilization.gpu", "clocks.sm", "pstate",
    "clocks_throttle_reasons.hw_thermal_slowdown", "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown", "clocks_throttle_reasons.sw_power_cap",
)


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _milli_celsius(path: Path) -> float | None:
    text = _read(path)
    try:
        return int(text) / 1000.0
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Sensor discovery and reading
# ---------------------------------------------------------------------------


@dataclass
class Sensors:
    """Where the readings live on this machine."""

    cpu_package: Path | None = None
    cpu_package_limit: float | None = None
    cpu_cores: list[Path] = field(default_factory=list)
    nvme: list[tuple[str, Path, float | None]] = field(default_factory=list)   # (controller, input, high)
    fans: list[tuple[str, Path]] = field(default_factory=list)
    watched_controllers: set[str] = field(default_factory=set)


def controller_for_path(path: Path, run=subprocess.run) -> str | None:
    """The NVMe controller (``nvme2``) of the drive holding ``path``, best effort."""
    try:
        done = run(["findmnt", "-no", "SOURCE", "--target", str(path)],
                   capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(nvme\d+)n\d+", done.stdout or "")
    return match.group(1) if match else None


def discover(hwmon_root: Path = HWMON_ROOT, watch_paths=(), run=subprocess.run) -> Sensors:
    sensors = Sensors()
    for hwmon in sorted(hwmon_root.glob("hwmon*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        name = _read(hwmon / "name")
        if name == "coretemp":
            for label_path in sorted(hwmon.glob("temp*_label")):
                label = _read(label_path) or ""
                input_path = hwmon / label_path.name.replace("_label", "_input")
                if label.startswith("Package"):
                    sensors.cpu_package = input_path
                    sensors.cpu_package_limit = _milli_celsius(
                        hwmon / label_path.name.replace("_label", "_crit"))
                elif label.startswith("Core"):
                    sensors.cpu_cores.append(input_path)
        elif name == "nvme":
            controller = Path((hwmon / "device").resolve()).name
            sensors.nvme.append((controller, hwmon / "temp1_input", _milli_celsius(hwmon / "temp1_max")))
        elif (hwmon / "fan1_input").exists():
            sensors.fans.append((name or hwmon.name, hwmon / "fan1_input"))
    sensors.nvme.sort(key=lambda item: item[0])
    for path in watch_paths:
        controller = controller_for_path(Path(path), run)
        if controller:
            sensors.watched_controllers.add(controller)
    return sensors


def read_gpu(run=subprocess.run) -> dict | None:
    """The GPU's readings from nvidia-smi, or ``None`` when it cannot be queried."""
    try:
        done = run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    values = [item.strip() for item in done.stdout.strip().splitlines()[0].split(",")]
    if len(values) != len(GPU_FIELDS):
        return None

    def number(text):
        try:
            return float(text)
        except ValueError:
            return None

    def active(text):
        return None if text in ("N/A", "[N/A]") else int(text == "Active")

    return {
        "gpu_c": number(values[0]), "gpu_w": number(values[1]), "gpu_util_pct": number(values[2]),
        "gpu_sm_mhz": number(values[3]), "gpu_pstate": values[4],
        "gpu_thermal_slowdown": max(active(values[5]) or 0, active(values[6]) or 0),
        "gpu_power_brake": active(values[7]), "gpu_power_cap": active(values[8]),
    }


def take_sample(sensors: Sensors, gpu_reader=read_gpu, loadavg_path: Path = Path("/proc/loadavg")) -> dict:
    sample: dict = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    sample.update(gpu_reader() or {})
    package = _milli_celsius(sensors.cpu_package) if sensors.cpu_package else None
    cores = [t for t in (_milli_celsius(p) for p in sensors.cpu_cores) if t is not None]
    sample["cpu_package_c"] = package
    sample["cpu_hottest_core_c"] = max(cores) if cores else None
    load = _read(loadavg_path)
    sample["load_1min"] = float(load.split()[0]) if load else None
    for controller, path, _high in sensors.nvme:
        sample[f"{controller}_c"] = _milli_celsius(path)
    for index, (name, path) in enumerate(sensors.fans, start=1):
        text = _read(path)
        sample[f"fan{index}_{name}_rpm"] = float(text) if text and text.isdigit() else None
    return sample


def warnings(sample: dict, sensors: Sensors, margin_c: float = WARN_MARGIN_C) -> list[str]:
    found = []
    package, limit = sample.get("cpu_package_c"), sensors.cpu_package_limit
    if package is not None and limit is not None and package >= limit - margin_c:
        found.append(f"CPU package {package:.0f} C is within {margin_c:.0f} C of its {limit:.0f} C limit")
    for controller, _path, high in sensors.nvme:
        value = sample.get(f"{controller}_c")
        if value is not None and high is not None and value >= high - margin_c:
            found.append(f"{controller} {value:.0f} C is within {margin_c:.0f} C of its {high:.0f} C limit")
    if sample.get("gpu_thermal_slowdown"):
        found.append("GPU reports a thermal slowdown (clocks are already being reduced)")
    if sample.get("gpu_power_brake"):
        found.append("GPU reports a hardware power brake")
    return found


def format_line(sample: dict, sensors: Sensors, flags: list[str]) -> str:
    def value(key, unit="", digits=0):
        item = sample.get(key)
        return "n/a" if item is None else f"{item:.{digits}f}{unit}"

    parts = [sample["time"][11:]]
    if "gpu_c" in sample:
        slow = " THERMAL-SLOWDOWN" if sample.get("gpu_thermal_slowdown") else ""
        cap = " power-cap" if sample.get("gpu_power_cap") else ""
        parts.append(f"GPU {value('gpu_c', 'C')} {value('gpu_w', 'W')} {value('gpu_util_pct', '%')} "
                     f"{value('gpu_sm_mhz', 'MHz')} {sample.get('gpu_pstate', 'n/a')}{slow}{cap}")
    else:
        parts.append("GPU n/a")
    parts.append(f"CPU pkg {value('cpu_package_c', 'C')} hottest-core {value('cpu_hottest_core_c', 'C')} "
                 f"load {value('load_1min', '', 1)}")
    drives = " ".join(
        f"{c}{'*' if c in sensors.watched_controllers else ''} {value(f'{c}_c', 'C')}"
        for c, _p, _h in sensors.nvme)
    parts.append(f"NVMe {drives}" if drives else "NVMe n/a")
    fans = [k for k in sample if k.startswith("fan")]
    if fans:
        parts.append("fans " + "/".join(value(k) for k in fans) + " rpm")
    line = " | ".join(parts)
    return line + "".join(f"\n    !! {flag}" for flag in flags)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class Summary:
    """Minimum, mean and maximum of every numeric reading, and how often warnings fired."""

    def __init__(self) -> None:
        self.samples = 0
        self.warned_samples = 0
        self.thermal_slowdown_samples = 0
        self._stats: dict[str, list[float]] = {}     # key -> [min, sum, max, count]
        self.first = self.last = ""

    def add(self, sample: dict, flags: list[str]) -> None:
        self.samples += 1
        self.warned_samples += bool(flags)
        self.thermal_slowdown_samples += bool(sample.get("gpu_thermal_slowdown"))
        self.first = self.first or sample["time"]
        self.last = sample["time"]
        for key, item in sample.items():
            if isinstance(item, (int, float)):
                stat = self._stats.setdefault(key, [item, 0.0, item, 0])
                stat[0], stat[1], stat[2], stat[3] = min(stat[0], item), stat[1] + item, max(stat[2], item), stat[3] + 1

    def lines(self) -> list[str]:
        if not self.samples:
            return ["No samples were taken."]
        out = [f"{self.samples} samples from {self.first} to {self.last}.",
               f"Samples with a warning: {self.warned_samples}. "
               f"Samples with a GPU thermal slowdown: {self.thermal_slowdown_samples}.",
               f"{'reading':<28}{'min':>10}{'mean':>10}{'max':>10}"]
        for key, (low, total, high, count) in self._stats.items():
            out.append(f"{key:<28}{low:>10.1f}{total / count:>10.1f}{high:>10.1f}")
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


class _Stop(Exception):
    pass


def main(argv=None, *, sensors: Sensors | None = None, gpu_reader=read_gpu, clock=time, out=print) -> int:
    parser = argparse.ArgumentParser(description="Print and log GPU, CPU, NVMe and fan readings.")
    parser.add_argument("--interval", type=float, default=INTERVAL_SECONDS, help="seconds between samples")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    parser.add_argument("--log", type=Path, default=None, help="CSV path (default: a new file in LOG_DIRECTORY)")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    sensors = sensors or discover(watch_paths=WATCH_PATHS)
    log_path = args.log or LOG_DIRECTORY / f"temperatures_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def stop(*_):
        raise _Stop

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    summary = Summary()
    out(f"Sampling every {args.interval:g} s; Ctrl-C stops and prints a summary. Log: {log_path}")
    started = clock.monotonic()
    try:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = None
            while True:
                sample = take_sample(sensors, gpu_reader)
                flags = warnings(sample, sensors)
                summary.add(sample, flags)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(sample) + ["warnings"], restval="")
                    writer.writeheader()
                writer.writerow({**sample, "warnings": "; ".join(flags)})
                handle.flush()
                out(format_line(sample, sensors, flags))
                if args.duration is not None and clock.monotonic() - started + args.interval > args.duration:
                    break
                clock.sleep(args.interval)
    except _Stop:
        pass
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    out("")
    for line in summary.lines():
        out(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
