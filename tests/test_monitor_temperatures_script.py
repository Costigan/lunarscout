"""scripts/monitor_temperatures.py against a fake /sys/class/hwmon tree and fake nvidia-smi."""

from __future__ import annotations

import csv
import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "monitor_temperatures.py"


def load():
    spec = importlib.util.spec_from_file_location("monitor_temperatures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


monitor = load()


def write(path: Path, text) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text))


@pytest.fixture()
def hwmon(tmp_path):
    root = tmp_path / "hwmon"
    write(root / "hwmon0" / "name", "AC")
    cpu = root / "hwmon11"
    write(cpu / "name", "coretemp")
    for index, (label, value) in enumerate([("Package id 0", 67000), ("Core 0", 60000), ("Core 1", 63000)], start=1):
        write(cpu / f"temp{index}_label", label)
        write(cpu / f"temp{index}_input", value)
        write(cpu / f"temp{index}_crit", 105000)
    for hwmon_number, controller, temperature in [(7, "nvme1", 51850), (5, "nvme0", 38850), (6, "nvme2", 22850)]:   # this machine's hwmon order
        directory = root / f"hwmon{hwmon_number}"
        write(directory / "name", "nvme")
        write(directory / "temp1_input", temperature)
        write(directory / "temp1_max", 83850)
        device = tmp_path / "devices" / controller
        device.mkdir(parents=True)
        (directory / "device").symlink_to(device)
    write(root / "hwmon1" / "name", "acpi_fan")
    write(root / "hwmon1" / "fan1_input", 3417)
    write(root / "hwmon2" / "name", "acpi_fan")
    write(root / "hwmon2" / "fan1_input", "oops")
    return root


def test_discovery_finds_the_cpu_nvme_and_fans_and_ignores_the_rest(hwmon) -> None:
    sensors = monitor.discover(hwmon)
    assert sensors.cpu_package.parent.name == "hwmon11" and sensors.cpu_package_limit == 105.0
    assert len(sensors.cpu_cores) == 2
    assert [(c, high) for c, _p, high in sensors.nvme] == [("nvme0", 83.85), ("nvme1", 83.85), ("nvme2", 83.85)]   # by name, not hwmon number
    assert len(sensors.fans) == 2


def test_a_sample_holds_every_reading_and_survives_missing_ones(hwmon, tmp_path) -> None:
    sensors = monitor.discover(hwmon)
    write(tmp_path / "loadavg", "0.64 0.50 0.30 1/900 12345")
    gpu = {"gpu_c": 53.0, "gpu_thermal_slowdown": 0}
    sample = monitor.take_sample(sensors, lambda: gpu, tmp_path / "loadavg")
    assert sample["cpu_package_c"] == 67.0 and sample["cpu_hottest_core_c"] == 63.0
    assert sample["nvme0_c"] == 38.85 and sample["nvme1_c"] == 51.85
    assert sample["fan1_acpi_fan_rpm"] == 3417.0 and sample["fan2_acpi_fan_rpm"] is None
    assert sample["load_1min"] == 0.64 and sample["gpu_c"] == 53.0
    (sensors.cpu_package).unlink()
    gone = monitor.take_sample(sensors, lambda: None, tmp_path / "no_such_loadavg")
    assert gone["cpu_package_c"] is None and gone["load_1min"] is None and "gpu_c" not in gone


def test_warnings_use_the_limits_the_sensors_report(hwmon) -> None:
    sensors = monitor.discover(hwmon)
    base = {"cpu_package_c": 94.9, "nvme0_c": 73.8, "nvme1_c": 20.0, "gpu_thermal_slowdown": 0}
    assert monitor.warnings(base, sensors) == []
    hot = {**base, "cpu_package_c": 95.0, "nvme0_c": 73.85, "gpu_thermal_slowdown": 1, "gpu_power_brake": 1}
    found = monitor.warnings(hot, sensors)
    assert len(found) == 4
    assert "CPU package 95 C" in found[0] and "nvme0 74 C" in found[1] and "thermal slowdown" in found[2]
    assert "power brake" in found[3]
    wide = monitor.warnings({**base, "cpu_package_c": 80.0}, sensors, margin_c=30.0)   # a wider margin warns earlier
    assert len(wide) == 2 and "CPU package 80 C is within 30 C" in wide[0] and "nvme0" in wide[1]


def fake_run(stdout="", returncode=0, error=None):
    def run(command, **kwargs):
        if error:
            raise error
        return SimpleNamespace(stdout=stdout, returncode=returncode)
    return run


def test_nvidia_smi_output_is_parsed() -> None:
    line = "53, 33.18, 47, 1492, P4, Not Active, Active, Not Active, [N/A]\n"
    gpu = monitor.read_gpu(fake_run(line))
    assert gpu == {"gpu_c": 53.0, "gpu_w": 33.18, "gpu_util_pct": 47.0, "gpu_sm_mhz": 1492.0, "gpu_pstate": "P4",
                   "gpu_thermal_slowdown": 1, "gpu_power_brake": 0, "gpu_power_cap": None}
    assert monitor.read_gpu(fake_run("53, 33\n")) is None                       # wrong field count
    assert monitor.read_gpu(fake_run("", returncode=9)) is None
    assert monitor.read_gpu(fake_run(error=FileNotFoundError())) is None
    assert monitor.read_gpu(fake_run(error=subprocess.TimeoutExpired("nvidia-smi", 10))) is None


def test_the_drive_holding_a_path_is_found_from_findmnt() -> None:
    assert monitor.controller_for_path(Path("/e/x"), fake_run("/dev/nvme2n1p2\n")) == "nvme2"
    assert monitor.controller_for_path(Path("/e/x"), fake_run("/dev/sda1\n")) is None
    assert monitor.controller_for_path(Path("/e/x"), fake_run(error=FileNotFoundError())) is None


def test_the_line_marks_watched_drives_and_flags(hwmon) -> None:
    sensors = monitor.discover(hwmon)
    sensors.watched_controllers = {"nvme1"}
    sample = {"time": "2026-10-23 15:52:22", "gpu_c": 50.0, "gpu_w": 15.9, "gpu_util_pct": 2.0, "gpu_sm_mhz": 660.0,
              "gpu_pstate": "P8", "cpu_package_c": 59.0, "cpu_hottest_core_c": 56.0, "load_1min": 0.45,
              "nvme0_c": 38.85, "nvme1_c": 51.85, "fan1_acpi_fan_rpm": 2420.0}
    line = monitor.format_line(sample, sensors, ["something is hot"])
    assert line.startswith("15:52:22 | GPU 50C 16W 2% 660MHz P8 |")
    assert "NVMe nvme0 39C nvme1* 52C" in line and "fans 2420 rpm" in line and "\n    !! something is hot" in line
    assert "GPU n/a" in monitor.format_line({"time": "2026-10-23 15:52:22"}, sensors, [])


def test_the_summary_reports_min_mean_max_and_warning_counts() -> None:
    summary = monitor.Summary()
    summary.add({"time": "t1", "gpu_c": 50.0, "gpu_thermal_slowdown": 0, "gpu_pstate": "P8"}, [])
    summary.add({"time": "t2", "gpu_c": 70.0, "gpu_thermal_slowdown": 1}, ["x"])
    text = "\n".join(summary.lines())
    assert "2 samples from t1 to t2" in text and "Samples with a warning: 1" in text
    assert "thermal slowdown: 1" in text
    assert any(line.startswith("gpu_c") and line.split()[1:] == ["50.0", "60.0", "70.0"] for line in summary.lines())
    assert monitor.Summary().lines() == ["No samples were taken."]


class FakeClock:
    def __init__(self, on_sleep=None):
        self.now, self.on_sleep, self.sleeps = 0.0, on_sleep, 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps += 1
        self.now += seconds
        if self.on_sleep:
            self.on_sleep(self.sleeps)


def run_main(hwmon, tmp_path, argv, clock, gpu=None):
    lines = []
    sensors = monitor.discover(hwmon)
    code = monitor.main(argv, sensors=sensors, gpu_reader=lambda: gpu, clock=clock, out=lines.append)
    return code, lines


def test_the_run_logs_each_sample_and_stops_after_the_duration(hwmon, tmp_path) -> None:
    log = tmp_path / "logs" / "t.csv"
    clock = FakeClock()
    code, lines = run_main(hwmon, tmp_path, ["--interval", "10", "--duration", "30", "--log", str(log)], clock,
                           gpu={"gpu_c": 50.0, "gpu_thermal_slowdown": 1})
    assert code == 0 and clock.sleeps == 3                      # samples at 0, 10, 20 and 30 seconds
    rows = list(csv.DictReader(log.open()))
    assert len(rows) == 4 and rows[0]["gpu_c"] == "50.0" and "thermal slowdown" in rows[0]["warnings"]
    assert rows[0]["cpu_package_c"] == "67.0"
    assert any("4 samples" in line for line in lines) and any("Sampling every 10 s" in line for line in lines)


def test_ctrl_c_stops_the_run_and_still_prints_the_summary(hwmon, tmp_path) -> None:
    log = tmp_path / "ctrlc.csv"
    previous = signal.getsignal(signal.SIGINT)
    clock = FakeClock(on_sleep=lambda n: os.kill(os.getpid(), signal.SIGINT) if n == 2 else None)
    code, lines = run_main(hwmon, tmp_path, ["--interval", "1", "--log", str(log)], clock, gpu=None)
    assert code == 0 and any("3 samples" in line or "2 samples" in line for line in lines)
    assert signal.getsignal(signal.SIGINT) == previous                    # the handler is restored
    assert len(list(csv.DictReader(log.open()))) >= 2


def test_a_bad_interval_is_refused_and_importing_starts_nothing(hwmon, tmp_path) -> None:
    with pytest.raises(SystemExit):
        monitor.main(["--interval", "0"], sensors=monitor.discover(hwmon))
    assert monitor.INTERVAL_SECONDS > 0 and monitor.WARN_MARGIN_C > 0
