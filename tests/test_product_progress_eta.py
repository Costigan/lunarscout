"""The verbose progress line of the product functions carries a rate and an ETA.

The rate counts only the patches this run finished, so it is right after a resume,
and it follows recent speed, so a slow start does not stay in the estimate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunarscout.products import _ProgressAdapter

WALL_START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clocks:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def wall(self):
        from datetime import timedelta

        return WALL_START + timedelta(seconds=self.now)


def make(clocks, **options):
    return _ProgressAdapter(
        "safe_havens", Path("out.tif"), verbose=options.pop("verbose", True),
        progress_callback=None, progress_event_callback=options.pop("event", None),
        clock=clocks.monotonic, wall_clock=clocks.wall,
    )


def event(state, completed, total=2000):
    return SimpleNamespace(
        state=state, completed_patches=completed, total_patches=total,
        tile_y=None, tile_x=None, backend="cpu", invalid_patches=0,
    )


def lines(capsys):
    return capsys.readouterr().out.splitlines()


def test_a_resumed_run_reports_the_rate_of_this_run_not_of_its_history(capsys) -> None:
    clocks = Clocks()
    adapter = make(clocks)
    adapter(event("start", 1000))
    for step in range(1, 11):
        clocks.now = 2.0 * step                 # one patch every 2 seconds
        adapter(event("valid", 1000 + step))
    last = lines(capsys)[-1]
    # 990 patches remain at 2 s each: 1,980 s (33 minutes) after the clock's 20 s.
    # Dividing the elapsed time by all 1,010 completed patches would say 0.02 s each.
    assert last == (
        "safe_havens: valid 1010/2000 | 2.00 s/patch | "
        "ETA 2026-01-01 12:33:20 (33m left)"
    )


def test_the_eta_waits_for_enough_samples(capsys) -> None:
    clocks = Clocks()
    adapter = make(clocks)
    adapter(event("start", 0))
    for step in range(1, 4):
        clocks.now = float(step)
        adapter(event("valid", step))
    assert lines(capsys)[-1] == "safe_havens: valid 3/2000 | ETA pending"
    clocks.now = 4.0
    adapter(event("valid", 4))
    assert "ETA pending" not in lines(capsys)[-1]


def test_slow_work_is_shown_in_seconds_per_patch(capsys) -> None:
    clocks = Clocks()
    adapter = make(clocks)
    adapter(event("start", 0))
    for step in range(1, 8):
        clocks.now = 4.0 * step
        adapter(event("valid", step))
    assert "| 4.00 s/patch |" in lines(capsys)[-1]


def test_the_estimate_follows_recent_speed_and_forgets_a_slow_start(capsys) -> None:
    clocks = Clocks()
    adapter = make(clocks)
    adapter(event("start", 0))
    completed = 0
    for _ in range(20):                         # slow start: 10 s per patch
        clocks.now += 10.0
        completed += 1
        adapter(event("valid", completed, 5000))
    for _ in range(1100):                       # then 0.5 s per patch
        clocks.now += 0.5
        completed += 1
        adapter(event("valid", completed, 5000))
    assert "| 2.00 patches/s |" in lines(capsys)[-1]


def test_invalid_patches_count_and_the_complete_line_has_no_eta(capsys) -> None:
    clocks = Clocks()
    adapter = make(clocks)
    adapter(event("start", 0, 10))
    for step in range(1, 7):
        clocks.now = float(step)
        adapter(event("invalid" if step == 6 else "valid", step, 10))
    clocks.now = 10.0
    adapter(event("complete", 10, 10))
    output = lines(capsys)
    invalid = [line for line in output if line.startswith("safe_havens: invalid 6/10")]
    assert len(invalid) == 1 and "| 1.00 patches/s | ETA " in invalid[0]
    assert output[-1] == "safe_havens: complete 10/10 (0 invalid patches)"


def test_nothing_is_printed_unless_verbose_and_callbacks_still_run(capsys) -> None:
    clocks = Clocks()
    seen = []
    adapter = make(clocks, verbose=False, event=seen.append)
    adapter(event("start", 0))
    adapter(event("valid", 1))
    assert capsys.readouterr().out == ""
    assert [item.stage for item in seen] == ["start", "valid"]


def test_the_public_safe_haven_function_prints_the_timing(tmp_path, capsys) -> None:
    import numpy as np

    import lunarscout as ls
    from test_public_safe_haven_products import _TIMES, _SUN, _EARTH, _inputs, _vectors

    dem_path, horizons_path = _inputs(tmp_path, np.zeros((1, 1)))
    ls.generate_safe_haven_products(
        dem_path, horizons_path, tmp_path / "out", times=_TIMES,
        sun_vectors_m=_vectors(_SUN), earth_vectors_m=_vectors(_EARTH),
        backend="cpu", verbose=True,
    )
    output = capsys.readouterr().out.splitlines()
    assert any(line.startswith("safe_havens: valid 1/1 | ETA pending") for line in output)
    assert output[-1].startswith("safe_havens: complete 1/1")


def test_durations_round_to_the_nearest_minute_and_carry_into_hours() -> None:
    from lunarscout.products import _format_duration

    assert _format_duration(20.0) == "20s"
    assert _format_duration(31.0) == "1m"           # rounds up from half a minute
    assert _format_duration(90.0) == "2m"
    assert _format_duration(3590.0) == "1h 00m"     # 59.8 minutes is an hour
    assert _format_duration(3.1 * 3600) == "3h 06m"
    assert _format_duration(0.0) == "0s"


def test_invalid_patches_also_set_the_rate(capsys) -> None:
    clocks = Clocks()
    adapter = make(clocks)
    adapter(event("start", 0, 100))
    for step in range(1, 9):
        clocks.now = float(step)
        adapter(event("invalid", step, 100))
    assert "| 1.00 patches/s | ETA " in lines(capsys)[-1]
