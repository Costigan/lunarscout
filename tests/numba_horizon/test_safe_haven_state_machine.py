"""The safe-haven state machine against an independent offline implementation.

The offline oracle builds the whole shadow and communications timelines first
and then walks intervals, the way a list-based implementation would.  The
streaming machine in ``safe_haven.py`` sees one sample at a time.  They must
agree exactly on random timelines, including flicker, one-sample gaps, hundreds
of month boundaries and permanent shadow.  See
``docs/safe-haven-state-machine.md`` for the definitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunarscout._numba_horizon.safe_haven import (
    OPERATING_FUZZ,
    reduce_safe_haven_products_patch_stream,
)

SUN_THRESHOLD = 0.2
EARTH_THRESHOLD = 2.0
SHADOW, PARTIAL, FULL = 0.1, 0.5, 1.0
BELOW, ABOVE = 1.0, 3.0


def offline_reference(fraction, margin, band, *, step, merge_gap_hours, band_count,
                      operating=FULL):
    """Hibernation, joint-total and joint-longest hours for one pixel's timeline."""
    count = len(fraction)
    shadow = fraction < SUN_THRESHOLD
    outage = margin < EARTH_THRESHOLD

    # Shadow runs: shadow samples whose sunlit gap is shorter than the merge gap.
    runs: list[list[int]] = []
    for index in np.flatnonzero(shadow):
        if runs:
            gap = index - runs[-1][1] - 1
            if gap == 0 or gap * step < merge_gap_hours:
                runs[-1][1] = int(index)
                continue
        runs.append([int(index), int(index)])

    # Hibernations: maximal outage intervals, scored by the longest run touching them.
    hibernation = np.full(band_count, np.nan)
    start = None
    for index in range(count + 1):
        active = index < count and bool(outage[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            first, last = start, index - 1
            start = None
            if first == 0 or band[first] < 0:
                continue  # start unknown, or outside every band
            score = max(
                (b - a + 1 for a, b in runs if a <= last and b >= first), default=0
            )
            current = hibernation[band[first]]
            hibernation[band[first]] = score if np.isnan(current) else min(current, score)

    # Joint operating: Earth available and the Sun fully visible.
    joint = (~outage) & (fraction >= operating - OPERATING_FUZZ)
    total = np.zeros(band_count)
    longest = np.zeros(band_count)
    run_start = None
    for index in range(count + 1):
        active = index < count and bool(joint[index])
        if active:
            if band[index] >= 0:
                total[band[index]] += 1
            if run_start is None:
                run_start = index
            if band[run_start] >= 0:
                longest[band[run_start]] = max(longest[band[run_start]], index - run_start + 1)
        else:
            run_start = None
    return hibernation * step, total * step, longest * step


def random_timelines(rng, pixels, count, band_count):
    """Markov timelines with a wide spread of run, gap and outage lengths."""
    shadow_stay = rng.uniform(0.3, 0.99, pixels)
    lit_stay = rng.uniform(0.3, 0.99, pixels)
    outage_stay = rng.uniform(0.3, 0.99, pixels)
    visible_stay = rng.uniform(0.3, 0.99, pixels)
    fraction = np.empty((count, pixels), dtype=np.float32)
    margin = np.empty((count, pixels), dtype=np.float32)
    in_shadow = rng.random(pixels) < 0.5
    in_outage = rng.random(pixels) < 0.5
    for t in range(count):
        flip = rng.random(pixels)
        in_shadow = np.where(in_shadow, flip < shadow_stay, flip >= lit_stay)
        flip = rng.random(pixels)
        in_outage = np.where(in_outage, flip < outage_stay, flip >= visible_stay)
        lit_value = np.where(rng.random(pixels) < 0.5, FULL, PARTIAL)
        fraction[t] = np.where(in_shadow, SHADOW, lit_value)
        margin[t] = np.where(in_outage, BELOW, ABOVE)
    # Monotone bands with random boundaries, some samples outside every band.
    cuts = np.sort(rng.integers(0, count, band_count - 1))
    band = np.searchsorted(cuts, np.arange(count), side="right").astype(np.int32)
    if rng.random() < 0.5:
        band[: rng.integers(0, 4)] = -1
    if rng.random() < 0.5:
        band[count - rng.integers(0, 4):] = -1
    return fraction, margin, band


def run_machine(fraction, margin, band, *, step, merge_gap_hours, band_count,
                use_numba=True):
    count, pixels = fraction.shape
    return reduce_safe_haven_products_patch_stream(
        (fraction[t].reshape(pixels, 1) for t in range(count)),
        (margin[t].reshape(pixels, 1) for t in range(count)),
        count,
        tuple((None, None) for _ in range(band_count)),
        month_index_of=band,
        sunlight_threshold=SUN_THRESHOLD,
        earth_threshold_deg=EARTH_THRESHOLD,
        operating_sunlight_fraction=FULL,
        merge_gap_hours=merge_gap_hours,
        time_step_hours=step,
        use_numba=use_numba,
    )


@pytest.mark.parametrize("step", [6.0, 1.0])
@pytest.mark.parametrize("merge_gap_hours", [0.0, 1.0, 12.0, 24.0, 100.0])
def test_state_machine_matches_the_offline_reference(step, merge_gap_hours) -> None:
    rng = np.random.default_rng(int(step * 1000 + merge_gap_hours))
    band_count = 7
    fraction, margin, band = random_timelines(rng, 400, 260, band_count)

    hibernation, total, longest = run_machine(
        fraction, margin, band, step=step, merge_gap_hours=merge_gap_hours,
        band_count=band_count,
    )

    for pixel in range(fraction.shape[1]):
        expected = offline_reference(
            fraction[:, pixel], margin[:, pixel], band, step=step,
            merge_gap_hours=merge_gap_hours, band_count=band_count,
        )
        for name, layers, want in zip(
            ("hibernation", "joint_total", "joint_longest"),
            (hibernation, total, longest),
            expected,
        ):
            got = np.array([layers[b][pixel, 0] for b in range(band_count)])
            np.testing.assert_allclose(
                got, want, err_msg=f"{name}, pixel {pixel}", equal_nan=True, rtol=0, atol=1e-5
            )


def test_the_plain_python_machine_matches_the_compiled_one() -> None:
    rng = np.random.default_rng(7)
    fraction, margin, band = random_timelines(rng, 40, 200, 5)
    kwargs = dict(step=6.0, merge_gap_hours=24.0, band_count=5)

    compiled = run_machine(fraction, margin, band, **kwargs)
    plain = run_machine(fraction, margin, band, use_numba=False, **kwargs)

    for layers_a, layers_b in zip(compiled, plain):
        np.testing.assert_array_equal(np.stack(layers_a), np.stack(layers_b))


def timeline(spec: str):
    """'.' sunlit and visible, 'S' shadow, 'o' outage, 'X' shadow and outage."""
    fraction = np.array([SHADOW if c in "SX" else FULL for c in spec], dtype=np.float32)
    margin = np.array([BELOW if c in "oX" else ABOVE for c in spec], dtype=np.float32)
    return fraction, margin


NAN = float("nan")


def check(spec, bands, expected, *, merge_gap_hours=24.0, step=6.0):
    """The machine and the offline reference both give ``expected`` hibernation hours."""
    fraction, margin = timeline(spec)
    band = np.zeros(len(spec), dtype=np.int32) if bands is None else np.asarray(bands, dtype=np.int32)
    assert len(band) == len(spec)
    band_count = int(band.max()) + 1
    machine = run_machine(
        fraction.reshape(-1, 1), margin.reshape(-1, 1), band, step=step,
        merge_gap_hours=merge_gap_hours, band_count=band_count,
    )[0]
    got = [float(layer[0, 0]) for layer in machine]
    reference = offline_reference(
        fraction, margin, band, step=step, merge_gap_hours=merge_gap_hours,
        band_count=band_count,
    )[0]
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(reference, expected)


def test_sunny_outage_scores_zero_and_a_month_without_a_start_is_nodata() -> None:
    check("........" "..oooo..", [0] * 8 + [1] * 8, [NAN, 0.0])


def test_the_score_goes_to_the_month_the_outage_starts() -> None:
    # The outage starts in band 0; its shadow begins in band 1.  Then a second,
    # sunny outage starts in band 1, after the shadow run has ended.
    check("..oooooo" "XXX.o...", [0] * 8 + [1] * 8, [18.0, 0.0])


def test_a_run_crossing_a_month_boundary_is_credited_once() -> None:
    check("...XXXXX" "XX......", [0] * 8 + [1] * 8, [42.0, NAN])


def test_two_starts_in_a_month_report_the_better_one() -> None:
    # 24 h of shadow on the first outage, 72 h on the second: the month reports 24.
    check("..XXXX" "....." "XXXXXXXXXXXX" "..", None, [24.0])
    # A sunny first outage beats a 72 h second one: the month reports 0.
    check("..oo......" "XXXXXXXXXXXX" "..", None, [0.0])


def test_short_sunlit_gaps_are_bridged_into_one_run() -> None:
    # 3 sunlit samples (18 h < 24 h) join two outages to one 8-sample run: both
    # outages score 48 h.
    check("..XX" "..." "XXX" "..", None, [48.0])
    # An outage inside the bridged gap belongs to the merged run too.
    check("..XX" "oo." "XXX" "..", None, [48.0])
    # 4 sunlit samples (24 h is not less than 24 h): two runs, scores 12 h and
    # 36 h, and the month reports the better one.
    check("..XX" "...." "XXXXXX" "..", None, [12.0])
    # Without merging, the same 3-sample gap leaves two runs: 12 h and 18 h.
    check("..XX" "..." "XXX" "..", None, [12.0], merge_gap_hours=0.0)


def test_a_hibernation_in_progress_at_the_first_sample_is_not_credited() -> None:
    check("XXXX.." "..oo..", [0] * 6 + [1] * 6, [NAN, 0.0])


def test_permanent_shadow_gives_every_month_the_whole_run() -> None:
    spec = "SSS" + "XXXX" + "S" * 8 + "XXXX" + "S" * 8 + "XXXX" + "SSS"
    bands = [0] * 10 + [1] * 14 + [2] * (len(spec) - 24)
    check(spec, bands, [len(spec) * 6.0] * 3)


def test_joint_operating_layers_count_totals_by_sample_and_runs_by_start() -> None:
    #           . . S . . . o . .   joint = visible Earth and a fully lit Sun
    fraction = np.array([1, 1, 0.1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    margin = np.array([3, 3, 3, 3, 3, 3, 1, 3, 3], dtype=np.float32)
    band = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    _, total, longest = run_machine(
        fraction.reshape(-1, 1), margin.reshape(-1, 1), band, step=6.0,
        merge_gap_hours=24.0, band_count=2,
    )
    assert [float(layer[0, 0]) for layer in total] == [24.0, 18.0]
    # The run at samples 3-5 starts in band 0 and spans the boundary: 18 h to band 0.
    assert [float(layer[0, 0]) for layer in longest] == [18.0, 12.0]


def test_operating_threshold_has_a_small_tolerance_for_float_rounding() -> None:
    fraction = np.array([1.0 - 5e-7, 1.0 - 2e-6, 0.5], dtype=np.float32).reshape(-1, 1)
    margin = np.full((3, 1), ABOVE, dtype=np.float32)
    _, total, _ = run_machine(
        fraction, margin, np.zeros(3, dtype=np.int32), step=6.0,
        merge_gap_hours=24.0, band_count=1,
    )
    assert float(total[0][0, 0]) == 6.0


def test_invalid_dem_pixels_are_skipped_but_other_nonfinite_signals_raise() -> None:
    # Pixel 1 sits on an invalid DEM node: its signals are NaN throughout.
    fraction = np.array([[0.1, np.nan]] * 4, dtype=np.float32)
    margin = np.array([[3.0, np.nan], [1.0, np.nan], [1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    kwargs = dict(
        month_index_of=np.zeros(4, dtype=np.int32), sunlight_threshold=SUN_THRESHOLD,
        earth_threshold_deg=EARTH_THRESHOLD, operating_sunlight_fraction=FULL,
        merge_gap_hours=24.0, time_step_hours=6.0,
    )

    def run(invalid):
        return reduce_safe_haven_products_patch_stream(
            (fraction[t].reshape(1, 2) for t in range(4)),
            (margin[t].reshape(1, 2) for t in range(4)),
            4, ((None, None),), invalid_pixels=invalid, **kwargs,
        )

    hibernation, total, longest = run(np.array([[False, True]]))
    assert hibernation[0][0, 0] == 24.0            # the valid pixel: one 4-sample run
    assert np.isnan(hibernation[0][0, 1])          # skipped: no hibernation, no value
    assert total[0][0, 1] == 0.0 and longest[0][0, 1] == 0.0
    with pytest.raises(ValueError, match="finite"):
        run(None)                                  # NaN with no mask is still an error
    with pytest.raises(ValueError, match="finite"):
        run(np.array([[True, False]]))             # NaN at a pixel not marked invalid
