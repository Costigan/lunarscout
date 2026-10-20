"""The degenerate-outage rule (docs/safe-haven-state-machine.md, section 2.1).

An Earth cycle with no outage gets a one-sample outage at the minimum of Earth's
elevation above a flat horizon, provided the margin above the real horizon stays at
or above the threshold for W samples either side.  The machine runs W samples behind
the data to decide that; these tests compare it with an offline implementation of the
rule as stated, on random smooth timelines, and check the edges by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunarscout._numba_horizon.safe_haven import (
    MAX_DEGENERATE_WINDOW,
    reduce_safe_haven_products_batches,
    reduce_safe_haven_products_patch_stream,
)
from test_safe_haven_state_machine import (
    EARTH_THRESHOLD,
    SUN_THRESHOLD,
    hibernation_values,
    joint_operating_values,
    random_timelines,
)

STEP = 6.0


# ---------------------------------------------------------------------------
# The offline reference: the rule as stated, on whole timelines.
# ---------------------------------------------------------------------------


def degenerate_outages(flat, margin, window):
    """Rule: sample k is a degenerate outage when Earth's flat-horizon elevation has a
    minimum there (lower than the sample before, no higher than the sample after), the
    margin is at or above the threshold on every sample k - W .. k + W, and the whole
    window lies in the series."""
    flags = np.zeros(len(flat), dtype=bool)
    if window == 0:
        return flags
    for k in range(window, len(flat) - window):
        is_minimum = flat[k] < flat[k - 1] and flat[k] <= flat[k + 1]
        clear = all(not (margin[j] < EARTH_THRESHOLD) for j in range(k - window, k + window + 1))
        flags[k] = is_minimum and clear
    return flags


def reference(fraction, flat, margin, band, *, window, merge_gap_hours=24.0, band_count):
    is_shadow = fraction < SUN_THRESHOLD
    is_outage = (margin < EARTH_THRESHOLD) | degenerate_outages(flat, margin, window)
    hibernation = hibernation_values(is_shadow, is_outage, band, band_count, STEP, merge_gap_hours)
    total, longest = joint_operating_values(is_outage, fraction, band, band_count, STEP, 1.0)
    return hibernation, total, longest


def smooth_timelines(rng, pixels, count, band_count):
    """Earth elevation as a smooth cycle per pixel (sometimes quantised, to make ties),
    a horizon that hides Earth at some pixels, and Markov Sun and band arrays."""
    fraction, _unused, band = random_timelines(rng, pixels, count, band_count)
    time = np.arange(count)[:, None]
    period = rng.uniform(12, 70, pixels)
    flat = rng.uniform(3, 12, pixels) * np.sin(2 * np.pi * time / period + rng.uniform(0, 6.3, pixels))
    flat = flat + rng.uniform(-4, 8, pixels)
    quantised = rng.random(pixels) < 0.3
    flat = np.where(quantised, np.round(flat * 2) / 2, flat).astype(np.float32)
    horizon = rng.uniform(-2, 7, pixels) + rng.uniform(0, 1.5, pixels) * np.sin(time / 5.0)
    margin = (flat - horizon).astype(np.float32)
    return fraction, flat, margin, band


def run(fraction, flat, margin, band, *, window, band_count, merge_gap_hours=24.0,
        use_numba=True, batch_size=32):
    count, pixels = fraction.shape

    def tiles(array):
        return (array[t].reshape(pixels, 1) for t in range(count))

    return reduce_safe_haven_products_patch_stream(
        tiles(fraction), tiles(margin), count, tuple((None, None) for _ in range(band_count)),
        month_index_of=band, sunlight_threshold=SUN_THRESHOLD,
        earth_threshold_deg=EARTH_THRESHOLD, operating_sunlight_fraction=1.0,
        merge_gap_hours=merge_gap_hours, time_step_hours=STEP, use_numba=use_numba,
        batch_size=batch_size, earth_flat_elevation_tiles=tiles(flat),
        degenerate_window_samples=window,
    )


def assert_matches(result, expected, pixel):
    for got, want in zip(result, expected):
        got_values = np.array([layer[pixel, 0] for layer in got], dtype=np.float64)
        np.testing.assert_array_equal(got_values, want)


@pytest.mark.parametrize("window", [0, 1, 2, 4, 7])
def test_the_machine_matches_the_offline_rule_on_smooth_random_timelines(window) -> None:
    band_count = 6
    rng = np.random.default_rng(100 + window)
    fraction, flat, margin, band = smooth_timelines(rng, 220, 260, band_count)
    triggered = sum(
        int(degenerate_outages(flat[:, p], margin[:, p], max(window, 1)).any())
        for p in range(fraction.shape[1])
    )

    result = run(fraction, flat, margin, band, window=window, band_count=band_count)

    for pixel in range(fraction.shape[1]):
        assert_matches(result, reference(
            fraction[:, pixel], flat[:, pixel], margin[:, pixel], band,
            window=window, band_count=band_count,
        ), pixel)
    assert triggered > 20          # the scenes do exercise the rule


def test_the_plain_python_machine_matches_the_compiled_one_with_the_rule() -> None:
    band_count = 4
    fraction, flat, margin, band = smooth_timelines(np.random.default_rng(5), 12, 120, band_count)
    compiled = run(fraction, flat, margin, band, window=3, band_count=band_count)
    plain = run(fraction, flat, margin, band, window=3, band_count=band_count, use_numba=False)
    for left, right in zip(compiled, plain):
        for band_left, band_right in zip(left, right):
            np.testing.assert_array_equal(band_left, band_right)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 32, 500])
def test_the_result_does_not_depend_on_how_time_steps_are_batched(batch_size) -> None:
    band_count = 5
    fraction, flat, margin, band = smooth_timelines(np.random.default_rng(9), 40, 150, band_count)
    whole = run(fraction, flat, margin, band, window=4, band_count=band_count, batch_size=500)
    batched = run(fraction, flat, margin, band, window=4, band_count=band_count, batch_size=batch_size)
    for left, right in zip(whole, batched):
        for band_left, band_right in zip(left, right):
            np.testing.assert_array_equal(band_left, band_right)


# ---------------------------------------------------------------------------
# By hand: one pixel, one band, 40 samples of 6 hours.
# ---------------------------------------------------------------------------


COUNT = 40


def timeline(minimum_at=20, depth=5.0):
    """Earth's flat elevation has its minimum at ``minimum_at``; the margin equals it."""
    flat = (depth + 0.01 * (np.arange(COUNT) - minimum_at) ** 2).astype(np.float32)
    return np.ones(COUNT, dtype=np.float32), flat, flat.copy()


def hand(fraction, flat, margin, *, window=4):
    band = np.zeros(COUNT, dtype=np.int32)
    result = run(fraction[:, None], flat[:, None], margin[:, None], band, window=window, band_count=1)
    return tuple(float(layer[0][0, 0]) for layer in result)


def test_a_sunlit_cycle_without_an_outage_gets_a_zero_hour_hibernation() -> None:
    fraction, flat, margin = timeline()
    hibernation, total, longest = hand(fraction, flat, margin)
    assert hibernation == 0.0
    # Earth is unavailable at the one outage sample: 39 of 40 samples operate, and the
    # longest unbroken stretch is the 20 samples before it.
    assert total == 39 * STEP and longest == 20 * STEP


def test_the_rule_can_be_switched_off() -> None:
    fraction, flat, margin = timeline()
    hibernation, total, longest = hand(fraction, flat, margin, window=0)
    assert np.isnan(hibernation) and total == 40 * STEP and longest == 40 * STEP


def test_shadow_on_the_minimum_scores_the_whole_run() -> None:
    fraction, flat, margin = timeline()
    fraction[18:24] = 0.1                  # six shadow samples around the minimum
    assert hand(fraction, flat, margin)[0] == 6 * STEP


def test_a_real_outage_within_the_window_means_no_degenerate_outage() -> None:
    fraction, flat, margin = timeline()
    margin[24] = 1.0                       # a real outage 4 samples from the minimum
    fraction[24] = 0.1                     # touched by a one-sample shadow run
    assert hand(fraction, flat, margin, window=4)[0] == STEP          # only the real outage
    margin[24] = 3.0
    margin[25] = 1.0                       # five samples away: outside a window of 4
    fraction[24] = 1.0
    fraction[25] = 0.1
    assert hand(fraction, flat, margin, window=4)[0] == 0.0           # the degenerate one is better
    assert hand(fraction, flat, margin, window=5)[0] == STEP          # window 5 reaches it


def test_of_two_equal_minimum_samples_the_first_is_the_outage() -> None:
    fraction, flat, margin = timeline()
    flat[21] = flat[20]                    # equal adjacent minima at 20 and 21
    margin[:] = flat
    fraction[21] = 0.1                     # shadow on the second only
    assert hand(fraction, flat, margin)[0] == 0.0
    fraction[21], fraction[20] = 1.0, 0.1  # shadow on the first
    assert hand(fraction, flat, margin)[0] == STEP


def test_a_minimum_within_the_window_of_either_end_is_not_credited() -> None:
    for position, credited in ((3, False), (4, True), (35, True), (36, False)):
        fraction, flat, margin = timeline(minimum_at=position)
        assert bool(np.isnan(hand(fraction, flat, margin)[0])) == (not credited), position


def test_the_last_samples_are_flushed_with_their_real_outage_flags() -> None:
    fraction = np.ones(COUNT, dtype=np.float32)
    flat = np.linspace(10, 30, COUNT).astype(np.float32)            # no minimum at all
    margin = flat.copy()
    margin[-2:] = 1.0                      # a real outage in the last two samples
    fraction[-1] = 0.1
    plain = hand(fraction, flat, margin, window=0)
    delayed = hand(fraction, flat, margin, window=4)
    assert plain == delayed and delayed[0] == STEP


def test_a_series_shorter_than_the_window_still_works() -> None:
    fraction = np.ones((3, 1), dtype=np.float32)
    flat = np.array([[5.0], [4.0], [6.0]], dtype=np.float32)
    margin = np.array([[5.0], [4.0], [1.0]], dtype=np.float32)
    band = np.zeros(3, dtype=np.int32)
    for window in (0, 1, 2, 5):
        result = run(fraction, flat, margin, band, window=window, band_count=1)
        assert float(result[0][0][0, 0]) == 0.0         # the real outage at sample 2, unshadowed
        assert float(result[1][0][0, 0]) == 2 * STEP


def test_the_window_and_its_inputs_are_checked() -> None:
    fraction, flat, margin = timeline()
    band = np.zeros(COUNT, dtype=np.int32)
    with pytest.raises(ValueError, match="degenerate_window_samples"):
        run(fraction[:, None], flat[:, None], margin[:, None], band, window=MAX_DEGENERATE_WINDOW + 1, band_count=1)
    with pytest.raises(ValueError, match="degenerate_window_samples"):
        run(fraction[:, None], flat[:, None], margin[:, None], band, window=-1, band_count=1)
    with pytest.raises(ValueError, match="needs earth_flat_elevation_batches"):
        reduce_safe_haven_products_batches(
            [fraction.reshape(COUNT, 1, 1)], [margin.reshape(COUNT, 1, 1)], COUNT,
            ((None, None),), month_index_of=band, sunlight_threshold=SUN_THRESHOLD,
            earth_threshold_deg=EARTH_THRESHOLD, operating_sunlight_fraction=1.0,
            merge_gap_hours=24.0, time_step_hours=STEP, degenerate_window_samples=2,
        )
    nan_flat = flat.copy()
    nan_flat[5] = np.nan
    with pytest.raises(ValueError, match="flat-horizon"):
        run(fraction[:, None], nan_flat[:, None], margin[:, None], band, window=2, band_count=1)


@pytest.mark.parametrize("count", [1, 2, 3, 5, 9])
def test_when_no_cycle_minimum_exists_any_window_gives_the_plain_machine_on_short_series(count) -> None:
    # Strictly rising elevation has no minimum, so the delay must change nothing: this
    # exercises series shorter than, and equal to, the window, and the flush at the end.
    rng = np.random.default_rng(count)
    pixels = 300
    fraction = rng.choice([0.1, 0.5, 1.0], size=(count, pixels)).astype(np.float32)
    margin = rng.choice([1.0, 3.0], size=(count, pixels)).astype(np.float32)
    flat = np.tile(np.arange(count, dtype=np.float32)[:, None] + 10.0, (1, pixels))
    band = np.zeros(count, dtype=np.int32)
    plain = run(fraction, flat, margin, band, window=0, band_count=1, merge_gap_hours=12.0)
    for window in (1, 2, 4, 8, MAX_DEGENERATE_WINDOW):
        delayed = run(fraction, flat, margin, band, window=window, band_count=1, merge_gap_hours=12.0)
        for left, right in zip(plain, delayed):
            np.testing.assert_array_equal(left[0], right[0])
