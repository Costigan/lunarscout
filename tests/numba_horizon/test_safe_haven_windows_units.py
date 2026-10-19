"""Pure rules of stage-2 pass 1 (docs/safe-haven-windows-spec.md), without any horizon data."""

from __future__ import annotations

import numpy as np
import pytest

from lunarscout._numba_horizon.safe_haven import merge_gap_limit
from lunarscout._numba_horizon.safe_haven_windows import (
    FloodGrid,
    WindowParameters,
    box_sums,
    disk_offsets,
    flood_step_minutes,
    hibernation_of_record,
    nearby_haven_counts,
)
from test_safe_haven_state_machine import hibernation_values, random_timelines

SUN_THRESHOLD = 0.2
EARTH_THRESHOLD = 2.0


# --- the flood time grid -----------------------------------------------------


@pytest.mark.parametrize(
    ("pixel_m", "speed", "minutes"),
    [(20.0, 0.01, 32), (20.0, 0.0111111, 30), (20.0, 0.02, 16), (10.0, 0.01, 16)],
)
def test_the_flood_step_is_the_largest_divisor_of_a_day_not_above_the_crossing_time(
    pixel_m, speed, minutes
) -> None:
    assert flood_step_minutes(pixel_m, speed) == minutes
    assert 1440 % minutes == 0
    assert minutes <= pixel_m / speed / 60.0 + 1e-6


def test_a_rover_that_crosses_a_pixel_in_under_a_minute_has_no_flood_step() -> None:
    with pytest.raises(ValueError):
        flood_step_minutes(1.0, 1.0)


def test_the_flood_grid_rounds_toward_the_hibernation_correctly() -> None:
    grid = FloodGrid(30)
    step = grid.step_us
    assert (grid.floor_index(5 * step), grid.ceil_index(5 * step)) == (5, 5)
    assert (grid.floor_index(5 * step + 1), grid.ceil_index(5 * step + 1)) == (5, 6)
    assert (grid.floor_index(-1), grid.ceil_index(-1)) == (-1, 0)
    assert grid.steps_per_chunk == 48
    assert grid.step_count(48.0) == 96
    assert FloodGrid(32).step_count(48.0) == 90
    assert str(grid.times([0, 48])[1]) == "1970-01-02T00:00:00.000000"
    with pytest.raises(ValueError):
        FloodGrid(7)


def test_invalid_parameters_are_rejected() -> None:
    for bad in (dict(connectivity=6), dict(min_nearby_haven_fraction=1.5),
                dict(survival_limit_hours=0.0), dict(max_seeds_per_cell_month=0)):
        with pytest.raises(ValueError):
            WindowParameters(**bad)


# --- neighbourhoods ------------------------------------------------------------


def test_a_100_m_disk_on_20_m_pixels_has_81_pixels() -> None:
    assert len(disk_offsets(100.0, 20.0)) == 81


def brute_force_disk(haven, radius_m, pixel_m):
    height, width = haven.shape
    offsets = disk_offsets(radius_m, pixel_m)
    found = np.zeros(haven.shape, dtype=int)
    inside = np.zeros(haven.shape, dtype=int)
    for row in range(height):
        for column in range(width):
            for dr, dc in offsets:
                r, c = row + dr, column + dc
                if 0 <= r < height and 0 <= c < width:
                    inside[row, column] += 1
                    found[row, column] += int(haven[r, c])
    return found, inside


@pytest.mark.parametrize("shape", [(3, 3), (17, 23), (30, 9)])
def test_nearby_counts_match_a_brute_force_disk_including_the_edges(shape) -> None:
    haven = np.random.default_rng(shape[0]).random(shape) < 0.3
    found, inside = nearby_haven_counts(haven, 100.0, 20.0)
    expected_found, expected_inside = brute_force_disk(haven, 100.0, 20.0)
    assert np.array_equal(found, expected_found)
    assert np.array_equal(inside, expected_inside)


def test_an_isolated_haven_pixel_has_a_low_fraction_and_a_cluster_a_high_one() -> None:
    haven = np.zeros((30, 30), dtype=bool)
    haven[5, 5] = True
    haven[15:25, 15:25] = True
    found, inside = nearby_haven_counts(haven, 100.0, 20.0)
    assert found[5, 5] / inside[5, 5] == pytest.approx(1 / 81)
    assert found[20, 20] / inside[20, 20] > 0.9
    assert inside[0, 0] < 81            # the corner sees fewer pixels inside the DEM


def test_box_sums_match_brute_force_and_treat_nan_as_zero() -> None:
    values = np.random.default_rng(3).random((11, 14))
    values[2, 3] = np.nan
    clean = np.nan_to_num(values)
    for half_width in (0, 1, 4, 30):
        expected = np.empty_like(clean)
        for row in range(clean.shape[0]):
            for column in range(clean.shape[1]):
                expected[row, column] = clean[
                    max(0, row - half_width):row + half_width + 1,
                    max(0, column - half_width):column + half_width + 1,
                ].sum()
        assert np.allclose(box_sums(values, half_width), expected)


# --- the hibernation of record ----------------------------------------------------


def span_arrays(fraction, margin):
    return fraction < SUN_THRESHOLD, margin < EARTH_THRESHOLD


@pytest.mark.parametrize("merge_gap_hours", [0.0, 12.0, 24.0, 100.0])
def test_the_hibernation_of_record_scores_like_the_stage_1_reference(merge_gap_hours) -> None:
    step, band_count = 6.0, 7
    rng = np.random.default_rng(int(merge_gap_hours) + 5)
    fraction, margin, band = random_timelines(rng, 150, 260, band_count)
    gap_limit = merge_gap_limit(merge_gap_hours, step)
    for pixel in range(fraction.shape[1]):
        is_shadow, is_outage = span_arrays(fraction[:, pixel], margin[:, pixel])
        expected = hibernation_values(is_shadow, is_outage, band, band_count, step, merge_gap_hours)
        for month in range(band_count):
            record = hibernation_of_record(
                is_shadow, is_outage, band, month, gap_limit=gap_limit, offset=0
            ).record
            if np.isnan(expected[month]):
                assert record is None
            else:
                assert record is not None
                assert record.score_samples * step == expected[month]
                # the record is the earliest hibernation with that score, and starts in the month
                assert band[record.outage_first] == month
                assert is_outage[record.outage_first] and not is_outage[record.outage_first - 1]


def test_a_short_span_gives_the_full_answer_whenever_it_reports_itself_closed() -> None:
    step, band_count, merge_gap_hours = 6.0, 5, 24.0
    gap_limit = merge_gap_limit(merge_gap_hours, step)
    rng = np.random.default_rng(11)
    fraction, margin, band = random_timelines(rng, 300, 200, band_count)
    closed = 0
    for pixel in range(fraction.shape[1]):
        is_shadow, is_outage = span_arrays(fraction[:, pixel], margin[:, pixel])
        for month in range(band_count):
            full = hibernation_of_record(
                is_shadow, is_outage, band, month, gap_limit=gap_limit, offset=0
            )
            members = np.flatnonzero(band == month)
            if members.size == 0:
                continue
            start = max(0, members[0] - 12)
            stop = min(len(band), members[-1] + 1 + 30)
            part = hibernation_of_record(
                is_shadow[start:stop], is_outage[start:stop], band[start:stop], month,
                gap_limit=gap_limit, offset=start,
            )
            if part.open_start or part.open_end:
                continue
            closed += 1
            assert part.record == full.record
    assert closed > 100


def test_a_hibernation_under_way_at_the_first_sample_is_not_credited() -> None:
    is_outage = np.array([True, True, False, False, True, False])
    is_shadow = np.zeros(6, dtype=bool)
    band = np.zeros(6, dtype=int)
    record = hibernation_of_record(is_shadow, is_outage, band, 0, gap_limit=4, offset=0).record
    assert (record.outage_first, record.outage_last, record.score_samples) == (4, 4, 0)


# --- exact boundaries of the closure rules -----------------------------------------------


def record_for(shadow_at, outage_at, count, *, gap_limit, offset=0, month=None):
    is_shadow = np.zeros(count, dtype=bool)
    is_outage = np.zeros(count, dtype=bool)
    is_shadow[list(shadow_at)] = True
    is_outage[list(outage_at)] = True
    months = np.zeros(count, dtype=int) if month is None else month
    return hibernation_of_record(is_shadow, is_outage, months, 0, gap_limit=gap_limit, offset=offset)


def test_a_run_is_closed_once_gap_limit_sunlit_samples_follow_it() -> None:
    # shadow at sample 3 touches the outage 2..3; the span ends 4 samples after it.
    closed = record_for([3], [2, 3], 8, gap_limit=4)       # samples 4..7 follow: 4 = gap_limit
    assert not closed.open_end
    assert record_for([3], [2, 3], 7, gap_limit=4).open_end   # only 3 follow: a shadow could join


def test_a_run_is_closed_at_the_start_once_gap_limit_sunlit_samples_precede_it() -> None:
    # The run starts at sample 4 and the span does not begin the series (offset > 0).
    assert not record_for([4], [4, 5], 12, gap_limit=4, offset=7).open_start    # 4 sunlit before
    assert record_for([3], [3, 4], 12, gap_limit=4, offset=7).open_start        # only 3 before
    assert not record_for([3], [3, 4], 12, gap_limit=4, offset=0).open_start    # the series starts here


def test_a_hibernation_that_reaches_the_last_sample_is_open() -> None:
    assert record_for([], [3, 4, 5], 6, gap_limit=4).open_end
    assert not record_for([], [3, 4, 5], 7, gap_limit=4).open_end


def test_of_two_hibernations_with_one_score_the_earlier_is_the_record() -> None:
    # Outages 2..3 and 8..9, each touched by a one-sample run, and both start in the month.
    found = record_for([2, 8], [2, 3, 8, 9], 14, gap_limit=2).record
    assert (found.outage_first, found.score_samples) == (2, 1)
    # A shorter run on the later one makes it the better opportunity.
    found = record_for([2, 3, 8], [2, 3, 8, 9], 14, gap_limit=2).record
    assert (found.outage_first, found.score_samples) == (8, 1)


def test_only_the_record_needs_a_known_start() -> None:
    # Two hibernations in the month. The first (outage 6..7) has a run that reaches back
    # to the start of the span, so its score is understated; the second (outage 20..21) has
    # a short run and is the real best. The span does not begin the series (offset > 0).
    shadow = list(range(0, 8)) + [20]
    found = record_for(shadow, [6, 7, 20, 21], 30, gap_limit=2, offset=10)
    assert found.record.outage_first == 30 and not found.open_start
    # With the short run on the first one instead, the record itself starts the span.
    found = record_for([0, 1, 20, 21, 22], [6, 7, 20, 21], 30, gap_limit=2, offset=10)
    assert found.record.outage_first == 16
    found = record_for(list(range(0, 8)), [6, 7, 20, 21], 30, gap_limit=2, offset=10)
    assert found.record.outage_first == 30 and found.record.score_samples == 0   # no run touches 20..21
    # The record is the one with a run that may reach back.
    found = record_for(list(range(0, 8)), [6, 7], 30, gap_limit=2, offset=10)
    assert found.record.outage_first == 16 and found.open_start


# --- choosing cells ------------------------------------------------------------------------


def test_a_coordinate_maps_to_its_cell() -> None:
    from lunarscout._numba_horizon.safe_haven_windows import cell_of_coordinate

    # The 20 m south-pole DEM: origin (-304000, 304000). Easting 71534, northing 117030 is
    # row 9348, column 18776 (checked against rasterio's index), inside cell (36, 73).
    transform = (-304000.0, 20.0, 0.0, 304000.0, 0.0, -20.0)
    assert cell_of_coordinate(transform, 71534.0, 117030.0, 256) == (36, 73)
    assert cell_of_coordinate(transform, -304000.0, 304000.0, 256) == (0, 0)
    assert cell_of_coordinate(transform, -304000.0 + 20.0 * 256 - 0.01, 304000.0, 256) == (0, 0)
    assert cell_of_coordinate(transform, -304000.0 + 20.0 * 256, 304000.0, 256) == (0, 1)
    with pytest.raises(ValueError):
        cell_of_coordinate(transform, -304001.0, 0.0, 256)
    with pytest.raises(ValueError):
        cell_of_coordinate((0, 20.0, 1.0, 0, 0, -20.0), 1.0, -1.0, 256)


def test_cells_around_are_clipped_to_the_raster() -> None:
    from lunarscout._numba_horizon.safe_haven_windows import cells_around

    kwargs = dict(height=600, width=300, cell_size=256)       # 3 cell rows, 2 cell columns
    assert cells_around((1, 1), 1, **kwargs) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert cells_around((0, 0), 0, **kwargs) == [(0, 0)]
    assert cells_around((2, 1), 1, **kwargs) == [(1, 0), (1, 1), (2, 0), (2, 1)]
