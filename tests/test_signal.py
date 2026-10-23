import numpy as np
import pytest

from pupilad.signal import (
    block_means,
    gap_length_at,
    interpolate_short_gaps,
    invalid_sample_mask,
    linear_slope,
    lowpass_ignoring_nan,
    pad_mask,
)


def test_invalid_sample_mask_flags_each_rule():
    confidence = np.array([0.9, 0.5, 0.9, 0.9, 0.9, 0.9])
    blink = np.array([0, 0, 1, 0, 0, 0])
    diameter = np.array([4.0, 4.0, 4.0, 1.0, 10.0, np.nan])
    mask = invalid_sample_mask(confidence, blink, diameter, 0.6, 1.5, 9.0)
    assert mask.tolist() == [False, True, True, True, True, True]


def test_pad_mask_extends_artifacts_by_pad_on_both_sides():
    times = np.arange(0, 1.0, 0.04)
    mask = np.zeros(len(times), dtype=bool)
    mask[10] = True
    padded = pad_mask(times, mask, 0.1)
    assert np.flatnonzero(padded).tolist() == [8, 9, 10, 11, 12]


def test_pad_mask_without_artifacts_is_unchanged():
    times = np.arange(5, dtype=float)
    mask = np.zeros(5, dtype=bool)
    assert not pad_mask(times, mask, 0.5).any()


def test_gap_length_is_zero_on_samples_and_span_between_them():
    valid_times = np.array([0.0, 1.0, 3.0])
    grid = np.array([0.0, 0.5, 2.0, 3.0, 4.0])
    gaps = gap_length_at(grid, valid_times)
    assert gaps.tolist() == [0.0, 1.0, 2.0, 0.0, np.inf]


def test_interpolate_short_gaps_fills_short_and_blanks_long_gaps():
    times = np.arange(0, 2.0, 0.1)
    values = np.linspace(4.0, 5.0, len(times))
    invalid = np.zeros(len(times), dtype=bool)
    invalid[3] = True
    invalid[8:16] = True
    resampled = interpolate_short_gaps(times, values, invalid, times, max_gap_s=0.5)
    assert resampled[3] == pytest.approx(values[3])
    assert np.isnan(resampled[8:16]).all()
    assert np.isfinite(resampled[:8]).all()


def test_lowpass_removes_fast_oscillation_and_keeps_nan_positions():
    sample_rate = 120.0
    t = np.arange(0, 5, 1 / sample_rate)
    slow = 4 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    signal = slow + 0.3 * np.sin(2 * np.pi * 30 * t)
    signal[100:110] = np.nan
    filtered = lowpass_ignoring_nan(signal, sample_rate, 4.0)
    assert np.isnan(filtered[100:110]).all()
    interior = slice(200, 400)
    assert np.max(np.abs(filtered[interior] - slow[interior])) < 0.02


def test_block_means_handles_partial_last_block():
    assert block_means(np.array([1.0, 3.0, 5.0, 7.0, 9.0]), 2).tolist() == [2.0, 6.0, 9.0]


def test_linear_slope_recovers_known_slope():
    digits = np.arange(1, 6)
    assert linear_slope(digits, 0.05 * digits + 0.2) == pytest.approx(0.05)


def test_linear_slope_is_nan_with_missing_values():
    assert np.isnan(linear_slope([1, 2, 3], [0.1, np.nan, 0.3]))
