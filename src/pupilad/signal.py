import numpy as np
from scipy.signal import butter, filtfilt


def invalid_sample_mask(confidence, blink, diameter_mm, min_confidence, min_mm, max_mm):
    diameter_mm = np.asarray(diameter_mm, dtype=float)
    return (
        (np.asarray(confidence) < min_confidence)
        | (np.asarray(blink) == 1)
        | ~np.isfinite(diameter_mm)
        | (diameter_mm < min_mm)
        | (diameter_mm > max_mm)
    )


def pad_mask(times, mask, pad_s):
    flagged_times = times[mask]
    if len(flagged_times) == 0 or pad_s <= 0:
        return mask.copy()
    right = np.searchsorted(flagged_times, times)
    distance = np.full(len(times), np.inf)
    has_right = right < len(flagged_times)
    distance[has_right] = flagged_times[right[has_right]] - times[has_right]
    has_left = right > 0
    distance[has_left] = np.minimum(
        distance[has_left], times[has_left] - flagged_times[right[has_left] - 1]
    )
    return mask | (np.abs(distance) <= pad_s)


def gap_length_at(grid, valid_times):
    right = np.searchsorted(valid_times, grid)
    gap = np.full(len(grid), np.inf)
    inside = (right > 0) & (right < len(valid_times))
    gap[inside] = valid_times[right[inside]] - valid_times[right[inside] - 1]
    on_sample = right < len(valid_times)
    on_sample[on_sample] = np.isclose(valid_times[right[on_sample]], grid[on_sample])
    gap[on_sample] = 0.0
    return gap


def interpolate_short_gaps(times, values, invalid, grid, max_gap_s):
    valid_times = times[~invalid]
    resampled = np.interp(grid, valid_times, values[~invalid])
    resampled[gap_length_at(grid, valid_times) > max_gap_s] = np.nan
    return resampled


def lowpass_ignoring_nan(signal, sample_rate_hz, cutoff_hz, order=4):
    missing = np.isnan(signal)
    if missing.all():
        return signal.copy()
    filled = np.interp(np.arange(len(signal)), np.flatnonzero(~missing), signal[~missing])
    b, a = butter(order, cutoff_hz / (sample_rate_hz / 2), btype="low")
    filtered = filtfilt(b, a, filled)
    filtered[missing] = np.nan
    return filtered


def block_means(values, block_size):
    starts = np.arange(0, len(values), block_size)
    sums = np.add.reduceat(values, starts)
    counts = np.diff(np.append(starts, len(values)))
    return sums / counts


def linear_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or not np.isfinite(y).all():
        return np.nan
    return float(np.polyfit(x, y, 1)[0])
