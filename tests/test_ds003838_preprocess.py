import numpy as np
import pandas as pd
import pytest

from pupilad.ds003838.events import digit_events, is_complete_sequence, parse_event_label
from pupilad.ds003838.preprocess import (
    LEFT_EYE_ID,
    OUTPUT_RATE_HZ,
    RIGHT_EYE_ID,
    CleanTrace,
    choose_eye,
    clean_eye,
    digit_window_means,
    epoch_trial,
)


def test_parse_memory_label():
    assert parse_event_label(6003091) == {
        "condition": "memory",
        "position": 3,
        "load": 9,
        "recalled": 1,
    }


def test_parse_control_label():
    parsed = parse_event_label(501313)
    assert (parsed["condition"], parsed["position"], parsed["load"]) == ("control", 13, 13)
    assert np.isnan(parsed["recalled"])


def test_unrelated_label_is_ignored():
    assert parse_event_label(12345) is None


def make_events(sequences):
    rows, time = [], 100.0
    for condition, load in sequences:
        for position in range(1, load + 1):
            code = (
                6_000_000 + position * 1000 + load * 10 + 1
                if condition == "memory"
                else 500_000 + position * 100 + load
            )
            rows.append({"timestamp": time, "label": code})
            time += 2.0
        time += 10.0
    return pd.DataFrame(rows)


def test_digit_events_number_trials_by_first_digit():
    digits = digit_events(make_events([("memory", 5), ("control", 9)]))
    assert digits.groupby("trial")["load"].first().tolist() == [5, 9]
    assert all(is_complete_sequence(group) for _, group in digits.groupby("trial"))


def test_incomplete_sequence_is_detected():
    digits = digit_events(make_events([("memory", 5)])).iloc[:4]
    assert not is_complete_sequence(digits)


def synthetic_eye(seconds=60, rate=120.0, diameter=4.0):
    t = np.arange(0, seconds, 1 / rate)
    return pd.DataFrame({"t": t, "eye": 0, "conf": 0.95, "diam": diameter + 0.0 * t, "blink": 0})


def test_clean_eye_keeps_good_signal_and_downsamples():
    trace = clean_eye(synthetic_eye())
    assert trace.valid_fraction == pytest.approx(1.0)
    assert np.nanmax(np.abs(trace.diameter_mm - 4.0)) < 1e-6
    assert np.median(np.diff(trace.times)) == pytest.approx(1 / OUTPUT_RATE_HZ)


def test_clean_eye_marks_long_dropout_missing():
    samples = synthetic_eye()
    samples.loc[(samples["t"] > 20) & (samples["t"] < 22), "conf"] = 0.1
    trace = clean_eye(samples)
    in_dropout = (trace.times > 20.5) & (trace.times < 21.5)
    assert np.isnan(trace.diameter_mm[in_dropout]).all()
    assert (trace.missing_fraction[in_dropout] == 1).all()


def test_choose_eye_switches_only_when_other_eye_is_much_better():
    def trace(valid):
        return CleanTrace(np.zeros(1), np.zeros(1), np.zeros(1), valid)

    assert (
        choose_eye({RIGHT_EYE_ID: trace(0.8), LEFT_EYE_ID: trace(0.9)}, RIGHT_EYE_ID)
        == RIGHT_EYE_ID
    )
    assert (
        choose_eye({RIGHT_EYE_ID: trace(0.5), LEFT_EYE_ID: trace(0.9)}, RIGHT_EYE_ID) == LEFT_EYE_ID
    )
    assert choose_eye({LEFT_EYE_ID: trace(0.5)}, RIGHT_EYE_ID) == LEFT_EYE_ID


def test_digit_window_means_average_each_two_second_window():
    times = np.arange(0, 12, 0.05)
    dilation = np.floor(times / 2.0) * 0.1
    means = digit_window_means(times, dilation, load=5)
    assert [round(means[f"d{digit}"], 6) for digit in range(1, 7)] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert np.isnan(means["d7"])


def ramp_trace(onset, load, slope_mm_per_s, missing=0.0):
    times = np.arange(onset - 5, onset + 2 * load + 10, 1 / OUTPUT_RATE_HZ)
    diameter = 4.0 + np.clip(times - onset, 0, None) * slope_mm_per_s
    return CleanTrace(times, diameter, np.full(len(times), missing), 1.0)


def test_epoch_trial_baseline_corrects_and_keeps_good_trial():
    digits = digit_events(make_events([("memory", 5)]))
    onset = digits["time"].iloc[0]
    record, trace_row = epoch_trial(ramp_trace(onset, 5, 0.05), digits)
    assert record["keep"] and record["baseline_mm"] == pytest.approx(4.0)
    assert record["d1"] == pytest.approx(0.05, abs=0.003)
    assert record["d5"] > record["d1"]
    assert np.nanmax(np.abs(trace_row[:5])) == pytest.approx(0.0)


def test_epoch_trial_rejects_trial_with_too_much_missing_data():
    digits = digit_events(make_events([("memory", 5)]))
    record, _ = epoch_trial(ramp_trace(digits["time"].iloc[0], 5, 0.05, missing=0.5), digits)
    assert not record["keep"]
