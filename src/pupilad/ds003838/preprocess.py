import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pupilad.ds003838.events import digit_events, is_complete_sequence
from pupilad.ds003838.loaders import (
    load_ds003838_events,
    load_ds003838_participants,
    load_ds003838_samples,
    subjects_with_samples,
)
from pupilad.paths import DS003838_DERIVED
from pupilad.signal import (
    block_means,
    interpolate_short_gaps,
    invalid_sample_mask,
    lowpass_ignoring_nan,
    pad_mask,
)

SAMPLE_RATE_HZ = 120.0
OUTPUT_RATE_HZ = 20.0
DECIMATION = int(SAMPLE_RATE_HZ / OUTPUT_RATE_HZ)
MIN_CONFIDENCE = 0.6
MIN_DIAMETER_MM = 1.5
MAX_DIAMETER_MM = 9.0
ARTIFACT_PAD_S = 0.1
MAX_INTERPOLATED_GAP_S = 0.5
LOWPASS_CUTOFF_HZ = 4.0
MIN_VALID_SAMPLES = 100

BASELINE_S = 0.5
DIGIT_SOA_S = 2.0
POST_SEQUENCE_S = 2.0
MAX_LOAD = 13
MAX_TRIAL_MISSING_FRACTION = 0.30
DIGIT_WINDOWS = MAX_LOAD + 1
TRACE_LENGTH = int((DIGIT_SOA_S * MAX_LOAD + POST_SEQUENCE_S + BASELINE_S) * OUTPUT_RATE_HZ)

RIGHT_EYE_ID = 0
LEFT_EYE_ID = 1
EYE_SWITCH_MARGIN = 0.15


@dataclass
class CleanTrace:
    times: np.ndarray
    diameter_mm: np.ndarray
    missing_fraction: np.ndarray
    valid_fraction: float


def empty_trace(grid):
    times = grid[::DECIMATION]
    return CleanTrace(times, np.full(len(times), np.nan), np.ones(len(times)), 0.0)


def clean_eye(samples):
    samples = samples.sort_values("t").drop_duplicates("t")
    times = samples["t"].to_numpy()
    diameter = samples["diam"].to_numpy(dtype=float)
    invalid = invalid_sample_mask(
        samples["conf"].to_numpy(),
        samples["blink"].to_numpy(),
        diameter,
        MIN_CONFIDENCE,
        MIN_DIAMETER_MM,
        MAX_DIAMETER_MM,
    )
    invalid = pad_mask(times, invalid, ARTIFACT_PAD_S)
    grid = np.arange(times[0], times[-1], 1 / SAMPLE_RATE_HZ)
    if (~invalid).sum() < MIN_VALID_SAMPLES:
        return empty_trace(grid)
    resampled = interpolate_short_gaps(times, diameter, invalid, grid, MAX_INTERPOLATED_GAP_S)
    filtered = lowpass_ignoring_nan(resampled, SAMPLE_RATE_HZ, LOWPASS_CUTOFF_HZ)
    decimated = filtered[::DECIMATION]
    invalid_on_grid = np.interp(grid, times, invalid.astype(float))
    missing = np.maximum(
        block_means(invalid_on_grid, DECIMATION), np.isnan(decimated).astype(float)
    )
    return CleanTrace(grid[::DECIMATION], decimated, missing, float(1 - invalid.mean()))


def choose_eye(traces_by_eye, dominant_eye_id):
    chosen = dominant_eye_id if dominant_eye_id in traces_by_eye else next(iter(traces_by_eye))
    other = LEFT_EYE_ID if chosen == RIGHT_EYE_ID else RIGHT_EYE_ID
    if (
        other in traces_by_eye
        and traces_by_eye[other].valid_fraction
        > traces_by_eye[chosen].valid_fraction + EYE_SWITCH_MARGIN
    ):
        chosen = other
    return chosen


def digit_window_means(relative_times, relative_diameter, load):
    means = {}
    for digit in range(1, DIGIT_WINDOWS + 1):
        in_window = (relative_times >= DIGIT_SOA_S * (digit - 1)) & (
            relative_times < DIGIT_SOA_S * digit
        )
        usable = (
            digit <= load + 1
            and in_window.any()
            and np.isfinite(relative_diameter[in_window]).any()
        )
        means[f"d{digit}"] = float(np.nanmean(relative_diameter[in_window])) if usable else np.nan
    return means


def epoch_trial(trace, trial_digits):
    onset = trial_digits["time"].iloc[0]
    load = int(trial_digits["load"].iloc[0])
    condition = trial_digits["condition"].iloc[0]
    in_window = (trace.times >= onset - BASELINE_S) & (
        trace.times < onset + DIGIT_SOA_S * load + POST_SEQUENCE_S
    )
    relative_times = trace.times[in_window] - onset
    segment = trace.diameter_mm[in_window]
    missing = float(trace.missing_fraction[in_window].mean()) if len(segment) else 1.0
    pre_onset = relative_times < 0
    baseline = np.nanmean(segment[pre_onset]) if pre_onset.any() else np.nan
    relative_diameter = segment - baseline
    complete = is_complete_sequence(trial_digits)
    record = {
        "condition": condition,
        "load": load,
        "complete": complete,
        "n_recalled": trial_digits["recalled"].sum() if condition == "memory" else np.nan,
        "baseline_mm": baseline,
        "missing": missing,
        "keep": bool(complete and missing <= MAX_TRIAL_MISSING_FRACTION and np.isfinite(baseline)),
        **digit_window_means(relative_times, relative_diameter, load),
    }
    trace_row = np.full(TRACE_LENGTH, np.nan, dtype=np.float32)
    n_samples = min(len(relative_diameter), TRACE_LENGTH)
    trace_row[:n_samples] = relative_diameter[:n_samples]
    return record, trace_row


def epoch_subject(subject, trace, digits):
    records, traces = [], []
    for trial_number, trial_digits in digits.groupby("trial"):
        record, trace_row = epoch_trial(trace, trial_digits)
        records.append({"subject": subject, "trial": trial_number, **record})
        traces.append(trace_row)
    return pd.DataFrame(records), np.array(traces, dtype=np.float32)


def dominant_eye_id(subject, participants):
    return LEFT_EYE_ID if participants.loc[subject, "eye"] == "l" else RIGHT_EYE_ID


def quality_summary(subject, participants, eye_id, trace, trials):
    kept = trials[trials["keep"]]
    return {
        "subject": subject,
        "eye_used": "left" if eye_id == LEFT_EYE_ID else "right",
        "dominant": participants.loc[subject, "eye"],
        "valid_frac": round(trace.valid_fraction, 3),
        "n_trials": len(trials),
        "n_kept": len(kept),
        "n_mem_kept": int((kept["condition"] == "memory").sum()),
        "n_ctrl_kept": int((kept["condition"] == "control").sum()),
        "mean_baseline_mm": round(float(trials["baseline_mm"].mean()), 3),
    }


def preprocess_subject(subject):
    participants = load_ds003838_participants()
    samples = load_ds003838_samples(subject)
    traces_by_eye = {
        eye_id: clean_eye(eye_samples)
        for eye_id, eye_samples in samples.groupby("eye")
        if len(eye_samples)
    }
    eye_id = choose_eye(traces_by_eye, dominant_eye_id(subject, participants))
    trace = traces_by_eye[eye_id]
    trials, traces = epoch_subject(subject, trace, digit_events(load_ds003838_events(subject)))
    output_dir = DS003838_DERIVED / "trials"
    output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_dir / f"{subject}_trials.csv", index=False)
    np.save(output_dir / f"{subject}_traces.npy", traces)
    summary = quality_summary(subject, participants, eye_id, trace, trials)
    (output_dir / f"{subject}_qc.json").write_text(json.dumps(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad preprocess")
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args(argv)
    for subject in args.subjects or subjects_with_samples():
        print(json.dumps(preprocess_subject(subject)), flush=True)


if __name__ == "__main__":
    main()
