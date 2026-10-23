from pathlib import Path

import numpy as np
import pandas as pd

from pupilad.tobii.ingest import TASK_EVENTS

SAMPLE_RATE_HZ = 60
TASK_DURATION_S = 90
BASELINE_DURATION_S = 10
INTER_TASK_GAP_S = 5
TOBII_INVALID_CODE = 4
BLINKS_PER_TASK = 3
BLINK_LENGTH_SAMPLES = 15
MIN_AMPLITUDE_MM = 0.02


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plateau_curve(t, amplitude, rise_center, fall_center, width):
    return amplitude * sigmoid((t - rise_center) / width) * sigmoid(-(t - fall_center) / width)


def early_peak_curve(t, amplitude, peak_time, width):
    return amplitude * np.exp(-((t - peak_time) ** 2) / (2 * width**2))


def noisy_amplitude(rng, low, high, noise_sd):
    return max(rng.uniform(low, high) + rng.normal(0, noise_sd), MIN_AMPLITUDE_MM)


def shape_profile(profile, rng, difficulty):
    if rng.random() < difficulty * 0.4:
        return "mci_like" if profile == "healthy" else "healthy"
    return profile


def dilation_curve(t, profile, rng, difficulty):
    duration = t[-1]
    noise_sd = difficulty * 0.35
    if shape_profile(profile, rng, difficulty) == "healthy":
        amplitude = noisy_amplitude(rng, 0.5, 0.8, noise_sd)
        return plateau_curve(t, amplitude, duration * 0.15, duration * 0.85, 5)
    if rng.choice(["early_exaggerated", "blunted"]) == "early_exaggerated":
        amplitude = noisy_amplitude(rng, 0.7, 1.1, noise_sd)
        return early_peak_curve(t, amplitude, duration * rng.uniform(0.08, 0.15), duration * 0.08)
    amplitude = noisy_amplitude(rng, 0.15, 0.3, noise_sd)
    return plateau_curve(t, amplitude, duration * 0.3, duration * 0.7, 8)


def simulate_task_trace(baseline_mm, profile, rng, difficulty):
    n_samples = int(TASK_DURATION_S * SAMPLE_RATE_HZ)
    t = np.linspace(0, TASK_DURATION_S, n_samples)
    return (
        baseline_mm + dilation_curve(t, profile, rng, difficulty) + rng.normal(0, 0.03, n_samples)
    )


def insert_blinks(trace, rng):
    trace = trace.copy()
    for _ in range(BLINKS_PER_TASK):
        start = rng.integers(0, max(1, len(trace) - BLINK_LENGTH_SAMPLES))
        trace[start : start + BLINK_LENGTH_SAMPLES] = np.nan
    return trace


def export_rows(subject_id, start_ms, left, right, validity, events):
    step_ms = 1000.0 / SAMPLE_RATE_HZ
    return pd.DataFrame(
        {
            "Recording timestamp": start_ms + step_ms * np.arange(len(left)),
            "Pupil diameter left": left,
            "Pupil diameter right": right,
            "Validity left": validity,
            "Validity right": validity,
            "Event": events,
            "Event value": "",
            "Participant name": subject_id,
        }
    )


def baseline_block(subject_id, start_ms, baseline_mm, rng):
    n_samples = int(BASELINE_DURATION_S * SAMPLE_RATE_HZ)
    left = baseline_mm + rng.normal(0, 0.02, n_samples)
    right = left + rng.normal(0, 0.01, n_samples)
    return export_rows(subject_id, start_ms, left, right, 0, [""] * n_samples)


def task_block(subject_id, start_ms, baseline_mm, profile, labels, rng, difficulty):
    left = insert_blinks(simulate_task_trace(baseline_mm, profile, rng, difficulty), rng)
    recorded = ~np.isnan(left)
    right = left.copy()
    right[recorded] += rng.normal(0, 0.01, recorded.sum())
    validity = np.where(recorded, 0, TOBII_INVALID_CODE)
    events = [""] * len(left)
    events[0], events[-1] = labels
    return export_rows(subject_id, start_ms, left, right, validity, events)


def generate_subject_session(subject_id, profile="healthy", seed=None, difficulty=0.0):
    rng = np.random.default_rng(seed)
    baseline_mm = rng.uniform(3.0, 4.5)
    step_ms = 1000.0 / SAMPLE_RATE_HZ
    blocks, cursor_ms = [], 0.0
    for labels in TASK_EVENTS.values():
        baseline = baseline_block(subject_id, cursor_ms, baseline_mm, rng)
        cursor_ms += len(baseline) * step_ms
        task = task_block(subject_id, cursor_ms, baseline_mm, profile, labels, rng, difficulty)
        cursor_ms += len(task) * step_ms + INTER_TASK_GAP_S * 1000
        blocks += [baseline, task]
    return pd.concat(blocks, ignore_index=True)


def generate_batch(n_healthy=5, n_mci_like=5, output_dir="synthetic_exports", difficulty=0.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = [(f"SYN_HEALTHY_{i + 1:03d}", "healthy", 1000 + i) for i in range(n_healthy)]
    cohort += [(f"SYN_MCILIKE_{i + 1:03d}", "mci_like", 2000 + i) for i in range(n_mci_like)]
    paths = []
    for subject_id, profile, seed in cohort:
        path = output_dir / f"{subject_id}.tsv"
        generate_subject_session(subject_id, profile, seed, difficulty).to_csv(
            path, sep="\t", index=False
        )
        paths.append(path)
    return paths


if __name__ == "__main__":
    print(f"Generated {len(generate_batch())} synthetic sessions")
