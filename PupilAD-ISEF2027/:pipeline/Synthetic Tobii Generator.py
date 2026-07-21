"""
PupilAD 2.0 — Synthetic Data Generator (Task-Evoked / Tobii-style)

Generates fake but realistically-shaped raw pupillometry exports matching
the column format expected by ingest_tobii.py. Includes injected noise,
blinks, and gaps so the cleaning/interpolation code gets tested against
something closer to real data than a clean sine wave.

Two synthetic subject profiles are generated: a "healthy" profile (normal
task-evoked dilation) and an "MCI-like" profile (blunted/early peaks),
matching the physiological patterns described in the project's background
research, purely so downstream code has two distinguishable classes to
test classification logic against. These are NOT real physiological
claims — just illustrative synthetic patterns for pipeline testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SAMPLE_RATE_HZ = 60
TASK_DURATION_SEC = 90  # per task
BASELINE_DURATION_SEC = 10
INTER_TASK_GAP_SEC = 5

TASKS = ["nback", "digit_span", "stroop", "semantic_fluency"]
TASK_EVENT_LABELS = {
    "nback": ("NBACK_START", "NBACK_END"),
    "digit_span": ("DIGITSPAN_START", "DIGITSPAN_END"),
    "stroop": ("STROOP_START", "STROOP_END"),
    "semantic_fluency": ("FLUENCY_START", "FLUENCY_END"),
}


def _simulate_pupil_trace(duration_sec, sample_rate, baseline_mm, profile="healthy", rng=None):
    """
    Simulate a single task's pupil diameter trace.
    'healthy': gradual rise to a steady peak, holds, gradual return.
    'mci_like': either an exaggerated early peak that decays (early MCI)
                or a blunted peak that never rises much (moderate MCI) —
                picked randomly per call to give some variability.
    """
    rng = rng or np.random.default_rng()
    n = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n)

    if profile == "healthy":
        # smooth ramp up, steady plateau, smooth ramp down
        amplitude = rng.uniform(0.5, 0.8)  # mm above baseline
        rise = amplitude * (1 / (1 + np.exp(-(t - duration_sec * 0.15) / 5)))
        fall = amplitude * (1 / (1 + np.exp((t - duration_sec * 0.85) / 5)))
        dilation = np.minimum(rise, fall) if False else rise - (amplitude - fall)
        dilation = rise * (fall / amplitude)  # simple bell-ish combination
    else:
        variant = rng.choice(["early_exaggerated", "blunted"])
        if variant == "early_exaggerated":
            amplitude = rng.uniform(0.7, 1.1)
            peak_time = duration_sec * rng.uniform(0.08, 0.15)  # peaks early
            dilation = amplitude * np.exp(-((t - peak_time) ** 2) / (2 * (duration_sec * 0.08) ** 2))
        else:  # blunted
            amplitude = rng.uniform(0.15, 0.3)  # much smaller peak
            rise = amplitude * (1 / (1 + np.exp(-(t - duration_sec * 0.3) / 8)))
            fall = amplitude * (1 / (1 + np.exp((t - duration_sec * 0.7) / 8)))
            dilation = rise * (fall / amplitude)

    noise = rng.normal(0, 0.03, n)
    trace = baseline_mm + dilation + noise
    return trace


def _inject_blinks(trace, rng, n_blinks=3, blink_len_samples=15):
    """Randomly zero out short windows to simulate blinks (removed as NaN)."""
    trace = trace.copy()
    n = len(trace)
    for _ in range(n_blinks):
        start = rng.integers(0, max(1, n - blink_len_samples))
        trace[start:start + blink_len_samples] = np.nan
    return trace


def generate_subject_session(subject_id: str, profile: str = "healthy", seed: int = None) -> pd.DataFrame:
    """
    Generate one full synthetic session (all 4 tasks + baselines + event
    markers) in the raw column format ingest_tobii.py expects.
    """
    rng = np.random.default_rng(seed)
    baseline_mm = rng.uniform(3.0, 4.5)  # resting pupil size, varies by subject

    rows = []
    t_cursor_ms = 0.0
    dt_ms = 1000.0 / SAMPLE_RATE_HZ

    for task in TASKS:
        start_label, end_label = TASK_EVENT_LABELS[task]

        # --- baseline window (quiet fixation, no event) ---
        n_baseline = int(BASELINE_DURATION_SEC * SAMPLE_RATE_HZ)
        baseline_trace = baseline_mm + rng.normal(0, 0.02, n_baseline)
        for i, val in enumerate(baseline_trace):
            rows.append({
                "Recording timestamp": t_cursor_ms,
                "Pupil diameter left": val,
                "Pupil diameter right": val + rng.normal(0, 0.01),
                "Validity left": 0,
                "Validity right": 0,
                "Event": "",
                "Event value": "",
                "Participant name": subject_id,
            })
            t_cursor_ms += dt_ms

        # --- task window ---
        task_trace = _simulate_pupil_trace(TASK_DURATION_SEC, SAMPLE_RATE_HZ, baseline_mm, profile, rng)
        task_trace = _inject_blinks(task_trace, rng)

        task_start_ts = t_cursor_ms
        for i, val in enumerate(task_trace):
            event = start_label if i == 0 else ("" if i != len(task_trace) - 1 else end_label)
            valid = 0 if not np.isnan(val) else 4  # Tobii: nonzero = invalid
            rows.append({
                "Recording timestamp": t_cursor_ms,
                "Pupil diameter left": val,
                "Pupil diameter right": val + (rng.normal(0, 0.01) if not np.isnan(val) else np.nan),
                "Validity left": valid,
                "Validity right": valid,
                "Event": event,
                "Event value": "",
                "Participant name": subject_id,
            })
            t_cursor_ms += dt_ms

        # --- inter-task gap ---
        t_cursor_ms += INTER_TASK_GAP_SEC * 1000

    return pd.DataFrame(rows)


def generate_batch(n_healthy: int = 5, n_mci_like: int = 5, output_dir: str = "synthetic_exports") -> list:
    """Generate a batch of synthetic subject sessions and save as .tsv files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepaths = []

    for i in range(n_healthy):
        subject_id = f"SYN_HEALTHY_{i+1:03d}"
        df = generate_subject_session(subject_id, profile="healthy", seed=1000 + i)
        fp = f"{output_dir}/{subject_id}.tsv"
        df.to_csv(fp, sep="\t", index=False)
        filepaths.append(fp)

    for i in range(n_mci_like):
        subject_id = f"SYN_MCILIKE_{i+1:03d}"
        df = generate_subject_session(subject_id, profile="mci_like", seed=2000 + i)
        fp = f"{output_dir}/{subject_id}.tsv"
        df.to_csv(fp, sep="\t", index=False)
        filepaths.append(fp)

    print(f"Generated {len(filepaths)} synthetic sessions in {output_dir}/")
    return filepaths


if __name__ == "__main__":
    generate_batch(n_healthy=5, n_mci_like=5)
