import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pupilad.validation.benchmarks import (
    ADNI_CSF_REFERENCE,
    LITERATURE_BENCHMARKS,
    PLAUSIBLE_PUPIL_RANGE_MM,
)

DEFAULT_COLUMN_MAP = {
    "subject_id": "subject_id",
    "trial_id": "trial_id",
    "task_name": "task_name",
    "load_level": "load_level",
    "timestamp_s": "timestamp_s",
    "pupil_diameter_mm": "pupil_diameter_mm",
    "correct": "correct",
    "baseline_window_s": (-1.0, 0.0),
    "response_window_s": (0.0, 4.0),
}
MAX_OUT_OF_RANGE_PERCENT = 5.0
MIN_FRACTION_IN_VELOCITY_BAND = 0.8
UNCALIBRATED_MEDIAN_FACTOR = 2
MIN_TRIALS_FOR_TESTS = 10
MIN_MATCHED_SUBJECTS = 5
EXPLORATORY_ALPHA = 0.10
ADNI_GROUPS = ("CN", "MCI", "Dementia")
LOAD_CITATION = LITERATURE_BENCHMARKS["dilation_amplitude_vs_cognitive_load"]["citation"]
VELOCITY_LOW = LITERATURE_BENCHMARKS["constriction_velocity_cutoff_low"]
VELOCITY_HIGH = LITERATURE_BENCHMARKS["constriction_velocity_cutoff_high"]


@dataclass
class TrialFeatures:
    subject_id: str
    trial_id: str
    task_name: str
    load_level: object
    correct: int
    baseline_diameter_mm: float
    peak_dilation_amplitude_mm: float
    peak_latency_s: float
    mean_constriction_velocity_mm_s: float


def check(name, citation, passed, detail):
    return {"check": name, "citation": citation, "pass": passed, "detail": detail}


def window_mask(times, window):
    low, high = window
    return (times >= low) & (times <= high)


def response_features(times, diameters, response_window, baseline):
    response = window_mask(times, response_window)
    if not response.any():
        return np.nan, np.nan, np.nan
    response_times = times[response]
    response_diameters = diameters[response]
    peak_index = int(np.nanargmax(response_diameters))
    peak_value = float(response_diameters[peak_index])
    latency = float(response_times[peak_index] - response_window[0])
    velocity = np.nan
    if peak_index < len(response_diameters) - 1:
        duration = response_times[-1] - response_times[peak_index]
        if duration > 0:
            velocity = float((peak_value - response_diameters[-1]) / duration)
    return peak_value - baseline, latency, velocity


def extract_trial_features(trial, column_map):
    times = trial[column_map["timestamp_s"]].to_numpy()
    diameters = trial[column_map["pupil_diameter_mm"]].to_numpy()
    baseline_mask = window_mask(times, column_map["baseline_window_s"])
    baseline = float(np.nanmean(diameters[baseline_mask])) if baseline_mask.any() else np.nan
    amplitude, latency, velocity = response_features(
        times, diameters, column_map["response_window_s"], baseline
    )
    first = trial.iloc[0]
    return TrialFeatures(
        subject_id=first[column_map["subject_id"]],
        trial_id=first[column_map["trial_id"]],
        task_name=first[column_map["task_name"]],
        load_level=first.get(column_map["load_level"], None),
        correct=int(first[column_map["correct"]]),
        baseline_diameter_mm=baseline,
        peak_dilation_amplitude_mm=amplitude,
        peak_latency_s=latency,
        mean_constriction_velocity_mm_s=velocity,
    )


def build_feature_table(pupil_table, column_map):
    groups = pupil_table.groupby([column_map["subject_id"], column_map["trial_id"]])
    return pd.DataFrame([asdict(extract_trial_features(trial, column_map)) for _, trial in groups])


def looks_uncalibrated(features):
    median = features["baseline_diameter_mm"].median()
    return pd.notna(median) and median > PLAUSIBLE_PUPIL_RANGE_MM[1] * UNCALIBRATED_MEDIAN_FACTOR


def baseline_range_check(features):
    low, high = PLAUSIBLE_PUPIL_RANGE_MM
    name = "Baseline pupil diameter within plausible physiological range"
    note = LITERATURE_BENCHMARKS["plausible_pupil_diameter_mm"]["note"]
    if looks_uncalibrated(features):
        return check(
            name,
            note,
            None,
            "SKIPPED: values look like uncalibrated units (e.g. webcam pixels), not mm.",
        )
    baseline = features["baseline_diameter_mm"]
    percent_bad = 100 * ((baseline < low) | (baseline > high)).sum() / max(len(features), 1)
    return check(
        name,
        note,
        percent_bad < MAX_OUT_OF_RANGE_PERCENT,
        f"{percent_bad:.1f}% of trials outside {low}-{high} mm (fail if >{MAX_OUT_OF_RANGE_PERCENT}%).",
    )


def velocity_band_check(features):
    name = "Constriction velocity in same order of magnitude as published cutoffs"
    citation = VELOCITY_LOW["citation"]
    if looks_uncalibrated(features):
        return check(
            name, citation, None, "SKIPPED: cutoff assumes mm/s and data look uncalibrated."
        )
    low, high = VELOCITY_LOW["value_mm_s"], VELOCITY_HIGH["value_mm_s"]
    velocities = features["mean_constriction_velocity_mm_s"].dropna()
    fraction = (
        ((velocities >= low * 0.5) & (velocities <= high * 2)).mean() if len(velocities) else 0.0
    )
    return check(
        name,
        citation,
        fraction > MIN_FRACTION_IN_VELOCITY_BAND,
        f"{fraction:.1%} of trials within a generous band around {low}-{high} mm/s.",
    )


def measurement_validity(features):
    return [baseline_range_check(features), velocity_band_check(features)]


def accuracy_check(features):
    usable = features.dropna(subset=["peak_dilation_amplitude_mm", "correct"])
    if len(usable) < MIN_TRIALS_FOR_TESTS:
        return check(
            "Dilation amplitude vs. accuracy",
            "n/a",
            None,
            f"Fewer than {MIN_TRIALS_FOR_TESTS} usable trials.",
        )
    correct = usable.loc[usable["correct"] == 1, "peak_dilation_amplitude_mm"]
    incorrect = usable.loc[usable["correct"] == 0, "peak_dilation_amplitude_mm"]
    if len(correct) < 2 or len(incorrect) < 2:
        return check(
            "Dilation amplitude vs. accuracy", "n/a", None, "Too few correct or incorrect trials."
        )
    test = stats.ttest_ind(correct, incorrect, equal_var=False)
    return check(
        "Dilation amplitude differs between correct and incorrect trials",
        LOAD_CITATION,
        test.pvalue < EXPLORATORY_ALPHA,
        f"t={test.statistic:.2f}, p={test.pvalue:.3f} (correct {correct.mean():.3f}, incorrect {incorrect.mean():.3f}). "
        "A null result does not invalidate the score: MCI-risk dilation can rise while accuracy stays normal.",
    )


def load_trend_check(features):
    load = pd.to_numeric(features.get("load_level"), errors="coerce")
    usable = features.assign(numeric_load=load).dropna(
        subset=["numeric_load", "peak_dilation_amplitude_mm"]
    )
    if len(usable) < MIN_TRIALS_FOR_TESTS:
        return None
    result = stats.spearmanr(usable["numeric_load"], usable["peak_dilation_amplitude_mm"])
    return check(
        "Dilation amplitude increases with cognitive load",
        LOAD_CITATION,
        result.statistic > 0 and result.pvalue < EXPLORATORY_ALPHA,
        f"Spearman rho={result.statistic:.2f}, p={result.pvalue:.3f}; expected positive below capacity.",
    )


def construct_validity(features):
    results = [accuracy_check(features)]
    trend = load_trend_check(features)
    if trend is not None:
        results.append(trend)
    return results


def lc_score_correlation(features, lc_scores):
    merged = features.merge(lc_scores, on="subject_id", how="inner")
    if merged["subject_id"].nunique() < MIN_MATCHED_SUBJECTS:
        return [check("LC score vs. subject accuracy", "n/a", None, "Not enough matched subjects.")]
    accuracy = merged.groupby("subject_id")["correct"].mean()
    score = merged.groupby("subject_id")["lc_score"].first()
    result = stats.spearmanr(accuracy, score)
    return [
        check(
            "LC score correlates with subject-level task accuracy",
            "General construct-validity expectation, not a specific paper",
            None,
            f"Spearman rho={result.statistic:.2f}, p={result.pvalue:.3f} across {len(accuracy)} subjects.",
        )
    ]


def ptau181_reference_check(ptau181):
    reference = ADNI_CSF_REFERENCE["markers"]["PTAU"]
    distances = {
        group: abs(ptau181 - reference[group]["mean"]) / reference[group]["std"]
        for group in ADNI_GROUPS
    }
    closest = min(distances, key=distances.get)
    details = "; ".join(
        f"{group}: mean {reference[group]['mean']:.1f} (sd {reference[group]['std']:.1f}, n={reference[group]['n']}), "
        f"{distances[group]:.2f} SD away"
        for group in ADNI_GROUPS
    )
    return [
        check(
            f"pTau181={ptau181:.1f} pg/mL vs. ADNI reference",
            ADNI_CSF_REFERENCE["source"],
            None,
            f"Closest to {closest}. {details}. Normative comparison only; assays may not be comparable.",
        )
    ]


def print_report(sections):
    labels = {True: "PASS", False: "FAIL", None: "INFO"}
    for section, checks in sections:
        print(f"\n--- {section} ---")
        for result in checks:
            print(f"[{labels[result['pass']]}] {result['check']}")
            print(f"   Source: {result['citation']}")
            print(f"   {result['detail']}\n")


def demo_trial_rows(subject_id, trial_number, at_risk, rng, times):
    load = int(rng.integers(1, 4))
    baseline = rng.normal(4.5, 0.4)
    amplitude = 0.15 * load + (0.25 if at_risk and load <= 2 else 0.0) + rng.normal(0, 0.05)
    correct = int(rng.random() > (0.05 * load - (0.1 if at_risk else 0)))
    response = np.where(
        times < 0, 0, amplitude * (1 - np.exp(-times / 0.8)) * np.exp(-0.05 * times)
    )
    trace = baseline + response + rng.normal(0, 0.02, size=times.shape)
    return pd.DataFrame(
        {
            "subject_id": subject_id,
            "trial_id": f"{subject_id}_T{trial_number}",
            "task_name": "n_back",
            "load_level": load,
            "timestamp_s": times,
            "pupil_diameter_mm": trace,
            "correct": correct,
        }
    )


def make_demo_data(n_subjects=20, n_trials=8, seed=0):
    rng = np.random.default_rng(seed)
    times = np.linspace(-1, 4, 50)
    tables = []
    for subject in range(n_subjects):
        subject_id = f"S{subject:03d}"
        at_risk = rng.random() < 0.3
        tables += [
            demo_trial_rows(subject_id, trial, at_risk, rng, times) for trial in range(n_trials)
        ]
    lc_scores = pd.DataFrame(
        {
            "subject_id": [f"S{subject:03d}" for subject in range(n_subjects)],
            "lc_score": rng.normal(65, 15, size=n_subjects).clip(0, 100),
        }
    )
    return pd.concat(tables, ignore_index=True), lc_scores


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad lc-check")
    parser.add_argument("--pupil_csv", type=Path)
    parser.add_argument("--lc_scores", type=Path)
    parser.add_argument("--column_map", type=Path)
    parser.add_argument("--ptau181", type=float)
    parser.add_argument("--output", type=Path, default=Path("extracted_features.csv"))
    args = parser.parse_args(argv)
    column_map = dict(DEFAULT_COLUMN_MAP)
    if args.column_map:
        column_map.update(json.loads(args.column_map.read_text()))
    if args.pupil_csv:
        pupil_table = pd.read_csv(args.pupil_csv)
        lc_scores = pd.read_csv(args.lc_scores) if args.lc_scores else None
    else:
        print("No --pupil_csv given; running on synthetic demo data.")
        pupil_table, lc_scores = make_demo_data()
    features = build_feature_table(pupil_table, column_map)
    sections = [
        ("Layer 1: Measurement validity", measurement_validity(features)),
        ("Layer 2: Construct validity", construct_validity(features)),
    ]
    if lc_scores is not None:
        sections.append(("Layer 2: LC score", lc_score_correlation(features, lc_scores)))
    if args.ptau181 is not None:
        sections.append(("Layer 3: ADNI reference context", ptau181_reference_check(args.ptau181)))
    print_report(sections)
    features.to_csv(args.output, index=False)
    print(f"Per-trial features written to {args.output}")


if __name__ == "__main__":
    main()
