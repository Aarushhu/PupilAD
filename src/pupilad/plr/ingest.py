import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_MAP = {
    "Timestamp": "timestamp_ms",
    "Pupil Diameter": "pupil_mm",
    "Event": "event",
    "Subject ID": "subject_id",
}
SUMMARY_COLUMN_MAP = {
    "NPi": "npi_score",
    "Latency": "plr_latency_ms",
    "CV": "plr_constriction_velocity",
    "MCV": "plr_max_constriction_velocity",
    "%CH": "plr_percent_change",
}
FLASH_ON_LABEL = "LIGHT_ON"
PLR_WINDOW_MS = 4000
RECOVERY_FRACTION = 0.75
ONSET_FRACTION = 0.05
MIN_VALID_PUPIL_MM = 1.5
MAX_VALID_PUPIL_MM = 9.0
MAX_INTERPOLATED_GAP_MS = 250
MIN_TRIAL_SAMPLES = 5
MIN_POST_FLASH_SAMPLES = 3


@dataclass
class PLRFeatures:
    baseline_diameter_mm: float = np.nan
    min_diameter_mm: float = np.nan
    latency_to_onset_ms: float = np.nan
    time_to_min_ms: float = np.nan
    constriction_amplitude_mm: float = np.nan
    constriction_percent: float = np.nan
    constriction_velocity_mm_s: float = np.nan
    max_constriction_velocity_mm_s: float = np.nan
    redilation_velocity_mm_s: float = np.nan
    t75_recovery_ms: float = np.nan
    valid_sample_frac: float = np.nan


def load_plr_export(path):
    export = pd.read_csv(path, low_memory=False)
    missing = [column for column in COLUMN_MAP if column not in export.columns]
    if missing:
        raise ValueError(f"Expected columns not found in PLR export: {missing}")
    return export.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]


def load_plr_summary_export(path):
    export = pd.read_csv(path)
    missing = [column for column in SUMMARY_COLUMN_MAP if column not in export.columns]
    if missing:
        raise ValueError(f"Expected summary columns not found: {missing}")
    return export.rename(columns=SUMMARY_COLUMN_MAP)


def clean_pupil_signal(recording):
    recording = recording.copy()
    out_of_range = (recording["pupil_mm"] < MIN_VALID_PUPIL_MM) | (
        recording["pupil_mm"] > MAX_VALID_PUPIL_MM
    )
    recording["pupil_mm"] = recording["pupil_mm"].mask(out_of_range)
    sample_interval_ms = recording["timestamp_ms"].diff().median()
    max_gap_samples = (
        int(MAX_INTERPOLATED_GAP_MS / sample_interval_ms) if sample_interval_ms else 10
    )
    recording["pupil_mm"] = recording["pupil_mm"].interpolate(
        method="linear", limit=max_gap_samples, limit_area="inside"
    )
    return recording


def segment_flash_trials(recording):
    onsets = recording.loc[recording["event"] == FLASH_ON_LABEL, "timestamp_ms"]
    trials = []
    for onset in onsets:
        in_window = recording["timestamp_ms"].between(onset, onset + PLR_WINDOW_MS)
        trial = recording[in_window].copy()
        trial["t_ms"] = trial["timestamp_ms"] - onset
        trials.append(trial)
    return trials


def peak_constriction_velocity(constriction):
    if len(constriction) < 2:
        return np.nan
    velocity = (constriction["pupil_mm"].diff() / (constriction["t_ms"].diff() / 1000)).dropna()
    return velocity.min() if len(velocity) else np.nan


def recovery_metrics(recovery, min_diameter, amplitude, time_to_min):
    target = min_diameter + RECOVERY_FRACTION * amplitude
    recovered = recovery[recovery["pupil_mm"] >= target]
    if len(recovery) < 2 or recovered.empty:
        return np.nan, np.nan
    t75 = recovered["t_ms"].iloc[0] - time_to_min
    velocity = (target - min_diameter) / (t75 / 1000) if t75 > 0 else np.nan
    return t75, velocity


def extract_plr_features(trial):
    features = PLRFeatures()
    signal = trial["pupil_mm"]
    if signal.notna().sum() < MIN_TRIAL_SAMPLES:
        return features
    features.valid_sample_frac = signal.notna().mean()
    pre_flash = trial.loc[trial["t_ms"] <= 0, "pupil_mm"]
    baseline = pre_flash.mean() if not pre_flash.empty else signal.iloc[0]
    features.baseline_diameter_mm = baseline
    post_flash = trial[trial["t_ms"] > 0]
    if post_flash["pupil_mm"].notna().sum() < MIN_POST_FLASH_SAMPLES:
        return features
    min_index = post_flash["pupil_mm"].idxmin()
    min_diameter = post_flash.loc[min_index, "pupil_mm"]
    time_to_min = post_flash.loc[min_index, "t_ms"]
    amplitude = baseline - min_diameter
    features.min_diameter_mm = min_diameter
    features.time_to_min_ms = time_to_min
    features.constriction_amplitude_mm = amplitude
    features.constriction_percent = amplitude / baseline * 100 if baseline else np.nan
    onset_samples = post_flash[post_flash["pupil_mm"] <= baseline - ONSET_FRACTION * amplitude]
    if not onset_samples.empty:
        features.latency_to_onset_ms = onset_samples["t_ms"].iloc[0]
    if time_to_min > 0:
        features.constriction_velocity_mm_s = amplitude / (time_to_min / 1000)
    features.max_constriction_velocity_mm_s = peak_constriction_velocity(
        post_flash[post_flash["t_ms"] <= time_to_min]
    )
    features.t75_recovery_ms, features.redilation_velocity_mm_s = recovery_metrics(
        post_flash[post_flash["t_ms"] > time_to_min], min_diameter, amplitude, time_to_min
    )
    return features


def subject_feature_row(subject_id, trials):
    usable = [
        asdict(features)
        for features in map(extract_plr_features, trials)
        if not np.isnan(features.baseline_diameter_mm)
    ]
    row = {"subject_id": subject_id, "n_valid_trials": len(usable)}
    if usable:
        row.update({f"plr_{name}": value for name, value in pd.DataFrame(usable).mean().items()})
    return row


def subject_summary_row(path):
    summary = load_plr_summary_export(path)
    return {
        "subject_id": Path(path).stem,
        **summary[list(SUMMARY_COLUMN_MAP.values())].mean().to_dict(),
    }


def process_recording(path):
    recording = load_plr_export(path)
    subject_id = recording["subject_id"].iloc[0] if len(recording) else Path(path).stem
    return subject_feature_row(subject_id, segment_flash_trials(clean_pupil_signal(recording)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad plr-features")
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("pupilad_plr_features.csv"))
    args = parser.parse_args(argv)
    build_row = subject_summary_row if args.summary else process_recording
    features = pd.DataFrame([build_row(path) for path in args.exports])
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} subject rows to {args.output}")


if __name__ == "__main__":
    main()
