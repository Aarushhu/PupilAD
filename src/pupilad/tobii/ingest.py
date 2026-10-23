import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_MAP = {
    "Recording timestamp": "timestamp_ms",
    "Pupil diameter left": "pupil_left_mm",
    "Pupil diameter right": "pupil_right_mm",
    "Validity left": "valid_left",
    "Validity right": "valid_right",
    "Event": "event",
    "Event value": "event_value",
    "Participant name": "subject_id",
}
TASK_EVENTS = {
    "nback": ("NBACK_START", "NBACK_END"),
    "digit_span": ("DIGITSPAN_START", "DIGITSPAN_END"),
    "stroop": ("STROOP_START", "STROOP_END"),
    "semantic_fluency": ("FLUENCY_START", "FLUENCY_END"),
}
BASELINE_WINDOW_S = 5.0
MIN_VALID_PUPIL_MM = 1.5
MAX_VALID_PUPIL_MM = 9.0
MAX_INTERPOLATED_GAP_MS = 250
TOBII_VALID_CODE = 0


@dataclass
class TaskFeatures:
    mean_dilation: float = np.nan
    peak_dilation: float = np.nan
    latency_to_peak_ms: float = np.nan
    dilation_auc: float = np.nan
    dilation_slope: float = np.nan
    baseline_pupil_mm: float = np.nan
    pupil_variability: float = np.nan
    valid_sample_frac: float = np.nan

    def prefixed(self, prefix):
        return {f"{prefix}_{name}": value for name, value in asdict(self).items()}


FEATURE_NAMES = list(TaskFeatures.__dataclass_fields__)


def load_tobii_export(path):
    export = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [column for column in COLUMN_MAP if column not in export.columns]
    if missing:
        raise ValueError(f"Expected columns not found in Tobii export: {missing}")
    return export.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]


def invalidate_eye(recording, eye):
    diameter = recording[f"pupil_{eye}_mm"]
    invalid = (
        (diameter < MIN_VALID_PUPIL_MM)
        | (diameter > MAX_VALID_PUPIL_MM)
        | (recording[f"valid_{eye}"] != TOBII_VALID_CODE)
    )
    return diameter.mask(invalid)


def clean_pupil_signal(recording):
    recording = recording.copy()
    for eye in ("left", "right"):
        recording[f"pupil_{eye}_mm"] = invalidate_eye(recording, eye)
    recording["pupil_mm"] = recording[["pupil_left_mm", "pupil_right_mm"]].mean(axis=1)
    sample_interval_ms = recording["timestamp_ms"].diff().median()
    max_gap_samples = (
        int(MAX_INTERPOLATED_GAP_MS / sample_interval_ms) if sample_interval_ms else 10
    )
    recording["pupil_mm"] = recording["pupil_mm"].interpolate(
        method="linear", limit=max_gap_samples, limit_area="inside"
    )
    return recording


def first_event_time(recording, label):
    matches = recording.loc[recording["event"] == label, "timestamp_ms"]
    return matches.iloc[0] if len(matches) else None


def segment_tasks(recording):
    segments = {}
    time_ms = recording["timestamp_ms"]
    for task_name, (start_label, end_label) in TASK_EVENTS.items():
        start = first_event_time(recording, start_label)
        end = first_event_time(recording, end_label)
        if start is None or end is None:
            continue
        baseline_start = start - BASELINE_WINDOW_S * 1000
        segments[task_name] = {
            "baseline": recording[(time_ms >= baseline_start) & (time_ms < start)],
            "task": recording[(time_ms >= start) & (time_ms <= end)],
        }
    return segments


def extract_task_features(baseline, task):
    features = TaskFeatures()
    if baseline["pupil_mm"].notna().sum() == 0 or task["pupil_mm"].notna().sum() == 0:
        return features
    baseline_mean = baseline["pupil_mm"].mean()
    dilation = task["pupil_mm"] - baseline_mean
    elapsed_ms = task["timestamp_ms"] - task["timestamp_ms"].iloc[0]
    valid = dilation.notna()
    features.baseline_pupil_mm = baseline_mean
    features.mean_dilation = dilation.mean()
    features.peak_dilation = dilation.max()
    features.latency_to_peak_ms = elapsed_ms.loc[dilation.idxmax()]
    features.dilation_auc = np.trapezoid(dilation.fillna(0), x=elapsed_ms)
    features.pupil_variability = task["pupil_mm"].std()
    features.valid_sample_frac = valid.mean()
    if valid.sum() > 1:
        features.dilation_slope = np.polyfit(elapsed_ms[valid], dilation[valid], 1)[0]
    return features


def subject_feature_row(subject_id, segments):
    row = {"subject_id": subject_id}
    for task_name, windows in segments.items():
        row.update(extract_task_features(windows["baseline"], windows["task"]).prefixed(task_name))
    return row


def process_recording(path):
    recording = load_tobii_export(path)
    subject_id = recording["subject_id"].iloc[0] if len(recording) else Path(path).stem
    return subject_feature_row(subject_id, segment_tasks(clean_pupil_signal(recording)))


def process_batch(paths):
    return pd.DataFrame([process_recording(path) for path in paths])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad tobii-features")
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("pupilad_features.csv"))
    args = parser.parse_args(argv)
    features = process_batch(args.exports)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} subject rows to {args.output}")


if __name__ == "__main__":
    main()
