import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pupilad.paths import DATA_DIR

DEFAULT_LOGS_DIR = DATA_DIR / "task_battery_logs"
BASELINE_LOOKBACK_S = 1.0
LAST_TRIAL_DURATION_S = {"nback": 2.0, "stroop": 3.5}


def load_task_battery_csv(path):
    return pd.read_csv(path) if path.exists() else None


def window_rows(pupil_trace, onset, window_end, trial):
    clock = pupil_trace["psychopy_clock_time"]
    in_window = (clock >= onset - BASELINE_LOOKBACK_S) & (clock <= window_end)
    window = pupil_trace.loc[in_window]
    return pd.DataFrame(
        {
            "subject_id": trial["subject_id"],
            "trial_id": trial["trial_id"],
            "task_name": trial["task_name"],
            "load_level": trial["load_level"],
            "timestamp_s": window["psychopy_clock_time"].to_numpy() - onset,
            "pupil_diameter_px": window["pupil_diameter_px"].to_numpy(),
            "correct": int(bool(trial["correct"])),
        }
    )


def next_onset_or_default(onsets, index, default_duration):
    if index + 1 < len(onsets) and not pd.isna(onsets[index + 1]):
        return onsets[index + 1]
    return onsets[index] + default_duration


def paced_task_rows(pupil_trace, task_log, subject_id, task_name):
    onsets = task_log["onset_time"].to_numpy()
    tables = []
    for index, row in task_log.iterrows():
        if pd.isna(onsets[index]):
            continue
        trial = {
            "subject_id": subject_id,
            "trial_id": f"{task_name}_{index}",
            "task_name": task_name,
            "load_level": np.nan,
            "correct": row["correct"],
        }
        window_end = next_onset_or_default(onsets, index, LAST_TRIAL_DURATION_S[task_name])
        tables.append(window_rows(pupil_trace, onsets[index], window_end, trial))
    return tables


def digit_span_rows(pupil_trace, task_log, subject_id):
    tables = []
    for index, row in task_log.iterrows():
        if pd.isna(row["stim_onset_time"]) or pd.isna(row["recall_end_time"]):
            continue
        trial = {
            "subject_id": subject_id,
            "trial_id": f"digitspan_{row['direction']}_{row['length']}_{index}",
            "task_name": "digit_span",
            "load_level": row["length"],
            "correct": row["correct"],
        }
        tables.append(
            window_rows(pupil_trace, row["stim_onset_time"], row["recall_end_time"], trial)
        )
    return tables


def build_trial_table(logs_dir, subject_id):
    pupil_trace = load_task_battery_csv(logs_dir / f"{subject_id}_pupil_trace.csv")
    if pupil_trace is None:
        raise SystemExit(f"No pupil trace for {subject_id}; run extract-webcam first.")
    pupil_trace = pupil_trace.sort_values("psychopy_clock_time").reset_index(drop=True)
    tables = []
    for task_name in ("nback", "stroop"):
        task_log = load_task_battery_csv(logs_dir / f"{subject_id}_{task_name}.csv")
        if task_log is not None:
            tables += paced_task_rows(pupil_trace, task_log, subject_id, task_name)
    digit_span_log = load_task_battery_csv(logs_dir / f"{subject_id}_digitspan.csv")
    if digit_span_log is not None:
        tables += digit_span_rows(pupil_trace, digit_span_log, subject_id)
    if not tables:
        raise SystemExit("No trials could be matched to pupil samples.")
    return pd.concat(tables, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad webcam-trials")
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--logs_dir", type=Path, default=DEFAULT_LOGS_DIR)
    args = parser.parse_args(argv)
    table = build_trial_table(args.logs_dir, args.subject_id)
    output_path = args.logs_dir / f"{args.subject_id}_trial_pupil_table.csv"
    table.to_csv(output_path, index=False)
    counts = table.groupby("task_name")["trial_id"].nunique().to_dict()
    print(f"Trials matched per task: {counts}; wrote {output_path}")


if __name__ == "__main__":
    main()
