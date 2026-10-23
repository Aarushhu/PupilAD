"""
PupilAD — Build a Trial-Level Pupil Table from a Recorded Session
====================================================================

Takes the outputs of one recorded task_battery session —
    {subject_id}_pupil_trace.csv    (from extract_pupil_from_video.py)
    {subject_id}_markers.csv        (task start/end events)
    {subject_id}_nback.csv
    {subject_id}_stroop.csv
    {subject_id}_digitspan.csv
— and produces ONE long-format CSV in the schema validate_lc_score.py
expects: subject_id, trial_id, task_name, load_level, timestamp_s,
pupil_diameter_px, correct.

Trial windows are built from the ACTUAL onset/end timestamps logged by
run_task_battery.py (onset_time for N-back/Stroop, stim_onset_time /
recall_end_time for Digit Span) rather than nominal timing constants, since
per-trial duration varies with response speed. A trial's window runs from
1.0s before its onset (baseline) to the NEXT trial's onset (response) — or,
for the last trial in a task / Digit Span, to its own logged end time.

Semantic Fluency isn't included: it has no per-trial correctness in this
pipeline (responses are scored later from audio), so there's nothing to
build a trial_id/correct row around.

Usage
-----
    python task_battery/build_trial_pupil_table.py --subject_id S001

Then feed the result straight into validate_lc_score.py:
    python validate_lc_score.py \\
        --pupil_csv task_battery/task_battery_logs/S001_trial_pupil_table.csv \\
        --column_map task_battery/column_map_webcam.json

Only needs pandas/numpy — runs fine under .venv/ (no psychopy/opencv needed
here; this script is pure data wrangling over already-extracted CSVs).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_LOOKBACK_S = 1.0

# Fallback max trial duration when a trial has no "next trial" to bound its
# response window against (the last trial in a task). Kept in sync with
# run_task_battery.py's NBACK_SOA / Stroop maxWait+ISI by hand — these are
# generous upper bounds, not exact, since any pupil samples beyond the real
# trial end just get cut off by the next task's own baseline window anyway.
FALLBACK_DURATION_S = {
    "nback": 2.0,       # NBACK_SOA
    "stroop": 3.5,       # 3.0s response window + 0.5s ISI
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[warn] {path} not found — skipping that task.")
        return None
    return pd.read_csv(path)


def _rows_for_window(pupil_trace: pd.DataFrame, onset: float, window_end: float,
                      subject_id: str, trial_id: str, task_name: str,
                      load_level, correct) -> list:
    mask = (
        (pupil_trace["psychopy_clock_time"] >= onset - BASELINE_LOOKBACK_S)
        & (pupil_trace["psychopy_clock_time"] <= window_end)
    )
    sliced = pupil_trace.loc[mask]
    rows = []
    for _, r in sliced.iterrows():
        rows.append({
            "subject_id": subject_id,
            "trial_id": trial_id,
            "task_name": task_name,
            "load_level": load_level,
            "timestamp_s": r["psychopy_clock_time"] - onset,
            "pupil_diameter_px": r["pupil_diameter_px"],
            "correct": int(bool(correct)),
        })
    return rows


def build_nback_rows(pupil_trace: pd.DataFrame, nback_df: pd.DataFrame, subject_id: str) -> list:
    rows = []
    for i, row in nback_df.iterrows():
        onset = row["onset_time"]
        if pd.isna(onset):
            continue
        if i + 1 < len(nback_df) and not pd.isna(nback_df.loc[i + 1, "onset_time"]):
            window_end = nback_df.loc[i + 1, "onset_time"]
        else:
            window_end = onset + FALLBACK_DURATION_S["nback"]
        rows.extend(_rows_for_window(
            pupil_trace, onset, window_end, subject_id,
            trial_id=f"nback_{i}", task_name="nback",
            load_level=np.nan, correct=row["correct"],
        ))
    return rows


def build_stroop_rows(pupil_trace: pd.DataFrame, stroop_df: pd.DataFrame, subject_id: str) -> list:
    rows = []
    for i, row in stroop_df.iterrows():
        onset = row["onset_time"]
        if pd.isna(onset):
            continue
        if i + 1 < len(stroop_df) and not pd.isna(stroop_df.loc[i + 1, "onset_time"]):
            window_end = stroop_df.loc[i + 1, "onset_time"]
        else:
            window_end = onset + FALLBACK_DURATION_S["stroop"]
        rows.extend(_rows_for_window(
            pupil_trace, onset, window_end, subject_id,
            trial_id=f"stroop_{i}", task_name="stroop",
            load_level=np.nan, correct=row["correct"],
        ))
    return rows


def build_digitspan_rows(pupil_trace: pd.DataFrame, digitspan_df: pd.DataFrame, subject_id: str) -> list:
    rows = []
    skipped = 0
    for i, row in digitspan_df.iterrows():
        onset = row["stim_onset_time"]
        window_end = row["recall_end_time"]
        if pd.isna(onset) or pd.isna(window_end):
            skipped += 1
            continue
        rows.extend(_rows_for_window(
            pupil_trace, onset, window_end, subject_id,
            trial_id=f"digitspan_{row['direction']}_{row['length']}_{i}",
            task_name="digit_span",
            load_level=row["length"],  # sequence length as cognitive load
            correct=row["correct"],
        ))
    if skipped:
        print(f"[warn] {skipped} digit span trial(s) missing onset/end timestamps — skipped "
              f"(these are from a run before that logging was added; re-record to fix).")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build a trial-level pupil table from one recorded task_battery session."
    )
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--logs_dir", default="task_battery/task_battery_logs")
    parser.add_argument("--output", default=None,
                         help="Output CSV path (default: {logs_dir}/{subject_id}_trial_pupil_table.csv)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    subject_id = args.subject_id

    pupil_trace = _load_csv(logs_dir / f"{subject_id}_pupil_trace.csv")
    if pupil_trace is None:
        raise SystemExit(
            f"No pupil trace found. Run extract_pupil_from_video.py first:\n"
            f"  .venv-psychopy/bin/python task_battery/extract_pupil_from_video.py --subject_id {subject_id}"
        )
    pupil_trace = pupil_trace.sort_values("psychopy_clock_time").reset_index(drop=True)

    all_rows = []
    task_counts = {}

    nback_df = _load_csv(logs_dir / f"{subject_id}_nback.csv")
    if nback_df is not None:
        rows = build_nback_rows(pupil_trace, nback_df, subject_id)
        all_rows.extend(rows)
        task_counts["nback"] = nback_df["trial"].nunique() if "trial" in nback_df else len(nback_df)

    stroop_df = _load_csv(logs_dir / f"{subject_id}_stroop.csv")
    if stroop_df is not None:
        rows = build_stroop_rows(pupil_trace, stroop_df, subject_id)
        all_rows.extend(rows)
        task_counts["stroop"] = len(stroop_df)

    digitspan_df = _load_csv(logs_dir / f"{subject_id}_digitspan.csv")
    if digitspan_df is not None:
        rows = build_digitspan_rows(pupil_trace, digitspan_df, subject_id)
        all_rows.extend(rows)
        task_counts["digit_span"] = len(digitspan_df)

    if not all_rows:
        raise SystemExit("No trials could be matched to pupil data — check that the behavioral "
                          "CSVs have onset_time/stim_onset_time/recall_end_time columns (requires "
                          "the version of run_task_battery.py that logs those).")

    result = pd.DataFrame(all_rows)
    output_path = Path(args.output) if args.output else logs_dir / f"{subject_id}_trial_pupil_table.csv"
    result.to_csv(output_path, index=False)

    print(f"Built {len(result)} sample-rows across {result['trial_id'].nunique()} trials:")
    for task, n in task_counts.items():
        n_matched = result.loc[result["task_name"] == task, "trial_id"].nunique()
        print(f"  {task}: {n_matched}/{n} trials matched to pupil data")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
