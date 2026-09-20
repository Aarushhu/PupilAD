"""
PupilAD 2.0 — Stage 2 Data Ingestion & Feature Extraction
Clinical pupillometry pipeline for Tobii Pro Lab exports

-------------------------------------------------------------------
WHAT THIS DOES
-------------------------------------------------------------------
Tobii Pro Lab exports a tab-separated (.tsv) file per recording with one
row per gaze sample (typically 60-1200 Hz depending on device/config).
This script:
  1. Loads the raw export
  2. Cleans blinks/artifacts and interpolates gaps
  3. Segments the recording into task windows using Event markers
  4. Extracts task-evoked pupillometry features per task
  5. Aggregates into one feature row per subject/session, ready to feed
     into the Stage 2 LSTM (replacing the old Haar-cascade feature set)

-------------------------------------------------------------------
BEFORE YOU RUN THIS FOR REAL
-------------------------------------------------------------------
Tobii Pro Lab's exact column names/order can vary slightly depending on
export settings (media study vs. external presentation, TTL sync, etc.).
Once you have a real export file, run `inspect_columns()` first and
adjust COLUMN_MAP below to match. Everything downstream reads from the
standardized names on the right side of COLUMN_MAP, so that's the only
place you should need to edit.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


# =====================================================================
# 1. CONFIG — adjust once you have a real export file
# =====================================================================

# Map from Tobii Pro Lab's raw column names -> standardized internal names.
# Run inspect_columns() on a real file and fix the left-hand side.
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

# Names of the four cognitive tasks as they'll appear in the Event column.
# These need to match whatever labels you send from your task-presentation
# software (e.g. PsychoPy/E-Prime) as TTL/log markers into Tobii Pro Lab.
TASK_EVENTS = {
    "nback": {"start": "NBACK_START", "end": "NBACK_END"},
    "digit_span": {"start": "DIGITSPAN_START", "end": "DIGITSPAN_END"},
    "stroop": {"start": "STROOP_START", "end": "STROOP_END"},
    "semantic_fluency": {"start": "FLUENCY_START", "end": "FLUENCY_END"},
}

# Baseline window: seconds of quiet fixation immediately before each task,
# used to baseline-correct pupil dilation (removes subject-level offset).
BASELINE_WINDOW_SEC = 5.0

# Blink/artifact thresholds
MIN_VALID_PUPIL_MM = 1.5
MAX_VALID_PUPIL_MM = 9.0
MAX_GAP_TO_INTERPOLATE_MS = 250  # gaps longer than this are left as NaN


# =====================================================================
# 2. LOAD + INSPECT
# =====================================================================

def inspect_columns(filepath: str) -> None:
    """Run this first on a real export to check actual column names."""
    df = pd.read_csv(filepath, sep="\t", nrows=20)
    print("Columns found in export:")
    for col in df.columns:
        print(f"  - {col}")
    print("\nFirst few rows:")
    print(df.head())


def load_raw_export(filepath: str) -> pd.DataFrame:
    """Load a Tobii Pro Lab .tsv export and rename to standardized columns."""
    df = pd.read_csv(filepath, sep="\t", low_memory=False)

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns not found in export: {missing}\n"
            f"Run inspect_columns() on this file and update COLUMN_MAP."
        )

    df = df.rename(columns=COLUMN_MAP)
    return df[list(COLUMN_MAP.values())]


# =====================================================================
# 3. CLEANING
# =====================================================================

def clean_pupil_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average left/right pupil diameter, remove blink artifacts, and
    interpolate short gaps. Long gaps are left as NaN and excluded
    from feature computation rather than interpolated.
    """
    df = df.copy()

    # Invalidate out-of-range or flagged-invalid samples
    for eye in ["left", "right"]:
        bad = (
            (df[f"pupil_{eye}_mm"] < MIN_VALID_PUPIL_MM)
            | (df[f"pupil_{eye}_mm"] > MAX_VALID_PUPIL_MM)
            | (df[f"valid_{eye}"] != 0)  # Tobii: 0 = valid, >0 = invalid
        )
        df.loc[bad, f"pupil_{eye}_mm"] = np.nan

    # Binocular average when both eyes valid, fall back to single eye
    df["pupil_mm"] = df[["pupil_left_mm", "pupil_right_mm"]].mean(axis=1, skipna=True)

    # Interpolate short gaps (typical blink duration), leave long gaps NaN
    sample_interval_ms = df["timestamp_ms"].diff().median()
    max_gap_samples = int(MAX_GAP_TO_INTERPOLATE_MS / sample_interval_ms) if sample_interval_ms else 10
    df["pupil_mm"] = df["pupil_mm"].interpolate(
        method="linear", limit=max_gap_samples, limit_area="inside"
    )

    return df


# =====================================================================
# 4. TASK SEGMENTATION
# =====================================================================

def segment_tasks(df: pd.DataFrame) -> dict:
    """
    Split the cleaned recording into per-task windows plus their
    preceding baseline windows, using Event start/end markers.
    Returns {task_name: {"baseline": df, "task": df}}
    """
    segments = {}
    for task_name, markers in TASK_EVENTS.items():
        start_rows = df[df["event"] == markers["start"]]
        end_rows = df[df["event"] == markers["end"]]

        if start_rows.empty or end_rows.empty:
            print(f"  [warn] Missing start/end markers for {task_name}, skipping")
            continue

        start_ts = start_rows["timestamp_ms"].iloc[0]
        end_ts = end_rows["timestamp_ms"].iloc[0]

        task_window = df[(df["timestamp_ms"] >= start_ts) & (df["timestamp_ms"] <= end_ts)]
        baseline_start = start_ts - (BASELINE_WINDOW_SEC * 1000)
        baseline_window = df[(df["timestamp_ms"] >= baseline_start) & (df["timestamp_ms"] < start_ts)]

        segments[task_name] = {"baseline": baseline_window, "task": task_window}

    return segments


# =====================================================================
# 5. FEATURE EXTRACTION
# =====================================================================

@dataclass
class TaskFeatures:
    task_name: str
    mean_dilation: float = np.nan       # mean baseline-corrected pupil size during task
    peak_dilation: float = np.nan       # max baseline-corrected pupil size
    latency_to_peak_ms: float = np.nan  # time from task start to peak dilation
    dilation_auc: float = np.nan        # area under the dilation curve (sustained effort)
    dilation_slope: float = np.nan      # linear slope of dilation over task duration
    baseline_pupil_mm: float = np.nan   # raw baseline pupil size (tonic level)
    pupil_variability: float = np.nan   # std dev during task (signal instability)
    valid_sample_frac: float = np.nan   # data quality: fraction of non-NaN samples

    def to_dict(self, prefix: str) -> dict:
        return {
            f"{prefix}_mean_dilation": self.mean_dilation,
            f"{prefix}_peak_dilation": self.peak_dilation,
            f"{prefix}_latency_to_peak_ms": self.latency_to_peak_ms,
            f"{prefix}_dilation_auc": self.dilation_auc,
            f"{prefix}_dilation_slope": self.dilation_slope,
            f"{prefix}_baseline_pupil_mm": self.baseline_pupil_mm,
            f"{prefix}_pupil_variability": self.pupil_variability,
            f"{prefix}_valid_sample_frac": self.valid_sample_frac,
        }


def extract_task_features(task_name: str, baseline: pd.DataFrame, task: pd.DataFrame) -> TaskFeatures:
    feats = TaskFeatures(task_name=task_name)

    if baseline["pupil_mm"].notna().sum() == 0 or task["pupil_mm"].notna().sum() == 0:
        return feats  # insufficient data, return all-NaN (handle downstream)

    baseline_mean = baseline["pupil_mm"].mean()
    feats.baseline_pupil_mm = baseline_mean

    dilation = task["pupil_mm"] - baseline_mean  # baseline-corrected signal
    t = task["timestamp_ms"] - task["timestamp_ms"].iloc[0]

    feats.mean_dilation = dilation.mean()
    feats.peak_dilation = dilation.max()

    peak_idx = dilation.idxmax()
    if pd.notna(peak_idx):
        feats.latency_to_peak_ms = t.loc[peak_idx]

    feats.dilation_auc = np.trapz(dilation.fillna(0), x=t)
    feats.pupil_variability = task["pupil_mm"].std()
    feats.valid_sample_frac = task["pupil_mm"].notna().mean()

    # Slope via simple linear regression (dilation ~ time), ignoring NaNs
    valid = dilation.notna()
    if valid.sum() > 1:
        feats.dilation_slope = np.polyfit(t[valid], dilation[valid], 1)[0]

    return feats


def build_subject_feature_row(subject_id: str, segments: dict) -> dict:
    """Combine per-task features into one flat row (feed to Stage 2 LSTM)."""
    row = {"subject_id": subject_id}
    for task_name, windows in segments.items():
        feats = extract_task_features(task_name, windows["baseline"], windows["task"])
        row.update(feats.to_dict(prefix=task_name))
    return row


# =====================================================================
# 6. PIPELINE ENTRY POINT
# =====================================================================

def process_recording(filepath: str) -> dict:
    """Full pipeline: raw export -> one feature row for this subject/session."""
    print(f"Processing {filepath}")
    raw = load_raw_export(filepath)
    subject_id = raw["subject_id"].iloc[0] if "subject_id" in raw.columns else Path(filepath).stem

    cleaned = clean_pupil_signal(raw)
    segments = segment_tasks(cleaned)
    row = build_subject_feature_row(subject_id, segments)
    return row


def process_batch(filepaths: list, output_csv: str = "pupilad_features.csv") -> pd.DataFrame:
    """Process multiple recordings and write a combined feature table."""
    rows = [process_recording(fp) for fp in filepaths]
    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(output_csv, index=False)
    print(f"\nWrote {len(feature_df)} subject rows to {output_csv}")
    return feature_df


if __name__ == "__main__":
    # Example usage once you have real export files:
    #
    # inspect_columns("raw_exports/subject001.tsv")   # run this first!
    #
    # feature_table = process_batch(
    #     filepaths=["raw_exports/subject001.tsv", "raw_exports/subject002.tsv"],
    #     output_csv="pupilad_features.csv",
    # )
    print("Scaffold ready. See __main__ block for usage once you have real Tobii exports.")